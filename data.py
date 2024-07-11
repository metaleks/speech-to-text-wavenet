import sugartensor as tf
import numpy as np
import csv
import string
import os

__author__ = 'namju.kim@kakaobrain.com'

# default data path
_data_path = 'asset/data/'

#
# vocabulary table
#

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# byte to index mapping
byte2index = {ch: i for i, ch in enumerate(index2byte)}

# vocabulary size
voca_size = len(index2byte)

# convert sentence to index list
def str2index(str_):
    """
    Convert a string to a list of indices based on the vocabulary table.
    Removes punctuation and converts to lower case.
    """
    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    str_ = str_.translate(str.maketrans('', '', string.punctuation)).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            # drop OOV (out-of-vocabulary) characters
            pass
    return res

# convert index list to string
def index2str(index_list):
    """
    Convert a list of indices back to a string.
    Stops at <EOS> token.
    """
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_

# print list of index list
def print_index(indices):
    """
    Print a list of index lists as strings.
    """
    for index_list in indices:
        print(index2str(index_list))

# real-time wave to mfcc conversion function
@tf.sg_producer_func
def _load_mfcc(src_list):
    """
    Load and process MFCC data for a given list of source files.
    """
    # label, wave_file
    label, mfcc_file = src_list

    # decode string to integer
    try:
        label = np.frombuffer(label, dtype=np.int)
    except ValueError as e:
        tf.sg_info(f"Error decoding label: {e}")
        return None, None

    # load mfcc
    try:
        mfcc = np.load(mfcc_file, allow_pickle=False)
    except (IOError, ValueError) as e:
        tf.sg_info(f"Error loading MFCC file {mfcc_file}: {e}")
        return None, None

    # speed perturbation augmenting
    mfcc = _augment_speech(mfcc)

    return label, mfcc

def _augment_speech(mfcc):
    """
    Apply speed perturbation to MFCC data by randomly shifting frequencies.
    """
    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc

# Speech Corpus
class SpeechCorpus(object):

    def __init__(self, batch_size=16, set_name='train'):
        """
        Initialize the SpeechCorpus with a given batch size and set name.
        Load and preprocess the data from the corresponding metadata file.
        """
        # load meta file
        label, mfcc_file = [], []
        meta_file_path = os.path.join(_data_path, 'preprocess', 'meta', f'{set_name}.csv')
        
        try:
            with open(meta_file_path) as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                for row in reader:
                    # Construct the path to the MFCC file
                    mfcc_path = os.path.join(_data_path, 'preprocess', 'mfcc', f'{row[0]}.npy')
                    if os.path.exists(mfcc_path):
                        mfcc_file.append(mfcc_path)
                        # Convert label to a byte string for variable-length support
                        label.append(np.asarray(row[1:], dtype=np.int).tobytes())
                    else:
                        tf.sg_info(f"MFCC file not found: {mfcc_path}")
        except (IOError, csv.Error) as e:
            tf.sg_info(f"Error reading meta file {meta_file_path}: {e}")
            return

        if not label or not mfcc_file:
            tf.sg_info("No valid data found.")
            return

        # Convert lists to constant tensors
        label_t = tf.convert_to_tensor(label)
        mfcc_file_t = tf.convert_to_tensor(mfcc_file)

        # Create queue from constant tensors
        label_q, mfcc_file_q = tf.train.slice_input_producer([label_t, mfcc_file_t], shuffle=True)

        # Create label and mfcc queues
        label_q, mfcc_q = _load_mfcc(source=[label_q, mfcc_file_q],
                                     dtypes=[tf.sg_intx, tf.sg_floatx],
                                     capacity=256, num_threads=64)

        # Create batch queue with dynamic padding
        batch_queue = tf.train.batch([label_q, mfcc_q], batch_size,
                                     shapes=[(None,), (20, None)],
                                     num_threads=64, capacity=batch_size*32,
                                     dynamic_pad=True)

        # Split data into label and mfcc tensors
        self.label, self.mfcc = batch_queue
        # Transpose mfcc to batch * time * dim format
        self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])
        # Calculate total batch count
        self.num_batch = len(label) // batch_size

        # Print dataset info
        tf.sg_info(f'{set_name.upper()} set loaded.(total data={len(label)}, total batch={self.num_batch})')