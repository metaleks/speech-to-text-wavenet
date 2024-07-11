import sugartensor as tf

# Function to calculate logits using atrous convolution
def get_logit(x, voca_size, num_blocks=3, num_dim=128):

    # Define a residual block with atrous convolution
    def res_block(tensor, size, rate, block, dim=num_dim):
      
        with tf.sg_context(name='block_%d_%d' % (block, rate)):

            # Filter convolution with tanh activation
            conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')

            # Gate convolution with sigmoid activation
            conv_gate = tensor.sg_aconv1d(size=size, rate=rate, act='sigmoid', bn=True, name='conv_gate')

            # Output by gate multiplying
            out = conv_filter * conv_gate

            # Final output
            out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')

            # Residual and skip output
            return out + tensor, out

    # Expand dimension
    with tf.sg_context(name='front'):
        z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')

    # Dilated conv block loop
    skip = 0  # Skip connections
    for i in range(num_blocks):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i, dim=num_dim)
            skip += s

    # Final logit layers
    with tf.sg_context(name='logit'):
        logit = (skip
                 .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                 .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

    return logit
