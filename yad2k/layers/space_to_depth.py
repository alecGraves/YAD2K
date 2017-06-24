from keras import backend as K
from keras.engine.topology import Layer, InputSpec

class SpaceToDepth(Layer):
    '''
    keras implementation of space_to_depth as a layer
    '''
    def __init__(self, block_size=2, **kwargs):
        self.block_size = block_size

        if K.image_dim_ordering() == 'channels_first':
            self.channls_first = True
        else:
            self.channels_first = False

        super(SpaceToDepth, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(SpaceToDepth, self).build(input_shape)

    def call(self, x, mask=None):
        """
        Uses phase shift algorithm to convert channels/depth for spatial resolution
        """
        out_shape = list(self.compute_output_shape(self.input_spec[0].shape))
        if out_shape[0] is None:
            out_shape[0] = 1
        #out = K.variable(K.placeholder(out_shape)*0)
        #out = K.variable(K.zeros_like(K.placeholder(out_shape))) 
        out = K.variable(K.placeholder(out_shape))
        #out = K.reshape(K.ones_like(x, dtype=K.floatx()), (-1, out_shape[1], out_shape[2], out_shape[3]))
        r = self.block_size
        if self.channels_first:
            for a, b in [(x, y) for x in range(r) for y in range(r)]: #itertools.product(range(r), repeat=2)
                K.update(out[:, r * a + b:: r * r, :, :], x[:, :, a::r, b::r])
                #out[:, r * a + b:: r * r, :, :] += x[:, :, a::r, b::r]
        else:
            for a, b in [(x, y) for x in range(r) for y in range(r)]:
                K.update(out[:, :, :, r * a + b:: r * r], x[:, a::r, b::r, :])
                #out[:, :, :, r * a + b:: r * r] += x[:, a::r, b::r, :]
        return out


    def compute_output_shape(self, input_shape):
        """
        Determine SpaceToDepth output shape
        """
        if self.channels_first:
            in_height = input_shape[2]
            in_width = input_shape[3]
            in_depth = input_shape[1]
        else:
            in_height = input_shape[1]
            in_width = input_shape[2]
            in_depth = input_shape[3]

        batch_size = input_shape[0]
        out_height = in_height // self.block_size
        out_width = in_width // self.block_size
        out_depth = in_depth * (self.block_size ** 2)

        if self.channels_first:
            return batch_size, out_depth, out_height, out_width
        else:
            return batch_size, out_height, out_width, out_depth