from keras import backend as K
from keras.engine.topology import Layer
import itertools


class SpaceToDepth(Layer):
    '''
    keras implementation of space_to_depth as a layer
    '''
    def __init__(self, scale_factor=2, **kwargs):
        self.scale_factor = scale_factor
        self.dim_ordering = K.image_dim_ordering()
        if self.dim_ordering == 'channels_first':
            self.channls_first = True
        else:
            self.channels_first = False
        super(SpaceToDepth, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpaceToDepth, self).build(input_shape)

    def call(self, x):
        """
        Uses phase shift algorithm to convert channels/depth for spatial resolution
        """
        scale = 2
        data_format = K.image_dim_ordering().lower()
        if self.channels_first == True:
            b, k, row, col = K.int_shape(x)
            if b ==  None: b = 0
            output_shape = (b, k // (scale ** 2), row * scale, col * scale)
            out = K.zeros(output_shape)
            r = scale
            for y, x in itertools.product(range(scale), repeat=2):
                out = K.update_add(out[:, :, y::r, x::r], x[:, r * y + x:: r * r, :, :])
        else:
            b, row, col, k = K.int_shape(x)
            if b ==  None: b = 0
            output_shape = (b, row * scale, col * scale, k // (scale ** 2))
            out = K.zeros(output_shape)
            r = scale
            for y, x in itertools.product(range(scale), repeat=2):
                out = K.update_add(out[:, y::r, x::r, :], x[:, :, :, r * y + x:: r * r])
        return out#K._postprocess_conv2d_output(out, x, None, None, None, data_format)


    def compute_output_shape(self, input_shape):
        """Determine SpaceToDepth output shape
        """
        if self.channels_first == True:
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))