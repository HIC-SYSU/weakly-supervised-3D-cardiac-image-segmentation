import torch.nn as nn

from .decoder import MyronenkoDecoder, MirroredDecoder
from .encoder import MyronenkoEncoder
from .basic_block import conv1x1x1
from copy import deepcopy
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                 encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None,
                 decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.base_width = base_width
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.encoder = encoder_class(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size)
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                                decoder_blocks)
        self.decoder = decoder_class(base_width=base_width, layer_blocks=decoder_blocks,
                                     upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size)
        self.set_final_convolution(n_features)
        self.set_activation(activation=activation)

    def set_final_convolution(self, n_outputs):
        self.final_convolution = conv1x1x1(in_planes=self.base_width, out_planes=n_outputs, stride=1)

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):
        if decoder_mirrors_encoder:
            decoder_blocks = encoder_blocks
            if decoder_class is None:
                decoder_class = MirroredDecoder
        elif decoder_blocks is None:
            decoder_blocks = [1] * len(encoder_blocks)
            if decoder_class is None:
                decoder_class = MyronenkoDecoder
        return decoder_class, decoder_blocks

    def forward(self, x):
        x = self.encoder(x)
        # return x[-1]#新加的
        features=x
        x = self.decoder(x)
        #return x
        x = self.final_convolution(x)
        if self.activation is not None:
            x = self.activation(x)
        return x#,features