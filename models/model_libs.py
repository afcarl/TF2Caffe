# --------------------------------------------------------
# TF2Caffe, model_libs.py
# Copyright (c) 2016
# Haozhi Qi (https://github.com/Oh233)
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from caffe import layers
from caffe import params


def unpack_variable(var, num):
    assert len > 0
    if type(var) is list and len(var) == num:
        return var
    else:
        ret = []
        if type(var) is list:
            assert len(var) == 1
            for i in xrange(0, num):
                ret.append(var[0])
        else:
            for i in xrange(0, num):
                ret.append(var)
    return ret


def conv_bn_layer(net, in_layer, out_layer, use_bn, use_relu,
                  num_output, kernel_size, pad, stride,
                  dilation=1, use_scale=True, eps=0.001,
                  conv_prefix='', conv_postfix='',
                  bn_prefix='', bn_postfix='_bn',
                  scale_prefix='', scale_postfix='_scale'):
    assert dilation == 1, 'dilated convolution is not supported yet'

    if use_bn:
        conv_kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'weight_filler': dict(type='gaussian', std=0.01),
            'bias_term': False,
        }

        bn_kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0),
                      dict(lr_mult=0, decay_mult=0),
                      dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
        }

        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [dict(lr_mult=1, decay_mult=0),
                          dict(lr_mult=1, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
            }

    # Add convolutional layer
    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = unpack_variable(kernel_size, 2)
    [pad_h, pad_w] = unpack_variable(pad, 2)
    [stride_h, stride_w] = unpack_variable(stride, 2)

    if kernel_h == kernel_w:
        net[conv_name] = layers.Convolution(net[in_layer],
                                            num_output=num_output, kernel_size=kernel_h,
                                            pad=pad_h, stride=stride_h,
                                            **conv_kwargs)
    else:
        net[conv_name] = layers.Convolution(net[in_layer],
                                            num_output=num_output,
                                            kernel_h=kernel_h, kernel_w=kernel_w,
                                            pad_h=pad_h, pad_w=pad_w,
                                            stride_h=stride_h, stride_w=stride_w,
                                            **conv_kwargs)

    # Add BN
    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = layers.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = layers.Scale(net[bn_name], in_place=True, **sb_kwargs)

    # Add ReLU
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = layers.ReLU(net[conv_name], in_place=True)

