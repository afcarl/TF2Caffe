# --------------------------------------------------------
# TF2Caffe, net_def.py
# Copyright (c) 2016
# Haozhi Qi (https://github.com/Oh233)
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import print_function
from caffe import layers
from caffe import params
from model_libs import conv_bn_layer


def inception_block_35(net, bottom_layer, block_num, scale=1.0):
    for block_idx in xrange(block_num):
        top_layer_branch0 = 'block35_{}_branch0_conv_0'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=1, pad=0, stride=1)

        top_layer_branch1 = 'block35_{}_branch1_conv_0a'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch1 = top_layer_branch1
        top_layer_branch1 = 'block35_{}_branch1_conv_0b'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=3, pad=1, stride=1)

        top_layer_branch2 = 'block35_{}_branch2_conv_0a'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch2 = top_layer_branch2
        top_layer_branch2 = 'block35_{}_branch2_conv_0b'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                      num_output=48, kernel_size=3, pad=1, stride=1)
        bottom_layer_branch2 = top_layer_branch2
        top_layer_branch2 = 'block35_{}_branch2_conv_0c'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                      num_output=64, kernel_size=3, pad=1, stride=1)

        top_layer_mixed = 'block35_{}_mixed'.format(block_idx)
        net[top_layer_mixed] = layers.Concat(*[net[top_layer_branch0],
                                               net[top_layer_branch1],
                                               net[top_layer_branch2]], axis=1)
        bottom_layer_residual = top_layer_mixed
        top_layer_residual = 'block35_{}_conv_1'.format(block_idx)
        net[top_layer_residual] = layers.Convolution(net[bottom_layer_residual],
                                                     num_output=320, kernel_size=1, pad=0, stride=1)

        bottom_layer_shortcut = bottom_layer
        bottom_layer_residual = top_layer_residual
        top_layer = 'block35_{}_eltsum'.format(block_idx)
        net[top_layer] = layers.Eltwise(*[net[bottom_layer_shortcut],
                                          net[bottom_layer_residual]], coeff=[1.0, scale])
        bottom_layer = top_layer
        relu_name = '{}_relu'.format(bottom_layer)
        net[relu_name] = layers.ReLU(net[bottom_layer], in_place=True)

    return top_layer


def inception_block_17(net, bottom_layer, block_num, scale=1.0):

    for block_idx in xrange(block_num):
        top_layer_branch0 = 'block17_{}_branch0_conv_0'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=1, pad=0, stride=1)

        top_layer_branch1 = 'block17_{}_branch1_conv_0a'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=128, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = 'block17_{}_branch1_conv_0b'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=160, kernel_size=[1, 7], pad=[0, 3], stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = 'block17_{}_branch1_conv_0c'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=1)

        top_layer_mixed = 'block17_{}_mixed'.format(block_idx)
        net[top_layer_mixed] = layers.Concat(*[net[top_layer_branch0],
                                               net[top_layer_branch1]], axis=1)

        bottom_layer_residual = top_layer_mixed
        top_layer_residual = 'block17_{}_conv_1'.format(block_idx)
        net[top_layer_residual] = layers.Convolution(net[bottom_layer_residual],
                                                     num_output=1088, kernel_size=1, pad=0, stride=1)

        bottom_layer_shortcut = bottom_layer
        bottom_layer_residual = top_layer_residual
        top_layer = 'block17_{}_eltsum'.format(block_idx)
        net[top_layer] = layers.Eltwise(*[net[bottom_layer_shortcut],
                                          net[bottom_layer_residual]], coeff=[1.0, scale])

        bottom_layer = top_layer
        relu_name = '{}_relu'.format(bottom_layer)
        net[relu_name] = layers.ReLU(net[bottom_layer], in_place=True)

    return top_layer


def inception_block_8(net, bottom_layer, block_num, scale=1.0, apply_last_relu=True):

    for block_idx in xrange(block_num):
        top_layer_branch0 = 'block8_{}_branch0_conv_0'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=1, pad=0, stride=1)

        top_layer_branch1 = 'block8_{}_branch1_conv_0a'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = 'block8_{}_branch1_conv_0b'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=224, kernel_size=[1, 3], pad=[0, 1], stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = 'block8_{}_branch1_conv_0c'.format(block_idx)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=256, kernel_size=[3, 1], pad=[1, 0], stride=1)

        top_layer_mixed = 'block8_{}_mixed'.format(block_idx)
        net[top_layer_mixed] = layers.Concat(*[net[top_layer_branch0],
                                               net[top_layer_branch1]], axis=1)

        bottom_layer_residual = top_layer_mixed
        top_layer_residual = 'block8_{}_conv_1'.format(block_idx)
        net[top_layer_residual] = layers.Convolution(net[bottom_layer_residual],
                                                     num_output=2080, kernel_size=1, pad=0, stride=1)

        bottom_layer_shortcut = bottom_layer
        bottom_layer_residual = top_layer_residual
        top_layer = 'block8_{}_eltsum'.format(block_idx)
        net[top_layer] = layers.Eltwise(*[net[bottom_layer_shortcut],
                                          net[bottom_layer_residual]], coeff=[1.0, scale])

        if block_idx < block_num - 1 or apply_last_relu:
            bottom_layer = top_layer
            relu_name = '{}_relu'.format(bottom_layer)
            net[relu_name] = layers.ReLU(net[bottom_layer], in_place=True)

    return top_layer


def mixed_5b(net, common_bottom_layer):
    # branch 0
    top_layer_branch0 = 'mixed5b_branch0_conv_0'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=96, kernel_size=1, pad=0, stride=1)
    # branch 1
    top_layer_branch1 = 'mixed5b_branch1_conv_0a'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=48, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'mixed5b_branch1_conv_0b'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=5, pad=2, stride=1)
    # branch 2
    top_layer_branch2 = 'mixed5b_branch2_conv_0a'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'mixed5b_branch2_conv_0b'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=96, kernel_size=3, pad=1, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'mixed5b_branch2_conv_0c'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=96, kernel_size=3, pad=1, stride=1)
    # branch 3
    top_layer_branch3 = 'mixed5b_branch3_avepool_0a'
    net[top_layer_branch3] = layers.Pooling(net[common_bottom_layer], pool=params.Pooling.AVE,
                                            kernel_size=3, stride=1, pad=1)

    bottom_layer_branch3 = top_layer_branch3
    top_layer_branch3 = 'mixed5b_branch3_conv_0b'
    conv_bn_layer(net, in_layer=bottom_layer_branch3, out_layer=top_layer_branch3, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=1, pad=0, stride=1)

    top_layer = 'mixed5b'
    net[top_layer] = layers.Concat(*[net[top_layer_branch0],
                                     net[top_layer_branch1],
                                     net[top_layer_branch2],
                                     net[top_layer_branch3]], axis=1)

    return top_layer


def mixed_6a(net, common_bottom_layer):
    # branch 0
    top_layer_branch0 = 'mixed6a_branch0_conv_0'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=384, kernel_size=3, pad=0, stride=2)
    # branch 1
    top_layer_branch1 = 'mixed6a_branch1_conv_0a'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'mixed6a_branch1_conv_0b'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=3, pad=1, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'mixed6a_branch1_conv_1a'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=384, kernel_size=3, pad=0, stride=2)

    # branch 2
    top_layer_branch2 = 'mixed6a_branch2_maxpool_0'
    net[top_layer_branch2] = layers.Pooling(net[common_bottom_layer], pool=params.Pooling.MAX,
                                            kernel_size=3, stride=2, pad=0)

    top_layer = 'mixed6a'
    net[top_layer] = layers.Concat(*[net[top_layer_branch0],
                                     net[top_layer_branch1],
                                     net[top_layer_branch2]], axis=1)
    return top_layer


def mixed_7a(net, common_bottom_layer):
    # branch 0
    top_layer_branch0 = 'mixed7a_branch0_conv_0a'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch0 = top_layer_branch0
    top_layer_branch0 = 'mixed7a_branch0_conv_1a'
    conv_bn_layer(net, in_layer=bottom_layer_branch0, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=384, kernel_size=3, pad=0, stride=2)
    # branch 1
    top_layer_branch1 = 'mixed7a_branch1_conv_0a'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'mixed7a_branch1_conv_1a'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=288, kernel_size=3, pad=0, stride=2)

    # branch 2
    top_layer_branch2 = 'mixed7a_branch2_conv_0a'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'mixed7a_branch2_conv_0b'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=288, kernel_size=3, pad=1, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'mixed7a_branch2_conv_1a'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=320, kernel_size=3, pad=0, stride=2)
    # branch 3
    top_layer_branch3 = 'mixed7a_branch3_maxpool_0'
    net[top_layer_branch3] = layers.Pooling(net[common_bottom_layer], pool=params.Pooling.MAX,
                                            kernel_size=3, stride=2, pad=0)

    top_layer = 'mixed7a'
    net[top_layer] = layers.Concat(*[net[top_layer_branch0],
                                     net[top_layer_branch1],
                                     net[top_layer_branch2],
                                     net[top_layer_branch3]], axis=1)
    return top_layer


def inception_resnet_v2(net):
    net['data'] = layers.DummyData(num=1, channels=3, height=299, width=299)
    # 149 x 149 x 32
    top_layer = 'conv_1a'
    conv_bn_layer(net, in_layer='data', out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=32, kernel_size=3, pad=0, stride=2)
    bottom_layer = top_layer
    # 147 x 147 x 32
    top_layer = 'conv_2a'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=32, kernel_size=3, pad=0, stride=1)
    bottom_layer = top_layer
    # 147 x 147 x 64
    top_layer = 'conv_2b'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=3, pad=0, stride=1)
    bottom_layer = top_layer
    # 73 x 73 x 64
    top_layer = 'maxpool_3a'
    net[top_layer] = layers.Pooling(net[bottom_layer], pool=params.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    bottom_layer = top_layer
    # 73 x 73 x 80
    top_layer = 'conv_3b'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=80, kernel_size=1, pad=0, stride=1)
    bottom_layer = top_layer
    # 71 x 71 x 192
    top_layer = 'conv_4a'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=192, kernel_size=3, pad=0, stride=1)
    bottom_layer = top_layer
    # 35 x 35 x 192
    top_layer = 'maxpool_5a'
    net[top_layer] = layers.Pooling(net[bottom_layer], pool=params.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    bottom_layer = top_layer

    bottom_layer = mixed_5b(net, bottom_layer)
    # 35 x 35 x 320 (Mixed 5a)
    bottom_layer = inception_block_35(net, bottom_layer, 10, 0.17)

    # 17 x 17 x 1088
    bottom_layer = mixed_6a(net, bottom_layer)

    bottom_layer = inception_block_17(net, bottom_layer, 20, 0.10)

    bottom_layer = mixed_7a(net, bottom_layer)

    bottom_layer = inception_block_8(net, bottom_layer, 10, 0.20, apply_last_relu=False)

    top_layer = 'conv_7b'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=1536, kernel_size=1, pad=0, stride=1)

    with open('gg.prototxt', 'w') as f:
        print(net.to_proto(), file=f)
