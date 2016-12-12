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


def inception_block_35(net, bottom_layer, block_num, scale=1.0, repeat_name='Repeat'):
    for block_idx in xrange(block_num):
        top_layer_branch0 = '{}/block35_{}/Branch_0/Conv2d_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=1, pad=0, stride=1)

        top_layer_branch1 = '{}/block35_{}/Branch_1/Conv2d_0a_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch1 = top_layer_branch1
        top_layer_branch1 = '{}/block35_{}/Branch_1/Conv2d_0b_3x3'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=3, pad=1, stride=1)

        top_layer_branch2 = '{}/block35_{}/Branch_2/Conv2d_0a_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                      num_output=32, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch2 = top_layer_branch2
        top_layer_branch2 = '{}/block35_{}/Branch_2/Conv2d_0b_3x3'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                      num_output=48, kernel_size=3, pad=1, stride=1)
        bottom_layer_branch2 = top_layer_branch2
        top_layer_branch2 = '{}/block35_{}/Branch_2/Conv2d_0c_3x3'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                      num_output=64, kernel_size=3, pad=1, stride=1)

        top_layer_mixed = 'block35_{}_mixed'.format(block_idx + 1)
        net[top_layer_mixed] = layers.Concat(*[net[top_layer_branch0],
                                               net[top_layer_branch1],
                                               net[top_layer_branch2]], axis=1)
        bottom_layer_residual = top_layer_mixed
        top_layer_residual = '{}/block35_{}/Conv2d_1x1'.format(repeat_name, block_idx + 1)
        net[top_layer_residual] = layers.Convolution(net[bottom_layer_residual],
                                                     num_output=320, kernel_size=1, pad=0, stride=1)

        bottom_layer_shortcut = bottom_layer
        bottom_layer_residual = top_layer_residual
        top_layer = 'block35_{}_eltsum'.format(block_idx + 1)
        net[top_layer] = layers.Eltwise(*[net[bottom_layer_shortcut],
                                          net[bottom_layer_residual]], coeff=[1.0, scale])
        bottom_layer = top_layer
        relu_name = '{}_relu'.format(bottom_layer)
        net[relu_name] = layers.ReLU(net[bottom_layer], in_place=True)

    return top_layer


def inception_block_17(net, bottom_layer, block_num, scale=1.0, repeat_name='Repeat'):

    for block_idx in xrange(block_num):
        top_layer_branch0 = '{}/block17_{}/Branch_0/Conv2d_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=1, pad=0, stride=1)

        top_layer_branch1 = '{}/block17_{}/Branch_0/Conv2d_0a_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=128, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = '{}/block17_{}/Branch_0/Conv2d_0b_1x7'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=160, kernel_size=[1, 7], pad=[0, 3], stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = '{}/block17_{}/Branch_0/Conv2d_0c_7x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=1)

        top_layer_mixed = 'block17_{}_mixed'.format(block_idx + 1)
        net[top_layer_mixed] = layers.Concat(*[net[top_layer_branch0],
                                               net[top_layer_branch1]], axis=1)

        bottom_layer_residual = top_layer_mixed
        top_layer_residual = '{}/block17_{}/Conv2d_1x1'.format(repeat_name, block_idx + 1)
        net[top_layer_residual] = layers.Convolution(net[bottom_layer_residual],
                                                     num_output=1088, kernel_size=1, pad=0, stride=1)

        bottom_layer_shortcut = bottom_layer
        bottom_layer_residual = top_layer_residual
        top_layer = 'block17_{}_eltsum'.format(block_idx + 1)
        net[top_layer] = layers.Eltwise(*[net[bottom_layer_shortcut],
                                          net[bottom_layer_residual]], coeff=[1.0, scale])

        bottom_layer = top_layer
        relu_name = '{}_relu'.format(bottom_layer)
        net[relu_name] = layers.ReLU(net[bottom_layer], in_place=True)

    return top_layer


def inception_block_8(net, bottom_layer, block_num, scale=1.0, repeat_name='Repeat', apply_last_relu=True):

    for block_idx in xrange(block_num):
        top_layer_branch0 = '{}/block8_{}/Branch_0/Conv2d_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=1, pad=0, stride=1)

        top_layer_branch1 = '{}/block8_{}/Branch_1/Conv2d_0a_1x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=192, kernel_size=1, pad=0, stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = '{}/block8_{}/Branch_1/Conv2d_0b_1x3'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=224, kernel_size=[1, 3], pad=[0, 1], stride=1)
        bottom_layer_branch1 = top_layer_branch1

        top_layer_branch1 = '{}/block8_{}/Branch_1/Conv2d_0c_3x1'.format(repeat_name, block_idx + 1)
        conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                      num_output=256, kernel_size=[3, 1], pad=[1, 0], stride=1)

        top_layer_mixed = '{}/block8_{}_mixed'.format(repeat_name, block_idx + 1)
        net[top_layer_mixed] = layers.Concat(*[net[top_layer_branch0],
                                               net[top_layer_branch1]], axis=1)

        bottom_layer_residual = top_layer_mixed
        top_layer_residual = '{}/block8_{}/Conv2d_1x1'.format(repeat_name, block_idx + 1)
        net[top_layer_residual] = layers.Convolution(net[bottom_layer_residual],
                                                     num_output=2080, kernel_size=1, pad=0, stride=1)

        bottom_layer_shortcut = bottom_layer
        bottom_layer_residual = top_layer_residual
        top_layer = '{}/block8_{}/eltsum'.format(repeat_name, block_idx + 1)
        net[top_layer] = layers.Eltwise(*[net[bottom_layer_shortcut],
                                          net[bottom_layer_residual]], coeff=[1.0, scale])

        if block_idx + 1 < block_num or apply_last_relu:
            bottom_layer = top_layer
            relu_name = '{}_relu'.format(bottom_layer)
            net[relu_name] = layers.ReLU(net[bottom_layer], in_place=True)

    return top_layer


def mixed_5b(net, common_bottom_layer):
    # branch 0
    top_layer_branch0 = 'Mixed_5b/Branch_0/Conv2d_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=96, kernel_size=1, pad=0, stride=1)
    # branch 1
    top_layer_branch1 = 'Mixed_5b/Branch_1/Conv2d_0a_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=48, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'Mixed_5b/Branch_1/Conv2d_0b_5x5'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=5, pad=2, stride=1)
    # branch 2
    top_layer_branch2 = 'Mixed_5b/Branch_2/Conv2d_0a_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'Mixed_5b/Branch_2/Conv2d_0b_3x3'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=96, kernel_size=3, pad=1, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'Mixed_5b/Branch_2/Conv2d_0c_3x3'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=96, kernel_size=3, pad=1, stride=1)
    # branch 3
    top_layer_branch3 = 'mixed5b_branch3_avepool_0a'
    net[top_layer_branch3] = layers.Pooling(net[common_bottom_layer], pool=params.Pooling.AVE,
                                            kernel_size=3, stride=1, pad=1)

    bottom_layer_branch3 = top_layer_branch3
    top_layer_branch3 = 'Mixed_5b/Branch_3/Conv2d_0b_1x1'
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
    top_layer_branch0 = 'Mixed_6a/Branch_0/Conv2d_1a_3x3'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=384, kernel_size=3, pad=0, stride=2)
    # branch 1
    top_layer_branch1 = 'Mixed_6a/Branch_1/Conv2d_0a_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'Mixed_6a/Branch_1/Conv2d_0b_3x3'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=3, pad=1, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'Mixed_6a/Branch_1/Conv2d_1a_3x3'
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
    top_layer_branch0 = 'Mixed_7a/Branch_0/Conv2d_0a_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch0 = top_layer_branch0
    top_layer_branch0 = 'Mixed_7a/Branch_0/Conv2d_1a_3x3'
    conv_bn_layer(net, in_layer=bottom_layer_branch0, out_layer=top_layer_branch0, use_bn=True, use_relu=True,
                  num_output=384, kernel_size=3, pad=0, stride=2)
    # branch 1
    top_layer_branch1 = 'Mixed_7a/Branch_1/Conv2d_0a_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch1 = top_layer_branch1
    top_layer_branch1 = 'Mixed_7a/Branch_1/Conv2d_1a_3x3'
    conv_bn_layer(net, in_layer=bottom_layer_branch1, out_layer=top_layer_branch1, use_bn=True, use_relu=True,
                  num_output=288, kernel_size=3, pad=0, stride=2)

    # branch 2
    top_layer_branch2 = 'Mixed_7a/Branch_2/Conv2d_0a_1x1'
    conv_bn_layer(net, in_layer=common_bottom_layer, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=256, kernel_size=1, pad=0, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'Mixed_7a/Branch_2/Conv2d_0b_3x3'
    conv_bn_layer(net, in_layer=bottom_layer_branch2, out_layer=top_layer_branch2, use_bn=True, use_relu=True,
                  num_output=288, kernel_size=3, pad=1, stride=1)

    bottom_layer_branch2 = top_layer_branch2
    top_layer_branch2 = 'Mixed_7a/Branch_2/Conv2d_1a_3x3'
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
    top_layer = 'Conv2d_1a_3x3'
    conv_bn_layer(net, in_layer='data', out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=32, kernel_size=3, pad=0, stride=2)
    bottom_layer = top_layer
    # 147 x 147 x 32
    top_layer = 'Conv2d_2a_3x3'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=32, kernel_size=3, pad=0, stride=1)
    bottom_layer = top_layer
    # 147 x 147 x 64
    top_layer = 'Conv2d_2b_3x3'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=64, kernel_size=3, pad=0, stride=1)
    bottom_layer = top_layer
    # 73 x 73 x 64
    top_layer = 'maxpool_3a'
    net[top_layer] = layers.Pooling(net[bottom_layer], pool=params.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    bottom_layer = top_layer
    # 73 x 73 x 80
    top_layer = 'Conv2d_3b_1x1'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=80, kernel_size=1, pad=0, stride=1)
    bottom_layer = top_layer
    # 71 x 71 x 192
    top_layer = 'Conv2d_4a_3x3'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=192, kernel_size=3, pad=0, stride=1)
    bottom_layer = top_layer
    # 35 x 35 x 192
    top_layer = 'maxpool_5a'
    net[top_layer] = layers.Pooling(net[bottom_layer], pool=params.Pooling.MAX, kernel_size=3, stride=2, pad=0)
    bottom_layer = top_layer

    bottom_layer = mixed_5b(net, bottom_layer)
    # 35 x 35 x 320 (Mixed 5a)
    bottom_layer = inception_block_35(net, bottom_layer, 10, 0.17, repeat_name='Repeat')

    # 17 x 17 x 1088
    bottom_layer = mixed_6a(net, bottom_layer)

    bottom_layer = inception_block_17(net, bottom_layer, 20, 0.10, repeat_name='Repeat_1')

    bottom_layer = mixed_7a(net, bottom_layer)

    bottom_layer = inception_block_8(net, bottom_layer, 9, 0.20, repeat_name='Repeat_2', apply_last_relu=True)
    bottom_layer = inception_block_8(net, bottom_layer, 1, 0.20, repeat_name='', apply_last_relu=False)

    top_layer = 'Conv2d_7b_1x1'
    conv_bn_layer(net, in_layer=bottom_layer, out_layer=top_layer, use_bn=True, use_relu=True,
                  num_output=1536, kernel_size=1, pad=0, stride=1)

    with open('gg.prototxt', 'w') as f:
        print(net.to_proto(), file=f)
