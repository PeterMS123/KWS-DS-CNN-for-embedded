# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications Copyright  2018 Peter Mølgaard Sørensen
# Adapted from label_wav.py to extract and quantize weights/biases
#
#
"""
Loads a frozen graphDef model and extracts weights/biases. Names of tensorflow-nodes containg weights must be specified.
Batch-norm parameters are fused into preceding convolution parameters and then quantized based on dynamic ranges of each
layer. Quantized weights/biases are exported to .h files for C++ implementation.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import tensor_util
import argparse
import sys
import numpy as np
import os
import tensorflow as tf
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

# pylint: enable=unused-import

FLAGS = None
Fs = 16000



def main(_):
    eps = 0.001
    with tf.Session() as sess:
        print("load graph")
        with tf.gfile.FastGFile(FLAGS.graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes = [n for n in graph_def.node]
    wts = [n for n in graph_nodes if n.op == 'Const']

    print("Number of nodes - "+ str(len(wts)))
    print("deleting old weights file")
    os.remove('weights.h')
    #Layer information [weights, biases, channel means, channel variances, input fractional bits, output fractional bits, name for .h file]
    conv_1 = ['DS-CNN/conv_1/weights','DS-CNN/conv_1/biases','DS-CNN/conv_1/batch_norm/moving_mean','DS-CNN/conv_1/batch_norm/moving_variance',3,4,'CONV1','DS-CNN/conv_1/batch_norm/beta']
    dw_conv_1 = ['DS-CNN/conv_ds_1/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_1/depthwise_conv/biases', 'DS-CNN/conv_ds_1/dw_batch_norm/moving_mean',
              'DS-CNN/conv_ds_1/dw_batch_norm/moving_variance',4,4,'DW_CONV1','DS-CNN/conv_ds_1/dw_batch_norm/beta']
    pw_conv_1 = ['DS-CNN/conv_ds_1/pointwise_conv/weights', 'DS-CNN/conv_ds_1/pointwise_conv/biases',
                 'DS-CNN/conv_ds_1/pw_batch_norm/moving_mean','DS-CNN/conv_ds_1/pw_batch_norm/moving_variance',4,4,'PW_CONV1','DS-CNN/conv_ds_1/pw_batch_norm/beta']
    dw_conv_2 = ['DS-CNN/conv_ds_2/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_2/depthwise_conv/biases',
                 'DS-CNN/conv_ds_2/dw_batch_norm/moving_mean',
                 'DS-CNN/conv_ds_2/dw_batch_norm/moving_variance', 4, 4,'DW_CONV2','DS-CNN/conv_ds_2/dw_batch_norm/beta']
    pw_conv_2 = ['DS-CNN/conv_ds_2/pointwise_conv/weights', 'DS-CNN/conv_ds_2/pointwise_conv/biases',
                 'DS-CNN/conv_ds_2/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_2/pw_batch_norm/moving_variance', 4, 4,'PW_CONV2','DS-CNN/conv_ds_2/pw_batch_norm/beta']
    dw_conv_3 = ['DS-CNN/conv_ds_3/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_3/depthwise_conv/biases',
                 'DS-CNN/conv_ds_3/dw_batch_norm/moving_mean',
                 'DS-CNN/conv_ds_3/dw_batch_norm/moving_variance', 4, 4, 'DW_CONV3','DS-CNN/conv_ds_3/dw_batch_norm/beta']
    pw_conv_3 = ['DS-CNN/conv_ds_3/pointwise_conv/weights', 'DS-CNN/conv_ds_3/pointwise_conv/biases',
                 'DS-CNN/conv_ds_3/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_3/pw_batch_norm/moving_variance', 4, 4,
                 'PW_CONV3','DS-CNN/conv_ds_3/pw_batch_norm/beta']
    dw_conv_4 = ['DS-CNN/conv_ds_4/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_4/depthwise_conv/biases',
                 'DS-CNN/conv_ds_4/dw_batch_norm/moving_mean',
                 'DS-CNN/conv_ds_4/dw_batch_norm/moving_variance', 4, 4, 'DW_CONV4','DS-CNN/conv_ds_4/dw_batch_norm/beta']
    pw_conv_4 = ['DS-CNN/conv_ds_4/pointwise_conv/weights', 'DS-CNN/conv_ds_4/pointwise_conv/biases',
                 'DS-CNN/conv_ds_4/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_4/pw_batch_norm/moving_variance', 4, 4,
                 'PW_CONV4','DS-CNN/conv_ds_4/pw_batch_norm/beta']
    dw_conv_5 = ['DS-CNN/conv_ds_5/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_5/depthwise_conv/biases',
                 'DS-CNN/conv_ds_5/dw_batch_norm/moving_mean',
                 'DS-CNN/conv_ds_5/dw_batch_norm/moving_variance', 4, 4, 'DW_CONV5',
                 'DS-CNN/conv_ds_5/dw_batch_norm/beta']
    pw_conv_5 = ['DS-CNN/conv_ds_5/pointwise_conv/weights', 'DS-CNN/conv_ds_5/pointwise_conv/biases',
                 'DS-CNN/conv_ds_5/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_5/pw_batch_norm/moving_variance', 4, 4,
                 'PW_CONV5', 'DS-CNN/conv_ds_5/pw_batch_norm/beta']
    dw_conv_6 = ['DS-CNN/conv_ds_6/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_6/depthwise_conv/biases',
                 'DS-CNN/conv_ds_6/dw_batch_norm/moving_mean',
                 'DS-CNN/conv_ds_6/dw_batch_norm/moving_variance', 4, 4, 'DW_CONV6',
                 'DS-CNN/conv_ds_6/dw_batch_norm/beta']
    pw_conv_6 = ['DS-CNN/conv_ds_6/pointwise_conv/weights', 'DS-CNN/conv_ds_6/pointwise_conv/biases',
                 'DS-CNN/conv_ds_6/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_6/pw_batch_norm/moving_variance', 4, 5,
                 'PW_CONV6', 'DS-CNN/conv_ds_6/pw_batch_norm/beta']
    layer_list = [conv_1, dw_conv_1, pw_conv_1,dw_conv_2, pw_conv_2, dw_conv_3,pw_conv_3,dw_conv_4,pw_conv_4,dw_conv_5, pw_conv_5,dw_conv_6, pw_conv_6]
    n_filters = 76
    print(len(layer_list))

    for layer in layer_list:
        layer_name = layer[6]
        PW =False
        if(layer_name[0:2]=='PW'):
            PW=True
        DW = False
        if (layer_name[0:2] == 'DW'):
            DW = True
        print("Name of node - " + layer[6])
        print(PW)
        for n in wts:
            if n.name == layer[0]:
                weights = tensor_util.MakeNdarray(n.attr['value'].tensor)
            if n.name == layer[1]:
                bias = tensor_util.MakeNdarray(n.attr['value'].tensor)
            if n.name == layer[2]:
                mean = tensor_util.MakeNdarray(n.attr['value'].tensor)
            if n.name == layer[3]:
                var = tensor_util.MakeNdarray(n.attr['value'].tensor)
            if n.name == layer[7]:
                beta = tensor_util.MakeNdarray(n.attr['value'].tensor)
        print(weights.shape)
        weights =weights.squeeze()
        print(weights.shape)
        weights_t1 = np.zeros(weights.shape)
        bias_t1 = np.zeros((1,n_filters))
        print('weights - '+ str(weights.shape))
        print('Bias - ' + str(bias.shape))
        print('Mean - '+str(mean.shape))
        print('variance - ' + str(var.shape))
        for i in range(0,len(bias)):
            if(PW):
                filter = weights[:, i]
            else:
                filter = weights[:,:,i]
            bias_temp = bias[i]
            mean_temp = mean[i]
            var_temp = var[i]
            beta_temp = beta[i]
            new_filter = filter/math.sqrt(var_temp+eps)
            new_bias = beta_temp +(bias_temp -mean_temp)/(math.sqrt(var_temp+eps))
            if(PW):
                weights_t1[:,i]=new_filter
            else:

                weights_t1[:,:,i] = new_filter
            bias_t1[0,i]=new_bias
            if(i==0):
                print('filters : ' +str(filter))
                print('Bias : '+str(bias_temp))
                print('Mean : '+ str(mean_temp))
                print('Variance : '+str(var_temp))

                print("New filter : " + str(new_filter))
                print("New Bias : "+ str(new_bias))

        print("Combined new weights shape: "+str(weights_t1.shape))
        if not PW:
            print(weights_t1[:,:,0])
        print(bias_t1.shape)
        print(bias_t1)
        min_value = weights_t1.min()
        max_value = weights_t1.max()
        int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        dec_bits_weight = min(7 - int_bits, 111)
        weights_quant = np.round(weights_t1 * 2 ** dec_bits_weight)
        print("input fractional bits: " +str(layer[4]))
        print("Weights min value: "+str(min_value))
        print("Weights max value: "+str(max_value))
        print("Weights fractional bits: "+str(dec_bits_weight))
        min_value = bias_t1.min()
        max_value = bias_t1.max()
        int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        dec_bits_bias = min(7 - int_bits, 10000)
        bias_quant = np.round(bias_t1 * 2 ** dec_bits_bias)
        bias_left_shift = layer[4]+dec_bits_weight-dec_bits_bias
        output_right_shift = layer[4] + dec_bits_weight - layer[5]
        print("Bias min value: " + str(min_value))
        print("Bias max value: " + str(max_value))
        print("Bias fractional bits: " + str(dec_bits_bias))
        print("output fractional bits: " +str(layer[5]))
        print("Bias left shift: "+ str(bias_left_shift))
        print("Output right shift: " + str(output_right_shift))
        if(PW):
            print("Quantized weights (filter 0):\n " + str(weights_quant[:,  0]))
            print("Quantized weights (filter 1):\n " + str(weights_quant[:,  1]))
        else:
            print("Quantized weights (filter 0):\n "+str(weights_quant[:,:,0]))
            print("Quantized weights (filter 1):\n " + str(weights_quant[:, :, 1]))

        #print("Quantized biases: \n "+str(bias_quant))
        print(weights_quant.shape)

        if ( PW):
            w_reshaped = np.zeros(weights_quant.size)
            for ch_out in range(0,n_filters):
                for ch_in in range(0,n_filters):
                    w_reshaped[ch_in + ch_out*n_filters] = weights_quant[ch_in,ch_out]
        else:
            w_reshaped = np.zeros(weights_quant.size)
            if (DW): # Detphwise conv TFC (HWC)
                filt_f = 3
                filt_t = 3

                for t in range(0, filt_t):
                    for f in range(0, filt_f):
                        for ch in range(0,n_filters):
                            w_reshaped[f*n_filters + t*n_filters*filt_f + ch] = weights_quant[t, f, ch]
            else: # Regular conv CTF (CHW)
                filt_f =4
                filt_t = 10

                for t in range(0,filt_t):
                    for f in range(0,filt_f):
                        for ch in range(0,n_filters):
                            w_reshaped[ch+ f*n_filters + t*filt_f*n_filters]=weights_quant[t,f,ch]




        print("element 0: "+str(w_reshaped[0]))
        print("element 1: " + str(w_reshaped[1]))
        print("element 5: " + str(w_reshaped[5]))
        print("element 40: " + str(w_reshaped[40]))
        print("element "+str(n_filters)+": " + str(w_reshaped[n_filters]))
        name = layer[6] + '_WEIGHTS'

        with open('weights.h', 'a') as f:
           f.write('#define ' + name + ' {')
        with open('weights.h', 'ab') as f:
           np.savetxt(f, w_reshaped, fmt='%d', delimiter=', ', newline=', ')
        with open('weights.h', 'a') as f:
           f.write('}\n')

        name = layer[6] + '_BIAS'
        with open('weights.h', 'a') as f:
           f.write('#define ' + name + ' {')
        with open('weights.h', 'ab') as f:
           np.savetxt(f, bias_quant, fmt='%d', delimiter=', ', newline=', ')
        with open('weights.h', 'a') as f:
           f.write('}\n')

        name = layer[6] + '_BIAS_LEFT_SHIFT'
        with open('weights.h', 'a') as f:
            f.write('#define ' + name +' '+ str(bias_left_shift) +'\n')

        name = layer[6] + '_OUTPUT_RIGHT_SHIFT'
        with open('weights.h', 'a') as f:
            f.write('#define ' + name +' '+ str(output_right_shift) +'\n')


    fc_layer = ['DS-CNN/fc1/weights','DS-CNN/fc1/biases',5,3,'FC']
    for n in wts:
        if n.name == fc_layer[0]:
            weights = tensor_util.MakeNdarray(n.attr['value'].tensor)
        if n.name == fc_layer[1]:
            bias = tensor_util.MakeNdarray(n.attr['value'].tensor)
    print("FC weights : "+str(weights.shape))
    print(weights)
    print("FC bias : " + str(bias.shape))
    print(bias)
    min_value = weights.min()
    max_value = weights.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    dec_bits_weight = min(7 - int_bits, 111)
    weights_quant = np.round(weights * 2 ** dec_bits_weight)
    print("input fractional bits: " + str(fc_layer[2]))
    print("Weights min value: " + str(min_value))
    print("Weights max value: " + str(max_value))
    print("Weights fractional bits: " + str(dec_bits_weight))
    min_value = bias.min()
    max_value = bias.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
    dec_bits_bias = min(7 - int_bits, 10000)
    bias_quant = np.round(bias * 2 ** dec_bits_bias)
    bias_left_shift = fc_layer[2] + dec_bits_weight - dec_bits_bias
    output_right_shift = fc_layer[2] + dec_bits_weight - fc_layer[3]
    print("Bias min value: " + str(min_value))
    print("Bias max value: " + str(max_value))
    print("Bias fractional bits: " + str(dec_bits_bias))
    print("output fractional bits: " + str(fc_layer[3]))
    print("Bias left shift: " + str(bias_left_shift))
    print("Output right shift: " + str(output_right_shift))
    print("node 0 weights: ")
    print(weights_quant[:,0])
    print("node 1 weights: ")
    print(weights_quant[:, 1])
    print("Shape of weights: " +str(weights_quant.shape) )
    w_reshaped = np.zeros(weights_quant.size)
    for node_out in range(0,12):
        for node_in in range(0,n_filters):
            w_reshaped[node_out*n_filters+node_in]= weights_quant[node_in,node_out]
    print("element 0: " + str(w_reshaped[0]))
    print("element 1: " + str(w_reshaped[1]))
    print("element 172: " + str(w_reshaped[172]))
    print("element 173: " + str(w_reshaped[173]))
    name = fc_layer[4]+'_WEIGHTS'
    with open('weights.h','a') as f:
        f.write('#define '+name +' {')
    with open('weights.h','ab') as f:
        np.savetxt(f,w_reshaped,fmt='%d',delimiter=', ',newline=', ')
    with open('weights.h','a') as f:
        f.write('}\n')

    name = fc_layer[4] + '_BIAS'
    with open('weights.h', 'a') as f:
        f.write('#define ' + name + ' {')
    with open('weights.h', 'ab') as f:
        np.savetxt(f, bias_quant.transpose(), fmt='%d', delimiter=', ', newline=', ')
    with open('weights.h', 'a') as f:
        f.write('}\n')

    name = fc_layer[4] + '_BIAS_LEFT_SHIFT'
    with open('weights.h', 'a') as f:
        f.write('#define ' + name + ' ' + str(bias_left_shift) + '\n')

    name = fc_layer[4] + '_OUTPUT_RIGHT_SHIFT'
    with open('weights.h', 'a') as f:
        f.write('#define ' + name + ' ' + str(output_right_shift) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph', type=str, default='', help='Model to use for identification.')
    parser.add_argument(
        '--result_path',
        type=str,
        default='/tmp/KWS-tensorflow-results',
        help='Where to put test results')
    parser.add_argument(
        '--input_name',
        type=str,
        default='wav_data:0',
        help='Name of WAVE data input node in model.')
    parser.add_argument(
        '--output_name',
        type=str,
        default='labels_softmax:0',
        help='Name of node outputting a prediction in the model.')
    parser.add_argument(
        '--how_many_labels',
        type=int,
        default=3,
        help='Number of results to show.')


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
