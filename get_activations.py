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
# Adapted from label_wav.py to run inference on a large number of speech samples
# and generate statistical parameters of activations
#
"""
Loads a frozen graphDef model and runs inference on approximately 1000 speech samples from the test data. Outputs
distributions of activations of each layer. This is useful for determining dynamic ranges for fixed-point implementation

"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import argparse
import os.path
import sys
import math
from tensorflow.python.framework import tensor_util
import tensorflow.contrib.slim as slim
from scipy.stats import norm
from scipy.stats import poisson
wav_name = '/tmp/speech_dataset_train_clean/stop/0b56bcfe_nohash_0.wav'
text_file = open("/tmp/speech_dataset_train_clean/testing_list.txt", "r")
lines = text_file.readlines()
print(len(lines))
text_file.close()
print(lines[0])
print(lines[1])

with tf.gfile.FastGFile('C:/tmp\KWS-tensorflow-results/log_mel_20_noisy/DS_CNN_7layers_76f_lm20n/DS_CNN_7layers_76f_lm20n.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    log_mel_total = np.array([])
    conv1_total = np.array([])
    dw_conv1_total = np.array([])
    pw_conv1_total = np.array([])
    dw_conv2_total = np.array([])
    pw_conv2_total = np.array([])
    dw_conv3_total = np.array([])
    pw_conv3_total = np.array([])
    dw_conv4_total = np.array([])
    pw_conv4_total = np.array([])
    dw_conv5_total = np.array([])
    pw_conv5_total = np.array([])
    dw_conv6_total = np.array([])
    pw_conv6_total = np.array([])
    avg_pool_total = np.array([])
    FC_total = np.array([])


    with tf.Session() as sess:
        log_mel = sess.graph.get_tensor_by_name('Reshape_1:0')
        conv1 = sess.graph.get_tensor_by_name('DS-CNN/conv_1/batch_norm/FusedBatchNorm:0')
        dw_conv1 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_1/dw_batch_norm/FusedBatchNorm:0')
        pw_conv1 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_1/pw_batch_norm/FusedBatchNorm:0')
        dw_conv2 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_2/dw_batch_norm/FusedBatchNorm:0')
        pw_conv2 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_2/pw_batch_norm/FusedBatchNorm:0')
        dw_conv3 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_3/dw_batch_norm/FusedBatchNorm:0')
        pw_conv3 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_3/pw_batch_norm/FusedBatchNorm:0')
        dw_conv4 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_4/dw_batch_norm/FusedBatchNorm:0')
        pw_conv4 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_4/pw_batch_norm/FusedBatchNorm:0')
        dw_conv5 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_5/dw_batch_norm/FusedBatchNorm:0')
        pw_conv5 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_5/pw_batch_norm/FusedBatchNorm:0')
        dw_conv6 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_6/dw_batch_norm/FusedBatchNorm:0')
        pw_conv6 = sess.graph.get_tensor_by_name('DS-CNN/conv_ds_6/pw_batch_norm/FusedBatchNorm:0')
        avg_pool = sess.graph.get_tensor_by_name('DS-CNN/avg_pool/AvgPool:0')
        FC = sess.graph.get_tensor_by_name('DS-CNN/fc1/BiasAdd:0')

        for i in range(0,round(len(lines)/5)):
            i = i *5
            if (i%200 == 0):
                print(str(i)+'/'+str(len(lines)))
            wav_name = '/tmp/speech_dataset_train_clean/'+lines[i]
            wav_name = wav_name[:-1]
            with open(wav_name, 'rb') as wav_file:
                wav_data = wav_file.read()
            softmax_tensor = sess.graph.get_tensor_by_name('labels_softmax:0')
            predictions,log_mel_out,conv1_out, dw_conv1_out, pw_conv1_out, dw_conv2_out, pw_conv2_out, dw_conv3_out, pw_conv3_out \
                , dw_conv4_out, pw_conv4_out, dw_conv5_out, pw_conv5_out, dw_conv6_out, pw_conv6_out,\
                avg_pool_out, FC_out = sess.run([softmax_tensor,log_mel,conv1,dw_conv1,pw_conv1,dw_conv2,pw_conv2,dw_conv3,pw_conv3
                                                            , dw_conv4, pw_conv4,dw_conv5,pw_conv5,dw_conv6,pw_conv6,
                                                 avg_pool,FC], {'wav_data:0': wav_data})
            #text_file = open("7layers_76f_node_names.txt", "w")
            #for n in tf.get_default_graph().as_graph_def().node:
            #    print(n.name)
            #    text_file.write(str(n.name)+'\n')
            #text_file.close()


            log_mel_total = np.append(log_mel_total,log_mel_out.flatten())
            conv1_total = np.append(conv1_total, conv1_out.flatten())
            dw_conv1_total = np.append(dw_conv1_total, dw_conv1_out.flatten())
            pw_conv1_total = np.append(pw_conv1_total, pw_conv1_out.flatten())
            dw_conv2_total = np.append(dw_conv2_total, dw_conv2_out.flatten())
            pw_conv2_total = np.append(pw_conv2_total, pw_conv2_out.flatten())
            dw_conv3_total = np.append(dw_conv3_total, dw_conv3_out.flatten())
            pw_conv3_total = np.append(pw_conv3_total, pw_conv3_out.flatten())
            dw_conv4_total = np.append(dw_conv4_total, dw_conv4_out.flatten())
            pw_conv4_total = np.append(pw_conv4_total, pw_conv4_out.flatten())
            dw_conv5_total = np.append(dw_conv5_total, dw_conv5_out.flatten())
            pw_conv5_total = np.append(pw_conv5_total, pw_conv5_out.flatten())
            dw_conv6_total = np.append(dw_conv6_total, dw_conv6_out.flatten())
            pw_conv6_total = np.append(pw_conv6_total, pw_conv6_out.flatten())
            avg_pool_total = np.append(avg_pool_total, avg_pool_out.flatten())
            FC_total = np.append(FC_total, FC_out.flatten())


        log_mel_hist, log_mel_bin_edges =np.histogram(log_mel_total,bins = 100)
        mu, std = norm.fit(log_mel_total)
        vals = np.array([mu,std])
        np.savetxt('log_mel_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('log_mel_hist.csv', log_mel_hist, fmt='%.2f', delimiter=',')
        np.savetxt('log_mel_hist_bin_edges.csv', log_mel_bin_edges, fmt='%.3f', delimiter=',')
        conv1_hist, conv1_bin_edges = np.histogram(conv1_total, bins=100)
        mu, std = norm.fit(conv1_total)
        vals = np.array([mu, std])
        np.savetxt('conv1_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('conv1_hist.csv', conv1_hist, fmt='%.2f', delimiter=',')
        np.savetxt('conv1_hist_bin_edges.csv', conv1_bin_edges, fmt='%.3f', delimiter=',')

        dw_conv1_hist, dw_conv1_bin_edges = np.histogram(dw_conv1_total, bins=100)
        mu, std = norm.fit(dw_conv1_total)
        vals = np.array([mu, std])
        np.savetxt('dw_conv1_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('dw_conv1_hist.csv', dw_conv1_hist, fmt='%.2f', delimiter=',')
        np.savetxt('dw_conv1_hist_bin_edges.csv', dw_conv1_bin_edges, fmt='%.3f', delimiter=',')
        pw_conv1_hist, pw_conv1_bin_edges = np.histogram(pw_conv1_total, bins=100)
        mu, std = norm.fit(pw_conv1_total)
        vals = np.array([mu, std])
        np.savetxt('pw_conv1_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('pw_conv1_hist.csv', pw_conv1_hist, fmt='%.2f', delimiter=',')
        np.savetxt('pw_conv1_hist_bin_edges.csv', pw_conv1_bin_edges, fmt='%.3f', delimiter=',')

        dw_conv2_hist, dw_conv2_bin_edges = np.histogram(dw_conv2_total, bins=100)
        mu, std = norm.fit(dw_conv2_total)
        vals = np.array([mu, std])
        np.savetxt('dw_conv2_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('dw_conv2_hist.csv', dw_conv2_hist, fmt='%.2f', delimiter=',')
        np.savetxt('dw_conv2_hist_bin_edges.csv', dw_conv2_bin_edges, fmt='%.3f', delimiter=',')
        pw_conv2_hist, pw_conv2_bin_edges = np.histogram(pw_conv2_total, bins=100)
        mu, std = norm.fit(pw_conv2_total)
        vals = np.array([mu, std])
        np.savetxt('pw_conv2_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('pw_conv2_hist.csv', pw_conv2_hist, fmt='%.2f', delimiter=',')
        np.savetxt('pw_conv2_hist_bin_edges.csv', pw_conv2_bin_edges, fmt='%.3f', delimiter=',')

        dw_conv3_hist, dw_conv3_bin_edges = np.histogram(dw_conv3_total, bins=100)
        mu, std = norm.fit(dw_conv3_total)
        vals = np.array([mu, std])
        np.savetxt('dw_conv3_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('dw_conv3_hist.csv', dw_conv3_hist, fmt='%.2f', delimiter=',')
        np.savetxt('dw_conv3_hist_bin_edges.csv', dw_conv3_bin_edges, fmt='%.3f', delimiter=',')
        pw_conv3_hist, pw_conv3_bin_edges = np.histogram(pw_conv3_total, bins=100)
        mu, std = norm.fit(pw_conv3_total)
        vals = np.array([mu, std])
        np.savetxt('pw_conv3_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('pw_conv3_hist.csv', pw_conv3_hist, fmt='%.2f', delimiter=',')
        np.savetxt('pw_conv3_hist_bin_edges.csv', pw_conv3_bin_edges, fmt='%.3f', delimiter=',')

        dw_conv4_hist, dw_conv4_bin_edges = np.histogram(dw_conv4_total, bins=100)
        mu, std = norm.fit(dw_conv4_total)
        vals = np.array([mu, std])
        np.savetxt('dw_conv4_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('dw_conv4_hist.csv', dw_conv4_hist, fmt='%.2f', delimiter=',')
        np.savetxt('dw_conv4_hist_bin_edges.csv', dw_conv4_bin_edges, fmt='%.3f', delimiter=',')
        mu, std = norm.fit(pw_conv4_total)
        vals = np.array([mu, std])
        np.savetxt('pw_conv4_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        pw_conv4_hist, pw_conv4_bin_edges = np.histogram(pw_conv4_total, bins=100)
        np.savetxt('pw_conv4_hist.csv', pw_conv4_hist, fmt='%.2f', delimiter=',')
        np.savetxt('pw_conv4_hist_bin_edges.csv', pw_conv4_bin_edges, fmt='%.3f', delimiter=',')

        dw_conv5_hist, dw_conv5_bin_edges = np.histogram(dw_conv5_total, bins=100)
        mu, std = norm.fit(dw_conv5_total)
        vals = np.array([mu, std])
        np.savetxt('dw_conv5_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('dw_conv5_hist.csv', dw_conv5_hist, fmt='%.2f', delimiter=',')
        np.savetxt('dw_conv5_hist_bin_edges.csv', dw_conv5_bin_edges, fmt='%.3f', delimiter=',')
        pw_conv5_hist, pw_conv5_bin_edges = np.histogram(pw_conv5_total, bins=100)
        mu, std = norm.fit(pw_conv5_total)
        vals = np.array([mu, std])
        np.savetxt('pw_conv5_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('pw_conv5_hist.csv', pw_conv5_hist, fmt='%.2f', delimiter=',')
        np.savetxt('pw_conv5_hist_bin_edges.csv', pw_conv5_bin_edges, fmt='%.3f', delimiter=',')

        dw_conv6_hist, dw_conv6_bin_edges = np.histogram(dw_conv6_total, bins=100)
        mu, std = norm.fit(dw_conv6_total)
        vals = np.array([mu, std])
        np.savetxt('dw_conv6_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('dw_conv6_hist.csv', dw_conv6_hist, fmt='%.2f', delimiter=',')
        np.savetxt('dw_conv6_hist_bin_edges.csv', dw_conv6_bin_edges, fmt='%.3f', delimiter=',')
        pw_conv6_hist, pw_conv6_bin_edges = np.histogram(pw_conv6_total, bins=100)
        mu, std = norm.fit(pw_conv6_total)
        vals = np.array([mu, std])
        np.savetxt('pw_conv6_hist_stat.csv', vals, fmt='%.5f', delimiter=',')
        np.savetxt('pw_conv6_hist.csv', pw_conv6_hist, fmt='%.2f', delimiter=',')
        np.savetxt('pw_conv6_hist_bin_edges.csv', pw_conv6_bin_edges, fmt='%.3f', delimiter=',')

        avg_pool_hist, avg_pool_bin_edges = np.histogram(avg_pool_total, bins=100)
        np.savetxt('avg_pool_hist.csv', avg_pool_hist, fmt='%.2f', delimiter=',')
        np.savetxt('avg_pool_hist_bin_edges.csv', avg_pool_bin_edges, fmt='%.3f', delimiter=',')

        FC_hist, FC_bin_edges = np.histogram(FC_total, bins=100)
        np.savetxt('FC_hist.csv', FC_hist, fmt='%.2f', delimiter=',')
        np.savetxt('FC_hist_bin_edges.csv', FC_bin_edges, fmt='%.3f', delimiter=',')
        mu, std = norm.fit(FC_total)
        vals = np.array([mu, std])
        np.savetxt('FC_hist_stat.csv', vals, fmt='%.5f', delimiter=',')