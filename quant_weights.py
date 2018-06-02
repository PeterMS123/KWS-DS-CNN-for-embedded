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
# Adapted from freeze.py, to create a checkpoint file with quantized weights
#
r"""
Loads a checkpoint file and quantizes weights based on bitwidths command line argument.
The quantized weights are then saved to a separate checkpoint file which can then be converted to a GraphDef file using
freeze.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import math
import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import input_data
import models
from tensorflow.python.framework import graph_util

FLAGS = None


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture, input_type,
                           model_size_info):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    clip_stride_ms: How often to run recognition. Useful for models with cache.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
  """


  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count,100)
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  #input_spectrogram = tf.placeholder(tf.float32, shape=[49,513], name='speech_signal')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)
  #spectrogram = input_spectrogram
  if (input_type == 'log-mel'):
      print("log-mel energies")
      # Warp the linear-scale, magnitude spectrograms into the mel-scale.
      num_spectrogram_bins = spectrogram.shape[-1].value  # magnitude_spectrograms.shape[-1].value
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, model_settings['dct_coefficient_count']
      linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, model_settings['sample_rate'], lower_edge_hertz,
          upper_edge_hertz)
      mel_spectrograms = tf.tensordot(
          spectrogram, linear_to_mel_weight_matrix, 1)
      # Note: Shape inference for `tf.tensordot` does not currently handle this case.
      mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))
      log_offset = 1e-6
      log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
      fingerprint_input = log_mel_spectrograms
  elif (input_type == 'MFCC'):
      print('MFCC-features')
      fingerprint_input = contrib_audio.mfcc(
          spectrogram,
          decoded_sample_data.sample_rate,
          dct_coefficient_count=model_settings['dct_coefficient_count'])
  #fingerprint_input = tf.placeholder(tf.float32,shape=[49,20],name='fingerprint')
  fingerprint_frequency_size = model_settings['dct_coefficient_count']
  fingerprint_time_size = model_settings['spectrogram_length']
  reshaped_input = tf.reshape(fingerprint_input, [
      -1, fingerprint_time_size * fingerprint_frequency_size
  ])

  logits,dropout_prob = models.create_model(
      reshaped_input, model_settings, model_architecture, model_size_info,
      is_training=True, runtime_settings=runtime_settings)

  # Create an output to use for inference.
  tf.nn.softmax(logits, name='labels_softmax')


def main(_):

  print(FLAGS.model_size_info)
  reg_conv_bits = FLAGS.bit_widths[0]
  dw_conv_bits = FLAGS.bit_widths[1]
  pw_conv_bits = FLAGS.bit_widths[2]
  fc_bits = FLAGS.bit_widths[3]
  activations_bits = FLAGS.bit_widths[4]

  print("Regular Conv-weights bit width: " +str(reg_conv_bits))
  print("Depthwise Conv-weights bit width: " + str(dw_conv_bits))
  print("Pointwise Conv-weights bit width: " + str(pw_conv_bits))
  print("FC-weights bit width: " + str(fc_bits))
  print("Activations bit width: " + str(activations_bits))
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(FLAGS.wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, 100)
  clip_stride_ms = 260
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  # input_spectrogram = tf.placeholder(tf.float32, shape=[49,513], name='speech_signal')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)
  # spectrogram = input_spectrogram
  if (FLAGS.input_type == 'log-mel'):
      print("log-mel energies")
      # Warp the linear-scale, magnitude spectrograms into the mel-scale.
      num_spectrogram_bins = spectrogram.shape[-1].value  # magnitude_spectrograms.shape[-1].value
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, model_settings['dct_coefficient_count']
      linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, model_settings['sample_rate'], lower_edge_hertz,
          upper_edge_hertz)
      mel_spectrograms = tf.tensordot(
          spectrogram, linear_to_mel_weight_matrix, 1)
      # Note: Shape inference for `tf.tensordot` does not currently handle this case.
      mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))
      log_offset = 1e-6
      log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
      fingerprint_input = log_mel_spectrograms
  elif (FLAGS.input_type == 'MFCC'):
      print('MFCC-features')
      fingerprint_input = contrib_audio.mfcc(
          spectrogram,
          decoded_sample_data.sample_rate,
          dct_coefficient_count=model_settings['dct_coefficient_count'])
  # fingerprint_input = tf.placeholder(tf.float32,shape=[49,20],name='fingerprint')
  fingerprint_frequency_size = model_settings['dct_coefficient_count']
  fingerprint_time_size = model_settings['spectrogram_length']
  reshaped_input = tf.reshape(fingerprint_input, [
      -1, fingerprint_time_size * fingerprint_frequency_size
  ])

  training = tf.placeholder(tf.bool, name='training')

  logits, net_c1 = models.create_model(
      reshaped_input, model_settings, FLAGS.model_architecture, FLAGS.model_size_info,
      is_training=True, runtime_settings=runtime_settings)
  # Create an output to use for inference.
  tf.nn.softmax(logits, name='labels_softmax')



  saver = tf.train.Saver(tf.global_variables())

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
  for v in tf.trainable_variables():
      print(v.name)
  v_backup = tf.trainable_variables()
  eps = 0.001
  # Layer information [weights, biases, channel means, channel variances, input fractional bits, output fractional bits, name for .h file]
  conv_1 = ['DS-CNN/conv_1/weights', 'DS-CNN/conv_1/biases', 'DS-CNN/conv_1/batch_norm/moving_mean',
            'DS-CNN/conv_1/batch_norm/moving_variance', 2, 5, 'CONV1', 'DS-CNN/conv_1/batch_norm/beta']
  dw_conv_1 = ['DS-CNN/conv_ds_1/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_1/depthwise_conv/biases',
               'DS-CNN/conv_ds_1/dw_batch_norm/moving_mean',
               'DS-CNN/conv_ds_1/dw_batch_norm/moving_variance', 5, 5, 'DW_CONV1',
               'DS-CNN/conv_ds_1/dw_batch_norm/beta']
  pw_conv_1 = ['DS-CNN/conv_ds_1/pointwise_conv/weights', 'DS-CNN/conv_ds_1/pointwise_conv/biases',
               'DS-CNN/conv_ds_1/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_1/pw_batch_norm/moving_variance', 5, 5,
               'PW_CONV1', 'DS-CNN/conv_ds_1/pw_batch_norm/beta']
  dw_conv_2 = ['DS-CNN/conv_ds_2/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_2/depthwise_conv/biases',
               'DS-CNN/conv_ds_2/dw_batch_norm/moving_mean',
               'DS-CNN/conv_ds_2/dw_batch_norm/moving_variance', 5, 5, 'DW_CONV2',
               'DS-CNN/conv_ds_2/dw_batch_norm/beta']
  pw_conv_2 = ['DS-CNN/conv_ds_2/pointwise_conv/weights', 'DS-CNN/conv_ds_2/pointwise_conv/biases',
               'DS-CNN/conv_ds_2/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_2/pw_batch_norm/moving_variance', 5, 5,
               'PW_CONV2', 'DS-CNN/conv_ds_2/pw_batch_norm/beta']
  dw_conv_3 = ['DS-CNN/conv_ds_3/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_3/depthwise_conv/biases',
               'DS-CNN/conv_ds_3/dw_batch_norm/moving_mean',
               'DS-CNN/conv_ds_3/dw_batch_norm/moving_variance', 5, 5, 'DW_CONV3',
               'DS-CNN/conv_ds_3/dw_batch_norm/beta']
  pw_conv_3 = ['DS-CNN/conv_ds_3/pointwise_conv/weights', 'DS-CNN/conv_ds_3/pointwise_conv/biases',
               'DS-CNN/conv_ds_3/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_3/pw_batch_norm/moving_variance', 5, 5,
               'PW_CONV3', 'DS-CNN/conv_ds_3/pw_batch_norm/beta']
  dw_conv_4 = ['DS-CNN/conv_ds_4/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_4/depthwise_conv/biases',
               'DS-CNN/conv_ds_4/dw_batch_norm/moving_mean',
               'DS-CNN/conv_ds_4/dw_batch_norm/moving_variance', 5, 5, 'DW_CONV4',
               'DS-CNN/conv_ds_4/dw_batch_norm/beta']
  pw_conv_4 = ['DS-CNN/conv_ds_4/pointwise_conv/weights', 'DS-CNN/conv_ds_4/pointwise_conv/biases',
               'DS-CNN/conv_ds_4/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_4/pw_batch_norm/moving_variance', 5, 5,
               'PW_CONV4', 'DS-CNN/conv_ds_4/pw_batch_norm/beta']
  dw_conv_5 = ['DS-CNN/conv_ds_5/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_5/depthwise_conv/biases',
               'DS-CNN/conv_ds_5/dw_batch_norm/moving_mean',
               'DS-CNN/conv_ds_5/dw_batch_norm/moving_variance', 5, 5, 'DW_CONV5',
               'DS-CNN/conv_ds_5/dw_batch_norm/beta']
  pw_conv_5 = ['DS-CNN/conv_ds_5/pointwise_conv/weights', 'DS-CNN/conv_ds_5/pointwise_conv/biases',
               'DS-CNN/conv_ds_5/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_5/pw_batch_norm/moving_variance', 5, 5,
               'PW_CONV5', 'DS-CNN/conv_ds_5/pw_batch_norm/beta']
  dw_conv_6 = ['DS-CNN/conv_ds_6/depthwise_conv/depthwise_weights', 'DS-CNN/conv_ds_6/depthwise_conv/biases',
               'DS-CNN/conv_ds_6/dw_batch_norm/moving_mean',
               'DS-CNN/conv_ds_6/dw_batch_norm/moving_variance', 5, 5, 'DW_CONV6',
               'DS-CNN/conv_ds_6/dw_batch_norm/beta']
  pw_conv_6 = ['DS-CNN/conv_ds_6/pointwise_conv/weights', 'DS-CNN/conv_ds_6/pointwise_conv/biases',
               'DS-CNN/conv_ds_6/pw_batch_norm/moving_mean', 'DS-CNN/conv_ds_6/pw_batch_norm/moving_variance', 5, 5,
               'PW_CONV6', 'DS-CNN/conv_ds_6/pw_batch_norm/beta']
  layer_list = [conv_1, dw_conv_1, pw_conv_1, dw_conv_2, pw_conv_2, dw_conv_3, pw_conv_3, dw_conv_4, pw_conv_4,
                dw_conv_5, pw_conv_5, dw_conv_6, pw_conv_6]
  n_filters = 76
  for layer in layer_list:
     bit_width = reg_conv_bits
     layer_name = layer[6]
     PW = False
     if (layer_name[0:2] == 'PW'):
        PW = True
        bit_width = pw_conv_bits
     DW = False
     if (layer_name[0:2] == 'DW'):
        DW = True
        bit_width = dw_conv_bits
     print("Name of node - " + layer[6])
     for v in tf.trainable_variables():
        if v.name == layer[0]+':0':
           v_weights = v
        if v.name == layer[1]+':0':
           v_bias = v
        if v.name == layer[7]+':0':
           v_beta = v
     for v in tf.global_variables():
        if v.name == layer[2]+':0':
           v_mean = v
        if v.name == layer[3]+':0':
           v_var = v
     weights = sess.run(v_weights)
     bias = sess.run(v_bias)
     beta = sess.run(v_beta)
     mean = sess.run(v_mean)
     var = sess.run(v_var)
     #print("Weights shape: " + str(weights.shape))
     #print("Bias shape: " + str(bias.shape))
     #print("Var shape: " + str(var.shape))
     #print("Mean shape: " + str(mean.shape))
     #print("Beta shape: " + str(beta.shape))

     w_shape = weights.shape
     b_shape = bias.shape
     weights = weights.squeeze()
     weights_t1 = np.zeros(weights.shape)
     bias_t1 = np.zeros((1, n_filters))
     for i in range(0, len(bias)):
        if (PW):
           filter = weights[:, i]
        else:
           filter = weights[:, :, i]
        bias_temp = bias[i]
        mean_temp = mean[i]
        var_temp = var[i]
        beta_temp = beta[i]
        new_filter = filter / math.sqrt(var_temp + eps)
        new_bias = beta_temp + (bias_temp - mean_temp) / (math.sqrt(var_temp + eps))
        if (PW):
           weights_t1[:, i] = new_filter
        else:
           weights_t1[:, :, i] = new_filter
        bias_t1[0, i] = new_bias
        #if (i == 0):
            #print('filters : ' + str(filter))
            #print('Bias : ' + str(bias_temp))
            #print('Mean : ' + str(mean_temp))
            #print('Variance : ' + str(var_temp))
            #print("New filter : " + str(new_filter))
            #print("New Bias : " + str(new_bias))
     min_value = weights_t1.min()
     max_value = weights_t1.max()
     int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))

     dec_bits_weight = min((bit_width-1) - int_bits, 111)
     weights_quant = np.round(weights_t1 * 2 ** dec_bits_weight)
     weights_quant = weights_quant/(2**dec_bits_weight)
     weights_quant = weights_quant.reshape(w_shape)
     #print("input fractional bits: " + str(layer[4]))
     #print("Weights min value: " + str(min_value))
     #print("Weights max value: " + str(max_value))
     #print("Weights fractional bits: " + str(dec_bits_weight))
     min_value = bias_t1.min()
     max_value = bias_t1.max()
     int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
     dec_bits_bias = min((bit_width-1) - int_bits, 10000)
     bias_quant = np.round(bias_t1 * 2 ** dec_bits_bias)
     bias_quant = bias_quant/(2**dec_bits_bias)
     bias_quant = bias_quant.reshape(b_shape)
     bias_left_shift = layer[4] + dec_bits_weight - dec_bits_bias
     #print("Bias min value: " + str(min_value))
     #print("Bias max value: " + str(max_value))
     #print("Bias fractional bits: " + str(dec_bits_bias))

     # update the weights in tensorflow graph for quantizing the activations
     updated_weights = sess.run(tf.assign(v_weights, weights_quant))
     updated_bias = sess.run(tf.assign(v_bias, bias_quant))

  fc_layer = ['DS-CNN/fc1/weights', 'DS-CNN/fc1/biases', 5, 3, 'FC']
  for v in tf.trainable_variables():
      if v.name == fc_layer[0]+':0':
          v_fc_weights = v
      if v.name == fc_layer[1]+':0':
          v_fc_bias = v
  weights = sess.run(v_fc_weights)
  bias = sess.run(v_fc_bias)
  w_shape = weights.shape
  b_shape = bias.shape
  #print("FC weights : " + str(weights.shape))
  #print(weights)
  #print("FC bias : " + str(bias.shape))
  #print(bias)
  min_value = weights.min()
  max_value = weights.max()
  int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
  dec_bits_weight = min((fc_bits-1) - int_bits, 111)
  weights_quant = np.round(weights * 2 ** dec_bits_weight)
  weights_quant = weights_quant / (2 ** dec_bits_weight)
  weights_quant = weights_quant.reshape(w_shape)
  #print("input fractional bits: " + str(fc_layer[2]))
  #print("Weights min value: " + str(min_value))
  #print("Weights max value: " + str(max_value))
  #print("Weights fractional bits: " + str(dec_bits_weight))
  min_value = bias.min()
  max_value = bias.max()
  int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
  dec_bits_bias = min((fc_bits-1) - int_bits, 10000)
  bias_quant = np.round(bias * 2 ** dec_bits_bias)
  #print("Bias min value: " + str(min_value))
  #print("Bias max value: " + str(max_value))
  #print("Bias fractional bits: " + str(dec_bits_bias))
  bias_quant = bias_quant / (2 ** dec_bits_bias)
  bias_quant = bias_quant.reshape(b_shape)
  #print("Quantized weights: " + str(weights_quant))
  #print("Quantized bias: " +str(bias_quant))
  updated_weights = sess.run(tf.assign(v_fc_weights, weights_quant))
  updated_bias = sess.run(tf.assign(v_fc_bias, bias_quant))
  #print("bias[0] : " + str(bias[0]))
  #print("bias_quant[0] : " + str(bias_quant[0]))


  training_step = 30000
  checkpoint_path = os.path.join(FLAGS.train_dir, 'quant',
                                 FLAGS.model_architecture + '.ckpt')
  tf.logging.info('Saving best model to "%s-%d"', checkpoint_path, training_step)
  saver.save(sess, checkpoint_path, global_step=training_step)




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition. Useful for models with cache.',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--output_file', type=str, help='Where to save the frozen graph.')
  parser.add_argument(
      '--input_type',
      type=str,
      default='MFCC',
      help='MFCC if DCT should be applied, log_mel if not')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128, 128, 128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--bit_widths',
      type=int,
      nargs="+",
      default=[8, 8, 8, 8, 8],
      help='Bit width for regular Conv-weights, Depthwise-conv weights, Pointwise-conv weights, FC-weights and activations')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
