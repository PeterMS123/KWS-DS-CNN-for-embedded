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
#
# Modifications Copyright  2018 Peter Mølgaard Sørensen
# Adapted from train.py to run single inference test on quantized network
# Bit width of variables are specified through command line argument
r"""
Loads network from checkpoint file, and quantizes weights. Weight nodes names must be specified in the script.
Fuses batch-norm variables into preceding conv-weights and runs single-inference test using quantized activations
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
import math
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
  print(FLAGS.data_dir)
  print(FLAGS.model_size_info)
  reg_conv_bits = FLAGS.bit_widths[0]
  dw_conv_bits = FLAGS.bit_widths[1]
  pw_conv_bits = FLAGS.bit_widths[2]
  fc_bits = FLAGS.bit_widths[3]
  activations_bits = FLAGS.bit_widths[4]
  print('Zero out threshold: '+str(FLAGS.zero_out_percentage))
  print("Regular Conv-weights bit width: " +str(reg_conv_bits))
  print("Depthwise Conv-weights bit width: " + str(dw_conv_bits))
  print("Pointwise Conv-weights bit width: " + str(pw_conv_bits))
  print("FC-weights bit width: " + str(fc_bits))
  print("Activations bit width: " + str(activations_bits))
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()
  mag_squared_bool = FLAGS.spectrogram_magnitude_squared == 'True'
  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.dct_coefficient_count, activations_bits)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings,FLAGS.input_type, mag_squared_bool)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      FLAGS.model_size_info,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(cross_entropy_mean)
  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)
  for v in tf.trainable_variables():
      print(v.name)
  v_backup = tf.trainable_variables()
  tf.logging.info('Training from step: %d ', start_step)
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
     layer_name = layer[6]
     bit_width = reg_conv_bits
     PW = False
     if (layer_name[0:2] == 'PW'):
        PW = True
        bit_width = pw_conv_bits
     DW = False
     if (layer_name[0:2] == 'DW'):
        DW = True
        bit_width = dw_conv_bits
     #print("Name of node - " + layer[6])
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
     if(bit_width < 9):
     	int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
     	dec_bits_weight = min((bit_width-1) - int_bits, 111)
     	weights_quant = np.round(weights_t1 * 2 ** dec_bits_weight)
     	weights_quant = weights_quant/(2**dec_bits_weight)
     	weights_quant = weights_quant.reshape(w_shape)
     elif(bit_width > 8):
        weights_quant = weights_t1.reshape(w_shape)
     #print("input fractional bits: " + str(layer[4]))
     #print("Weights min value: " + str(min_value))
     #print("Weights max value: " + str(max_value))
     #print("Weights fractional bits: " + str(dec_bits_weight))
     min_value = bias_t1.min()
     max_value = bias_t1.max()
     if(bit_width < 9):	
        int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        dec_bits_bias = min((bit_width-1) - int_bits, 10000)
        bias_quant = np.round(bias_t1 * 2 ** dec_bits_bias)
        bias_quant = bias_quant/(2**dec_bits_bias)
        bias_quant = bias_quant.reshape(b_shape)
        bias_left_shift = layer[4] + dec_bits_weight - dec_bits_bias
     elif(bit_width > 8):
        bias_quant = bias_t1.reshape(b_shape)
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
  if(fc_bits < 9):
     int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
     dec_bits_weight = min((fc_bits-1) - int_bits, 111)
     weights_quant = np.round(weights * 2 ** dec_bits_weight)
     weights_quant = weights_quant / (2 ** dec_bits_weight)
     weights_quant = weights_quant.reshape(w_shape)
  elif(fc_bits > 8):
     weights_quant = weights.reshape(w_shape)
  #print("input fractional bits: " + str(fc_layer[2]))
  #print("Weights min value: " + str(min_value))
  #print("Weights max value: " + str(max_value))
  #print("Weights fractional bits: " + str(dec_bits_weight))
  min_value = bias.min()
  max_value = bias.max()
  if(fc_bits < 9):
     int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
     dec_bits_bias = min((fc_bits-1) - int_bits, 10000)
     bias_quant = np.round(bias * 2 ** dec_bits_bias)
  #print("Bias min value: " + str(min_value))
  #print("Bias max value: " + str(max_value))
  #print("Bias fractional bits: " + str(dec_bits_bias))
     bias_quant = bias_quant / (2 ** dec_bits_bias)
     bias_quant = bias_quant.reshape(b_shape)
  elif(fc_bits > 8):
     bias_quant = bias.reshape(b_shape)
  #print("Quantized weights: " + str(weights_quant))
  #print("Quantized bias: " +str(bias_quant))
  updated_weights = sess.run(tf.assign(v_fc_weights, weights_quant))
  updated_bias = sess.run(tf.assign(v_fc_bias, bias_quant))
  #print("bias[0] : " + str(bias[0]))
  #print("bias_quant[0] : " + str(bias_quant[0]))


  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings,  FLAGS.background_frequency,
        FLAGS.background_volume, 0, 'testing', sess)

    # Quantize input features
    if (activations_bits <= 8):
        test_fingerprints = tf.fake_quant_with_min_max_args(tf.convert_to_tensor(test_fingerprints,dtype=tf.float32), min = -16, max = 16, num_bits = activations_bits)
        test_fingerprints = test_fingerprints.eval()
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))

  text_file = open(FLAGS.result_path+'/accuracy_'+FLAGS.noise_set+"_quant_"+str(reg_conv_bits)+"_"+ str(dw_conv_bits)+ "_" +str(pw_conv_bits)+"_"+str(fc_bits)+"_" +str(activations_bits)+".txt", "a")
  text_file.write(str(total_accuracy) + ',')
  text_file.close()

  text_file = open(FLAGS.result_path + '/conf_' + FLAGS.noise_set + "_"+FLAGS.SNR+"_quant_"+str(reg_conv_bits)+"_"+ str(dw_conv_bits)+ "_" +str(pw_conv_bits)+"_"+str(fc_bits)+"_" +str(activations_bits)+".txt", "w")
  text_file.write(str(total_conf_matrix))
  text_file.close()

  #print("Saving original weights")
  for v in v_backup:
    var_val = sess.run(v)
    #var_val = sess.run(tf.assign(v, var_val))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
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
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
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
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--SNR',
      type=str,
      default='',
      help='SNR of test data')
  parser.add_argument(
      '--noise_set',
      type=str,
      default='',
      help='If there are unknown noise tyoes included')
  parser.add_argument(
      '--result_path',
      type=str,
      default='',
      help='path to where results should be put')
  parser.add_argument(
      '--input_type',
      type=str,
      default='MFCC',
      help='MFCC if DCT should be applied, log_mel if not')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  parser.add_argument(
      '--spectrogram_magnitude_squared',
      type=str,
      default='True',
      help='Whether to use squared magnitude spectrogram or not')
  parser.add_argument(
      '--bit_widths',
      type=int,
      nargs="+",
      default=[8, 8, 8, 8, 8],
      help='Bit width for regular Conv-weights, Depthwise-conv weights, Pointwise-conv weights, FC-weights and activations')
  parser.add_argument(
      '--zero_out_percentage',
      type=int,
      default=0,
      help='Threshold for removing weights')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
