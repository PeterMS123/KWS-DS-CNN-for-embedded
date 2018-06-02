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
# Adapted from label_wav.py to run detection test in continuous audio stream
#
r"""
Loads a model from a frozen graph and runs automated streaming tests on specified wav-file for different values of detection threshold,
averaging window and input frame shift.
Wav-file must have a corresponding groundtruth document. Both can be generated using generate_streaming_test_wav.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None
Fs = 16000

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

  return 0

def check_keyword(keyword, groundtruth, tolerance, check_list):
    for index, x in enumerate(groundtruth):
        if (index%2 ==1):
            if(x > keyword[1] - tolerance and x < keyword[1] + tolerance):
                if (groundtruth[index-1] == keyword[0] and check_list[round((index-1)/2)] == 0):
                    return ['hit', round((index-1)/2)]
    return ['false alarm', -1]

def check_ground_truth(groundtruth, check_list, current_time_ms):
    for index, x in enumerate(groundtruth):
        if (index%2 ==1):
            if(check_list[round((index-1)/2)]==0 and current_time_ms > x + 1000):
                if (groundtruth[index-1]== '_unknown_'):
                    return ['unknown', round((index-1)/2)]
                else:
                    return ['miss', round((index-1)/2)]

    return ['ok',-1]

def label_wav(wav, labels, graph, input_name, output_name, how_many_labels, window_stride, frame_length,w_smooth_ms
              ,ground_truth,detection_threshold, suppression_ms, detection_tolerance_ms,detection_threshold_conf, use_conf
              , use_method_2, direct_method):
  float_formatter = lambda x: "%.2f" % x
  np.set_printoptions(formatter={'float_kind': float_formatter})
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not ground_truth or not tf.gfile.Exists(ground_truth):
    tf.logging.fatal('Ground truth file does not exist %s', ground_truth)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)
  print(labels_list)
  ground_truth_list = load_labels(ground_truth)
  ground_truth_list = [x for xs in ground_truth_list for x in xs.split(',')]
  n_words = round(sum(1 for item in ground_truth_list)/2)
  checked_keywords = np.zeros([n_words])

  n_keywords = 0
  for index, x in enumerate(ground_truth_list): # Convert timestamps to floats
    if(index%2 == 1):
      ground_truth_list[index] = float(ground_truth_list[index])
    else:
      if(ground_truth_list[index] != '_unknown_'):
        n_keywords += 1

  w_smooth = max(round(w_smooth_ms/window_stride),1)
  # load graph, which is stored in the default session
  load_graph(graph)
  print("Network loaded \n")
  print("Window length is " + str(frame_length) + " ms")
  print("Window stride in streaming test:  "+str(window_stride) + " ms")
  print("Predictions are averaged over " + str(w_smooth) + " frames, approximately "+ str(w_smooth_ms)+" ms")
  print("Detection threshold is : " + str(detection_threshold))
  print("Detection threshold (confidence score) is : " + str(detection_threshold_conf))
  print("Detected keywords are suppressed for "+ str(suppression_ms) + " ms")

  with open("/tmp/speech_dataset/go/0ab3b47d_nohash_1.wav", 'rb') as wav_file:
    wav_dummy = wav_file.read()
  head = wav_dummy[:44]
  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  signal = wav_data[44:]
  file_length_bytes = sum(1 for item in signal)
  print("Length of read audio (bytes): " + str(file_length_bytes))
  n_samples = file_length_bytes/2;
  print("Number of samples (16 bit pr sample): "+ str(n_samples))
  print("File length in seconds maybe : " + str(n_samples/Fs))

  hits = 0
  false_alarms = 0
  misses = 0
  suppression_frames = int(suppression_ms/window_stride)
  n_frames = int((n_samples/Fs -1)/(window_stride/1000))
  #n_frames = 50
  predict_array = np.zeros([12,n_frames])
  predict_smooth = np.zeros([12,n_frames])
  predict_one_hot = np.zeros([12, n_frames])
  suppressed = np.zeros([12,1])
  keywords_found = []
  time_offset_ms = 0
  time_n = 0

  with tf.Session() as sess:
    for i in range(0,n_frames): #  Loop through audio file
      if i%int((n_frames/20)) == 0:
        print("Progress : "+ str(int(100*i/n_frames)) + "%")
      bytes_offset = round((time_offset_ms/1000)*Fs*2)
      sub_sig = signal[bytes_offset:bytes_offset + round((frame_length/1000)*Fs*2)]
      proc_sig = head + sub_sig

      """Runs the audio data through the graph and prints predictions."""

      # Feed the audio data as input to the graph.
      #   predictions  will contain a two-dimensional array, where one
      #   dimension represents the input image count, and the other has
            #   predictions per class
      softmax_tensor = sess.graph.get_tensor_by_name(output_name)
      pred, = sess.run(softmax_tensor, {input_name: proc_sig})
      predict_array[:,i] = np.array(pred)
      

      #Smooth predictions
      h_smooth = max(0, i - w_smooth + 1)
      predict_smooth[:,i] = np.mean(predict_array[:,h_smooth:i+1], axis = 1)

      # Detection using the confidence score
      if(use_conf):
          w_max = max(1,round(w_smooth_ms / window_stride))
          h_max = max(0, i - w_max + 1)
          # Calculate confidence score
          max_smoothed_predicts = np.max(predict_smooth[1:12, h_max:i + 1]*100, axis=0)
          confidence_score = (np.prod(max_smoothed_predicts)) ** (1 / w_max)
          #print(confidence_score)
          #detection_threshold_conf = 50
          if(confidence_score > detection_threshold_conf):
            index = np.argmax(np.max(predict_smooth[1:12, h_max:i + 1]*100, axis=1))+1
            if(suppressed[index]==0 and labels_list[index]!= '_silence_'):
                suppressed[index] = suppression_frames  # Suppress the keyword for a number of frames
                keywords_found.append([labels_list[index], time_offset_ms])  # Add found keyword to the list
                detection_status, word_index = check_keyword(keywords_found[-1], ground_truth_list,
                                                             detection_tolerance_ms, checked_keywords)
                #print('Index:' + str(index))
                if (detection_status == 'hit'):
                    if (labels_list[index] != '_unknown_'):
                        hits += 1

                    checked_keywords[word_index] = 1
                elif (detection_status == 'false alarm'):
                    if (labels_list[index] != '_unknown_'):
                        false_alarms += 1
                print("Detected keyword! :'" + labels_list[index] + "' at time " + str(time_offset_ms) + " ms"
                      + " - It is a " + detection_status + "!")
      elif(use_method_2):
          #print(predict_smooth[:,i-1:i+1])
          p_conf = predict_smooth[:,i]*predict_smooth[:,i-1]
          #print('haps')
          for index, p in enumerate(p_conf):
              if p > detection_threshold and suppressed[index] == 0 and labels_list[index] != '_silence_':
                  suppressed[index] = suppression_frames  # Suppress the keyword for a number of frames
                  keywords_found.append([labels_list[index], time_offset_ms])  # Add found keyword to the list
                  detection_status, word_index = check_keyword(keywords_found[-1], ground_truth_list,
                                                               detection_tolerance_ms, checked_keywords)
                  # print('Index:'+str(index))
                  if (detection_status == 'hit'):
                      if (labels_list[index] != '_unknown_'):
                          hits += 1

                      checked_keywords[word_index] = 1
                  elif (detection_status == 'false alarm'):
                      false_alarms += 1
                  print("Detected keyword! :'" + labels_list[index] + "' at time " + str(time_offset_ms) + " ms"
                        + " - It is a " + detection_status + "!")
      elif (direct_method):
          max_index = np.argmax(predict_array[:,i])
          predict_one_hot[max_index,i] = 1

          for index in range(0,12):

              if (predict_one_hot[index,i]+ predict_one_hot[index,i-1]) > 1.5  and suppressed[index] == 0 and labels_list[index] != '_silence_':
                  suppressed[index] = suppression_frames  # Suppress the keyword for a number of frames
                  keywords_found.append([labels_list[index], time_offset_ms])  # Add found keyword to the list
                  detection_status, word_index = check_keyword(keywords_found[-1], ground_truth_list,
                                                               detection_tolerance_ms, checked_keywords)
                  # print('Index:'+str(index))
                  if (detection_status == 'hit'):
                      if (labels_list[index] != '_unknown_'):
                          hits += 1

                      checked_keywords[word_index] = 1
                  elif (detection_status == 'false alarm'):
                      false_alarms += 1
                  print("Detected keyword! :'" + labels_list[index] + "' at time " + str(time_offset_ms) + " ms"
                        + " - It is a " + detection_status + "!")
      else:
          # Check if a smoothed prediction is larger than the detection threshold and is not suppressed
          # Keyword detected
          for index ,p in enumerate(predict_smooth[:,i]):
            if p > detection_threshold and suppressed[index] == 0 and labels_list[index]!= '_silence_':
              suppressed[index] = suppression_frames # Suppress the keyword for a number of frames
              keywords_found.append([labels_list[index], time_offset_ms]) # Add found keyword to the list
              detection_status, word_index = check_keyword(keywords_found[-1], ground_truth_list, detection_tolerance_ms, checked_keywords)
              #print('Index:'+str(index))
              if(detection_status == 'hit'):
                if(labels_list[index] != '_unknown_'):
                  hits += 1

                checked_keywords[word_index] = 1
              elif (detection_status == 'false alarm'):
                false_alarms += 1
              print("Detected keyword! :'" + labels_list[index] + "' at time " + str(time_offset_ms) + " ms"
                            + " - It is a "+ detection_status+ "!")


      # Update suppression list
      for index,x in enumerate(suppressed):
        if x != 0:
          suppressed[index] = suppressed[index] -1

      #Check once every second if a word from the ground truth has been missed
      if(time_offset_ms > 1000*time_n):
        time_n += 1
        status, word_index = check_ground_truth(ground_truth_list, checked_keywords, time_offset_ms)
        if(status != 'ok'):
          checked_keywords[word_index] =1
          if(status == 'miss'):
            misses += 1
            print("System failed to detect the keyword -"+ground_truth_list[word_index*2]+ "- at "+str(ground_truth_list[word_index*2+1])+ " ms")
            #print(predict_smooth[:,round(time_offset_ms/window_stride)-40:round(time_offset_ms/window_stride)-10])
          elif(status == 'unknown'):
            print("System failed to detect the keyword -" + ground_truth_list[word_index * 2] + "- at " + str(
              ground_truth_list[word_index * 2 + 1]) + " ms. But that is okay!")

      time_offset_ms += window_stride


    #print("Keywords detected by system:")
    #print(keywords_found)
    print("#words in test: " + str(n_words))
    print("#Keywords in test: " + str(n_keywords))
    print("#hits: "+ str(hits))
    print("#false alarms: " + str(false_alarms))
    print("#misses: " + str(misses))
    print("Hit rate = " + str(hits/n_keywords))
    print("Miss rate = "+ str(misses/n_keywords))

  return [n_frames, n_keywords, hits,false_alarms,misses]

def main(_):
  names = np.array(['Default detection threshold ', '\n Default averaging window (ms) ', '\n  Default window stride (ms)', '\n  Default detection threshold - conf'])
  vals = np.array([FLAGS.detection_threshold,FLAGS.w_smooth_ms, FLAGS.window_stride, FLAGS.detection_threshold_conf])
  dat = np.column_stack((names, vals));
  start = FLAGS.wav.find('g_test_')
  print(start)
  noise_config = FLAGS.wav[start + 7:-4]
  print(noise_config)
  np.savetxt(FLAGS.result_path +'/streaming_test_settings_'+noise_config+'.txt', dat , delimiter=" ", fmt="%s")
  use_conf = False
  method_2 = False
  direct_method = False
  # test 1
  active_tests = [0,1,0,0,1,1]
  test_crtieria = ['n_words', 'hits', 'false alarms', 'misses', 'hitrate', 'false alarm rate', 'missrate']
  if (active_tests[0]):
      use_conf = True
      print("Running tests for detection threshold _conf score")
      detection_thresholds_conf=[40,50,60,70,80,90,95]
      n_vals = 7
      test_output = np.zeros([n_vals, 9])
      for i in range(0, n_vals):
          FLAGS.detection_threshold_conf = detection_thresholds_conf[i]
          n_population, n_keywords, hits, false_alarms, misses = label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
                                                             FLAGS.output_name, FLAGS.how_many_labels,
                                                             FLAGS.window_stride, FLAGS.frame_length, FLAGS.w_smooth_ms,
                                                             FLAGS.ground_truth, FLAGS.detection_threshold,
                                                             FLAGS.suppression_ms, FLAGS.detection_tolerance_ms,
                                                             FLAGS.detection_threshold_conf, use_conf, method_2,direct_method)
          test_output[i, :] = [FLAGS.detection_threshold_conf,n_population,  n_keywords, hits, false_alarms, misses, hits / n_keywords,
                               false_alarms / (n_population-n_keywords), misses / n_keywords]
      print("detection threshold using conf score results")
      print(test_crtieria)
      print(test_output)
      np.savetxt(FLAGS.result_path + '/test_detection_threshold_conf_'+noise_config+'.csv', test_output, fmt='%.3f', delimiter=',',
                 header="Detection threshold, n_keywords, n_population, _hits, false_alarms, misses, True-Posistive rate, False-Positive Rate, missrate")

  FLAGS.detection_threshold_conf = 80
  use_conf = False
  if(active_tests[1]):

      print("Running tests for detection threshold")
      detection_thresholds =[0.6,0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.96,0.98]
      n_vals = 10
      test_output = np.zeros([n_vals,9])
      for i in range(0,n_vals):
        FLAGS.detection_threshold = detection_thresholds[i]
        n_population, n_keywords, hits, false_alarms, misses =label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
                FLAGS.output_name, FLAGS.how_many_labels, FLAGS.window_stride, FLAGS.frame_length, FLAGS.w_smooth_ms,
                FLAGS.ground_truth, FLAGS.detection_threshold, FLAGS.suppression_ms, FLAGS.detection_tolerance_ms,
                FLAGS.detection_threshold_conf,use_conf, method_2,direct_method)
        test_output[i,:]=[FLAGS.detection_threshold,n_population, n_keywords,hits, false_alarms, misses, hits/n_keywords, false_alarms/(n_population-n_keywords), misses/n_keywords]
      print("detection threshold results")
      print(test_crtieria)
      print(test_output)
      if(FLAGS.quant_graph =='yep' ):
          quant_status = 'quant'
      else:
          quant_status=''

      np.savetxt(FLAGS.result_path + '/test_detection_threshold_'+noise_config+'_'+quant_status+'.csv', test_output, fmt='%.3f', delimiter=',',
                 header="Detection threshold, n_keywords, n_population, _hits, false_alarms, misses, True-Posistive rate, False-Positive Rate, missrate")

  FLAGS.detection_threshold = 0.8

  if (active_tests[2]):
      method_2 = True
      print("Running tests for detection threshold - method 2")
      detection_thresholds = [ 0.4, 0.5, 0.6,0.65, 0.7,0.75,0.8,0.85,0.9]
      n_vals = 9
      test_output = np.zeros([n_vals, 9])
      for i in range(0, n_vals):
          FLAGS.detection_threshold = detection_thresholds[i]
          n_population, n_keywords, hits, false_alarms, misses = label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
                                                             FLAGS.output_name, FLAGS.how_many_labels,
                                                             FLAGS.window_stride, FLAGS.frame_length, FLAGS.w_smooth_ms,
                                                             FLAGS.ground_truth, FLAGS.detection_threshold,
                                                             FLAGS.suppression_ms, FLAGS.detection_tolerance_ms,
                                                             FLAGS.detection_threshold_conf, use_conf, method_2,direct_method)
          test_output[i, :] = [FLAGS.detection_threshold,n_population,  n_keywords, hits, false_alarms, misses, hits / n_keywords,
                               false_alarms / (n_population-n_keywords), misses / n_keywords]
      print("detection threshold results -method 2")
      print(test_crtieria)
      print(test_output)
      np.savetxt(FLAGS.result_path + '/test_detection_threshold_' + noise_config + '_method_2.csv', test_output, fmt='%.3f',
                 delimiter=',',
                 header="Detection threshold, n_keywords, n_population, _hits, false_alarms, misses, True-Posistive rate, False-Positive Rate, missrate")

  FLAGS.detection_threshold = 0.8
  method_2 = False

  if (active_tests[3]):
      direct_method = True
      print("Running test for direct method")
      detection_thresholds = [1]
      n_vals = 1
      test_output = np.zeros([n_vals, 9])
      for i in range(0, n_vals):
          FLAGS.detection_threshold = detection_thresholds[i]
          n_population, n_keywords, hits, false_alarms, misses = label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
                                                             FLAGS.output_name, FLAGS.how_many_labels,
                                                             FLAGS.window_stride, FLAGS.frame_length, FLAGS.w_smooth_ms,
                                                             FLAGS.ground_truth, FLAGS.detection_threshold,
                                                             FLAGS.suppression_ms, FLAGS.detection_tolerance_ms,
                                                             FLAGS.detection_threshold_conf, use_conf, method_2,direct_method)
          test_output[i, :] = [FLAGS.detection_threshold,n_population,  n_keywords, hits, false_alarms, misses, hits / n_keywords,
                               false_alarms / (n_population-n_keywords), misses / n_keywords]
      print("detection threshold results -method 2")
      print(test_crtieria)
      print(test_output)
      np.savetxt(FLAGS.result_path + '/test_detection_threshold_' + noise_config + '_direct_method.csv', test_output, fmt='%.3f',
                 delimiter=',',
                 header=" Detection threshold, n_keywords, n_population, _hits, false_alarms, misses, True-Posistive rate, False-Positive Rate, missrate")

  FLAGS.detection_threshold = 0.8
  direct_method = False

  # Smoothing method
  if(active_tests[4]):
      print("Running tests for w_smooth")
      w_smooth = [1, 2, 3, 4]
      n_vals = 4
      test_output = np.zeros([n_vals, 9])
      for i in range(0, n_vals):
          FLAGS.w_smooth_ms = w_smooth[i]*FLAGS.window_stride
          n_population, n_keywords, hits, false_alarms, misses = label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
                                                             FLAGS.output_name, FLAGS.how_many_labels,
                                                             FLAGS.window_stride, FLAGS.frame_length, FLAGS.w_smooth_ms,
                                                             FLAGS.ground_truth, FLAGS.detection_threshold,
                                                             FLAGS.suppression_ms, FLAGS.detection_tolerance_ms
                                                             ,FLAGS.detection_threshold_conf,use_conf, method_2,direct_method)
          test_output[i, :] = [max(round(FLAGS.w_smooth_ms/FLAGS.window_stride),1)*FLAGS.window_stride, n_population, n_keywords, hits, false_alarms,
                               misses, hits / n_keywords, false_alarms / (n_population-n_keywords),
                               misses / n_keywords]
      print("w_smooth results")
      print(test_crtieria)
      print(test_output)
      np.savetxt(FLAGS.result_path+'/test_w_smooth_'+noise_config+'.csv', test_output, fmt='%.3f', delimiter=',',
                 header=" W-smooth, n_keywords, n_population, _hits, false_alarms, misses, True-Posistive rate, False-Positive Rate, missrate")


  FLAGS.w_smooth_ms = 500
  if (active_tests[5]):
      print("Running tests for window_stride")
      window_stride = [20, 50, 100, 250,333 , 500, 750, 1000]
      n_vals = 8
      test_output = np.zeros([n_vals, 9])
      for i in range(0, n_vals):
          FLAGS.window_stride = window_stride[i]
          n_population, n_keywords, hits, false_alarms, misses = label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
                                                             FLAGS.output_name, FLAGS.how_many_labels,
                                                             FLAGS.window_stride, FLAGS.frame_length, FLAGS.w_smooth_ms,
                                                             FLAGS.ground_truth, FLAGS.detection_threshold,
                                                             FLAGS.suppression_ms, FLAGS.detection_tolerance_ms
                                                             ,FLAGS.detection_threshold_conf,use_conf,method_2,direct_method)
          test_output[i, :] = [FLAGS.window_stride,n_population,  n_keywords, hits, false_alarms, misses, hits / n_keywords, false_alarms / (n_population-n_keywords),
                               misses / n_keywords]
      print("window_stride results")
      print(test_crtieria)
      print(test_output)
      np.savetxt(FLAGS.result_path+'/test_window_stride_'+noise_config+'.csv', test_output, fmt='%.3f', delimiter=',',
                 header="  W-stride, n_keywords, n_population, _hits, false_alarms, misses, True-Posistive rate, False-Positive Rate, missrate")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--ground_truth', type=str, default='', help='Path to file containing labels and timesteps of test wav')
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
  parser.add_argument(
      '--window_stride',
      type=int,
      default=250,
      help='time between the network is applied (ms)')
  parser.add_argument(
      '--frame_length',
      type=int,
      default=1000,
      help='Length of each input frame (ms)')
  parser.add_argument(
      '--w_smooth_ms',
      type=int,
      default=500,
      help='time to average predictions over')
  parser.add_argument(
      '--detection_threshold',
      type=float,
      default=0.8,
      help='Detection threshold of smoothed predictions')
  parser.add_argument(
      '--detection_threshold_conf',
      type=float,
      default=70,
      help='Detection threshold of smoothed predictions using confidence score')
  parser.add_argument(
      '--suppression_ms',
      type=int,
      default=1750,
      help='Amount of time to ignore the same predicted word')
  parser.add_argument(
      '--detection_tolerance_ms',
      type=int,
      default=750,
      help='Maximum detection time deviation from ground truth')
  parser.add_argument(
      '--quant_graph',
      type=str,
      default='nope',
      help='Wheter the network is quantized or not')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
