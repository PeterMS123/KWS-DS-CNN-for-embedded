/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************
 *
 * Modifications Copyright 2018 Peter Mølgaard Sørensen
 * KWS system uses MFSC (log-mel) features instead of MFCC. DNN classifier replaced with depthwise separable CNN 
 */
 */

#ifndef __KWS_H__
#define __KWS_H__

#include "arm_math.h"
#include "mbed.h"
#include "ds_cnn.h"
#include "log_mel.h"

#define MAX_SLIDING_WINDOW 10

class KWS_DS_CNN{

public:
  KWS_DS_CNN(int16_t* audio_buffer, q7_t* scratch_buffer);
  ~KWS_DS_CNN();
  
  void extract_features();
  //overloaded function for 
  void extract_features(uint16_t num_frames);
  void classify();
  void average_predictions(int window_len);
  int get_top_detection(q7_t* prediction);
  int16_t* audio_buffer;
  q7_t log_mel_buffer[LOG_MEL_BUFFER_SIZE];
  q7_t output[OUT_DIM];
  q7_t predictions[MAX_SLIDING_WINDOW][OUT_DIM];
  q7_t averaged_output[OUT_DIM];
  
private:
  LOG_MEL *log_mel;
  DS_CNN *nn;
  
};

#endif
