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

/*
 * Description: Keyword spotting example code using MFSC feature extraction
 * and DS-CNN model. 
 */

#include "kws_ds_cnn.h"
#include "string.h"

KWS_DS_CNN::KWS_DS_CNN(int16_t* audio_buffer, q7_t* scratch_buffer)
: audio_buffer(audio_buffer)
{
  log_mel = new LOG_MEL;
  nn = new DS_CNN(scratch_buffer);
}

KWS_DS_CNN::~KWS_DS_CNN()
{
  delete log_mel;
  delete nn;
}

void KWS_DS_CNN::extract_features()
{
  int32_t log_mel_buffer_head=0;
  for (uint16_t f = 0; f < NUM_FRAMES; f++) {
    log_mel->log_mel_compute(audio_buffer+(f*FRAME_SHIFT),3,&log_mel_buffer[log_mel_buffer_head]);
    log_mel_buffer_head += NUM_FBANK_BINS;
  }
  
}

/* This overloaded function is used in streaming audio case */
void KWS_DS_CNN::extract_features(uint16_t num_frames) 
{

  //move old features left 
  memmove(log_mel_buffer,log_mel_buffer+(num_frames*NUM_FBANK_BINS),(NUM_FRAMES-num_frames)*NUM_FBANK_BINS);
  //compute features only for the newly recorded audio
  int32_t log_mel_buffer_head = (NUM_FRAMES-num_frames)*NUM_FBANK_BINS; 

  for (uint16_t f = 0; f < num_frames; f++) {
    log_mel->log_mel_compute(audio_buffer+(f*FRAME_SHIFT),2,&log_mel_buffer[log_mel_buffer_head]);

    log_mel_buffer_head += NUM_FBANK_BINS;
  }
}

void KWS_DS_CNN::classify()
{
  nn->run_nn(log_mel_buffer, output);

  // Softmax
  arm_softmax_q7(output,OUT_DIM,output);
  	
  //do any post processing here
}

int KWS_DS_CNN::get_top_detection(q7_t* prediction)
{
  int max_ind=0;
  int max_val=-128;
  for(int i=0;i<OUT_DIM;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }    
  }
  return max_ind;
}

void KWS_DS_CNN::average_predictions(int window_len)
{
  //shift right old predictions 
  for(int i=window_len-1;i>0;i--) {
    for(int j=0;j<OUT_DIM;j++)
      predictions[i][j]=predictions[i-1][j];
  }
  //add new predictions
  for(int j=0;j<OUT_DIM;j++)
    predictions[0][j]=output[j];
  //compute averages
  int sum;
  for(int j=0;j<OUT_DIM;j++) {
    sum=0;
    for(int i=0;i<window_len;i++) 
      sum += predictions[i][j];
    averaged_output[j] = (q7_t)(sum/window_len);
  }   
}
