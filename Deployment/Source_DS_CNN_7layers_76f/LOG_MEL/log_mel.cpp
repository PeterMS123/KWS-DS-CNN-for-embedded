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
 * Removed DCT to generate MFSC features instead of MFCC. Power spectrogram is used instead of magnitude
 */


/*
 * Description: MFSC feature extraction
 */

#include <string.h>

#include "log_mel.h"
#include "float.h"

LOG_MEL::LOG_MEL() {

  // Round-up to nearest power of 2.
  frame_len_padded = pow(2,ceil((log(FRAME_LEN)/log(2))));

  frame = new float[frame_len_padded];
  buffer = new float[frame_len_padded];
  mel_energies = new float[NUM_FBANK_BINS];

  //create window function
  window_func = new float[FRAME_LEN];
  for (int i = 0; i < FRAME_LEN; i++)
    window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (FRAME_LEN));

  //create mel filterbank
  fbank_filter_first = new int32_t[NUM_FBANK_BINS];
  fbank_filter_last = new int32_t[NUM_FBANK_BINS];;
  mel_fbank = create_mel_fbank();
  

  //initialize FFT
  rfft = new arm_rfft_fast_instance_f32;
  arm_rfft_fast_init_f32(rfft, frame_len_padded);

}

LOG_MEL::~LOG_MEL() {
  delete []frame;
  delete [] buffer;
  delete []mel_energies;
  delete []window_func;
  delete []fbank_filter_first;
  delete []fbank_filter_last;
  delete rfft;
  for(int i=0;i<NUM_FBANK_BINS;i++)
    delete []mel_fbank;
}


float ** LOG_MEL::create_mel_fbank() {

  int32_t bin, i;

  int32_t num_fft_bins = frame_len_padded/2;
  float fft_bin_width = ((float)SAMP_FREQ) / frame_len_padded;
  float mel_low_freq = MelScale(MEL_LOW_FREQ);
  float mel_high_freq = MelScale(MEL_HIGH_FREQ); 
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

  float *this_bin = new float[num_fft_bins];

  float ** mel_fbank =  new float*[NUM_FBANK_BINS];

  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {

    float left_mel = mel_low_freq + bin * mel_freq_delta;
    float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
    float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    int32_t first_index = -1, last_index = -1;

    for (i = 0; i < num_fft_bins; i++) {

      float freq = (fft_bin_width * i);  // center freq of this fft bin.
      float mel = MelScale(freq);
      this_bin[i] = 0.0;

      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel-mel) / (right_mel-center_mel);
        }
        this_bin[i] = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }

    fbank_filter_first[bin] = first_index;
    fbank_filter_last[bin] = last_index;
    mel_fbank[bin] = new float[last_index-first_index+1]; 

    int32_t j = 0;
    //copy the part we care about
    for (i = first_index; i <= last_index; i++) {
      mel_fbank[bin][j++] = this_bin[i];
    }
  }
  delete []this_bin;
  return mel_fbank;
}

void LOG_MEL::log_mel_compute(const int16_t * data, uint16_t dec_bits, int8_t * log_mel_out) {

  int32_t i, j, bin;

  //TensorFlow way of normalizing .wav data to (-1,1)
  for (i = 0; i < FRAME_LEN; i++) {
    frame[i] = (float)data[i]/(1<<15); 
  }
  //Fill up remaining with zeros
  memset(&frame[FRAME_LEN], 0, sizeof(float) * (frame_len_padded-FRAME_LEN));

  for (i = 0; i < FRAME_LEN; i++) {
    frame[i] *= window_func[i];
  }

  //Compute FFT
  arm_rfft_fast_f32(rfft, frame, buffer, 0);

  //Convert to power spectrum
  //frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32_t half_dim = frame_len_padded/2;
  float first_energy = buffer[0] * buffer[0],
        last_energy =  buffer[1] * buffer[1];  // handle this special case
  for (i = 1; i < half_dim; i++) {
    float real = buffer[i*2], im = buffer[i*2 + 1];
    buffer[i] = real*real + im*im;
  }
  buffer[0] = first_energy;
  buffer[half_dim] = last_energy;  
 
  float sqrt_data;
  //Apply mel filterbanks
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    j = 0;
    float mel_energy = 0;
    int32_t first_index = fbank_filter_first[bin];
    int32_t last_index = fbank_filter_last[bin];
    for (i = first_index; i <= last_index; i++) {
      //arm_sqrt_f32(buffer[i],&sqrt_data);
	  sqrt_data = buffer[i];
      mel_energy += (sqrt_data) * mel_fbank[bin][j++];
    }
    mel_energies[bin] = mel_energy;

    //avoid log of zero
    //if (mel_energy == 0.0)
    //  mel_energies[bin] = FLT_MIN;
  }

  //Take log
  for (bin = 0; bin < NUM_FBANK_BINS; bin++){
    mel_energies[bin] = logf(mel_energies[bin] + 1e-6);
	//Input is Qx.dec_bits (from quantization step)
    mel_energies[bin] *= (0x1<<dec_bits);
    mel_energies[bin] = round(mel_energies[bin]); 
    if(mel_energies[bin] >= 127)
      log_mel_out[bin] = 127;
    else if(mel_energies[bin] <= -128)
      log_mel_out[bin] = -128;
    else
      log_mel_out[bin] = mel_energies[bin];
  }

}