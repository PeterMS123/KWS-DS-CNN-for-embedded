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
 * Implementation of depthwise separable CNN instead of DNN
 */

#ifndef KWS_DS_CNN_H
#define KWS_DS_CNN_H

#include "ds_cnn_weights.h"
#include "arm_nnfunctions.h"
#include "arm_math.h"
#include "log_mel.h"
#include "mbed.h"
/* Network Structure 

  49x20 input features
    |
   CONV1: 2D convolution with 76 kernels (10x4). T_stride = 2, F_stride = 1 (ReLu actication) -> Output size (25*20*76)
    |
   Depthwise convolution 1, 76 kernels (3x3), T_stride =2, F_stride = 2 (ReLu actication) -> Output size (13*10*76)
    |
   Pointwise convolution 1, 76 kernels (1x1), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Depthwise convolution 2, 76 kernels (3x3), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Pointwise convolution 2, 76 kernels (1x1), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Depthwise convolution 3, 76 kernels (3x3), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Pointwise convolution 3, 76 kernels (1x1), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Depthwise convolution 4, 76 kernels (3x3), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Pointwise convolution 4, 76 kernels (1x1), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Depthwise convolution 5, 76 kernels (3x3), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Pointwise convolution 5, 76 kernels (1x1), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Depthwise convolution 6, 76 kernels (3x3), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |
   Pointwise convolution 6, 76 kernels (1x1), T_stride =1, F_stride = 1 (ReLu actication) -> Output size (13*10*76)
    |	
   Average pooling, window size (13*10) -> Output size (76)
    |
   Fully Connected layer, 12 output nodes (Softmax activation)

*/

#define IN_DIM (NUM_FRAMES*NUM_FBANK_BINS)
#define OUT_DIM 12
#define CONV1_OUT_DIM (25*20*76)
#define DW_CONV_OUT_DIM (13*10*76)
#define PW_CONV_OUT_DIM (13*10*76)
#define AVG_POOL_OUT_DIM 76
#define CONV1_WT_DIM (10*4*76)
#define CONV1_BIAS_DIM 76
#define DW_CONV_WT_DIM (3*3*76)
#define DW_CONV_BIAS_DIM 76
#define PW_CONV_WT_DIM (76*76)
#define PW_CONV_BIAS_DIM 76
#define FC_WT_DIM (12*76)
#define FC_BIAS_DIM 12
#define VEC_BUFFER_MAX (2*10*4*76) 
#define SCRATCH_BUFFER_SIZE (CONV1_OUT_DIM + DW_CONV_OUT_DIM + VEC_BUFFER_MAX)

// Layer 1 parameters
#define CONV1_IN_DIM_X 20 // Frequency along x-axis
#define CONV1_IN_DIM_Y 49 // Time along y-axis
#define CONV1_IN_CH 76 // Fake, actually 1
#define CONV1_KERNEL_DIM_X 4
#define CONV1_KERNEL_DIM_Y 10
#define CONV1_PADDING_X 1  // Left side padding
#define CONV1_PADDING_Y 4 // Top padding
#define CONV1_STRIDE_X 1
#define CONV1_STRIDE_Y 2
#define CONV1_OUT_DIM_X 20 // Frequency along x-axis
#define CONV1_OUT_DIM_Y 25 // Time along y-axis
#define CONV1_OUT_CH 76

// Layer 2 depthwise parameters
#define DW_CONV1_IN_DIM_X 20 // Frequency along x-axis
#define DW_CONV1_IN_DIM_Y 25 // Time along y-axis
#define DW_CONV_IN_CH 76
#define DW_CONV_KERNEL_DIM_X 3
#define DW_CONV_KERNEL_DIM_Y 3
#define DW_CONV1_PADDING_X 0  // Left side padding
#define DW_CONV1_PADDING_Y 1 // Top padding
#define DW_CONV1_STRIDE_X 2
#define DW_CONV1_STRIDE_Y 2
#define DW_CONV_OUT_DIM_X 10 // Frequency along x-axis
#define DW_CONV_OUT_DIM_Y 13 // Time along y-axis
#define DW_CONV_OUT_CH 76

//layer 3-5 depthwise paramters
#define DW_CONV_IN_DIM_X 10 // Frequency along x-axis
#define DW_CONV_IN_DIM_Y 13 // Time along y-axis
#define DW_CONV_PADDING_X 1  // Left side padding
#define DW_CONV_PADDING_Y 1 // Top padding
#define DW_CONV_STRIDE_X 1
#define DW_CONV_STRIDE_Y 1


// Pointwise conv paramters
#define PW_CONV_IN_DIM_X 10 // Frequency along x-axis
#define PW_CONV_IN_DIM_Y 13 // Time along y-axis
#define PW_CONV_IN_CH 76
#define PW_CONV_KERNEL_DIM_X 1
#define PW_CONV_KERNEL_DIM_Y 1
#define PW_CONV_PADDING_X 0  // Left side padding
#define PW_CONV_PADDING_Y 0 // Top padding
#define PW_CONV_STRIDE_X 1
#define PW_CONV_STRIDE_Y 1
#define PW_CONV_OUT_DIM_X 10 // Frequency along x-axis
#define PW_CONV_OUT_DIM_Y 13 // Time along y-axis
#define PW_CONV_OUT_CH 76



class DS_CNN {

  public:
    DS_CNN(q7_t* scratch_pad);
    ~DS_CNN();
    void run_nn(q7_t* in_data, q7_t* out_data);

  private:
	
    q7_t* conv1_out;
    q7_t* dw_conv1_out;
    q7_t* pw_conv1_out;
	q7_t* dw_conv2_out;
    q7_t* pw_conv2_out;
	q7_t* dw_conv3_out;
    q7_t* pw_conv3_out;
	q7_t* dw_conv4_out;
    q7_t* pw_conv4_out;
	q7_t* dw_conv5_out;
    q7_t* pw_conv5_out;
	q7_t* dw_conv6_out;
    q7_t* pw_conv6_out;
	q7_t* avg_pool_out;
	q7_t* fc_out;
    q15_t* vec_buffer;
	static q7_t const conv1_wt[CONV1_WT_DIM];
	static q7_t const conv1_bias[CONV1_BIAS_DIM];
	static q7_t const dw_conv1_wt[DW_CONV_WT_DIM];
	static q7_t const dw_conv1_bias[DW_CONV_BIAS_DIM];
	static q7_t const pw_conv1_wt[PW_CONV_WT_DIM];
	static q7_t const pw_conv1_bias[PW_CONV_BIAS_DIM];
	static q7_t const dw_conv2_wt[DW_CONV_WT_DIM];
	static q7_t const dw_conv2_bias[DW_CONV_BIAS_DIM];
	static q7_t const pw_conv2_wt[PW_CONV_WT_DIM];
	static q7_t const pw_conv2_bias[PW_CONV_BIAS_DIM];
	static q7_t const dw_conv3_wt[DW_CONV_WT_DIM];
	static q7_t const dw_conv3_bias[DW_CONV_BIAS_DIM];
	static q7_t const pw_conv3_wt[PW_CONV_WT_DIM];
	static q7_t const pw_conv3_bias[PW_CONV_BIAS_DIM];
	static q7_t const dw_conv4_wt[DW_CONV_WT_DIM];
	static q7_t const dw_conv4_bias[DW_CONV_BIAS_DIM];
	static q7_t const pw_conv4_wt[PW_CONV_WT_DIM];
	static q7_t const pw_conv4_bias[PW_CONV_BIAS_DIM];
	static q7_t const dw_conv5_wt[DW_CONV_WT_DIM];
	static q7_t const dw_conv5_bias[DW_CONV_BIAS_DIM];
	static q7_t const pw_conv5_wt[PW_CONV_WT_DIM];
	static q7_t const pw_conv5_bias[PW_CONV_BIAS_DIM];
	static q7_t const dw_conv6_wt[DW_CONV_WT_DIM];
	static q7_t const dw_conv6_bias[DW_CONV_BIAS_DIM];
	static q7_t const pw_conv6_wt[PW_CONV_WT_DIM];
	static q7_t const pw_conv6_bias[PW_CONV_BIAS_DIM];
	static q7_t const fc_wt[FC_WT_DIM];
	static q7_t const fc_bias[FC_BIAS_DIM];
	
	

};

#endif
