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
#define TEST_TIMING 0
#include "ds_cnn.h"
const q7_t DS_CNN::conv1_wt[CONV1_WT_DIM] = CONV1_WEIGHTS;
const q7_t DS_CNN::conv1_bias[CONV1_BIAS_DIM] = CONV1_BIAS;
const q7_t DS_CNN::dw_conv1_wt[DW_CONV_WT_DIM] = DW_CONV1_WEIGHTS;
const q7_t DS_CNN::dw_conv1_bias[DW_CONV_BIAS_DIM] = DW_CONV1_BIAS;
const q7_t DS_CNN::pw_conv1_wt[PW_CONV_WT_DIM] = PW_CONV1_WEIGHTS;
const q7_t DS_CNN::pw_conv1_bias[PW_CONV_BIAS_DIM] = PW_CONV1_BIAS;
const q7_t DS_CNN::dw_conv2_wt[DW_CONV_WT_DIM] = DW_CONV2_WEIGHTS;
const q7_t DS_CNN::dw_conv2_bias[DW_CONV_BIAS_DIM] = DW_CONV2_BIAS;
const q7_t DS_CNN::pw_conv2_wt[PW_CONV_WT_DIM] =PW_CONV2_WEIGHTS;
const q7_t DS_CNN::pw_conv2_bias[PW_CONV_BIAS_DIM] = PW_CONV2_BIAS;
const q7_t DS_CNN::dw_conv3_wt[DW_CONV_WT_DIM] = DW_CONV3_WEIGHTS;
const q7_t DS_CNN::dw_conv3_bias[DW_CONV_BIAS_DIM] =DW_CONV3_BIAS;
const q7_t DS_CNN::pw_conv3_wt[PW_CONV_WT_DIM] = PW_CONV3_WEIGHTS;
const q7_t DS_CNN::pw_conv3_bias[PW_CONV_BIAS_DIM]= PW_CONV3_BIAS;
const q7_t DS_CNN::dw_conv4_wt[DW_CONV_WT_DIM] = DW_CONV4_WEIGHTS;
const q7_t DS_CNN::dw_conv4_bias[DW_CONV_BIAS_DIM]= DW_CONV4_BIAS;
const q7_t DS_CNN::pw_conv4_wt[PW_CONV_WT_DIM] = PW_CONV4_WEIGHTS;
const q7_t DS_CNN::pw_conv4_bias[PW_CONV_BIAS_DIM] = PW_CONV4_BIAS;
const q7_t DS_CNN::dw_conv5_wt[DW_CONV_WT_DIM] = DW_CONV5_WEIGHTS;
const q7_t DS_CNN::dw_conv5_bias[DW_CONV_BIAS_DIM]= DW_CONV5_BIAS;
const q7_t DS_CNN::pw_conv5_wt[PW_CONV_WT_DIM] = PW_CONV5_WEIGHTS;
const q7_t DS_CNN::pw_conv5_bias[PW_CONV_BIAS_DIM] = PW_CONV5_BIAS;
const q7_t DS_CNN::dw_conv6_wt[DW_CONV_WT_DIM] = DW_CONV6_WEIGHTS;
const q7_t DS_CNN::dw_conv6_bias[DW_CONV_BIAS_DIM]= DW_CONV6_BIAS;
const q7_t DS_CNN::pw_conv6_wt[PW_CONV_WT_DIM] = PW_CONV6_WEIGHTS;
const q7_t DS_CNN::pw_conv6_bias[PW_CONV_BIAS_DIM] = PW_CONV6_BIAS;
const q7_t DS_CNN::fc_wt[FC_WT_DIM] = FC_WEIGHTS;
const q7_t DS_CNN::fc_bias[FC_BIAS_DIM] = FC_BIAS;
	
/* 	
const q7_t DS_CNN::ip1_wt[IP1_WT_DIM]=IP1_WT;
const q7_t DS_CNN::ip1_bias[IP1_OUT_DIM]=IP1_BIAS;
const q7_t DS_CNN::ip2_wt[IP2_WT_DIM]=IP2_WT;
const q7_t DS_CNN::ip2_bias[IP2_OUT_DIM]=IP2_BIAS;
const q7_t DS_CNN::ip3_wt[IP3_WT_DIM]=IP3_WT;
const q7_t DS_CNN::ip3_bias[IP3_OUT_DIM]=IP3_BIAS;
const q7_t DS_CNN::ip4_wt[IP4_WT_DIM]=IP4_WT;
const q7_t DS_CNN::ip4_bias[OUT_DIM]=IP4_BIAS;
 */
DS_CNN::DS_CNN(q7_t* scratch_pad)
{
  conv1_out = scratch_pad;
  dw_conv1_out = conv1_out + CONV1_OUT_DIM;
  pw_conv1_out = conv1_out;
  dw_conv2_out = conv1_out + PW_CONV_OUT_DIM;
  pw_conv2_out = conv1_out;
  dw_conv3_out = conv1_out + PW_CONV_OUT_DIM;
  pw_conv3_out = conv1_out;
  dw_conv4_out = conv1_out + PW_CONV_OUT_DIM;
  pw_conv4_out = conv1_out;
  dw_conv5_out = conv1_out + PW_CONV_OUT_DIM;
  pw_conv5_out = conv1_out;
  dw_conv6_out = conv1_out + PW_CONV_OUT_DIM;
  pw_conv6_out = conv1_out;
  avg_pool_out = conv1_out + PW_CONV_OUT_DIM;
  fc_out = conv1_out;
  
  vec_buffer = (q15_t*)(conv1_out + CONV1_OUT_DIM + PW_CONV_OUT_DIM);
  
 /*  ip1_out = scratch_pad;
  ip2_out = ip1_out+IP1_OUT_DIM;
  ip3_out = ip1_out;
  vec_buffer = (q15_t*)(ip1_out+IP1_OUT_DIM+IP2_OUT_DIM); */
}

DS_CNN::~DS_CNN()
{
}

void DS_CNN::run_nn(q7_t* in_data, q7_t* out_data)
{
	
	Serial pc_c(USBTX, USBRX);	
	Timer T;
	T.start();
	int start;
	int stop;
	
	/* for(int i = 0; i< NUM_FRAMES; i++){
	  for(int j = 0; j< NUM_FBANK_BINS; j++){
		  pc_c.printf("%.2f : ",(float)((int)in_data[i*NUM_FBANK_BINS+j])*0.25);
	  }	
	  pc_c.printf("\r\n");
    }	
	arm_relu_q7(in_data,20*49);
	for(int i = 0; i< NUM_FRAMES; i++){
	  for(int j = 0; j< NUM_FBANK_BINS; j++){
		  pc_c.printf("%.2f : ",(float)((int)in_data[i*NUM_FBANK_BINS+j])*0.25);
	  }	
	  pc_c.printf("\r\n");
    } */
	// Run all layers
	// Conv 1
	if(TEST_TIMING){
		start=T.read_us();
	}
	
	arm_depthwise_separable_conv_HWC_q7_nonsquare(in_data, CONV1_IN_DIM_X, CONV1_IN_DIM_Y, CONV1_IN_CH, conv1_wt,
					CONV1_OUT_CH, CONV1_KERNEL_DIM_X, CONV1_KERNEL_DIM_Y, CONV1_PADDING_X, CONV1_PADDING_Y, 
					CONV1_STRIDE_X, CONV1_STRIDE_Y, conv1_bias, CONV1_BIAS_LEFT_SHIFT, CONV1_OUTPUT_RIGHT_SHIFT,
					conv1_out, CONV1_OUT_DIM_X, CONV1_OUT_DIM_Y, vec_buffer,NULL,1);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("Conv1 execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	// Split ReLu because size of conv1_out is to big for the uint16_t size variable to handle
	if(TEST_TIMING){
		start=T.read_us();
	}
	arm_relu_q7(conv1_out, CONV1_OUT_DIM/2);
	arm_relu_q7(conv1_out+CONV1_OUT_DIM/2, CONV1_OUT_DIM/2);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("Relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	/* pc_c.printf("Hello - Number of elements in Conv 1 output: %d \r\n", CONV1_OUT_DIM);
	for (int t = 0; t < 25;t++){
		for(int f = 0; f< 20; f++){
			pc_c.printf("%.3f, ", ((float)conv1_out[layer_no + 172*f + 172*20*t])/32.0);
		}
		pc_c.printf("\r\n");
	}	
    int max_val = 0;
	for(int i = 0; i < CONV1_OUT_DIM;i++){
		if((int)conv1_out[i] > max_val){
			max_val =(int) conv1_out[i];
        }
	}
	pc_c.printf("Max value in conv1 out layer: %.3f\r\n", ((float)max_val)/32.0);
	
	for(int i = 0; i< 172; i++){
		pc_c.printf("%.3f, ", ((float)conv1_out[i])/32.0);
    } */
	// Depthwise conv 1
	if(TEST_TIMING){
		start=T.read_us();
	}
	arm_depthwise_separable_conv_HWC_q7_nonsquare(conv1_out, DW_CONV1_IN_DIM_X, DW_CONV1_IN_DIM_Y, DW_CONV_IN_CH, dw_conv1_wt,
					DW_CONV_OUT_CH, DW_CONV_KERNEL_DIM_X, DW_CONV_KERNEL_DIM_Y, DW_CONV1_PADDING_X, DW_CONV1_PADDING_Y, 
					DW_CONV1_STRIDE_X, DW_CONV1_STRIDE_Y, dw_conv1_bias, DW_CONV1_BIAS_LEFT_SHIFT, DW_CONV1_OUTPUT_RIGHT_SHIFT,
					dw_conv1_out, DW_CONV_OUT_DIM_X, DW_CONV_OUT_DIM_Y, vec_buffer,NULL,0);
	arm_relu_q7(dw_conv1_out, DW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("DW-conv1+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Pointwise conv 1
	arm_convolve_1x1_HWC_q7_fast_nonsquare(dw_conv1_out, PW_CONV_IN_DIM_X, PW_CONV_IN_DIM_Y, PW_CONV_IN_CH, pw_conv1_wt,
					PW_CONV_OUT_CH, PW_CONV_KERNEL_DIM_X, PW_CONV_KERNEL_DIM_Y, PW_CONV_PADDING_X, PW_CONV_PADDING_Y, 
					PW_CONV_STRIDE_X, PW_CONV_STRIDE_Y, pw_conv1_bias, PW_CONV1_BIAS_LEFT_SHIFT, PW_CONV1_OUTPUT_RIGHT_SHIFT,
					pw_conv1_out, PW_CONV_OUT_DIM_X, PW_CONV_OUT_DIM_Y, vec_buffer,NULL);
	arm_relu_q7(pw_conv1_out, PW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("PW-conv1+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Depthwise conv 2
	arm_depthwise_separable_conv_HWC_q7_nonsquare(pw_conv1_out, DW_CONV_IN_DIM_X, DW_CONV_IN_DIM_Y, DW_CONV_IN_CH, dw_conv2_wt,
					DW_CONV_OUT_CH, DW_CONV_KERNEL_DIM_X, DW_CONV_KERNEL_DIM_Y, DW_CONV_PADDING_X, DW_CONV_PADDING_Y, 
					DW_CONV_STRIDE_X, DW_CONV_STRIDE_Y, dw_conv2_bias, DW_CONV2_BIAS_LEFT_SHIFT, DW_CONV2_OUTPUT_RIGHT_SHIFT,
					dw_conv2_out, DW_CONV_OUT_DIM_X, DW_CONV_OUT_DIM_Y, vec_buffer,NULL,0);
	arm_relu_q7(dw_conv2_out, DW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("DW-conv2+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Pointwise conv 2
	arm_convolve_1x1_HWC_q7_fast_nonsquare(dw_conv2_out, PW_CONV_IN_DIM_X, PW_CONV_IN_DIM_Y, PW_CONV_IN_CH, pw_conv2_wt,
					PW_CONV_OUT_CH, PW_CONV_KERNEL_DIM_X, PW_CONV_KERNEL_DIM_Y, PW_CONV_PADDING_X, PW_CONV_PADDING_Y, 
					PW_CONV_STRIDE_X, PW_CONV_STRIDE_Y, pw_conv2_bias, PW_CONV2_BIAS_LEFT_SHIFT, PW_CONV2_OUTPUT_RIGHT_SHIFT,
					pw_conv2_out, PW_CONV_OUT_DIM_X, PW_CONV_OUT_DIM_Y, vec_buffer,NULL);
	arm_relu_q7(pw_conv2_out, PW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("PW-conv2+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Depthwise conv 3
	arm_depthwise_separable_conv_HWC_q7_nonsquare(pw_conv2_out, DW_CONV_IN_DIM_X, DW_CONV_IN_DIM_Y, DW_CONV_IN_CH, dw_conv3_wt,
					DW_CONV_OUT_CH, DW_CONV_KERNEL_DIM_X, DW_CONV_KERNEL_DIM_Y, DW_CONV_PADDING_X, DW_CONV_PADDING_Y, 
					DW_CONV_STRIDE_X, DW_CONV_STRIDE_Y, dw_conv3_bias, DW_CONV3_BIAS_LEFT_SHIFT, DW_CONV3_OUTPUT_RIGHT_SHIFT,
					dw_conv3_out, DW_CONV_OUT_DIM_X, DW_CONV_OUT_DIM_Y, vec_buffer,NULL,0);
	arm_relu_q7(dw_conv3_out, DW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("DW-conv3+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Pointwise conv 3
	arm_convolve_1x1_HWC_q7_fast_nonsquare(dw_conv3_out, PW_CONV_IN_DIM_X, PW_CONV_IN_DIM_Y, PW_CONV_IN_CH, pw_conv3_wt,
					PW_CONV_OUT_CH, PW_CONV_KERNEL_DIM_X, PW_CONV_KERNEL_DIM_Y, PW_CONV_PADDING_X, PW_CONV_PADDING_Y, 
					PW_CONV_STRIDE_X, PW_CONV_STRIDE_Y, pw_conv3_bias, PW_CONV3_BIAS_LEFT_SHIFT, PW_CONV3_OUTPUT_RIGHT_SHIFT,
					pw_conv3_out, PW_CONV_OUT_DIM_X, PW_CONV_OUT_DIM_Y, vec_buffer,NULL);
	arm_relu_q7(pw_conv3_out, PW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("PW-conv3+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Depthwise conv 4
	arm_depthwise_separable_conv_HWC_q7_nonsquare(pw_conv3_out, DW_CONV_IN_DIM_X, DW_CONV_IN_DIM_Y, DW_CONV_IN_CH, dw_conv4_wt,
					DW_CONV_OUT_CH, DW_CONV_KERNEL_DIM_X, DW_CONV_KERNEL_DIM_Y, DW_CONV_PADDING_X, DW_CONV_PADDING_Y, 
					DW_CONV_STRIDE_X, DW_CONV_STRIDE_Y, dw_conv4_bias, DW_CONV4_BIAS_LEFT_SHIFT, DW_CONV4_OUTPUT_RIGHT_SHIFT,
					dw_conv4_out, DW_CONV_OUT_DIM_X, DW_CONV_OUT_DIM_Y, vec_buffer,NULL,0);
	arm_relu_q7(dw_conv4_out, DW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("DW-conv4+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Pointwise conv 4
	arm_convolve_1x1_HWC_q7_fast_nonsquare(dw_conv4_out, PW_CONV_IN_DIM_X, PW_CONV_IN_DIM_Y, PW_CONV_IN_CH, pw_conv4_wt,
					PW_CONV_OUT_CH, PW_CONV_KERNEL_DIM_X, PW_CONV_KERNEL_DIM_Y, PW_CONV_PADDING_X, PW_CONV_PADDING_Y, 
					PW_CONV_STRIDE_X, PW_CONV_STRIDE_Y, pw_conv4_bias, PW_CONV4_BIAS_LEFT_SHIFT, PW_CONV4_OUTPUT_RIGHT_SHIFT,
					pw_conv4_out, PW_CONV_OUT_DIM_X, PW_CONV_OUT_DIM_Y, vec_buffer,NULL);
	arm_relu_q7(pw_conv4_out, PW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("PW-conv4+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Depthwise conv 5
	arm_depthwise_separable_conv_HWC_q7_nonsquare(pw_conv4_out, DW_CONV_IN_DIM_X, DW_CONV_IN_DIM_Y, DW_CONV_IN_CH, dw_conv5_wt,
					DW_CONV_OUT_CH, DW_CONV_KERNEL_DIM_X, DW_CONV_KERNEL_DIM_Y, DW_CONV_PADDING_X, DW_CONV_PADDING_Y, 
					DW_CONV_STRIDE_X, DW_CONV_STRIDE_Y, dw_conv5_bias, DW_CONV5_BIAS_LEFT_SHIFT, DW_CONV5_OUTPUT_RIGHT_SHIFT,
					dw_conv5_out, DW_CONV_OUT_DIM_X, DW_CONV_OUT_DIM_Y, vec_buffer,NULL,0);
	arm_relu_q7(dw_conv5_out, DW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("DW-conv5+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Pointwise conv 5
	arm_convolve_1x1_HWC_q7_fast_nonsquare(dw_conv5_out, PW_CONV_IN_DIM_X, PW_CONV_IN_DIM_Y, PW_CONV_IN_CH, pw_conv5_wt,
					PW_CONV_OUT_CH, PW_CONV_KERNEL_DIM_X, PW_CONV_KERNEL_DIM_Y, PW_CONV_PADDING_X, PW_CONV_PADDING_Y, 
					PW_CONV_STRIDE_X, PW_CONV_STRIDE_Y, pw_conv5_bias, PW_CONV5_BIAS_LEFT_SHIFT, PW_CONV5_OUTPUT_RIGHT_SHIFT,
					pw_conv5_out, PW_CONV_OUT_DIM_X, PW_CONV_OUT_DIM_Y, vec_buffer,NULL);
	arm_relu_q7(pw_conv5_out, PW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("PW-conv5+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Depthwise conv 6
	arm_depthwise_separable_conv_HWC_q7_nonsquare(pw_conv5_out, DW_CONV_IN_DIM_X, DW_CONV_IN_DIM_Y, DW_CONV_IN_CH, dw_conv6_wt,
					DW_CONV_OUT_CH, DW_CONV_KERNEL_DIM_X, DW_CONV_KERNEL_DIM_Y, DW_CONV_PADDING_X, DW_CONV_PADDING_Y, 
					DW_CONV_STRIDE_X, DW_CONV_STRIDE_Y, dw_conv6_bias, DW_CONV6_BIAS_LEFT_SHIFT, DW_CONV6_OUTPUT_RIGHT_SHIFT,
					dw_conv6_out, DW_CONV_OUT_DIM_X, DW_CONV_OUT_DIM_Y, vec_buffer,NULL,0);
	arm_relu_q7(dw_conv6_out, DW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("DW-conv6+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Pointwise conv 6
	arm_convolve_1x1_HWC_q7_fast_nonsquare(dw_conv6_out, PW_CONV_IN_DIM_X, PW_CONV_IN_DIM_Y, PW_CONV_IN_CH, pw_conv6_wt,
					PW_CONV_OUT_CH, PW_CONV_KERNEL_DIM_X, PW_CONV_KERNEL_DIM_Y, PW_CONV_PADDING_X, PW_CONV_PADDING_Y, 
					PW_CONV_STRIDE_X, PW_CONV_STRIDE_Y, pw_conv6_bias, PW_CONV6_BIAS_LEFT_SHIFT, PW_CONV6_OUTPUT_RIGHT_SHIFT,
					pw_conv6_out, PW_CONV_OUT_DIM_X, PW_CONV_OUT_DIM_Y, vec_buffer,NULL);
	arm_relu_q7(pw_conv6_out, PW_CONV_OUT_DIM);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("PW-conv6+relu execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// Average Pooling
	for (int t = 0; t < 13;t++){
		for(int f = 0; f < 10;f++){
			for(int ch = 0; ch < CONV1_IN_CH; ch++){
				vec_buffer[ch] += (int16_t)pw_conv4_out[t*10*CONV1_IN_CH+f*CONV1_IN_CH+ch];
			}
		}
	}
	float temp;
	for (int ch = 0; ch < CONV1_IN_CH; ch++){
		temp = (float)vec_buffer[ch];
		temp = temp/130.0;
		avg_pool_out[ch] = (q7_t)(round(temp)); // Convert to Q2.5
	}
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("AVG-pool execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	
	if(TEST_TIMING){
		start=T.read_us();
	}
	// FC-layer
	arm_fully_connected_q7(avg_pool_out, fc_wt, AVG_POOL_OUT_DIM, OUT_DIM, FC_BIAS_LEFT_SHIFT, FC_OUTPUT_RIGHT_SHIFT, fc_bias, out_data, vec_buffer);
	if(TEST_TIMING){
		stop=T.read_us();
		pc_c.printf("FC-layer execution time: %.1f ms\r\n", (float)(stop-start)/1000.0);
	}
	
	T.stop();
	/* 
	// IP1 
	arm_fully_connected_q7(in_data, ip1_wt, IN_DIM, IP1_OUT_DIM, 1, 7, ip1_bias, ip1_out, vec_buffer);
        // RELU1
	arm_relu_q7(ip1_out, IP1_OUT_DIM);

	// IP2 
	arm_fully_connected_q7(ip1_out, ip2_wt, IP1_OUT_DIM, IP2_OUT_DIM, 2, 8, ip2_bias, ip2_out, vec_buffer);
        // RELU2
	arm_relu_q7(ip2_out, IP2_OUT_DIM);

	// IP3 
	arm_fully_connected_q7(ip2_out, ip3_wt, IP2_OUT_DIM, IP3_OUT_DIM, 2, 9, ip3_bias, ip3_out, vec_buffer);
        // RELU3
	arm_relu_q7(ip3_out, IP3_OUT_DIM);

	// IP4 
	arm_fully_connected_q7(ip3_out, ip4_wt, IP3_OUT_DIM, OUT_DIM, 0, 6, ip4_bias, out_data, vec_buffer);
 */
}


