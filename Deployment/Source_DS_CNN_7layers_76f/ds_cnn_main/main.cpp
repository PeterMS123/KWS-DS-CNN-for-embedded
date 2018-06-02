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
 * Modified for implementation on FRDM K66F development board, using AUX or on-board mic input.
 * Switched DNN-classifier out with Depthwise separable convolutional neural network classifier
 */

/*
 * Description: Example code for running keyword spotting on Cortex-M boards
 */

#include "kws_ds_cnn.h"
#include "arm_nnfunctions.h"
#include "arm_math.h"
#include "fsl_clock_config.h"
#include "fsl_sai.h"
#include "fsl_port.h"
#include "DA7212.h"
#include "fsl_common.h"

#include "mbed.h"



DA7212 codec(I2C_SDA, I2C_SCL);
Serial *pc;
Timer *T;

char output_string[256];
char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};




/* AUDIO_WINDOW_SIZE is the number of audio samples in each processing window */
// 16kHz sampling freq, 40 ms frames, 20 ms frame shift - should store 14 frames = 300 ms
#define AUDIO_WINDOW_SIZE   ((int16_t)(300*SAMP_FREQ*0.001))
/* AUDIO_RECORD_SIZE is the number of audio samples in each recording window */
// should record 280 ms audio
#define AUDIO_RECORD_SIZE   ((int16_t)(280*SAMP_FREQ*0.001))
/* AUDIO_BUFFER_SIZE is the number of audio samples in the recording buffer, */
// 16kHz sampling freq, 40 ms frames, 20 ms frame shift - Must have room for 2 recording windows plus 1 frame-shift = 580 ms
#define AUDIO_BUFFER_SIZE   ((int16_t)(580*SAMP_FREQ*0.001))
/* AUDIO_BUFFER_OFFSET is the number of audio samples to offset the recording with */
// The first half recording must start after 20 ms
#define AUDIO_BUFFER_OFFSET   ((int16_t)(20*SAMP_FREQ*0.001))
int16_t audio_io_buffer[AUDIO_BUFFER_SIZE]; // ping-pong buffer
//int16_t audio_buffer[AUDIO_WINDOW_SIZE];


int16_t* AUDIO_BUFFER_IN = audio_io_buffer;

q7_t scratch_buffer[SCRATCH_BUFFER_SIZE];
//KWS_DS_CNN *kws_ds_cnn;

uint8_t g_buffer[60];
uint8_t g_counter = 0;
uint8_t g_IdleCounter = 0;
uint8_t g_NotIdleCounter = 0;
uint8_t recording_done_flag = 0;


void initTFT(void)
{
    //Configure the display driver    
    TFT.FastWindow(true) ;
    TFT.background(Black);
    TFT.foreground(White);
    wait(0.01) ;
    TFT.cls();
}


void i2scallback(I2S_Type *base, sai_handle_t *handle, status_t status, void *userData)
{
  g_counter++;
  if (status == kStatus_SAI_RxIdle)
  {
    g_IdleCounter++;

    sai_transfer_t transfer;
    if (g_counter & 0x1) // Odd means first is full and second part of buffer is free
    {
      //firstSemaphore.release();
      transfer.data = (uint8_t*)(AUDIO_BUFFER_IN + AUDIO_BUFFER_OFFSET + AUDIO_RECORD_SIZE);
	  
	  //audio_buffer = buffer_1;//AUDIO_BUFFER_IN;
    }
    else // even means second is full and first part of buffer is free
    {
      //secondSemaphore.release();
      transfer.data = (uint8_t*)(AUDIO_BUFFER_IN + AUDIO_BUFFER_OFFSET);
	  //audio_buffer = buffer_2;//AUDIO_BUFFER_IN + AUDIO_RECORD_SIZE;
    }
    transfer.dataSize = AUDIO_RECORD_SIZE * 2; // 16bit to 8bit
    recording_done_flag++;;
    status_t s = SAI_TransferReceiveNonBlocking(I2S0, handle, &transfer);
	//pc->printf("g_counter: %d\r\n",g_counter);
    if (s != kStatus_Success)
    {
      pc->printf("SAI_TransferReceiveNonBlocking error %d\n", s);
    }
  
  }
  else
  {
    pc->printf("I2S error\r\n");
    g_NotIdleCounter++;
  }
}

int main()
{
  BOARD_BootClockHSRUN();
  wait_ms(100);
  
  CLOCK_EnableClock(kCLOCK_PortE);
  PORT_SetPinMux(PORTE, 7U, kPORT_MuxAlt4);
  PORT_SetPinMux(PORTE, 8U, kPORT_MuxAlt4);
  PORT_SetPinMux(PORTE, 9U, kPORT_MuxAlt4);
  PORT_SetPinMux(PORTC, 6U, kPORT_MuxAlt6);
  
  int16_t audio_buffer[AUDIO_WINDOW_SIZE];
  pc = new Serial(USBTX,USBRX);
  T = new Timer;
  //pc->baud(115200);
  pc->printf("SAMPLING FREQUENCY: %d\r\n", SAMP_FREQ);
  pc->printf("AUDIO_WINDOW_SIZE: %d\r\n", AUDIO_WINDOW_SIZE);
  pc->printf("AUDIO_RECORD_SIZE: %d\r\n", AUDIO_RECORD_SIZE);
  pc->printf("AUDIO_BUFFER_SIZE: %d\r\n", AUDIO_BUFFER_SIZE);
  pc->printf("AUDIO_BUFFER_OFFSET: %d\r\n", AUDIO_BUFFER_OFFSET);
    // uint8_t aux, mix, mixsel, adc;

  
  // AUX input
  if(1){

	   // SYSTEM INITIALIZATION REGISTERS
	  uint8_t CIF_REG_SOFT_RESET = 0x80;
	  codec.i2c_register_write(DA7212::REG_CIF_CTRL, CIF_REG_SOFT_RESET);

	  wait_ms(50);
	  // Setup clocks and references
	  // WRITE DA7212 0x23 0x08  //Enable Bias
	  codec.i2c_register_write(DA7212::REG_REFERENCES, 0x08);
	  
	  // WRITE DA7212 0x92 0x02  //Set Ramp rate to 1 second
	  // codec.i2c_register_write(DA7212::REG_GAIN_RAMP_CTRL, 0x02);
	  wait_ms(50);
	  // WRITE DA7212 0x90 0xFF  //Enable Digital LDO

	  codec.i2c_register_write(DA7212::REG_LDO_CTRL, 0xFF);
	  // WRITE DA7212 0x47 0xCD  //Enable Charge Pump (fixed VDD/1)

	  // codec.i2c_register_write(DA7212::REG_CP_CTRL, 0xCD);

	  // WRITE DA7212 0x29 0xC8  //Enable AIF 16bit I2S mode
	  // codec.i2c_register_write(DA7212::REG_DAI_CTRL, 0xC8);

	  uint8_t DAI_EN = 0x80, DAI_OE = 0x40;
	  codec.i2c_register_write(DA7212::REG_DAI_CTRL, DAI_EN | DAI_OE);

	  // WRITE DA7212 0x28 0x01  //Slave Mode
	  // codec.i2c_register_write(DA7212::REG_DAI_CLK_MODE, 0x01);

	  uint8_t DAI_CLK_ENABLE = 0x80;
	  codec.i2c_register_write(DA7212::REG_DAI_CLK_MODE, DAI_CLK_ENABLE);

	  // WRITE DA7212 0x22 0x0B  //Set incoming sample rate to 48kHz
	  // codec.i2c_register_write(DA7212::REG_SR, 0x0B);

	  uint8_t sample_rate_16kHz = 0b0101;
	  codec.i2c_register_write(DA7212::REG_SR, sample_rate_16kHz);
	  // WRITE DA7212 0x62 0xAA  //Enable MICBIAS 1 & 2
	  codec.i2c_register_write(DA7212::REG_MICBIAS_CTRL, 0xAA);

	  // WRITE DA7212 0x21 0x10  //DIG Routing ADC
	  codec.i2c_register_write(DA7212::REG_DIG_ROUTING_DAI, 0x10);

	  // WRITE DA7212 0x22 0x0B  //SR assuming 48KHz
	  // codec.i2c_register_write(DA7212::REG_SR, 0x0B); // duplicate?
	  // codec.i2c_register_write(DA7212::REG_SR, sample_rate_16kHz);

	  // WRITE DA7212 0x24 0x06  //PLL_FRAC_TOP
	  codec.i2c_register_write(DA7212::REG_PLL_FRAC_TOP, 0x06);

	  // WRITE DA7212 0x25 0xDC  //PLL_FRAC_BOT
	  codec.i2c_register_write(DA7212::REG_PLL_FRAC_BOT, 0xDC);

	  // WRITE DA7212 0x20 0x1A  //PLL_FBDIV_INT
	  codec.i2c_register_write(DA7212::REG_PLL_INTEGER, 0x1A);

	  // WRITE DA7212 0x27 0xC4  //MCLK 15MHz

	  // codec.i2c_register_write(DA7212::REG_PLL_CTRL, 0xC4);

	  codec.i2c_register_write(DA7212::REG_PLL_CTRL, 0x84);

	  // WRITE DA7212 0x67 0x80  //ADC_L_CTRL
	  codec.i2c_register_write(DA7212::REG_ADC_L_CTRL, 0x80);

	  // WRITE DA7212 0x68 0x80  //ADC_R_CTRL
	  codec.i2c_register_write(DA7212::REG_ADC_R_CTRL, 0x80);

	  wait_ms(20);
	  // WRITE DA7212 0x94 0x00  //PC_COUT_Resync
	  // codec.i2c_register_write(DA7212::REG_PC_COUNT, 0x00);

	  uint8_t PC_RESYNC_AUTO = 0x20;
	  codec.i2c_register_write(DA7212::REG_PC_COUNT, PC_RESYNC_AUTO);
	  //Configure Inputs/Outputs
	  /* // WRITE DA7212 0x32 0x80  //DMIC_L_EN
	  codec.i2c_register_write(DA7212::REG_MIXIN_L_SELECT, 0x80);

	  // WRITE DA7212 0x33 0x80  //DMIC_R_EN
	  codec.i2c_register_write(DA7212::REG_MIXIN_R_SELECT, 0x80); */

	  uint8_t AUX_L_SEL = 0x01;
	  codec.i2c_register_write(DA7212::REG_MIXIN_L_SELECT, AUX_L_SEL);

	  uint8_t AUX_R_SEL = 0x01;
	  codec.i2c_register_write(DA7212::REG_MIXIN_R_SELECT, AUX_R_SEL); 

	  // WRITE DA7212 0x65 0xA8  //Enable Left input mixer
	  codec.i2c_register_write(DA7212::REG_MIXIN_L_CTRL, 0xA8);	 

	  // WRITE DA7212 0x66 0xA8  //Enable right input mixer
	  codec.i2c_register_write(DA7212::REG_MIXIN_R_CTRL, 0xA8);
	  
	  
   // MIC input  
  }else{
	  // SYSTEM INITIALIZATION REGISTERS
	  uint8_t CIF_REG_SOFT_RESET = 0x80;
	  codec.i2c_register_write(DA7212::REG_CIF_CTRL, CIF_REG_SOFT_RESET);

	  wait_ms(50);
	  // Setup clocks and references
	  // WRITE DA7212 0x23 0x08  //Enable Bias
	  codec.i2c_register_write(DA7212::REG_REFERENCES, 0x08);
	  
	  // WRITE DA7212 0x92 0x02  //Set Ramp rate to 1 second
	  // codec.i2c_register_write(DA7212::REG_GAIN_RAMP_CTRL, 0x02);
	  wait_ms(50);
	  // WRITE DA7212 0x90 0xFF  //Enable Digital LDO

	  codec.i2c_register_write(DA7212::REG_LDO_CTRL, 0xFF);
	  // WRITE DA7212 0x47 0xCD  //Enable Charge Pump (fixed VDD/1)

	  // codec.i2c_register_write(DA7212::REG_CP_CTRL, 0xCD);

	  // WRITE DA7212 0x29 0xC8  //Enable AIF 16bit I2S mode
	  // codec.i2c_register_write(DA7212::REG_DAI_CTRL, 0xC8);

	  uint8_t DAI_EN = 0x80, DAI_OE = 0x40;
	  codec.i2c_register_write(DA7212::REG_DAI_CTRL, DAI_EN | DAI_OE);

	  // WRITE DA7212 0x28 0x01  //Slave Mode
	  // codec.i2c_register_write(DA7212::REG_DAI_CLK_MODE, 0x01);

	  uint8_t DAI_CLK_ENABLE = 0x80;
	  codec.i2c_register_write(DA7212::REG_DAI_CLK_MODE, DAI_CLK_ENABLE);

	  // WRITE DA7212 0x22 0x0B  //Set incoming sample rate to 48kHz
	  // codec.i2c_register_write(DA7212::REG_SR, 0x0B);

	  uint8_t sample_rate_16kHz = 0b0101;
	  codec.i2c_register_write(DA7212::REG_SR, sample_rate_16kHz);
	  // WRITE DA7212 0x62 0xAA  //Enable MICBIAS 1 & 2
	  codec.i2c_register_write(DA7212::REG_MICBIAS_CTRL, 0xAA);

	  // WRITE DA7212 0x21 0x10  //DIG Routing ADC
	  codec.i2c_register_write(DA7212::REG_DIG_ROUTING_DAI, 0x10);

	  // WRITE DA7212 0x22 0x0B  //SR assuming 48KHz
	  // codec.i2c_register_write(DA7212::REG_SR, 0x0B); // duplicate?
	  // codec.i2c_register_write(DA7212::REG_SR, sample_rate_16kHz);

	  // WRITE DA7212 0x24 0x06  //PLL_FRAC_TOP
	  codec.i2c_register_write(DA7212::REG_PLL_FRAC_TOP, 0x06);

	  // WRITE DA7212 0x25 0xDC  //PLL_FRAC_BOT
	  codec.i2c_register_write(DA7212::REG_PLL_FRAC_BOT, 0xDC);

	  // WRITE DA7212 0x20 0x1A  //PLL_FBDIV_INT
	  codec.i2c_register_write(DA7212::REG_PLL_INTEGER, 0x1A);

	  // WRITE DA7212 0x27 0xC4  //MCLK 15MHz

	  // codec.i2c_register_write(DA7212::REG_PLL_CTRL, 0xC4);

	  codec.i2c_register_write(DA7212::REG_PLL_CTRL, 0x84);

	  // WRITE DA7212 0x67 0x80  //ADC_L_CTRL
	  codec.i2c_register_write(DA7212::REG_ADC_L_CTRL, 0x80);

	  // WRITE DA7212 0x68 0x80  //ADC_R_CTRL
	  codec.i2c_register_write(DA7212::REG_ADC_R_CTRL, 0x80);

	  wait_ms(20);
	  // WRITE DA7212 0x94 0x00  //PC_COUT_Resync
	  // codec.i2c_register_write(DA7212::REG_PC_COUNT, 0x00);

	  uint8_t PC_RESYNC_AUTO = 0x20;
	  codec.i2c_register_write(DA7212::REG_PC_COUNT, PC_RESYNC_AUTO);
	  //Configure Inputs/Outputs
	  // WRITE DA7212 0x32 0x80  //DMIC_L_EN
	  codec.i2c_register_write(DA7212::REG_MIXIN_L_SELECT, 0x80);

	  // WRITE DA7212 0x33 0x80  //DMIC_R_EN
	  codec.i2c_register_write(DA7212::REG_MIXIN_R_SELECT, 0x80);	 

	  // WRITE DA7212 0x65 0xA8  //Enable Left input mixer
	  codec.i2c_register_write(DA7212::REG_MIXIN_L_CTRL, 0xA8);	 

	  // WRITE DA7212 0x66 0xA8  //Enable right input mixer
	  codec.i2c_register_write(DA7212::REG_MIXIN_R_CTRL, 0xA8);
  }
  
  sai_config_t config;
  sai_transfer_format_t format;
  sai_transfer_t transfer;

  sai_handle_t handle;

  config.protocol = kSAI_BusI2S;
  config.syncMode = kSAI_ModeAsync;
  config.mclkOutputEnable = true;
  config.masterSlave = kSAI_Slave;
  config.mclkSource = kSAI_MclkSourceSysclk;
  // config.bclkSource = kSAI_BclkSourceOtherSai0;
  
  SAI_RxInit(I2S0, &config);
  // pc.printf("RX Status: 0x%8X\n",SAI_RxGetStatusFlag(I2S0));

  uint32_t mclkSourceClockHz = 180000000; // core running 180 MHz
  uint32_t bclkSourceClockHz = 500000; // 500kHz

  format.sampleRate_Hz = kSAI_SampleRate16KHz;
  format.bitWidth = kSAI_WordWidth16bits;
  format.stereo = kSAI_Stereo;
  format.masterClockHz = 15000000; // 
  format.watermark = 4;
  format.channel = 0;
  format.protocol = kSAI_BusI2S;

  status_t s;
  SAI_TransferRxCreateHandle(I2S0, &handle, (sai_transfer_callback_t)i2scallback, &transfer);
  s = SAI_TransferRxSetFormat(I2S0, &handle, &format, mclkSourceClockHz, bclkSourceClockHz);
  if (s != kStatus_Success)
  {
    pc->printf("SAI_TransferRxSetFormat error %d\n", s);
  }
  wait_ms(500);
  //kws_ds_cnn = new KWS_DS_CNN(audio_buffer,scratch_buffer);
  KWS_DS_CNN kws_ds_cnn(audio_buffer,scratch_buffer);
  /* Initialize buffer */
  memset(AUDIO_BUFFER_IN, 0, AUDIO_BUFFER_SIZE);
  /* Start Recording */
  transfer.data = (uint8_t*)(AUDIO_BUFFER_IN + AUDIO_BUFFER_OFFSET);
  transfer.dataSize = AUDIO_RECORD_SIZE * 2; // Size of data to record [bytes]
  
  s = SAI_TransferReceiveNonBlocking(I2S0, &handle, &transfer);
  if (s != kStatus_Success)
  {
    pc->printf("SAI_TransferReceiveNonBlocking error %d\n", s);
  }
  
  pc->printf("SystemCoreClock = %d MHz\r\n", SystemCoreClock/1000000);
  pc->printf("idle: %d, not idle: %d , g_counter: %d \r\n", g_IdleCounter, g_NotIdleCounter, g_counter);

  T->start();
  int t1 = T->read_us();
  int t2;
  int n_runs = 0;
  int t_start = T->read_us();
  int a = 5000;
  q7_t prev_avg_prob[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  int pred_prev = 0;
  int current_pred = 0;
  int averaging_window_len = 2;  
  int detection_threshold = 95;  //in percent
  
 
  while(1){
	 // Wait around for a little while
	if(g_NotIdleCounter){
		pc->printf("g_NotIdleCounter: %u \r\n ", g_NotIdleCounter);
	}
	  
	if(recording_done_flag){
		if(recording_done_flag > 1){
			pc->printf("Oh shite!, recording_done_flag = %d \r\n", recording_done_flag);
			
		}
		n_runs++;
		
		t2 = T->read_us();
		//pc->printf("Running classifier, Time since last run: %.2f [ms] audio_buffer[0]: %d \r\n", (t2-t1)/1000.0, audio_buffer[0]);
		t1=t2;
		recording_done_flag--;
		int start = T->read_us();
		if(g_counter & 0x1){
			// Move the last 20 ms from previous recording to the front of the buffer
			memcpy(AUDIO_BUFFER_IN, AUDIO_BUFFER_IN + (AUDIO_BUFFER_SIZE - AUDIO_BUFFER_OFFSET), AUDIO_BUFFER_OFFSET*2);
			memcpy(audio_buffer, audio_io_buffer,AUDIO_WINDOW_SIZE*2);
		}else{
			memcpy(audio_buffer, audio_io_buffer+AUDIO_RECORD_SIZE, AUDIO_WINDOW_SIZE*2);
		}
        
		kws_ds_cnn.extract_features(14); //extract mfsc features
		kws_ds_cnn.classify();	    //classify using ds-cnn 
		memcpy(prev_avg_prob, kws_ds_cnn.averaged_output, 12);
		kws_ds_cnn.average_predictions(averaging_window_len);
		int max_ind = kws_ds_cnn.get_top_detection(kws_ds_cnn.averaged_output);

		current_pred = 0;
		if(((float)kws_ds_cnn.averaged_output[max_ind]*100/128.0) > detection_threshold){
			pc->printf("Detected %s (%.1f)\r\n",output_class[max_ind],((float)kws_ds_cnn.averaged_output[max_ind]*100/128.0));
			current_pred = max_ind+1;
		}
	}	  
 }
  return 0;
}

