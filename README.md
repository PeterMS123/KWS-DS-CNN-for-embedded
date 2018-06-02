# KWS-DS-CNN-for-embedded
This repository contains modified python scripts, based on the Speech Commands Tensorflow example, for training tensorflow models based on depthwise separable convolutional neural networks for keyword spotting. It also contains scripts for quantizing trained networks and evaluating performance in single-inference tests and on continuous audio streams.
C++ source code for implementation of a pretrained network on the Cortex M4 based FRDM K66F development board is also included.

# Training and deployment
To train networks and deploying them to ARM Cortex-M boards, the guides from [ARM's](https://github.com/ARM-software/ML-KWS-for-MCU) should be followed. 

**Note:** After cloning the CMSIS-NN library, the "CMSIS\NN\Source\ConvolutionFunctions\arm_depthwise_separable_conv_HWC_q7_nonsquare.c" function must be replaced with the slightly modified version found in this repository.


