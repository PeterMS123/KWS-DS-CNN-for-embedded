/**
* @author Giles Barton-Owen; fork by Kazuki Yamamoto
*
* @section LICENSE
*
* Copyright (c) 2016 k4zuki
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* @section DESCRIPTION
*    A Driver set for the I2C half of the DA7212
*
*/

#include "DA7212.h"

const uint8_t base_address = 0x34;  // 0x1A in 7bit address

DA7212::DA7212(PinName i2c_sda, PinName i2c_scl) : i2c(i2c_sda, i2c_scl) {
    address = base_address;
    // init();
    // defaulter();

    // form_cmd(all);
}

DA7212::DA7212(PinName i2c_sda, PinName i2c_scl, int i2c_addr) : i2c(i2c_sda, i2c_scl) {
    address = (uint8_t)i2c_addr;
    init();
    defaulter();

    form_cmd(all);
}

// DA7212::DA7212(PinName i2c_sda, PinName i2c_scl, bool cs_level) : i2c(i2c_sda, i2c_scl) {
//     address = base_address + (1 * cs_level);
//     defaulter();
//     form_cmd(all);
// }

void DA7212::init() {
    mic2_gain.min = mic1_gain.min = MIC_PGA_MIN;
    mic2_gain.max = mic1_gain.max = MIC_PGA_MAX;
    mic2_gain.step = mic1_gain.step = MIC_PGA_STEP;
    mic2_gain.por = mic1_gain.por = MIC_PGA_POR;
    mic2_gain.mask = mic1_gain.mask = 0x07;
    mic2_gain.width = mic1_gain.width = 3;

    aux_r_gain.min = aux_l_gain.min = AUX_PGA_MIN;
    aux_r_gain.max = aux_l_gain.max = AUX_PGA_MAX;
    aux_r_gain.step = aux_l_gain.step = AUX_PGA_STEP;
    aux_r_gain.por = aux_l_gain.por = AUX_PGA_POR;
    aux_r_gain.mask = aux_l_gain.mask = 0x3F;
    aux_r_gain.width = aux_l_gain.width = 6;

    mix_r_in_gain.min = mix_l_in_gain.min = MIXIN_PGA_MIN;
    mix_r_in_gain.max = mix_l_in_gain.max = MIXIN_PGA_MAX;
    mix_r_in_gain.step = mix_l_in_gain.step = MIXIN_PGA_STEP;
    mix_r_in_gain.por = mix_l_in_gain.por = MIXIN_PGA_POR;
    mix_r_in_gain.mask = mix_l_in_gain.mask = 0x0F;
    mix_r_in_gain.width = mix_l_in_gain.width = 4;

    dac_r_gain.min = dac_l_gain.min = DAC_PGA_MIN;
    dac_r_gain.max = dac_l_gain.max = DAC_PGA_MAX;
    dac_r_gain.step = dac_l_gain.step = DIGITAL_PGA_STEP;
    dac_r_gain.por = dac_l_gain.por = DIGITAL_PGA_POR;
    dac_r_gain.mask = dac_l_gain.mask = 0x7F;
    dac_r_gain.width = dac_l_gain.width = 7;

    hp_r_gain.min = hp_l_gain.min = HP_PGA_MIN;
    hp_r_gain.max = hp_l_gain.max = HP_PGA_MAX;
    hp_r_gain.step = hp_l_gain.step = OUT_PGA_STEP;
    hp_r_gain.por = hp_l_gain.por = HP_PGA_POR;
    hp_r_gain.mask = hp_l_gain.mask = 0x3F;
    hp_r_gain.width = hp_l_gain.width = 6;

    spk_gain.min = SPK_PGA_MIN;
    spk_gain.max = SPK_PGA_MAX;
    spk_gain.step = OUT_PGA_STEP;
    spk_gain.por = SPK_PGA_POR;
    spk_gain.mask = 0x3F;
    spk_gain.width = 6;

    mixin_l.dmic = mixin_r.dmic = 0;
    mixin_l.mixin = mixin_r.mixin = 0;
    mixin_l.mic1 = mixin_r.mic1 = 0;
    mixin_l.mic2 = mixin_r.mic2 = 0;
    mixin_l.aux = mixin_r.aux = 0;

    mixout_l.mixinv1 = mixout_r.mixinv1 = 0;
    mixout_l.mixinv2 = mixout_r.mixinv2 = 0;
    mixout_l.auxinv = mixout_r.auxinv = 0;
    mixout_l.dac = mixout_r.dac = 0;
    mixout_l.mixin1 = mixout_r.mixin1 = 0;
    mixout_l.mixin2 = mixout_r.mixin2 = 0;
    mixout_l.aux = mixout_r.aux = 0;
}

void DA7212::power(bool on_off) {
    device_all_pwr = on_off;
    form_cmd(power_control);
}

void DA7212::input_select(int input) {
    mixin_l.dmic = mixin_r.dmic = 0;
    mixin_l.mixin = mixin_r.mixin = 0;
    mixin_l.mic1 = mixin_r.mic1 = 0;
    mixin_l.mic2 = mixin_r.mic2 = 0;
    mixin_l.aux = mixin_r.aux = 0;
    switch (input) {
        case DA7212_NO_IN:
            break;
        case DA7212_LINE:
            mixin_l.aux = mixin_r.aux = 1;
            break;
        case DA7212_MIC:
            mixin_l.mic1 = mixin_r.mic1 = 1;
            mixin_l.mic2 = mixin_r.mic2 = 1;
            break;
        default:
            mixin_l.aux = mixin_r.aux = 1;
            break;
    }
    i2c_register_write(REG_MIXIN_L_CTRL, set_input(mixin_l));
    i2c_register_write(REG_MIXIN_R_CTRL, set_input(mixin_r));
    REG_SYSTEM_MODES_INPUT;
    ADC_source_old = ADC_source;
}

void DA7212::headphone_volume(int volume) {
    i2c_register_write(REG_HP_L_GAIN, set_volume(hp_l_gain, volume));
    i2c_register_write(REG_HP_R_GAIN, set_volume(hp_r_gain, volume));
}

void DA7212::linein_volume(int volume) {
    i2c_register_write(REG_AUX_L_GAIN, set_volume(aux_l_gain, volume));
    i2c_register_write(REG_AUX_R_GAIN, set_volume(aux_l_gain, volume));
}

void DA7212::microphone_boost(int mic_boost) {
    // mic_boost = mic_boost & 0x07;
    i2c_register_write(REG_MIC_1_GAIN, set_volume(mic1_gain, mic_boost));
}

void DA7212::input_mute(bool mute) {
    uint8_t mask = 0;
    uint8_t read = 0;
    if (mute) {
        mask = DA721X_MUTE_EN;
    } else {
        mask = 0;
    }
    if (ADC_source == DA7212_MIC) {
        // uint8_t read = 0;
        // uint8_t mask = 0;
        // mic_mute = mute;
        // form_cmd(path_analog);
        //         case DA721X_MIC1:
        read = i2c_register_read(REG_MIC1_CTRL);
        read &= DA721X_MUTE_DIS;
        read |= mask;
        // i2c_register_write(REG_MIC1_CTRL, read);
        // i2c_reg_update_bits(REG_MIC1_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
        // i2c_reg_update_bits(REG_MIC2_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
        // break;
    } else {  // DA7212_LINE
        read = i2c_register_read(REG_AUX_L_CTRL);
        read &= DA721X_MUTE_DIS;
        read |= mask;
        i2c_register_write(REG_AUX_L_CTRL, read);
        read = i2c_register_read(REG_AUX_R_CTRL);
        read &= DA721X_MUTE_DIS;
        read |= mask;
        i2c_register_write(REG_AUX_R_CTRL, read);
        // LineIn_mute_left = mute;
        // LineIn_mute_right = mute;
        // form_cmd(line_in_vol_left);
        // form_cmd(line_in_vol_right);
    }
}

void DA7212::output_mute(bool mute) {
    // out_mute = mute;
    // form_cmd(path_digital);
    // case DA721X_DAC:
    //     i2c_reg_update_bits(REG_DAC_L_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     i2c_reg_update_bits(REG_DAC_R_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     break;
    // case DA721X_HP:
    //     i2c_reg_update_bits(REG_HP_L_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     i2c_reg_update_bits(REG_HP_R_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     break;
    // case DA721X_SPEAKER:
    //     i2c_reg_update_bits(REG_LINE_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     break;
    // default:
    //     break;
}

void DA7212::input_power(bool on_off) {
    device_adc_pwr = on_off;

    if (ADC_source == DA7212_MIC) {
        device_mic_pwr = on_off;
        device_lni_pwr = false;
    } else {
        device_mic_pwr = false;
        device_lni_pwr = on_off;
    }

    form_cmd(power_control);
}

void DA7212::output_power(bool on_off) {
    device_dac_pwr = on_off;
    device_out_pwr = on_off;
    // case DA721X_DAC:
    //     i2c_reg_update_bits(REG_DAC_L_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     i2c_reg_update_bits(REG_DAC_R_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     break;
    // case DA721X_HP:
    //     i2c_reg_update_bits(REG_HP_L_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     i2c_reg_update_bits(REG_HP_R_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     break;
    // case DA721X_SPEAKER:
    //     i2c_reg_update_bits(REG_LINE_CTRL, DA721X_MUTE_EN, (mute ? DA721X_MUTE_EN : 0));
    //     break;
    // default:
    //     break;
    //
    //     form_cmd(power_control);
}

void DA7212::wordsize(int words) {
    // 0x29 DAI_CTRL| DAI_EN[7](0)| DAI_OE[6](0) |DAI_TDM_MODE_EN[5](0)|
    // DAI_MONO_MODE_EN[4](0) | DAI_WORD_LENGTH[3..2](10)| DAI_FORMAT[1..0](00)
    // |
    uint8_t temp = 0;
    switch (words) {
        case 16:
            temp = 0;
            break;
        case 20:
            temp = 1;
            break;
        case 24:
            temp = 2;
            break;
        case 32:
            temp = 3;
            break;
    }
    i2c_register_write(REG_DAI_CTRL, temp);
    // form_cmd(interface_format);
}

void DA7212::master(bool master) {
    // 0x28 DAI_CLK_MODE| DAI_CLK_EN[7](0)| Reserved[6..4](000)|
    // DAI_WCLK_POL[3](0)| DAI_CLK_POL[2](0)| DAI_BCLKS_PER_WCLK[1..0](01)|
    uint8_t temp = 0x01;
    if (master) {
        temp |= 0x80;
    } else {
        temp |= 0x00;
    }
    i2c_register_write(REG_DAI_CLK_MODE, temp);
    // device_master = master;
    // form_cmd(interface_format);
}

void DA7212::frequency(int freq) {
    // Sample_rate = freq;
    // ADC_rate = freq;
    // DAC_rate = freq;
    uint8_t temp = SR32k;
    uint8_t mixedrate = 0;
    switch (freq) {
        // case 96000:
        //     temp = SR96k;
        //     break;
        // case 88200:
        //     temp = SR88k;
        //     break;
        case 48000:
            temp = SR48k;
            mixedrate = 1;
            break;
        case 44100:
            temp = SR44k;
            break;
        case 32000:
            temp = SR32k;
            break;
        case 24000:
            temp = SR24k;
            break;
        case 22050:
            temp = SR22k;
            break;
        case 16000:
            temp = SR16k;
            break;
        case 12000:
            temp = SR12k;
            break;
        case 11025:
            temp = SR11k;
            break;
        case 8000:
            temp = SR8k;
            break;
        default:
            temp = SR32k;
            break;
    }
    i2c_register_write(REG_SR, temp);
    i2c_register_write(REG_MIXED_SAMPLE_MODE, mixedrate);
}

// void DA7212::input_highpass(bool enabled) {
//     ADC_highpass_enable = enabled;
//     form_cmd(path_digital);
// }

void DA7212::output_softmute(bool enabled) {
    // out_mute = enabled;

    // form_cmd(path_digital);
    i2c_register_write(REG_DAC_FILTERS5, (enabled ? 0x80 : 0x00));  // SOFT MUTE ON! for DAC
}

void DA7212::interface_switch(bool on_off) {
    // 0x29 DAI_CTRL| DAI_EN[7]| DAI_OE[6] |DAI_TDM_MODE_EN[5]|
    // DAI_MONO_MODE_EN[4]| DAI_WORD_LENGTH[3..2]| DAI_FORMAT[1..0]|
    i2c_register_write(REG_DAI_CTRL, (on_off ? 0x80 : 0x00));  // DAI_EN[7]
    // device_interface_active = on_off;
    // form_cmd(interface_activation);
}

// void DA7212::sidetone(float sidetone_vol) {
//     sdt_vol = sidetone_vol;
//     form_cmd(path_analog);
// }

// void DA7212::deemphasis(char code) {
//     de_emph_code = code & 0x03;
//     form_cmd(path_digital);
// }

void DA7212::reset() {
    i2c_register_write(REG_CIF_CTRL, 0x80);
    // form_cmd(reset_reg);
}

void DA7212::start() { interface_switch(true); }

// void DA7212::bypass(bool enable) {
//     bypass_ = enable;
//     form_cmd(path_analog);
// }

void DA7212::stop() { interface_switch(false); }

void DA7212::command(reg_address add, uint16_t cmd) {
    char temp[2];
    temp[0] = (char(add) << 1) | ((cmd >> 6) & 0x01);
    temp[1] = (cmd & 0xFF);
    i2c.write((address << 1), temp, 2);
}

void DA7212::i2c_register_write(DA7212Registers reg, uint8_t command) {
    char temp[2];
    temp[0] = (char)reg;
    temp[1] = (char)command;
    i2c.write((address | 0), (const char*)temp, 2);
}

uint8_t DA7212::i2c_register_read(DA7212Registers reg) {
    char temp = (char)reg;
    i2c.write((address | 0), (const char*)&temp, 1, true);  // will do repeated start
    i2c.read((address | 1), &temp, 1);
    return temp;
}

void DA7212::form_cmd(reg_address add) {
    uint16_t cmd = 0;
    int temp = 0;
    bool mute;
    switch (add) {
        // case line_in_vol_left:
        //     temp = int(LineIn_vol_left * 32) - 1;
        //     mute = LineIn_mute_left;
        //
        //     if (temp < 0) {
        //         temp = 0;
        //         mute = true;
        //     }
        //     cmd = temp & 0x1F;
        //     cmd |= mute << 7;
        //     break;
        // case line_in_vol_right:
        //     temp = int(LineIn_vol_right * 32) - 1;
        //     mute = LineIn_mute_right;
        //     if (temp < 0) {
        //         temp = 0;
        //         mute = true;
        //     }
        //     cmd = temp & 0x1F;
        //     cmd |= mute << 7;
        //     break;
        //
        // case headphone_vol_left:
        //     temp = int(hp_vol_left * 80) + 47;
        //     cmd = DA7212_HP_VOL_DF_MASK;
        //     cmd |= temp & 0x7F;
        //     break;
        // case headphone_vol_right:
        //     temp = int(hp_vol_right * 80) + 47;
        //     cmd = DA7212_HP_VOL_DF_MASK;
        //     cmd |= temp & 0x7F;
        //     break;
        //
        // case path_analog:
        //     temp = int(sdt_vol * 5);
        //     char vol_code = 0;
        //     switch (temp) {
        //         case 5:
        //             vol_code = 0x0C;
        //             break;
        //         case 0:
        //             vol_code = 0x00;
        //             break;
        //         default:
        //             vol_code = ((0x04 - temp) & 0x07) | 0x08;
        //             break;
        //     }
        //     cmd = vol_code << 5;
        //     cmd |= 1 << 4;
        //     cmd |= bypass_ << 3;
        //     cmd |= ADC_source << 2;
        //     cmd |= mic_mute << 1;
        //     cmd |= mic_boost_;
        //     break;
        //
        // case path_digital:
        //     cmd |= out_mute << 3;
        //     cmd |= ((de_emph_code & 0x3) << 1);
        //     cmd |= ADC_highpass_enable;
        //     break;

        case power_control:
            cmd |= !device_all_pwr << 7;
            cmd |= !device_clk_pwr << 6;
            cmd |= !device_osc_pwr << 5;
            cmd |= !device_out_pwr << 4;
            cmd |= !device_dac_pwr << 3;
            cmd |= !device_adc_pwr << 2;
            cmd |= !device_mic_pwr << 1;
            cmd |= !device_lni_pwr << 0;
            break;

        // case interface_format:
        //     cmd |= device_master << 6;
        //     cmd |= device_lrswap << 5;
        //     cmd |= device_lrws << 4;
        //     temp = 0;
        //     switch (device_bitlength) {
        //         case 16:
        //             temp = 0;
        //             break;
        //         case 20:
        //             temp = 1;
        //             break;
        //         case 24:
        //             temp = 2;
        //             break;
        //         case 32:
        //             temp = 3;
        //             break;
        //     }
        //     cmd |= (temp & 0x03) << 2;
        //     cmd |= (device_data_form & 0x03);
        //     break;
        //
        // case sample_rate:
        //     temp = gen_samplerate();
        //     cmd = device_usb_mode;
        //     cmd |= (temp & 0x03) << 1;
        //     cmd |= device_clk_in_div << 6;
        //     cmd |= device_clk_out_div << 7;
        //     break;

        case interface_activation:
            cmd = device_interface_active;
            break;

        case reset_reg:
            cmd = 0;
            break;

        case all:
            for (int i = line_in_vol_left; i <= reset_reg; i++) {
                form_cmd((reg_address)i);
            }
            break;
    }
    if (add != all) command(add, cmd);
}

void DA7212::defaulter() {
    hp_vol_left = hp_l_gain.por;
    hp_vol_right = hp_r_gain.por;
    LineIn_vol_left = aux_l_gain.por;
    LineIn_vol_right = aux_r_gain.por;
    // sdt_vol           = DA7212_DF_sdt_vol;
    bypass_ = Default_bypass_;

    ADC_source = Default_ADC_source;
    ADC_source_old = Default_ADC_source;

    mic_mute = Default_mic_mute;
    LineIn_mute_left = Default_LineIn_mute_left;
    LineIn_mute_right = Default_LineIn_mute_right;

    mic_boost = Default_mic_boost;
    out_mute = Default_out_mute;
    de_emph_code = Default_de_emph_code;
    ADC_highpass_enable = Default_ADC_highpass_enable;

    // device_all_pwr  = Default_device_all_pwr;
    // device_clk_pwr  = Default_device_clk_pwr;
    // device_osc_pwr  = Default_device_osc_pwr;
    device_out_pwr = Default_device_out_pwr;
    device_dac_pwr = Default_device_dac_pwr;
    device_adc_pwr = Default_device_dac_pwr;
    device_mic_pwr = Default_device_mic_pwr;
    device_lni_pwr = Default_device_lni_pwr;

    device_master = Default_device_master;
    device_lrswap = Default_device_lrswap;
    device_lrws = Default_device_lrws;
    device_bitlength = Default_device_bitlength;

    Sample_rate = Default_Sample_rate;
    // ADC_rate  = Default_ADC_rate;
    // DAC_rate  = Default_DAC_rate;

    device_interface_active = Default_device_interface_active;
}

// char DA7212::gen_samplerate() {
//     uint8_t temp = DA721X_SR_32000;
//     uint8_t mixedrate = 0;
//     switch (Sample_rate) {
//         case 96000:
//             temp = DA721X_SR_96000;
//             break;
//         case 88200:
//             temp = DA721X_SR_88200;
//             break;
//         case 48000:
//             temp = DA721X_SR_48000;
//             mixedrate = 1;
//             break;
//         case 44100:
//             temp = DA721X_SR_44100;
//             break;
//         case 32000:
//             temp = DA721X_SR_32000;
//             break;
//         case 24000:
//             temp = DA721X_SR_24000;
//             break;
//         case 22050:
//             temp = DA721X_SR_22050;
//             break;
//         case 16000:
//             temp = DA721X_SR_16000;
//             break;
//         case 12000:
//             temp = DA721X_SR_12000;
//             break;
//         case 11025:
//             temp = DA721X_SR_11025;
//             break;
//         case 8000:
//             temp = DA721X_SR_8000;
//             break;
//         default:
//             temp = DA721X_SR_32000;
//             break;
//     }
//     i2c_register_write(REG_SR, temp);
//     i2c_register_write(REG_MIXED_SAMPLE_MODE, mixedrate);
//     return temp;
// }

// int da7212_I2Splayback(void)  // Init path from Digital Audio Interface to
//     HP
//     or SPK {
//     printf("Start da7212_I2Splayback!\n");
//     i2c_register_write(REG_GAIN_RAMP_CTRL, 0x00);  // Set Ramp rate to
//     default 0x92
//         GAIN_RAMP_CTRL
//         i2c_register_write(REG_REFERENCES, 0x08);  // Enable master bias
//         wait_ms(40);                               // 40ms delay
//         i2c_register_write(REG_LDO_CTRL, 0x80);    // Enable Digital LDO
//
//         i2c_register_write(REG_DAI_CTRL, 0xC0);         // Enable AIF 16bit I2S
//         mode i2c_register_write(REG_PC_COUNT, 0x02);    // Set PC sync to resync
//         i2c_register_write(REG_DAC_FILTERS5, 0x80);     // SOFT MUTE ON!
//         i2c_register_write(REG_DIG_ROUTING_DAC, 0x32);  // DACR and DACL source.
//         //    i2c_register_write(REG_MIXOUT_L_SELECT, 0x08); //MIXOUT_L input
//         from DACL i2c_register_write(REG_MIXOUT_R_SELECT, 0x08);  // MIXOUT_R input from
//         DACR
// #ifdef DA7212_HP
//             i2c_register_write(REG_CP_CTRL, 0xF1);                     // CP_CTRL - Signal
//         size + Boost i2c_register_write(REG_CP_VOL_THRESHOLD1, 0x36);  // CP_VOL_THRESHOLD
//         i2c_register_write(REG_CP_DELAY, 0xA5);                        // CP_DELAY
//         i2c_register_write(REG_HP_L_GAIN, 0x39);                       // Set HP_L gain to 0dB
//         i2c_register_write(REG_HP_R_GAIN, 0x39);                       // Set HP_R gain to 0dB
//
//         i2c_register_write(REG_DAC_L_CTRL, 0x80);     // DAC_L
//         i2c_register_write(REG_DAC_R_CTRL, 0x80);     // DAC_R
//         i2c_register_write(REG_HP_L_CTRL, 0xA8);      // HP_L
//         i2c_register_write(REG_HP_R_CTRL, 0xA8);      // HP_R
//         i2c_register_write(REG_MIXOUT_L_CTRL, 0x88);  // MIXOUT_L
//         i2c_register_write(REG_MIXOUT_R_CTRL, 0x88);  // MIXOUT_R
//
//         // if use DA7212_SYSTEM_MODE
//         i2c_register_write(REG_SYSTEM_MODES_OUTPUT, 0xF1);  // Enable DAC, HP
// #else                                                       // DA7212_SPK
//             i2c_register_write(REG_LINE_GAIN, 0x30);  // SPEAKER GAIN 0dB
//
//         //    i2c_register_write(REG_DAC_L_CTRL     , 0x80); // DAC_L
//         i2c_register_write(REG_DAC_R_CTRL, 0x80);  // DAC_R
//         i2c_register_write(REG_LINE_CTRL, 0xA8);   // SPEAKER
//         //    i2c_register_write(REG_MIXOUT_L_CTRL  , 0x88); // MIXOUT_L
//         i2c_register_write(REG_MIXOUT_R_CTRL, 0x88);  // MIXOUT_R
//
//         // if use DA7212_SYSTEM_MODE
//         // i2c_register_write(REG_SYSTEM_MODES_OUTPUT, 0xC9); // SET Enable DAC,
//         SPEAKER(LINE)
// #endif
//         i2c_register_write(REG_TONE_GEN_CFG2, 0x60);  // Tone generater.
//         da7212_set_clock(CONFIG_SAMPLE_RATE);
//         da7212_set_dai_fmt(CPU_I2S_MASTER);
//         wait_ms(40);                                 // 40ms delay
//         i2c_register_write(REG_DAC_FILTERS5, 0x00);  // SOFT MUTE OFF!
//         return 0;
// }

// int da7212_duplex(void)  // Init path from MIC1 to Digital Audio Interface
// {
//     // DEBUG_PRINTF("---ticks : %d\n",ticks);
//     i2c_register_write(REG_REFERENCES, 0x08);      // Enable master bias
//     i2c_register_write(REG_GAIN_RAMP_CTRL, 0x02);  // Set Ramp rate to 1S
//     wait_ms(40);                                   // 40ms delay
//     i2c_register_write(REG_LDO_CTRL, 0x80);        // Enable Digital LDO
//     da7212_set_clock(CONFIG_SAMPLE_RATE);
//     da7212_set_dai_fmt(CPU_I2S_MASTER);
//     i2c_register_write(REG_DAI_CTRL, 0xC0);                                     // Enable AIF 16bit I2S
//     mode i2c_register_write(REG_PC_COUNT, 0x02);                                // Set PC sync to resync
//     i2c_register_write(REG_DIG_ROUTING_DAI, 0x10);                              // DIG_ROUTING_DAI, from
//     DAI L / R i2c_register_write(REG_MICBIAS_CTRL, 0x0A);                       // Enable MICBIAS1
//     i2c_register_write(REG_MIC_1_CTRL, 0x84);                                   // Set Mic1 to be single
//     ended from MIC1_SE connector i2c_register_write(REG_MIXIN_L_SELECT, 0x02);  // MIXIN_L input from
//     MIC1 i2c_register_write(REG_MIXIN_R_SELECT, 0x04);                          // MIXIN_R input from
//     MIC1 i2c_register_write(REG_GAIN_RAMP_CTRL, 0x00);                          // Set Ramp rate to
//     default
//
//         i2c_register_write
//         (REG_MIXIN_L_CTRL, 0x88);                         // Enable MIXIN Left
//         i2c_register_write(REG_MIXIN_R_CTRL, 0x88);       // Enable MIXIN Left
//         i2c_register_write(REG_ADC_L_CTRL, 0xA0);         // Enable ADC Left and
//         unmute.i2c_register_write(REG_ADC_R_CTRL, 0xA0);  // Enable ADC Right
//
//         da7212_I2Splayback();
//         // DEBUG_PRINTF("---ticks : %d\n",ticks);
//
//         printf("Start da7212_duplex!\n");
//         return 0;
// }

// // int da7212_record(void)    //Init path from MIC1 to Digital Audio Interface
// {
//     i2c_register_write(REG_REFERENCES, 0x08);      // Enable master bias
//     i2c_register_write(REG_GAIN_RAMP_CTRL, 0x02);  // Set Ramp rate to 1S
//     wait_ms(40);                                   // 40ms delay
//     i2c_register_write(REG_LDO_CTRL, 0x80);        // Enable Digital LDO
//     da7212_set_clock(CONFIG_SAMPLE_RATE);
//     da7212_set_dai_fmt(CPU_I2S_MASTER);
//     i2c_register_write(REG_DAI_CTRL, 0xC0);                                     // Enable AIF 16bit I2S
//     mode i2c_register_write(REG_PC_COUNT, 0x02);                                // Set PC sync to resync
//     i2c_register_write(REG_DIG_ROUTING_DAI, 0x10);                              // DIG_ROUTING_DAI, from
//     DAI L / R i2c_register_write(REG_MICBIAS_CTRL, 0x0A);                       // Enable MICBIAS1
//     i2c_register_write(REG_MIC_1_CTRL, 0x84);                                   // Set Mic1 to be single
//     ended from MIC1_SE connector i2c_register_write(REG_MIXIN_L_SELECT, 0x02);  // MIXIN_L input from
//     MIC1 i2c_register_write(REG_MIXIN_R_SELECT, 0x04);                          // MIXIN_R input from
//     MIC1 i2c_register_write(REG_GAIN_RAMP_CTRL, 0x00);                          // Set Ramp rate to
//     default
// #ifndef DA7212_SYSTEM_MODE
//         i2c_register_write
//         (REG_MIXIN_L_CTRL, 0x88);                         // Enable MIXIN Left
//         i2c_register_write(REG_MIXIN_R_CTRL, 0x88);       // Enable MIXIN Left
//         i2c_register_write(REG_ADC_L_CTRL, 0xA0);         // Enable ADC Left and
//         unmute.i2c_register_write(REG_ADC_R_CTRL, 0xA0);  // Enable ADC Right
// #else
//         i2c_register_write
//         (REG_SYSTEM_MODES_INPUT, 0xF4);  // Enable MIXIN,ADC.
// #endif
//         return 0;
// }

/* Supported PLL input frequencies are 5MHz - 54MHz. */

// int da7212_set_dai_pll(uint8_t sampling_rate) {
//     uint8_t pll_ctrl, indiv_bits, indiv;
//     uint8_t pll_frac_top, pll_frac_bot, pll_integer;
//     uint8_t use_pll = 0x80;  // ENABLE PLL
//     /* Reset PLL configuration */
//     i2c_register_write(REG_DAC_FILTERS5, 0x80);  // SOFT MUTE ON!
//
//     i2c_register_write(REG_PLL_CTRL, 0);  //  system clock is MCLK; SRM
//     disabled;
//     32 kHz mode disabled;
//     squarer at the MCLK disabled; input clock
//     range for the PLL= 2 - 10 MHz
//
//     if (i2c_register_write(REG_SR, sampling_rate) < 0) {
//         printf("codec_set_sample_rate: error in write reg .\n");
//         return -1;
//     }
//
//     pll_ctrl = 0;
//     /* Workout input divider based on MCLK rate */
//     if (DA721X_MCLK < 5000000) {
//         goto pll_err;
//     } else if (DA721X_MCLK <= 10000000) {
//         indiv_bits = DA721X_PLL_INDIV_5_10_MHZ;
//         indiv = DA721X_PLL_INDIV_5_10_MHZ_VAL;
//     } else if (DA721X_MCLK <= 20000000) {
//         indiv_bits = DA721X_PLL_INDIV_10_20_MHZ;
//         indiv = DA721X_PLL_INDIV_10_20_MHZ_VAL;
//     } else if (DA721X_MCLK <= 40000000) {
//         indiv_bits = DA721X_PLL_INDIV_20_40_MHZ;
//         indiv = DA721X_PLL_INDIV_20_40_MHZ_VAL;
//     } else if (DA721X_MCLK <= 54000000) {
//         indiv_bits = DA721X_PLL_INDIV_40_54_MHZ;
//         indiv = DA721X_PLL_INDIV_40_54_MHZ_VAL;
//     } else if (DA721X_MCLK == 12288000) {
//         use_pll = 0;
//     } else {
//         goto pll_err;
//     }
//     pll_ctrl |= indiv_bits;
//
//     /*
//      * If Codec is slave and SRM enabled,
//      */
//     if (CPU_I2S_MASTER && DA721X_SRM_EN) {
//         pll_ctrl |= DA721X_PLL_SRM_EN;
//         pll_frac_top = 0x0D;
//         pll_frac_bot = 0xFA;
//         pll_integer = 0x1F;
//     } else {
//         switch (sampling_rate) {
//             case DA721X_SR_8000:
//             case DA721X_SR_12000:
//             case DA721X_SR_16000:
//             case DA721X_SR_24000:
//             case DA721X_SR_32000:
//             case DA721X_SR_48000:
//             case DA721X_SR_96000:
//                 pll_frac_top = 0x18;
//                 pll_frac_bot = 0x93;
//                 pll_integer = 0x20;
//                 break;
//             case DA721X_SR_11025:
//             case DA721X_SR_22050:
//             case DA721X_SR_44100:
//             case DA721X_SR_88200:
//                 pll_frac_top = 0x03;
//                 pll_frac_bot = 0x61;
//                 pll_integer = 0x1E;
//                 break;
//             default:
//                 printf("codec_set_sample_rate: invalid parameter. \n");
//                 return -1;
//         }
//     }
//
//     /* Write PLL dividers */
//     i2c_register_write(REG_PLL_FRAC_TOP, pll_frac_top);
//     i2c_register_write(REG_PLL_FRAC_BOT, pll_frac_bot);
//     i2c_register_write(REG_PLL_INTEGER, pll_integer);
//
// /* Enable MCLK squarer if required */
// #if (DA721X_MCLK_SQR_EN)
//     pll_ctrl |= DA721X_PLL_MCLK_SQR_EN;
// #endif
//     /* Enable PLL */
//     pll_ctrl |= use_pll;
//     i2c_register_write(REG_PLL_CTRL, pll_ctrl);
//     wait_us(10 * 1000);                          // 10ms delay
//     i2c_register_write(REG_DAC_FILTERS5, 0x00);  // SOFT MUTE OFF!
//     return 0;
//
// pll_err:
//     printf("Unsupported PLL input frequency %d\n", DA721X_MCLK);
//     return -1;
// }
