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

#ifndef DA7212_H_
#define DA7212_H_
#include "mbed.h"  //NOLINT

#define DA7212_CS_HIGH (true)
#define DA7212_CS_LOW false

#define DA7212_ON true
#define DA7212_OFF false

#define DA7212_MUTE true
#define DA7212_UNMUTE false

#define DA7212_MASTER true
#define DA7212_SLAVE false

#define DA7212_LINE 0
#define DA7212_MIC 1
#define DA7212_NO_IN -1

#define DA7212_DE_EMPH_DISABLED 0
#define DA7212_DE_EMPH_32KHZ 1
#define DA7212_DE_EMPH_44KHZ 2
#define DA7212_DE_EMPH_48KHZ 3

#define DA721X_MCLK 12288000
#define DA721X_MUTE_EN (0x40)
#define DA721X_MUTE_DIS (0xBF)
#define DA721X_POWER_EN (0x80)

/** A class to control the I2C part of the DA7212
 * - 12.288MHz MCLK sent from FRDM
 * - BitClk aka. BCLK for DA7212
 * - WordSelect is WCLK for DA7212
 */
class DA7212 {
   public:
    /** Create an instance of the DA7212 class
     *
     * @param i2c_sda The SDA pin of the I2C
     * @param i2c_scl The SCL pin of the I2C
     */
    DA7212(PinName i2c_sda, PinName i2c_scl);

    /** Create an instance of the DA7212 class
     *
     * @param i2c_sda The SDA pin of the I2C
     * @param i2c_scl The SCL pin of the I2C
     * @param i2c_addr 8-bit I2C slave address
     */
    DA7212(PinName i2c_sda, PinName i2c_scl, int i2c_addr);

    /* Create an instance of the DA7212 class
     *
     * @param i2c_sda The SDA pin of the I2C
     * @param i2c_scl The SCL pin of the I2C
     * @param cs_level The level of the CS pin on the DA7212
     */
    // DA7212(PinName i2c_sda, PinName i2c_scl, bool cs_level);

    /** Control the power of the device
     *
     * @param on_off The power state
     */
    void power(bool on_off);

    /** Control the input source of the device
     *
     * @param input Select the source of the input of the device: DA7212_LINE,
     * DA7212_MIC, DA7212_NO_IN
     */
    void input_select(int input);

    /** Set the headphone volume
     * 1x(-57~+6):64
     * 0b111001 = 57 = 0 dB
     * @param h_volume The desired headphone volume: -57->+6
     */
    void headphone_volume(int h_volume);

    /** Set the line in pre-amp volume
     * 150 x (-27~+36):64
     * 0b110101 = 53 = 0 dB (default)
     * @param LineIn_volume The desired line in volume: -27->+36
     */
    void linein_volume(int LineIn_volume);

    /** Turn on/off the microphone pre-amp boost
     * 600 x (-1~+6):8
     * 0b001 = 1 = 0 dB (default)
     * @param mic_boost Boost gain -1->+6
     */
    void microphone_boost(int mic_boost);

    /** Mute the input
     *
     * @param mute Mute on/off
     */
    void input_mute(bool mute);

    /** Mute the output
     *
     * @param mute Mute on/off
     */
    void output_mute(bool mute);

    /** Turn on/off the input stage
     *
     * @param on_off Input stage on(true)/off(false)
     */
    void input_power(bool on_off);

    /** Turn on/off the output stage
     *
     * @param on_off Output stage on(true)/off(false)
     */
    void output_power(bool on_off);

    /** Select the word size
     *
     * @param words 16/20/24/32 bits
     */
    void wordsize(int words);

    /** Select interface mode: Master or Slave
     *
     * @param master Interface mode: master(true)/slave
     */
    void master(bool master);

    /** Select the sample rate
     *
     * @param freq Frequency: 96/48/32/8 kHz
     */
    void frequency(int freq);

    /** Enable the input highpass filter
     *
     * @param enabled Input highpass filter enabled
     */
    void input_highpass(bool enabled);

    /** Enable the output soft mute
     *
     * @param enabled Output soft mute enabled
     */
    void output_softmute(bool enabled);

    /** Turn on and off the I2S
     *
     * @param on_off Switch the I2S interface on(true)/off(false)
     */
    void interface_switch(bool on_off);

    /** Reset the device and settings
     *
     */
    void reset();

    /** Set the microphone sidetone volume
     *
     * @param sidetone_volume The volume of the sidetone: 0->1
     * ;does not exist in 7212?
     */
    void sidetone(float sidetone_vol);

    /** Set the analog bypass
     *
     * @param bypass_en Enable the bypass: enabled(true)
     */
    void bypass(bool bypass_en);

    /** Set the deemphasis frequency
     *
     * @param code The deemphasis code: DA7212_DE_EMPH_DISABLED,
     * DA7212_DE_EMPH_32KHZ, DA7212_DE_EMPH_44KHZ, DA7212_DE_EMPH_48KHZ
     */
    void deemphasis(char code);

    /** Enable the input highpass filter
     *
     * @param enable Enable the input highpass filter enabled(true)
     */
    void adc_highpass(bool enable);

    /** Start the device sending/recieving etc
    */
    void start();

    /** Stop the device sending/recieving etc
    */
    void stop();
   
    enum DA7212Registers {
        REG_STATUS1 = (0x02),
        REG_PLL_STATUS,
        REG_AUX_L_GAIN_STATUS,
        REG_AUX_R_GAIN_STATUS,
        REG_MIC_1_GAIN_STATUS,
        REG_MIC_2_GAIN_STATUS,
        REG_MIXIN_L_GAIN_STATUS,
        REG_MIXIN_R_GAIN_STATUS,
        REG_ADC_L_GAIN_STATUS,
        REG_ADC_R_GAIN_STATUS,
        REG_DAC_L_GAIN_STATUS,
        REG_DAC_R_GAIN_STATUS,
        REG_HP_L_GAIN_STATUS,
        REG_HP_R_GAIN_STATUS,
        REG_LINE_GAIN_STATUS,
        REG_CIF_CTRL = (0x1D),
        REG_DIG_ROUTING_DAI = (0x21),
        REG_SR,
        REG_REFERENCES,
        REG_PLL_FRAC_TOP,
        REG_PLL_FRAC_BOT,
        REG_PLL_INTEGER,
        REG_PLL_CTRL,
        REG_DAI_CLK_MODE,
        REG_DAI_CTRL,
        REG_DIG_ROUTING_DAC,
        REG_ALC_CTRL1,
        REG_AUX_L_GAIN = (0x30),
        REG_AUX_R_GAIN,
        REG_MIXIN_L_SELECT,
        REG_MIXIN_R_SELECT,
        REG_MIXIN_L_GAIN,
        REG_MIXIN_R_GAIN,
        REG_ADC_L_GAIN,
        REG_ADC_R_GAIN,
        REG_ADC_FILTERS1,
        REG_MIC_1_GAIN,
        REG_MIC_2_GAIN,
        REG_DAC_FILTERS5 = (0x40),
        REG_DAC_FILTERS2,
        REG_DAC_FILTERS3,
        REG_DAC_FILTERS4,
        REG_DAC_FILTERS1,
        REG_DAC_L_GAIN,
        REG_DAC_R_GAIN,
        REG_CP_CTRL,
        REG_HP_L_GAIN,
        REG_HP_R_GAIN,
        REG_LINE_GAIN,
        REG_MIXOUT_L_SELECT,
        REG_MIXOUT_R_SELECT,
        REG_SYSTEM_MODES_INPUT = (0x50),
        REG_SYSTEM_MODES_OUTPUT,
        REG_AUX_L_CTRL = (0x60),
        REG_AUX_R_CTRL,
        REG_MICBIAS_CTRL,
        REG_MIC1_CTRL,
        REG_MIC2_CTRL,
        REG_MIXIN_L_CTRL,
        REG_MIXIN_R_CTRL,
        REG_ADC_L_CTRL,
        REG_ADC_R_CTRL,
        REG_DAC_L_CTRL,
        REG_DAC_R_CTRL,
        REG_HP_L_CTRL,
        REG_HP_R_CTRL,
        REG_LINE_CTRL,
        REG_MIXOUT_L_CTRL,
        REG_MIXOUT_R_CTRL,
        REG_MIXED_SAMPLE_MODE = (0x84),
        REG_LDO_CTRL = (0x90),
        REG_GAIN_RAMP_CTRL = (0x92),
        REG_MIC_CONFIG,
        REG_PC_COUNT,
        REG_CP_VOL_THRESHOLD1,
        REG_CP_DELAY,
        REG_CP_DETECTOR,
        REG_DAI_OFFSET,
        REG_DIG_CTRL,
        REG_ALC_CTRL2,
        REG_ALC_CTRL3,
        REG_ALC_NOISE,
        REG_ALC_TARGET_MIN,
        REG_ALC_TARGET_MAX,
        REG_ALC_GAIN_LIMITS,
        REG_ALC_ANA_GAIN_LIMITS,
        REG_ALC_ANTICLIP_CTRL,
        REG_ALC_ANTICLIP_LEVEL,
        REG_ALC_OFFSET_AUTO_M_L,
        REG_ALC_OFFSET_AUTO_U_L,
        REG_ALC_OFFSET_MAN_M_L,
        REG_ALC_OFFSET_MAN_U_L,
        REG_ALC_OFFSET_AUTO_M_R,
        REG_ALC_OFFSET_AUTO_U_R,
        REG_ALC_OFFSET_MAN_M_R,
        REG_ALC_OFFSET_MAN_U_R,
        REG_ALC_CIC_OP_LVL_CTRL,
        REG_ALC_CIC_OP_LVL_DATA,
        REG_DAC_NG_SETUP_TIME,
        REG_DAC_NG_OFF_THRESHOLD,
        REG_DAC_NG_ON_THRESHOLD,
        REG_DAC_NG_CTRL,
        REG_TONE_GEN_CFG1 = (0xB4),
        REG_TONE_GEN_CFG2,
        REG_TONE_GEN_CYCLES,
        REG_TONE_GEN_FREQ1_L,
        REG_TONE_GEN_FREQ1_U,
        REG_TONE_GEN_FREQ2_L,
        REG_TONE_GEN_FREQ2_U,
        REG_TONE_GEN_ON_PER,
        REG_TONE_GEN_OFF_PER,
        REG_SYSTEM_STATUS = (0xE0),
        REG_SYSTEM_ACTIVE = (0xFD)
    };
    // Definitions for each Gains
// private:
    /**
    600 x (-1~+6):8
    0b001 = 1 = 0 dB (default)
    */
    enum DA7212MicGain {
        MIC_PGA_MIN = (-600),  // -6dB
        MIC_PGA_MAX = (3600),  // 36dB
        MIC_PGA_STEP = 600,
        MIC_PGA_POR = 600,
    };

    /**
    150 x (-27~+36):64
    0b110101 = 53 = 0 dB (default)
    */
    enum DA7212AuxGain {
        AUX_PGA_MIN = (-5400),  // -54dB
        AUX_PGA_MAX = (1500),   // 15dB
        AUX_PGA_STEP = 150,
        AUX_PGA_POR = 5400,
    };

    /**
    150 x (-3~12):16
    0b0011 = 3 = 0 dB
    */
    enum DA7212MixerInGain {
        MIXIN_PGA_MIN = (-450),  // -4.5 dB
        MIXIN_PGA_MAX = (1800),  // 18dB
        MIXIN_PGA_STEP = 150,
        MIXIN_PGA_POR = 450,
    };

    /**
    75 x (-111~16):128
    0b1101111 = 111(decimal!) = 0 dB (default)
    */
    enum DA7212DigitalGain {
        ADC_PGA_MIN = (-8325),  // -78 ~ 12dB
        ADC_PGA_MAX = (1200),
        DAC_PGA_MIN = (-8325),  // -78 ~ 12dB
        DAC_PGA_MAX = (1200),
        DIGITAL_PGA_STEP = 75,
        DIGITAL_PGA_POR = 8325,

        DAC_MIN_VOL = (DAC_PGA_MIN / 100),
        DAC_MAX_VOL = (DAC_PGA_MAX / 100),
        ADC_MIN_VOL = (ADC_PGA_MIN / 100),
        ADC_MAX_VOL = (ADC_PGA_MAX / 100),
    };

    /**
    1x(-57~+6):64
    0b111001 = 57 = 0 dB
    */
    enum DA7212HeadPhoneGain {
        HP_PGA_MIN = (-5700),  // -57 ~ 6dB
        HP_PGA_MAX = (600),
        OUT_PGA_STEP = 100,
        HP_PGA_POR = 5700,
    };

    /**
    1x(-48~+15):64
    0b110000 = 48 = 0 dB
    */
    enum DA7212SpeakerGain {
        SPK_PGA_MIN = (-4800),  // -48 ~ 15dB
        SPK_PGA_MAX = (1500),
        SPK_PGA_POR = 4800,
    };

    /* REG_SR = r0x22 */
    enum DA7212SampleRate {
        SR8k = (0x1 << 0),
        SR11k = (0x2 << 0),
        SR12k = (0x3 << 0),
        SR16k = (0x5 << 0),
        SR22k = (0x6 << 0),
        SR24k = (0x7 << 0),
        SR32k = (0x9 << 0),
        SR44k = (0xA << 0),
        SR48k = (0xB << 0),
        SR88k = (0xE << 0),
        SR96k = (0xF << 0),
    };

    enum reg_address {
        line_in_vol_left = 0x00,
        line_in_vol_right = 0x01,
        headphone_vol_left = 0x02,
        headphone_vol_right = 0x03,
        path_analog = 0x04,
        path_digital = 0x05,
        power_control = 0x06,
        interface_format = 0x07,
        sample_rate = 0x08,
        interface_activation = 0x09,
        reset_reg = 0x0A,
        all = 0xFF
    };

    enum DA7212_defaults {
        Default_bypass_ = 0,
        Default_ADC_source = DA7212_LINE,
        Default_mic_mute = DA7212_UNMUTE,
        Default_LineIn_mute_left = 0,
        Default_LineIn_mute_right = 0,
        Default_mic_boost = 0,
        Default_out_mute = DA7212_UNMUTE,

        Default_de_emph_code = 0x00,
        Default_ADC_highpass_enable = 0,

        // Default_device_all_pwr = 1,
        // Default_device_clk_pwr = 1,
        // Default_device_osc_pwr = 1,
        Default_device_out_pwr = 1,
        Default_device_dac_pwr = 1,
        Default_device_adc_pwr = 1,
        Default_device_mic_pwr = 0,
        Default_device_lni_pwr = 1,

        Default_device_master = 0,
        Default_device_lrswap = 0,
        Default_device_lrws = 0,
        Default_device_bitlength = 32,

        Default_Sample_rate = 32000,
        // Default_ADC_rate = 32000,
        // Default_DAC_rate = 32000,

        Default_device_interface_active = 0
    };

    typedef struct gain_t {
        int min;
        int max;
        int step;
        int por;
        int mask;
        int width;
    };

    gain_t mic1_gain, mic2_gain;
    gain_t aux_l_gain, aux_r_gain;
    gain_t mix_l_in_gain, mix_r_in_gain;
    gain_t dac_l_gain, dac_r_gain;
    gain_t hp_l_gain, hp_r_gain;
    gain_t spk_gain;

    typedef struct mixin_t {
        int dmic;
        int mixin;
        int mic1;
        int mic2;
        int aux;
    };

    mixin_t mixin_l, mixin_r;

    typedef struct mixout_t {
        int mixinv1;
        int mixinv2;
        int auxinv;
        int dac;
        int mixin1;
        int mixin2;
        int aux;
    };
    mixout_t mixout_l, mixout_r;

    I2C i2c;
    uint8_t address;
    void command(reg_address add, uint16_t byte);
    void i2c_register_write(DA7212Registers reg, uint8_t command);
    uint8_t i2c_register_read(DA7212Registers reg);
    void form_cmd(reg_address add);
    void defaulter();
    void init();
    char gen_samplerate();
    int reset_gain(gain_t gain) { return (gain.por / gain.step); }
    int set_volume(gain_t channel, int gain) {
        int vol = 0;
        vol = (gain * 100 - channel.min) / channel.step;
        vol &= channel.mask;
        return vol;
    }

    int set_input(mixin_t channel) {
        int mixin_ = 0;
        mixin_ = mixin_l.dmic << 7 | mixin_l.mixin << 3 | mixin_l.mic2 << 2 | mixin_l.mic1 << 1 | mixin_l.aux;
        return mixin_;
    }

    // I2S i2s_tx(I2S_TRANSMIT, p5, p6 , p7);
    // I2S i2s_rx(I2S_RECIEVE , p8, p29, p30);

    float hp_vol_left, hp_vol_right;
    float LineIn_vol_left, LineIn_vol_right;
    float sdt_vol;
    bool LineIn_mute_left, LineIn_mute_right;
    bool bypass_;
    bool ADC_source;
    bool ADC_source_old;
    bool mic_mute;
    uint8_t mic_boost;
    // bool mic_boost_;
    bool out_mute;
    char de_emph_code;
    bool ADC_highpass_enable;

    bool device_all_pwr;
    bool device_clk_pwr;
    bool device_osc_pwr;
    bool device_out_pwr;
    bool device_dac_pwr;
    bool device_adc_pwr;
    bool device_mic_pwr;
    bool device_lni_pwr;

    bool device_master;
    bool device_lrswap;
    bool device_lrws;
    char device_bitlength;
    static const char device_data_form = 0x02;

    // int ADC_rate;
    // int DAC_rate;
    int Sample_rate;
    static const bool device_usb_mode = false;
    static const bool device_clk_in_div = false;
    static const bool device_clk_out_div = false;
    bool device_interface_active;
};

#endif  // DA7212_H_
