 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
#include "Protocols.h"
#include "IST3020.h"

/*this is a quite standard config, similar to ST7565, except bigger screen and diff resistor ratio value*/

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
#define IC_X_SEGS    256 // IST3020 SEG has range 0-255 (255-0 if ADC=1), check your datasheet, important for the orientation
#define IC_Y_COMS    64  // IST3020 COM has range 0-63 (63-0 if SHL=1), check your datasheet, important for the orientation
// put in constructor
//#define LCDSIZE_X       192 // display X pixels, IST3020 is advertised as 224x65 but display size could be smaller
//#define LCDSIZE_Y       64  // display Y pixels, the 65th is for accessing "icons"



IST3020::IST3020(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : LCD(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, IC_X_SEGS, IC_Y_COMS, name)
{
    hw_reset();
    BusEnable(true);
    init();
    cls();
    set_orientation(1);
    locate(0,0);
}
IST3020::IST3020(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : LCD(displayproto, Hz, mosi, miso, sclk, CS, reset, DC, LCDSIZE_X, LCDSIZE_Y, IC_X_SEGS, IC_Y_COMS, name)
{
    hw_reset();
    BusEnable(true);
    init();
    cls();
    set_orientation(1);
    locate(0,0);
}
// reset and init the lcd controller
// init sequence is manufacturer specific
void IST3020::init()
{
    /* Start Initial Sequence ----------------------------------------------------*/
    
    wr_cmd8(0xE2);   //  sw reset
    wait_ms(10);
    
    wr_cmd8(0xAE);   //  display off
    wr_cmd8(0xAB);   //  built-in OSC on
    wr_cmd8(0xA2);   //  bias voltage (1/9)
  //  wr_cmd8(0xA3);   //  bias voltage (1/7)

    wr_cmd8(0xA0);   // ADC select seg0-seg223
    //wr_cmd8(0xA1);   // ADC select seg223-seg0
    wr_cmd8(0xC8);   // SHL select com63-com0
    //wr_cmd8(0xC0);   // SHL select com0-com63

    wr_cmd8(0x2C);   //  Internal Voltage Converter ON
    wait_ms(10);
    wr_cmd8(0x2E);   //  Internal Voltage Regulator ON
    wait_ms(10);
    wr_cmd8(0x2F);   //  Internal Voltage Follower ON
    wait_ms(10);
    wr_cmd8(0x20);   //  Regulor_Resistor_Select resistor ratio 20-27 20=4.5(default) 27=8.0, 0.5 steps
    set_contrast(46);
    //wr_cmd8(0x81);   //  set contrast (reference voltage register set)
    //wr_cmd8(0x20);   //  contrast 00-3F default 20
    
    wr_cmd8(0xA4);   //  LCD display ram (EntireDisplayOn disable)
    //wr_cmd8(0x70);   //  External Capacitors Discharge function enable (should be enabled by default)
    //wr_cmd8(0x77);   //  External Capacitors Discharge function disable
    wr_cmd8(0x40);   // start line = 0
    wr_cmd8(0xA6);     // display normal (1 = illuminated)
    wr_cmd8(0xAF);     // display ON 

}