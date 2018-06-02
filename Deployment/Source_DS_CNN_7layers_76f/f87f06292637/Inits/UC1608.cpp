 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
#include "Protocols.h"
#include "UC1608.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
#define IC_X_SEGS    240 // UC1608 SEG has range 0-239 (239-0 if MX=1), check your datasheet, important for the orientation
#define IC_Y_COMS    128  // UC1608 COM has range 0-127 (127-0 if MY=1), check your datasheet, important for the orientation
//#define LCDSIZE_X       240 // display X pixels
//#define LCDSIZE_Y       120  // display Y pixels, UC1608 is advertised as 240x128 but display size could be smaller



UC1608::UC1608(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : LCD(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, IC_X_SEGS, IC_Y_COMS, name)
{
    hw_reset();
    BusEnable(true);
    init();
    cls();
    set_orientation(1);
    locate(0,0);
}
UC1608::UC1608(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
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
void UC1608::init()
{
    /* Start Initial Sequence ----------------------------------------------------*/
    
  //  wr_cmd8(0xE2);   //  sw reset
    wait_ms(15);
    
    wr_cmd8(0x27);   // Multiplex rate :128   set temperature consenpation 0%
    wr_cmd8(0xEA);   //set bias:1/12bias

    wr_cmd8(0xC4);   // set mirror MX=1,MY=0 (controller->display SEGs wiring inverted)
   // wr_cmd8(0xA0);   // ADC select seg0-seg223
    //wr_cmd8(0xA1);   // ADC select seg223-seg0
   // wr_cmd8(0xC8);   // SHL select com63-com0
    //wr_cmd8(0xC0);   // SHL select com0-com63

    wr_cmd8(0x2F);   //  //Power Control:internal, LCD capacitance 60nf-90nf
    wait_ms(10);
    
   // wr_cmd8(0x81);//Set Gain and Potentiometer
  //  wr_cmd8(0x40|26);//Set Gain and Potentiometer  xx xxxxxx
    set_contrast(26);
    
    wr_cmd8(0x88);   //disable colum/page address wraparound
    wr_cmd8(0xA4);   //  LCD display ram (EntireDisplayOn disable)
    wr_cmd8(0x40);   // start line = 0
    wr_cmd8(0xA6);     // display normal (1 = illuminated)
    wr_cmd8(0xAF);     // display ON 

}
////////////////////////////////////////////////////////////////////
// functions that overrides the standard ones implemented in LCD.cpp
////////////////////////////////////////////////////////////////////
void UC1608::mirrorXY(mirror_t mode)
{
    switch (mode)
    {
        case(NONE):
            wr_cmd8(0xC4); // this is in real X mirror command, but my display have SEGs wired inverted, so assume this is the default no-x-mirror
            break;
        case(X):
            wr_cmd8(0xC0);
            break;
        case(Y):
            wr_cmd8(0xCC);
            break;
        case(XY):
            wr_cmd8(0xC8);
            break;
    }
}
void UC1608::set_contrast(int o)
{
    contrast = o;
  //  wr_cmd8(0x81);      //  set volume
  //  wr_cmd8(0x40|(o&0x3F));
    wr_cmd16(0x8140|(o&0x3F));
}
void UC1608::BusEnable(bool enable)
{
    LCD::BusEnable(!enable); // crap IC has CS not inverted (active HIGH)
}