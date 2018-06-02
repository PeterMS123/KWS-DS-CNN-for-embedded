/* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Peter Drescher
 * Released under the MIT License: http://mbed.org/license/mit
 */
  
#include "Protocols.h"
#include "SSD1306.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


#define IC_X_SEGS    128 // UC1608 SEG has range 0-239 (239-0 if MX=1), check your datasheet, important for the orientation
#define IC_Y_COMS    64  // UC1608 COM has range 0-127 (127-0 if MY=1), check your datasheet, important for the orientation

#define SSD1306_SETCONTRAST         0x81
#define SSD1306_DISPLAYALLON_RESUME 0xA4
#define SSD1306_DISPLAYALLON        0xA5
#define SSD1306_NORMALDISPLAY       0xA6
#define SSD1306_INVERTDISPLAY       0xA7
#define SSD1306_DISPLAYOFF          0xAE
#define SSD1306_DISPLAYON           0xAF
#define SSD1306_SETDISPLAYOFFSET    0xD3
#define SSD1306_SETCOMPINS          0xDA
#define SSD1306_SETVCOMDETECT       0xDB
#define SSD1306_SETDISPLAYCLOCKDIV  0xD5
#define SSD1306_SETPRECHARGE        0xD9
#define SSD1306_SETMULTIPLEX        0xA8
#define SSD1306_SETLOWCOLUMN        0x00
#define SSD1306_SETHIGHCOLUMN       0x10
#define SSD1306_SETSTARTLINE        0x40
#define SSD1306_MEMORYMODE          0x20
#define SSD1306_COMSCANINC          0xC0
#define SSD1306_COMSCANDEC          0xC8
#define SSD1306_SEGREMAP            0xA0
#define SSD1306_CHARGEPUMP          0x8D

SSD1306::SSD1306(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : LCD(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, IC_X_SEGS, IC_Y_COMS, name)
{
    hw_reset();
    BusEnable(true);
    init();
    cls();
    set_orientation(1);
    locate(0,0);
}
SSD1306::SSD1306(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : LCD(displayproto, Hz, mosi, miso, sclk, CS, reset, DC, LCDSIZE_X, LCDSIZE_Y, IC_X_SEGS, IC_Y_COMS, name)
{
    hw_reset();
    BusEnable(true);
    init();
    cls();
    set_orientation(1);
    locate(0,0);
}

SSD1306::SSD1306(proto_t displayproto, int Hz, int address, PinName sda, PinName scl, const char* name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : LCD(displayproto, Hz, address, sda, scl, LCDSIZE_X, LCDSIZE_Y, IC_X_SEGS, IC_Y_COMS, name)
{
    init();
    cls();
    set_orientation(1);
    locate(0,0);
}


// reset and init the lcd controller
void SSD1306::init()
{
    /* Start Initial Sequence ----------------------------------------------------*/
    
  //  wr_cmd8(0xE2);   //  sw reset
    wait_ms(15);
    
    wr_cmd8(SSD1306_DISPLAYOFF);    // no problem in SPI_16 for single byte cmds
    wr_cmd16((SSD1306_SETDISPLAYCLOCKDIV<<8)|0x80);   // wr_cmd16 for multibyte cmds issue in SPI16 mode
  //  wr_cmd8(0x80); // in SPI_16 it would become 0xE380 and will break things up
    wr_cmd16((SSD1306_SETMULTIPLEX<<8)|63);
 //   wr_cmd8(63);
    
    wr_cmd16((SSD1306_SETDISPLAYOFFSET<<8)|0x00);
 //   wr_cmd8(0x0);
    
    wr_cmd8(SSD1306_SETSTARTLINE | 0x0);            // line #0

    wr_cmd16((SSD1306_CHARGEPUMP<<8)|0x14);
  //  wr_cmd8(0x14);                         // 0x10 

    wr_cmd16((SSD1306_MEMORYMODE<<8)|0x00);
 //   wr_cmd8(0x00);                                  // 0x0 act like ks0108

    wr_cmd8(SSD1306_SEGREMAP ); //| 0x1);

    wr_cmd8(SSD1306_COMSCANDEC);

    wr_cmd16((SSD1306_SETCOMPINS<<8)|0x12);
  //  wr_cmd8(0x12);                           //        LCDSIZE_Y == 32 ? 0x02 : 0x12);        

    wr_cmd16((SSD1306_SETCONTRAST<<8)|0xCF);
 //   wr_cmd8(0xCF);                              //  _rawHeight == 32 ? 0x8F : ((vccstate == SSD1306_EXTERNALVCC) ? 0x9F : 0xCF) );

    wr_cmd16((SSD1306_SETPRECHARGE<<8)|0xF1);
 //   wr_cmd8(0xF1);                               // : 0x22);

    wr_cmd16((SSD1306_SETVCOMDETECT<<8)|0x40);
 //   wr_cmd8(0x40);

    wr_cmd8(SSD1306_DISPLAYALLON_RESUME);

    //wr_cmd8(SSD1306_NORMALDISPLAY);
    wr_cmd8(SSD1306_INVERTDISPLAY);
    
    wr_cmd8(SSD1306_DISPLAYON);
}

////////////////////////////////////////////////////////////////////
// functions that overrides the standard ones implemented in LCD.cpp
////////////////////////////////////////////////////////////////////

void SSD1306::mirrorXY(mirror_t mode)
{
    switch (mode)
    {
        case(NONE):
            wr_cmd16(0xA0C0); 
            break;
        case(X):
            wr_cmd16(0xA1C0);
            break;
        case(Y):
            wr_cmd16(0xA0C8);
            break;
        case(XY):
            wr_cmd16(0xA1C8);
            break;
    }
}

void SSD1306::set_contrast(int o)
{
    contrast = o;
  
    wr_cmd16(0x8100|(o&0xFF));
}

////////////////////////////////////////////////////////////////////
// functions that overrides the standard ones implemented in LCD.cpp
////////////////////////////////////////////////////////////////////


const uint8_t scroll_speed[8]={3,2,1,6,0,5,4,7};

////////////////////////////////////////////////////////////////////
// functions addon  to LCD.cpp
////////////////////////////////////////////////////////////////////
void SSD1306::horizontal_scroll(int l_r,int s_page,int e_page,int speed){
    wr_cmd8(0x2E);                      // deactivate scroll before change
    if(l_r == 1){
        wr_cmd16(0x2700);               // horizontal scroll left
    }
    else {
        wr_cmd16(0x2600);
    }
    wr_cmd16((s_page & 0x07)<<8 | (scroll_speed[speed & 0x07]));
    wr_cmd16((e_page & 0x07)<<8 );
    wr_cmd16(0xFF2F);
}

void SSD1306::horiz_vert_scroll(int l_r,int s_page,int e_page,int v_off,int speed){
    wr_cmd8(0x2E);                      // deactivate scroll before change
    if(l_r == 1){
            wr_cmd16(0x2A00);               // horizontal scroll left
        }
        else {
            wr_cmd16(0x2900);
        }
        wr_cmd16((s_page & 0x07)<<8 | (scroll_speed[speed & 0x07]));
        wr_cmd16((e_page & 0x07)<<8 | (v_off & 0x3F) );
        wr_cmd8(0x2F);

}

void SSD1306::end_scroll(){
    wr_cmd8(0x2E);
}
