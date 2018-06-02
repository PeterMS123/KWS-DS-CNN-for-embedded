 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
#include "Protocols.h"
#include "S6D04D1.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// put in constructor
//#define LCDSIZE_X       240 // display X pixels, TFTs are usually portrait view
//#define LCDSIZE_Y       400  // display Y pixels



S6D04D1::S6D04D1(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);
    identify(); // will collect tftID, set mipistd flag
    init();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
//    scrollbugfix=1; // when scrolling 1 line, the last line disappears, set to 1 to fix it, for ili9481 is set automatically in identify()
    set_orientation(0);
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. Give a try
    cls();
    locate(0,0); 
}
S6D04D1::S6D04D1(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, buspins, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);
    identify(); // will collect tftID, set mipistd flag
    init();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
//    scrollbugfix=1; // when scrolling 1 line, the last line disappears, set to 1 to fix it, for ili9481 is set automatically in identify()
    set_orientation(0);
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. Give a try
    cls();
    locate(0,0); 
}
S6D04D1::S6D04D1(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name , unsigned int LCDSIZE_X , unsigned  int LCDSIZE_Y )
    : TFT(displayproto, Hz, mosi, miso, sclk, CS, reset, DC, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset(); //TFT class forwards to Protocol class
    BusEnable(true); //TFT class forwards to Protocol class
    identify(); // will collect tftID and set mipistd flag
    init(); // per display custom init cmd sequence, implemented here
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
 //   scrollbugfix=1; // when scrolling 1 line, the last line disappears, set to 1 to fix it, for ili9481 is set automatically in identify()
    set_orientation(0); //TFT class does for MIPI standard and some ILIxxx
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. Give a try
    cls();
    locate(0,0); 
}
// reset and init the lcd controller
void S6D04D1::init()
{
    /**********************************************
        TFT1P CODE Initialization of Truly 
       
     ************************************************        
         Panel:3.0 240400 
         Driver IC:S6D04D1X21-BAF8
     
     ************************************************/
wr_cmd8(0xE0); 
wr_data8(0x01); 

wr_cmd8(0x11); 
wait_ms(150); 

wr_cmd8(0xF3); 
wr_data8(0x01); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x0C);//Do not set any higher VCI1 level than VCI -0.15V. 0C 0A 
wr_data8(0x03);//VGH和VGL 01 02VGH=6VCI1,VGL=-4VCI1. 
wr_data8(0x75); 
wr_data8(0x75); 
wr_data8(0x30); 

wr_cmd8(0xF4); 
wr_data8(0x4C); 
wr_data8(0x4C); 
wr_data8(0x44); 
wr_data8(0x44); 
wr_data8(0x22); 

wr_cmd8(0xF5); 
wr_data8(0x10); 
wr_data8(0x22); 
wr_data8(0x05); 
wr_data8(0xF0); 
wr_data8(0x70); 
wr_data8(0x1F); 
wait_ms(30); 

wr_cmd8(0xF3); 
wr_data8(0x03); 
wait_ms(30); 
wr_cmd8(0xF3); 
wr_data8(0x07); 
wait_ms(30); 
wr_cmd8(0xF3); 
wr_data8(0x0F); 
wait_ms(30); 
wr_cmd8(0xF3); 
wr_data8(0x1F); 
wait_ms(30); 
wr_cmd8(0xF3); 
wr_data8(0x7F); 
wait_ms(30); 


wr_cmd8(0xF7); 
wr_data8(0x80); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x05); 
wr_data8(0x0D); 
wr_data8(0x1F); 
wr_data8(0x26); 
wr_data8(0x2D); 
wr_data8(0x14); 
wr_data8(0x15); 
wr_data8(0x26); 
wr_data8(0x20); 
wr_data8(0x01); 
wr_data8(0x22); 
wr_data8(0x22); 

wr_cmd8(0xF8); 
wr_data8(0x80); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x07); 
wr_data8(0x1E); 
wr_data8(0x2A); 
wr_data8(0x32); 
wr_data8(0x10); 
wr_data8(0x16); 
wr_data8(0x36); 
wr_data8(0x3C); 
wr_data8(0x3B); 
wr_data8(0x22); 
wr_data8(0x22); 

wr_cmd8(0xF9); 
wr_data8(0x80); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x05); 
wr_data8(0x0D); 
wr_data8(0x1F); 
wr_data8(0x26); 
wr_data8(0x2D); 
wr_data8(0x14); 
wr_data8(0x15); 
wr_data8(0x26); 
wr_data8(0x20); 
wr_data8(0x01); 
wr_data8(0x22); 
wr_data8(0x22); 


wr_cmd8(0xFA); 
wr_data8(0x80); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x07); 
wr_data8(0x1E); 
wr_data8(0x2A); 
wr_data8(0x32); 
wr_data8(0x10); 
wr_data8(0x16); 
wr_data8(0x36); 
wr_data8(0x3C); 
wr_data8(0x3B); 
wr_data8(0x22); 
wr_data8(0x22); 


wr_cmd8(0xFB); 
wr_data8(0x80); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x05); 
wr_data8(0x0D); 
wr_data8(0x1F); 
wr_data8(0x26); 
wr_data8(0x2D); 
wr_data8(0x14); 
wr_data8(0x15); 
wr_data8(0x26); 
wr_data8(0x20); 
wr_data8(0x01); 
wr_data8(0x22); 
wr_data8(0x22); 

wr_cmd8(0xFC); 
wr_data8(0x80); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x07); 
wr_data8(0x1E); 
wr_data8(0x2A); 
wr_data8(0x32); 
wr_data8(0x10); 
wr_data8(0x16); 
wr_data8(0x36); 
wr_data8(0x3C); 
wr_data8(0x3B); 
wr_data8(0x22); 
wr_data8(0x22); 

//wr_cmd8(0x35); 
wr_cmd8(0x34); // tearing effect line off

wr_cmd8(0x36); 
wr_data8(0x48);//08 

wr_cmd8(0x3A); 
wr_data8(0x05); 

wr_cmd8(0xF2); 
wr_data8(0x17); 
wr_data8(0x17); 
wr_data8(0x0F); 
wr_data8(0x08); 
wr_data8(0x08); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x00); 
wr_data8(0x13); 
wr_data8(0x00); 

wr_cmd8(0xF6); 
wr_data8(0x00); 
wr_data8(0x08); 
wr_data8(0x00); 
wr_data8(0x00); 

wr_cmd8(0xFD); 
wr_data8(0x02); 
wr_data8(0x01);//240*400 
 
wait_ms(20); 
wr_cmd8(0x29); // display on
wait_ms(20); 
    
}
