 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
#include "Protocols.h"
#include "ILI9327.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// put in constructor
//#define LCDSIZE_X       240 // display X pixels, TFTs are usually portrait view
//#define LCDSIZE_Y       400  // display Y pixels



ILI9327::ILI9327(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
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
ILI9327::ILI9327(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
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
ILI9327::ILI9327(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name , unsigned int LCDSIZE_X , unsigned  int LCDSIZE_Y )
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
// read display ID
void ILI9327::identify()
{
    // ILI9327 custom cmd
    tftID=rd_reg_data32(0xEF);
    mipistd=true;
}
// reset and init the lcd controller
void ILI9327::init()
{
    wr_cmd8(0x11);
    wait_ms(150);
    
    wr_cmd8(0xF3); //Set EQ 
    wr_data8(0x08);    
    wr_data8(0x20);    
    wr_data8(0x20);    
    wr_data8(0x08);  
    
    wr_cmd8(0xE7);  
    wr_data8(0x60);   //OPON 
    
    wr_cmd8(0xD1);  // Set VCOM 
    wr_data8(0x00);    
    wr_data8(0x5D);    
    wr_data8(0x15);  
    
    wr_cmd8(0xD0); //Set power related setting 
    wr_data8(0x07);    
    wr_data8(0x02);   //VGH:15V,VGL:-7.16V (BOE LCD: VGH:12~18V,VGL:-6~10V)  
    wr_data8(0x8B);    
    wr_data8(0x03);    
    wr_data8(0xD4); 

    wr_cmd8(0x3A); // Set_pixel_format
    wr_data8(0x55); //0x55:16bit/pixel,65k;0x66:18bit/pixel,262k;

    wr_cmd8(0x36); //Set_address_mode
    wr_data8(0x38);
    
    wr_cmd8(0x20); //Exit_invert_mode 
    
    wr_cmd8(0xC1);   //Set Normal/Partial mode display timing  
    wr_data8(0x10);    
    wr_data8(0x1A);    
    wr_data8(0x02);    
    wr_data8(0x02); 
    
    wr_cmd8(0xC0);  //Set display related setting  
    wr_data8(0x10);    
    wr_data8(0x31);    
    wr_data8(0x00);    
    wr_data8(0x00);    
    wr_data8(0x01);    
    wr_data8(0x02);  
    
    wr_cmd8(0xC4);  //Set waveform timing 
    wr_data8(0x01);    
    
    wr_cmd8(0xC5);  //Set oscillator 
    wr_data8(0x04);   //72Hz
    wr_data8(0x01); 
    
    wr_cmd8(0xD2);  //Set power for normal mode 
    wr_data8(0x01);    
    wr_data8(0x44);   
    
    wr_cmd8(0xC8);  //Set gamma
    wr_data8(0x00);    
    wr_data8(0x77);    
    wr_data8(0x77);    
    wr_data8(0x00);    
    wr_data8(0x04);    
    wr_data8(0x00);    
    wr_data8(0x00);    
    wr_data8(0x00);    
    wr_data8(0x77);    
    wr_data8(0x00);    
    wr_data8(0x00);    
    wr_data8(0x08);    
    wr_data8(0x00);    
    wr_data8(0x00);    
    wr_data8(0x00); 
    
    wr_cmd8(0xCA);  //Set DGC LUT
    wr_data8(0x00); 
    
    wr_cmd8(0xEA);  //3-Gamma Function Control
    wr_data8(0x80);  // enable
    
    wr_cmd8(0x29);  // Set_display_on  
    wait_ms(150);
    
}