 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
#include "Protocols.h"
#include "ILI9486.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// put in constructor
//#define LCDSIZE_X       320 // display X pixels, TFTs are usually portrait view
//#define LCDSIZE_Y       480  // display Y pixels



ILI9486::ILI9486(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);
    identify(); // will collect tftID and set mipistd flag
    init();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
    set_orientation(0);
 //   FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. ILI9486 does not, at least in par mode
    cls();
    locate(0,0); 
}
ILI9486::ILI9486(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, Hz, mosi, miso, sclk, CS, reset, DC, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset(); //TFT class forwards to Protocol class
    BusEnable(true); //TFT class forwards to Protocol class
    identify(); // will collect tftID and set mipistd flag
    init(); // per display custom init cmd sequence, implemented here
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
    set_orientation(0); //TFT class does for MIPI standard and some ILIxxx
 //   FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. ILI9486 does not, at least in par mode
    cls();
    locate(0,0); 
}
// reset and init the lcd controller
void ILI9486::init()
{
    /* Start Initial Sequence ----------------------------------------------------*/
    
    wr_cmd8(0xF1);
    wr_data8(0x36);
    wr_data8(0x04);
    wr_data8(0x00);
    wr_data8(0x3C);
    wr_data8(0x0F);
    wr_data8(0x8F);


    wr_cmd8(0xF2);
    wr_data8(0x18);
    wr_data8(0xA3);
    wr_data8(0x12);
    wr_data8(0x02);
    wr_data8(0xb2);
    wr_data8(0x12);
    wr_data8(0xFF);
    wr_data8(0x10);
    wr_data8(0x00);

    wr_cmd8(0xF8);
    wr_data8(0x21);
    wr_data8(0x04);

    wr_cmd8(0xF9);
    wr_data8(0x00);
    wr_data8(0x08);  

    wr_cmd8(0xC0);
    wr_data8(0x0f); //13
    wr_data8(0x0f); //10

    wr_cmd8(0xC1);
    wr_data8(0x42); //43

    wr_cmd8(0xC2);
    wr_data8(0x22);

    wr_cmd8(0xC5);
    wr_data8(0x01); //00
    wr_data8(0x29); //4D
    wr_data8(0x80);

    wr_cmd8(0xB6);
    wr_data8(0x00);
    wr_data8(0x02); //42
    wr_data8(0x3b);

    wr_cmd8(0xB1);
    wr_data8(0xB0); //C0
    wr_data8(0x11);

    wr_cmd8(0xB4);
    wr_data8(0x02); //01

    wr_cmd8(0xE0);
    wr_data8(0x0F);
    wr_data8(0x18);
    wr_data8(0x15);
    wr_data8(0x09);
    wr_data8(0x0B);
    wr_data8(0x04);
    wr_data8(0x49);
    wr_data8(0x64);
    wr_data8(0x3D);
    wr_data8(0x08);
    wr_data8(0x15);
    wr_data8(0x06);
    wr_data8(0x12);
    wr_data8(0x07);
    wr_data8(0x00);

    wr_cmd8(0xE1);
    wr_data8(0x0F);
    wr_data8(0x38);
    wr_data8(0x35);
    wr_data8(0x0a);
    wr_data8(0x0c);
    wr_data8(0x03);
    wr_data8(0x4A);
    wr_data8(0x42);
    wr_data8(0x36);
    wr_data8(0x04);
    wr_data8(0x0F);
    wr_data8(0x03);
    wr_data8(0x1F);
    wr_data8(0x1B);
    wr_data8(0x00);

    wr_cmd8(0x20);                     // display inversion OFF
  
    wr_cmd8(0x36);      // MEMORY_ACCESS_CONTROL (orientation stuff)
    wr_data8(0x48);
     
    wr_cmd8(0x3A);      // COLMOD_PIXEL_FORMAT_SET
    wr_data8(0x55);     // 16 bit pixel 

    wr_cmd8(0x13); // Nomal Displaymode
    
    wr_cmd8(0x11);                     // sleep out
    wait_ms(150);
     
    wr_cmd8(0x29);                     // display on
    wait_ms(150);
}