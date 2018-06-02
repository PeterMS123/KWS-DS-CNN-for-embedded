 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
#include "Protocols.h"
#include "TFT_MIPI.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

// put in constructor
//#define LCDSIZE_X       320 // display X pixels, TFTs are usually portrait view
//#define LCDSIZE_Y       480  // display Y pixels



TFT_MIPI::TFT_MIPI(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);
    identify(); // will collect tftID, set mipistd flag
    init();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
//    scrollbugfix=1; // when scrolling 1 line, the last line disappears, set to 1 to fix it, for ili9481 is set automatically in identify()
    set_orientation(0);
 //   FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. Give a try
    cls();
    locate(0,0); 
}
TFT_MIPI::TFT_MIPI(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name , unsigned int LCDSIZE_X , unsigned  int LCDSIZE_Y )
    : TFT(displayproto, Hz, mosi, miso, sclk, CS, reset, DC, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset(); //TFT class forwards to Protocol class
    BusEnable(true); //TFT class forwards to Protocol class
    identify(); // will collect tftID and set mipistd flag
    init(); // per display custom init cmd sequence, implemented here
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
 //   scrollbugfix=1; // when scrolling 1 line, the last line disappears, set to 1 to fix it, for ili9481 is set automatically in identify()
    set_orientation(0); //TFT class does for MIPI standard and some ILIxxx
 //   FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. Give a try
    cls();
    locate(0,0); 
}
// reset and init the lcd controller
void TFT_MIPI::init()
{
    /* Start Initial Sequence ----------------------------------------------------*/
    
    /* Start Initial Sequence ----------------------------------------------------*/
    wr_cmd8(0xD0);  // POWER SETTING             
    wr_data8(0x07);
    wr_data8(0x42);
    wr_data8(0x18);
    
    wr_cmd8(0xD1);  //  VCOM control          
    wr_data8(0x00);
    wr_data8(0x07);  
    wr_data8(0x10);
    
    wr_cmd8(0xD2);  //   Power_Setting for Normal Mode 
    wr_data8(0x01);   // LCD power supply current
    wr_data8(0x02);  // charge pumps
    
    wr_cmd8(0xC0);  // Panel Driving Setting             
    wr_data8(0x10);    // 10 orig
    wr_data8(0x3B);   //number of lines+1 *8
    wr_data8(0x00);
    wr_data8(0x02);
    wr_data8(0x11);
    
     // C1 missing? Display_Timing_Setting for Normal Mode
    
    //renesas does not have this
   // wr_cmd8(0xC5); // Frame Rate and Inversion Control                     
   // wr_data8(0x03); // 72hz, datashet tells default 02=85hz
    
    wr_cmd8(0xC8);  // Gamma settings        
    wr_data8(0x00);
    wr_data8(0x32);
    wr_data8(0x36);
    wr_data8(0x45);
    wr_data8(0x06);
    wr_data8(0x16);
    wr_data8(0x37);
    wr_data8(0x75);
    wr_data8(0x77);
    wr_data8(0x54);
    wr_data8(0x0C);
    wr_data8(0x00);
     
    

    wr_cmd8(0x36);   // MEMORY_ACCESS_CONTROL (orientation stuff)
    wr_data8(0x0A);     // 0A as per chinese example (vertical flipped)
   
    wr_cmd8(0x3A);                     // COLMOD_PIXEL_FORMAT_SET, not present in AN
    wr_data8(0x55);                 // 16 bit pixel 
    
    wr_cmd8(0x13); // Nomal Displaymode
    
    wr_cmd8(0x11);                     // sleep out
    wait_ms(150);
     
    wr_cmd8(0x29);                     // display on
    wait_ms(150);

}