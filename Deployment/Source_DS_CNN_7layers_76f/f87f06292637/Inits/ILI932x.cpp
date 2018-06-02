 /* mbed UniGraphic library - Device specific class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
 
#include "Protocols.h"
#include "ILI932x.h"

//////////////////////////////////////////////////////////////////////////////////
// display settings ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


ILI932x::ILI932x(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT932x(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);  //set CS low, will stay low untill manually set high with BusEnable(false);
    identify(); // will collect tftID
    if(tftID==0x9325) init9325();
    else if(tftID==0x9320) init9320();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
    set_orientation(0);
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. 
    cls();
    locate(0,0); 
}
ILI932x::ILI932x(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name , unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT932x(displayproto, buspins, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);  //set CS low, will stay low untill manually set high with BusEnable(false);
    identify(); // will collect tftID
    if(tftID==0x9325) init9325();
    else if(tftID==0x9320) init9320();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
    set_orientation(0);
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. 
    cls();
    locate(0,0); 
}
ILI932x::ILI932x(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT932x(displayproto, Hz, mosi, miso, sclk, CS, reset, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset(); //TFT class forwards to Protocol class
    BusEnable(true); //set CS low, TFT932x class will toggle CS every transfer
    identify(); // will collect tftID
    if(tftID==0x9325) init9325();
    else if(tftID==0x9320) init9320();
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
    set_orientation(0); 
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. 
    cls();
    locate(0,0); 
}
// reset and init the lcd controller

void ILI932x::init9325()
{
    /* Example for ILI9325 ----------------------------------------------------*/

 flipped=FLIP_NONE; // FLIP_NONE, FLIP_X, FLIP_Y, FLIP_X|FLIP_Y
 
 reg_write(0x0001,0x0100); 
 reg_write(0x0002,0x0700); 
 reg_write(0x0003,0x1030); 
 reg_write(0x0004,0x0000); 
 reg_write(0x0008,0x0207);  
 reg_write(0x0009,0x0000);
 reg_write(0x000A,0x0000); 
 reg_write(0x000C,0x0000); 
 reg_write(0x000D,0x0000);
 reg_write(0x000F,0x0000);
//power on sequence VGHVGL
 reg_write(0x0010,0x0000);   
 reg_write(0x0011,0x0007);  
 reg_write(0x0012,0x0000);  
 reg_write(0x0013,0x0000); 
 reg_write(0x0007,0x0001);
wait_ms(200);
//vgh 
 reg_write(0x0010,0x1290);   
 reg_write(0x0011,0x0227);
wait_ms(50);
 //vregiout 
 reg_write(0x0012,0x001d); //0x001b
 wait_ms(50);
 //vom amplitude
 reg_write(0x0013,0x1500);
 wait_ms(50); 
 //vom H
 reg_write(0x0029,0x0018); 
 reg_write(0x002B,0x000D); 
wait_ms(50);
//gamma
 reg_write(0x0030,0x0004);
 reg_write(0x0031,0x0307);
 reg_write(0x0032,0x0002);// 0006
 reg_write(0x0035,0x0206);
 reg_write(0x0036,0x0408);
 reg_write(0x0037,0x0507); 
 reg_write(0x0038,0x0204);//0200
 reg_write(0x0039,0x0707); 
 reg_write(0x003C,0x0405);// 0504
 reg_write(0x003D,0x0F02); 
 //ram
 reg_write(0x0050,0x0000); 
 reg_write(0x0051,0x00EF);
 reg_write(0x0052,0x0000); 
 reg_write(0x0053,0x013F);  
 reg_write(0x0060,0xA700); 
 reg_write(0x0061,0x0001); 
 reg_write(0x006A,0x0000); 
 //
 reg_write(0x0080,0x0000); 
 reg_write(0x0081,0x0000); 
 reg_write(0x0082,0x0000); 
 reg_write(0x0083,0x0000); 
 reg_write(0x0084,0x0000); 
 reg_write(0x0085,0x0000); 
 //
 reg_write(0x0090,0x0010); 
 reg_write(0x0092,0x0600); 
 reg_write(0x0093,0x0003); 
 reg_write(0x0095,0x0110); 
 reg_write(0x0097,0x0000); 
 reg_write(0x0098,0x0000);
 
 reg_write(0x0007,0x0133); // display on

}
void ILI932x::init9320()
{
    /* Example for ILI9320 ----------------------------------------------------*/

 flipped=FLIP_X; // FLIP_NONE, FLIP_X, FLIP_Y, FLIP_X|FLIP_Y
 
 reg_write(0x0001,0x0100); 
 reg_write(0x0002,0x0700); 
 reg_write(0x0003,0x1030); 
 reg_write(0x0004,0x0000); 
 reg_write(0x0008,0x0202);  
 reg_write(0x0009,0x0000);
 reg_write(0x000A,0x0000); 
 reg_write(0x000C,0x0000); 
 reg_write(0x000D,0x0000);
 reg_write(0x000F,0x0000);
//power on sequence
 reg_write(0x0010,0x0000);   
 reg_write(0x0011,0x0007);  
 reg_write(0x0012,0x0000);  
 reg_write(0x0013,0x0000); 
 reg_write(0x0007,0x0001);
wait_ms(200);

 reg_write(0x0010,0x10C0);   
 reg_write(0x0011,0x0007);
wait_ms(50);

 reg_write(0x0012,0x0110);
 wait_ms(50);

 reg_write(0x0013,0x0b00);
 wait_ms(50); 

 reg_write(0x0029,0x0000); 
 reg_write(0x002B,0x4010); // bit 14???
wait_ms(50);
//gamma
/*
 reg_write(0x0030,0x0004);
 reg_write(0x0031,0x0307);
 reg_write(0x0032,0x0002);// 0006
 reg_write(0x0035,0x0206);
 reg_write(0x0036,0x0408);
 reg_write(0x0037,0x0507); 
 reg_write(0x0038,0x0204);//0200
 reg_write(0x0039,0x0707); 
 reg_write(0x003C,0x0405);// 0504
 reg_write(0x003D,0x0F02);
 */
 //ram
 reg_write(0x0050,0x0000); 
 reg_write(0x0051,0x00EF);
 reg_write(0x0052,0x0000); 
 reg_write(0x0053,0x013F);  
 reg_write(0x0060,0x2700); 
 reg_write(0x0061,0x0001); 
 reg_write(0x006A,0x0000); 
 //
 reg_write(0x0080,0x0000); 
 reg_write(0x0081,0x0000); 
 reg_write(0x0082,0x0000); 
 reg_write(0x0083,0x0000); 
 reg_write(0x0084,0x0000); 
 reg_write(0x0085,0x0000); 
 //
 reg_write(0x0090,0x0000); 
 reg_write(0x0092,0x0000); 
 reg_write(0x0093,0x0001); 
 reg_write(0x0095,0x0110); 
 reg_write(0x0097,0x0000); 
 reg_write(0x0098,0x0000);
 
 reg_write(0x0007,0x0133); // display on

}