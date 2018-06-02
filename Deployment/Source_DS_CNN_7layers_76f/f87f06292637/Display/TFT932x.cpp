 /* mbed UniGraphic library - custom TFT driver class, ILI932x specific 
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 */
 
#include "platform.h"
#include "TFT932x.h"

#if DEVICE_PORTINOUT
#include "PAR8.h"
#include "PAR16.h"
#endif

//#include "mbed_debug.h"

#define SWAP(a, b)  { a ^= b; b ^= a; a ^= b; }

#if DEVICE_PORTINOUT
TFT932x::TFT932x(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const int lcdsize_x, const int lcdsize_y, const char *name)
    : GraphicsDisplay(name), screensize_X(lcdsize_x), screensize_Y(lcdsize_y)
{
    if(displayproto==PAR_8)
    {
        proto = new PAR8(port, CS, reset, DC, WR, RD);
        dummycycles=1;
    }
    else if(displayproto==PAR_16)
    {
        proto = new PAR16(port, CS, reset, DC, WR, RD);
        dummycycles=0;
    }
  //  set_orientation(0);
    foreground(White);
    background(Black);
    set_auto_up(false); //we don't have framebuffer
    usefastwindow=false;
    fastwindowready=false;
    is18bit=false;
    isBGR=false;
    flipped=0;
  //  cls();
  //  locate(0,0);
}
#endif

TFT932x::TFT932x(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const int lcdsize_x, const int lcdsize_y, const char *name)
    : GraphicsDisplay(name), screensize_X(lcdsize_x), screensize_Y(lcdsize_y)
{
    if(displayproto==BUS_8)
    {
        PinName pins[16];
        for(int i=0; i<16; i++) pins[i]=NC;
        for(int i=0; i<8; i++) pins[i]=buspins[i];
        proto = new BUS8(pins, CS, reset, DC, WR, RD);
        dummycycles=1;
    }
    else if(displayproto==BUS_16)
    {
        proto = new BUS16(buspins, CS, reset, DC, WR, RD);
        dummycycles=0;
    }
  //  set_orientation(0);
    foreground(White);
    background(Black);
    set_auto_up(false); //we don't have framebuffer
    usefastwindow=false;
    fastwindowready=false;
    is18bit=false;
    isBGR=false;
    flipped=0;
  //  cls();
  //  locate(0,0);
}
TFT932x::TFT932x(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, const int lcdsize_x, const int lcdsize_y, const char *name)
    : GraphicsDisplay(name), screensize_X(lcdsize_x), screensize_Y(lcdsize_y)
{
    if(displayproto==SPI_8)
    {
        proto = new SPI8(Hz, mosi, miso, sclk, CS, reset);
        dummycycles=4;
    }
    else if(displayproto==SPI_16)
    {
        proto = new SPI16(Hz, mosi, miso, sclk, CS, reset);
        dummycycles=2;
    }
 //   set_orientation(0);
    foreground(White);
    background(Black);
    set_auto_up(false);
    usefastwindow=false;
    fastwindowready=false;
    is18bit=false;
    isBGR=false;
    flipped=0;
  //  locate(0,0);
}
// dummy read needed before read gram
// read gram protocol function does 1 dymmy read as for MIPI standard, but ILI932x needs more and protocol specific number of cycles
// for example in spi mode, 5 dummy byte read needed, so for SPI16 2x16bit clocks done here and the 5th dummy will be handled by read gram function
void TFT932x::dummyread()
    {
        for(unsigned int i=0; i<dummycycles; i++) proto->dummyread(); 
    }
void TFT932x::reg_select(unsigned char reg, bool forread)
    {
        proto->reg_select(reg, forread); 
    }
void TFT932x::reg_write(unsigned char reg, unsigned short data)
    {
        proto->reg_write(reg, data); 
    }
unsigned short TFT932x::reg_read(unsigned char reg)
    {
        return proto->reg_read(reg); 
    }
void TFT932x::wr_gram(unsigned short data)
    {
        proto->wr_gram(data);
    }
void TFT932x::wr_gram(unsigned short data, unsigned int count)
    {
        proto->wr_gram(data, count);
    }
void TFT932x::wr_grambuf(unsigned short* data, unsigned int lenght)
    {
        proto->wr_grambuf(data, lenght);
    }
unsigned short TFT932x::rd_gram()
    {
        return proto->rd_gram(is18bit); // protocol will handle 18to16 bit conversion
        
    }
//for TFT, just send data, position counters are in hw
void TFT932x::window_pushpixel(unsigned short color)
{
    proto->wr_gram(color);
}
void TFT932x::window_pushpixel(unsigned short color, unsigned int count)
{
    proto->wr_gram(color, count);
}
void TFT932x::window_pushpixelbuf(unsigned short* color, unsigned int lenght)
    {
        proto->wr_grambuf(color, lenght);
    }
void TFT932x::hw_reset()
    {
        proto->hw_reset();
        BusEnable(true);
    }
void TFT932x::BusEnable(bool enable)
    {
        proto->BusEnable(enable);
    }
// ILI932x can't rotate in hw (swap raw<->columns) for landscape views,
// but can change the way address counter is auto incremented/decremented
void TFT932x::set_orientation(int o)
{
  //  if(orientation == o) return;
    orientation = o;
    switch (orientation)
    // BGR bit set for all modes, seems most TFT are like that, in case override set_orientation() in init
    // ORG bit set for all modes
    {
        case 0:// default, portrait view 0°
            reg_write(0x0001,((flipped&FLIP_X)==0) ? 0x0100:0x0000); // S720toS1 or S1toS720
            reg_write(0x0060,((flipped&FLIP_Y)==0) ? 0xA700:0x2700); // G320toG1 or G1toG320
            reg_write(0x03, 0x10B0);
            set_width(screensize_X);
            set_height(screensize_Y);
            break;
        case 1:// landscape view +90°
            reg_write(0x0001,((flipped&FLIP_X)==0) ? 0x0000:0x0100); // S1toS720 or S720toS1
            reg_write(0x0060,((flipped&FLIP_Y)==0) ? 0xA700:0x2700); // G320toG1 or G1toG320
            reg_write(0x03, 0x10B8); // AM=1 increase addr ctr first vertically then horizontally
            set_width(screensize_Y);
            set_height(screensize_X);
            break;
        case 2:// portrait view +180°
            reg_write(0x0001,((flipped&FLIP_X)==0) ? 0x0000:0x0100); // S1toS720 or S720toS1
            reg_write(0x0060,((flipped&FLIP_Y)==0) ? 0x2700:0xA700); // G1toG320 or G320toG1
            reg_write(0x03, 0x10B0);
            set_width(screensize_X);
            set_height(screensize_Y);
            break;
        case 3:// landscape view -90°
            reg_write(0x0001,((flipped&FLIP_X)==0) ? 0x0100:0x0000); // S720toS1 or S1toS720
            reg_write(0x0060,((flipped&FLIP_Y)==0) ? 0x2700:0xA700); // G1toG320 or G320toG1
            reg_write(0x03, 0x10B8); // AM=1 increase addr ctr first vertically then horizontally
            set_width(screensize_Y);
            set_height(screensize_X);
            break;
    }
}
void TFT932x::invert(unsigned char o)
{
    unsigned short oldreg = reg_read(0x61);
    if(o == 0) reg_write(0x61, oldreg|1); // seems most TFT have REV bit enabled for normal display
    else reg_write(0x61, oldreg&0xFFFE);
}
void TFT932x::FastWindow(bool enable)
    {
        usefastwindow=enable;
    }
// TFT have both column and raw autoincrement inside a window, with internal counters
void TFT932x::window(int x, int y, int w, int h)
{
    if(orientation==1 || orientation==3)
    {
        SWAP(x,y);
        SWAP(w,h);
    }
    fastwindowready=false; // end raw/column going to be set to lower value than bottom-right corner
    reg_write(0x50, x);//start column
    reg_write(0x51, x+w-1);//end column
    reg_write(0x52, y);//start page
    reg_write(0x53, y+h-1);//end page
    
    reg_write(0x20, 0); // since ORG bit is set, address is windows relative, so should be set always to 0000
    reg_write(0x21, 0);
    
    reg_select(0x22, false);  //write mem, just write gram next
}
void TFT932x::window4read(int x, int y, int w, int h)
{
    if(orientation==1 || orientation==3)
    {
        SWAP(x,y);
        SWAP(w,h);
    }
    fastwindowready=false; // end raw/column going to be set to lower value than bottom-right corner
    reg_write(0x50, x);//start column
    reg_write(0x51, x+w-1);//end column
    reg_write(0x52, y);//start page
    reg_write(0x53, y+h-1);//end page
    
    reg_write(0x20, 0); // since ORG bit is set, address is windows relative, so should be set always to 0000
    reg_write(0x21, 0);
    
    reg_select(0x22, true);  //read mem, just read gram next
    dummyread();
}
void TFT932x::pixel(int x, int y, unsigned short color)
{
    if(usefastwindow)
    {
        if(fastwindowready) //setting only start column/page does speedup, but needs end raw/column previously set to bottom-right corner
        {
            if(orientation==1 || orientation==3) SWAP(x,y);
            reg_write(0x50, x);//start column only
            reg_write(0x52, y);//start page only
            reg_write(0x20, 0); // since ORG bit is set, address is window relative, so should be set always to 0000
            reg_write(0x21, 0);
            reg_select(0x22, false);  //write mem, just write gram next
        }
        else
        {
            window(x,y,width()-x,height()-y); // set also end raw/column to bottom-right corner
            fastwindowready=true;
        }
    }
    else window(x,y,1,1);
    wr_gram(color);
}
unsigned short TFT932x::pixelread(int x, int y)
{
  /*  if(usefastwindow) // for ILI9325 fastwindows for reading works only in PAR16
    {
        if(fastwindowready) //setting only start column/page does speedup, but needs end raw/column previously set to bottom-right corner
        {
            if(orientation==1 || orientation==3) SWAP(x,y);
            reg_write(0x50, x);//start column only
            reg_write(0x52, y);//start page only
            reg_write(0x20, 0); // since ORG bit is set, address is window relative, so should be set always to 0000
            reg_write(0x21, 0);
            reg_select(0x22, true);  //read mem, just read gram next
        }
        else
        {
            window4read(x,y,width()-x,height()-y); // set also end raw/column to bottom-right corner
            fastwindowready=true;
        }
    }
    else*/
    window4read(x,y,1,1);
    
    unsigned short color;
    color = rd_gram();
    if(isBGR) color = BGR2RGB(color); // in case, convert BGR to RGB 
    return color;
}
void TFT932x::setscrollarea (int startY, int areasize) // ie 0,480 for whole screen
{
    // ILI932x allows only ful lscreen scrolling
    unsigned short oldreg = reg_read(0x61);
    reg_write(0x61, oldreg|2); // enable scroll
}
void TFT932x::scroll (int lines) // ie 1= scrollup 1, 479= scrolldown 1
{
    reg_write(0x6A, lines%screensize_Y); // select the (absolute)line which will be displayed as first line 
}
void TFT932x::scrollreset()
{
    unsigned short oldreg = reg_read(0x61);
 //   reg_write(0x61, oldreg&0xFFFD); // disable scroll
    reg_write(0x6A, 0);
}
void TFT932x::cls (void)
{
    WindowMax();
    wr_gram(_background,screensize_X*screensize_Y);
}
// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR
void TFT932x::auto_gram_read_format()
{
    unsigned short px=0xCDB1;
    unsigned short rback, rback18;
    pixel(0,0,px);
    window4read(0,0,1,1);
    rback=proto->rd_gram(0); // try 16bit
    window4read(0,0,1,1);
    rback18=proto->rd_gram(1); // try 18bit converted to 16
    if((rback18==px) || (BGR2RGB(rback18)==px))
    {
        is18bit=true;
        if(BGR2RGB(rback18)==px) isBGR=true;
    }
    else if((rback==px) || (BGR2RGB(rback)==px))
    {
        if(BGR2RGB(rback)==px) isBGR=true;
    }
 //   debug("\r\nIdentify gram read color format,\r\nsent %.4X read16 %.4X(bgr%.4X) read18 %.4X(bgr%.4X)", px, rback, BGR2RGB(rback), rback18, BGR2RGB(rback18));    
}
// try to identify display controller
void TFT932x::identify()
{
    tftID = reg_read(0x00);
    hw_reset(); // in case wrong cmd messed up important settings
}
int TFT932x::sizeX()
{
    return screensize_X;
}
int TFT932x::sizeY()
{
    return screensize_Y;
}