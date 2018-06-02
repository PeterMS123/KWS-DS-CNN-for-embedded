 /* mbed UniGraphic library - universal TFT driver class
 * Copyright (c) 2015 Giuliano Dianda
 * Released under the MIT License: http://mbed.org/license/mit
 *
 * Derived work of:
 *
 * mbed library for 240*320 pixel display TFT based on ILI9341 LCD Controller
 * Copyright (c) 2013 Peter Drescher - DC2PD
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TFT.h"

//#include "mbed_debug.h"

#define SWAP(a, b)  { a ^= b; b ^= a; a ^= b; }

#if DEVICE_PORTINOUT 
TFT::TFT(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const int lcdsize_x, const int lcdsize_y, const char *name)
    : GraphicsDisplay(name), screensize_X(lcdsize_x), screensize_Y(lcdsize_y)
{
    if(displayproto==PAR_8) proto = new PAR8(port, CS, reset, DC, WR, RD);
    else if(displayproto==PAR_16) proto = new PAR16(port, CS, reset, DC, WR, RD);   
    useNOP=false;
    scrollbugfix=0;
    mipistd=false;
    set_orientation(0);
    foreground(White);
    background(Black);
    set_auto_up(false); //we don't have framebuffer
    topfixedareasize=0;
    scrollareasize=0;
    usefastwindow=false;
    fastwindowready=false;
    is18bit=false;
    isBGR=false;
  //  cls();
  //  locate(0,0);
}
#endif 

TFT::TFT(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const int lcdsize_x, const int lcdsize_y, const char *name)
    : GraphicsDisplay(name), screensize_X(lcdsize_x), screensize_Y(lcdsize_y)
{
    if(displayproto==BUS_8)
    {
        PinName pins[16];
        for(int i=0; i<16; i++) pins[i]=NC;
        for(int i=0; i<8; i++) pins[i]=buspins[i];
        proto = new BUS8(pins, CS, reset, DC, WR, RD);
    }
    else if(displayproto==BUS_16)
    {
        proto = new BUS16(buspins, CS, reset, DC, WR, RD);
    }
    useNOP=false;
    scrollbugfix=0;
    mipistd=false;
    set_orientation(0);
    foreground(White);
    background(Black);
    set_auto_up(false); //we don't have framebuffer
    topfixedareasize=0;
    scrollareasize=0;
    usefastwindow=false;
    fastwindowready=false;
    is18bit=false;
    isBGR=false;
  //  cls();
  //  locate(0,0);
}
TFT::TFT(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const int lcdsize_x, const int lcdsize_y, const char *name)
    : GraphicsDisplay(name), screensize_X(lcdsize_x), screensize_Y(lcdsize_y)
{
    if(displayproto==SPI_8)
    {
        proto = new SPI8(Hz, mosi, miso, sclk, CS, reset, DC);
        useNOP=false;
    }
    else if(displayproto==SPI_16)
    {
        proto = new SPI16(Hz, mosi, miso, sclk, CS, reset, DC);
        useNOP=true;
    }
    scrollbugfix=0;
    mipistd=false;
    set_orientation(0);
    foreground(White);
    background(Black);
    set_auto_up(false);
    topfixedareasize=0;
    scrollareasize=0;
    usefastwindow=false;
    fastwindowready=false;
    is18bit=false;
    isBGR=false;
  //  locate(0,0);
}
void TFT::wr_cmd8(unsigned char cmd)
    {
        if(useNOP) proto->wr_cmd16(cmd); // 0x0000|cmd, 00 is NOP cmd for TFT
        else proto->wr_cmd8(cmd);
    }
void TFT::wr_data8(unsigned char data)
    {
        proto->wr_data8(data);
    }
void TFT::wr_data16(unsigned short data)
    {
        proto->wr_data16(data);
    }
void TFT::wr_gram(unsigned short data)
    {
        proto->wr_gram(data);
    }
void TFT::wr_gram(unsigned short data, unsigned int count)
    {
        proto->wr_gram(data, count);
    }
void TFT::wr_grambuf(unsigned short* data, unsigned int lenght)
    {
        proto->wr_grambuf(data, lenght);
    }
unsigned short TFT::rd_gram()
    {
        return proto->rd_gram(is18bit); // protocol will handle 18to16 bit conversion
    }
unsigned int TFT::rd_reg_data32(unsigned char reg)
    {
        return proto->rd_reg_data32(reg);
    }
unsigned int TFT::rd_extcreg_data32(unsigned char reg, unsigned char SPIreadenablecmd)
    {
        return proto->rd_extcreg_data32(reg, SPIreadenablecmd);
    }
//for TFT, just send data, position counters are in hw
void TFT::window_pushpixel(unsigned short color)
{
    proto->wr_gram(color);
}
void TFT::window_pushpixel(unsigned short color, unsigned int count)
{
    proto->wr_gram(color, count);
}
void TFT::window_pushpixelbuf(unsigned short* color, unsigned int lenght)
    {
        proto->wr_grambuf(color, lenght);
    }
void TFT::hw_reset()
    {
        proto->hw_reset();
        BusEnable(true);
    }
void TFT::BusEnable(bool enable)
    {
        proto->BusEnable(enable);
    }
// color TFT can rotate in hw (swap raw<->columns) for landscape views
void TFT::set_orientation(int o)
{
    orientation = o;
    wr_cmd8(0x36);
    switch (orientation) {
        case 0:// default, portrait view 0°
            if(mipistd) wr_data8(0x0A); // this is in real a vertical flip enabled, seems most displays are vertical flipped
            else wr_data8(0x48); //for some other ILIxxxx
            set_width(screensize_X);
            set_height(screensize_Y);
            break;
        case 1:// landscape view +90°
            if(mipistd) wr_data8(0x28); 
            else wr_data8(0x29);//for some other ILIxxxx
            set_width(screensize_Y);
            set_height(screensize_X);
            break;
        case 2:// portrait view +180°
            if(mipistd) wr_data8(0x09); 
            else wr_data8(0x99);//for some other ILIxxxx
            set_width(screensize_X);
            set_height(screensize_Y);
            break;
        case 3:// landscape view -90°
            if(mipistd) wr_data8(0x2B); 
            else wr_data8(0xF8);//for some other ILIxxxx
            set_width(screensize_Y);
            set_height(screensize_X);
            break;
    }
}
void TFT::invert(unsigned char o)
{
    if(o == 0) wr_cmd8(0x20);
    else wr_cmd8(0x21);
}
void TFT::FastWindow(bool enable)
    {
        usefastwindow=enable;
    }
// TFT have both column and raw autoincrement inside a window, with internal counters
void TFT::window(int x, int y, int w, int h)
{
    fastwindowready=false; // end raw/column going to be set to lower value than bottom-right corner
    wr_cmd8(0x2A);
    wr_data16(x);   //start column
    wr_data16(x+w-1);//end column

    wr_cmd8(0x2B);
    wr_data16(y);   //start page
    wr_data16(y+h-1);//end page
    
    wr_cmd8(0x2C);  //write mem, just send pixels color next
}
void TFT::window4read(int x, int y, int w, int h)
{
    fastwindowready=false;
    wr_cmd8(0x2A);
    wr_data16(x);   //start column
    wr_data16(x+w-1);//end column

    wr_cmd8(0x2B);
    wr_data16(y);   //start page
    wr_data16(y+h-1);//end page
    
    wr_cmd8(0x2E);  //read mem, just pixelread next
}
void TFT::pixel(int x, int y, unsigned short color)
{
    if(usefastwindow) //ili9486 does not like truncated 2A/2B cmds, at least in par mode
    {
        if(fastwindowready) //setting only start column/page does speedup, but needs end raw/column previously set to bottom-right corner
        {
            wr_cmd8(0x2A);
            wr_data16(x);   //start column only
            wr_cmd8(0x2B);
            wr_data16(y);   //start page only
            wr_cmd8(0x2C);  //write mem, just send pixels color next
        }
        else
        {
            window(x,y,width()-x,height()-y); // set also end raw/column to bottom-right corner
            fastwindowready=true;
        }
    }
    else window(x,y,1,1);
  //  proto->wr_gram(color);   // 2C expects 16bit parameters
    wr_gram(color);
}
unsigned short TFT::pixelread(int x, int y)
{
    if(usefastwindow) //ili9486 does not like truncated 2A/2B cmds, at least in par mode
    {
        if(fastwindowready) //setting only start column/page does speedup, but needs end raw/column previously set to bottom-right corner
        {
            wr_cmd8(0x2A);
            wr_data16(x);   //start column only
            wr_cmd8(0x2B);
            wr_data16(y);   //start page only
            wr_cmd8(0x2E);  //read mem, just pixelread next
        }
        else
        {
            window4read(x,y,width()-x,height()-y); // set also end raw/column to bottom-right corner
            fastwindowready=true;
        }
    }
    else window4read(x,y,1,1);
    
    unsigned short color;
  //  proto->wr_gram(color);   // 2C expects 16bit parameters
    color = rd_gram();
    if(isBGR) color = BGR2RGB(color); // in case, convert BGR to RGB (should depend on cmd36 bit3) but maybe is device specific
    return color;
}
void TFT::setscrollarea (int startY, int areasize) // ie 0,480 for whole screen
{
    unsigned int bfa;
    topfixedareasize=startY;
    scrollareasize=areasize;
    wr_cmd8(0x33);
    wr_data16(topfixedareasize); //num lines of top fixed area
    wr_data16(scrollareasize+scrollbugfix); //num lines of vertical scroll area, +1 for ILI9481 fix
    if((areasize+startY)>screensize_Y) bfa=0;
    else bfa = screensize_Y-(areasize+startY);
    wr_data16(bfa); //num lines of bottom fixed area
}
void TFT::scroll (int lines) // ie 1= scrollup 1, 479= scrolldown 1
{
    wr_cmd8(0x37);
    wr_data16(topfixedareasize+(lines%scrollareasize)); // select the (absolute)line which will be displayed as first scrollarea line 
}
void TFT::scrollreset()
{
    wr_cmd8(0x13);  //normal display mode
}
void TFT::cls (void)
{
    WindowMax();
  //  proto->wr_gram(_background,screensize_X*screensize_Y);
  //  proto->wr_gram(0,screensize_X*screensize_Y);
    wr_gram(_background,screensize_X*screensize_Y);
}
// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR
void TFT::auto_gram_read_format()
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
void TFT::identify()
{
    // MIPI std read ID cmd
    tftID=rd_reg_data32(0xBF);
    mipistd=true;
 //   debug("ID MIPI : 0x%8X\r\n",tftID);
    if(((tftID&0xFF)==((tftID>>8)&0xFF)) && ((tftID&0xFF)==((tftID>>16)&0xFF)))
    {
        mipistd=false;
        // ILI specfic read ID cmd
        tftID=rd_reg_data32(0xD3)>>8; 
    //    debug("ID ILI : 0x%8X\r\n",tftID);
    }
    if(((tftID&0xFF)==((tftID>>8)&0xFF)) && ((tftID&0xFF)==((tftID>>16)&0xFF)))
    {
        // ILI specfic read ID cmd with ili9341 specific spi read-in enable 0xD9 cmd
        tftID=rd_extcreg_data32(0xD3, 0xD9);
    //    debug("ID D9 extc ILI : 0x%8X\r\n",tftID);
    }
    if(((tftID&0xFF)==((tftID>>8)&0xFF)) && ((tftID&0xFF)==((tftID>>16)&0xFF)))
    {
        // ILI specfic read ID cmd with ili9486/88 specific spi read-in enable 0xFB cmd
        tftID=rd_extcreg_data32(0xD3, 0xFB);
    //    debug("ID D9 extc ILI : 0x%8X\r\n",tftID);
    }
    if(((tftID&0xFF)==((tftID>>8)&0xFF)) && ((tftID&0xFF)==((tftID>>16)&0xFF))) tftID=0xDEAD;
    if ((tftID&0xFFFF)==0x9481) scrollbugfix=1;
    else scrollbugfix=0;
    hw_reset(); // in case wrong cmds messed up important settings
}
int TFT::sizeX()
{
    return screensize_X;
}
int TFT::sizeY()
{
    return screensize_Y;
}