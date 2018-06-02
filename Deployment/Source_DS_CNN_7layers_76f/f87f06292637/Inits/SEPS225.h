#ifndef SEPS225_H
#define SEPS225_H included

#include "mbed.h"
#include "TFT.h"
//#include "vt100.h"    
//extern vt100 *tty ;
 
/** Class for SEPS225 Syncoam Co.,Ltd
 * 128 x 128 Dots, 262K Colors PM-OLED Display Driver and Controller
 */
 
class SEPS225 : public TFT
{

public:

    /** Create a PAR display interface
    * @param displayproto PAR_8 or PAR_16
    * @param port GPIO port name to use
    * @param CS pin connected to CS of display
    * @param reset pin connected to RESET of display
    * @param DC pin connected to data/command of display
    * @param WR pin connected to SDI of display
    * @param RD pin connected to RS of display
    * @param name The name used by the parent class to access the interface
    * @param LCDSIZE_X x size in pixel - optional
    * @param LCDSIZE_Y y size in pixel - optional
    */ 
    SEPS225(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name ,const unsigned int LCDSIZE_X = 128, unsigned  int LCDSIZE_Y = 128);
    
    /** Create a BUS display interface
    * @param displayproto BUS_8 or BUS_16
    * @param buspins array of PinName to group as Bus
    * @param CS pin connected to CS of display
    * @param reset pin connected to RESET of display
    * @param DC pin connected to data/command of display
    * @param WR pin connected to SDI of display
    * @param RD pin connected to RS of display
    * @param name The name used by the parent class to access the interface
    * @param LCDSIZE_X x size in pixel - optional
    * @param LCDSIZE_Y y size in pixel - optional
    */ 
    SEPS225(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name ,const unsigned int LCDSIZE_X = 128, unsigned  int LCDSIZE_Y = 128);
  
    /** Create an SPI display interface
    * @param displayproto SPI_8 or SPI_16
    * @param Hz SPI speed in Hz
    * @param mosi SPI pin
    * @param miso SPI pin
    * @param sclk SPI pin
    * @param CS pin connected to CS of display
    * @param reset pin connected to RESET of display
    * @param DC pin connected to data/command of display
    * @param name The name used by the parent class to access the interface
    * @param LCDSIZE_X x size in pixel - optional
    * @param LCDSIZE_Y y size in pixel - optional
    */ 
    SEPS225(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char* name ,unsigned int LCDSIZE_X = 128, unsigned  int LCDSIZE_Y = 128);
  

    virtual void pixel(int x, int y, unsigned short color);
    virtual void window(int x, int y, int w, int h);
    virtual void rect(int x0, int y0, int x1, int y1, unsigned short color) ;
    virtual void cls(void) ; // virtual
    virtual unsigned short pixelread(int x, int y);
    virtual void window4read(int x, int y, int w, int h);
    virtual void window_pushpixel(unsigned short color);
    virtual void window_pushpixel(unsigned short color, unsigned int count);
    virtual void window_pushpixelbuf(unsigned short* color, unsigned int lenght);
    void display(int onoff) ;
    
    void reg_write(unsigned char cmd, unsigned char data) ;
    void cmd_write(unsigned char cmd) ;
    void data_write(unsigned char data) ;
    void write8(unsigned char data) ;
    void write16(unsigned short sdata) ;
    void bufwrite8(unsigned char *data, unsigned long len) ;
    void bufwrite16(unsigned short *sdata, unsigned long len) ;
  
protected:
    
    
    /** Init command sequence  
    */
    void init();


private:
    DigitalOut *_cs ;
    DigitalOut *_rs ;
} ;
 
 #endif /* SEPS225_H */
 
