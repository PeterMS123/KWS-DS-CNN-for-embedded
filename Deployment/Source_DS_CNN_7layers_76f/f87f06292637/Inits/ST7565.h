#ifndef MBED_ST7565_H
#define MBED_ST7565_H



#include "mbed.h"
#include "LCD.h"

/** Class for ST7565 and similar display controllers
* to be copypasted and adapted for other controllers
*/
class ST7565 : public LCD
{
 
 public:

    /** Create a PAR display interface
    * @param displayproto only supports PAR_8
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
    ST7565(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name, unsigned int LCDSIZE_X = 132, unsigned  int LCDSIZE_Y = 64);
    
    /** Create a BUS display interface
    * @param displayproto only supports BUS_8
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
    ST7565(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name, unsigned int LCDSIZE_X = 132, unsigned  int LCDSIZE_Y = 64);
  
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
    ST7565(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char* name, unsigned int LCDSIZE_X = 132, unsigned  int LCDSIZE_Y = 64);
  

  
protected:
    
    
    /** Init command sequence  
    */
    void init();



};
#endif