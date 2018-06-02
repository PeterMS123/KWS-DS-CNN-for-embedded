#ifndef MBED_UC1608_H
#define MBED_UC1608_H



#include "mbed.h"
#include "LCD.h"

/** Class for UC1608 display controller
* to be copypasted and adapted for other controllers
*/
class UC1608 : public LCD
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
    UC1608(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name, unsigned int LCDSIZE_X = 240, unsigned  int LCDSIZE_Y = 120);
  
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
    UC1608(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char* name , unsigned int LCDSIZE_X = 240, unsigned  int LCDSIZE_Y = 120);
  
    /** set the contrast of the screen
      * @note here overrided because of not standard command
      * @param o contrast 0-63
      */
    virtual void set_contrast(int o);
    
    /** Set ChipSelect high or low
    * @note here overriding the std one, cause CS inverted
    * @param enable 0/1   
    */
    virtual void BusEnable(bool enable);
  
protected:
    
    
    /** Init command sequence  
    */
    void init();
    
    /** set mirror mode
      * @note here overriding the LCD class default one because of not standard commands
      * @param mode NONE, X, Y, XY 
      */
    virtual void mirrorXY(mirror_t mode);
    
    



};
#endif