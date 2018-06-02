#ifndef MBED_ILI932x_H
#define MBED_ILI932x_H



#include "mbed.h"
#include "TFT932x.h"

/** Class for ILI932x tft display controller
* to be copypasted and adapted for other controllers
*/
class ILI932x : public TFT932x
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
    ILI932x(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name, unsigned int LCDSIZE_X = 240, unsigned  int LCDSIZE_Y = 320);
    
    /** Create a BUS display interface
    * @param displayproto BUS_8 or BUS_16
    * @param buspins array of PinName to group as Bus
    * , i.e. PinName buspins[8]={PC_0,PC_1,PC_2,PC_3,D9,D8,D7,D6};
    * @param CS pin connected to CS of display
    * @param reset pin connected to RESET of display
    * @param DC pin connected to data/command of display
    * @param WR pin connected to SDI of display
    * @param RD pin connected to RS of display
    * @param name The name used by the parent class to access the interface
    * @param LCDSIZE_X x size in pixel - optional
    * @param LCDSIZE_Y y size in pixel - optional
    */ 
    ILI932x(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name, unsigned int LCDSIZE_X = 240, unsigned  int LCDSIZE_Y = 320);
  
    /** Create an SPI display interface
    * @param displayproto SPI_8 or SPI_16
    * @param Hz SPI speed in Hz
    * @param mosi SPI pin
    * @param miso SPI pin
    * @param sclk SPI pin
    * @param CS pin connected to CS of display
    * @param reset pin connected to RESET of display
    * @param name The name used by the parent class to access the interface
    * @param LCDSIZE_X x size in pixel - optional
    * @param LCDSIZE_Y y size in pixel - optional
    */ 
    ILI932x(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, const char* name, unsigned int LCDSIZE_X = 240, unsigned  int LCDSIZE_Y= 320);
  

  
protected:
    
    
    /** Init command sequence  
    */
    void init9325();
    void init9320();

};
#endif