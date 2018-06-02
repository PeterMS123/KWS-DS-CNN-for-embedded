#ifndef MBED_SSD1306_H
#define MBED_SSD1306_H

#include "mbed.h"
#include "LCD.h"

/** Class for SSD1306 display controller
* to be copypasted and adapted for other controllers
*/
class SSD1306 : public LCD
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
    SSD1306(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char* name, unsigned int LCDSIZE_X = 128, unsigned  int LCDSIZE_Y = 64);
  
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
    SSD1306(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char* name , unsigned int LCDSIZE_X = 128, unsigned  int LCDSIZE_Y = 64);

    /** Create an I2C display interface
    * @param displayproto I2C_
    * @param Hz I2C speed in Hz
    * @param address I2C address
    * @param sda I2C pin
    * @param scl I2C pin
    * @param name The name used by the parent class to access the interface
    * @param LCDSIZE_X x size in pixel - optional
    * @param LCDSIZE_Y y size in pixel - optional
    */ 
    SSD1306(proto_t displayproto, int Hz, int address, PinName sda, PinName scl, const char* name , unsigned int LCDSIZE_X = 128, unsigned  int LCDSIZE_Y = 64);



    /** set the contrast of the screen
      * @note here overrided because of not standard value range
      * @param o contrast 0-255
      */
    virtual void set_contrast(int o);
    
    /** set automatc horizontal scroll mode
     * @param l_r direction - left = 0, right = 1
     * @param s_page start page
     * @param e_page end page
     * @param speed time between horizontal shift. 0 slow .. 7 fast
     */
    void horizontal_scroll(int l_r,int s_page,int e_page,int speed);

    /** automatic horizontal + vertical scroll mode
     * @param l_r direction - left = 0, right = 1
     * @param s_page start page
     * @param e_page end page
     * @param v_off vertical offset for scroll
     * @param speed time between horizontal shift. 0 slow .. 7 fast
     */
    void horiz_vert_scroll(int l_r,int s_page,int e_page,int v_off,int speed);

    /** end scroll mode
     *
     */
    void end_scroll(void);
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