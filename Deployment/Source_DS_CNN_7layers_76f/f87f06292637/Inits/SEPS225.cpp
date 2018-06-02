#include "mbed.h"

#include "SEPS225.h"

#ifndef OLED_LCD_REG_H_
#define OLED_LCD_REG_H_

#define CMD_INDEX                       0x00
#define CMD_STATUS_RD                   0x01
#define CMD_OSC_CTL                     0x02
#define CMD_CLOCK_DIV                   0x03
#define CMD_REDUCE_CURRENT              0x04
#define CMD_SOFT_RST                    0x05
#define CMD_DISP_ON_OFF                 0x06
#define CMD_PRECHARGE_TIME_R            0x08
#define CMD_PRECHARGE_TIME_G            0x09
#define CMD_PRECHARGE_TIME_B            0x0A
#define CMD_PRECHARGE_CURRENT_R         0x0B
#define CMD_PRECHARGE_CURRENT_G         0x0C
#define CMD_PRECHARGE_CURRENT_B         0x0D
#define CMD_DRIVING_CURRENT_R           0x10
#define CMD_DRIVING_CURRENT_G           0x11
#define CMD_DRIVING_CURRENT_B           0x12
#define CMD_DISPLAY_MODE_SET            0x13
#define CMD_RGB_IF                      0x14
#define CMD_RGB_POL                     0x15
#define CMD_MEMORY_WRITE_MODE           0x16
#define CMD_MX1_ADDR                    0x17
#define CMD_MX2_ADDR                    0x18
#define CMD_MY1_ADDR                    0x19
#define CMD_MY2_ADDR                    0x1A
#define CMD_MEMORY_ACCESS_POINTER_X     0x20
#define CMD_MEMORY_ACCESS_POINTER_Y     0x21
#define CMD_DDRAM_DATA_ACCESS_PORT      0x22
#define CMD_DUTY                        0x28
#define CMD_DSL                         0x29
#define CMD_D1_DDRAM_FAC                0x2E
#define CMD_D1_DDRAM_FAR                0x2F
#define CMD_D2_DDRAM_SAC                0x31
#define CMD_D2_DDRAM_SAR                0x32
#define CMD_SCR1_FX1                    0x33
#define CMD_SCR1_FX2                    0x34
#define CMD_SCR1_FY1                    0x35
#define CMD_SCR1_FY2                    0x36
#define CMD_SCR2_SX1                    0x37
#define CMD_SCR2_SX2                    0x38
#define CMD_SCR2_SY1                    0x39
#define CMD_SCR2_SY2                    0x3A
#define CMD_SCREEN_SAVER_CONTEROL       0x3B
#define CMD_SS_SLEEP_TIMER              0x3C
#define CMD_SCREEN_SAVER_MODE           0x3D
#define CMD_SS_SCR1_FU                  0x3E
#define CMD_SS_SCR1_MXY                 0x3F
#define CMD_SS_SCR2_FU                  0x40
#define CMD_SS_SCR2_MXY                 0x41
#define CMD_MOVING_DIRECTION            0x42
#define CMD_SS_SCR2_SX1                 0x47
#define CMD_SS_SCR2_SX2                 0x48
#define CMD_SS_SCR2_SY1                 0x49
#define CMD_SS_SCR2_SY2                 0x4A
#define CMD_IREF                        0x80


#define LCD_RESET   0
#define LCD_CLEAR   1
#define LCD_PRINT   2


#define COLOR_RED     0xF100
#define COLOR_GREEN   0x07E0
#define COLOR_BLUE    0x001F
#define COLOR_CYAN    0x07FF
#define COLOR_MAGENTA 0xF11F
#define COLOR_YELLOW  0xFFE0
#define COLOR_BLACK   0x0000
#define COLOR_WHITE   0xFFFF

#define OLED_WIDTH   0x80
#define OLED_HEIGHT  0x80

#endif /* OLED_LCD_REG_H_ */

SEPS225::SEPS225(proto_t displayproto, PortName port, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, port, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);
    identify(); // will collect tftID and set mipistd flag
    init();
    auto_gram_read_format();
    set_orientation(0);
    cls();
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. 
    locate(0,0); 
}
SEPS225::SEPS225(proto_t displayproto, PinName* buspins, PinName CS, PinName reset, PinName DC, PinName WR, PinName RD, const char *name, unsigned int LCDSIZE_X, unsigned  int LCDSIZE_Y)
    : TFT(displayproto, buspins, CS, reset, DC, WR, RD, LCDSIZE_X, LCDSIZE_Y, name)
{
    hw_reset();
    BusEnable(true);
    identify(); // will collect tftID and set mipistd flag
    init();
    auto_gram_read_format();
    set_orientation(0);
    cls();
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. 
    locate(0,0); 
}
SEPS225::SEPS225(proto_t displayproto, int Hz, PinName mosi, PinName miso, PinName sclk, PinName CS, PinName reset, PinName DC, const char *name, unsigned int LCDSIZE_X , unsigned  int LCDSIZE_Y)
    : TFT(displayproto, Hz, mosi, miso, sclk, CS, reset, DC, LCDSIZE_X, LCDSIZE_Y, name)
{
    _cs = new DigitalOut(CS, 1) ;
    _rs = new DigitalOut(DC, 1) ;
    
    hw_reset(); //TFT class forwards to Protocol class
    BusEnable(true); //TFT class forwards to Protocol class
    identify(); // will collect tftID and set mipistd flag
    init(); // per display custom init cmd sequence, implemented here
    auto_gram_read_format();// try to get read gram pixel format, could be 16bit or 18bit, RGB or BGR. Will set flags accordingly
    set_orientation(0); //TFT class does for MIPI standard and some ILIxxx
    FastWindow(true); // most but not all controllers support this, even if datasheet tells they should. 
    cls();
    locate(0,0); 
}

// reset and init the lcd controller
void SEPS225::init()
{
#if 0
   oled = new SPI(PIN_OLED_MOSI, PIN_OLED_MISO, PIN_OLED_SCLK) ;
    oled_cs = new DigitalOut(PIN_OLED_CS, 1) ;
    oled_rs = new DigitalOut(PIN_OLED_RS, 1) ;
    oled->format(8, 0) ;
    oled->frequency(1000000) ;
#endif 
     
    reg_write(CMD_REDUCE_CURRENT, 0x01) ;  wait(0.01) ; //    oled_delay(100000) ;
    reg_write(CMD_REDUCE_CURRENT, 0x00) ;  wait(0.01) ; //    oled_delay(100000) ;
    
    reg_write(CMD_OSC_CTL, 0x01) ;
    reg_write(CMD_CLOCK_DIV, 0x30) ;
//    reg_write(CMD_PRECHARGE_TIME_R, 0x03) ;
//    reg_write(CMD_PRECHARGE_TIME_G, 0x04) ;
//    reg_write(CMD_PRECHARGE_TIME_B, 0x05) ;
  reg_write(CMD_PRECHARGE_TIME_R, 0x0E) ;
  reg_write(CMD_PRECHARGE_TIME_G, 0x0E) ;
  reg_write(CMD_PRECHARGE_TIME_B, 0x0E) ;
//    reg_write(CMD_PRECHARGE_CURRENT_R, 0x0B) ;
//    reg_write(CMD_PRECHARGE_CURRENT_G, 0x0B) ;
//    reg_write(CMD_PRECHARGE_CURRENT_B, 0x0B) ;
    reg_write(CMD_PRECHARGE_CURRENT_R, 0x3E) ;
    reg_write(CMD_PRECHARGE_CURRENT_G, 0x32) ;
    reg_write(CMD_PRECHARGE_CURRENT_B, 0x3D) ;
  reg_write(CMD_DRIVING_CURRENT_R, 0x3E) ;
  reg_write(CMD_DRIVING_CURRENT_G, 0x32) ;
  reg_write(CMD_DRIVING_CURRENT_B, 0x3D) ;
//    reg_write(CMD_DRIVING_CURRENT_R, 0x0B) ;
//    reg_write(CMD_DRIVING_CURRENT_G, 0x0B) ;
//    reg_write(CMD_DRIVING_CURRENT_B, 0x0B) ;
    // Memory Write Mode
    // 16bit * 1 transfer mode, R[5], G[6], B[5]
    // H:Inc, V:Inc, Method:V
//    reg_write(CMD_MEMORY_WRITE_MODE, 0x27) ;
    reg_write(CMD_RGB_IF, 0x11) ;
//    reg_write(CMD_MEMORY_WRITE_MODE, 0x66) ;
    reg_write(CMD_MEMORY_WRITE_MODE, 0x26) ;
    reg_write(CMD_IREF, 0x01) ; // Voltage ctrl by internal
    // 
    reg_write(CMD_MX1_ADDR, 0x00) ;
    reg_write(CMD_MX2_ADDR, 0x7F) ;
    reg_write(CMD_MY1_ADDR, 0x00) ;
    reg_write(CMD_MY2_ADDR, 0x7F) ;
    reg_write(CMD_SCR1_FX1, 0x00) ;
    reg_write(CMD_SCR1_FX2, 0x7F) ;
    reg_write(CMD_SCR1_FY1, 0x00) ;
    reg_write(CMD_SCR1_FY2, 0x7F) ;
    reg_write(CMD_DSL, 0x00) ; // display start line
    reg_write(CMD_DUTY, 0x7F) ; //
    reg_write(CMD_DISPLAY_MODE_SET, 0x04) ; // was 0x00
//    reg_write(CMD_DISPLAY_MODE_SET, 0x00) ; // was 0x00
    reg_write(CMD_D1_DDRAM_FAC, 0x00) ;
    reg_write(CMD_D1_DDRAM_FAR, 0x00) ;
    // Clear
    // oled_lcd_clear() ;
    cls() ;
    
    // DISP_ON_OFF
//  reg_write(CMD_DISP_ON_OFF, 0x01) ;
}

/** Draw a pixel in the specified color.
* @param x is the horizontal offset to this pixel.
* @param y is the vertical offset to this pixel.
* @param color defines the color for the pixel.
*/
void SEPS225::pixel(int x, int y, unsigned short color)
{
    unsigned char data[2] ;
    data[0] = (unsigned char)((color & 0xFF00) >> 8) ;
    data[1] = (unsigned char)(color & 0x00FF) ;
  
    reg_write(CMD_MEMORY_ACCESS_POINTER_X, x) ;
    reg_write(CMD_MEMORY_ACCESS_POINTER_Y, y) ;
    *_cs = 0 ;
    *_rs = 0 ;
    wr_cmd8(CMD_DDRAM_DATA_ACCESS_PORT) ;
    *_rs = 1 ;  

    wr_data8(data[0]) ;
    wr_data8(data[1]) ;
    *_cs = 1 ;
}

void SEPS225::window(int x, int y, int w, int h)
{
    reg_write(CMD_MX1_ADDR, x) ;
    reg_write(CMD_MX2_ADDR, x+w-1) ;
    reg_write(CMD_MY1_ADDR, y) ;
    reg_write(CMD_MY2_ADDR, y+h-1) ;
    reg_write(CMD_MEMORY_ACCESS_POINTER_X, x) ;
    reg_write(CMD_MEMORY_ACCESS_POINTER_Y, y) ;
}

void SEPS225::cls(void)
{
    window(0, 0, OLED_WIDTH, OLED_HEIGHT) ;
    reg_write(CMD_MEMORY_ACCESS_POINTER_X, 0) ;
    reg_write(CMD_MEMORY_ACCESS_POINTER_Y, 0) ;
    *_cs = 0 ;
    *_rs = 0 ;  
    wr_cmd8(CMD_DDRAM_DATA_ACCESS_PORT) ;
    *_rs = 1 ;
    for (int i = 0 ; i < OLED_WIDTH * OLED_HEIGHT ; i++ ) {
        write16(COLOR_BLACK) ;
    }
    *_cs = 1 ;
}

unsigned short SEPS225::pixelread(int x, int y)
{
    unsigned short value = 0x0000 ;
    //printf("SEPS225::pixelread not implemented\n\r") ;
    return(value) ;
}

void SEPS225::rect(int x0, int y0, int x1, int y1, unsigned short color)
{
    float interval = 0.01 ;
//    window(x0, y0, x1-x0+1, y1-y0+1) ;
    *_cs = 0 ;
    wait(interval) ;
    line(x0, y0, x1, y0, color) ;
    *_cs = 1 ;
    wait(interval) ;  
    *_cs = 0 ;
    line(x1, y0, x1, y1, color) ;
    *_cs = 1 ;
    wait(interval) ;
    *_cs = 0 ;
    line(x0, y0, x0, y1, color) ;
    *_cs = 1 ;
    wait(interval) ;
    *_cs = 0 ;
    line(x0, y1, x1, y1, color) ;
    wait(interval) ;
    *_cs = 1 ;
//    *_cs = 1 ;
}
    
void SEPS225::window4read(int x, int y, int w, int h)
{
    //printf("SEPS225::window4read not implemented\n\r") ;
}

void SEPS225::window_pushpixel(unsigned short color)
{
    //printf("SEPS225::window_pushpixel(unsigned short color) not implemented\n\r") ;
    *_cs = 0 ;
    *_rs = 0 ;
     wr_cmd8(CMD_DDRAM_DATA_ACCESS_PORT) ;
    *_rs = 1 ;
    wr_data16(color) ;
//    write16(color) ;
    *_cs = 1 ;
}

void SEPS225::window_pushpixel(unsigned short color, unsigned int count)
{
    //printf("SEPS225::window_pushpixel(unsigned short color, unsigned int count) not implemented\n\r") ;
    *_cs = 0 ;
    *_rs = 0 ;
     wr_cmd8(CMD_DDRAM_DATA_ACCESS_PORT) ;
    *_rs = 1 ;
    for (unsigned int i = 0 ; i < count ; i++ ) {
 //       write16(color) ;
        wr_data16(color) ;
    }
    *_cs = 1 ;
}

void SEPS225::window_pushpixelbuf(unsigned short* color, unsigned int lenght)
{
    //printf("SEPS225::window_pushpixelbuf(unsigned short color, unsigned int length) not implemented\n\r") ;
    *_cs = 0 ;
    *_rs = 0 ;
     wr_cmd8(CMD_DDRAM_DATA_ACCESS_PORT) ;
    *_rs = 1 ;
    for (unsigned int i = 0 ; i < lenght ; i++ ) {
  //      write16(color[i]) ;
        wr_data16(color[i]) ;
    }
    *_cs = 1 ;
}

void SEPS225::reg_write(unsigned char cmd, unsigned char data)
{
   *_cs = 0 ;
   *_rs = 0 ;
    wr_cmd8(cmd) ;
    *_rs = 1 ;
    wr_data8(data) ;
    *_cs = 1 ;
}
    
void SEPS225::display(int onoff) 
{
    reg_write(CMD_DISP_ON_OFF, onoff) ;
} 

void SEPS225::cmd_write(unsigned char cmd) 
{
    *_cs = 0 ;
    *_rs = 0 ;
    wr_cmd8(cmd) ;
    *_rs = 1 ;
    *_cs = 1 ;
}

void SEPS225::data_write(unsigned char data)
{
    *_cs = 0 ;
    wr_data8(data) ;
    *_cs = 1 ;
}

void SEPS225::write8(unsigned char data)
{
    wr_data8(data) ;
}

void SEPS225::write16(unsigned short sdata)
{
    wr_data8((sdata >> 8)&0xFF) ;
    wr_data8(sdata & 0xFF) ;
}

void SEPS225::bufwrite8(unsigned char *data, unsigned long len) 
{
    unsigned long i;
    for (i = 0 ; i < len ; i++ ) {
        wr_data8(data[i]) ;
    }
}

void SEPS225::bufwrite16(unsigned short *sdata, unsigned long len)
{
    unsigned long i ;
    for (i = 0 ; i < len ; i++) {
        write8((*sdata >> 8)&0xFF) ;
        write8(*sdata & 0xFF) ;
    }
}