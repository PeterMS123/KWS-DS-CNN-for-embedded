/* mbed UniGraphic library - I2C protocol class
 * Copyright (c) 2017 Peter Drescher
 * Released under the MIT License: http://mbed.org/license/mit
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
 
#include "I2C_bus.h"

I2C_bus::I2C_bus(int Hz, int address, PinName sda, PinName scl)
    : _i2c(sda,scl)
{
    _i2c.frequency(Hz);
    _address = address;
    //hw_reset();    
}

void I2C_bus::wr_cmd8(unsigned char cmd)
{     
    char tmp[2];
    tmp[0] = 0x00;  //command 
    tmp[1] = cmd;
    _i2c.write(_address,tmp,2);
}
void I2C_bus::wr_data8(unsigned char data)
{
    _i2c.write(data);    // write 8bit
}
void I2C_bus::wr_cmd16(unsigned short cmd)
{     
    char tmp[3];
    tmp[0] = 00; //command
    tmp[1] = cmd>>8;
    tmp[2] = cmd&0xFF;
    
    _i2c.write(_address,tmp,3);
}
void I2C_bus::wr_data16(unsigned short data)
{
    _i2c.write(data>>8);    // write 8bit
    _i2c.write(data&0xFF);    // write 8bit
}
void I2C_bus::wr_gram(unsigned short data)
{
    _i2c.write(data>>8);    // write 8bit
    _i2c.write(data&0xFF);    // write 8bit
}
void I2C_bus::wr_gram(unsigned short data, unsigned int count)
{
    _i2c.start();
    _i2c.write(_address);
    _i2c.write(0x40);          // data continue
    if((data>>8)==(data&0xFF))
    {
        count<<=1;
        while(count)
        {
            _i2c.write(data);    // write 8bit
            count--;
        }
    }
    else
    {
        while(count)
        {
            _i2c.write(data>>8);    // write 8bit
            _i2c.write(data&0xFF);    // write 8bit
            count--;
        }
    }
   _i2c.stop();
}
void I2C_bus::wr_grambuf(unsigned short* data, unsigned int lenght)
{
    _i2c.start();
    _i2c.write(_address);
    _i2c.write(0x40);          // data continue
    while(lenght)
    {
        _i2c.write((*data)>>8);    // write 8bit
        _i2c.write((*data)&0xFF);    // write 8bit
        data++;
        lenght--;
    }
    _i2c.stop();
}

void I2C_bus::hw_reset()
{
    
}
void I2C_bus::BusEnable(bool enable)
{
}

void I2C_bus::reg_select(unsigned char reg, bool forread)
{    
}

unsigned int I2C_bus::rd_reg_data32(unsigned char reg)
{
     return 0;
}

unsigned int I2C_bus::rd_extcreg_data32(unsigned char reg, unsigned char SPIreadenablecmd)
{
    return 0;
}

void I2C_bus::dummyread()
{
}

unsigned short I2C_bus::rd_gram(bool convert)
{
    return (0);    
}

unsigned short I2C_bus::reg_read(unsigned char reg)
{
    return (0);
}

void I2C_bus::reg_write(unsigned char reg, unsigned short data)
{
}