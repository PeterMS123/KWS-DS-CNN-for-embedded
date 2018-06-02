#ifndef I2C_bus_H
#define I2C_bus_H

#include "mbed.h"
#include "Protocols.h"

/** I2C interface
*/
class I2C_bus : public Protocols
{
 public:

    /** Create an I2C display interface 
    *
    * @param I2C frquency
    * @param I2C address
    * @param I2C pin sda
    * @param I2C pin scl
    */ 
    I2C_bus(int Hz, int address,PinName sda, PinName scl);
 
protected:
   
    /** Send 8bit command to display controller 
    *
    * @param cmd: byte to send  
    *
    */   
    virtual void wr_cmd8(unsigned char cmd);
    
    /** Send 8bit data to display controller 
    *
    * @param data: byte to send   
    *
    */   
    virtual void wr_data8(unsigned char data);
    
    /** Send 2x8bit command to display controller 
    *
    * @param cmd: halfword to send  
    * @note in SPI_16 mode a single 16bit transfer will be done
    */   
    virtual void wr_cmd16(unsigned short cmd);
    
    /** Send 2x8bit data to display controller 
    *
    * @param data: halfword to send   
    * @note in SPI_16 mode a single 16bit transfer will be done
    */   
    virtual void wr_data16(unsigned short data);
    
    /** Send 16bit pixeldata to display controller 
    *
    * @param data: halfword to send   
    *
    */   
    virtual void wr_gram(unsigned short data);
    
    /** Send same 16bit pixeldata to display controller multiple times
    *
    * @param data: halfword to send
    * @param count: how many
    *
    */   
    virtual void wr_gram(unsigned short data, unsigned int count);
    
    /** Send array of pixeldata shorts to display controller
    *
    * @param data: unsigned short pixeldata array
    * @param lenght: lenght (in shorts)
    *
    */   
    virtual void wr_grambuf(unsigned short* data, unsigned int lenght);
    
    /** Read 16bit pixeldata from display controller (with dummy cycle)
    *
    * @param convert true/false. Convert 18bit to 16bit, some controllers returns 18bit
    * @returns 16bit color
    */ 
    virtual unsigned short rd_gram(bool convert);
    
    /** Read 4x8bit register data (
    *   reading from display ia I2C is not implemented in most controllers !
    * 
    */ 
    virtual unsigned int rd_reg_data32(unsigned char reg);
    
    /** Read 3x8bit ExtendedCommands register data
    *   reading from display ia I2C is not implemented in most controllers !
    */ 
    virtual unsigned int rd_extcreg_data32(unsigned char reg, unsigned char SPIreadenablecmd);
    
    /** ILI932x specific, does a dummy read cycle, number of bits is protocol dependent
    *   reading from display ia I2C is not implemented in most controllers !
    */   
    virtual void dummyread ();
    
    /** ILI932x specific, select register for a successive write or read
    *
    *   reading from display ia I2C is not implemented in most controllers !
    */   
    virtual void reg_select(unsigned char reg, bool forread =false);
    
    /** ILI932x specific, write register with data
    *
    * @param reg register to write
    * @param data 16bit data
    *  not implemented for I2C !
    */   
    virtual void reg_write(unsigned char reg, unsigned short data);
    
    /** ILI932x specific, read register
    *
    * @param reg register to be read
    * @returns 16bit register value
    *  not implemented for I2C !
    */ 
    virtual unsigned short reg_read(unsigned char reg);
    
    /** HW reset sequence (without display init commands)
    *  most I2C displays have no reset signal !   
    */
    virtual void hw_reset();
    
    /** Set ChipSelect high or low
    * @param enable 0/1   
    *  not implemented for I2C !
    */
    virtual void BusEnable(bool enable);

private:

    I2C _i2c;
    int _address;
    
};


#endif