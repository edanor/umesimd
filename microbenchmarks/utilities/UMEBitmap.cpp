// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
//
// Author: Przemyslaw Karpinski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
//  This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//

#include "UMEBitmap.h"
#include "UMEEndianness.h"
#include "../../UMEMemory.h"

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cstring>

bool UME::operator == (UME::BitmapFileHeader & h0, UME::BitmapFileHeader & h1)
{
    if(     h0.fileSize != h1.fileSize// Check size first for better performance
        ||  h0.headerID != h1.headerID
        ||  h0.imageOffset != h1.imageOffset
        ||  h0.reserved1 != h1.reserved1
        ||  h0.reserved2 != h1.reserved2)
    {
        return false;
    }

    return true;
}


bool UME::operator != (UME::BitmapFileHeader & h0, UME::BitmapFileHeader & h1)
{
    if(     h0.fileSize != h1.fileSize // Check size first for better performance
        ||  h0.headerID != h1.headerID
        ||  h0.imageOffset != h1.imageOffset
        ||  h0.reserved1 != h1.reserved1
        ||  h0.reserved2 != h1.reserved2)
    {
        return true;
    }

    return false;
}

bool UME::operator == (UME::BitmapDIBHeader & h0, UME::BitmapDIBHeader & h1)
{
    if(     h0.headerSize != h1.headerSize // Check size first for better performance
        ||  h0.bitsPerPixel != h1.bitsPerPixel
        ||  h0.colorPlanes != h1.colorPlanes
        ||  h0.height != h1.height
        ||  h0.width != h1.width)
    {
        return true;
    }

    return false;
}

bool UME::operator != (UME::BitmapDIBHeader & h0, UME::BitmapDIBHeader & h1)
{
    if(     h0.headerSize != h1.headerSize // Check size first for better performance
        ||  h0.bitsPerPixel != h1.bitsPerPixel
        ||  h0.colorPlanes != h1.colorPlanes
        ||  h0.height != h1.height
        ||  h0.width != h1.width)
    {
        return false;
    }

    return true;
}

UME::Bitmap::Bitmap(uint32_t width, uint32_t height, PIXEL_TYPE type)
{
    (void)type; // ignore unuesed parameter

    uint32_t imageSize;// = ((24*width + 31)/32)*4*height; // Size of data. Every horizontal line has to be 32b aligned.
    //mPaddedWidth = (uint32_t) std::ceil(((double)width*3) / 32) * 32;
    mPaddedWidth = (uint32_t) std::ceil((double)width*24 / 32)*4;
    imageSize = height*mPaddedWidth;
    // header length is 14 bytes
    mHeader.headerID = 0x4d42;
        WRITE_WORD(mHeader.raw + 0, mHeader.headerID);
    mHeader.fileSize = imageSize + 14 + 40; // image size + header size + DIB header size
        WRITE_DWORD(mHeader.raw + 2, mHeader.fileSize);
    mHeader.reserved1 = 0;
        WRITE_WORD(mHeader.raw + 6, mHeader.reserved1);
    mHeader.reserved2 = 0;
        WRITE_WORD(mHeader.raw + 8, mHeader.reserved2);
    mHeader.imageOffset = 0x36;
        WRITE_DWORD(mHeader.raw + 10, mHeader.imageOffset);
    
    // DIB header length is 40 bytes
    mDIBHeader.headerSize = 0x28;
        WRITE_DWORD(mDIBHeader.raw + 0, mDIBHeader.headerSize);
    mDIBHeader.width = width;
        WRITE_DWORD(mDIBHeader.raw + 4, mDIBHeader.width);
    mDIBHeader.height = height;
        WRITE_DWORD(mDIBHeader.raw + 8, mDIBHeader.height);
    mDIBHeader.colorPlanes = 0x1;   // number of color planes
        WRITE_WORD(mDIBHeader.raw + 12, mDIBHeader.colorPlanes);
    mDIBHeader.bitsPerPixel = 24;   // use RGB only data
        WRITE_WORD(mDIBHeader.raw + 14, mDIBHeader.bitsPerPixel);
    
        WRITE_DWORD(mDIBHeader.raw + 16, 0); // compresion method (0 == BI_RGB - none)
        WRITE_DWORD(mDIBHeader.raw + 20, imageSize); // image size 
        WRITE_DWORD(mDIBHeader.raw + 24, 0); // horizontal resolution
        WRITE_DWORD(mDIBHeader.raw + 28, 0); // vertical re solution
        WRITE_DWORD(mDIBHeader.raw + 32, 0); // number of colors in the color palette (0 for 2^n)
        WRITE_DWORD(mDIBHeader.raw + 36, 0); // number of important colors

    this->mRasterData = (uint8_t*)UME::DynamicMemory::Malloc(GetBitmapSize());
}

UME::Bitmap::Bitmap(std::string const & fileName)
{
    if (!LoadFromFile(fileName))
    {
        std::cerr << "Failed to load file: " + fileName  <<  std::endl;
    }
}


UME::Bitmap::Bitmap(Bitmap & original, bool copyData)
{
    this->mHeader = original.mHeader;
    this->mDIBHeader = original.mDIBHeader;
    this->mPaddedWidth = original.mPaddedWidth;
    
    unsigned int bitmapSize = GetBitmapSize();
    this->mRasterData = (uint8_t*)UME::DynamicMemory::Malloc(bitmapSize);
    
    if(copyData) // copy whole blob
    {
        std::memcpy(this->mRasterData, original.mRasterData, bitmapSize);
    }
    else
    {
        std::memset(this->mRasterData, 0, bitmapSize);
    }
}

UME::Bitmap::Bitmap(UME::BitmapFileHeader &header, UME::BitmapDIBHeader &dIBHeader)
{
    uint32_t rasterSize = GetBitmapSize();
    this->mHeader = header;
    this->mDIBHeader = dIBHeader;

    memset(this->mRasterData, 0, rasterSize);
    
    mPaddedWidth = (uint32_t) std::ceil((double)mDIBHeader.width*mDIBHeader.bitsPerPixel / 32)*4;
}

UME::Bitmap::~Bitmap()
{
    UME::DynamicMemory::Free(mRasterData);
}

bool UME::Bitmap::LoadFromFile(std::string const & fileName)
{
    FILE *file = NULL;
    bool retval = true;
    size_t read_size = 0;
    
    do
    {
#if defined (_MSC_VER)
        fopen_s(&file, fileName.c_str(), "rb");
#else
        file = fopen(fileName.c_str(), "rb");
#endif
        if (!file)
        {
            std::cerr << "Error: cannot read file: " << fileName << std::endl;
            retval = false;
            break;
        }

        // Read the file header
        read_size = fread(mHeader.raw, 1, UME_BITMAP_HEADER_LENGTH, file);
        if(read_size != UME_BITMAP_HEADER_LENGTH) {
            std::cerr << "Error: reading bitmap header: " << fileName << std::endl;
        }

        // Parse the header 
        mHeader.headerID = READ_WORD(mHeader.raw);
        mHeader.fileSize = READ_DWORD(mHeader.raw + 2);
        mHeader.reserved1 = READ_WORD(mHeader.raw + 6);
        mHeader.reserved2 = READ_WORD(mHeader.raw + 8);
        mHeader.imageOffset = READ_DWORD(mHeader.raw + 10);

        // Read the DIB header

        read_size = fread(mDIBHeader.raw, 1, 4, file);
        if(read_size != 4) {
            std::cerr << "Error: reading DIB header size: " << fileName << std::endl;
        }
        mDIBHeader.headerSize = READ_DWORD(mDIBHeader.raw);

        // TODO: different types of headers can be supported. Differentiation depends on header size.
        if (mDIBHeader.headerSize != UME_BITMAP_DIB_HEADER_LENGTH)
        {
            std::cerr << "Error: invalid size of dIBHeader: " << mDIBHeader.headerSize << std::endl;
            retval = false;
            break;
        }

        read_size = fread(mDIBHeader.raw + 4, 1, mDIBHeader.headerSize - 4, file);
        if(read_size != mDIBHeader.headerSize - 4) {
            std::cerr << "Error: reading DIB header: " << fileName << std::endl;
        }

        mDIBHeader.width  = READ_DWORD(mDIBHeader.raw + 4);
        mDIBHeader.height = READ_DWORD(mDIBHeader.raw + 8);
        mDIBHeader.colorPlanes = READ_WORD(mDIBHeader.raw + 12);
        mDIBHeader.bitsPerPixel = READ_WORD(mDIBHeader.raw + 14);
        
        mPaddedWidth = (uint32_t) std::ceil((double)mDIBHeader.width*mDIBHeader.bitsPerPixel / 32)*4;

        // Read the bitmap
        unsigned int bitmapSize = GetBitmapSize();
        // Sanity check
        if(bitmapSize != mPaddedWidth*mDIBHeader.height)
        {
            std::cout << "UMEBitmap: error - invalid line padding!" << std::endl;
        }

        mRasterData = (uint8_t*)UME::DynamicMemory::Malloc(bitmapSize);
        read_size = fread(mRasterData, 1, bitmapSize, file);
        if(read_size != bitmapSize) {
            std::cerr << "Error: reading bitmap data: " << fileName << std::endl;
        }
    } while (0);

    fclose(file);
    return retval;
}

bool UME::Bitmap::SaveToFile(std::string const & fileName)
{
    FILE *file = NULL;
    bool retval = true;

    do
    {
#if defined (_MSC_VER)
        fopen_s( &file, fileName.c_str(), "wb");
#else
        file = fopen(fileName.c_str(), "wb");
#endif
        if (!file)
        {
            std::cerr << "Error: cannot open file: " << fileName << std::endl;
            retval = false;
            break;
        }

        // Read the file header
        fwrite(mHeader.raw, 1, UME_BITMAP_HEADER_LENGTH, file);
        fwrite(mDIBHeader.raw, 1, mDIBHeader.headerSize, file);

        fwrite(mRasterData, 1, GetBitmapSize(), file);

    } while (0);

    fclose(file);

    return retval;
}

uint8_t* UME::Bitmap::GetRasterData()
{
    return mRasterData;
}

void UME::Bitmap::CopyRasterData(uint8_t* data)
{
    memcpy(mRasterData, data, GetBitmapSize());
}
        

uint32_t UME::Bitmap::GetPixelCount()
{
    return mDIBHeader.width * mDIBHeader.height;
}

uint8_t UME::Bitmap::GetPixelSize()
{
    return mDIBHeader.bitsPerPixel / 8;
}

// every line has to be 32b aligned
UME::PixelCoord2D UME::Bitmap::GetPixelCoord(uint8_t *pixel)
{
    uint32_t offset = (uint32_t)(pixel - mRasterData);
    uint32_t y = offset / (mDIBHeader.width * GetPixelSize());
    uint32_t x = offset/GetPixelSize() - y*mDIBHeader.width;

    return PixelCoord2D(x, y);
}

uint32_t UME::Bitmap::GetPixelValue(uint32_t x, uint32_t y)
{
    uint32_t rasterDataOffset = y*GetPaddedWidth() + x*GetPixelSize();
    uint32_t value = 0;

    for (int i = 0; i < GetPixelSize(); i++)
    {
        value += mRasterData[rasterDataOffset + i] << 8*i;
    }
    
    return value;
}

// TODO: this doesn't seem right!!!
uint32_t * UME::Bitmap::GetPixelValues(uint32_t start_x, uint32_t start_y, int count, uint32_t *destination)
{
    uint32_t rasterDataOffset = (start_x*GetWidth() + start_y)*GetPixelSize();
    uint32_t *value = destination;
    uint32_t pixelSize = GetPixelSize();

    for(int currPixel = 0; currPixel < count; currPixel++)
    {
        *value = 0;
        for(int channel = 0; channel < GetPixelSize(); channel++)
        {
            (*value) += mRasterData[rasterDataOffset + currPixel*pixelSize + channel] << 8*channel;
        }
    }

    return destination;
}

void UME::Bitmap::SetPixelValue(uint32_t x, uint32_t y, uint32_t value)
{
    uint32_t rasterDataOffset = y*GetPaddedWidth() + x*GetPixelSize();
   
    for(int channel = 0; channel < GetPixelSize(); channel++)
    {
        mRasterData[rasterDataOffset + channel] = (uint8_t)((( 0xFF << 8*channel) & value) >> 8*channel);
    }
}

void UME::Bitmap::GetHeader(UME::BitmapFileHeader *headerContainer)
{
    headerContainer->headerID = mHeader.headerID;
    headerContainer->imageOffset = mHeader.imageOffset;
    headerContainer->fileSize = mHeader.fileSize;
    headerContainer->reserved1 = mHeader.reserved1;
    headerContainer->reserved2 = mHeader.reserved2;

    memcpy(headerContainer->raw, mHeader.raw, UME_BITMAP_HEADER_LENGTH);
}

void UME::Bitmap::GetDIBHeader(UME::BitmapDIBHeader *headerContainer)
{
    headerContainer->bitsPerPixel = mDIBHeader.bitsPerPixel;
    headerContainer->colorPlanes = mDIBHeader.colorPlanes;
    headerContainer->headerSize = mDIBHeader.headerSize;
    headerContainer->height = mDIBHeader.height;
    headerContainer->width = mDIBHeader.width;

    memset(headerContainer->zeros, 0, UME_BITMAP_DIB_HEADER_ZEROS_LENGTH);
    memcpy(headerContainer->raw, mDIBHeader.raw, UME_BITMAP_DIB_HEADER_LENGTH);
}

void UME::Bitmap::ClearTarget(uint8_t r, uint8_t g, uint8_t b)
{
    uint32_t pixelSize = GetPixelSize();
    for(uint32_t i = 0; i < GetPixelCount()*pixelSize; i+=pixelSize)
    {
        mRasterData[i] = r;
        mRasterData[i + 1] = g;
        mRasterData[i + 2] = b;
    }
}

void UME::Bitmap::DrawLine(double r, double theta, Color color)
{
    const double EPS = 0.001;
    const uint32_t WIDTH = this->GetWidth();
    const uint32_t HEIGHT = this->GetHeight();
    const double SIN_THETA = sin(theta);
    const double COS_THETA = cos(theta);
    const double TAN_THETA = SIN_THETA/COS_THETA;
    const double CTAN_THETA = COS_THETA/SIN_THETA;
    const double R_D_COS_THETA = r/COS_THETA;
    const double R_D_SIN_THETA = r/SIN_THETA;

    // Check if the line is vertical
    //if((theta < (PI_HALF + EPS) && theta > (PI_HALF - EPS) ) ||
    //   (theta > (3*PI_HALF - EPS) && theta < (3*PI_HALF - EPS)))
    if((theta < EPS || theta > 2*UME::CONSTANTS::PI) ||
        (theta > UME::CONSTANTS::PI - EPS && theta <UME::CONSTANTS::PI + EPS))
    {
        if( r > 0.0 && r < (double)WIDTH)
        {
            for(uint32_t i = 0; i < HEIGHT; i++)
            {
                SetPixelValue((uint32_t)r, i, color);
            }
        }
    }
    else
    {
        // convert (r,theta) to (a,b)
        for(uint32_t y = 0; y < HEIGHT; y++)
        {
           double x = R_D_COS_THETA - (double)y * TAN_THETA;
           if(x > 0.0 && x < (double)WIDTH)
           {
                SetPixelValue((uint32_t)x, y, color);
               // this->SaveToFile(std::string("..\\..\\out_debug.bmp"));
           }
        }
        for(uint32_t x = 0; x < WIDTH; x++)
        {
            double y = R_D_SIN_THETA - (double)x * CTAN_THETA;
            if(y > 0.0 && y < (double)HEIGHT)
            {
                SetPixelValue(x, (uint32_t)y, color);
            }
        }
    }
}
