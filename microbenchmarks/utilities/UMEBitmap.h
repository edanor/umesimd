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

#ifndef BITMAP_H_
#define BITMAP_H_

#include <string>
#include "../../UMEBasicTypes.h"
#include "../utilities/UMEConstants.h"

namespace UME
{
    typedef struct{
        uint16_t headerID;
        uint32_t fileSize;
        uint16_t reserved1;
        uint16_t reserved2;
        uint32_t imageOffset;
        // Also store a raw copy of the header for debugging
#define UME_BITMAP_HEADER_LENGTH 14
        uint8_t  raw[UME_BITMAP_HEADER_LENGTH]; 
    }BitmapFileHeader;

    // Declar comparison operators for BitmapFileHeader
    bool operator == (BitmapFileHeader & h0, BitmapFileHeader & h1);
    bool operator != (BitmapFileHeader & h0, BitmapFileHeader & h1);

    typedef struct{
        uint32_t headerSize;
        uint32_t width;
        uint32_t height;
        uint16_t colorPlanes;
        uint16_t bitsPerPixel;
#define UME_BITMAP_DIB_HEADER_ZEROS_LENGTH 24
        uint8_t  zeros[UME_BITMAP_DIB_HEADER_ZEROS_LENGTH];
#define UME_BITMAP_DIB_HEADER_LENGTH 40
        uint8_t raw[UME_BITMAP_DIB_HEADER_LENGTH];
    }BitmapDIBHeader;

    typedef enum{
       PIXEL_TYPE_RGB,
       PIXEL_TYPE_ARGB
    }PIXEL_TYPE;

    // Delare comparison operators for BitmpaDIBHeader
    bool operator == (BitmapDIBHeader & h0, BitmapDIBHeader & h1);
    bool operator != (BitmapDIBHeader & h0, BitmapDIBHeader & h1);

    struct PixelCoord2D
    {
        PixelCoord2D(uint32_t x, uint32_t y):x(x), y(y){};
        uint32_t x;
        uint32_t y;
    };

    typedef uint32_t Color;
    #define COLOR_RED   0x00FF0000
    #define COLOR_GREEN 0x0000FF00
    #define COLOR_BLUE  0x000000FF

    class Bitmap
    {
    public:
        Bitmap() {};
        Bitmap(uint32_t x, uint32_t y, PIXEL_TYPE type);
        Bitmap(std::string const & fileName);
        Bitmap(Bitmap & original, bool copyData);
        Bitmap(BitmapFileHeader &header, BitmapDIBHeader &dIBHeader);
        ~Bitmap();

        bool LoadFromFile(std::string const & fileName);
        bool SaveToFile(std::string const & fileName);


        uint8_t* GetRasterData();   // get pointer to the pixels info
        void     CopyRasterData(uint8_t* data); // copy raster data from external location
        uint32_t GetPixelCount();   // get number of pixels available in picture
        uint8_t GetPixelSize();     // get pixel size in bytes

        PixelCoord2D GetPixelCoord(uint8_t *pixel);
        inline uint32_t GetWidth() { return mDIBHeader.width; }
        inline uint32_t GetPaddedWidth() { return mPaddedWidth; } // Bitmap definition states that every bitmap data line has to be 32b aligned
        inline uint32_t GetHeight() { return mDIBHeader.height; }
        uint32_t GetPixelValue(uint32_t x, uint32_t y);
        // Load multiple pixel values as 32b integers into array. Pixels are read horiozontal-wide.
        uint32_t* GetPixelValues(uint32_t start_x, uint32_t start_y, int count, uint32_t *destination);
        void     SetPixelValue(uint32_t x, uint32_t y, uint32_t value);
        void GetHeader(BitmapFileHeader *headerContainter);
        void GetDIBHeader(BitmapDIBHeader *headerContainer);
        inline uint32_t GetBitmapSize() { return mHeader.fileSize - UME_BITMAP_DIB_HEADER_LENGTH - UME_BITMAP_HEADER_LENGTH; }
        inline uint32_t GetPixelsOffset() { return UME_BITMAP_DIB_HEADER_LENGTH + UME_BITMAP_HEADER_LENGTH; }
  
        void ClearTarget(uint8_t r, uint8_t g, uint8_t b);
        void DrawLine(double r, double theta, Color color);
    private:
        // Parsed data
        BitmapFileHeader mHeader;
        BitmapDIBHeader  mDIBHeader;	

        // Padded width is number of bytes (not pixels!) used to store single bitmap line. This value has to be 32b aligned.
        uint32_t mPaddedWidth;

        // Beginning of the pixel image
        uint8_t *mRasterData;
    };
} // UME

#endif //BITMAP_H_

