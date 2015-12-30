#ifndef BASIC_TYPES_H_
#define BASIC_TYPES_H_

#ifdef _MSC_VER

#include <stdint.h>
/*
#define int8_t __int8
#define int16_t __int16
#define int32_t __int32
#define int64_t __int64

#define uint8_t unsigned __int8
#define uint16_t unsigned __int16
#define uint32_t unsigned __int32
#define uint64_t unsigned __int64
*/
#else
#include <inttypes.h>
#endif //_MSC_VER


#endif
