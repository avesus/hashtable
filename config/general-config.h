#ifndef _GENERAL_CONFIG_H_
#define _GENERAL_CONFIG_H_

#if __has_include(<compile-config.h>)
#include <compile-config.h>
#endif

#ifndef NCORES
#define NCORES ((uint64_t)sysconf(_SC_NPROCESSORS_ONLN))
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE ((uint64_t)sysconf(_SC_PAGE_SIZE))
#endif

#ifndef L1_CACHE_LINE_SIZE
#ifdef L1_LOG_CACHE_LINE_SIZE

#error "No idea how log(cache line size) was created w/o cache line size..."
BROKEN //this is just to cause compiler error

#endif
#define L1_CACHE_LINE_SIZE 64
#define L1_LOG_CACHE_LINE_SIZE 6
#endif

#ifndef TEMP_PATH
#define TEMP_PATH ""
#endif
#ifndef TEMP_PATH_TYPE
#define TEMP_PATH_TYPE 0
#endif

//#define STACK_TRACE_ON
//#define FRAME_DEBUGGER
#define DEBUG_MODE
#define USAGE_CHECK

#define ERROR_VERBOSE 0
#define LOW_VERBOSE 1
#define MED_VERBOSE 2
#define HIGH_VERBOSE 3

#if defined ERROR_VERBOSE || LOW_VERBOSE || defined MED_VERBOSE || defined HIGH_VERBOSE
#define VERBOSE
#include <stdint.h>
extern int32_t verbose;
#endif


#define SMALL_READ_LEN 16


#define SMALL_PATH_LEN 128
#define MED_PATH_LEN 256
#define BIG_PATH_LEN 512


#define SMALL_BUF_LEN 32
#define MED_BUF_LEN 128
#define BIG_BUF_LEN 512


#define ptype ms //default unit for timing output


// gcc has typeof, c++ equivilent is decltype
#ifdef __cpp_attributes
#define CPP_MODE
#else
#define C_MODE
#endif


#define INT32_MASK 0xffffffff

#endif
