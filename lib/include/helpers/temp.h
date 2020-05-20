#ifndef _TEMP_H_
#define _TEMP_H_


//////////////////////////////////////////////////////////////////////
#include <general-config.h>
#include <helpers/util.h>
#include <local/core.h>
//////////////////////////////////////////////////////////////////////

// Return value if the
// directory containing
// temperature files
// was not found
#define NOT_FOUND 0

// Return value if the
// directory containing
// temperature files
// was found
#define FOUND_DIR 1


// Return value if the
// zone containing
// temperature files
// was found
#define FOUND_ZONE 2

// Scale between file
// content and Celcius
//(i.e file 48000 ->
// 48 degrees C)
#define SCALE_TO_C 1000

// opens all temp files we are tracking as indicated by
// cpu_set_t. Returns array of file descriptors mapped to the open
// temperature files. The array is dynamically allocated.
// ncores : Number of cores to get files for
// cpus   : cpu_set_t specifying which which cores' temperature files will be
//          opened
int32_t * getTempFiles(int32_t ncores, cpu_set_t * cpus);

// reads all files that the fds array contains and returns dynamically
// allocated array with the temperatures of each file
// ncores : number of cores/file
// fds    : pointer to array of fds to temp files
float * readNStore(int32_t ncores, int32_t * fds);

// reads temperature of a given fd. Basically this just reads file,
// resets file offset so it can be read again, and returns the content
// as a float
// fd: fd to file to read temp from
float getTemp(int32_t fd);


// finds base path for where temp files can be found returns FOUND if
// was able to find coretemp directory returns NOT_FOUND if it was not
// able to find coretemp directory. Uses the recursive strategy
// described above to find the directory.
// outstr : optional argument if you want to copy the path to another string
//          (current used for creating compile-config.h)
int32_t createTempBasePath(char * outstr);


#endif
