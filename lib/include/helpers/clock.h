#ifndef _CLOCK_H_
#define _CLOCK_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////
#include <general-config.h>
#include <helpers/util.h>
//////////////////////////////////////////////////////////////////////


/*
For timing with clock_gettime. Units is determined by "ptype". This
should be used if we are coordinating times across cores as tsc are
not guranteed to be constant between processors.
 */


// lightweight struct for storing times
typedef struct ts_tracker {
    struct timespec * ts_stamps;
    int32_t           len;
    int32_t           idx;
    struct timespec   start_time;
    struct timespec   end_time;
} ts_tracker;


// initializes a new tracker with length len
ts_tracker ts_initTracker(int32_t len);

// frees memory assosiated with a given tracker
void ts_freeTracker(ts_tracker tracker);

// prints tracker summary. csv_flag will print as csv, header will
// include a generic csv header.
void ts_printTracker(ts_tracker   tracker,
                     int32_t      csv_flag,
                     int32_t      csv_header,
                     int32_t      include_raw,
                     FILE *       outfile,
                     const char * header);


// adds new time to tracker
void ts_takeTime(ts_tracker * tracker);

// set at start of trial
void ts_startTrial(ts_tracker * tracker);

// set at end of trial
void ts_endTrial(ts_tracker * tracker);


#endif
