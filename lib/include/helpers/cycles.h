#ifndef _CYCLES_H_
#define _CYCLES_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////
#include <helpers/util.h>
//////////////////////////////////////////////////////////////////////


/*
From https://en.wikipedia.org/wiki/Time_Stamp_Counter

"Relying on the TSC also reduces portability, as other
processors may not have a similar feature. Recent Intel processors
include a constant rate TSC (identified by the
kern.timecounter.invariant_tsc sysctl on FreeBSD or by the
"constant_tsc" flag in Linux's /proc/cpuinfo). With these processors,
the TSC ticks at the processor's nominal frequency, regardless of the
actual CPU clock frequency due to turbo or power saving states. Hence
TSC ticks are counting the passage of time, not the number of CPU
clock cycles elapsed."

Since this is so precise I decided against implementing any structures
to track for the programmer as coding it generically would cause some
overhead.
*/

typedef struct tsc_tracker {
    uint64_t *      tsc_stamps;
    int32_t         len;
    int32_t         idx;
    struct timespec start_time;
    struct timespec end_time;
} tsc_tracker;


// initializes a new tracker with length len
tsc_tracker tsc_initTracker(int32_t len);

// frees memory assosiated with a given tracker
void tsc_freeTracker(tsc_tracker tracker);

// prints tracker summary. csv_flag will print as csv, header will
// include a generic csv header.
void tsc_printTracker(tsc_tracker  tracker,
                      int32_t      csv_flag,
                      int32_t      csv_header,
                      int32_t      include_raw,
                      FILE *       outfile,
                      const char * header);

// get current cycle count
uint64_t grabTSC();

// adds new time to tracker
void tsc_takeTime(tsc_tracker * tracker);

// set at start of trial
void tsc_startTrial(tsc_tracker * tracker);

// set at end of trial
void tsc_endTrial(tsc_tracker * tracker);

// stores a tsc time in tracker. This allows to time to be taken before
// overhead of adding it to struct
void tsc_storeTime(uint64_t t, tsc_tracker * tracker);
#endif
