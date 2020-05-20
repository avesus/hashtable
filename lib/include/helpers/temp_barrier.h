#ifndef _TEMP_BARRIER_H_
#define _TEMP_BARRIER_H_

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

//////////////////////////////////////////////////////////////////////
#include <general-config.h>
#include <helpers/temp.h>
#include <helpers/util.h>
//////////////////////////////////////////////////////////////////////

// package to monitor temperature of cores in a multicore processor.
// Specific to linux.  Tested on Ubuntu 18.04 & Ubuntu 16.04.


// for initial allocation if
// STORE_DATA is defined and
// num trials is not
#define TRIALS_GUESS 32

// for initial allocation if
// STORE_DATA is defined and
// timeout_sec is not defined
#define MAX_SLEEPS_GUESS 256

// define to store data after
// each temperature checl
#define STORE_DATA

// set in lower bit of
// init_temps if allocation was
// done in init (i.e init_temps
// was not set in attr)
#define INTERNAL_ALLOC 0x1

#define local_get_ptr(X) ((float *)low_bits_get_ptr((X)))


#define MAINTAIN(X, Y)  ((uint64_t)(((double)(X)) / (Y)))
#define MIN_GRAN_FACTOR 100  // After consecutives
// sleeps/spinning will
// increase duration until
// MIN_GRAN_FACTOR
typedef enum enforcer_modes {
    BELOW_THRESH = 0,
    WITHIN_RANGE = 1
} enforcer_modes;


// temp barrier struct. Note that while this is meant to coordinate
// multiple threads. Most of the functionality is meant to be called by
// a single thread
typedef struct temp_barrier {

    // cpus that are being regulated
    cpu_set_t * cpus;


    // number of cpus that are
    // being regulated. Not a
    // necessary argument but
    // CPU_COUNT() is not free and
    // it aligns with the next
    // float
    int32_t ncores;

    // allowed variance to reach
    // required cooldown. I.e if
    // init temperature is c, to
    // meet threshold current temp
    // must be bellow c *
    // allowed_variance
    float allowed_variance;


    // enforcer mode, either
    // ensures within range (i.e if
    // temp drops to much will
    // increase) or just ensures
    // falls below threshold
    enforcer_modes enf_mode;

    // init temperature for
    // cooldown threshold
    float * init_temps;

    // file descriptors to
    // temperature files for each
    // cpu being regulated
    int32_t * fds;

    // sleep time between reaching
    // if temperature threshold has
    // been met
    int32_t usec_sleep_time;

    // timeout in seconds. If
    // temperatures have not
    // dropped below threshold
    //(cooled off) by timeout
    // error occurs
    int32_t timeout_sec;

    // Todo fix this stuff
#ifdef STORE_DATA

    // time stamp for when this data was gotten
    struct timespec ** timestamps;

    // this is by definition not on a critical path so
    // will just live with amortized O(N)
    float *** temps;

    // lengths[i] stores amount of temps collected at
    // temps[i]
    int32_t * index;

    // bounds[i] stores current allocated space for temps[i]
    int32_t * bounds;

    // current max trials allocated
    int32_t trial_bound;

    // cur trial number we are collecting for
    int32_t cur_trial;
#endif

    // actual barrier that all this
    // temperature bullshit wraps
    pthread_barrier_t barrier;
} temp_barrier;

#define NO_TIMEOUT (~0U)

// default values for temp_barrier_attr fields.


// default is no timeout
#define DEFAULT_TIMEOUT (NO_TIMEOUT)

// if store data no trial max
#define DEFAULT_TRIALS (-1)

// default sleep time is once per sec
#define DEFAULT_SLEEP us_per_sec

// default variance is 1.1
#define DEFAULT_VARIANCE 1.1

// default enforcer mode
#define DEFAULT_ENF_MODE BELOW_THRESH

// attributes struct is for setting various fields in
// temp_barrier. This is entirely for a cleaner api. Variable names in
// attr correspond 1-1 with variable names in temp_barrier
typedef struct temp_barrier_attr {
    cpu_set_t *    cpus;              // to set cpus
    int32_t        usec_sleep_time;   // to set sleep time
    uint32_t       timeout_sec;       // to set timeout
    float          allowed_variance;  // to set allowed_variance
    float *        init_temps;        // to set init_temps
    enforcer_modes enf_mode;          // to set enforcer mode

    // Todo fix this stuff
#ifdef STORE_DATA
    int32_t ntrials;
#endif
} temp_barrier_attr;

// initialize attr struct with reasonable values
void initAttr(temp_barrier_attr * attr  // attr struct to be
                                        // initialized
);


// This set of functions initializes a specific field in an
// temp_barrier_attr struct. I bet you can guess which one each
// function initializes
void attrSetSleepUS(temp_barrier_attr * attr, int32_t usec_sleep_time);
void attrSetTimeoutSec(temp_barrier_attr * attr, uint32_t timeout_sec);
void attrSetAllowedVariance(temp_barrier_attr * attr, float allowed_variance);
void attrSetInitTemps(temp_barrier_attr * attr, float * init_temps);
void attrSetTrials(temp_barrier_attr * attr, int32_t ntrials);
void attrSetEnforcer(temp_barrier_attr * attr, enforcer_modes mode);
void attrSetCPUS(temp_barrier_attr * attr, cpu_set_t * cpus);


// initializes new temp barrier. Returns ptr to dynamically allocated
// region of memory storing the barrier.
// barrier_num      : for initializing pthread_barrier_t. Is number of waits for
//                    barrier to expect for releasing.
// attr             : attr struct with values for initializing the barrier
temp_barrier * initTempBarrier(int32_t barrier_num, temp_barrier_attr * attr);


// the api function to do barrier synchronization. Coordinator needs to
// be set by 1 thread and 1 thread only to actually do the call to
// regulate temp. Just about nothing in the temp barrier is safe for
// concurrency
// tmp_barirer: barrier for sleeping/cooling off.
// coordinator: boolean to say whether this call should actually make the
//              enforce temperature call or just hang at pthread_barrier. Must
//              be set by at least one of the waiting threads (and example might
//              be passing (!tid)
void tempBarrierWait(temp_barrier * tmp_barrier, int32_t coordinator);

// frees all memory barrier that will have all the dynamic memory assosiated
// with it free (this includes freeing tmp_barrier)
// tmp_barrier: barrier being freed;
void freeTempBarrier(temp_barrier * tmp_barrier);

// print temp barrier data, really does nothing if STORE_DATA is not
// defined
// tmp_barrier : barrier whose data is being printed
// outfile     : file to write the data to (can be stdout or stderr)
void printBarrierData(temp_barrier * tmp_barrier, FILE * outfile);
#endif
