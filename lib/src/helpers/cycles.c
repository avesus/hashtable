#include <helpers/cycles.h>


// simply allocs tsc_stamps as specified by length. We are
// preallocating because allocation time is almost definetly not
// acceptable overhead if need the precision of rdtsc
tsc_tracker
tsc_initTracker(int32_t len) {
#ifdef USAGE_CHECK
    if (!len) {
        errdie("invalid length\n");
    }
#endif
    tsc_tracker new_tracker;
    new_tracker.len = len;
    new_tracker.idx = 0;
    memset(&new_tracker.start_time, 0, sizeof(struct timespec));
    memset(&new_tracker.end_time, 0, sizeof(struct timespec));
    new_tracker.tsc_stamps = (uint64_t *)mycalloc(len, sizeof(uint64_t));
    return new_tracker;
}


// free memory
void
tsc_freeTracker(tsc_tracker tracker) {
    myfree(tracker.tsc_stamps);
}

// prints data, same way as in clocks
void
tsc_printTracker(tsc_tracker  tracker,
                 int32_t      csv_flag,
                 int32_t      csv_header,
                 int32_t      include_raw,
                 FILE *       outfile,
                 const char * header) {
#ifdef USAGE_CHECK
    if (tracker.idx == 0) {
        errdie("No cycles to print!\n");
    }
    if (tracker.tsc_stamps == NULL) {
        errdie("Cant print: array is NULL\n");
    }
    DBG_ASSERT(tracker.idx && (!(tracker.idx & 0x1)),
               "Invalid number of times to print: %d\n",
               tracker.idx);
#endif
    int32_t    ntimes = tracker.idx >> 1;
    uint64_t * difs   = (uint64_t *)mycalloc(ntimes, sizeof(uint64_t));

    for (int32_t i = 0; i < ntimes; i++) {
        difs[i] =
            tracker.tsc_stamps[(i << 1) + 1] - tracker.tsc_stamps[(i << 1)];
    }
    if (outfile == NULL) {
        outfile = stdout;
    }
    if (csv_flag) {
        if (csv_header) {
            fprintf(outfile,
                    "HEADER,START,END,UNITS,N,MIN,MAX,MEAN,MEDIAN,SD,VAR");
            if (include_raw) {
                fprintf(outfile, ",RAW[0-%d]", ntimes - 1);
            }
            fprintf(outfile, "\n");
        }
        fprintf(outfile,
                "%s,%lu,%lu,%s,%d,%.3lf,%.3lf,%.3lf,%.3lf,%.3lf,%.3lf",
                header,
                to_nsecs(tracker.start_time),
                to_nsecs(tracker.end_time),
                "cycles",
                ntimes,
                getMin(difs, ntimes),
                getMax(difs, ntimes),
                getMean(difs, ntimes),
                getMedian(difs, ntimes),
                getSD(difs, ntimes),
                getVar(difs, ntimes));
        if (include_raw) {
            for (int32_t i = 0; i < ntimes; i++) {
                fprintf(outfile, ",%lu", difs[i]);
            }
        }
        fprintf(outfile, "\n");
    }
    else {
        fprintf(outfile, "%s -> {\n", header);
        fprintf(outfile, "\tStart : %lu ns\n", to_nsecs(tracker.start_time));
        fprintf(outfile, "\tEnd   : %lu ns\n", to_nsecs(tracker.end_time));
        fprintf(outfile, "\tUnits : %s\n", "cycles");
        fprintf(outfile, "\tN     : %d\n", ntimes);
        fprintf(outfile, "\tMin   : %.3lf cycles\n", getMin(difs, ntimes));
        fprintf(outfile, "\tMax   : %.3lf cycles\n", getMax(difs, ntimes));
        fprintf(outfile, "\tMean  : %.3lf cycles\n", getMean(difs, ntimes));
        fprintf(outfile, "\tMed   : %.3lf cycles\n", getMedian(difs, ntimes));
        fprintf(outfile, "\tSD    : %.3lf cycles\n", getSD(difs, ntimes));
        fprintf(outfile, "\tVar   : %.3lf cycles\n", getVar(difs, ntimes));
        if (include_raw) {
            fprintf(stderr, "\tData  : [");
            for (int32_t i = 0; i < (ntimes - 1); i++) {
                fprintf(outfile, "%lu, ", difs[i]);
            }
            fprintf(outfile, "%lu ]\n", difs[ntimes - 1]);
        }
        fprintf(outfile, "\t}\n\n");
    }
    myfree(difs);
}

// set start time for a trial. This is for coordinating with temp/freq
// data
void
tsc_startTrial(tsc_tracker * tracker) {
    clock_gettime(CLOCK_MONOTONIC, &tracker->start_time);
}

// set end time for a trial. This is for coordinating with temp/freq
// data
void
tsc_endTrial(tsc_tracker * tracker) {
    clock_gettime(CLOCK_MONOTONIC, &tracker->end_time);
}


// just a wrapper for getting time with rdtsc
uint64_t
grabTSC() {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (((uint64_t)lo) | (((uint64_t)hi) << 32));
}


// store a tsc value provided
void
tsc_storeTime(uint64_t t, tsc_tracker * tracker) {
#ifdef DEBUG_MODE
    if (tracker->idx >= tracker->len) {
        errdie("Cant add time: index out of bounds (%d/%d)\n",
               tracker->idx,
               tracker->len);
    }
    if (tracker->tsc_stamps == NULL) {
        errdie("Cant add time: array is NULL\n");
    }
#endif
    tracker->tsc_stamps[tracker->idx++] = t;
}


// gets tsc and stores in current idx
void
tsc_takeTime(tsc_tracker * tracker) {
#ifdef DEBUG_MODE
    if (tracker->idx >= tracker->len) {
        errdie("Cant add time: index out of bounds (%d/%d)\n",
               tracker->idx,
               tracker->len);
    }
    if (tracker->tsc_stamps == NULL) {
        errdie("Cant add time: array is NULL\n");
    }
#endif
    tracker->tsc_stamps[tracker->idx++] = grabTSC();
}
