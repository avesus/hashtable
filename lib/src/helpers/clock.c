#include <helpers/clock.h>

// simply allocs ts_stamps as specified by length. We are preallocating
// because allocation time is probably not acceptable for precise
// timing.
ts_tracker
ts_initTracker(int32_t len) {
#ifdef USAGE_CHECK
    if (!len) {
        errdie("invalid length\n");
    }
#endif
    ts_tracker new_tracker;
    new_tracker.len = len;
    new_tracker.idx = 0;
    memset(&new_tracker.start_time, 0, sizeof(struct timespec));
    memset(&new_tracker.end_time, 0, sizeof(struct timespec));
    new_tracker.ts_stamps = (timespec *)mycalloc(len, sizeof(struct timespec));
    return new_tracker;
}


// free all memory
void
ts_freeTracker(ts_tracker tracker) {
    myfree(tracker.ts_stamps);
}

// print memory. Csv flag will print as csv, otherwise a humanreadable
// format
void
ts_printTracker(ts_tracker   tracker,
                int32_t      csv_flag,
                int32_t      csv_header,
                int32_t      include_raw,
                FILE *       outfile,
                const char * header) {
    if (tracker.idx == 0) {
        errdie("No times to print!\n");
    }
    else if (tracker.ts_stamps == NULL) {
        errdie("Cant print: array is NULL\n");
    }
    else if (tracker.idx & 0x1) {
        errdie("Invalid number of times to print: %d\n", tracker.idx);
    }
    int32_t    ntimes = tracker.idx >> 1;
    uint64_t * difs   = (uint64_t *)mycalloc(ntimes, sizeof(uint64_t));

    // assuming all evens are start time, all odds are end times for a
    // given timing window
    for (int32_t i = 0; i < ntimes; i++) {
        difs[i] = ns_diff(tracker.ts_stamps[(i << 1) + 1],
                          tracker.ts_stamps[(i << 1)]);
    }
    if (outfile == NULL) {
        outfile = stdout;
    }
    const char * units = unit_to_str(ptype);
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
                units,
                ntimes,
                unit_convert(getMin(difs, ntimes), ptype),
                unit_convert(getMax(difs, ntimes), ptype),
                unit_convert(getMean(difs, ntimes), ptype),
                unit_convert(getMedian(difs, ntimes), ptype),
                unit_convert(getSD(difs, ntimes), ptype),
                unit_convert(getVar(difs, ntimes), ptype));
        if (include_raw) {
            for (int32_t i = 0; i < ntimes; i++) {
                fprintf(outfile,
                        ",%.3lf",
                        unit_convert((double)difs[i], ptype));
            }
        }
        fprintf(outfile, "\n");
    }
    else {
        fprintf(outfile, "%s -> {\n", header);
        fprintf(outfile, "\tStart : %lu ns\n", to_nsecs(tracker.start_time));
        fprintf(outfile, "\tEnd   : %lu ns\n", to_nsecs(tracker.end_time));
        fprintf(outfile, "\tUnits : %s\n", units);
        fprintf(outfile, "\tN     : %d\n", ntimes);
        fprintf(outfile,
                "\tMin   : %.3lf %s\n",
                unit_convert(getMin(difs, ntimes), ptype),
                units);
        fprintf(outfile,
                "\tMax   : %.3lf %s\n",
                unit_convert(getMax(difs, ntimes), ptype),
                units);
        fprintf(outfile,
                "\tMean  : %.3lf %s\n",
                unit_convert(getMean(difs, ntimes), ptype),
                units);
        fprintf(outfile,
                "\tMed   : %.3lf %s\n",
                unit_convert(getMedian(difs, ntimes), ptype),
                units);
        fprintf(outfile,
                "\tSD    : %.3lf %s\n",
                unit_convert(getSD(difs, ntimes), ptype),
                units);
        fprintf(outfile,
                "\tVar   : %.3lf %s\n",
                unit_convert(getVar(difs, ntimes), ptype),
                units);
        if (include_raw) {
            fprintf(stderr, "\tData  : [");
            for (int32_t i = 0; i < (ntimes - 1); i++) {
                fprintf(outfile,
                        "%.3lf, ",
                        unit_convert((double)difs[i], ptype));
            }
            fprintf(outfile,
                    "%.3lf]\n",
                    unit_convert((double)difs[ntimes - 1], ptype));
        }
        fprintf(outfile, "\t}\n\n");
    }
    myfree(difs);
}


// set start time for a trial. This is for coordinating with temp/freq
// data
void
ts_startTrial(ts_tracker * tracker) {
    clock_gettime(CLOCK_MONOTONIC, &tracker->start_time);
}

// set end time for a trial. This is for coordinating with temp/freq
// data
void
ts_endTrial(ts_tracker * tracker) {
    clock_gettime(CLOCK_MONOTONIC, &tracker->end_time);
}

// stores time int32_to current idx value of tracker
void
ts_takeTime(ts_tracker * tracker) {
#ifdef DEBUG_MODE
    if (tracker->idx >= tracker->len) {
        errdie("Cant add time: index out of bounds (%d/%d)\n",
               tracker->idx,
               tracker->len);
    }
    if (tracker->ts_stamps == NULL) {
        errdie("Cant add time: array is NULL\n");
    }
#endif
    clock_gettime(CLOCK_MONOTONIC, tracker->ts_stamps + tracker->idx++);
}
