#include <helpers/temp_barrier.h>

typedef struct core_enforcer_args {
    pthread_mutex_t *   fd_lock;
    volatile uint64_t * done;
    int32_t             base_sleep_time;
    int32_t             fd;
    float               init_temp;
    float               allowed_variance;
    int32_t             core_num;
} core_enforcer_args;

// stores temperature data after a given test
// tmp_barrier  : temp barrier to store data in
// temp_line    : most recent set of temperature reads
#ifdef STORE_DATA
static void
storeData(temp_barrier * tmp_barrier, float * temp_line) {

    int32_t  cur_trial       = tmp_barrier->cur_trial;
    int32_t  usec_sleep_time = tmp_barrier->usec_sleep_time;
    uint32_t timeout_sec     = tmp_barrier->timeout_sec;
    if (cur_trial >= tmp_barrier->trial_bound) {
        int32_t max_sleep_alloc =
            (timeout_sec != (NO_TIMEOUT))
                ? ((us_per_sec * (uint64_t)timeout_sec) / usec_sleep_time)
                : MAX_SLEEPS_GUESS;

        int32_t new_trial_bound = tmp_barrier->trial_bound << 1;
        tmp_barrier->temps =
            (float ***)realloc(tmp_barrier->temps,
                               new_trial_bound * sizeof(float **));
        tmp_barrier->bounds =
            (int32_t *)realloc(tmp_barrier->bounds,
                               new_trial_bound * sizeof(int32_t));
        tmp_barrier->index =
            (int32_t *)realloc(tmp_barrier->index,
                               new_trial_bound * sizeof(int32_t));

        tmp_barrier->timestamps = (struct timespec **)realloc(
            tmp_barrier->timestamps,
            new_trial_bound * sizeof(struct timespec *));

        for (int32_t i = tmp_barrier->trial_bound; i < new_trial_bound; i++) {
            tmp_barrier->temps[i] =
                (float **)mycalloc(max_sleep_alloc, sizeof(float *));
            tmp_barrier->timestamps[i] =
                (struct timespec *)mycalloc(max_sleep_alloc,
                                            sizeof(struct timespec));

            tmp_barrier->bounds[i] = max_sleep_alloc;
            tmp_barrier->index[i]  = 0;
        }

        tmp_barrier->trial_bound = new_trial_bound;
    }

    int32_t len   = tmp_barrier->bounds[cur_trial];
    int32_t index = tmp_barrier->index[cur_trial];
    clock_gettime(CLOCK_MONOTONIC, &tmp_barrier->timestamps[cur_trial][index]);
    tmp_barrier->temps[cur_trial][index] = temp_line;
    tmp_barrier->index[cur_trial]++;

    if ((index + 1) >= len) {
        int32_t new_len = len << 1;

        tmp_barrier->temps[cur_trial] =
            (float **)realloc(tmp_barrier->temps[cur_trial],
                              new_len * sizeof(float *));

        tmp_barrier->timestamps[cur_trial] =
            (struct timespec *)realloc(tmp_barrier->timestamps[cur_trial],
                                       new_len * sizeof(struct timespec));

        // unnecissary but 0 memory is nicer
        memset(tmp_barrier->temps[cur_trial] + len, 0, len * sizeof(float *));

        memset(tmp_barrier->timestamps[cur_trial] + len,
               0,
               len * sizeof(struct timespec));

        tmp_barrier->bounds[cur_trial] = new_len;
    }
}
#endif

// returns first set core in cpu_set (very slow, for debug purposes
// only)
static uint32_t
getSetCore(cpu_set_t * check_cores) {
    uint64_t * cpu_mask = (uint64_t *)check_cores->__bits;
    uint64_t * end      = cpu_mask + ((NCORES + (sizeof_bits(uint64_t) - 1)) /
                                 sizeof_bits(uint64_t));

    uint32_t ret = 0;
    while (!(*cpu_mask) && cpu_mask < end) {
        cpu_mask++;
        ret += sizeof_bits(uint64_t);
    }
    DBG_ASSERT(cpu_mask != end, "No core set in cpu set\n");
    ret += ff1_64_ctz_nz(*cpu_mask);
    return ret;
}

// returns next cpu in the set after n
// cpus    : cpuset of current active cpus
// n       : last cpu that was returned.
static uint32_t
getNextCPU(cpu_set_t * cpus, int32_t n) {

    n++;
    // basically a cpu_set_t is a just a 1024 bit bitvector. This bitcount
    uint64_t * cpu_mask_start = (uint64_t *)cpus->__bits;
    cpu_mask_start += n / sizeof_bits(uint64_t);

    uint32_t i         = 0;
    uint64_t temp_mask = *(cpu_mask_start);
    temp_mask &= ~(((1UL) << (n % sizeof_bits(uint64_t))) - 1);

    // in practice ever hitting this loop would be suprising
    while (!temp_mask) {
        i++;
        DBG_ASSERT(((uint64_t)(cpu_mask_start + i)) <= ((uint64_t)(cpus + 1)),
                   "Error: Accessing invalid memory searching for next "
                   "CPU index: %p > %p\n",
                   (cpu_mask_start + i),
                   (cpus + 1));

        temp_mask = *(cpu_mask_start + i);
    }

    int32_t next_core = ff1_64_ctz_nz(temp_mask);

    next_core += ((n / sizeof_bits(uint64_t)) + i) * sizeof_bits(uint64_t);
    return next_core;
}

#define IS_IN_RANGE 0

// Returns IS_IN_RANGE if temp is within temp range. If not returns
// current temp - init_temp as integether (rounds up to 1 as temps are
// floats and return is int). Can check positive/negative to see if to
// hot/cold if return is not in range
// init_temp         : initial temperature for the core
// allowed_variance  : how close current temperature and init temp need to be
// cur_temp          : variable to store current temperature in.
// fd                : file descriptor with opened temperature file
// fd_lock           : lock to avoid race condition on reading from file
static int32_t
inRange(float             init_temp,
        float             allowed_variance,
        float *           cur_temp,
        int32_t           fd,
        pthread_mutex_t * fd_lock) {

    pthread_mutex_lock(fd_lock);
    *cur_temp = getTemp(fd);
    pthread_mutex_unlock(fd_lock);

    // temp above init
    if (*cur_temp >= init_temp) {

        // within threshold
        if (*cur_temp < (init_temp * allowed_variance)) {
            return IS_IN_RANGE;
        }
        uint32_t ret = *cur_temp - init_temp;
        return ret | (ret == 0);
    }
    else {  // temp below init
        if (*cur_temp >
            (init_temp / allowed_variance)) {  // temp within threshold
            return IS_IN_RANGE;
        }
        uint32_t ret = *cur_temp - init_temp;
        return ret - (ret == 0);
    }
}

// Will try and get the temperature of each core being tracked within range of
// its initial value. For some cores this might mean raising its temperature
// (done via spinning). For others this might mean lowering the temperature
// (done via sleeping). This will be called by a seperate thread for each core.
// targs    : void * cast of core_enforce_args struct.
static void *
perCoreEnforcer(void * targs) {
    core_enforcer_args * args             = (core_enforcer_args *)targs;
    pthread_mutex_t *    fd_lock          = args->fd_lock;
    volatile uint64_t *  done             = args->done;
    int32_t              base_sleep_time  = args->base_sleep_time;
    int32_t              fd               = args->fd;
    float                init_temp        = args->init_temp;
    float                allowed_variance = args->allowed_variance;
    int32_t              core_num         = args->core_num;

    int32_t max_above = 0, max_below = 0;

#ifdef DEBUG_MODE
    cpu_set_t check_cores;
    CPU_ZERO(&check_cores);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &check_cores);
    DBG_ASSERT(CPU_COUNT(&check_cores) == 1,
               "CPU_COUNT within core enforcer thread invalid (%d)\n",
               CPU_COUNT(&check_cores));
    DBG_ASSERT(CPU_ISSET(core_num, &check_cores),
               "Incorrect CPU set\n\tExpec:  %d\n\tActual: %d)\n",
               core_num,
               getSetCore(&check_cores));
#endif

    int32_t consecutives = 0;
    int32_t last_todo    = 0;

    float          cur_temp    = 0.0;
    float          scale_denum = init_temp;
    volatile float sink        = init_temp;

    // continue until done is triggered when all threads are within range. The
    // reason this needs to continue for a given thread even after it has
    // reached it's desired temperature is that other threads might still be
    // cooling/heating so this thread must maintain its temperature (if it just
    // returns it will effectively sleep and cool off its core)
    while (!(*done)) {
        int32_t todo =
            inRange(init_temp, allowed_variance, &cur_temp, fd, fd_lock);

        consecutives = (todo == last_todo) ? (consecutives + 1) : 1;
        last_todo    = todo;

        PRINT(HIGH_VERBOSE,
              "Core (%d) -> %.3f within [%.3f, %.3f] -> %d\n",
              core_num,
              cur_temp,
              init_temp / allowed_variance,
              init_temp * allowed_variance,
              todo);

        if (todo == IS_IN_RANGE) {
            PRINT(HIGH_VERBOSE,
                  "\t(%d) IS_IN_RANGE: sleeping %d us\n",
                  core_num,
                  base_sleep_time);
            sink *= sink;
            usleep(MAINTAIN(base_sleep_time, scale_denum));
        }
        else if (todo < 0) {  // need to heat up (cur temp - init temp < 0)

            if (last_todo == IS_IN_RANGE) {
                scale_denum += 1.0;
            }

            todo = (~todo) + 1;
            todo /= SCALE_TO_C;
            max_below = MAX(max_below, todo);
            DBG_ASSERT(
                max_below - todo >= 0,
                "Something went wrong with MAX calculation, overflow likely\n");

            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            clock_gettime(CLOCK_MONOTONIC, &end);

            // idea is as we get closer start user finer granularity, might
            // need to add some constants here or change the scaling.
            uint32_t spin_time = base_sleep_time / ((max_below + 1) - todo);

            PRINT(HIGH_VERBOSE,
                  "\t(%d) BELOW_RANGE: spinning(%d) (%d / ((%d + 1) - %d) -> "
                  "%d us\n",
                  core_num,
                  MIN(consecutives, MIN_GRAN_FACTOR),
                  base_sleep_time,
                  max_below,
                  todo,
                  spin_time);
            while (us_diff(end, start) < spin_time) {
                for (int c = 0; c < MIN(consecutives, MIN_GRAN_FACTOR); c++) {
                    sink *= sink;
                }
                clock_gettime(CLOCK_MONOTONIC, &end);
            }
        }
        else {  // need to cool down (cur temp - init temp > 0)

            if (last_todo == IS_IN_RANGE) {
                if (scale_denum > 2.0) {
                    scale_denum -= 1.0;
                }
            }
            todo /= 1000;
            max_above = MAX(max_above, todo);
            DBG_ASSERT(
                max_above - todo >= 0,
                "Something went wrong with MAX calculation, overflow likely\n");
            PRINT(HIGH_VERBOSE,
                  "\t(%d) ABOVE_RANGE: sleeping %d * (%d / ((%d + 1) - %d) -> "
                  "%d us\n",
                  core_num,
                  base_sleep_time,
                  MIN(consecutives, MIN_GRAN_FACTOR),
                  max_above,
                  todo,
                  (MIN(consecutives, MIN_GRAN_FACTOR) * base_sleep_time) /
                      ((max_above + 1) - todo));
            usleep((MIN(consecutives, MIN_GRAN_FACTOR) * base_sleep_time) /
                   ((max_above + 1) - todo));
        }
    }
    return NULL;
}

// ensures temperature is within a certain range based on temps and
// allowed variance.
// tmp_barrier : barrier we are waiting on
static void
enforceWithinRange(temp_barrier * tmp_barrier) {

    // store barrier variables in locals for cleanliness
    uint32_t timeout_sec     = tmp_barrier->timeout_sec;
    int32_t  usec_sleep_time = tmp_barrier->usec_sleep_time;

    int32_t   ncores = tmp_barrier->ncores;
    int32_t * fds    = tmp_barrier->fds;

    float   allowed_variance = tmp_barrier->allowed_variance;
    float * init_temps       = local_get_ptr(tmp_barrier->init_temps);


    pthread_attr_t attr;
    pthread_attr_init(&attr);

    cpu_set_t new_cpu_set;
    CPU_ZERO(&new_cpu_set);

    pthread_t * tids = (pthread_t *)mycalloc(ncores, sizeof(pthread_t));


    core_enforcer_args * targs =
        (core_enforcer_args *)mycalloc(ncores, sizeof(core_enforcer_args));

    pthread_mutex_t * fd_locks =
        (pthread_mutex_t *)mycalloc(ncores, sizeof(pthread_mutex_t));
    for (int i = 0; i < ncores; i++) {
        if (pthread_mutex_init(fd_locks + i, NULL)) {
            errdie("Error initializing mutex\n");
        }
    }

    volatile uint64_t done = 0;

    uint32_t set_cpu        = (~0);
    uint32_t debug_prev_cpu = (~0);

    uint64_t physical_cores = 0;
    for (int i = 0; i < ncores; i++) {

        set_cpu = getNextCPU(tmp_barrier->cpus, set_cpu);

        DBG_ASSERT(set_cpu != debug_prev_cpu,
                   "getNextCPU returned same value: %d == %d\n",
                   set_cpu,
                   debug_prev_cpu);
        debug_prev_cpu = set_cpu;

        int32_t phys_core = getCoreID(set_cpu);
        if (!(physical_cores & (1 << phys_core))) {
            CPU_SET(set_cpu, &new_cpu_set);

            if (pthread_attr_setaffinity_np(&attr,
                                            sizeof(cpu_set_t),
                                            &new_cpu_set)) {
                errdie("Error setting cpu attr\n");
            }


            physical_cores |= (1 << phys_core);
            targs[i].fd_lock          = fd_locks + phys_core;
            targs[i].done             = &done;
            targs[i].base_sleep_time  = usec_sleep_time / MIN_GRAN_FACTOR;
            targs[i].fd               = fds[i];
            targs[i].init_temp        = init_temps[i];
            targs[i].allowed_variance = allowed_variance;
            targs[i].core_num         = set_cpu;

            mypthread_create(tids + i,
                             &attr,
                             perCoreEnforcer,
                             (void *)(targs + i));
            PRINT(HIGH_VERBOSE,
                  "Thread (%d) -> (%lu, %d)\n",
                  i,
                  tids[i],
                  set_cpu);

            CPU_CLR(set_cpu, &new_cpu_set);
        }
        else {
            PRINT(MED_VERBOSE,
                  "Logical Core (%d) already being enforced by Physical (%d)\n",
                  set_cpu,
                  phys_core);
            targs[i].fd_lock = fd_locks + phys_core;
        }
#ifdef DEBUG_MODE
        DBG_ASSERT(CPU_COUNT(&new_cpu_set) == 0,
                   "Unknown cpu still set after loop (%d)\n",
                   i);
#endif
    }
    if (pthread_attr_destroy(&attr)) {
        errdie("Error destroying attr\n");
    }

    int32_t         i = 0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // continue looping until timeout or if timeout is not set (-1)
    // continue indefinetly
    while ((s_diff(end, start) < timeout_sec) ||
           (timeout_sec == (NO_TIMEOUT))) {

        // if store data read line. This will be removed
#ifdef STORE_DATA
        // we are being inefficient here (double iterating/reading but at
        // this point32_t in the program should be nothing running by
        // definition
        uint64_t temp_phys_cores = physical_cores;
        while (temp_phys_cores) {
            i = ff1_64_ctz_nz(temp_phys_cores);
            temp_phys_cores ^= (1UL) << i;
            pthread_mutex_lock(fd_locks + i);
        }
        float * store_temps = readNStore(ncores, fds);

        temp_phys_cores = physical_cores;
        while (temp_phys_cores) {
            i = ff1_64_ctz_nz(temp_phys_cores);
            temp_phys_cores ^= (1UL) << i;
            pthread_mutex_unlock(fd_locks + i);
        }

        PRINT(HIGH_VERBOSE, "\nAll: ");
    #ifdef VERBOSE
        for (i = 0; i < ncores; i++) {
            PRINT(HIGH_VERBOSE,
                  "(%.3f vs %.3f) ",
                  init_temps[i],
                  store_temps[i]);
        }
    #endif
        PRINT(HIGH_VERBOSE, "\n\t")


        storeData(tmp_barrier, store_temps);

#endif

#ifdef VERBOSE
        clock_gettime(CLOCK_MONOTONIC, &end);
#endif
        PRINT(MED_VERBOSE, "Time(us): %lu", us_diff(end, start));
        for (i = 0; i < ncores; i++) {

            float cur_temp = 0.0;
            if (inRange(init_temps[i],
                        allowed_variance,
                        &cur_temp,
                        fds[i],
                        targs[i].fd_lock) != IS_IN_RANGE) {
                PRINT(MED_VERBOSE, "(%.3f vs %.3f) ", init_temps[i], cur_temp);
                break;
            }
            PRINT(MED_VERBOSE, "(%.3f vs %.3f) ", init_temps[i], cur_temp);
        }

        // all are in range so cleanup and return
        if (i == ncores) {
            PRINT(MED_VERBOSE, "Passed\n");
            done = 1;  // done tell all threads to stop looping

            for (int c = 0; c < ncores; c++) {
                pthread_join(tids[c], NULL);
                if (pthread_mutex_destroy(fd_locks + c)) {
                    errdie("Error destroying pthread mutex\n");
                }
            }

            myfree(tids);
            myfree(targs);
            myfree(fd_locks);
            return;
        }
        PRINT(MED_VERBOSE, "Failed\n");


        // sleep between checks so that cpu usage on whatever core this is
        // on doesnt impact threads meaningfully
        usleep(usec_sleep_time);

        // get time for timeout check
        clock_gettime(CLOCK_MONOTONIC, &end);
    }


    // throw error if never cooled off
    errdie("Error, temperatures never cooled off\n");
}

// regulates temperature so that it falls below a threshold determined
// temps and allowed variance.
// tmp_barrier : barrier we are waiting on
static void
enforceBelowThresh(temp_barrier * tmp_barrier) {

    // store barrier variables in locals for cleanliness
    uint32_t timeout_sec     = tmp_barrier->timeout_sec;
    int32_t  usec_sleep_time = tmp_barrier->usec_sleep_time;

    int32_t   ncores = tmp_barrier->ncores;
    int32_t * fds    = tmp_barrier->fds;

    float   allowed_variance = tmp_barrier->allowed_variance;
    float * init_temps       = local_get_ptr(tmp_barrier->init_temps);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // continue looping until timeout or if timeout is not set (-1)
    // continue indefinetly
    while ((s_diff(end, start) < timeout_sec) || (timeout_sec == NO_TIMEOUT)) {

        // if store data read line. This will be removed
#ifdef STORE_DATA
        // we are being inefficient here (double iterating/reading but at
        // this point32_t in the program should be nothing running by
        // definition
        storeData(tmp_barrier, readNStore(ncores, fds));
#endif

#ifdef VERBOSE
        clock_gettime(CLOCK_MONOTONIC, &end);
#endif
        PRINT(MED_VERBOSE, "Time(us): %lu", us_diff(end, start));


        // iterate through ncores and check temperature. On first failure
        // break which will bring control back to while loop
        for (int32_t i = 0; i < ncores; i++) {
            PRINT(MED_VERBOSE,
                  "(%.3f vs %.3f) ",
                  init_temps[i],
                  getTemp(fds[i]));


            // check if given core has cooled off
            if (allowed_variance * init_temps[i] < getTemp(fds[i])) {
                PRINT(MED_VERBOSE, "Failed\n");

                // go back to while loop for sleep/timeout check
                break;
            }
            else if ((i + 1) == ncores) {  // all cores passed
                PRINT(MED_VERBOSE, "Passed\n");
                return;
            }
        }

        // sleep to cooloff
        usleep(usec_sleep_time);

        // get time for timeout check
        clock_gettime(CLOCK_MONOTONIC, &end);
    }

    // throw error if never cooled off
    errdie("Error, temperatures never cooled off\n");
}


/*
  initializes a temp_barrier_attr with default values.
  attr  : attr to be initialized
*/
void
initAttr(temp_barrier_attr * attr) {
    attr->cpus             = NULL;
    attr->usec_sleep_time  = DEFAULT_SLEEP;
    attr->timeout_sec      = DEFAULT_TIMEOUT;
    attr->allowed_variance = DEFAULT_VARIANCE;
    attr->init_temps       = NULL;
    attr->enf_mode         = DEFAULT_ENF_MODE;
#ifdef STORE_DATA
    attr->ntrials = DEFAULT_TRIALS;
#endif
}


// sets a given field in attr struct. Each function corresponds to a
// field I bet you can figure out.
// attr             : attr being set
// usec_sleep_time  : sleep time to set (between checking temperatures)
void
attrSetSleepUS(temp_barrier_attr * attr, int32_t usec_sleep_time) {

    DBG_ASSERT(attr, "attr not set\n");
    attr->usec_sleep_time = usec_sleep_time;
}

// sets the timeout field of attr struct
// attr         : attr being set
// timeout_sec  : timeout for when to give up on enforcement
void
attrSetTimeoutSec(temp_barrier_attr * attr, uint32_t timeout_sec) {
    DBG_ASSERT(attr, "attr not set\n");
    attr->timeout_sec = timeout_sec;
}


// sets the allowed variance field of attr struct
// attr             : attr being set
// allowed_variance : allowed variance for cur temperature vs initial value when
//                    cooling
void
attrSetAllowedVariance(temp_barrier_attr * attr, float allowed_variance) {
    DBG_ASSERT(attr, "attr not set\n");
    attr->allowed_variance = allowed_variance;
}


// sets the init_temps field of attr struct
// attr             : attr being set
// init_temps       : array with temperatures to enforce to for each core
void
attrSetInitTemps(temp_barrier_attr * attr, float * init_temps) {
    DBG_ASSERT(attr, "attr not set\n");
    attr->init_temps = init_temps;
}


// sets the ntrials field of attr struct
// attr             : attr being set
// ntrials          : only relevant if storing for each temperature read. This
//                    just allows exact allocation. It is not necessary for
//                    correctness.
void
attrSetTrials(temp_barrier_attr * attr, int32_t ntrials) {
#ifdef STORE_DATA
    DBG_ASSERT(attr, "attr not set\n");
    attr->ntrials = ntrials;
#endif
}


// sets the mode field of attr struct
// attr             : attr being set
// mode             : mode for temperature enforcer. This is either within range
//                    or below threshold.
void
attrSetEnforcer(temp_barrier_attr * attr, enforcer_modes mode) {
    DBG_ASSERT(attr, "attr not set\n");
    attr->enf_mode = mode;
}


// sets the init_temps field of attr struct
// attr             : attr being set
// cpus             : cpu set to tell barrier which core temperatures to enforce
//                    around. If not set will default to all cores on machine.
void
attrSetCPUS(temp_barrier_attr * attr, cpu_set_t * cpus) {
    DBG_ASSERT(attr, "attr not set\n");
    attr->cpus = cpus;
}


// for defaulting if CPUS passed in null
static cpu_set_t cpus_default;
static void
setDefaultCPU() {
    CPU_ZERO(&cpus_default);

    // basically just set regulator for all cores the current process has
    // access to
    sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpus_default);
}

// initializes new temp barrier. Returns ptr to dynamically allocated
// region of memory storing the barrier.
// barrier_num      : for initializing pthread_barrier_t. Is number of waits for
//                    barrier to expect for releasing.
// attr             : attr struct with values for initializing the barrier
temp_barrier *
initTempBarrier(int32_t barrier_num, temp_barrier_attr * attr) {

    // allocate tmp_barrier
    temp_barrier * tmp_barrier =
        (temp_barrier *)mycalloc(1, sizeof(temp_barrier));

    // if attr is null use default values
    if (attr == NULL) {
        temp_barrier_attr default_attr;
        initAttr(&default_attr);
        attr = &default_attr;
    }

    // if cpu set is null use default
    tmp_barrier->cpus = attr->cpus;
    if (tmp_barrier->cpus == NULL) {
        setDefaultCPU();
        tmp_barrier->cpus = &cpus_default;
    }

    // dont need ncores here for correctness but CPU_COUNT is not free
    // but memory basically is
    int32_t ncores      = CPU_COUNT(tmp_barrier->cpus);
    tmp_barrier->ncores = ncores;

    // open the file descriptors
    int32_t * fds = getTempFiles(ncores, tmp_barrier->cpus);

    // if init_temps where specified use them, else read them
    if (attr->init_temps) {
        tmp_barrier->init_temps = attr->init_temps;
    }
    else {
        tmp_barrier->init_temps = readNStore(ncores, fds);

        // set 1s bit so know to free
        low_bits_set((tmp_barrier->init_temps), INTERNAL_ALLOC);
    }

#ifdef VERBOSE
    PRINT(LOW_VERBOSE,
          "init: [%.3f",
          local_get_ptr(tmp_barrier->init_temps)[0]);
    for (int32_t i = 1; i < ncores; i++) {
        PRINT(LOW_VERBOSE, ", %.3f", local_get_ptr(tmp_barrier->init_temps)[i]);
    }
    PRINT(LOW_VERBOSE, "]\n");
#endif


    // set fields
    tmp_barrier->fds = fds;

    tmp_barrier->allowed_variance = attr->allowed_variance;
    tmp_barrier->enf_mode         = attr->enf_mode;

    tmp_barrier->timeout_sec     = attr->timeout_sec;
    tmp_barrier->usec_sleep_time = attr->usec_sleep_time;


    // this will be removed. Purely for development
#ifdef STORE_DATA
    int32_t trials_alloc = (attr->ntrials != -1) ? attr->ntrials : TRIALS_GUESS;
    int32_t max_sleep_alloc =
        (attr->timeout_sec != NO_TIMEOUT)
            ? ((us_per_sec * (uint64_t)attr->timeout_sec) /
               attr->usec_sleep_time)
            : MAX_SLEEPS_GUESS;

    tmp_barrier->trial_bound = trials_alloc;
    tmp_barrier->temps  = (float ***)mycalloc(trials_alloc, sizeof(float **));
    tmp_barrier->bounds = (int32_t *)mycalloc(trials_alloc, sizeof(int32_t));
    tmp_barrier->index  = (int32_t *)mycalloc(trials_alloc, sizeof(int32_t));
    tmp_barrier->timestamps =
        (struct timespec **)mycalloc(trials_alloc, sizeof(struct timespec *));

    for (int32_t i = 0; i < trials_alloc; i++) {
        tmp_barrier->timestamps[i] =
            (struct timespec *)mycalloc(max_sleep_alloc,
                                        sizeof(struct timespec));
        tmp_barrier->temps[i] =
            (float **)mycalloc(max_sleep_alloc, sizeof(float *));
        tmp_barrier->bounds[i] = max_sleep_alloc;
    }
#endif

    // initialize the actual barrier
    mybarrierinit(&tmp_barrier->barrier, barrier_num);
    return tmp_barrier;
}


// the api function to do barrier synchronization. Coordinator needs to
// be set by 1 thread and 1 thread only to actually do the call to
// regulate temp. Just about nothing in the temp barrier is safe for
// concurrency
// tmp_barirer: barrier for sleeping/cooling off.
// coordinator: boolean to say whether this call should actually make the
//              enforce temperature call or just hang at pthread_barrier. Must
//              be set by at least one of the waiting threads (and example might
//              be passing (!tid)
void
tempBarrierWait(temp_barrier * tmp_barrier, int32_t coordinator) {

    // need 2 barrier wait calls. Otherwise there is a race condition in
    // that coordinator thread could finish before others started using
    // serious CPU resources, path the regulator but by the time barrier
    // is passed cores could be hot
    pthread_barrier_wait(&tmp_barrier->barrier);

    if (coordinator) {
        if (tmp_barrier->enf_mode == BELOW_THRESH) {
            enforceBelowThresh(tmp_barrier);
        }
        else if (tmp_barrier->enf_mode == WITHIN_RANGE) {
            enforceWithinRange(tmp_barrier);
        }
        else {
            errdie("Unknown enforcer mode\n");
        }
#ifdef STORE_DATA
        tmp_barrier->cur_trial++;
        DBG_ASSERT(tmp_barrier->cur_trial <= tmp_barrier->trial_bound,
                   "invalid storage index for trial number: %d / %d\n",
                   tmp_barrier->cur_trial,
                   tmp_barrier->trial_bound);
#endif
    }

    pthread_barrier_wait(&tmp_barrier->barrier);
}


// free all memory assosiated with barrier
// tmp_barrier: barrier being freed;
void
freeTempBarrier(temp_barrier * tmp_barrier
) {
    pthread_barrier_destroy(&tmp_barrier->barrier);
    myfree(tmp_barrier->fds);
    if (low_bits_get(tmp_barrier->init_temps) == INTERNAL_ALLOC) {
        myfree(local_get_ptr(tmp_barrier->init_temps));
    }

#ifdef STORE_DATA
    for (int32_t i = 0; i < tmp_barrier->trial_bound; i++) {
        for (int32_t j = 0; j < tmp_barrier->index[i]; j++) {
            myfree(tmp_barrier->temps[i][j]);
        }
        myfree(tmp_barrier->timestamps[i]);
        myfree(tmp_barrier->temps[i]);
    }
    myfree(tmp_barrier->timestamps);
    myfree(tmp_barrier->temps);
    myfree(tmp_barrier->bounds);
    myfree(tmp_barrier->index);
#endif
    myfree(tmp_barrier);
}


// prints data from sleeping. only does anything if STORE_DATA is defined
// tmp_barrier : barrier whose data is being printed
// outfile     : file to write the data to (can be stdout or stderr)
void
printBarrierData(temp_barrier * tmp_barrier, FILE * outfile) {
#ifdef STORE_DATA
    int32_t ncores = tmp_barrier->ncores;

    char header[512]  = "Trial,Time Since First Sleep(us)";
    char core_buf[16] = "";
    char cur_offset   = strlen(header);

    uint64_t * cpu_mask = (uint64_t *)tmp_barrier->cpus->__bits;
    uint64_t * end      = cpu_mask + (sizeof(cpu_set_t) / sizeof(uint64_t));
    int32_t    iter     = 0;
    while (cpu_mask < end) {
        uint64_t tmp_core_mask = *cpu_mask;
        cpu_mask++;
        while (tmp_core_mask) {
            int32_t i = ff1_64_ctz_nz(tmp_core_mask);
            tmp_core_mask ^= ((1UL) << i);
            i += iter * sizeof_bits(uint64_t);

            sprintf(core_buf, ",Core_%d", i);
            int32_t core_len = strlen(core_buf);

            memcpy(header + cur_offset, core_buf, core_len);

            cur_offset += core_len;
        }
        iter++;
    }

    fprintf(outfile, "%s\n", header);
    for (int32_t i = 0; i <= tmp_barrier->cur_trial; i++) {
        struct timespec first_trial_stamp = tmp_barrier->timestamps[i][0];
        for (int32_t j = 0; j < tmp_barrier->index[i]; j++) {
            fprintf(outfile,
                    "%d,%lu",
                    i,
                    us_diff(tmp_barrier->timestamps[i][j], first_trial_stamp));
            for (int32_t c = 0; c < ncores; c++) {
                fprintf(outfile, ",%.3f", tmp_barrier->temps[i][j][c]);
            }
            fprintf(outfile, "\n");
        }
    }

    fflush(outfile);
#endif
}
