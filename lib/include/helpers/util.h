#ifndef _UTIL_H_
#define _UTIL_H_


#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <regex.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

//////////////////////////////////////////////////////////////////////
#include <general-config.h>
#include <helpers/bits.h>
#include <helpers/debug.h>
#include <helpers/opt.h>
//////////////////////////////////////////////////////////////////////

extern const char * progname;

#define DBG_ASSERT(X, msg, args...)                                            \
    {                                                                          \
        if (__builtin_expect((!(X)), 0)) {                                     \
            errdie(msg, ##args);                                               \
        }                                                                      \
    }


#define DBG_ASSERT_ID(X, ID, msg, args...)                                     \
    {                                                                          \
        if (__builtin_expect((!(X)), 0)) {                                     \
            errdie_ID(ID, msg, ##args);                                        \
        }                                                                      \
    }

#define PRINT(V_LEVEL, ...)                                                    \
    {                                                                          \
        if (verbose >= V_LEVEL) {                                              \
            fprintf(stderr, __VA_ARGS__);                                      \
        }                                                                      \
    }

// error handling
#define MAX_CALL_STACK_OUTPUT 256
#define errdie(msg, args...)                                                   \
    dieOnErrno(__FILE__, __LINE__, pthread_self(), errno, msg, ##args);        \
    assert(0)

#define errdie_ID(ID, msg, args...)                                            \
    dieOnErrno(__FILE__, __LINE__, ID, errno, msg, ##args);                    \
    assert(0)

void dieOnErrno(const char * fn,
                int32_t      ln,
                uint64_t     ID,
                int32_t      en,
                const char * msg,
                ...);

#define die(msg, args...)                                                      \
    _die(__FILE__, __LINE__, msg, ##args);                                     \
    assert(0)

void _die(const char * fn, int32_t ln, const char * fmt, ...);

// vm alloc wrappers
#define mymmap(A, B, C, D, E, F)                                               \
    myMmap((A), (B), (C), (D), (E), (F), __FILE__, __LINE__)
void * myMmap(void *        addr,
              uint64_t      length,
              int32_t       prot_flags,
              int32_t       mmap_flags,
              int32_t       fd,
              int32_t       offset,
              const char *  fname,
              const int32_t ln);


// allocation with mmap
#define mymmap_alloc(X)                                                        \
    myMmap(NULL,                                                               \
           (X),                                                                \
           (PROT_READ | PROT_WRITE),                                           \
           (MAP_ANONYMOUS | MAP_PRIVATE),                                      \
           -1,                                                                 \
           0,                                                                  \
           __FILE__,                                                           \
           __LINE__)

// allocation with mmap option to pass a flag
#define mymmap_falloc(X, Y)                                                    \
    myMmap(NULL,                                                               \
           (X),                                                                \
           (PROT_READ | PROT_WRITE),                                           \
           ((Y) | MAP_ANONYMOUS | MAP_PRIVATE),                                \
           -1,                                                                 \
           0,                                                                  \
           __FILE__,                                                           \
           __LINE__)

#define mymunmap(X, Y) myMunmap((X), (Y), __FILE__, __LINE__)
void myMunmap(void *        addr,
              uint64_t      length,
              const char *  fname,
              const int32_t ln);

// alloc wrappers
#define mycalloc(x, y) myCalloc((x), (y), __FILE__, __LINE__)
void * myCalloc(size_t        nmemb,
                size_t        size,
                const char *  fname,
                const int32_t ln);

#define mymalloc(x) myMalloc((x), __FILE__, __LINE__)
void * myMalloc(size_t size, const char * fname, const int32_t ln);

#define myacalloc(x, y, z) myACalloc((x), (y), (z), __FILE__, __LINE__);
void * myACalloc(size_t        alignment,
                 size_t        nmemb,
                 size_t        size,
                 const char *  fname,
                 const int32_t ln);

#define myaalloc(x, y) myAAlloc((x), (y), __FILE__, __LINE__);
void * myAAlloc(size_t        alignment,
                size_t        size,
                const char *  fname,
                const int32_t ln);

#define myfree(x) myFree((x), __FILE__, __LINE__);
void myFree(void * ptr, const char * fname, const int32_t ln);

// thread creation
#define mypthread_create(w, x, y, z)                                           \
    myPthread_Create((w), (x), (y), (z), __FILE__, __LINE__)
void myPthread_Create(pthread_t *      tid,
                      pthread_attr_t * attr,
                      void *(fun)(void *),
                      void *        args,
                      const char *  fname,
                      const int32_t ln);


// set core affinity for thread attr
#define myset_core(x, y) mySet_Core((x), (y), __FILE__, __LINE__)
void mySet_Core(pthread_attr_t * attr,
                size_t           core,
                const char *     fname,
                const int32_t    ln);

#define mybarrierinit(x, y) myBarrierInit((x), (y), __FILE__, __LINE__);
void myBarrierInit(pthread_barrier_t * barrier,
                   uint32_t            nthreads,
                   const char *        fname,
                   const int32_t       ln);

// IO
#define myfexists(X) (access((X), F_OK) != -1)

#define myfcntl(X, Y, Z) myFnctl((X), (Y), (Z), __FILE__, __LINE__)
int32_t myFnctl(uint32_t      fd,
                uint32_t      cmd,
                uint32_t      flags,
                const char *  fname,
                const int32_t ln);

void make_nonblock(int32_t fd);
void make_blocking(int32_t fd);

#define myopen2(x, y) myOpen2((x), (y), __FILE__, __LINE__)
int32_t myOpen2(const char *  path,
                int32_t       flags,
                const char *  fname,
                const int32_t ln);

#define myopen3(x, y, z) myOpen3((x), (y), (z), __FILE__, __LINE__)
int32_t myOpen3(const char *  path,
                int32_t       flags,
                mode_t        mode,
                const char *  fname,
                const int32_t ln);

#define myread(x, y, z) myRead((x), (y), (z), __FILE__, __LINE__)
int32_t myRead(int32_t       fd,
               void *        buf,
               size_t        count,
               const char *  fname,
               const int32_t ln);


#define myrobustread(x, y, z) myrobustwrite((x), (y), (z), __FILE__, __LINE__)
int32_t myRobustRead(int32_t       fd,
                     void *        buf,
                     size_t        nbytes,
                     const char *  fname,
                     const int32_t ln);


#define mywrite(x, y, z) myWrite((x), (y), (z), __FILE__, __LINE__)
int32_t myWrite(int32_t       fd,
                void *        buf,
                size_t        nbytes,
                const char *  fname,
                const int32_t ln);


#define myrobustwrite(x, y, z) myRobustWrite((x), (y), (z), __FILE__, __LINE__)
int32_t myRobustWrite(int32_t       fd,
                      void *        buf,
                      size_t        nbytes,
                      const char *  fname,
                      const int32_t ln);


#define myfopen(x, y) myFOpen((x), (y), __FILE__, __LINE__)
FILE * myFOpen(const char *  path,
               const char *  mode,
               const char *  fname,
               const int32_t ln);

#define myfread(w, x, y, z) myFRead((w), (x), (y), (z), __FILE__, __LINE__)
int32_t myFRead(void *        ptr,
                size_t        size,
                size_t        nmemb,
                FILE *        fp,
                const char *  fname,
                const int32_t ln);

#define myfwrite(w, x, y, z) myFWrite((w), (x), (y), (z), __FILE__, __LINE__)
int32_t myFWrite(void *        ptr,
                 size_t        size,
                 size_t        nmemb,
                 FILE *        fp,
                 const char *  fname,
                 const int32_t ln);


double getMedian(uint64_t * arr, uint32_t len);
double getMean(uint64_t * arr, uint32_t len);
double getSD(uint64_t * arr, uint32_t len);
double getVar(uint64_t * arr, uint32_t len);
double getMin(uint64_t * arr, uint32_t len);
double getMax(uint64_t * arr, uint32_t len);


#define unit_change 1000
#define ns_per_sec  ((uint64_t)1000000000)
#define us_per_sec  ((uint64_t)1000000)
#define ms_per_sec  ((uint64_t)1000)

enum time_unit {
    s  = 1,
    ms = ms_per_sec,
    us = us_per_sec,
    ns = ns_per_sec,
};
static const char time_unit_str[4][4] = {
    "s",
    "ms",
    "us",
    "ns",
};

uint64_t     to_nsecs(struct timespec t);
uint64_t     ns_diff(struct timespec t1, struct timespec t2);
uint64_t     to_usecs(struct timespec t);
uint64_t     us_diff(struct timespec t1, struct timespec t2);
uint64_t     to_msecs(struct timespec t);
uint64_t     ms_diff(struct timespec t1, struct timespec t2);
uint64_t     to_secs(struct timespec t);
uint64_t     s_diff(struct timespec t1, struct timespec t2);
double       unit_convert(double time_ns, enum time_unit desired);
const char * unit_to_str(enum time_unit u);


// misc
#define sizeof_bits(X) ((sizeof(X)) << 3)


typedef union four_byte_cast {
    uint32_t ui;
    int32_t  i;
    float    f;
} fb_cast;


typedef union eight_byte_cast {
    void *   p;
    void **  pp;
    uint64_t ui;
    int64_t  i;
    double   d;
} eb_cast;


#define byte_f_to_i(X) ((fb_cast)(X)).i
#define byte_d_to_i(X) ((eb_cast)(X)).i
#define byte_i_to_f(X) ((fb_cast)(X)).f
#define byte_i_to_d(X) ((fb_cast)(X)).d


// For generating macros
#define COMBINE_(X, Y) X##Y
#define COMBINE(X, Y)  COMBINE_(X, Y)

#define COMBINE_3(X, Y, Z) X##Y##Z
#define COMBINE3(X, Y, Z)  COMBINE_3(X, Y, Z)

#endif
