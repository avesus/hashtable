#ifndef _LOCKS_H_
#define _LOCKS_H_

#include <general-config.h>
#include <helpers/bits.h>
#include <helpers/util.h>
#include <unistd.h>

#define unlocked    0
#define usleep_time 50
#define do_sleep    usleep(usleep_time);

#ifndef NORMAL_ALIGNMENT
    // set how you like (i.e I usually use 64 byte aligned because cache
    // line size)
    #define lb_max_threads  62
    #define lb_write_locked 63
#else
    #define lb_max_threads  6
    #define lb_write_locked 7
#endif

#define hb_max_threads  (65534UL)
#define hb_write_locked (65535UL)

// this is kind of unnecissary...
#define ab_max_threads  ((~(0UL)) - 1)
#define ab_write_locked (~(0UL))

// specify hb, lb, or ab for which type of lock to use
#define lockWR(X, Y)   COMBINE(Y, _writeLock)((uint64_t *)&((X)))
#define lockRD(X, Y)   COMBINE(Y, _readLock)((uint64_t *)(&(X)))
#define unlockWR(X, Y) COMBINE(Y, _unlock_wr)((uint64_t *)(&(X)))
#define unlockRD(X, Y) COMBINE(Y, _unlock_rd)((uint64_t *)(&(X)))

// does locking with CAS on high bits of ptr
void hb_unlock_wr(uint64_t * volatile ptr);
void hb_unlock_rd(uint64_t * volatile ptr);
void hb_writeLock(uint64_t * volatile ptr);
void hb_readLock(uint64_t * volatile ptr);

// does lock with CAS on low bits of ptr
void lb_unlock_wr(uint64_t * volatile ptr);
void lb_unlock_rd(uint64_t * volatile ptr);
void lb_writeLock(uint64_t * volatile ptr);
void lb_readLock(uint64_t * volatile ptr);

// does lock with CAS on an uint64_t
void ab_unlock_wr(uint64_t * volatile ptr);
void ab_unlock_rd(uint64_t * volatile ptr);
void ab_writeLock(uint64_t * volatile ptr);
void ab_readLock(uint64_t * volatile ptr);


#endif
