#include <helpers/locks.h>

void
lb_readLock(uint64_t * volatile ptr) {
    uint64_t expec = (uint64_t)(*ptr);
    while ((expec & lb_write_locked) == lb_write_locked) {
        do_sleep;
        expec = (uint64_t)(*ptr);
    }
    uint64_t new_val = expec + 1;
    while (!__atomic_compare_exchange(ptr,
                                      (&expec),
                                      (&new_val),
                                      1,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        do_sleep;
        expec = (uint64_t)(*ptr);
        while ((expec & lb_write_locked) == lb_write_locked) {
            do_sleep;
            expec = (uint64_t)(*ptr);
        }
        new_val = expec + 1;
    }
}


void
lb_writeLock(uint64_t * volatile ptr) {

    uint64_t expec = (uint64_t)(*ptr);
    expec &= lowBitsPtrMask;
    uint64_t new_val = expec | lb_write_locked;
    while (!__atomic_compare_exchange(ptr,
                                      (&expec),
                                      (&new_val),
                                      1,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        do_sleep;
        expec = (uint64_t)(*ptr);
        expec &= lowBitsPtrMask;
        new_val = expec | lb_write_locked;
    }
}


void
lb_unlock_rd(uint64_t * volatile ptr) {
    __atomic_sub_fetch((uint64_t *)ptr, 1, __ATOMIC_RELAXED);
}


void
lb_unlock_wr(uint64_t * volatile ptr) {
    uint64_t new_val = (uint64_t)(*ptr);
    new_val &= lowBitsPtrMask;
    __atomic_store_n((uint64_t *)ptr, new_val, __ATOMIC_RELAXED);
}


void
hb_readLock(uint64_t * volatile ptr) {
    uint64_t expec = (uint64_t)(*ptr);
    while (((expec >> highBits) & hb_write_locked) == hb_write_locked) {
        do_sleep;
        expec = (uint64_t)(*ptr);
    }
    uint64_t new_val = expec + (1UL << highBits);
    while (!__atomic_compare_exchange(ptr,
                                      (&expec),
                                      (&new_val),
                                      1,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        do_sleep;
        expec = (uint64_t)(*ptr);
        while (((expec >> highBits) & hb_write_locked) == hb_write_locked) {
            do_sleep;
            expec = (uint64_t)(*ptr);
        }
        new_val = expec + (1UL << highBits);
    }
}


void
hb_writeLock(uint64_t * volatile ptr) {

    uint64_t expec = (uint64_t)(*ptr);
    expec &= highBitsPtrMask;
    uint64_t new_val = expec | (hb_write_locked << highBits);
    while (!__atomic_compare_exchange(ptr,
                                      (&expec),
                                      (&new_val),
                                      1,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        do_sleep;
        expec = (uint64_t)(*ptr);
        expec &= highBitsPtrMask;
        new_val = expec | (hb_write_locked << highBits);
    }
}


void
hb_unlock_rd(uint64_t * volatile ptr) {
    __atomic_sub_fetch((uint64_t *)ptr, (1UL << highBits), __ATOMIC_RELAXED);
}


void
hb_unlock_wr(uint64_t * volatile ptr) {
    uint64_t new_val = (uint64_t)(*ptr);
    new_val &= highBitsPtrMask;
    __atomic_store_n((uint64_t *)ptr, new_val, __ATOMIC_RELAXED);
}


void
ab_readLock(uint64_t * volatile ptr) {
    uint64_t expec = (uint64_t)(*ptr);
    while (expec == ab_write_locked) {
        do_sleep;
        expec = (uint64_t)(*ptr);
    }
    uint64_t new_val = expec + 1;
    while (!__atomic_compare_exchange(ptr,
                                      (uint64_t *)(&expec),
                                      (uint64_t *)(&new_val),
                                      1,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        do_sleep;
        expec = (uint64_t)(*ptr);
        while (expec == ab_write_locked) {
            do_sleep;
            expec = (uint64_t)(*ptr);
        }
        new_val = expec + 1;
    }
}


void
ab_writeLock(uint64_t * volatile ptr) {
    uint64_t expec   = unlocked;
    uint64_t new_val = ab_write_locked;
    while (!__atomic_compare_exchange(ptr,
                                      (uint64_t *)(&expec),
                                      (uint64_t *)(&new_val),
                                      1,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        do_sleep;
        expec   = unlocked;
        new_val = ab_write_locked;
    }
}


void
ab_unlock_rd(uint64_t * volatile ptr) {
    __atomic_sub_fetch((uint64_t *)ptr, 1, __ATOMIC_RELAXED);
}


void
ab_unlock_wr(uint64_t * volatile ptr) {
    __atomic_store_n((uint64_t *)ptr, unlocked, __ATOMIC_RELAXED);
}
