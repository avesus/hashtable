#ifndef _MY_OPT_H_
#define _MY_OPT_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#define MIN(x, y) (((x) < (y) ? (x) : (y)))
#define MAX(x, y) (((x) > (y) ? (x) : (y)))
#define ROUNDUP_P2(X, Y) (((X) + ((Y) - 1)) & (~((Y) - 1)))

/*
  X	x index returned from IDX_TO_X
  Y	log_2(init_size)
*/
#define X_TO_SIZE(X, Y) ((1U) << ((X) + (Y) + ((X) == 0)))

/*
  X	index value
  Y	log_2(init_size)
*/
#define IDX_TO_X(X, Y) (sizeof_bits(uint32_t) - (fl1_32_clz((X) >> (Y))))

/*
  X	index value
  Z	x index returned from IDX_TO_X
  Y	log_2(init_size)
*/
#define IDX_TO_Y(X, Y, Z) ((X) & ((1 << ((Y) + ((Z) - 1) + ((Y) == 0))) - 1))

// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_32_ctz(X) ((X) ? __builtin_ctz((X)) : 32)
#define ff0_32_ctz(X) ff1_32_ctz(~(X))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define fl1_32_clz(X) ((X) ? __builtin_clz((X)) : 32)
#define fl0_32_clz(X) fl1_32_clz(~(X))

// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_64_ctz(X) ((X) ? __builtin_ctzll((X)) : 64)
#define ff0_64_ctz(X) ff1_64_ctz(~(X))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define fl1_64_clz(X) ((X) ? __builtin_clzll((X)) : 64)
#define fl0_64_clz(X) fl1_64_clz(~(X))


// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_32_ctz_nz(X) __builtin_ctz((X))
#define ff0_32_ctz_nz(X) ff1_32_ctz_nz(~(X))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define fl1_32_clz_nz(X) __builtin_clz((X))
#define fl0_32_clz_nz(X) fl1_32_clz_nz(~(X))

// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_64_ctz_nz(X) __builtin_ctzll((X))
#define ff0_64_ctz_nz(X) ff1_64_ctz_nz(~(X))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define fl1_64_clz_nz(X) __builtin_clzll((X))
#define fl0_64_clz_nz(X) fl1_64_clz_nz(~(X))


// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_asm_tz(X, Y) __asm__("tzcnt %1, %0" : "=r"((Y)) : "rm"((X)));
#define ff0_asm_tz(X, Y) ff1_asm_tz(~(X), (Y))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define fl1_asm_lz(X, Y) __asm__("lzcnt %1, %0" : "=r"((Y)) : "rm"((X)));
#define fl0_asm_lz(X, Y) fl1_asm_lz(~(X), (Y))

#define ff1_32_tz(X) _tzcnt_u32((X))
#define ff0_32_tz(X) ff1_64_tz(~(X))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define ff1_64_tz(X) _tzcnt_u64((X))
#define fl0_64_tz(X) fl1_64_tz(~(X))

#define fl1_32_tz(X) _lzcnt_u32((X))
#define ff0_32_tz(X) ff1_64_tz(~(X))

// ff -> find last, i.e fl1 of 0b10101000 would return 7
#define fl1_64_tz(X) _lzcnt_u64((X))
#define fl0_64_tz(X) fl1_64_tz(~(X))


// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_32_ffs(X) __builtin_ffs((X))
#define ff0_32_ffs(X) ff1_32_ffs(~(X))

// ff -> find first, i.e ff1 of 0b10101000 would return 3
#define ff1_64_ffs(X) __builtin_ffsll((X))
#define ff0_64_ffs(X) ff1_64_ffs(~(X))

uint32_t ulog2_32(uint32_t n);
uint32_t ulog2_64(uint64_t n);

// these get optimized to popcnt
uint32_t bitcount_32(uint32_t v);
uint32_t bitcount_64(uint64_t v);

uint32_t roundup_32(uint32_t v);
uint64_t roundup_64(uint64_t v);

#endif
