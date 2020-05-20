#include <helpers/opt.h>


void
fast_memset(void * loc, const uint64_t val, const uint64_t len) {

    uint64_t len_8 = (len >> 3);

    uint64_t * loc_u64 = (uint64_t *)loc;
    uint32_t   i;
    for (i = 0; i < len_8; i++) {
        loc_u64[i] = val;
    }
    i <<= 3;
    // if mem for some reason is not 8 byte aligned
    uint8_t * loc_u8    = (uint8_t *)loc;
    uint8_t * val_bytes = (uint8_t *)(&val);
    for (; i < len; i++) {
        loc_u8[i] = val_bytes[i & 0x7];
    }
}

void
fast_bytecopy(void * dst, void * src, uint64_t len) {
    uint64_t * dst_u64 = (uint64_t *)dst;
    uint64_t * src_u64 = (uint64_t *)src;
    uint64_t   len_8   = len >> 3;
    uint32_t   i       = 0;

    for (i = 0; i < len_8; i++) {
        dst_u64[i] = src_u64[i];
    }
    i <<= 3;
    uint8_t * dst_u8 = (uint8_t *)dst;
    uint8_t * src_u8 = (uint8_t *)src;
    for (; i < len; i++) {
        dst_u8[i] = src_u8[i];
    }
}

int32_t
fast_bytecmp(const void * a, const void * b, uint64_t len) {
    uint64_t * a_u64 = (uint64_t *)a;
    uint64_t * b_u64 = (uint64_t *)b;
    uint32_t   i;
    uint64_t   len_8 = len >> 3;
    for (i = 0; i < len_8; i++) {
        if (a_u64[i] != b_u64[i]) {
            return a_u64[i] > b_u64[i] ? 1 : -1;
        }
    }
    i <<= 3;
    uint8_t * a_u8 = (uint8_t *)a;
    uint8_t * b_u8 = (uint8_t *)b;
    for (; i < len; i++) {
        if (a_u8[i] != b_u8[i]) {
            return a_u8[i] > b_u8[i] ? 1 : -1;
        }
    }
    return 0;
}


uint32_t
ulog2_32(uint32_t n) {
    uint32_t s, t;
    t = (n > 0xffff) << 4;
    n >>= t;
    s = (n > 0xff) << 3;
    n >>= s, t |= s;
    s = (n > 0xf) << 2;
    n >>= s, t |= s;
    s = (n > 0x3) << 1;
    n >>= s, t |= s;
    return (t | (n >> 1));
}


uint32_t
ulog2_64(uint64_t n) {
    uint64_t s, t;
    t = (n > 0xffffffff) << 5;
    n >>= t;
    t = (n > 0xffff) << 4;
    n >>= t;
    s = (n > 0xff) << 3;
    n >>= s, t |= s;
    s = (n > 0xf) << 2;
    n >>= s, t |= s;
    s = (n > 0x3) << 1;
    n >>= s, t |= s;
    return (t | (n >> 1));
}

uint32_t
bitcount_64(uint64_t v) {
    uint64_t c;
    c = v - ((v >> 1) & 0x5555555555555555UL);
    c = ((c >> 2) & 0x3333333333333333UL) + (c & 0x3333333333333333UL);
    c = ((c >> 4) + c) & 0x0F0F0F0F0F0F0F0FUL;
    c = ((c >> 8) + c) & 0x00FF00FF00FF00FFUL;
    c = ((c >> 16) + c) & 0x0000FFFF0000FFFFUL;
    c = ((c >> 32) + c) & 0x00000000FFFFFFFFUL;
    return c;
}

uint32_t
bitcount_32(uint32_t v) {
    uint32_t c;
    c = v - ((v >> 1) & 0x55555555);
    c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
    c = ((c >> 4) + c) & 0x0F0F0F0F;
    c = ((c >> 8) + c) & 0x00FF00FF;
    c = ((c >> 16) + c) & 0x0000FFFF;
    return c;
}

uint32_t
roundup_32(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

uint64_t
roundup_64(uint64_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}
