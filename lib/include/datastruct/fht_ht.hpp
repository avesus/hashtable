#ifndef _FHT_HT_H_
#define _FHT_HT_H_

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>


static uint32_t
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

static uint32_t
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


static void *
myMmap(void *        addr,
       uint64_t      length,
       int32_t       prot_flags,
       int32_t       mmap_flags,
       int32_t       fd,
       int32_t       offset,
       const char *  fname,
       const int32_t ln) {

    void * p = mmap(addr, length, prot_flags, mmap_flags, fd, offset);
    if (p == MAP_FAILED && length) {
        assert(0);
    }
    return p;
}

static void
myMunmap(void * addr, uint64_t length, const char * fname, const int32_t ln) {
    if (addr && length) {
        if ((((uint64_t)addr) % PAGE_SIZE) != 0) {
            assert(0);
        }

        if (munmap(addr, length) == -1) {
            assert(0);
        }
    }
}


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

#define mymunmap(X, Y) myMunmap((X), (Y), __FILE__, __LINE__)


//#define FHT_STATS
#ifdef FHT_STATS
#define FHT_STATS_INCR(X) this->X++
#define FHT_STATS_SUMMARY this->stats_summary()
#else
#define FHT_STATS_INCR(X)
#define FHT_STATS_SUMMARY
#endif


// tunable
const uint32_t FHT_DEFAULT_INIT_SIZE = PAGE_SIZE;
const uint32_t FHT_HASH_SEED         = 0;
const uint32_t FHT_SEARCH_NUMBER     = (L1_CACHE_LINE_SIZE - 16);

// return values
const uint32_t FHT_NOT_ADDED = 0;
const uint32_t FHT_ADDED     = 1;

const uint32_t FHT_NOT_FOUND = 0;
const uint32_t FHT_FOUND     = 1;

const uint32_t FHT_NOT_DELETED = 0;
const uint32_t FHT_DELETED     = 1;

typedef uint8_t tag_type_t;

template<typename K, typename V>
struct fht_node {
    K key;
    V val;
};

template<typename K, typename V>
struct fht_chunk {
    tag_type_t tags[L1_CACHE_LINE_SIZE];

    fht_node<K, V> nodes[L1_CACHE_LINE_SIZE];
};

static const uint32_t murmur3_32(const uint8_t * key, const uint32_t len);
template<typename K>
struct DEFAULT_HASH_32 {


    const uint32_t
    operator()(const K key) const {
        return murmur3_32((const uint8_t *)(&key), sizeof(K));
    }
};

template<typename K, typename V, typename Hasher = DEFAULT_HASH_32<K>>
class fht_table {
    fht_chunk<K, V> * chunks;
    uint32_t          log_incr;
    Hasher            hash_32;

#ifdef FHT_STATS
    uint64_t add_att;
    uint64_t find_att;
    uint64_t remove_att;

    uint64_t add_iter;
    uint64_t add_success;
    uint64_t add_duplicate;
    uint64_t add_tag_removed;
    uint64_t add_tag_match;
    uint64_t add_tag_fail;
    uint64_t add_tag_success;

    uint64_t add_resize_att;
    uint64_t add_resize_iter;

    uint64_t find_iter;
    uint64_t find_complete;
    uint64_t find_tag_removed;
    uint64_t find_tag_match;
    uint64_t find_tag_fail;
    uint64_t find_tag_success;

    uint64_t remove_iter;
    uint64_t remove_complete;
    uint64_t remove_tag_removed;
    uint64_t remove_tag_match;
    uint64_t remove_tag_fail;
    uint64_t remove_tag_success;

    uint64_t resize_att;
    uint64_t resize_iter;
    uint64_t resize_sub_iter;
    uint64_t resize_invalid;
    uint64_t resize_valid;
    double   u64div(uint64_t num, uint64_t den);
    void     stats_summary();
#endif


    fht_chunk<K, V> * const resize();

   public:
    fht_table(uint32_t init_size);
    fht_table();
    ~fht_table();


    uint32_t add(const K new_key, const V new_val);
    uint32_t find(const K key);
    uint32_t remove(const K key);
};


//////////////////////////////////////////////////////////////////////
// Actual Implemenation cuz templating kinda sucks imo
//////////////////////////////////////////////////////////////////////
// Constants
#define FHT_NODES_PER_CACHE_LINE     L1_CACHE_LINE_SIZE
#define FHT_LOG_NODES_PER_CACHE_LINE L1_LOG_CACHE_LINE_SIZE
#define FHT_CACHE_IDX_MASK           (FHT_NODES_PER_CACHE_LINE - 1)
#define FHT_CACHE_ALIGN_MASK         (~(FHT_CACHE_IDX_MASK))
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Helpers
// mask of n bits
#define TO_MASK(n) ((1 << (n)) - 1)

// for extracting a bit
#define GET_NTH_BIT(X, n) (((X) >> (n)) & 0x1)
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Tag Fields
#define VALID_MASK   (0x1)
#define DELETE_MASK  (0x2)
#define CONTENT_MASK (~(VALID_MASK | DELETE_MASK))

#define IS_VALID(tag)    ((tag)&VALID_MASK)
#define SET_VALID(tag)   ((tag) |= VALID_MASK)
#define SET_UNVALID(tag) ((tag) ^= VALID_MASK)

#define IS_DELETED(tag)    ((tag)&DELETE_MASK)
#define SET_DELETED(tag)   ((tag) |= DELETE_MASK)
#define SET_UNDELETED(tag) ((tag) ^= DELETE_MASK)

// skip in resize if either not valid or deleted
#define RESIZE_SKIP(tag) ((tag & (VALID_MASK | DELETE_MASK)) != VALID_MASK)

// valid bit + delete bit
#define CONTENT_OFFSET   (1 + 1)
#define GET_CONTENT(tag) ((tag)&CONTENT_MASK)
//////////////////////////////////////////////////////////////////////
// Calculating Tag and Start Idx from a raw hash
#define GEN_TAG(hash_val) (((hash_val) << CONTENT_OFFSET))
#define GEN_START_IDX(hash_val)                                                \
    ((hash_val) >> (32 - FHT_LOG_NODES_PER_CACHE_LINE))
//////////////////////////////////////////////////////////////////////
// For getting test idx from start_idx and j in loop
#define IDX_MOD(idx)             (3 * (idx))
#define GEN_TEST_IDX(start, idx) (((start) ^ IDX_MOD(idx)) & FHT_CACHE_IDX_MASK)
//////////////////////////////////////////////////////////////////////
// Node helper functions
#define SET_KEY_VAL(node, _key, _val)                                          \
    (node).key = (_key);                                                       \
    (node).val = (_val)
#define COMPARE_KEYS(key1, key2) ((key1) == (key2))
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Constructor / Destructor
template<typename K, typename V, typename Hasher>
fht_table<K, V, Hasher>::fht_table(const uint32_t init_size) {
#ifdef FHT_STATS
    memset(this, 0, sizeof(fht_table));
#endif
    const uint64_t _init_size =
        init_size > FHT_DEFAULT_INIT_SIZE
            ? (init_size ? roundup_32(init_size) : FHT_DEFAULT_INIT_SIZE)
            : FHT_DEFAULT_INIT_SIZE;

    const uint32_t _log_init_size = ulog2_64(_init_size);

    //    hash_func hf;
    //    this->hash_32 = hf.get_hash_32();

    this->chunks = (fht_chunk<K, V> *)mymmap_alloc(
        (_init_size / FHT_NODES_PER_CACHE_LINE) * sizeof(fht_chunk<K, V>));
    this->log_incr = _log_init_size;
}
template<typename K, typename V, typename Hasher>
fht_table<K, V, Hasher>::fht_table() : fht_table(FHT_DEFAULT_INIT_SIZE) {}

template<typename K, typename V, typename Hasher>
fht_table<K, V, Hasher>::~fht_table() {
    FHT_STATS_SUMMARY;
    mymunmap(this->chunks,
             ((1 << (this->log_incr)) / FHT_NODES_PER_CACHE_LINE) *
                 sizeof(fht_chunk<K, V>));
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Default hash function
static const uint32_t
murmur3_32(const uint8_t * key, const uint32_t len) {
    uint32_t h = FHT_HASH_SEED;
    if (len > 3) {
        const uint32_t * key_x4 = (const uint32_t *)key;
        uint32_t         i      = len >> 2;
        do {

            uint32_t k = *key_x4++;
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xe6546b64;
        } while (--i);
        key = (const uint8_t *)key_x4;
    }
    if (len & 3) {
        uint32_t i = len & 3;
        uint32_t k = 0;
        key        = &key[i - 1];
        do {
            k <<= 8;
            k |= *key--;
        } while (--i);
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Resize
template<typename K, typename V, typename Hasher>
fht_chunk<K, V> * const
fht_table<K, V, Hasher>::resize() {
    FHT_STATS_INCR(resize_att);
    const uint32_t                _new_log_incr = ++(this->log_incr);
    const fht_chunk<K, V> * const old_chunks    = this->chunks;

    const uint32_t _num_chunks =
        (1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE;

    fht_chunk<K, V> * const new_chunks = (fht_chunk<K, V> *)mymmap_alloc(
        (2 * _num_chunks) * sizeof(fht_chunk<K, V>));

    // set this while its definetly still in cache
    this->chunks = new_chunks;


    for (uint32_t i = 0; i < _num_chunks; i++) {
        const fht_node<K, V> * const nodes = old_chunks[i].nodes;
        const tag_type_t * const     tags  = old_chunks[i].tags;
        //        const uint64_t * const       fast_tags = (const uint64_t *
        //        const)tags;

        uint32_t raw_slots[4][56];
        uint8_t  raw_idx[4][56], incr[4] = { 0, 0 };
        ;
        uint64_t ideals[2] = { 0, 0 };
        //        for (uint32_t out_j = 0; out_j < (FHT_NODES_PER_CACHE_LINE /
        //        8);
        //             out_j++) {
        //            uint64_t fast_bytes = fast_tags[out_j] &
        //            0x0101010101010101; uint64_t j;
        //  while(fast_bytes) {
        //      __asm__("tzcnt %1, %0" : "=r"((j)) : "rm"((fast_bytes)));
        //      fast_bytes ^= ((1UL) << j);
        for (uint32_t j = 0; j < (FHT_NODES_PER_CACHE_LINE); j++) {
            if (!RESIZE_SKIP(tags[j])) {

                const uint32_t raw_slot = this->hash_32(nodes[j].key);
                const uint32_t start_idx =
                    GEN_START_IDX(raw_slot) & FHT_CACHE_IDX_MASK;
                const uint32_t next_bit =
                    (GET_NTH_BIT(raw_slot, _new_log_incr - 1));
                if (ideals[next_bit] & ((1UL) << (start_idx))) {
                    raw_slots[next_bit + 2][incr[next_bit + 2]] = raw_slot;
                    raw_idx[next_bit + 2][incr[next_bit + 2]++] = j;
                }
                else {
                    ideals[next_bit] |= ((1UL) << (start_idx));
                    raw_slots[next_bit][incr[next_bit]] = raw_slot;
                    raw_idx[next_bit][incr[next_bit]++] = j;
                }
            }
            //            }
        }
        for (uint32_t b = 0; b < 2; b++) {
            const uint32_t b_max = incr[b];

            tag_type_t * const new_tags =
                new_chunks[i | ((b & 0x1) ? _num_chunks : 0)].tags;
            fht_node<K, V> * const new_nodes =
                new_chunks[i | ((b & 0x1) ? _num_chunks : 0)].nodes;

            for (uint32_t j = 0; j < b_max; j++) {
                const uint32_t   raw_slot  = raw_slots[b][j];
                const tag_type_t tag       = GEN_TAG(raw_slot);
                const uint32_t   start_idx = GEN_START_IDX(raw_slot);

                new_tags[start_idx] = tag | VALID_MASK;
                SET_KEY_VAL(new_nodes[start_idx],
                            nodes[raw_idx[b][j]].key,
                            nodes[raw_idx[b][j]].val);

            }
        }

        for (uint32_t b = 2; b < 4; b++) {
            const uint32_t b_max = incr[b];

            tag_type_t * const new_tags =
                new_chunks[i | ((b & 0x1) ? _num_chunks : 0)].tags;
            fht_node<K, V> * const new_nodes =
                new_chunks[i | ((b & 0x1) ? _num_chunks : 0)].nodes;

            for (uint32_t j = 0; j < b_max; j++) {
                const uint32_t   raw_slot  = raw_slots[b][j];
                const tag_type_t tag       = GEN_TAG(raw_slot);
                const uint32_t   start_idx = GEN_START_IDX(raw_slot);

                for (uint32_t new_j = 0; new_j < FHT_SEARCH_NUMBER; new_j++) {
                    FHT_STATS_INCR(resize_sub_iter);
                    const uint32_t test_idx = GEN_TEST_IDX(start_idx, new_j);
                    if (__builtin_expect(!IS_VALID(new_tags[test_idx]), 1)) {
                        new_tags[test_idx] = tag | VALID_MASK;
                        SET_KEY_VAL(new_nodes[test_idx],
                                    nodes[raw_idx[b][j]].key,
                                    nodes[raw_idx[b][j]].val);
                        break;
                    }
                }
            }
        }
    }

    mymunmap((void *)old_chunks,
             ((1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE) *
                 sizeof(fht_chunk<K, V>));

    return new_chunks;
}

//////////////////////////////////////////////////////////////////////
// Add Key Val
template<typename K, typename V, typename Hasher>
uint32_t
fht_table<K, V, Hasher>::add(const K new_key, const V new_val) {
    FHT_STATS_INCR(add_att);
    const uint32_t          _log_incr = this->log_incr;
    fht_chunk<K, V> * const chunks    = this->chunks;
    const uint32_t          raw_slot  = this->hash_32(new_key);

    const uint32_t   next_bit  = GET_NTH_BIT(raw_slot, _log_incr - 1);
    const tag_type_t tag       = GEN_TAG(raw_slot);
    const uint32_t   start_idx = GEN_START_IDX(raw_slot);

    tag_type_t * const tags = (tag_type_t * const)(
        (chunks) + ((raw_slot & (TO_MASK(_log_incr) & FHT_CACHE_ALIGN_MASK)) /
                    FHT_NODES_PER_CACHE_LINE));
    fht_node<K, V> * const nodes =
        (fht_node<K, V> * const)(tags + FHT_NODES_PER_CACHE_LINE);

    // the prefetch on nodes is particularly important
    __builtin_prefetch(nodes + (start_idx & FHT_CACHE_IDX_MASK));
    __builtin_prefetch(tags + (start_idx & FHT_CACHE_IDX_MASK));

    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(add_iter);
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(tags[test_idx]), 1)) {
            FHT_STATS_INCR(add_tag_match);
            tags[test_idx] = (tag | VALID_MASK);
            SET_KEY_VAL(nodes[test_idx], new_key, new_val);
            return FHT_ADDED;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, new_key)) {
                FHT_STATS_INCR(add_tag_success);
                if (IS_DELETED(tags[test_idx])) {
                    SET_UNDELETED(tags[test_idx]);
                    FHT_STATS_INCR(add_tag_removed);
                    return FHT_ADDED;
                }
                FHT_STATS_INCR(add_duplicate);
                return FHT_NOT_ADDED;
            }
            FHT_STATS_INCR(add_tag_fail);
        }
    }

    // no valid slot found so resize
    tag_type_t * const new_tags = (tag_type_t * const)(
        this->resize() +
        ((raw_slot & TO_MASK(_log_incr + 1)) / FHT_NODES_PER_CACHE_LINE));

    fht_node<K, V> * const new_nodes =
        (fht_node<K, V> * const)(new_tags + FHT_NODES_PER_CACHE_LINE);

    const uint32_t new_next_bit = GET_NTH_BIT(raw_slot, _log_incr);
    FHT_STATS_INCR(add_resize_att);
    // after resize add without duplication check
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(add_resize_iter);
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(new_tags[test_idx]), 1)) {
            FHT_STATS_INCR(add_tag_match);
            new_tags[test_idx] = (tag | VALID_MASK);
            SET_KEY_VAL(new_nodes[test_idx], new_key, new_val);
            return FHT_ADDED;
        }
    }

    // probability of this is 1 / (2 ^ FHT_SEARCH_NUMBER)
    assert(0);
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Find
template<typename K, typename V, typename Hasher>
uint32_t
fht_table<K, V, Hasher>::find(const K key) {
    FHT_STATS_INCR(find_att);
    const uint32_t                _log_incr = this->log_incr;
    const fht_chunk<K, V> * const chunks    = this->chunks;
    const uint32_t                raw_slot  = this->hash_32(key);

    const uint32_t   next_bit  = GET_NTH_BIT(raw_slot, _log_incr - 1);
    const tag_type_t tag       = GEN_TAG(raw_slot);
    const uint32_t   start_idx = GEN_START_IDX(raw_slot);

    tag_type_t * const tags = (tag_type_t * const)(
        (chunks) + ((raw_slot & (TO_MASK(_log_incr) & FHT_CACHE_ALIGN_MASK)) /
                    FHT_NODES_PER_CACHE_LINE));
    fht_node<K, V> * const nodes =
        (fht_node<K, V> * const)(tags + FHT_NODES_PER_CACHE_LINE);

    // the prefetch on nodes is particularly important
    __builtin_prefetch(nodes + (start_idx & FHT_CACHE_IDX_MASK));
    __builtin_prefetch(tags + (start_idx & FHT_CACHE_IDX_MASK));

    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(find_iter);
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(tags[test_idx]), 1)) {
            FHT_STATS_INCR(find_tag_match);
            return FHT_NOT_FOUND;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, key)) {
                FHT_STATS_INCR(find_tag_success);
                if (IS_DELETED(tags[test_idx])) {
                    FHT_STATS_INCR(find_tag_removed);
                    return FHT_NOT_FOUND;
                }
                return FHT_FOUND;
            }
            FHT_STATS_INCR(find_tag_fail);
        }
    }
    FHT_STATS_INCR(find_complete);
    return FHT_NOT_FOUND;
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Delete
template<typename K, typename V, typename Hasher>
uint32_t
fht_table<K, V, Hasher>::remove(const K key) {
    FHT_STATS_INCR(remove_att);
    const uint32_t                _log_incr = this->log_incr;
    const fht_chunk<K, V> * const chunks    = this->chunks;
    const uint32_t                raw_slot  = this->hash_32(key);

    const uint32_t   next_bit  = GET_NTH_BIT(raw_slot, _log_incr - 1);
    const tag_type_t tag       = GEN_TAG(raw_slot);
    const uint32_t   start_idx = GEN_START_IDX(raw_slot);

    tag_type_t * const tags = (tag_type_t * const)(
        (chunks) + ((raw_slot & (TO_MASK(_log_incr) & FHT_CACHE_ALIGN_MASK)) /
                    FHT_NODES_PER_CACHE_LINE));
    fht_node<K, V> * const nodes =
        (fht_node<K, V> * const)(tags + FHT_NODES_PER_CACHE_LINE);

    // the prefetch on nodes is particularly important
    __builtin_prefetch(nodes + (start_idx & FHT_CACHE_IDX_MASK));
    __builtin_prefetch(tags + (start_idx & FHT_CACHE_IDX_MASK));

    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(remove_iter);
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(tags[test_idx]), 1)) {
            FHT_STATS_INCR(remove_tag_match);
            return FHT_NOT_DELETED;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, key)) {
                FHT_STATS_INCR(remove_tag_success);
                if (IS_DELETED(tags[test_idx])) {
                    FHT_STATS_INCR(remove_tag_removed);
                    return FHT_NOT_DELETED;
                }
                SET_DELETED(tags[test_idx]);
                return FHT_DELETED;
            }
            FHT_STATS_INCR(remove_tag_fail);
        }
    }
    FHT_STATS_INCR(remove_complete);
    return FHT_NOT_FOUND;
}


//////////////////////////////////////////////////////////////////////
// Stats
#ifdef FHT_STATS
template<typename K, typename V, typename Hasher>
double
fht_table<K, V, Hasher>::u64div(uint64_t num, uint64_t den) {
    return (double)(((double)num) / ((double)den));
}

template<typename K, typename V, typename Hasher>
void
fht_table<K, V, Hasher>::stats_summary() {
    if (this->add_att) {
        fprintf(stderr, "\rAdd Stats\n");
        fprintf(stderr, "\tAttempts     : %lu\n", this->add_att);
        fprintf(stderr, "\tIter         : %lu\n", this->add_iter);
        fprintf(stderr,
                "\t\tIter Avg         : %.3lf\n",
                this->u64div(this->add_iter, this->add_att));
        fprintf(stderr, "\tResize Att   : %lu\n", this->add_resize_att);
        fprintf(stderr, "\tResize Iter  : %lu\n", this->add_resize_iter);
        fprintf(stderr, "\tDuplicate    : %lu\n", this->add_duplicate);
        fprintf(stderr,
                "\t\tDuplicate Rate   : %.3lf\n",
                this->u64div(this->add_duplicate, this->add_att));
        fprintf(stderr, "\tSuccess      : %lu\n", this->add_tag_match);
        fprintf(stderr,
                "\t\tSuccess Rate     : %.3lf\n",
                this->u64div(this->add_tag_match, this->add_att));
        fprintf(stderr, "\tRemoved      : %lu\n", this->add_tag_removed);
        fprintf(stderr,
                "\t\tRemoved Rate     : %.3lf\n",
                this->u64div(this->add_tag_removed, this->add_att));

        fprintf(stderr, "\tTag Match    : %lu\n", this->add_tag_match);
        fprintf(stderr, "\tTag Fail     : %lu\n", this->add_tag_fail);
        fprintf(stderr,
                "\t\tTag Fail Rate    : %.3lf\n",
                this->u64div(this->add_tag_fail, this->add_tag_match));
        fprintf(stderr, "\tTag Success  : %lu\n", this->add_tag_success);
        fprintf(stderr,
                "\t\tTag Success Rate : %.3lf\n",
                this->u64div(this->add_tag_success, this->add_tag_match));
    }
    if (this->find_att) {
        fprintf(stderr, "\n\rFind Stats\n");
        fprintf(stderr, "\tAttempts     : %lu\n", this->find_att);
        fprintf(stderr, "\tIter         : %lu\n", this->find_iter);
        fprintf(stderr,
                "\t\tIter Avg         : %.3lf\n",
                this->u64div(this->find_iter, this->find_att));
        fprintf(stderr, "\tComplete     : %lu\n", this->find_complete);
        fprintf(stderr,
                "\t\tComplete Rate    : %.3lf\n",
                this->u64div(this->find_complete, this->find_att));
        fprintf(stderr, "\tSuccess      : %lu\n", this->find_tag_match);
        fprintf(stderr,
                "\t\tSuccess Rate     : %.3lf\n",
                this->u64div(this->find_tag_match, this->find_att));
        fprintf(stderr, "\tRemoved      : %lu\n", this->find_tag_removed);
        fprintf(stderr,
                "\t\tRemoved Rate     : %.3lf\n",
                this->u64div(this->find_tag_removed, this->find_att));
        fprintf(stderr, "\tTag Match    : %lu\n", this->find_tag_match);
        fprintf(stderr, "\tTag Fail     : %lu\n", this->find_tag_fail);
        fprintf(stderr,
                "\t\tTag Fail Rate    : %.3lf\n",
                this->u64div(this->find_tag_fail, this->find_tag_match));
        fprintf(stderr, "\tTag Success  : %lu\n", this->find_tag_success);
        fprintf(stderr,
                "\t\tTag Success Rate : %.3lf\n",
                this->u64div(this->find_tag_success, this->find_tag_match));
    }
    if (this->remove_att) {
        fprintf(stderr, "\n\rRemove Stats\n");
        fprintf(stderr, "\tAttempts     : %lu\n", this->remove_att);
        fprintf(stderr, "\tIter         : %lu\n", this->remove_iter);
        fprintf(stderr,
                "\t\tIter Avg         : %.3lf\n",
                this->u64div(this->remove_iter, this->remove_att));
        fprintf(stderr, "\tComplete     : %lu\n", this->remove_complete);
        fprintf(stderr,
                "\t\tComplete Rate    : %.3lf\n",
                this->u64div(this->remove_complete, this->remove_att));
        fprintf(stderr, "\tSuccess      : %lu\n", this->remove_tag_match);
        fprintf(stderr,
                "\t\tSuccess Rate     : %.3lf\n",
                this->u64div(this->remove_tag_match, this->remove_att));
        fprintf(stderr, "\tRemoved      : %lu\n", this->remove_tag_removed);
        fprintf(stderr,
                "\t\tRemoved Rate     : %.3lf\n",
                this->u64div(this->remove_tag_removed, this->remove_att));
        fprintf(stderr, "\tTag Match    : %lu\n", this->remove_tag_match);
        fprintf(stderr, "\tTag Fail     : %lu\n", this->remove_tag_fail);
        fprintf(stderr,
                "\t\tTag Fail Rate    : %.3lf\n",
                this->u64div(this->remove_tag_fail, this->remove_tag_match));
        fprintf(stderr, "\tTag Success  : %lu\n", this->remove_tag_success);
        fprintf(stderr,
                "\t\tTag Success Rate : %.3lf\n",
                this->u64div(this->remove_tag_success, this->remove_tag_match));
    }
    if (this->resize_att) {
        fprintf(stderr, "\n\rResize Stats\n");
        fprintf(stderr, "\tAttempts     : %lu\n", this->resize_att);
        fprintf(stderr, "\tIter         : %lu\n", this->resize_iter);
        fprintf(stderr, "\tSub Iter     : %lu\n", this->resize_sub_iter);
        fprintf(stderr,
                "\t\tSub Iter Avg     : %.3lf\n",
                this->u64div(this->resize_sub_iter, this->resize_iter));
        fprintf(stderr, "\tInvalid      : %lu\n", this->resize_invalid);
        fprintf(stderr,
                "\t\tinValid Rate     : %.3lf\n",
                this->u64div(this->resize_invalid, this->resize_iter));
        fprintf(stderr, "\tValid        : %lu\n", this->resize_valid);
        fprintf(stderr,
                "\t\tValid Rate       : %.3lf\n",
                this->u64div(this->resize_valid, this->resize_iter));
    }
}
#endif

//////////////////////////////////////////////////////////////////////
// Various Optimized for Size Hash Functions
//////////////////////////////////////////////////////////////////////
// Default hash function
static const uint32_t
murmur3_32_4(const uint32_t key) {
    uint32_t h = FHT_HASH_SEED;

    uint32_t k = key;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;

    h ^= 4;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

static const uint32_t
murmur3_32_8(const uint64_t key) {
    uint32_t h = FHT_HASH_SEED;

    // 1st 4 bytes
    uint32_t k = key;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;

    // 2nd 4 bytes
    k = key >> 32;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;

    h ^= 8;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

template<typename K>
struct HASH_32_4 {

    const uint32_t
    operator()(const K key) const {
        return murmur3_32_4((key));
    }
};

template<typename K>
struct HASH_32_8 {

    const uint32_t
    operator()(const K key) const {
        return murmur3_32_8((key));
    }
};

template<typename K>
struct HASH_32_STR {

    const uint32_t
    operator()(K const & key) const {
        return murmur3_32((const uint8_t *)(key.c_str()), key.length());
    }
};
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Undefs
#undef FHT_NODES_PER_CACHE_LINE
#undef FHT_LOG_NODES_PER_CACHE_LINE
#undef FHT_CACHE_IDX_MASK
#undef FHT_CACHE_ALIGN_MASK
#undef TO_MASK
#undef GET_NTH_BIT
#undef VALID_MASK
#undef DELETE_MASK
#undef CONTENT_MASK
#undef IS_VALID
#undef SET_VALID
#undef SET_UNVALID
#undef IS_DELETED
#undef SET_DELETED
#undef SET_UNDELETED
#undef RESIZE_SKIP
#undef CONTENT_OFFSE
#undef GET_CONTENT
#undef GEN_TAG
#undef GEN_START_IDX
#undef IDX_MOD
#undef GEN_TEST_IDX
#undef SET_KEY_VAL
#undef COMPARE_KEYS
//////////////////////////////////////////////////////////////////////

#endif
