#ifndef _FHT_HT_H_
#define _FHT_HT_H_

//////////////////////////////////////////////////////////////////////
// kind of a middle ground between fully removing the stats/timing code and
// not having it intrusive
#define FHT_STATS_INCR(X)
#define FHT_TAKE_TIME(X)
//////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string>
#include <type_traits>


/* Todos
1) Optimize resize
*/



//////////////////////////////////////////////////////////////////////
// Table params

// tunable
const uint32_t FHT_MAX_MEMORY        = (1 << 30);
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


//////////////////////////////////////////////////////////////////////
// forward declaration of default helper struct

// default return expect ptr to val type if val type is a arithmetic or ptr (i.e
// if val is an int/float or int * / float * pass int * / float * or int ** or
// float ** respectively) in which case it will copy val for a found key into.
// If val is not a default type (i.e c++ string) it will expect a ** of val type
// (i.e val is c++ string, expect store return as string **). This is basically
// to ensure unnecissary copying of larger types doesn't happen but smaller
// types will still be copied cleanly. Returner that sets a different protocol
// can be implemented (or you can just use REF_RETURNER which has already been
// defined). If you implement your own make sure ret_type_t is defined!
template<typename V>
struct DEFAULT_RETURNER;

// depending on type chooses from a few optimized hash functions. Nothing too
// fancy.
template<typename K>
struct DEFAULT_HASH_32;

// if both K and V don't require a real constructor (i.e an int or really any C
// type) it will alloc with mmap and NOT define new (new is slower because even
// if constructor is unnecissary still wastes some time). If type is not builtin
// new is used though allocation backend is still mmap. If you write your own
// allocator be sure that is 1) 0s out the returned memory (this is necessary
// for correctness) and 2) returns at the very least cache line aligned memory
// (this is necessary for performance)
template<typename K, typename V>
struct DEFAULT_MMAP_ALLOC;

// forward declaration of some basic helpers
static uint32_t log_b2(uint64_t n);
static uint32_t roundup_next_p2(uint32_t v);


//////////////////////////////////////////////////////////////////////
// Really just typedef structs
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
//////////////////////////////////////////////////////////////////////
// Table class

template<typename K,
         typename V,
         typename Returner  = DEFAULT_RETURNER<V>,
         typename Hasher    = DEFAULT_HASH_32<K>,
         typename Allocator = DEFAULT_MMAP_ALLOC<K, V>>
class fht_table {
    fht_chunk<K, V> * chunks;
    uint32_t          log_incr;
    Hasher            hash_32;
    Allocator         alloc_mmap;
    Returner          returner;

#ifdef FHT_STATS
    fht_stats stats_collector;
#endif
#ifdef FHT_TIMER
    fht_timer timer;
#endif

    fht_chunk<K, V> * const resize();

    template<typename T>
    using pass_type_t =
        typename std::conditional<(std::is_arithmetic<T>::value ||
                                   std::is_pointer<T>::value),
                                  const T,
                                  T const &>::type;

    using ret_type_t = typename Returner::ret_type_t;

   public:
    fht_table(uint32_t init_size);
    fht_table();
    ~fht_table();


    const uint32_t add(pass_type_t<K> new_key, pass_type_t<V> new_val);
    const uint32_t find(pass_type_t<K> key, ret_type_t store_val);
    const uint32_t remove(pass_type_t<K> key);
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
    (node).key = K(_key);                                                      \
    (node).val = V(_val)
#define COMPARE_KEYS(key1, key2) ((key1) == (key2))
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Constructor / Destructor
template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
fht_table<K, V, Returner, Hasher, Allocator>::fht_table(
    const uint32_t init_size) {
    const uint64_t _init_size =
        init_size > FHT_DEFAULT_INIT_SIZE
            ? (init_size ? roundup_next_p2(init_size) : FHT_DEFAULT_INIT_SIZE)
            : FHT_DEFAULT_INIT_SIZE;

    const uint32_t _log_init_size = log_b2(_init_size);

    this->chunks =
        this->alloc_mmap.init_mem((_init_size / FHT_NODES_PER_CACHE_LINE));

    this->log_incr = _log_init_size;
}
template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
fht_table<K, V, Returner, Hasher, Allocator>::fht_table()
    : fht_table(FHT_DEFAULT_INIT_SIZE) {}

template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
fht_table<K, V, Returner, Hasher, Allocator>::~fht_table() {
    this->alloc_mmap.deinit_mem(
        this->chunks,
        ((1 << (this->log_incr)) / FHT_NODES_PER_CACHE_LINE));
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Resize
template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
fht_chunk<K, V> * const
fht_table<K, V, Returner, Hasher, Allocator>::resize() {
    FHT_TAKE_TIME(resize_timer_idx);
    FHT_STATS_INCR(resize_att);
    const uint32_t                _new_log_incr = ++(this->log_incr);
    const fht_chunk<K, V> * const old_chunks    = this->chunks;

    const uint32_t _num_chunks =
        (1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE;

    fht_chunk<K, V> * const new_chunks =
        this->alloc_mmap.init_mem(2 * _num_chunks);

    // set this while its definetly still in cache
    this->chunks = new_chunks;


    for (uint32_t i = 0; i < _num_chunks; i++) {
        const fht_node<K, V> * const nodes = old_chunks[i].nodes;
        const tag_type_t * const     tags  = old_chunks[i].tags;


        for (uint32_t j = 0; j < FHT_NODES_PER_CACHE_LINE; j++) {
            FHT_STATS_INCR(resize_iter);
            if (RESIZE_SKIP(tags[j])) {
                FHT_STATS_INCR(resize_invalid);
                continue;
            }
            FHT_STATS_INCR(resize_valid);

            const tag_type_t tag = tags[j];

            // annoying that we need to access object. C++ needs a better way to
            // do this.
            const uint32_t raw_slot  = this->hash_32(nodes[j].key);
            const uint32_t start_idx = GEN_START_IDX(raw_slot);

            tag_type_t * const new_tags =
                new_chunks[i | (GET_NTH_BIT(raw_slot, _new_log_incr - 1)
                                    ? _num_chunks
                                    : 0)]
                    .tags;
            fht_node<K, V> * const new_nodes =
                new_chunks[i | (GET_NTH_BIT(raw_slot, _new_log_incr - 1)
                                    ? _num_chunks
                                    : 0)]
                    .nodes;

            for (uint32_t new_j = 0; new_j < FHT_SEARCH_NUMBER; new_j++) {
                FHT_STATS_INCR(resize_sub_iter);
                const uint32_t test_idx = GEN_TEST_IDX(start_idx, new_j);
                if (__builtin_expect(!IS_VALID(new_tags[test_idx]), 1)) {
                    new_tags[test_idx] = tag;
                    SET_KEY_VAL(new_nodes[test_idx],
                                nodes[j].key,
                                nodes[j].val);
                    break;
                }
            }
        }
    }

    this->alloc_mmap.deinit_mem(
        (fht_chunk<K, V> * const)old_chunks,
        ((1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE));

    FHT_TAKE_TIME(resize_timer_idx);
    return new_chunks;
}

//////////////////////////////////////////////////////////////////////
// Add Key Val
template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
const uint32_t
fht_table<K, V, Returner, Hasher, Allocator>::add(pass_type_t<K> new_key,
                                                  pass_type_t<V> new_val) {
    FHT_TAKE_TIME(add_timer_idx);
    FHT_STATS_INCR(add_att);
    const uint32_t          _log_incr = this->log_incr;
    fht_chunk<K, V> * const chunks    = this->chunks;
    const uint32_t          raw_slot  = this->hash_32(new_key);

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


    uint32_t possible_idx = FHT_NODES_PER_CACHE_LINE;
    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(add_iter);
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);

        if (__builtin_expect(!IS_VALID(tags[test_idx]), 1)) {
            if (possible_idx != FHT_NODES_PER_CACHE_LINE) {
                FHT_STATS_INCR(add_place_possible_idx);
                tags[possible_idx] = (tag | VALID_MASK);
                SET_KEY_VAL(nodes[possible_idx], new_key, new_val);
                FHT_TAKE_TIME(add_timer_idx);
                return FHT_ADDED;
            }
            else {
                FHT_STATS_INCR(add_tag_match);
                tags[test_idx] = (tag | VALID_MASK);
                SET_KEY_VAL(nodes[test_idx], new_key, new_val);
                FHT_TAKE_TIME(add_timer_idx);
                return FHT_ADDED;
            }
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, new_key)) {
                FHT_STATS_INCR(add_tag_success);
                if (IS_DELETED(tags[test_idx])) {
                    SET_UNDELETED(tags[test_idx]);
                    FHT_STATS_INCR(add_tag_removed);
                    FHT_TAKE_TIME(add_timer_idx);
                    return FHT_ADDED;
                }
                FHT_STATS_INCR(add_duplicate);
                FHT_TAKE_TIME(add_timer_idx);
                return FHT_NOT_ADDED;
            }
            FHT_STATS_INCR(add_tag_fail);
        }
        if (possible_idx == FHT_NODES_PER_CACHE_LINE &&
            IS_DELETED(tags[test_idx])) {
            FHT_STATS_INCR(add_found_possible_idx);
            possible_idx = test_idx;
        }
    }

    if (possible_idx != FHT_NODES_PER_CACHE_LINE) {
        FHT_STATS_INCR(add_place_possible_idx);
        tags[possible_idx] = (tag | VALID_MASK);
        SET_KEY_VAL(nodes[possible_idx], new_key, new_val);
        FHT_TAKE_TIME(add_timer_idx);
        return FHT_ADDED;
    }

    // no valid slot found so resize
    tag_type_t * const new_tags = (tag_type_t * const)(
        this->resize() +
        ((raw_slot & TO_MASK(_log_incr + 1)) / FHT_NODES_PER_CACHE_LINE));

    fht_node<K, V> * const new_nodes =
        (fht_node<K, V> * const)(new_tags + FHT_NODES_PER_CACHE_LINE);

    FHT_STATS_INCR(add_resize_att);
    // after resize add without duplication check
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(add_resize_iter);
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(new_tags[test_idx]), 1)) {
            FHT_STATS_INCR(add_tag_match);
            new_tags[test_idx] = (tag | VALID_MASK);
            SET_KEY_VAL(new_nodes[test_idx], new_key, new_val);
            FHT_TAKE_TIME(add_timer_idx);
            return FHT_ADDED;
        }
    }

    // probability of this is 1 / (2 ^ FHT_SEARCH_NUMBER)
    assert(0);
}
//////////////////////////////////////////////////////////////////////
// Find
template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
const uint32_t
fht_table<K, V, Returner, Hasher, Allocator>::find(pass_type_t<K> key,
                                                   ret_type_t     store_val) {
    FHT_TAKE_TIME(find_timer_idx);
    FHT_STATS_INCR(find_att);
    const uint32_t                _log_incr = this->log_incr;
    const fht_chunk<K, V> * const chunks    = this->chunks;
    const uint32_t                raw_slot  = this->hash_32(key);

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
            FHT_TAKE_TIME(find_timer_idx);
            return FHT_NOT_FOUND;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, key)) {
                FHT_STATS_INCR(find_tag_success);
                if (IS_DELETED(tags[test_idx])) {
                    FHT_STATS_INCR(find_tag_removed);
                    FHT_TAKE_TIME(find_timer_idx);
                    return FHT_NOT_FOUND;
                }
                this->returner.to_ret_type(store_val, &(nodes[test_idx].val));
                FHT_TAKE_TIME(find_timer_idx);
                return FHT_FOUND;
            }
            FHT_STATS_INCR(find_tag_fail);
        }
    }
    FHT_STATS_INCR(find_complete);
    FHT_TAKE_TIME(find_timer_idx);
    return FHT_NOT_FOUND;
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Delete
template<typename K,
         typename V,
         typename Returner,
         typename Hasher,
         typename Allocator>
const uint32_t
fht_table<K, V, Returner, Hasher, Allocator>::remove(pass_type_t<K> key) {
    FHT_TAKE_TIME(remove_timer_idx);
    FHT_STATS_INCR(remove_att);
    const uint32_t                _log_incr = this->log_incr;
    const fht_chunk<K, V> * const chunks    = this->chunks;
    const uint32_t                raw_slot  = this->hash_32(key);

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
            FHT_TAKE_TIME(remove_timer_idx);
            return FHT_NOT_DELETED;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, key)) {
                FHT_STATS_INCR(remove_tag_success);
                if (IS_DELETED(tags[test_idx])) {
                    FHT_STATS_INCR(remove_tag_removed);
                    FHT_TAKE_TIME(remove_timer_idx);
                    return FHT_NOT_DELETED;
                }
                SET_DELETED(tags[test_idx]);
                FHT_TAKE_TIME(remove_timer_idx);
                return FHT_DELETED;
            }
            FHT_STATS_INCR(remove_tag_fail);
        }
    }
    FHT_STATS_INCR(remove_complete);
    FHT_TAKE_TIME(remove_timer_idx);
    return FHT_NOT_FOUND;
}


//////////////////////////////////////////////////////////////////////
// Default classes


template<typename V>
struct DEFAULT_RETURNER {

    template<typename _V = V>
    using local_ret_type_t =
        typename std::conditional<(std::is_arithmetic<_V>::value ||
                                   std::is_pointer<_V>::value),
                                  _V *,
                                  _V **>::type;

    typedef local_ret_type_t<V> ret_type_t;

    // this is case where builtin type is passed (imo ptr counts as thats
    // basically uint64)
    template<typename _V = V>
    typename std::enable_if<(std::is_arithmetic<_V>::value ||
                             std::is_pointer<_V>::value),
                            void>::type
    to_ret_type(ret_type_t store_val, V const * val) const {
        *store_val = *val;
    }

    // this is case where ** is passed (bigger types)
    template<typename _V = V>
    typename std::enable_if<(!(std::is_arithmetic<_V>::value ||
                               std::is_pointer<_V>::value)),
                            void>::type
    to_ret_type(ret_type_t store_val, V * val) const {
        *store_val = val;
    }
};

template<typename V>
struct REF_RETURNER {

    template<typename _V = V>
    using local_ret_type_t = V &;
    typedef local_ret_type_t<V> ret_type_t;

    void
    to_ret_type(ret_type_t store_val, V const * val) const {
        store_val = *val;
    }
};

// same as ref returner really, just a matter of personal preference
template<typename V>
struct PTR_RETURNER {

    template<typename _V = V>
    using local_ret_type_t = V *;
    typedef local_ret_type_t<V> ret_type_t;

    void
    to_ret_type(ret_type_t store_val, V const * val) const {
        *store_val = *val;
    }
};

template<typename V>
struct PTR_PTR_RETURNER {

    template<typename _V = V>
    using local_ret_type_t = V **;
    typedef local_ret_type_t<V> ret_type_t;

    void
    to_ret_type(ret_type_t store_val, V * val) const {
        *store_val = val;
    }
};


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
struct HASH_32 {
    const uint32_t
    operator()(K const & key) const {
        return murmur3_32((const uint8_t *)(&key), sizeof(K));
    }
};


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
struct HASH_32_CPP_STR {

    const uint32_t
    operator()(K const & key) const {
        return murmur3_32((const uint8_t *)(key.c_str()), key.length());
    }
};


template<typename K>
struct DEFAULT_HASH_32 {

    template<typename _K = K>
    typename std::enable_if<(std::is_arithmetic<_K>::value && sizeof(_K) <= 4),
                            const uint32_t>::type
    operator()(const K key) const {
        return murmur3_32_4(key);
    }

    template<typename _K = K>
    typename std::enable_if<(std::is_arithmetic<_K>::value && sizeof(_K) == 8),
                            const uint32_t>::type
    operator()(const K key) const {
        return murmur3_32_8(key);
    }

    template<typename _K = K>
    typename std::enable_if<(std::is_same<_K, std::string>::value),
                            const uint32_t>::type
    operator()(K const & key) const {
        return murmur3_32((const uint8_t *)(key.c_str()), key.length());
    }

    template<typename _K = K>
    typename std::enable_if<(!std::is_same<_K, std::string>::value) &&
                                (!std::is_arithmetic<_K>::value),
                            const uint32_t>::type
    operator()(K const & key) const {
        return murmur3_32((const uint8_t *)(&key), sizeof(K));
    }
};

//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Memory Allocators
#ifndef mymmap_alloc
#define USING_LOCAL_MMAP
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

#endif

#ifndef mymunmap
#define USING_LOCAL_MUNMAP
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


#define mymunmap(X, Y) myMunmap((X), (Y), __FILE__, __LINE__)
#endif


// less syscalls this way
template<typename K, typename V>
struct OPTIMIZED_MMAP_ALLOC {

    size_t start_offset;
    void * base_address;
    OPTIMIZED_MMAP_ALLOC() {
        this->base_address = mymmap_alloc(FHT_MAX_MEMORY);
        this->start_offset = 0;
    }
    ~OPTIMIZED_MMAP_ALLOC() {
        mymunmap(this->base_address, FHT_MAX_MEMORY);
    }

    fht_chunk<K, V> * const
    init_mem(const size_t size) {
        const size_t old_start_offset = this->start_offset;
        this->start_offset +=
            PAGE_SIZE *
            ((size * sizeof(fht_chunk<K, V>) + PAGE_SIZE - 1) / PAGE_SIZE);

        return (fht_chunk<K, V> * const)(this->base_address + old_start_offset);
    }
    void
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        return;
    }
};


template<typename K, typename V>
struct MMAP_ALLOC {

    fht_chunk<K, V> * const
    init_mem(const size_t size) const {
        return (fht_chunk<K, V> * const)mymmap_alloc(size *
                                                     sizeof(fht_chunk<K, V>));
    }
    void
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        mymunmap(ptr, size * sizeof(fht_chunk<K, V>));
    }
};

template<typename K, typename V>
struct NEW_MMAP_ALLOC {
    void *
    operator new(size_t size) {
        return mymmap_alloc(size);
    }
    void *
    operator new[](size_t size) {
        return mymmap_alloc(size);
    }

    void
    operator delete(void * ptr, const uint32_t size) {
        mymunmap(ptr, size);
    }

    void
    operator delete[](void * ptr, const uint32_t size) {
        mymunmap(ptr, size);
    }

    fht_chunk<K, V> * const
    init_mem(const size_t size) const {
        return new fht_chunk<K, V>[size];
    }
    void
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        delete[] ptr;
    }
};


template<typename K, typename V>
struct DEFAULT_MMAP_ALLOC {
    void *
    operator new(size_t size) {
        return mymmap_alloc(size);
    }
    void *
    operator new[](size_t size) {
        return mymmap_alloc(size);
    }

    void
    operator delete(void * ptr, const uint32_t size) {
        mymunmap(ptr, size);
    }

    void
    operator delete[](void * ptr, const uint32_t size) {
        mymunmap(ptr, size);
    }

    // basically if we need constructor to be called we go to the overloaded new
    // version. These are slower for simply types that can be initialized with
    // just memset. For those types our init mem is compiled to just mmap.
    template<typename _K = K, typename _V = V>
    typename std::enable_if<
        (std::is_arithmetic<_K>::value || std::is_pointer<_K>::value) &&
            (std::is_arithmetic<_V>::value || std::is_pointer<_V>::value),
        fht_chunk<K, V> * const>::type
    init_mem(const size_t size) const {
        return (fht_chunk<K, V> * const)mymmap_alloc(size *
                                                     sizeof(fht_chunk<K, V>));
    }
    template<typename _K = K, typename _V = V>
    typename std::enable_if<
        (std::is_arithmetic<_K>::value || std::is_pointer<_K>::value) &&
            (std::is_arithmetic<_V>::value || std::is_pointer<_V>::value),
        void>::type
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        mymunmap(ptr, size * sizeof(fht_chunk<K, V>));
    }

    template<typename _K = K, typename _V = V>
    typename std::enable_if<
        (!(std::is_arithmetic<_K>::value || std::is_pointer<_K>::value)) ||
            (!(std::is_arithmetic<_V>::value || std::is_pointer<_V>::value)),
        fht_chunk<K, V> * const>::type
    init_mem(const size_t size) const {
        return new fht_chunk<K, V>[size];
    }
    template<typename _K = K, typename _V = V>
    typename std::enable_if<
        (!(std::is_arithmetic<_K>::value || std::is_pointer<_K>::value)) ||
            (!(std::is_arithmetic<_V>::value || std::is_pointer<_V>::value)),
        void>::type
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        delete[] ptr;
    }
};

#ifdef USING_LOCAL_MMAP
#undef mymmap_alloc
#undef USING_LOCAL_MMAP
#endif

#ifdef USING_LOCAL_MUNMAP
#undef mymunmap
#undef USING_LOCAL_MUNMAP
#endif


//////////////////////////////////////////////////////////////////////
// Helpers for fht_table constructor
static uint32_t
log_b2(uint64_t n) {
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
roundup_next_p2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

//////////////////////////////////////////////////////////////////////
// Stats and Timing. These need to be moved to the top if you want to use them
//#define FHT_TIMER
//#define FHT_STATS

#ifdef FHT_TIMER
#include <math.h>
const uint32_t add_timer_idx    = 0;
const uint32_t find_timer_idx   = 1;
const uint32_t remove_timer_idx = 2;
const uint32_t resize_timer_idx = 3;

#undef FHT_TAKE_TIME
#define FHT_TAKE_TIME(X) this->timer.take_time(X);
const uint32_t MAX_TIMES = (1 << 23);

static int32_t
dblcomp(const void * a, const void * b) {
    return *(double *)b - *(double *)a;
}

struct fht_timer {
    uint64_t * timers[4];
    uint32_t   timer_idx[4];

    fht_timer() {
        for (uint32_t i = 0; i < 4; i++) {
            timers[i]    = (uint64_t *)calloc((MAX_TIMES), sizeof(uint64_t));
            timer_idx[i] = 0;
        }
    }
    ~fht_timer() {
        this->print_summary();
    }

    void
    print_stats(uint32_t idx) {
        assert((timer_idx[idx] & 0x1) == 0);
        const uint32_t ntimes = timer_idx[idx] / 2;
        uint64_t *     difs   = (uint64_t *)calloc(ntimes, sizeof(uint64_t));
        for (uint32_t i = 0; i < ntimes; i++) {
            difs[i] = timers[idx][2 * i + 1] - timers[idx][2 * i];
        }

        fprintf(stderr, "\tUnits : %s\n", "cycles");
        fprintf(stderr, "\tN     : %d\n", ntimes);
        fprintf(stderr, "\tMin   : %.3lf cycles\n", this->getMin(difs, ntimes));
        fprintf(stderr, "\tMax   : %.3lf cycles\n", this->getMax(difs, ntimes));
        fprintf(stderr,
                "\tMean  : %.3lf cycles\n",
                this->getMean(difs, ntimes));
        fprintf(stderr,
                "\tMed   : %.3lf cycles\n",
                this->getMedian(difs, ntimes));
        fprintf(stderr, "\tSD    : %.3lf cycles\n", this->getSD(difs, ntimes));
        fprintf(stderr, "\tVar   : %.3lf cycles\n", this->getVar(difs, ntimes));


        free(difs);
        free(timers[idx]);
    }

    void
    print_summary() {
        if (this->timer_idx[add_timer_idx]) {
            fprintf(stderr, "\rAdd Times\n");
            this->print_stats(add_timer_idx);
        }

        if (this->timer_idx[find_timer_idx]) {
            fprintf(stderr, "\rFind Times\n");
            this->print_stats(find_timer_idx);
        }

        if (this->timer_idx[remove_timer_idx]) {
            fprintf(stderr, "\rRemove Times\n");
            this->print_stats(remove_timer_idx);
        }

        if (this->timer_idx[resize_timer_idx]) {
            fprintf(stderr, "\rResize Times\n");
            this->print_stats(resize_timer_idx);
        }
    }

    void
    take_time(uint32_t idx) {
        this->timers[idx][timer_idx[idx]++] = grabTSC();
        if (timer_idx[idx] == MAX_TIMES) {
            timer_idx[idx] = 0;
        }
    }

    uint64_t
    grabTSC() {
        unsigned hi, lo;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return (((uint64_t)lo) | (((uint64_t)hi) << 32));
    }


    double
    getMedian(uint64_t * arr, uint32_t len) {
        if (len == 0 || arr == NULL) {
            assert(0);
        }
        double * arr_dbl = (double *)calloc(len, sizeof(double));
        for (uint32_t i = 0; i < len; i++) {
            arr_dbl[i] = (double)arr[i];
        }
        qsort(arr_dbl, len, sizeof(double), dblcomp);
        double median;
        if (len & 0x1) {
            median = arr_dbl[len >> 1];
        }
        else {
            median =
                (arr_dbl[(len - 1) >> 1] + arr_dbl[((len - 1) >> 1) + 1]) / 2.0;
        }
        free(arr_dbl);
        return median;
    }

    double
    getMean(uint64_t * arr, uint32_t len) {
        if (len == 0 || arr == NULL) {
            assert(0);
        }
        double total = 0.0;
        for (uint32_t i = 0; i < len; i++) {
            total += (double)arr[i];
        }
        return total / ((double)len);
    }

    double
    getSD(uint64_t * arr, uint32_t len) {
        if (len == 0 || arr == NULL) {
            assert(0);
        }
        if (len == 1) {
            return 0.0;
        }
        double sum = 0.0;
        double mean;
        double sd = 0.0;
        for (uint32_t i = 0; i < len; i++) {
            sum += (double)arr[i];
        }
        mean = sum / ((double)len);
        for (uint32_t i = 0; i < len; i++) {
            sd += pow(arr[i] - mean, 2);
        }
        return sqrt(sd / (len - 1));
    }

    double
    getVar(uint64_t * arr, uint32_t len) {
        assert(len);
        assert(arr);

        double sum = 0.0;
        double mean;
        double sd = 0.0;
        for (uint32_t i = 0; i < len; i++) {
            sum += (double)arr[i];
        }
        mean = sum / ((double)len);

        for (uint32_t i = 0; i < len; i++) {
            sd += pow(arr[i] - mean, 2);
        }
        return sqrt(sd / len);
    }


    double
    getMin(uint64_t * arr, uint32_t len) {
        if (len == 0 || arr == NULL) {
            assert(0);
        }
        double m = arr[0];
        for (uint32_t i = 0; i < len; i++)
            if (m > (double)arr[i]) {
                m = (double)arr[i];
            }

        return m;
    }


    double
    getMax(uint64_t * arr, uint32_t len) {
        if (len == 0 || arr == NULL) {
            assert(0);
        }
        double m = arr[0];
        for (uint32_t i = 0; i < len; i++)
            if (m < (double)arr[i]) {
                m = (double)arr[i];
            }
        return m;
    }
};

#endif

#ifdef FHT_STATS

#define FHT_STATS_INCR(X) this->stats_collector.X++

struct fht_stats {
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

    uint64_t add_found_possible_idx;
    uint64_t add_place_possible_idx;

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

    fht_stats() {
        memset(this, 0, sizeof(fht_stats));
    }
    ~fht_stats() {
        this->stats_summary();
    }
    double
    u64div(uint64_t num, uint64_t den) {
        return (double)(((double)num) / ((double)den));
    }
    void
    print_stat(std::string header, uint64_t val) {
        fprintf(stderr, "\t%s", header.c_str());
        for (uint32_t i = header.length(); i < 20; i++) {
            fprintf(stderr, " ");
        }
        fprintf(stderr, ": %lu\n", val);
    }
    void
    print_rate(std::string header, uint64_t num, uint64_t den) {
        fprintf(stderr, "\t -> %s", header.c_str());
        for (uint32_t i = header.length(); i < 16; i++) {
            fprintf(stderr, " ");
        }
        fprintf(stderr, ": %.3lf\n", this->u64div(num, den));
    }

    void
    stats_summary() {
        if (this->add_att) {
            fprintf(stderr, "\rAdd Stats\n");
            print_stat("Attempts", this->add_att);
            print_stat("Iter", this->add_iter);
            print_rate("Iter Avg", this->add_iter, this->add_att);
            print_stat("Resize Att", this->add_resize_att);
            print_stat("Resize Iter", this->add_resize_iter);
            print_stat("Found P Idx", this->add_found_possible_idx);
            print_rate("Found Rate",
                       this->add_found_possible_idx,
                       this->add_att);
            print_stat("Place P Idx", this->add_place_possible_idx);
            print_rate("Place Found Rate",
                       this->add_place_possible_idx,
                       this->add_found_possible_idx);
            print_rate("Place Raw Rate",
                       this->add_place_possible_idx,
                       this->add_att);
            print_stat("Duplicate", this->add_duplicate);
            print_rate("Duplicate Rate", this->add_duplicate, this->add_att);
            print_stat("Success", this->add_tag_match);
            print_rate("Success Rate", this->add_tag_match, this->add_att);
            print_stat("Removed", this->add_tag_removed);
            print_rate("Removed Rate", this->add_tag_removed, this->add_att);

            print_stat("Tag Match", this->add_tag_match);
            print_stat("Tag Fail", this->add_tag_fail);
            print_rate("Tag Fail Rate",
                       this->add_tag_fail,
                       this->add_tag_match);
            print_stat("Tag Success", this->add_tag_success);
            print_rate("Tag Success Rate",
                       this->add_tag_success,
                       this->add_tag_match);
        }
        if (this->find_att) {
            fprintf(stderr, "\rFind Stats\n");
            print_rate("Iter Avg", this->find_iter, this->find_att);
            print_stat("Complete", this->find_complete);
            print_rate("Complete Rate", this->find_complete, this->find_att);
            print_stat("Success", this->find_tag_match);
            print_rate("Success Rate", this->find_tag_match, this->find_att);
            print_stat("Removed", this->find_tag_removed);
            print_rate("Removed Rate", this->find_tag_removed, this->find_att);
            print_stat("Tag Match", this->find_tag_match);
            print_stat("Tag Fail", this->find_tag_fail);
            print_rate("Tag Fail Rate",
                       this->find_tag_fail,
                       this->find_tag_match);
            print_stat("Tag Success", this->find_tag_success);
            print_rate("Tag Success Rate",
                       this->find_tag_success,
                       this->find_tag_match);
        }
        if (this->remove_att) {
            fprintf(stderr, "\rRemove Stats\n");
            print_rate("Iter Avg", this->remove_iter, this->remove_att);
            print_stat("Complete", this->remove_complete);
            print_rate("Complete Rate",
                       this->remove_complete,
                       this->remove_att);
            print_stat("Success", this->remove_tag_match);
            print_rate("Success Rate",
                       this->remove_tag_match,
                       this->remove_att);
            print_stat("Removed", this->remove_tag_removed);
            print_rate("Removed Rate",
                       this->remove_tag_removed,
                       this->remove_att);
            print_stat("Tag Match", this->remove_tag_match);
            print_stat("Tag Fail", this->remove_tag_fail);
            print_rate("Tag Fail Rate",
                       this->remove_tag_fail,
                       this->remove_tag_match);
            print_stat("Tag Success", this->remove_tag_success);
            print_rate("Tag Success Rate",
                       this->remove_tag_success,
                       this->remove_tag_match);
        }
        if (this->resize_att) {
            fprintf(stderr, "\rResize Stats\n");
            print_rate("Sub Iter Avg",
                       this->resize_sub_iter,
                       this->resize_iter);
            print_stat("Invalid", this->resize_invalid);
            print_rate("Invalid Rate", this->resize_invalid, this->resize_iter);
            print_stat("Valid", this->resize_valid);
            print_rate("Valid Rate", this->resize_valid, this->resize_iter);
        }
    }
};

#endif
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Undefs
#undef FHT_STATS
#undef FHT_STATS_INCR
#undef FHT_TIMER
#undef FHT_TAKE_TIME
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
