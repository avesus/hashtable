#ifndef _FHT_HT_H_
#define _FHT_HT_H_

/* Todos
1) Recover perf loss from fht_chunk refactor
2) Optimize resize
*/


//////////////////////////////////////////////////////////////////////
// Macros
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
// Calculating Tag and Start Idx and chunk from a raw hash
#define HASH_TO_IDX(hash_val, tbl_log)                                         \
    ((hash_val & TO_MASK(tbl_log)) / FHT_NODES_PER_CACHE_LINE)
#define GEN_TAG(hash_val) (((hash_val) << CONTENT_OFFSET))
#define GEN_START_IDX(hash_val)                                                \
    ((hash_val) >> (32 - FHT_LOG_NODES_PER_CACHE_LINE))
//////////////////////////////////////////////////////////////////////
// For getting test idx from start_idx and j in loop
#define IDX_MOD(idx)             (3 * (idx))
#define GEN_TEST_IDX(start, idx) (((start) ^ IDX_MOD(idx)) & FHT_CACHE_IDX_MASK)
//////////////////////////////////////////////////////////////////////


#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string>
#include <type_traits>


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

// needs to be #define for compile time comparisons
#define FHT_PASS_BY_VAL_THRESH 8

// for optimized layout of node struct
#define FHT_SEPERATE_THRESH 8

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
struct fht_seperate_kv {
    K keys[L1_CACHE_LINE_SIZE];
    V vals[L1_CACHE_LINE_SIZE];
};

template<typename K, typename V>
struct fht_combined_kv {
    fht_node<K, V> nodes[L1_CACHE_LINE_SIZE];
};

template<typename K, typename V>
struct fht_chunk {

    template<typename T>
    using pass_type_t =
        typename std::conditional<(sizeof(T) <= FHT_PASS_BY_VAL_THRESH),
                                  const T,
                                  T const &>::type;

    template<typename _K = K, typename _V = V>
    using local_node_t =
        typename std::conditional<(sizeof(_K) <= FHT_SEPERATE_THRESH),
                                  fht_seperate_kv<_K, _V>,
                                  fht_combined_kv<_K, _V>>::type;


    typedef pass_type_t<K>     key_pass_t;
    typedef pass_type_t<V>     val_pass_t;
    typedef local_node_t<K, V> node_t;

    tag_type_t tags[L1_CACHE_LINE_SIZE];
    node_t     nodes;

    void
    undelete_tag_n(const uint32_t n) {
        SET_UNDELETED(this->tags[n]);
    }

    void
    delete_tag_n(const uint32_t n) {
        SET_DELETED(this->tags[n]);
    }

    // the following exist for key/val in a far more complicated format
    constexpr const tag_type_t
    get_tag_n(const uint32_t n) const {
        return this->tags[n];
    }

    constexpr tag_type_t * const
    get_tag_n_ptr(const uint32_t n) const {
        return (tag_type_t * const)(&(this->tags[n]));
    }

    //////////////////////////////////////////////////////////////////////
    // <= FHT_SEPERATE_THRESH byte value methods
    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) <= FHT_SEPERATE_THRESH),
                                      key_pass_t>::type
    get_key_n(const uint32_t n) const {
        return this->nodes.keys[n];
    }

    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) <= FHT_SEPERATE_THRESH),
                                      const uint32_t>::type
    compare_key_n(const uint32_t n, key_pass_t other_key) const {
        return this->nodes.keys[n] == other_key;
    }

    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) <= FHT_SEPERATE_THRESH),
                                      val_pass_t>::type
    get_val_n(const uint32_t n) const {
        return this->nodes.vals[n];
    }


    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) <= FHT_SEPERATE_THRESH),
                                      K * const>::type
    get_key_n_ptr(const uint32_t n) const {
        return (K * const)(&(this->nodes.keys[n]));
    }

    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) <= FHT_SEPERATE_THRESH),
                                      V * const>::type
    get_val_n_ptr(const uint32_t n) const {
        return (V * const)(&(this->nodes.vals[n]));
    }

    template<typename _K = K, typename _V = V>
    typename std::enable_if<(sizeof(_K) <= FHT_SEPERATE_THRESH), void>::type
    set_key_val_tag(const uint32_t   n,
                    key_pass_t       new_key,
                    val_pass_t       new_val,
                    const tag_type_t tag) {

        this->tags[n]       = tag;
        this->nodes.keys[n] = new_key;
        this->nodes.vals[n] = new_val;
    }


    //////////////////////////////////////////////////////////////////////
    // Non FHT_SEPERATE_THRESH byte value methods
    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) > FHT_SEPERATE_THRESH),
                                      key_pass_t>::type
    get_key_n(const uint32_t n) const {
        return this->nodes.nodes[n].key;
    }

    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) > FHT_SEPERATE_THRESH),
                                      const uint32_t>::type
    compare_key_n(const uint32_t n, key_pass_t other_key) const {
        return this->nodes.nodes[n].key == other_key;
    }


    template<typename _K = K, typename _V = V>
    constexpr typename std::enable_if<(sizeof(_K) > FHT_SEPERATE_THRESH),
                                      val_pass_t>::type
    get_val_n(const uint32_t n) const {
        return this->nodes.nodes[n].val;
    }


    template<typename _K = K, typename _V = V>
    constexpr
        typename std::enable_if<(sizeof(_K) > FHT_SEPERATE_THRESH), K * const>::type
        get_key_n_ptr(const uint32_t n) const {
        return (K * const)(&(this->nodes.nodes[n].key));
    }

    template<typename _K = K, typename _V = V>
    constexpr
        typename std::enable_if<(sizeof(_K) > FHT_SEPERATE_THRESH), V * const>::type
        get_val_n_ptr(const uint32_t n) const {
        return (V * const)(&(this->nodes.nodes[n].val));
    }


    template<typename _K = K, typename _V = V>
    typename std::enable_if<(sizeof(_K) > FHT_SEPERATE_THRESH), void>::type
    set_key_val_tag(const uint32_t   n,
                    key_pass_t       new_key,
                    val_pass_t       new_val,
                    const tag_type_t tag) {
        this->tags[n]            = tag;
        this->nodes.nodes[n].key = new_key;
        this->nodes.nodes[n].val = new_val;
    }
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


    fht_chunk<K, V> * const resize();

    template<typename T>
    using pass_type_t =
        typename std::conditional<(sizeof(T) <= FHT_PASS_BY_VAL_THRESH),
                                  const T,
                                  T const &>::type;

    using key_pass_t = typename fht_chunk<K, V>::key_pass_t;
    using val_pass_t = typename fht_chunk<K, V>::val_pass_t;
    using ret_type_t = typename Returner::ret_type_t;

   public:
    fht_table(uint32_t init_size);
    fht_table();
    ~fht_table();


    const uint32_t add(key_pass_t new_key, val_pass_t new_val);
    const uint32_t find(key_pass_t key, ret_type_t store_val);
    const uint32_t remove(key_pass_t key);
};


//////////////////////////////////////////////////////////////////////
// Actual Implemenation cuz templating kinda sucks imo


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

    const uint32_t                _new_log_incr = ++(this->log_incr);
    const fht_chunk<K, V> * const old_chunks    = this->chunks;


    const uint32_t _num_chunks =
        (1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE;

    fht_chunk<K, V> * const new_chunks =
        this->alloc_mmap.init_mem(2 * _num_chunks);

    // set this while its definetly still in cache
    this->chunks = new_chunks;


    for (uint32_t i = 0; i < _num_chunks; i++) {
        const fht_chunk<K, V> * const old_chunk = old_chunks + i;
        for (uint32_t j = 0; j < FHT_NODES_PER_CACHE_LINE; j++) {
            if (RESIZE_SKIP(old_chunk->get_tag_n(j))) {
                continue;
            }

            const tag_type_t tag = old_chunk->get_tag_n(j);

            const uint32_t raw_slot  = this->hash_32(old_chunk->get_key_n(j));
            const uint32_t start_idx = GEN_START_IDX(raw_slot);

            fht_chunk<K, V> * const new_chunk =
                new_chunks +
                (i |
                 (GET_NTH_BIT(raw_slot, _new_log_incr - 1) ? _num_chunks : 0));

            for (uint32_t new_j = 0; new_j < FHT_SEARCH_NUMBER; new_j++) {
                const uint32_t test_idx = GEN_TEST_IDX(start_idx, new_j);
                if (__builtin_expect(!IS_VALID(new_chunk->get_tag_n(test_idx)),
                                     1)) {
                    new_chunk->set_key_val_tag(test_idx,
                                               old_chunk->get_key_n(j),
                                               old_chunk->get_val_n(j),
                                               tag);
                    break;
                }
            }
        }
    }

    this->alloc_mmap.deinit_mem(
        (fht_chunk<K, V> * const)old_chunks,
        ((1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE));

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
fht_table<K, V, Returner, Hasher, Allocator>::add(key_pass_t new_key,
                                                  val_pass_t new_val) {
    const uint32_t          _log_incr = this->log_incr;
    const uint32_t          raw_slot  = this->hash_32(new_key);
    fht_chunk<K, V> * const chunk     = (fht_chunk<K, V> * const)(
        (this->chunks) + (HASH_TO_IDX(raw_slot, _log_incr)));

    const tag_type_t tag       = GEN_TAG(raw_slot);
    const uint32_t   start_idx = GEN_START_IDX(raw_slot);

    // the prefetch on nodes is particularly important
    __builtin_prefetch(chunk->get_tag_n_ptr(start_idx & FHT_CACHE_IDX_MASK));
    __builtin_prefetch(chunk->get_key_n_ptr((start_idx & FHT_CACHE_IDX_MASK)));

    uint32_t possible_idx = FHT_NODES_PER_CACHE_LINE;
    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);

        if (__builtin_expect(!IS_VALID(chunk->get_tag_n(test_idx)), 1)) {
            if (possible_idx != FHT_NODES_PER_CACHE_LINE) {
                chunk->set_key_val_tag(possible_idx,
                                       new_key,
                                       new_val,
                                       tag | VALID_MASK);
                return FHT_ADDED;
            }
            else {
                chunk->set_key_val_tag(test_idx,
                                       new_key,
                                       new_val,
                                       tag | VALID_MASK);
                return FHT_ADDED;
            }
        }
        else if ((GET_CONTENT(chunk->get_tag_n(test_idx)) == tag) &&
                 (chunk->compare_key_n(test_idx, new_key))) {
            if (IS_DELETED(chunk->get_tag_n(test_idx))) {
                chunk->undelete_tag_n(test_idx);
                return FHT_ADDED;
            }
            return FHT_NOT_ADDED;
        }

        if (possible_idx == FHT_NODES_PER_CACHE_LINE &&
            IS_DELETED(chunk->get_tag_n(test_idx))) {
            possible_idx = test_idx;
        }
    }

    if (possible_idx != FHT_NODES_PER_CACHE_LINE) {
        chunk->set_key_val_tag(possible_idx,
                               new_key,
                               new_val,
                               tag | VALID_MASK);
        return FHT_ADDED;
    }

    // no valid slot found so resize
    fht_chunk<K, V> * const new_chunk = (fht_chunk<K, V> * const)(
        this->resize() + HASH_TO_IDX(raw_slot, _log_incr + 1));


    // after resize add without duplication check
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(new_chunk->get_tag_n(test_idx)), 1)) {
            new_chunk->set_key_val_tag(test_idx,
                                       new_key,
                                       new_val,
                                       tag | VALID_MASK);
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
fht_table<K, V, Returner, Hasher, Allocator>::find(key_pass_t key,
                                                   ret_type_t store_val) {
    const uint32_t                _log_incr = this->log_incr;
    const uint32_t                raw_slot  = this->hash_32(key);
    const fht_chunk<K, V> * const chunk     = (const fht_chunk<K, V> * const)(
        (this->chunks) + (HASH_TO_IDX(raw_slot, _log_incr)));

    // by setting valid here we can remove delete check
    const tag_type_t tag       = GEN_TAG(raw_slot) | VALID_MASK;
    const uint32_t   start_idx = GEN_START_IDX(raw_slot);


    // the prefetch on nodes is particularly important
    __builtin_prefetch(chunk->get_tag_n_ptr(start_idx & FHT_CACHE_IDX_MASK));
    __builtin_prefetch(chunk->get_key_n_ptr(start_idx & FHT_CACHE_IDX_MASK));

    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(chunk->get_tag_n(test_idx)), 0)) {
            return FHT_NOT_FOUND;
        }
        else if ((chunk->get_tag_n(test_idx) == tag) &&
                 chunk->compare_key_n(test_idx, key)) {
            this->returner.to_ret_type(store_val,
                                       chunk->get_val_n_ptr(test_idx));
            return FHT_FOUND;
        }
    }
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
fht_table<K, V, Returner, Hasher, Allocator>::remove(key_pass_t key) {
    const uint32_t          _log_incr = this->log_incr;
    const uint32_t          raw_slot  = this->hash_32(key);
    fht_chunk<K, V> * const chunk     = (fht_chunk<K, V> * const)(
        (this->chunks) + (HASH_TO_IDX(raw_slot, _log_incr)));


    const tag_type_t tag       = GEN_TAG(raw_slot) | VALID_MASK;
    const uint32_t   start_idx = GEN_START_IDX(raw_slot);

    __builtin_prefetch(chunk->get_tag_n_ptr(start_idx & FHT_CACHE_IDX_MASK));
    __builtin_prefetch(chunk->get_key_n_ptr(start_idx & FHT_CACHE_IDX_MASK));


    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (__builtin_expect(!IS_VALID(chunk->get_tag_n(test_idx)), 0)) {
            return FHT_NOT_DELETED;
        }
        else if ((chunk->get_tag_n(test_idx) == tag) &&
                 chunk->compare_key_n(test_idx, key)) {
            chunk->delete_tag_n(test_idx);
            return FHT_DELETED;
        }
    }
    return FHT_NOT_FOUND;
}


//////////////////////////////////////////////////////////////////////
// Default classes


template<typename V>
struct DEFAULT_RETURNER {

    template<typename _V = V>
    using local_ret_type_t = typename std::
        conditional<(sizeof(_V) <= FHT_PASS_BY_VAL_THRESH), _V *, _V **>::type;

    typedef local_ret_type_t<V> ret_type_t;

    // this is case where builtin type is passed (imo ptr counts as thats
    // basically uint64)
    template<typename _V = V>
    typename std::enable_if<(sizeof(_V) <= FHT_PASS_BY_VAL_THRESH), void>::type
    to_ret_type(ret_type_t store_val, V const * val) const {
        if (store_val) {
            *store_val = *val;
        }
    }

    // this is case where ** is passed (bigger types)
    template<typename _V = V>
    typename std::enable_if<(sizeof(_V) > FHT_PASS_BY_VAL_THRESH), void>::type
    to_ret_type(ret_type_t store_val, V * val) const {
        if (store_val) {
            *store_val = val;
        }
    }
};

template<typename V>
struct REF_RETURNER {

    template<typename _V = V>
    using local_ret_type_t = V &;
    typedef local_ret_type_t<V> ret_type_t;

    void
    to_ret_type(ret_type_t store_val, V const * val) const {
        if (store_val) {
            store_val = *val;
        }
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
        if (store_val) {
            *store_val = *val;
        }
    }
};

template<typename V>
struct PTR_PTR_RETURNER {

    template<typename _V = V>
    using local_ret_type_t = V **;
    typedef local_ret_type_t<V> ret_type_t;

    void
    to_ret_type(ret_type_t store_val, V * val) const {
        if (store_val) {
            *store_val = val;
        }
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
// Undefs
#undef FHT_PASS_BY_VAL_THRESH
#undef FHT_SEPERATE_THRESH
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
