#ifndef _FHT_HT_H_
#define _FHT_HT_H_

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
//#define FHT_STATS
#ifdef FHT_STATS
#define FHT_STATS_INCR(X) this->X++
#define FHT_STATS_SUMMARY this->stats_summary()
#else
#define FHT_STATS_INCR(X)
#define FHT_STATS_SUMMARY
#endif


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

#undef mymmap_alloc
#undef mymunmap

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
// Table class

template<typename K,
         typename V,
         typename Hasher    = DEFAULT_HASH_32<K>,
         typename Allocator = DEFAULT_MMAP_ALLOC<K, V>>
class fht_table {
    fht_chunk<K, V> * chunks;
    uint32_t          log_incr;
    Hasher            hash_32;
    Allocator         alloc_mmap;

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
    double   u64div(uint64_t num, uint64_t den);
    void     stats_summary();
#endif


    fht_chunk<K, V> * const resize();

    template<typename T>
    using pass_type_t = typename std::
        conditional<(std::is_arithmetic<T>::value || std::is_pointer<T>::value), const T, T const &>::type;

   public:
    fht_table(uint32_t init_size);
    fht_table();
    ~fht_table();


    uint32_t add(pass_type_t<K> new_key, pass_type_t<V> new_val);
    uint32_t find(pass_type_t<K> key);
    uint32_t remove(pass_type_t<K> key);
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
template<typename K, typename V, typename Hasher, typename Allocator>
fht_table<K, V, Hasher, Allocator>::fht_table(const uint32_t init_size) {
#ifdef FHT_STATS
    memset(this, 0, sizeof(fht_table));
#endif
    const uint64_t _init_size =
        init_size > FHT_DEFAULT_INIT_SIZE
            ? (init_size ? roundup_next_p2(init_size) : FHT_DEFAULT_INIT_SIZE)
            : FHT_DEFAULT_INIT_SIZE;

    const uint32_t _log_init_size = log_b2(_init_size);

    this->chunks =
        this->alloc_mmap.init_mem((_init_size / FHT_NODES_PER_CACHE_LINE));

    this->log_incr = _log_init_size;
}
template<typename K, typename V, typename Hasher, typename Allocator>
fht_table<K, V, Hasher, Allocator>::fht_table()
    : fht_table(FHT_DEFAULT_INIT_SIZE) {}

template<typename K, typename V, typename Hasher, typename Allocator>
fht_table<K, V, Hasher, Allocator>::~fht_table() {
    FHT_STATS_SUMMARY;
    this->alloc_mmap.deinit_mem(
        this->chunks,
        ((1 << (this->log_incr)) / FHT_NODES_PER_CACHE_LINE));
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Resize
template<typename K, typename V, typename Hasher, typename Allocator>
fht_chunk<K, V> * const
fht_table<K, V, Hasher, Allocator>::resize() {
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

    return new_chunks;
}

//////////////////////////////////////////////////////////////////////
// Add Key Val
template<typename K, typename V, typename Hasher, typename Allocator>
uint32_t
fht_table<K, V, Hasher, Allocator>::add(pass_type_t<K> new_key,
                                        pass_type_t<V> new_val) {
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
                return FHT_ADDED;
            }
            else {
                FHT_STATS_INCR(add_tag_match);
                tags[test_idx] = (tag | VALID_MASK);
                SET_KEY_VAL(nodes[test_idx], new_key, new_val);
                return FHT_ADDED;
            }
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
            return FHT_ADDED;
        }
    }

    // probability of this is 1 / (2 ^ FHT_SEARCH_NUMBER)
    assert(0);
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Find
template<typename K, typename V, typename Hasher, typename Allocator>
uint32_t
fht_table<K, V, Hasher, Allocator>::find(pass_type_t<K> key) {
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
template<typename K, typename V, typename Hasher, typename Allocator>
uint32_t
fht_table<K, V, Hasher, Allocator>::remove(pass_type_t<K> key) {
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
template<typename K, typename V, typename Hasher, typename Allocator>
double
fht_table<K, V, Hasher, Allocator>::u64div(uint64_t num, uint64_t den) {
    return (double)(((double)num) / ((double)den));
}

template<typename K, typename V, typename Hasher, typename Allocator>
void
fht_table<K, V, Hasher, Allocator>::stats_summary() {
    if (this->add_att) {
        fprintf(stderr, "\rAdd Stats\n");
        fprintf(stderr, "\tAttempts     : %lu\n", this->add_att);
        fprintf(stderr, "\tIter         : %lu\n", this->add_iter);
        fprintf(stderr,
                "\t\tIter Avg         : %.3lf\n",
                this->u64div(this->add_iter, this->add_att));
        fprintf(stderr, "\tResize Att   : %lu\n", this->add_resize_att);
        fprintf(stderr, "\tResize Iter  : %lu\n", this->add_resize_iter);
        fprintf(stderr, "\tFound P Idx  : %lu\n", this->add_found_possible_idx);
        fprintf(stderr,
                "\t\tFound Rate       : %.3lf\n",
                this->u64div(this->add_found_possible_idx, this->add_att));
        fprintf(stderr, "\tPlace P Idx  : %lu\n", this->add_place_possible_idx);
        fprintf(stderr,
                "\t\tPlace Found Rate : %.3lf\n",
                this->u64div(this->add_place_possible_idx,
                             this->add_found_possible_idx));
        fprintf(stderr,
                "\t\tPlace Raw Rate   : %.3lf\n",
                this->u64div(this->add_place_possible_idx, this->add_att));
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
// Undefs
#undef FHT_STATS
#undef FHT_STATS_INCR
#undef FHT_STATS_SUMMARY
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
