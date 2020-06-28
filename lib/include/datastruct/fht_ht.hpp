#ifndef _FHT_HT_H_
#define _FHT_HT_H_


/* Todos
1) Find way to only have 1 find/remove w.o perf loss
2) Optimize resize
*/

// if using big pages might want to use a seperate allocator and redefine
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

// make sure these are correct. Something like $> cat /proc/cpuinfo should give
// you everything you need
#ifndef L1_CACHE_LINE_SIZE

#define L1_CACHE_LINE_SIZE     64
#define L1_LOG_CACHE_LINE_SIZE 6

#endif

//////////////////////////////////////////////////////////////////////
// Table params
// return values
const uint64_t FHT_NOT_ERASED = 0;
const uint64_t FHT_ERASED     = 1;


// tunable
//#define DESTROYABLE_INSERT
#ifdef DESTROYABLE_INSERT
#define SRC_WRAPPER(X) std::move(X)
#else
#define SRC_WRAPPER(X) (X)
#endif

#define NEW(type, dst, src) (new ((void *)(&(dst))) type(src))


// when to change pass by from actual value to reference
#define FHT_PASS_BY_VAL_THRESH 8

// for optimized layout of node struct. Keys with size <= get different layout
// than bigger key vals
#define FHT_SEPERATE_THRESH 8

// these will use the "optimized" find/remove. Basically I find this is
// important with string keys and thats about all. Seperate types with ,
// worth noting generally insert heavy does better with special.
struct _fht_empty_t {};
#define FHT_SPECIAL_TYPES std::string, _fht_empty_t


// tunable but less important

// max memory willing to use (this doesn't really have effect with default
// allocator)
const uint64_t FHT_DEFAULT_INIT_MEMORY = ((1UL) << 35);

// default init size (since mmap is backend for allocation less than page size
// has no effect)
const uint32_t FHT_DEFAULT_INIT_SIZE = PAGE_SIZE;

// literally does not matter unless you care about universal hashing or have
// some fancy shit in mind
const uint32_t FHT_HASH_SEED = 0;


//////////////////////////////////////////////////////////////////////
// Macros that are not really tunable
#include "FHT_HELPER_MACROS.h"
#include "FHT_SPECIAL_TYPE_MACROS.h"

// necessary includes
#include <assert.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <sys/mman.h>
#include <string>
#include <type_traits>


#define FHT_MM_SET(X)          _mm_set1_epi8(X)
#define FHT_MM_MASK(X, Y)      _mm_movemask_epi8(_mm_cmpeq_epi8(X, Y))
#define FHT_MM_EMPTY(X)        _mm_movemask_epi8(_mm_sign_epi8(X, X))
#define FHT_MM_EMPTY_OR_DEL(X) _mm_movemask_epi8(X)


static const __m256i FHT_RESET_VEC = _mm256_set1_epi8(0x80);


static const uint32_t FHT_MM_LINE = FHT_NODES_PER_CACHE_LINE / sizeof(__m128i);
static const uint32_t FHT_MM_LINE_MASK = FHT_MM_LINE - 1;

static const uint32_t FHT_MM_IDX_MULT = FHT_NODES_PER_CACHE_LINE / FHT_MM_LINE;
static const uint32_t FHT_MM_IDX_MASK = FHT_MM_IDX_MULT - 1;


//////////////////////////////////////////////////////////////////////
// forward declaration of default helper struct
// near their implementations below are some alternatives ive written

// depending on type chooses from a few optimized hash functions. Nothing too
// fancy.
// Must define:
// const uint32_t operator()(K) or const uint32_t operator()(K const &)
template<typename K>
struct DEFAULT_HASH_32;

// const uint64_t operator()(K) or const uint64_t operator()(K const &)
template<typename K>
struct DEFAULT_HASH_64;

// if both K and V don't require a real constructor (i.e an int or really any C
// type) it will alloc with mmap and NOT define new (new is slower because even
// if constructor is unnecissary still wastes some time). If type is not builtin
// new is used though allocation backend is still mmap. If you write your own
// allocator be sure that is 1) 0s out the returned memory (this is necessary
// for correctness) and 2) returns at the very least cache line aligned memory
// (this is necessary for performance)
// Must define:
// fht_chunk<K, V> * const init_mem(const size_t)
// deinit_mem(fht_chunk<K, V> * const, const size_t size)

// If table type (value or key) requires a constructor (i.e most non primitive
// classes) must also define:
// void * new(size_t size)
// void * new[](size_t size)
// delete(void * ptr, const uint32_t size)
// delete[](void * ptr, const uint32_t size)

template<typename K, typename V>
struct DEFAULT_MMAP_ALLOC;

// this will perform significantly better if the copy time on either keys or
// values is large (it tries to avoid at least a portion of the copying step in
// resize)
template<typename K, typename V>
struct INPLACE_MMAP_ALLOC;

//////////////////////////////////////////////////////////////////////
// helpers
inline uint64_t
log_b2(uint64_t n) {
    uint64_t s, t;
    t = (n > 0xffffffffffffffff) << 6;
    n >>= t;
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

inline uint64_t
roundup_next_p2(uint64_t v) {
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

// these get optimized to popcnt. For resize
static inline uint32_t
bitcount_32(uint32_t v) {
    uint32_t c;
    c = v - ((v >> 1) & 0x55555555);
    c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
    c = ((c >> 4) + c) & 0x0F0F0F0F;
    c = ((c >> 8) + c) & 0x00FF00FF;
    c = ((c >> 16) + c) & 0x0000FFFF;
    return c;
}


//////////////////////////////////////////////////////////////////////
// Really just typedef structs
template<typename K, typename V>
struct fht_node {
    K key;
    V val;
};

// node layout for smaller keys
template<typename K, typename V>
struct fht_seperate_kv {
    K keys[L1_CACHE_LINE_SIZE];
    V vals[L1_CACHE_LINE_SIZE];
};


// node layout for larger keys
template<typename K, typename V>
struct fht_combined_kv {
    fht_node<K, V> nodes[L1_CACHE_LINE_SIZE];
};

// chunk containing cache line of tags and a single node (either fht_combined_kv
// or fht_seperate_kv). Either way each chunk contains bytes per cache line
// number of key value pairs (on most 64 bit machines this will mean 64 Key
// value pairs)
template<typename K, typename V>
struct fht_chunk {


    // determine best way to pass K/V depending on size. Generally passing
    // values machine word size is ill advised
    template<typename T>
    using pass_type_t =
        typename std::conditional<(std::is_arithmetic<T>::value ||
                                   std::is_pointer<T>::value),
                                  const T,
                                  T const &>::type;

    // determine node type based on K/V
    template<typename _K = K, typename _V = V>
    using _node_t =
        typename std::conditional<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                   sizeof(_K) <= FHT_SEPERATE_THRESH),
                                  fht_seperate_kv<_K, _V>,
                                  fht_combined_kv<_K, _V>>::type;


    // typedefs to fht_table can access these variables
    typedef pass_type_t<K> key_pass_t;
    typedef pass_type_t<V> val_pass_t;
    typedef _node_t<K, V>  node_t;

    // actual content of chunk
    __m128i tags_vec[FHT_MM_LINE];
    node_t  nodes;

    inline constexpr uint32_t __attribute__((always_inline))
    get_del(const uint32_t idx) {
        return FHT_MM_EMPTY_OR_DEL(this->tags_vec[idx]);
    }

    inline constexpr uint32_t __attribute__((always_inline))
    get_empty_or_del(const uint32_t idx) {
        return FHT_MM_EMPTY_OR_DEL(this->tags_vec[idx]);
    }

    inline constexpr uint32_t __attribute__((always_inline))
    get_empty(const uint32_t idx) {
        return FHT_MM_EMPTY(this->tags_vec[idx]);
    }

    inline constexpr uint32_t __attribute__((always_inline))
    is_deleted_n(const uint32_t n) {
        return IS_DELETED(((const int8_t * const)this->tags_vec)[n]);
    }

    inline constexpr uint32_t __attribute__((always_inline))
    is_invalid_n(const uint32_t n) {
        return IS_INVALID(((const int8_t * const)this->tags_vec)[n]);
    }

    inline void __attribute__((always_inline)) delete_tag_n(const uint32_t n) {
        SET_DELETED(((int8_t * const)this->tags_vec)[n]);
    }

    // this undeletes
    inline void __attribute__((always_inline))
    set_tag_n(const uint32_t n, const int8_t new_tag) {
        ((int8_t * const)this->tags_vec)[n] = new_tag;
    }


    // the following exist for key/val in a far more complicated format
    inline constexpr const int8_t __attribute__((always_inline))
    get_tag_n(const uint32_t n) const {
        return ((int8_t * const)this->tags_vec)[n];
    }


    // overloaded key/value helpers
    //////////////////////////////////////////////////////////////////////
    // <= FHT_SEPERATE_THRESH byte value methods
    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                 sizeof(_K) <= FHT_SEPERATE_THRESH),
                                key_pass_t>::type __attribute__((always_inline))
        get_key_n(const uint32_t n) const {
        return this->nodes.keys[n];
    }

    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                 sizeof(_K) <= FHT_SEPERATE_THRESH),
                                const uint32_t>::type
        __attribute__((always_inline))
        compare_key_n(const uint32_t n, key_pass_t other_key) const {
        return this->nodes.keys[n] == other_key;
    }

    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                 sizeof(_K) <= FHT_SEPERATE_THRESH),
                                val_pass_t>::type __attribute__((always_inline))
        get_val_n(const uint32_t n) const {
        return this->nodes.vals[n];
    }


    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                 sizeof(_K) <= FHT_SEPERATE_THRESH),
                                const K * const>::type
        __attribute__((always_inline)) get_key_n_ptr(const uint32_t n) const {
        return (const K * const)(&(this->nodes.keys[n]));
    }

    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                 sizeof(_K) <= FHT_SEPERATE_THRESH),
                                V * const>::type __attribute__((always_inline))
        get_val_n_ptr(const uint32_t n) const {
        return (V * const)(&(this->nodes.vals[n]));
    }

    template<typename _K = K, typename _V = V>
    inline typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                    sizeof(_K) <= FHT_SEPERATE_THRESH &&
                                    sizeof(_V) <= FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    val_pass_t     new_val) {

        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.keys[n], SRC_WRAPPER(new_key));
        NEW(V, this->nodes.vals[n], SRC_WRAPPER(new_val));
    }


    template<typename _K = K, typename _V = V, typename... Args>
    inline typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                    sizeof(_K) <= FHT_SEPERATE_THRESH &&
                                    sizeof(_V) > FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    Args &&... args) {
        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.keys[n], SRC_WRAPPER(new_key));
        NEW(V, this->nodes.vals[n], std::forward<Args>(args)...);
    }

    template<typename _K = K, typename _V = V>
    inline typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                    sizeof(_K) <= FHT_SEPERATE_THRESH &&
                                    sizeof(_V) > FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    const V &      new_val) {
        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.keys[n], SRC_WRAPPER(new_key));
        NEW(V, this->nodes.vals[n], SRC_WRAPPER(new_val));
    }

    template<typename _K = K, typename _V = V>
    inline typename std::enable_if<(FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES) &&
                                    sizeof(_K) <= FHT_SEPERATE_THRESH &&
                                    sizeof(_V) > FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    V &            new_val) {
        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.keys[n], SRC_WRAPPER(new_key));
        NEW(V, this->nodes.vals[n], SRC_WRAPPER(new_val));
    }


    //////////////////////////////////////////////////////////////////////
    // Non FHT_SEPERATE_THRESH byte value methods
    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                 sizeof(_K) > FHT_SEPERATE_THRESH),
                                key_pass_t>::type __attribute__((always_inline))
        get_key_n(const uint32_t n) const {
        return this->nodes.nodes[n].key;
    }

    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                 sizeof(_K) > FHT_SEPERATE_THRESH),
                                const uint32_t>::type
        __attribute__((always_inline))
        compare_key_n(const uint32_t n, key_pass_t other_key) const {
        return this->nodes.nodes[n].key == other_key;
    }


    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                 sizeof(_K) > FHT_SEPERATE_THRESH),
                                val_pass_t>::type __attribute__((always_inline))
        get_val_n(const uint32_t n) const {
        return this->nodes.nodes[n].val;
    }


    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                 sizeof(_K) > FHT_SEPERATE_THRESH),
                                const K * const>::type
        __attribute__((always_inline)) get_key_n_ptr(const uint32_t n) const {
        return (const K * const)(&(this->nodes.nodes[n].key));
    }

    template<typename _K = K, typename _V = V>
    inline constexpr
        typename std::enable_if<(FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                 sizeof(_K) > FHT_SEPERATE_THRESH),
                                V * const>::type __attribute__((always_inline))
        get_val_n_ptr(const uint32_t n) const {
        return (V * const)(&(this->nodes.nodes[n].val));
    }


    template<typename _K = K, typename _V = V>
    inline typename std::enable_if<((FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                     sizeof(_K) > FHT_SEPERATE_THRESH) &&
                                    sizeof(_V) <= FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    val_pass_t     new_val) {
        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.nodes[n].key, SRC_WRAPPER(new_key));
        NEW(V, this->nodes.nodes[n].val, SRC_WRAPPER(new_val));
    }

    template<typename _K = K, typename _V = V, typename... Args>
    inline typename std::enable_if<((FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                     sizeof(_K) > FHT_SEPERATE_THRESH) &&
                                    sizeof(_V) > FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    Args &&... args) {
        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.nodes[n].key, SRC_WRAPPER(new_key));
        NEW(V, this->nodes.nodes[n].val, std::forward<Args>(args)...);
    }

    template<typename _K = K, typename _V = V>
    inline typename std::enable_if<((FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                     sizeof(_K) > FHT_SEPERATE_THRESH) &&
                                    sizeof(_V) > FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    const V &      new_val) {

        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.nodes[n].key, SRC_WRAPPER(new_key));
        NEW(V, this->nodes.nodes[n].val, SRC_WRAPPER(new_val));
    }

    template<typename _K = K, typename _V = V>
    inline typename std::enable_if<((FHT_IS_SPECIAL(FHT_SPECIAL_TYPES) ||
                                     sizeof(_K) > FHT_SEPERATE_THRESH) &&
                                    sizeof(_V) > FHT_PASS_BY_VAL_THRESH),
                                   void>::type __attribute__((always_inline))
    set_key_val_tag(const uint32_t n,
                    const int8_t   tag,
                    key_pass_t     new_key,
                    V &            new_val) {
        ((int8_t * const)this->tags_vec)[n] = tag;
        NEW(K, this->nodes.nodes[n].key, SRC_WRAPPER(new_key));
        NEW(V, this->nodes.nodes[n].val, SRC_WRAPPER(new_val));
    }
};
//////////////////////////////////////////////////////////////////////
// Table class
template<typename K,
         typename V,
         typename Hasher    = DEFAULT_HASH_64<K>,
         typename Allocator = INPLACE_MMAP_ALLOC<K, V>>
struct fht_table {


    // log of table size
    uint32_t log_incr;

    // chunk array
    fht_chunk<K, V> * chunks;

    // helper classes
    Hasher    hash;
    Allocator alloc_mmap;

    //////////////////////////////////////////////////////////////////////
    // very basic info
    inline constexpr bool
    empty() const {
        return !(this->npairs);
    }
    inline constexpr uint64_t
    size() const {
        return this->npairs;
    }
    inline constexpr uint64_t
    max_size() const {
        return (1UL) << this->log_size;
    }

    inline constexpr double
    load_factor() const {
        return ((double)this->size()) / ((double)this->max_size());
    }

    // aint gunna do anything about this
    inline constexpr double
    max_load_factor() const {
        return 1.0;
    }


    //////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////
    // stuff related to adding elements

    // in place resize, only works with INPLACE allocator
    template<typename _K         = K,
             typename _V         = V,
             typename _Hasher    = Hasher,
             typename _Allocator = Allocator>
    typename std::enable_if<
        std::is_same<_Allocator, INPLACE_MMAP_ALLOC<_K, _V>>::value,
        void>::type
    resize();

    // standard resize which copies all elements
    template<typename _K         = K,
             typename _V         = V,
             typename _Hasher    = Hasher,
             typename _Allocator = Allocator>
    typename std::enable_if<
        !(std::is_same<_Allocator, INPLACE_MMAP_ALLOC<_K, _V>>::value),
        void>::type
    resize();

    template<typename _K = K, typename _Hasher = Hasher>
    using _hash_type_t = typename std::result_of<_Hasher(K)>::type;
    typedef _hash_type_t<K, Hasher> hash_type_t;

    using key_pass_t = typename fht_chunk<K, V>::key_pass_t;
    using val_pass_t = typename fht_chunk<K, V>::val_pass_t;
    struct fht_iterator;

    fht_table(const uint64_t init_size);
    // defaults to FHT_DEFAULT_INIT_SIZE (really???)
    fht_table();
    ~fht_table();

    //////////////////////////////////////////////////////////////////////
    // add new key value pair stuff

    // slightly different logic than emplace...
    template<typename... Args>
    std::pair<fht_iterator, bool>
    insert_or_assign(key_pass_t new_key, Args &&... args) {
        const uint64_t res =
            (const uint64_t)add(new_key, std::forward<Args>(args)...);
        if (res & 0x1) {
            NEW(V,
                *((V * const)((res >> 1) << 1)),
                std::forward<Args>(args)...);
        }
        return std::pair<fht_iterator, bool>(
            ((const int8_t * const)((res >> 1) << 1)),
            !(res & 0x1));
    }

    template<typename... Args>
    std::pair<fht_iterator, bool>
    insert(key_pass_t new_key, Args &&... args) {
        return emplace(new_key, std::forward<Args>(args)...);
    }

    //////////////////////////////////////////////////////////////////////
    // Its probably a bad idea to use any other inserts below
    std::pair<fht_iterator, bool>
    insert(const std::pair<const K, V> & bad_pair) {
        return insert(bad_pair.first, bad_pair.second);
    }


    void
    insert(std::initializer_list<const std::pair<const K, V>> ilist) {

        // supposedly this is a few assembly ops faster than:
        // for(it = begin(); it != end(); ++it)
        // but you really should never be using this.
        for (auto p : ilist) {
            emplace(p.first, p.second);
        }
    }
    //////////////////////////////////////////////////////////////////////


    // at some point I will try and implement google's shit where they try and
    // find a key argument in pair arguments
    template<typename... Args>
    std::pair<fht_iterator, bool>
    emplace(Args &&... args) {
        return insert(std::forward<Args>(args)...);
    }

    // add new key value pair
    template<typename... Args>
    std::pair<fht_iterator, bool>
    emplace(key_pass_t new_key, Args &&... args) {
        // for now going to force explicit key & value
        const int8_t * const res = add(new_key, std::forward<Args>(args)...);
        return (((uint64_t)res) & 0x1)
                   ? (std::pair<fht_iterator, bool>(this->end(), false))
                   : (std::pair<fht_iterator, bool>(fht_iterator(res),
                                                    ++this->npairs));
    }

    template<typename... Args>
    const int8_t * const
    add(key_pass_t new_key, Args &&... args) {

        // get all derferncing of this out of the way
        const uint32_t          _log_incr = this->log_incr;
        const hash_type_t       raw_slot  = this->hash(new_key);
        fht_chunk<K, V> * const chunk     = (fht_chunk<K, V> * const)(
            (this->chunks) + (HASH_TO_IDX(raw_slot, _log_incr)));

        __builtin_prefetch(chunk);

        // get tag and start_idx from raw_slot
        const int8_t   tag       = GEN_TAG(raw_slot);
        const __m128i  tag_match = FHT_MM_SET(tag);
        const uint32_t start_idx = GEN_START_IDX(raw_slot);

        // prefetch is nice for performance here
        if (FHT_IS_SPECIAL_(FHT_SPECIAL_TYPES)) {
            __builtin_prefetch(
                chunk->get_key_n_ptr((FHT_MM_IDX_MULT * start_idx)));
        }

        // check for valid slot or duplicate
        uint32_t idx, slot_mask, del_idx = FHT_NODES_PER_CACHE_LINE;
        for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
            const uint32_t outer_idx = (j + start_idx) & FHT_MM_LINE_MASK;

            slot_mask = FHT_MM_MASK(tag_match, chunk->tags_vec[outer_idx]);
            while (slot_mask) {
                __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((slot_mask)));
                const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;

                if (__builtin_expect((chunk->compare_key_n(true_idx, new_key)),
                                     1)) {
                    return (const int8_t * const)(
                        (((uint64_t)chunk->get_val_n_ptr(true_idx)) | 0x1));
                }

                slot_mask ^= (1 << idx);
            }

            // we always go here 1st loop (where most add calls find a slot) and
            // if no deleted elements alwys go here as well
            if (__builtin_expect(del_idx & FHT_NODES_PER_CACHE_LINE, 1)) {
                const uint32_t _slot_mask = chunk->get_empty_or_del(outer_idx);
                if (__builtin_expect(_slot_mask, 1)) {
                    __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((_slot_mask)));
                    const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;

                    // some tunable param here would be useful
                    if (chunk->is_deleted_n(true_idx)) {
                        // even though this adds an operation it avoids worst
                        // case where alot of deleted items = unnecissary
                        // iterations. 2 iterations approx = this cost.
                        if (chunk->get_empty(outer_idx)) {

                            chunk->set_key_val_tag(true_idx,
                                                   tag,
                                                   new_key,
                                                   std::forward<Args>(args)...);
                            return ((const int8_t * const)chunk) + true_idx;
                        }

                        del_idx = true_idx;
                    }
                    else {
                        chunk->set_key_val_tag(true_idx,
                                               tag,
                                               new_key,
                                               std::forward<Args>(args)...);
                        return ((const int8_t * const)chunk) + true_idx;
                    }
                }
            }
            else if (chunk->get_empty(outer_idx)) {
                chunk->set_key_val_tag(del_idx,
                                       tag,
                                       new_key,
                                       std::forward<Args>(args)...);
                return ((const int8_t * const)chunk) + del_idx;
            }
        }

        if (del_idx != FHT_NODES_PER_CACHE_LINE) {
            chunk->set_key_val_tag(del_idx,
                                   tag,
                                   new_key,
                                   std::forward<Args>(args)...);
            return ((const int8_t * const)chunk) + del_idx;
        }

        // no valid slot found so resize
        this->resize();

        fht_chunk<K, V> * const new_chunk = (fht_chunk<K, V> * const)(
            this->chunks + HASH_TO_IDX(raw_slot, _log_incr + 1));


        // after resize add without duplication check
        for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
            const uint32_t outer_idx  = (j + start_idx) & FHT_MM_LINE_MASK;
            const uint32_t _slot_mask = new_chunk->get_empty(outer_idx);

            if (__builtin_expect(_slot_mask, 1)) {
                __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((_slot_mask)));
                const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;
                new_chunk->set_key_val_tag(true_idx,
                                           tag,
                                           new_key,
                                           std::forward<Args>(args)...);


                return ((const int8_t * const)new_chunk) + true_idx;
            }
        }
        // probability of this is 1 / (2 ^ 64)
        assert(0);
    }
    //////////////////////////////////////////////////////////////////////
    // stuff related to finding elements

    template<typename _K         = K,
             typename _V         = V,
             typename _Hasher    = Hasher,
             typename _Allocator = Allocator>
    typename std::enable_if<FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES),
                            const int8_t * const>::type
    _find(key_pass_t key) const;

    template<typename _K         = K,
             typename _V         = V,
             typename _Hasher    = Hasher,
             typename _Allocator = Allocator>
    typename std::enable_if<FHT_IS_SPECIAL(FHT_SPECIAL_TYPES),
                            const int8_t * const>::type
    _find(key_pass_t key) const;

    fht_iterator
    find(K && key) const {
        const int8_t * const res = _find(key);
        return (res == NULL) ? this->end() : fht_iterator(res);
    }

    fht_iterator
    find(const K & key) const {
        const int8_t * const res = _find(key);
        return (res == NULL) ? this->end() : fht_iterator(res);
    }

    uint64_t
    count(const K & key) const {
        return (_find(key) != NULL);
    }

    uint64_t
    count(K && key) const {
        return (_find(key) != NULL);
    }

    bool
    contains(const K & key) const {
        return count(key);
    }

    bool
    contains(K && key) const {
        return count(key);
    }

    V &
    at(const K & key) {
        const uint64_t                res   = (const uint64_t)_find(key);
        const fht_chunk<K, V> * const chunk = (const fht_chunk<K, V> * const)(
            res & (~(FHT_NODES_PER_CACHE_LINE - 1)));
        return *(chunk->get_val_ptr_n(res & (FHT_NODES_PER_CACHE_LINE - 1)));
    }

    V &
    at(K && key) {
        const uint64_t                res   = (const uint64_t)_find(key);
        const fht_chunk<K, V> * const chunk = (const fht_chunk<K, V> * const)(
            res & (~(FHT_NODES_PER_CACHE_LINE - 1)));
        return *(chunk->get_val_ptr_n(res & (FHT_NODES_PER_CACHE_LINE - 1)));
    }

    V & operator[](const K & key) {
        return at(key);
    }

    V & operator[](K && key) {
        return at(key);
    }


    //////////////////////////////////////////////////////////////////////
    // deleting stuff
    template<typename _K         = K,
             typename _V         = V,
             typename _Hasher    = Hasher,
             typename _Allocator = Allocator>
    typename std::enable_if<FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES), uint64_t>::type
    erase(key_pass_t key) const;


    template<typename _K         = K,
             typename _V         = V,
             typename _Hasher    = Hasher,
             typename _Allocator = Allocator>
    typename std::enable_if<FHT_IS_SPECIAL(FHT_SPECIAL_TYPES), uint64_t>::type
    erase(key_pass_t key) const;

    void
    clear() {
        const uint32_t num_chunks = 1 << this->log_incr;
        const __m256i  INV_SETTER = _mm256_set1_epi8(INVALID_MASK);

        for (uint32_t i = 0; i < num_chunks; i++) {
            ((__m256i * const)this->chunks[i])[0] = INV_SETTER;
            ((__m256i * const)this->chunks[i])[1] = INV_SETTER;
        }
    }


    // would be nice to implement in 4 bytes so iterator + bool fits in register
    struct fht_iterator {
        const int8_t * cur_tag;
        fht_iterator(const int8_t * const init_tag_pos) {
            this->cur_tag = init_tag_pos;
        }

        inline fht_iterator &
        operator=(const fht_iterator & other) {
            this->cur_tag = other.cur_tag;
        }

        fht_iterator &
        operator++() {
            do {
                if (__builtin_expect(((uint64_t)(this->cur_tag) %
                                      FHT_NODES_PER_CACHE_LINE) ==
                                         (FHT_NODES_PER_CACHE_LINE - 1),
                                     0)) {
                    this->cur_tag += sizeof(typename fht_chunk<K, V>::node_t);
                }
                this->cur_tag++;
            } while (RESIZE_SKIP(*(this->cur_tag)));
            return *this;
        }

        fht_iterator &
        operator++(int) {
            ++(*this);
            return *this;
        }

        inline fht_iterator &
        operator+=(uint32_t n) {
            while (n) {
                this ++;
            }
        }

        fht_iterator &
        operator--() {
            do {
                if (__builtin_expect(((uint64_t)(this->cur_tag) %
                                      FHT_NODES_PER_CACHE_LINE) == 0,
                                     0)) {
                    this->cur_tag -= sizeof(typename fht_chunk<K, V>::node_t);
                }
                this->cur_tag--;
            } while (RESIZE_SKIP(*cur_tag));
            return *this;
        }

        fht_iterator &
        operator--(int) {
            --(*this);
            return *this;
        }

        inline fht_iterator &
        operator-=(uint32_t n) {
            while (n) {
                this --;
            }
        }

        inline V & operator*() const {
            // i know this looks like a lot but really its just 7 operations, 1
            // out of cache line (probably)
            const fht_chunk<K, V> * const cur_chunk =
                (const fht_chunk<K, V> * const)(
                    ((uint64_t)(this->cur_tag)) &
                    (~(FHT_NODES_PER_CACHE_LINE - 1)));

            return *(
                cur_chunk->get_val_n_ptr(((uint64_t)(this->cur_tag)) &
                                         ((FHT_NODES_PER_CACHE_LINE - 1))));
        }

        inline V * operator->() const {
            const fht_chunk<K, V> * const cur_chunk =
                (const fht_chunk<K, V> * const)(
                    ((uint64_t)(this->cur_tag)) &
                    (~(FHT_NODES_PER_CACHE_LINE - 1)));

            return cur_chunk->get_val_n_ptr(((uint64_t)(this->cur_tag)) &
                                            ((FHT_NODES_PER_CACHE_LINE - 1)));
        }

        inline friend bool
        operator==(const fht_iterator & it_a, const fht_iterator & it_b) {
            return ((uint64_t)(it_a.cur_tag)) == ((uint64_t)(it_b.cur_tag));
        }

        inline friend bool
        operator!=(const fht_iterator & it_a, const fht_iterator & it_b) {
            return ((uint64_t)(it_a.cur_tag)) != ((uint64_t)(it_b.cur_tag));
        }

        inline friend bool
        operator<(const fht_iterator & it_a, const fht_iterator & it_b) {
            return ((uint64_t)(it_a.cur_tag)) < ((uint64_t)(it_b.cur_tag));
        }

        inline friend bool
        operator>(const fht_iterator & it_a, const fht_iterator & it_b) {
            return ((uint64_t)(it_a.cur_tag)) > ((uint64_t)(it_b.cur_tag));
        }

        inline friend bool
        operator<=(const fht_iterator & it_a, const fht_iterator & it_b) {
            return ((uint64_t)(it_a.cur_tag)) <= ((uint64_t)(it_b.cur_tag));
        }

        inline friend bool
        operator>=(const fht_iterator & it_a, const fht_iterator & it_b) {
            return ((uint64_t)(it_a.cur_tag)) >= ((uint64_t)(it_b.cur_tag));
        }
    };
    inline fht_iterator
    begin() const {
        return fht_iterator((const int8_t *)this->chunks);
    }

    inline fht_iterator
    end() const {
        return fht_iterator(((const int8_t *)this->chunks) +
                            (1UL << this->log_incr) + FHT_NODES_PER_CACHE_LINE);
    }
};


//////////////////////////////////////////////////////////////////////
// Actual Implemenation cuz templating kinda sucks imo


//////////////////////////////////////////////////////////////////////
// Constructor / Destructor
template<typename K, typename V, typename Hasher, typename Allocator>
fht_table<K, V, Hasher, Allocator>::fht_table(const uint64_t init_size) {

    // ensure init_size is above min
    const uint64_t _init_size =
        init_size > FHT_DEFAULT_INIT_SIZE
            ? (init_size ? roundup_next_p2(init_size) : FHT_DEFAULT_INIT_SIZE)
            : FHT_DEFAULT_INIT_SIZE;

    const uint32_t _log_init_size = log_b2(_init_size);
    //    int *  test = Allocator::new (NULL) int;

    // alloc chunks
    this->chunks =
        this->alloc_mmap.init_mem((_init_size / FHT_NODES_PER_CACHE_LINE));

    // might be faster with _m512 but this is a mile from any critical path and
    // makes less portable
    const __m256i INV_SETTER = _mm256_set1_epi8(INVALID_MASK);
    for (uint32_t i = 0; i < (_init_size / FHT_NODES_PER_CACHE_LINE); i++) {
        for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
            ((__m256i * const)(this->chunks + i))[0] = INV_SETTER;
            ((__m256i * const)(this->chunks + i))[1] = INV_SETTER;
        }
    }

    // set log
    this->log_incr = _log_init_size;
}

// call above with FHT_DEFAULT_INIT_SIZE
template<typename K, typename V, typename Hasher, typename Allocator>
fht_table<K, V, Hasher, Allocator>::fht_table()
    : fht_table(FHT_DEFAULT_INIT_SIZE) {}

// dealloc current chunk
template<typename K, typename V, typename Hasher, typename Allocator>
fht_table<K, V, Hasher, Allocator>::~fht_table() {
    this->alloc_mmap.deinit_mem(
        this->chunks,
        ((1 << (this->log_incr)) / FHT_NODES_PER_CACHE_LINE));
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

// Resize In Place
template<typename K, typename V, typename Hasher, typename Allocator>
template<typename _K, typename _V, typename _Hasher, typename _Allocator>
typename std::enable_if<
    std::is_same<_Allocator, INPLACE_MMAP_ALLOC<_K, _V>>::value,
    void>::type
fht_table<K, V, Hasher, Allocator>::resize() {

    // incr table log
    const uint32_t          _new_log_incr = ++(this->log_incr);
    fht_chunk<K, V> * const old_chunks    = this->chunks;


    const uint32_t _num_chunks =
        (1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE;

    // allocate new chunk array
    fht_chunk<K, V> * const new_chunks = this->alloc_mmap.init_mem(_num_chunks);

    uint32_t to_move = 0;
    uint32_t new_starts;
    uint32_t old_start_good_slots;
    // iterate through all chunks and re-place nodes
    for (uint32_t i = 0; i < _num_chunks; i++) {
        new_starts           = 0;
        old_start_good_slots = 0;

        uint32_t old_start_pos[FHT_MM_LINE]     = { 0 };
        uint64_t old_start_to_move[FHT_MM_LINE] = { 0 };


        fht_chunk<K, V> * const old_chunk = old_chunks + i;
        fht_chunk<K, V> * const new_chunk = new_chunks + i;


        // all intents and purposes not an important optimization but faster way
        // to reset deleted
        __m256i * const set_tags_vec = (__m256i * const)(old_chunk->tags_vec);

        // turn all deleted tags -> INVALID (reset basically)
        set_tags_vec[0] = _mm256_min_epu8(set_tags_vec[0], FHT_RESET_VEC);
        set_tags_vec[1] = _mm256_min_epu8(set_tags_vec[1], FHT_RESET_VEC);

        uint64_t j,
            iter_mask =
                ~((((uint64_t)_mm256_movemask_epi8(set_tags_vec[1])) << 32) |
                  (_mm256_movemask_epi8(set_tags_vec[0]) & 0xffffffff));

        while (iter_mask) {
            __asm__("tzcnt %1, %0" : "=r"((j)) : "rm"((iter_mask)));
            iter_mask ^= ((1UL) << j);


            // if node is invalid or deleted skip it. Can't just bvec iter here
            // because need to reset to invalid

            const hash_type_t raw_slot  = this->hash(old_chunk->get_key_n(j));
            const uint32_t    start_idx = GEN_START_IDX(raw_slot);

            if (GET_NTH_BIT(raw_slot, _new_log_incr - 1)) {
                const int8_t tag = old_chunk->get_tag_n(j);
                old_chunk->set_tag_n(j, INVALID_MASK);

                // place new node w.o duplicate check
                for (uint32_t new_j = 0; new_j < FHT_MM_LINE; new_j++) {
                    const uint32_t outer_idx =
                        (new_j + start_idx) & FHT_MM_LINE_MASK;
                    const uint32_t inner_idx =
                        (new_starts >> (8 * outer_idx)) & 0xff;

                    if (__builtin_expect(inner_idx != FHT_MM_IDX_MULT, 1)) {
                        const uint32_t true_idx =
                            FHT_MM_IDX_MULT * outer_idx + inner_idx;

                        ((int8_t * const)new_chunk->tags_vec)[true_idx] = tag;

                        NEW(K,
                            *(new_chunk->get_key_n_ptr(true_idx)),
                            std::move(*(old_chunk->get_key_n_ptr(j))));
                        NEW(V,
                            *(new_chunk->get_val_n_ptr(true_idx)),
                            std::move(*(old_chunk->get_val_n_ptr(j))));


                        new_starts += (1 << (8 * outer_idx));
                        break;
                    }
                }
            }
            else {
                // unplaceable slots
                old_start_pos[j / FHT_MM_IDX_MULT] |=
                    (1 << (j & FHT_MM_IDX_MASK));
                if ((j / FHT_MM_IDX_MULT) != start_idx) {
                    old_start_to_move[start_idx] |= ((1UL) << j);
                    to_move |= (1 << start_idx);
                }
                else {
                    old_start_good_slots += (1 << (8 * start_idx));
                }
            }
        }

        for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
            const uint32_t inner_idx = (new_starts >> (8 * j)) & 0xff;
            for (uint32_t _j = inner_idx; _j < FHT_MM_IDX_MULT; _j++) {
                new_chunk->set_tag_n(j * FHT_MM_IDX_MULT + _j, INVALID_MASK);
            }
        }

        uint64_t to_move_idx;
        uint32_t to_place_idx;

        while (to_move) {
            uint32_t j;
            __asm__("tzcnt %1, %0" : "=r"((j)) : "rm"((to_move)));

            // has space and has items to move to it
            while (old_start_pos[j] != 0xffff && old_start_to_move[j]) {

                __asm__("tzcnt %1, %0"
                        : "=r"((to_move_idx))
                        : "rm"((old_start_to_move[j])));

                __asm__("tzcnt %1, %0"
                        : "=r"((to_place_idx))
                        : "rm"((~old_start_pos[j])));

                old_start_to_move[j] ^= ((1UL) << to_move_idx);

                const uint32_t true_idx = FHT_MM_IDX_MULT * j + to_place_idx;

                old_chunk->set_tag_n(true_idx,
                                     old_chunk->get_tag_n(to_move_idx));


                NEW(K,
                    *(old_chunk->get_key_n_ptr(true_idx)),
                    std::move(*(old_chunk->get_key_n_ptr(to_move_idx))));
                NEW(V,
                    *(old_chunk->get_val_n_ptr(true_idx)),
                    std::move(*(old_chunk->get_val_n_ptr(to_move_idx))));


                old_chunk->set_tag_n(to_move_idx, INVALID_MASK);

                old_start_good_slots += (1 << (8 * j));
                old_start_pos[j] |= (1 << to_place_idx);
                old_start_pos[to_move_idx / FHT_MM_IDX_MULT] ^=
                    (1 << (to_move_idx & FHT_MM_IDX_MASK));
            }
            // j is full of items that belong but more entries wants to be in j
            if (__builtin_expect(
                    old_start_to_move[j] && ((old_start_good_slots >> (8 * j)) &
                                             0xff) == FHT_MM_IDX_MULT,
                    0)) {

                // move the indexes that this need to be placed (i.e if
                // any of the nodes that needed to be moved to j (now
                // j + 1) are already in j + 1 we can remove them from
                // to_move list
                old_start_to_move[(j + 1) & FHT_MM_LINE_MASK] |=
                    (~((0xffffUL)
                       << (FHT_MM_IDX_MULT * ((j + 1) & FHT_MM_LINE_MASK)))) &
                    old_start_to_move[j];


                const uint32_t new_mask =
                    (old_start_to_move[j] >>
                     (FHT_MM_IDX_MULT * ((j + 1) & FHT_MM_LINE_MASK))) &
                    0xffff;

                old_start_pos[(j + 1) & FHT_MM_LINE_MASK] |= new_mask;
                old_start_good_slots += bitcount_32(new_mask)
                                        << (8 * ((j + 1) & FHT_MM_LINE_MASK));

                // if j + 1 was done set it back
                if (old_start_to_move[(j + 1) & FHT_MM_LINE_MASK]) {
                    to_move |= (1 << ((j + 1) & FHT_MM_LINE_MASK));
                }
                old_start_to_move[j] = 0;
                to_move ^= (1 << j);
            }
            else if (__builtin_expect(!old_start_to_move[j], 1)) {
                to_move ^= (1 << j);
            }
        }
    }
}


// Resize Standard
template<typename K, typename V, typename Hasher, typename Allocator>
template<typename _K, typename _V, typename _Hasher, typename _Allocator>
typename std::enable_if<
    !(std::is_same<_Allocator, INPLACE_MMAP_ALLOC<_K, _V>>::value),
    void>::type
fht_table<K, V, Hasher, Allocator>::resize() {

    // incr table log
    const uint32_t                _new_log_incr = ++(this->log_incr);
    const fht_chunk<K, V> * const old_chunks    = this->chunks;


    const uint32_t _num_chunks =
        (1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE;

    // allocate new chunk array
    fht_chunk<K, V> * const new_chunks =
        this->alloc_mmap.init_mem(2 * _num_chunks);

    // set this while its definetly still in cache
    this->chunks = new_chunks;

    // iterate through all chunks and re-place nodes
    for (uint32_t i = 0; i < _num_chunks; i++) {
        uint64_t slot_idx = 0;

        const fht_chunk<K, V> * const old_chunk = old_chunks + i;

        // which one is optimal here really depends on the quality of the hash
        // function.
#ifdef BVEC_ITER
        uint64_t taken_slots, j;

        const uint32_t temp_taken_slots =
            (old_chunk->get_empty_or_del(1) << 16) |
            (old_chunk->get_empty_or_del(0) & 0xffff);

        taken_slots = (old_chunk->get_empty_or_del(3) << 16) |
                      (old_chunk->get_empty_or_del(2) & 0xffff);


        taken_slots = (taken_slots << 32) | temp_taken_slots;
        taken_slots = ~taken_slots;

        while (taken_slots) {
            __asm__("tzcnt %1, %0" : "=r"((j)) : "rm"((taken_slots)));
            taken_slots ^= ((1UL) << j);

#else
        for (uint32_t j = 0; j < FHT_NODES_PER_CACHE_LINE; j++) {

            // if node is invalid or deleted skip it
            if (__builtin_expect(RESIZE_SKIP(old_chunk->get_tag_n(j)), 0)) {
                continue;
            }
#endif

            const hash_type_t raw_slot  = this->hash(old_chunk->get_key_n(j));
            const uint32_t    start_idx = GEN_START_IDX(raw_slot);
            const uint32_t nth_bit = GET_NTH_BIT(raw_slot, _new_log_incr - 1);
            // 50 50 of hashing to same slot or slot + .5 * new table size
            fht_chunk<K, V> * const new_chunk =
                new_chunks + (i | (nth_bit ? _num_chunks : 0));

            // place new node w.o duplicate check
            for (uint32_t new_j = 0; new_j < FHT_MM_LINE; new_j++) {
                const uint32_t outer_idx =
                    (new_j + start_idx) & FHT_MM_LINE_MASK;
                const uint32_t inner_idx =
                    (slot_idx >> (8 * outer_idx + 32 * nth_bit)) & 0xff;

                if (__builtin_expect(inner_idx != FHT_MM_IDX_MULT, 1)) {
                    const uint32_t true_idx =
                        FHT_MM_IDX_MULT * outer_idx + inner_idx;

                    new_chunk->set_tag_n(true_idx, old_chunk->get_tag_n(j));
                    NEW(K,
                        *(new_chunk->get_key_n_ptr(true_idx)),
                        std::move(*(old_chunk->get_key_n_ptr(j))));
                    NEW(V,
                        *(new_chunk->get_val_n_ptr(true_idx)),
                        std::move(*(old_chunk->get_val_n_ptr(j))));


                    slot_idx += ((1UL) << (8 * outer_idx + 32 * nth_bit));
                    break;
                }
            }
        }
        // set remaining to INVALID_MASK
        for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
            const uint32_t inner_idx = (slot_idx >> (8 * j)) & 0xff;
            for (uint32_t _j = inner_idx; _j < FHT_MM_IDX_MULT; _j++) {
                new_chunks[i].set_tag_n(FHT_MM_IDX_MULT * j + _j, INVALID_MASK);
            }
        }
        for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
            const uint32_t inner_idx = (slot_idx >> (8 * j + 32)) & 0xff;
            for (uint32_t _j = inner_idx; _j < FHT_MM_IDX_MULT; _j++) {
                new_chunks[i | _num_chunks].set_tag_n(FHT_MM_IDX_MULT * j + _j,
                                                      INVALID_MASK);
            }
        }
    }

    // deallocate old table
    this->alloc_mmap.deinit_mem(
        (fht_chunk<K, V> * const)old_chunks,
        ((1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE));
}


//////////////////////////////////////////////////////////////////////
// Find
template<typename K, typename V, typename Hasher, typename Allocator>
template<typename _K, typename _V, typename _Hasher, typename _Allocator>
typename std::enable_if<FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES),
                        const int8_t * const>::type
fht_table<K, V, Hasher, Allocator>::_find(key_pass_t key) const {

    // same deal with add
    const uint32_t                _log_incr = this->log_incr;
    const hash_type_t             raw_slot  = this->hash(key);
    const fht_chunk<K, V> * const chunk     = (const fht_chunk<K, V> * const)(
        (this->chunks) + (HASH_TO_IDX(raw_slot, _log_incr)));
    __builtin_prefetch(chunk);

    // by setting valid here we can remove delete check
    const __m128i  tag_match = FHT_MM_SET(GEN_TAG(raw_slot));
    const uint32_t start_idx = GEN_START_IDX(raw_slot);

    // prefetch is good for perf
    __builtin_prefetch(chunk->get_key_n_ptr((FHT_MM_IDX_MULT * start_idx)));

    // check for valid slot of duplicate
    uint32_t idx, slot_mask;
    for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
        // seeded with start_idx we go through idx function
        const uint32_t outer_idx = (j + start_idx) & FHT_MM_LINE_MASK;


        slot_mask = FHT_MM_MASK(tag_match, chunk->tags_vec[outer_idx]);

        while (slot_mask) {
            __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((slot_mask)));
            const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;
            if (__builtin_expect((chunk->compare_key_n(true_idx, key)), 1)) {
                return ((const int8_t * const)chunk) + true_idx;
            }
            slot_mask ^= (1 << idx);
        }

        if (__builtin_expect(chunk->get_empty(outer_idx), 1)) {
            return NULL;
        }
    }
    return NULL;
}


//////////////////////////////////////////////////////////////////////
// Delete
template<typename K, typename V, typename Hasher, typename Allocator>
template<typename _K, typename _V, typename _Hasher, typename _Allocator>
typename std::enable_if<FHT_NOT_SPECIAL(FHT_SPECIAL_TYPES), uint64_t>::type
fht_table<K, V, Hasher, Allocator>::erase(key_pass_t key) const {

    // basically exact same as find but instead of storing the val just set
    // tag to deleted

    const uint32_t          _log_incr = this->log_incr;
    const hash_type_t       raw_slot  = this->hash(key);
    fht_chunk<K, V> * const chunk     = (fht_chunk<K, V> * const)(
        (this->chunks) + (HASH_TO_IDX(raw_slot, _log_incr)));
    __builtin_prefetch(chunk);

    const __m128i  tag_match = FHT_MM_SET(GEN_TAG(raw_slot));
    const uint32_t start_idx = GEN_START_IDX(raw_slot);


    __builtin_prefetch(chunk->get_key_n_ptr((FHT_MM_IDX_MULT * start_idx)));

    // check for valid slot of duplicate
    uint32_t idx, slot_mask;
    for (uint32_t j = 0; j < FHT_MM_LINE; j++) {

        const uint32_t outer_idx = (j + start_idx) & FHT_MM_LINE_MASK;

        slot_mask = FHT_MM_MASK(tag_match, chunk->tags_vec[outer_idx]);
        while (slot_mask) {
            __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((slot_mask)));
            const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;
            if ((chunk->compare_key_n(true_idx, key))) {
                chunk->delete_tag_n(true_idx);
                return FHT_ERASED;
            }
            slot_mask ^= (1 << idx);
        }

        if (__builtin_expect(chunk->get_empty(outer_idx), 1)) {
            return FHT_NOT_ERASED;
        }
    }

    return FHT_NOT_ERASED;
}

//////////////////////////////////////////////////////////////////////
// Optimized for larger sizes?

// find optimized for larger sizes?
template<typename K, typename V, typename Hasher, typename Allocator>
template<typename _K, typename _V, typename _Hasher, typename _Allocator>
typename std::enable_if<FHT_IS_SPECIAL(FHT_SPECIAL_TYPES),
                        const int8_t * const>::type
fht_table<K, V, Hasher, Allocator>::_find(key_pass_t key) const {

    // seperate version of find
    const uint32_t    _log_incr = this->log_incr;
    const hash_type_t raw_slot  = this->hash(key);

    // instead of doing everything through calls to just do directly. My
    // compiler at least does a bad job of optimizing out many of the
    // reference passes
    const __m128i * const tags = (const __m128i * const)(
        (this->chunks + HASH_TO_IDX(raw_slot, _log_incr)));

    // by setting valid here we can remove delete check
    const __m128i  tag_match = FHT_MM_SET(GEN_TAG(raw_slot));
    const uint32_t start_idx = GEN_START_IDX(raw_slot);

    // this is the key to seperate find. Basically instead of passing
    // reference to string to all of the chunk helper functions we can
    // gurantee inline and use direct values
    const fht_node<K, V> * const nodes =
        (const fht_node<K, V> * const)(tags + FHT_MM_LINE);

    __builtin_prefetch(tags + (start_idx));
    __builtin_prefetch(nodes + (FHT_MM_IDX_MULT * start_idx));

    // check for valid slot of duplicate
    uint32_t idx, slot_mask;
    for (uint32_t j = 0; j < FHT_MM_LINE; j++) {
        // seeded with start_idx we go through idx function
        const uint32_t outer_idx = (j + start_idx) & FHT_MM_LINE_MASK;
        slot_mask                = FHT_MM_MASK(tag_match, tags[outer_idx]);

        while (slot_mask) {
            __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((slot_mask)));
            const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;

            if ((nodes[true_idx].key == key)) {
                return ((const int8_t * const)tags) + true_idx;
            }
            slot_mask ^= (1 << idx);
        }

        if (__builtin_expect(FHT_MM_EMPTY(tags[outer_idx]), 1)) {
            return NULL;
        }
    }
    return NULL;
}

//////////////////////////////////////////////////////////////////////
// remove optimized for larger sizes?
template<typename K, typename V, typename Hasher, typename Allocator>
template<typename _K, typename _V, typename _Hasher, typename _Allocator>
typename std::enable_if<FHT_IS_SPECIAL(FHT_SPECIAL_TYPES), uint64_t>::type
fht_table<K, V, Hasher, Allocator>::erase(key_pass_t key) const {

    // same logic as the find function above
    const uint32_t    _log_incr = this->log_incr;
    const hash_type_t raw_slot  = this->hash(key);

    const __m128i * const tags = (const __m128i * const)(
        (this->chunks + HASH_TO_IDX(raw_slot, _log_incr)));

    int8_t * const tags8     = (int8_t * const)tags;
    const __m128i  tag_match = FHT_MM_SET(GEN_TAG(raw_slot));
    const uint32_t start_idx = GEN_START_IDX(raw_slot);

    const fht_node<K, V> * const nodes =
        (const fht_node<K, V> * const)(tags + FHT_MM_LINE);

    __builtin_prefetch(tags + (start_idx));
    __builtin_prefetch(nodes + (FHT_MM_IDX_MULT * start_idx));

    // check for valid slot of duplicate
    uint32_t idx, slot_mask;
    for (uint32_t j = 0; j < FHT_MM_LINE; j++) {

        const uint32_t outer_idx = (j + start_idx) & FHT_MM_LINE_MASK;

        slot_mask = FHT_MM_MASK(tag_match, tags[outer_idx]);
        while (slot_mask) {
            __asm__("tzcnt %1, %0" : "=r"((idx)) : "rm"((slot_mask)));
            const uint32_t true_idx = FHT_MM_IDX_MULT * outer_idx + idx;
            if ((nodes[true_idx].key == key)) {
                SET_DELETED(tags8[true_idx]);
                return FHT_ERASED;
            }
            slot_mask ^= (1 << idx);
        }

        if (__builtin_expect(FHT_MM_EMPTY(tags[outer_idx]), 1)) {
            return FHT_NOT_ERASED;
        }
    }

    return FHT_NOT_ERASED;
}

//////////////////////////////////////////////////////////////////////
// Default hash function
static const uint32_t
crc_32(const uint32_t * const data, const uint32_t len) {
    uint32_t       res = 0;
    const uint32_t l1  = len / sizeof(uint32_t);
    for (uint32_t i = 0; i < l1; i++) {
        res ^= __builtin_ia32_crc32si(FHT_HASH_SEED, data[i]);
    }

    if (len & 0x3) {
        uint32_t             final_k = 0;
        const int8_t * const data_8  = (const int8_t * const)(data + l1);
        for (uint32_t i = sizeof(uint32_t) * l1; i < len; i++) {
            final_k <<= 8;
            final_k |= data_8[i];
        }
        res ^= __builtin_ia32_crc32si(FHT_HASH_SEED, final_k);
    }

    return res;
}


template<typename K>
struct HASH_32 {

    constexpr const uint32_t
    operator()(K const & key) const {
        return crc_32((const uint32_t * const)(&key), sizeof(K));
    }
};


template<typename K>
struct HASH_32_4 {

    constexpr const uint32_t
    operator()(const K key) const {
        return __builtin_ia32_crc32si(FHT_HASH_SEED, key);
    }
};

template<typename K>
struct HASH_32_8 {

    constexpr const uint32_t
    operator()(const K key) const {
        return __builtin_ia32_crc32si(FHT_HASH_SEED, key) ^
               __builtin_ia32_crc32si(FHT_HASH_SEED, key >> 32);
    }
};

template<typename K>
struct HASH_32_CPP_STR {

    constexpr const uint32_t
    operator()(K const & key) const {
        return crc_32((const uint32_t * const)(key.c_str()), key.length());
    }
};


template<typename K>
struct DEFAULT_HASH_32 {


    template<typename _K = K>
    constexpr typename std::enable_if<(std::is_arithmetic<_K>::value &&
                                       sizeof(_K) <= 4),
                                      const uint32_t>::type
    operator()(const K key) const {
        return __builtin_ia32_crc32si(FHT_HASH_SEED, key);
    }

    template<typename _K = K>
    constexpr typename std::enable_if<(std::is_arithmetic<_K>::value &&
                                       sizeof(_K) == 8),
                                      const uint32_t>::type
    operator()(const K key) const {
        return __builtin_ia32_crc32si(FHT_HASH_SEED, key) ^
               __builtin_ia32_crc32si(FHT_HASH_SEED, key >> 32);
    }

    template<typename _K = K>
    constexpr typename std::enable_if<(std::is_same<_K, std::string>::value),
                                      const uint32_t>::type
    operator()(K const & key) const {
        return crc_32((const uint32_t * const)(key.c_str()), key.length());
    }

    template<typename _K = K>
    constexpr typename std::enable_if<(!std::is_same<_K, std::string>::value) &&
                                          (!std::is_arithmetic<_K>::value),
                                      const uint32_t>::type
    operator()(K const & key) const {
        return crc_32((const uint32_t * const)(&key), sizeof(K));
    }
};

//////////////////////////////////////////////////////////////////////
// 64 bit hashes

static const uint64_t
crc_64(const uint64_t * const data, const uint32_t len) {
    uint64_t       res = 0;
    const uint32_t l1  = len / sizeof(uint64_t);
    for (uint32_t i = 0; i < l1; i++) {
        res ^= _mm_crc32_u64(FHT_HASH_SEED, data[i]);
    }

    if (len & 0x7) {
        uint64_t             final_k = 0;
        const int8_t * const data_8  = (const int8_t * const)(data + l1);
        for (uint32_t i = sizeof(uint32_t) * l1; i < len; i++) {
            final_k <<= 8;
            final_k |= data_8[i];
        }
        res ^= _mm_crc32_u64(FHT_HASH_SEED, final_k);
    }

    return res;
}


template<typename K>
struct HASH_64 {

    constexpr const uint64_t
    operator()(K const & key) const {
        return crc_64((const uint64_t * const)(&key), sizeof(K));
    }
};


// really no reason to 64 bit hash a 32 bit value...
template<typename K>
struct HASH_64_4 {


    constexpr const uint32_t
    operator()(const K key) const {
        return __builtin_ia32_crc32si(FHT_HASH_SEED, key);
    }
};

template<typename K>
struct HASH_64_8 {

    constexpr const uint64_t
    operator()(const K key) const {
        return _mm_crc32_u64(FHT_HASH_SEED, key);
    }
};

template<typename K>
struct HASH_64_CPP_STR {

    constexpr const uint64_t
    operator()(K const & key) const {
        return crc_64((const uint64_t * const)(key.c_str()), key.length());
    }
};


template<typename K>
struct DEFAULT_HASH_64 {

    // we dont want 64 bit hash of 32 bit val....
    template<typename _K = K>
    constexpr typename std::enable_if<(std::is_arithmetic<_K>::value &&
                                       sizeof(_K) <= 4),
                                      const uint32_t>::type
    operator()(const K key) const {
        return __builtin_ia32_crc32si(FHT_HASH_SEED, key);
    }

    template<typename _K = K>
    constexpr typename std::enable_if<(std::is_arithmetic<_K>::value &&
                                       sizeof(_K) == 8),
                                      const uint64_t>::type
    operator()(const K key) const {
        return _mm_crc32_u64(FHT_HASH_SEED, key);
    }

    template<typename _K = K>
    constexpr typename std::enable_if<(std::is_same<_K, std::string>::value),
                                      const uint64_t>::type
    operator()(K const & key) const {
        return crc_64((const uint64_t * const)(key.c_str()), key.length());
    }

    template<typename _K = K>
    constexpr typename std::enable_if<(!std::is_same<_K, std::string>::value) &&
                                          (!std::is_arithmetic<_K>::value),
                                      const uint64_t>::type
    operator()(K const & key) const {
        return crc_64((const uint64_t * const)(&key), sizeof(K));
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


#define mymmap_alloc(Y, X)                                                     \
    myMmap((Y),                                                                \
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
struct SMALL_INPLACE_MMAP_ALLOC {
    SMALL_INPLACE_MMAP_ALLOC() {}

    ~SMALL_INPLACE_MMAP_ALLOC() {}

    constexpr fht_chunk<K, V> * const
    init_mem(const uint64_t size) const {
        assert(size <= sizeof(fht_chunk<K, V>) *
                           (FHT_DEFAULT_INIT_MEMORY / sizeof(fht_chunk<K, V>)));
        return (fht_chunk<K, V> *)myMmap(
            NULL,
            sizeof(fht_chunk<K, V>) *
                (FHT_DEFAULT_INIT_MEMORY / sizeof(fht_chunk<K, V>)),
            (PROT_READ | PROT_WRITE),
            (MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE),
            (-1),
            0,
            __FILE__,
            __LINE__);
    }

    constexpr void
    deinit_mem(fht_chunk<K, V> const * ptr, const size_t size) const {
        mymunmap((void *)ptr,
                 sizeof(fht_chunk<K, V>) *
                     (FHT_DEFAULT_INIT_MEMORY / sizeof(fht_chunk<K, V>)));
    }
};


// less syscalls this way
template<typename K, typename V>
struct INPLACE_MMAP_ALLOC {

    uint32_t          cur_size;
    uint32_t          start_offset;
    fht_chunk<K, V> * base_address;
    INPLACE_MMAP_ALLOC() {
        this->base_address = (fht_chunk<K, V> *)myMmap(
            NULL,
            sizeof(fht_chunk<K, V>) *
                (FHT_DEFAULT_INIT_MEMORY / sizeof(fht_chunk<K, V>)),
            (PROT_READ | PROT_WRITE),
            (MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE),
            (-1),
            0,
            __FILE__,
            __LINE__);

        this->cur_size     = FHT_DEFAULT_INIT_MEMORY / sizeof(fht_chunk<K, V>);
        this->start_offset = 0;
    }
    ~INPLACE_MMAP_ALLOC() {
        mymunmap(this->base_address, this->cur_size);
    }

    fht_chunk<K, V> * const
    init_mem(const size_t size) {
        const size_t old_start_offset = this->start_offset;
        this->start_offset += size;
        if (this->start_offset >= this->cur_size) {
            // maymove breaks inplace so no flags. This will very probably fail.
            // Can't really generically specify a unique addr. Assumption is
            // that FHT_DEFAULT_INIT_MEMORY will be sufficient
            if (MAP_FAILED ==
                mremap((void *)this->base_address,
                       sizeof(fht_chunk<K, V>) * this->cur_size,
                       2 * sizeof(fht_chunk<K, V>) * this->cur_size,
                       0)) {
                assert(0);
            }
            this->cur_size = 2 * this->cur_size;
        }
        return (fht_chunk<K, V> * const)(this->base_address + old_start_offset);
    }

    void
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        return;
    }
};


template<typename K, typename V>
struct DEFAULT_MMAP_ALLOC {

    fht_chunk<K, V> * const
    init_mem(const size_t size) const {
        return (fht_chunk<K, V> * const)mymmap_alloc(
            NULL,
            size * sizeof(fht_chunk<K, V>));
    }
    void
    deinit_mem(fht_chunk<K, V> * const ptr, const size_t size) const {
        mymunmap(ptr, size * sizeof(fht_chunk<K, V>));
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
// Undefs
#include "UNDEF_FHT_HELPER_MACROS.h"
#include "UNDEF_FHT_SPECIAL_TYPE_MACROS.h"
//////////////////////////////////////////////////////////////////////

#endif
