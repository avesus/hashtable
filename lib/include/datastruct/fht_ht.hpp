#ifndef _FHT_HT_H_
#define _FHT_HT_H_

#include <helpers/opt.h>
#include <helpers/util.h>


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
    union {
    V val;
        uint64_t wasted;
    };
};

template<typename K, typename V>
struct fht_chunk {
    tag_type_t     tags[L1_CACHE_LINE_SIZE];
    fht_node<K, V> nodes[L1_CACHE_LINE_SIZE];
};

static const uint32_t
murmur3_32(const uint8_t * key, const uint32_t len);
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
//Actual Implemenation cuz templating kinda sucks imo
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
    const uint32_t                _new_log_incr = ++(this->log_incr);
    const fht_chunk<K, V> * const old_chunks    = this->chunks;

    fht_chunk<K, V> * const new_chunks = (fht_chunk<K, V> *)mymmap_alloc(
        ((1 << (_new_log_incr)) / FHT_NODES_PER_CACHE_LINE) *
        sizeof(fht_chunk<K, V>));
    this->chunks = new_chunks;

    const uint32_t _num_chunks =
        (1 << (_new_log_incr - 1)) / FHT_NODES_PER_CACHE_LINE;

    for (uint32_t i = 0; i < _num_chunks; i++) {
        const fht_node<K, V> * const nodes = old_chunks[i].nodes;
        const tag_type_t * const     tags  = old_chunks[i].tags;

        fht_node<K, V> * const new_nodes[2] = {
            new_chunks[i].nodes,
            new_chunks[i + _num_chunks].nodes
        };

        tag_type_t * const new_tags[2] = { new_chunks[i].tags,
                                           new_chunks[i + _num_chunks].tags };

        uint32_t in = 0;
        for (uint32_t j = 0; j < FHT_NODES_PER_CACHE_LINE; j++) {
            if (RESIZE_SKIP(tags[j])) {
                continue;
            }

            in++;
            const tag_type_t tag = tags[j];

            // annoying that we need to access object. C++ needs a better way to
            // do this.
            const uint32_t raw_slot  = this->hash_32(nodes[j].key);
            const uint32_t start_idx = GEN_START_IDX(raw_slot);
            const uint32_t next_bit  = GET_NTH_BIT(raw_slot, _new_log_incr - 1);

            for (uint32_t new_j = 0; new_j < FHT_SEARCH_NUMBER; new_j++) {
                const uint32_t test_idx = GEN_TEST_IDX(start_idx, new_j);
                if (!IS_VALID(new_tags[next_bit][test_idx])) {
                    new_tags[next_bit][test_idx] = tag;
                    SET_KEY_VAL(new_nodes[next_bit][test_idx],
                                nodes[j].key,
                                nodes[j].val);
                    in--;
                    break;
                }
            }
        }
        assert(!in);
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

    // check for valid slot of duplicate
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);

        if (!IS_VALID(tags[test_idx])) {
            tags[test_idx] = (tag | VALID_MASK);
            SET_KEY_VAL(nodes[test_idx], new_key, new_val);
            return FHT_ADDED;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, new_key)) {
                if (IS_DELETED(tags[test_idx])) {
                    SET_UNDELETED(tags[test_idx]);
                    return FHT_ADDED;
                }
                return FHT_NOT_ADDED;
            }
        }
    }

    // no valid slot found so resize
    tag_type_t * const new_tags = (tag_type_t * const)(
        this->resize() +
        ((raw_slot & TO_MASK(_log_incr + 1)) / FHT_NODES_PER_CACHE_LINE));

    fht_node<K, V> * const new_nodes =
        (fht_node<K, V> * const)(new_tags + FHT_NODES_PER_CACHE_LINE);

    // after resize add without duplication check
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (!IS_VALID(new_tags[test_idx])) {
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
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (!IS_VALID(tags[test_idx])) {
            return FHT_NOT_FOUND;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, key)) {
                if (IS_DELETED(tags[test_idx])) {
                    return FHT_NOT_FOUND;
                }
                return FHT_FOUND;
            }
        }
    }
    return FHT_NOT_FOUND;
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Delete
template<typename K, typename V, typename Hasher>
uint32_t
fht_table<K, V, Hasher>::remove(const K key) {
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
        // seeded with start_idx we go through idx function
        const uint32_t test_idx = GEN_TEST_IDX(start_idx, j);
        if (!IS_VALID(tags[test_idx])) {
            return FHT_NOT_DELETED;
        }
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (COMPARE_KEYS(nodes[test_idx].key, key)) {
                if (IS_DELETED(tags[test_idx])) {
                    return FHT_NOT_DELETED;
                }
                SET_DELETED(tags[test_idx]);
                return FHT_DELETED;
            }
        }
    }
    return FHT_NOT_FOUND;
}


template class fht_table<uint32_t, uint32_t>;

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
