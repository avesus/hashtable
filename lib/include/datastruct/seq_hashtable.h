#ifndef _SEQ_HASHTABLE_H_
#define _SEQ_HASHTABLE_H_
#include <helpers/bits.h>
#include <helpers/opt.h>
#include <helpers/util.h>

// turn on for some trivial stats collection
//#define FHT_STATS
#define FHT_ALWAYS_REHASH
//#define FHT_HASH_ATTEMPTS 2

#if defined FHT_HASH_ATTEMPTS && !defined FHT_ALWAYS_REHASH
static_assert(0, "Bad defines\n");
#endif


//////////////////////////////////////////////////////////////////////
//....
#define DEFAULT_INIT_FHT_SIZE PAGE_SIZE
//////////////////////////////////////////////////////////////////////

// return values
#define FHT_NOT_ADDED (0)
#define FHT_ADDED     (1)

#define FHT_NOT_FOUND (0)
#define FHT_FOUND     (1)

#define FHT_NOT_DELETED (0)
#define FHT_DELETED     (1)
//////////////////////////////////////////////////////////////////////

// a node for storing key/val
typedef struct fht_node {
    uint32_t key;
    uint32_t val;
} fht_node_t;

typedef uint32_t fht_key_t;
typedef uint32_t fht_val_t;

// bullshit I write to tell myself this could work for any type and is thus
// generic and good
#define fht_get_key_size(X) sizeof(fht_key_t)
#define fht_get_val_size(X) sizeof(fht_val_t)

#define fht_get_key(X) (X).key
#define fht_get_val(X) (X).val


// actually tunable. Make sure you update log
typedef uint8_t tag_type_t;
// a compile time way to compute this would be nice but since I refused to past
// c++11 this seems only way
#define FHT_LOG_TAG_TYPE_SIZE 0


// a chunk the padding union portion is 100% unnecissary but i think it makes
// things cleared
typedef struct flat_chunk {
    union {
        uint8_t    tag_padding[L1_CACHE_LINE_SIZE];
        tag_type_t tags[L1_CACHE_LINE_SIZE / sizeof(tag_type_t)];
    };
    union {
        uint8_t    node_padding[(L1_CACHE_LINE_SIZE / sizeof(tag_type_t)) *
                             sizeof(fht_node_t)];
        fht_node_t nodes[(L1_CACHE_LINE_SIZE / sizeof(tag_type_t))];
    };
} flat_chunk_t;

// hashtable...
typedef struct flat_hashtable {
    flat_chunk_t * chunks;
#ifndef FHT_ALWAYS_REHASH
    uint32_t       log_init_size;
#endif
    uint32_t       log_incr;
} flat_hashtable_t;


// API
flat_hashtable_t * fht_init_table(uint32_t init_size);
void               fht_deinit_table(flat_hashtable_t * table);


int32_t fht_add_key(flat_hashtable_t * table,
                    fht_key_t          new_key,
                    fht_val_t          new_val);
int32_t fht_find_key(flat_hashtable_t * table, fht_key_t key);
int32_t fht_delete_key(flat_hashtable_t * table, fht_key_t key);


#endif
