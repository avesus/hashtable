#ifndef _HASHTABLE_H_
#define _HASHTABLE_H_

#include <helpers/bits.h>
#include <helpers/opt.h>
#include <helpers/util.h>

//////////////////////////////////////////////////////////////////////
// Configs
#include <general-config.h>

#define cache_line_size L1_CACHE_LINE_SIZE
#define log_cls         L1_LOG_CACHE_LINE_SIZE

#define ent_per_line cache_line_size / sizeof(node *)
// this is step length for iterating through
// int array where each index is on its own cache line
#define int_ca     (cache_line_size >> 2)
#define log_int_ca (log_cls - 2)

#define num_high_bits 16

//#define mark_null //table being copied will not add to null items (if they can
// mark it) #define next_hash //turns on storing next hash optimization #define
// lazy_resize //turns on lazy resizing (necessary for functioning delete)
#if defined next_hash && !defined lazy_resize
#undef next_hash
#endif
#if defined mark_null && !defined lazy_resize
#undef mark_null
#endif


#define DEFAULT_INIT_TSIZE         (sizeof_bits(cpu_set_t))
#define DEFAULT_SPREAD             (16)
#define DEFAULT_HASH_ATTEMPTS      (1)
#define DEFAULT_LINES              (1)
#define DEFAULT_READS_PER_LINE     (ent_per_line * DEFAULT_LINES)
#define DEFAULT_LOG_READS_PER_LINE (3)


#define RESIZE_THRESHOLD 2


//////////////////////////////////////////////////////////////////////
// actual node type
typedef struct frame_node {
    pthread_t key;
    void *    val;  // this is a frame_data_t ** defined in debug
} frame_node;

// define node and key type
typedef pthread_t         key_type;
typedef struct frame_node node;


//////////////////////////////////////////////////////////////////////
// table structs
// a sub table (this should be hidden)
typedef struct subTable {
    node ** innerTable;  // rows (table itself)

    // for keeping track of when all items from min sub table have been moved
#ifdef lazy_resize
    uint32_t * threadCopy;
#endif
    uint32_t tableSize;  // size
    uint32_t logTableSize;
} subTable;

// head of cache: this is the main hahstable
typedef struct hashTable {
    subTable **    tableArray;  // array of tables
    uint64_t       start;
    uint64_t       end;  // current max index (max exclusive)
    const uint32_t seeds[DEFAULT_HASH_ATTEMPTS];
} hashTable;  // worth it to aligned alloc this


//////////////////////////////////////////////////////////////////////
// general api start
// return 1 if inserted, 0 if already there

node * addNode(hashTable * head, node * n, uint32_t tid);

// see if node is in the table
node * findNode(hashTable * table, key_type key, uint32_t tid);


// initialize a new main hashtable
hashTable * initTable();


void freeTable(hashTable * head, void (*freeEnt)(void *));

//////////////////////////////////////////////////////////////////////
// general api end

#endif
