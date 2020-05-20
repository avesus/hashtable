#include <datastruct/seq_hashtable.h>

/* Imo the following are the best todos in no particular order:
 *
 * 1) try and find a way to find valid slot for node during resizing in O(1)
 *
 * 2) try and find a way to effectively use vectorized find method for matching
 * tags
 *
 * 3) I played around with storing next 16 bits of each hash (in uint16_t array
 * that would go along side tags) to avoid ever rehashing. I think the worse
 * overall cache performance / extra memory usage makes this not worth it
 * (rehashing is pretty cheap anyways) but is an idea. Really only needed if
 * uint8_t sized tags (which I get best performance with)
 *
 * 4) A way that either delete or valid flag could be dropped (every bit in tag
 * is a perf benefit). I was thinking valid could be dropped and tags that == 0
 * could be rehashed with new seed value. This would mean we get extra bit for
 * tag matching.
 *
 * 5) in general queries need to be optimized most imo
 */


#ifdef FHT_STATS
uint64_t niter_add    = 0;
uint64_t natt_add     = 0;
uint64_t niter_resize = 0;
uint64_t natt_resize  = 0;
uint64_t niter_find   = 0;
uint64_t natt_find    = 0;

uint64_t invalid_resize = 0;
uint64_t deleted_resize = 0;
uint64_t good_resize    = 0;
#define FHT_STATS_INCR(X) (X)++
#else
#define FHT_STATS_INCR(X)
#endif

//////////////////////////////////////////////////////////////////////
// These are basically variable for cache alignment

#define FHT_NODES_PER_CACHE_LINE (L1_CACHE_LINE_SIZE / sizeof(tag_type_t))
#define FHT_LOG_NODES_PER_CACHE_LINE                                           \
    (L1_LOG_CACHE_LINE_SIZE - FHT_LOG_TAG_TYPE_SIZE)
#define FHT_CACHE_IDX_MASK   (FHT_NODES_PER_CACHE_LINE - 1)
#define FHT_CACHE_ALIGN_MASK (~(FHT_CACHE_IDX_MASK))

// how many searches to try before giving up (I'm keeping as entire line but
// depending on ALOT of other things thats not always best)
#define FHT_SEARCH_NUMBER (FHT_NODES_PER_CACHE_LINE)
//////////////////////////////////////////////////////////////////////
#define TO_MASK(X) ((1 << (X)) - 1)
//////////////////////////////////////////////////////////////////////

// valid / invalid bits in the tag
#define VALID_FLAG  (0x1)
#define DELETE_FLAG (0x2)

#define IS_VALID(X)    ((X)&VALID_FLAG)
#define SET_VALID(X)   ((X) |= VALID_FLAG)
#define SET_UNVALID(X) ((X) ^= VALID_FLAG)

#define IS_DELETED(X)    ((X)&DELETE_FLAG)
#define SET_DELETED(X)   ((X) |= DELETE_FLAG)
#define SET_UNDELETED(X) ((X) ^= DELETE_FLAG)

// for calculating IDX from tag
#define IDX_MASK   (FHT_CACHE_IDX_MASK)
#define GET_IDX(X) (XOR_IDX((X) >> NEXT_HASH_OFFSET))

// if we have 1 byte tags this is enough
#if FHT_LOG_TAG_TYPE_SIZE == 0
#define XOR_IDX(X) ((X)&IDX_MASK)

#else

// if multi byte tags I find some xoring is nice
#define XOR_IDX(X)                                                             \
    (((X) >> (FHT_LOG_NODES_PER_CACHE_LINE) ^ ((X)) ^                          \
      ((X) >> (sizeof_bits(tag_type_t) - FHT_LOG_NODES_PER_CACHE_LINE))) &     \
     IDX_MASK)

#endif

// get the actual content (for matching)
#define CONTENT_MASK   (~(VALID_FLAG | DELETE_FLAG))
#define GET_CONTENT(X) ((X)&CONTENT_MASK)

// takes raw slot and log_init_size to get tag content
#define gen_tag(X, Y) ((((X) >> (Y)) << NEXT_HASH_OFFSET))

// delete flag + valid_flag
#define NEXT_HASH_OFFSET        (1 + 1)
#define GET_NEXT_HASH_BIT(X, Y) (((X) >> (NEXT_HASH_OFFSET + (Y))) & 0x1)

// tunable parameter determine how often need to rehash. REQ_UNIQUE... is
// basically how many bits in tag need be unknown assuming a table idx
#define REQ_UNIQUE_TAG_BITS (5)
#define REHASH_THRESH                                                          \
    (sizeof_bits(tag_type_t) - (REQ_UNIQUE_TAG_BITS + NEXT_HASH_OFFSET))
//////////////////////////////////////////////////////////////////////

// some bullshit so that you can technically change key type without rewriting
// actual code logic. Don't fuck with this as best todo is write in c++ once
// optimizations complete and make parameterized type

#define set_key_val(X, Y, Z)                                                   \
    fht_get_key((X)) = (Y);                                                    \
    fht_get_val((X)) = (Z)

#define EQUALS     1
#define NOT_EQUALS 0
static uint32_t
compare_keys(fht_key_t a, fht_key_t b) {
    return a == b;
}


//////////////////////////////////////////////////////////////////////
#define hseed           0  // literally doesn't matter
#define hash_fht_key(X) murmur_32((uint8_t *)(&(X)), fht_get_key_size(X), hseed)
//#define hash_fht_key(X) murmur_32_4(((X)), hseed)

// just murmur hash for 32 bit (imo better than 64)
static uint32_t
murmur_32(const uint8_t * key, size_t len, uint32_t seed) {
    uint32_t h = seed;
    if (len > 3) {
        const uint32_t * key_x4 = (const uint32_t *)key;
        size_t           i      = len >> 2;
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
        size_t   i = len & 3;
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

// murmur has 32 with 4 byte key in mind (0 difference between this and above)
static uint32_t
murmur_32_4(const uint32_t key, uint32_t seed) {
    uint32_t h = seed;
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

// 64 bit murmur hash. Kind of slow need to optimize it at some point
static uint64_t
murmur_64(const void * key, const uint32_t len, const uint32_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995;
    const uint32_t r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end  = data + (len / 8);

    while (data != end) {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    switch (len & 7) {
        case 7:
            h ^= ((*data) & ((0xFFUL) << 48));
        case 6:
            h ^= ((*data) & ((0xFFUL) << 40));
        case 5:
            h ^= ((*data) & ((0xFFUL) << 32));
        case 4:
            h ^= ((*data) & ((0xFFUL) << 24));
        case 3:
            h ^= ((*data) & ((0xFFUL) << 16));
        case 2:
            h ^= ((*data) & ((0xFFUL) << 8));
        case 1:
            h ^= ((*data) & ((0xFFUL)));
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

//////////////////////////////////////////////////////////////////////


#if 0

/* this implements a fast way to find a uint8_t in array. casts to uint64_t and
 * reduces results log time for speedup. Its faster but difficulties in actually
 * implementing well are there. Basically to implement it need to probably
 * implement as part of a loop that will cross out false positives. Also means
 * tags needs to be ordered which has its own performance costs. played around
 * with it a bit but ran into roadblock when a tag matches partway through the
 * vector. Then need to manually iterate through rest of sizeof(uint64_t). Got
 * it working (not totally optimially) but saw performance decrease (likely due
 * to fact that tags now need to be ordered) so decided to scrap it. If you have
 * an brilliant ideas go for it. Turn FHT_STATS on to see how many average
 * iterations before and ADD/FIND/RESIZE of a key takes to find good slot.*/

#define do_fast_find(X, Y) fast_find((uint64_t)(X), (const uint64_t * const)(Y))
#define NO_DELETE_MASK     0xfdfdfdfdfdfdfdfd
#define NO_FIND            (-1)
static int32_t
fast_find(uint64_t check_64, const uint64_t * const tags_64) {
    const uint64_t mask = 0x0101010101010101;
    check_64 |= check_64 << 32;
    check_64 |= check_64 << 16;
    check_64 |= check_64 << 8;
    for (int i = 0;
         i < (L1_CACHE_LINE_SIZE / (sizeof(uint64_t) / sizeof(tag_type_t)));
         i++) {
        const uint64_t tag_mask = (mask & (~tags_64[i])) - 1;
        uint64_t res = ~(check_64 ^ (tags_64[i] & tag_mask & NO_DELETE_MASK));
        res &= (res >> 4);
        res &= (res >> 2);
        res &= (res >> 1);
        res &= mask;
        if (res) {
            uint64_t ret;
            ff1_asm_tz(res, ret);
            return i * 8 + ret / 8;
        }
        else if (tag_mask != (~(0UL))) {
            uint64_t ret;
            ff0_asm_tz(tag_mask, ret);
            return i * 8 + ret / 8;
        }
    }
    return NO_FIND;
}
#endif


// initializes table. Required minimum size is kind of arbitrarily PAGE_SIZE
flat_hashtable_t *
fht_init_table(uint32_t init_size) {

    // alignment is important...
    assert((sizeof(flat_chunk_t) % L1_CACHE_LINE_SIZE) == 0);

    const uint64_t _init_size =
        init_size > DEFAULT_INIT_FHT_SIZE
            ? (init_size ? roundup_32(init_size) : DEFAULT_INIT_FHT_SIZE)
            : DEFAULT_INIT_FHT_SIZE;

    const uint32_t _log_init_size = ulog2_64(_init_size);

    // figure calloc call with header/footer will page in everything here (at
    // least with default initi size)
    uint64_t init_alloc = (uint64_t)mycalloc(
        1,
        sizeof(flat_hashtable_t) + L1_CACHE_LINE_SIZE +
            (_init_size / FHT_NODES_PER_CACHE_LINE) * sizeof(flat_chunk_t));

    flat_hashtable_t * new_table = (flat_hashtable_t *)init_alloc;
    init_alloc += sizeof(flat_hashtable_t);
    init_alloc +=
        (L1_CACHE_LINE_SIZE - (init_alloc & (L1_CACHE_LINE_SIZE - 1)));


    new_table->chunks        = (flat_chunk_t *)init_alloc;
    new_table->log_init_size = _log_init_size;
    new_table->log_incr      = 0;

    return new_table;
}

void
fht_deinit_table(flat_hashtable_t * table) {
    // absolutely a memory leak here. Not 100% sure how to fix.
    myfree(table);
}

// debugger
static void
print_chunk(const char * header, flat_chunk_t * chunk, uint32_t idx) {
    if (verbose) {
        fprintf(stderr, "[%s]: Chunk[%d] -> %p\n\t", header, idx, chunk);
        for (uint32_t i = 0; i < FHT_NODES_PER_CACHE_LINE; i++) {
            fprintf(stderr,
                    "%d: [%x] -> [%d][%d]\n\t",
                    i,
                    chunk->tags[i],
                    chunk->nodes[i].key,
                    chunk->nodes[i].val);
        }
        fprintf(stderr, "\r\n");
    }
}


// basically iterates thorugh all chunks in old table. nodes in each chunk can
// be remapped to 1 of 2 tables and does that. If this resize number will cause
// bits of stored hash to fall below belowthresh will rehash. I've found that
// maintaining higher threshold pays off moreso than avoiding hashing (though if
// either are totally ignored performance definetly drops)
static void
resize(flat_hashtable_t * table) {

    // constants...

    // reason we hold onto log_init_size is so that we can generate proper tag
    // for all incoming nodes
    const uint32_t _log_init_size = table->log_init_size;

    // incr are what actually give us table size
    const uint32_t _old_log_incr = table->log_incr;
    const uint32_t _new_log_incr = ++(table->log_incr);

    // num chunks to iterate through
    const uint32_t _num_chunks =
        (1 << (_log_init_size + _old_log_incr)) / FHT_NODES_PER_CACHE_LINE;

    // just some const variables. to make life easier. Recently decided if you
    // set things const properly compiler does a good job about what actually
    // need a register vs what should be recomputed on the fly vs stored in
    // memory (on the stack)
    const flat_chunk_t * const old_chunks    = table->chunks;
    flat_chunk_t * const       new_chunk_arr = (flat_chunk_t *)mymmap_alloc(
        ((1 << (_log_init_size + _new_log_incr)) / FHT_NODES_PER_CACHE_LINE) *
        sizeof(flat_chunk_t));

    // tbh this is a debug variable but really shouldnt affect perf and will
    // spot a nasty bug
    uint32_t nnode_counter = 0;

    for (uint32_t i = 0; i < _num_chunks; i++) {
        const fht_node_t * const nodes = old_chunks[i].nodes;
        const tag_type_t * const tags  = old_chunks[i].tags;

        // since we are adding 1 new bit 2 options for next chunk
        fht_node_t * const new_nodes[2] = {
            new_chunk_arr[i].nodes,
            new_chunk_arr[i + _num_chunks].nodes
        };

        tag_type_t * const new_tags[2] = {
            new_chunk_arr[i].tags,
            new_chunk_arr[i + _num_chunks].tags
        };

        // same as above counter
        uint32_t internal_nnode_counter = 0;

        // heres the meat. Basically iterate through all nodes in line, skip if
        // deleted or invalid
        for (uint32_t j = 0; j < FHT_NODES_PER_CACHE_LINE; j++) {
            if (!IS_VALID(tags[j]) || IS_DELETED(tags[j])) {

#ifdef FHT_STATS
                invalid_resize += (!IS_VALID(tags[j]));
                deleted_resize += (IS_DELETED(tags[j]));
#endif
                continue;
            }
            FHT_STATS_INCR(good_resize);

            nnode_counter++;
            internal_nnode_counter++;

            // here we are getting tag/start idx/hash for the node. If its a
            // rehash resize call will recompute hash value and recalculate
            // tag/next hash bits for everything. Else will use what is stored
            // in tag array
            uint32_t ltag;
            uint32_t start_idx;
            uint32_t next_hash_bit;
            if (_new_log_incr < REHASH_THRESH) {
                // use whats stored
                ltag          = tags[j];
                start_idx     = GET_IDX(ltag);
                next_hash_bit = GET_NEXT_HASH_BIT(ltag, _old_log_incr);
                assert(!!next_hash_bit == next_hash_bit);
            }
            else {
                // recalculate
                const uint32_t raw_slot = hash_fht_key(fht_get_key(nodes[j]));
                ltag = gen_tag(raw_slot, _log_init_size + _new_log_incr) |
                       VALID_FLAG;
                start_idx     = GET_IDX(ltag);
                next_hash_bit = GET_NEXT_HASH_BIT(tags[j], _old_log_incr);
            }

#ifdef DEBUG
            // this is something that breaks alot so keeping in debug macro
            assert(
                (i | (next_hash_bit << (_log_init_size + _old_log_incr -
                                        FHT_LOG_NODES_PER_CACHE_LINE))) ==
                ((hash_fht_key(nodes[j].key) >> FHT_LOG_NODES_PER_CACHE_LINE) &
                 (TO_MASK(_log_init_size + _new_log_incr) /
                  FHT_NODES_PER_CACHE_LINE)));
#endif

/* possible a faster way to select chunks */
#if 0
            tag_type_t * const new_tags =
                new_chunk_arr[i + (next_hash_bit ? _num_chunks : 0)].tags;
            tag_type_t * const new_nodes =
                new_chunk_arr[i + (next_hash_bit ? _num_chunks : 0)].nodes;
#endif

            FHT_STATS_INCR(natt_resize);

            // try and find available slot in new tables chunk. If this fails
            // (it technically can) abort(). Will try and think of a way to make
            // this robust. Probability this fails (assuming uniform hashing) is
            // 1 / (2 ^ FHT_NODES_PER_CACHE_LINE)
            for (uint32_t new_j = 0; new_j < FHT_SEARCH_NUMBER; new_j++) {
                FHT_STATS_INCR(niter_resize);
                const uint32_t test_idx =
                    (start_idx + new_j) & FHT_CACHE_IDX_MASK;

                // we dont need to check for matches here (all nodes are by
                // definition unique) so just find first invalid slot
                if (!IS_VALID(new_tags[next_hash_bit][test_idx])) {
                    new_tags[next_hash_bit][test_idx] = ltag;
                    set_key_val(new_nodes[next_hash_bit][test_idx],
                                fht_get_key(nodes[j]),
                                fht_get_val(nodes[j]));

                    internal_nnode_counter--;
                    nnode_counter--;
                    break;
                }
                assert(new_j != FHT_NODES_PER_CACHE_LINE);
            }
        }
        // debugging....
        if (internal_nnode_counter) {
            fprintf(stderr, "Error(%d) IDX(%d)\n", internal_nnode_counter, i);
            print_chunk("Old_Chunk", (flat_chunk_t *)(old_chunks + i), i);
            print_chunk("New_Chunk[0]", new_chunk_arr + i, i);
            print_chunk("New_Chunk[1]",
                        new_chunk_arr + i + _num_chunks,
                        i + _num_chunks);
        }
        assert(!internal_nnode_counter);
    }

    assert(!nnode_counter);
    table->chunks = new_chunk_arr;

    // update "log_init_size"  and log_incr if we rehashed. Basically you can
    // think of log_init_size is log of last size that we hashed and incr how
    // many resizes since then.
    if (_new_log_incr >= REHASH_THRESH) {
        table->log_init_size = _log_init_size + _new_log_incr;
        table->log_incr      = 0;
    }
}

// add new key/val pair (instead of this bullshit with typedefs if this table is
// good will make type T and c++
int32_t
fht_add_key(flat_hashtable_t * table, fht_key_t new_key, fht_val_t new_val) {
    const uint32_t _log_init_size = table->log_init_size;
    const uint32_t _log_incr      = table->log_incr;

    // this mask drops lower bits for getting proper chunk
    const uint32_t chunk_mask =
        (TO_MASK(_log_init_size + _log_incr) & FHT_CACHE_ALIGN_MASK);

    // computer hash and tag... # persay
    const uint32_t   raw_slot  = hash_fht_key(new_key);
    const tag_type_t tag       = gen_tag(raw_slot, _log_init_size);
    const uint32_t   start_idx = GET_IDX(tag);

    // get right chunk
    flat_chunk_t * const chunk =
        (table->chunks) + ((raw_slot & chunk_mask) / FHT_NODES_PER_CACHE_LINE);

    // get tags/nodes array
    tag_type_t * const tags  = chunk->tags;
    fht_node_t * const nodes = chunk->nodes;

    FHT_STATS_INCR(natt_add);

    // search through the array.
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(niter_add);

        // start idx is basically randomized. This helps avoid to many
        // iterations (i.e by starting randomly we average 2.2 iterations until
        // find a good slot (meaning VALID slot or iterm already exists). If
        // start_idx was always 0 average about 28... This does make some
        // optimizations harder/impossible (i.e it would be nice if we could
        // O(1) find a valid slot for resize but havent really put much effort
        // into optimizing that yet
        const uint32_t test_idx = (start_idx + j) & FHT_CACHE_IDX_MASK;

        // if block is invalid add there (valid basically means taken).
        if (!IS_VALID(tags[test_idx])) {
            tags[test_idx] = (tag | VALID_FLAG);
            set_key_val(nodes[test_idx], new_key, new_val);
            return FHT_ADDED;
        }

        // elseif it is valid and tags match compare actual keys
        else if ((GET_CONTENT(tags[test_idx]) == tag)) {
            if (compare_keys(fht_get_key(nodes[test_idx]), new_key) == EQUALS) {
                if (IS_DELETED(tags[test_idx])) {
                    SET_UNDELETED(tags[test_idx]);
                    return FHT_ADDED;
                }
                return FHT_NOT_ADDED;
            }
        }
    }

    // if we found no valid slots resize then add (without comparison checks
    // again)
    resize(table);

    flat_chunk_t * const new_chunk =
        (table->chunks) +
        ((raw_slot & TO_MASK(_log_init_size + _log_incr + 1)) /
         FHT_NODES_PER_CACHE_LINE);

    tag_type_t * const new_tags  = new_chunk->tags;
    fht_node_t * const new_nodes = new_chunk->nodes;

    const tag_type_t new_tag       = gen_tag(raw_slot, table->log_init_size);
    const uint32_t   new_start_idx = GET_IDX(new_tag);

    FHT_STATS_INCR(natt_add);
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(niter_add);
        const uint32_t test_idx = (new_start_idx + j) & FHT_CACHE_IDX_MASK;
        if (!IS_VALID(new_tags[test_idx])) {
            new_tags[test_idx] = (new_tag | VALID_FLAG);
            set_key_val(new_nodes[test_idx], new_key, new_val);
            return FHT_ADDED;
        }
    }
    assert(0);
}
// find and delete are basically the exact same as add w.o the actual adding
// step... Deletion just sets a flag.

// Reason delete need to set flag (as opposed to null out item) is that we
// assume first NULL means item not in table. If we NULL out deleted item could
// mean NULL block between hash location of item and where it was actually
// placed. If you want ot do true delete need to move items that missed exact
// hash location due to it being occupied.

int32_t
fht_find_key(flat_hashtable_t * table, fht_key_t key) {
    const uint32_t _log_init_size = table->log_init_size;
    const uint32_t _log_incr      = table->log_incr;

    // this mask drops lower bits for getting proper chunk
    const uint32_t chunk_mask =
        (TO_MASK(_log_init_size + _log_incr) & FHT_CACHE_ALIGN_MASK);

    const uint32_t   raw_slot  = hash_fht_key(key);
    const tag_type_t tag       = gen_tag(raw_slot, _log_init_size);
    const uint32_t   start_idx = GET_IDX(tag);

    flat_chunk_t * const chunk =
        (table->chunks) + ((raw_slot & chunk_mask) / FHT_NODES_PER_CACHE_LINE);

    tag_type_t * const tags  = chunk->tags;
    fht_node_t * const nodes = chunk->nodes;
    FHT_STATS_INCR(natt_find);
    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        FHT_STATS_INCR(niter_find);
        const uint32_t test_idx = (start_idx + j) & FHT_CACHE_IDX_MASK;


        if (!IS_VALID(tags[test_idx])) {
            return FHT_NOT_FOUND;
        }
        else if (IS_VALID(tags[test_idx]) &&
                 (GET_CONTENT(tags[test_idx]) == tag)) {
            if (compare_keys(fht_get_key(nodes[test_idx]), key) == EQUALS) {

                if (IS_DELETED(tags[test_idx])) {
                    return FHT_NOT_FOUND;
                }
                return FHT_FOUND;
            }
        }
    }
    return FHT_NOT_FOUND;
}


int32_t
fht_delete_key(flat_hashtable_t * table, fht_key_t key) {
    const uint32_t _log_init_size = table->log_init_size;
    const uint32_t _log_incr      = table->log_incr;

    // this mask drops lower bits for getting proper chunk
    const uint32_t chunk_mask =
        TO_MASK(_log_init_size + _log_incr) & FHT_CACHE_ALIGN_MASK;

    const uint32_t   raw_slot  = hash_fht_key(key);
    const tag_type_t tag       = gen_tag(raw_slot, _log_init_size);
    const uint32_t   start_idx = GET_IDX(tag);

    flat_chunk_t * const chunk =
        (table->chunks) + ((raw_slot & chunk_mask) / FHT_NODES_PER_CACHE_LINE);

    tag_type_t * const tags  = chunk->tags;
    fht_node_t * const nodes = chunk->nodes;

    for (uint32_t j = 0; j < FHT_SEARCH_NUMBER; j++) {
        const uint32_t test_idx = (start_idx + j) & FHT_CACHE_IDX_MASK;

        if (!IS_VALID(tags[test_idx])) {
            return FHT_NOT_DELETED;
        }
        else if (IS_VALID(tags[test_idx]) &&
                 (GET_CONTENT(tags[test_idx]) == tag)) {
            if (compare_keys(fht_get_key(nodes[test_idx]), key) == EQUALS) {
                if (IS_DELETED(tags[test_idx])) {
                    return FHT_NOT_DELETED;
                }
                SET_DELETED(tags[test_idx]);
                return FHT_DELETED;
            }
        }
    }
    return FHT_NOT_DELETED;
}
