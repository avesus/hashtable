#ifndef _FHT_HELPER_MACROS_H_
#define _FHT_HELPER_MACROS_H_

//////////////////////////////////////////////////////////////////////
// Constants
#define FHT_NODES_PER_CACHE_LINE     L1_CACHE_LINE_SIZE
#define FHT_LOG_NODES_PER_CACHE_LINE L1_LOG_CACHE_LINE_SIZE
#define FHT_CACHE_IDX_MASK           (FHT_NODES_PER_CACHE_LINE - 1)
#define FHT_CACHE_ALIGN_MASK         (~(FHT_CACHE_IDX_MASK))
//////////////////////////////////////////////////////////////////////
// Helpers
// mask of n bits
#define TO_MASK(n) ((1 << (n)) - 1)

// for extracting a bit
#define GET_NTH_BIT(X, n) ((((X) >> (TAG_BITS - L1_LOG_CACHE_LINE_SIZE)) >> ((n))) & 0x1)
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Tag Fields
#define VALID_MASK   (0x1)
#define DELETE_MASK  ((~VALID_MASK) | 0x2)
#define CONTENT_MASK (~(VALID_MASK))

#define IS_VALID(tag)    ((tag)&VALID_MASK)
#define SET_VALID(tag)   ((tag) |= VALID_MASK)
#define SET_UNVALID(tag) ((tag) ^= VALID_MASK)

#define IS_DELETED(tag)  ((tag) == DELETE_MASK)
#define SET_DELETED(tag) ((tag) = DELETE_MASK)

#define IS_PLACEABLE(tag) (!(tag))

// skip in resize if either not valid or deleted
#define RESIZE_SKIP(tag) (!IS_VALID(tag))

#define TAG_BITS 7
//////////////////////////////////////////////////////////////////////
// Calculating Tag and Start Idx and chunk from a raw hash
#define HASH_TO_IDX(hash_val, tbl_log)                                         \
    ((((hash_val) >> 1) & TO_MASK(tbl_log)) / FHT_NODES_PER_CACHE_LINE)
#define GEN_TAG(hash_val)       (((hash_val) << (TAG_BITS - L1_LOG_CACHE_LINE_SIZE)) | VALID_MASK)
#define GEN_START_IDX(hash_val) ((hash_val) >> (8 * sizeof(hash_type_t) - 2))


#endif
