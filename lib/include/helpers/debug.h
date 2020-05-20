#ifndef _DEBUG_H_
#define _DEBUG_H_

// must be power of 2. For truly race free code N_FRAMES >= max number
// of parallel frames that can be added to a given ID. For example 1000
// threads, using ID as ID N_FRAMES = 1 works, 1000 threads all going
// to ID = 0, N_FRAMES must be >= 1024
#define N_FRAMES 8

// for clean printing. Variables with name > PRINT_ALIGN chars will be an issue
#define PRINT_ALIGN 24

// low bit 0x1 means >= N_FRAMES have been set (this is necessary for
// indexing
#define WRAPPED 0x1

// active means another thread is currently operating on frames
// struct. This will throw and error.
#define ACTIVE 0x2

#include <helpers/bits.h>
#include <helpers/util.h>
#include <local/hashtable.h>
#include <local/macro-helper.h>

void init_debugger();
void free_debugger();

typedef struct frame_data {
    uint32_t nargs;
    uint32_t format_len;
    uint32_t var_name_len;
    uint32_t var_size_len;
    uint32_t size_align;
    uint32_t data_align;
    uint32_t line_num;
    uint64_t ID;
    char     data[];
} frame_data_t;

#define META_DATA_SIZE offsetof(frame_data_t, data)


#define DBG_ALIGNMENT      sizeof(uint64_t)     // for now this should be enough
#define DBG_ALIGNMENT_MASK (DBG_ALIGNMENT - 1)  // Mask of above


// helper to determine if it is var or region argument. A by-product of
// this method (couldn't find any others) is that a variable CANNOT
// start with parent, i.e passing (a + 1) would break things, passing a
//+ (1 + 2) would be fine.
#define M_CHECK(...)        M_CHECK_(__VA_ARGS__)
#define M_CHECK_(a, b, ...) b
#define M_IS_PAREN(x)       M_CHECK(M_IS_PAREN_ x, VAR)
#define M_IS_PAREN_(...)    REG, REG
#define M_CONCAT(a, b)      M_CONCAT_(a, b)
#define M_CONCAT_(a, b)     a##b
#define DEF(x)              M_IS_PAREN(x)

// if var we need to add argument for its size, if region size should
// have already be specified
#define DEF_VAR(X) X, sizeof(X)
#define DEF_REG(X) X

#define COND_DEF(X)       M_CONCAT(DEF_, DEF(X))(X)
#define COND_WRAPPER(...) APPLY(COND_DEF, __VA_ARGS__)


// sum up all data sizes
#define TOTAL_GET_SIZE(...) SUM_X_N_Y(GET_SIZE, __VA_ARGS__)

// get size of all data
#define GET_SIZE(X, Y) ROUNDUP_AL(Y)

// helper to get size for all and sum
#define TOTAL_TYPE_LEN(...) SUM(TO_SIZEOF, __VA_ARGS__)

// get sizeof strings (this is used on format/var names)
#define TO_SIZEOF(X) sizeof(X)


// sum size of variable names/format specifiers passed
#define TO_STRLEN(X)       sizeof(X)
#define TOTAL_STR_LEN(...) SUM(TO_STRLEN, __VA_ARGS__)

// to get names of all variables passed as strings
#define TO_NAME(X, Y)  #X
#define VAR_NAMES(...) APPLY_X_N_Y(TO_NAME, __VA_ARGS__)

// creates buffer to store all frame data
#define CREATE_BUFFER(ln, X) mycalloc((X + META_DATA_SIZE), sizeof(char));

// rounds up a size.
#define ROUNDUP_AL(X) (((X) + (DBG_ALIGNMENT_MASK)) & (~(DBG_ALIGNMENT_MASK)))

// for aligning so that data types are on DBG_ALIGNMENT byte boundary (can be
// important in some cases)
#define ALIGN(X) (DBG_ALIGNMENT - ((X) & (DBG_ALIGNMENT_MASK)));

// for getting block size from
#define MALLOC_SIZE_MASK (~(15UL))
#define BLOCK_SIZE(X)                                                          \
    (((*(((uint64_t *)X) - 1)) & MALLOC_SIZE_MASK) - (2 * sizeof(uint64_t)))

// copy strings helper
#define VAR_COPY_STR(X, ...) APPLY_X(TO_COPY_STR, X, __VA_ARGS__)

// copy string data. Assumed all strings are constant so can optimize
// with sizeof() instead of strlen. This is only used on format
// string/variable names
#define TO_COPY_STR(X, Y)                                                      \
    memcpy((X), (Y), (sizeof(Y)));                                             \
    ((X) += sizeof((Y)))


// helpers for copying either region or var. If var just cast and set,
// else memcpy (we want adding a frame to be low weight)
#define C_CHECK(...)              C_CHECK_(__VA_ARGS__)
#define C_CHECK_(a, b, c, d, ...) d
#define C_COND_DEF(X, Y, Z)       M_CONCAT(DEF_, C_DEF(Y, X, Z))(X, Y, Z)
#define C_DEF(x, y, z)            C_IS_PAREN(x, y, z)
#define C_IS_PAREN(x, y, z)       C_CHECK(C_IS_PAREN_ x, y, z, TO_COPY_VAL)
#define C_IS_PAREN_(...)          TO_COPY_REG, TO_COPY_REG, TO_COPY_REG, TO_COPY_REG
#define COPY_VARS(X, ...)         APPLY_X_Y_Z(C_COND_DEF, X, __VA_ARGS__)

// copy region
#define DEF_TO_COPY_REG(X, Y, Z)                                               \
    memcpy((X), (Y), (Z));                                                     \
    (X) += ROUNDUP_AL(Z)


// gcc has typeof, c++ equivilent is decltype
#ifdef CPP_MODE

// copy variable
#define DEF_TO_COPY_VAL(X, Y, Z)                                               \
    *(decltype(Y) *)(X) = (Y);                                                 \
    (X) += ROUNDUP_AL(Z)

#elif defined C_MODE

// copy variable
#define DEF_TO_COPY_VAL(X, Y, Z)                                               \
    *(typeof(Y) *)(X) = (Y);                                                   \
    (X) += ROUNDUP_AL(Z)


#endif


// copy sizeof data helper
#define COPY_SIZES(X, ...) APPLY_X_Y_Z(TO_COPY_SIZE, X, __VA_ARGS__)
// copy sizeof type (could probably make this shorts, regions can
// potentially be big though
#define TO_COPY_SIZE(X, Y, Z)                                                  \
    *((uint32_t *)(X)) = (Z);                                                  \
    (X) += sizeof(uint32_t)


// stores the frame symbols. In macro because alot of the information
// becomes lost/harder to find once passed to function. An idea is to
// reused buffers already allocated for the symbol frame to be over
// written. The pro is it will be faster the con is it requires some
// locking for parallelism. Best thing to do would be rewrite malloc.
#define NARGS       COMBINE(restricted_do_not_use_nargs, ln)
#define FMT_LEN     COMBINE(restricted_do_not_use_fmt_len, ln)
#define NAME_LEN    COMBINE(restricted_do_not_use_name_len, ln)
#define VAR_LEN     COMBINE(restricted_do_not_use_var_len, ln)
#define FILE_LEN    COMBINE(restricted_do_not_use_file_len, ln)
#define FUN_LEN     COMBINE(restricted_do_not_use_fun_len, ln)
#define SIZE_LEN    COMBINE(restricted_do_not_use_size_len, ln)
#define SIZE_ALIGN  COMBINE(restricted_do_not_use_size_align_len, ln)
#define DATA_ALIGN  COMBINE(restricted_do_not_use_data_align_len, ln)
#define TOTAL_SIZE  COMBINE(restricted_do_not_use_total_size, ln)
#define STRUCT_PTR  COMBINE(restricted_do_not_use_struct_ptr, ln)
#define BUF_PTR     COMBINE(restricted_do_not_use_buf_ptr, ln)
#define ID_NODE     COMBINE(restricted_do_not_use_id_node, ln)
#define ID_STRUCT   COMBINE(restricted_do_not_use_id_struct, ln)
#define FRAME_INDEX COMBINE(restricted_do_not_use_frame_index, ln)


// stores the frame symbols. In macro because alot of the information
// becomes lost/harder to find once passed to function
#define FRAME_AT_POINT(f, fn, ln, IDN, F, V)                                   \
    {                                                                          \
        do {                                                                   \
            assert(2 * PP_NARG(F) == PP_NARG(V));                              \
            const uint32_t NARGS      = PP_NARG(F);                            \
            const uint32_t FMT_LEN    = TOTAL_STR_LEN(F);                      \
            const uint32_t NAME_LEN   = TOTAL_STR_LEN(VAR_NAMES(V));           \
            const uint32_t VAR_LEN    = TOTAL_GET_SIZE(V);                     \
            const uint32_t FILE_LEN   = sizeof(f);                             \
            const uint32_t FUN_LEN    = sizeof(fn);                            \
            const uint32_t SIZE_LEN   = (NARGS * sizeof(uint32_t));            \
            const uint32_t SIZE_ALIGN = ALIGN(META_DATA_SIZE + FILE_LEN +      \
                                              FUN_LEN + FMT_LEN + NAME_LEN);   \
            const uint32_t DATA_ALIGN =                                        \
                ALIGN(META_DATA_SIZE + FILE_LEN + FUN_LEN + FMT_LEN +          \
                      NAME_LEN + SIZE_ALIGN + SIZE_LEN);                       \
            const uint32_t TOTAL_SIZE = FILE_LEN + FUN_LEN + FMT_LEN +         \
                                        VAR_LEN + NAME_LEN + META_DATA_SIZE +  \
                                        SIZE_ALIGN + DATA_ALIGN + SIZE_LEN;    \
                                                                               \
            DBG_ASSERT((TOTAL_SIZE % DBG_ALIGNMENT) == 0,                      \
                       "Unaligned Size (%d)\n",                                \
                       TOTAL_SIZE);                                            \
                                                                               \
            uint8_t * BUF_PTR = NULL;                                          \
            node *    ID_NODE = (node *)findFrame(IDN);                        \
            if (!ID_NODE) {                                                    \
                ID_NODE = (node *)newID_get(IDN);                              \
            }                                                                  \
            uint32_t FRAME_INDEX = high_bits_get(ID_NODE->val);                \
                                                                               \
            if (FRAME_INDEX == N_FRAMES) {                                     \
                low_bits_set_OR(ID_NODE->val, WRAPPED);                        \
            }                                                                  \
            FRAME_INDEX &= (N_FRAMES - 1);                                     \
                                                                               \
            high_bits_set_INCR(ID_NODE->val);                                  \
            frame_data_t ** ID_STRUCT =                                        \
                (frame_data_t **)get_ptr(ID_NODE->val);                        \
                                                                               \
            DBG_ASSERT(low_bits_set_XOR_atomic(ID_NODE->val, ACTIVE) & ACTIVE, \
                       "Duplicate ID(%lu)\n",                                  \
                       IDN);                                                   \
                                                                               \
            if (ID_STRUCT[FRAME_INDEX] && 0 &&                                 \
                (BLOCK_SIZE(ID_STRUCT[FRAME_INDEX]) >= TOTAL_SIZE)) {          \
                BUF_PTR = (uint8_t *)ID_STRUCT[FRAME_INDEX];                   \
            }                                                                  \
            else {                                                             \
                BUF_PTR = (uint8_t *)CREATE_BUFFER(ln, TOTAL_SIZE);            \
                myfree(ID_STRUCT[FRAME_INDEX]);                                \
                ID_STRUCT[FRAME_INDEX] = (frame_data_t *)BUF_PTR;              \
            }                                                                  \
                                                                               \
            frame_data_t * STRUCT_PTR = (frame_data_t *)BUF_PTR;               \
            BUF_PTR += META_DATA_SIZE;                                         \
            memcpy(BUF_PTR, f, FILE_LEN);                                      \
            BUF_PTR += FILE_LEN;                                               \
            memcpy(BUF_PTR, fn, FUN_LEN);                                      \
            BUF_PTR += FUN_LEN;                                                \
            VAR_COPY_STR(BUF_PTR, F);                                          \
            VAR_COPY_STR(BUF_PTR, VAR_NAMES(V));                               \
            BUF_PTR += SIZE_ALIGN;                                             \
            COPY_SIZES(BUF_PTR, V);                                            \
            BUF_PTR += DATA_ALIGN;                                             \
            COPY_VARS(BUF_PTR, V);                                             \
            STRUCT_PTR->nargs        = (uint32_t)NARGS;                        \
            STRUCT_PTR->format_len   = (uint32_t)FMT_LEN;                      \
            STRUCT_PTR->var_name_len = (uint32_t)NAME_LEN;                     \
            STRUCT_PTR->var_size_len = (uint32_t)VAR_LEN;                      \
            STRUCT_PTR->line_num     = (uint32_t)ln;                           \
            STRUCT_PTR->ID           = (uint32_t)IDN;                          \
            STRUCT_PTR->data_align   = (uint32_t)DATA_ALIGN;                   \
            STRUCT_PTR->size_align   = (uint32_t)SIZE_ALIGN;                   \
            assert(TOTAL_SIZE == (uint64_t)BUF_PTR - (uint64_t)STRUCT_PTR);    \
                                                                               \
            DBG_ASSERT(                                                        \
                !(low_bits_set_XOR_atomic(ID_NODE->val, ACTIVE) & ACTIVE),     \
                "Duplicate ID(%lu)\n",                                         \
                IDN);                                                          \
                                                                               \
                                                                               \
        } while (0);                                                           \
    }

//////////////////////////////////////////////////////////////////////
// API MACROS


// All formats should be wrapped in FMT(fmt1, fmt2, fmt3....). Likewise for
// variables/regions
#define VARS(...) __VA_ARGS__
#define FMTS(...) __VA_ARGS__

// A region must be wrapped in REG(ptr, size of region).
#define REG(X, Y) (X), (Y)

#ifdef FRAME_DEBUGGER

// Init/deinit functions
#define INIT_DEBUGGER init_debugger()
#define FREE_DEBUGGER free_debugger()


#define PRINT_FRAME_N(FRAMES, N) printFrameN(FRAMES, N)
#define GET_NFRAMES(FRAMES)      getNFrames(FRAMES)

#define PRINT_FRAMES printFrames(pthread_self())
#define GET_FRAMES   getFrames(pthread_self())
#define HAS_FRAMES   checkFrames(pthread_self())
#define RESET_FRAMES resetFrames(pthread_self())
#define NEW_FRAME(F, V)                                                        \
    FRAME_AT_POINT(__FILE__,                                                   \
                   __func__,                                                   \
                   __LINE__,                                                   \
                   pthread_self(),                                             \
                   FMTS(F),                                                    \
                   VARS(COND_WRAPPER(V)));

#define PRINT_FRAMES_ID(ID) printFrames(ID)
#define GET_FRAMES_ID(ID)   getFrames(ID)
#define HAS_FRAMES_ID(ID)   checkFrames(ID)
#define RESET_FRAMES_ID(ID) resetFrames(ID)
#define NEW_FRAME_ID(ID, F, V)                                                 \
    FRAME_AT_POINT(__FILE__,                                                   \
                   __func__,                                                   \
                   __LINE__,                                                   \
                   ID,                                                         \
                   FMTS(F),                                                    \
                   VARS(COND_WRAPPER(V)));
#else

#define INIT_DEBUGGER
#define FREE_DEBUGGER

#define PRINT_FRAME_N(FRAMES, N)
#define GET_NFRAMES(FRAMES) 0

#define PRINT_FRAMES
#define GET_FRAMES
#define HAS_FRAMES   0
#define RESET_FRAMES 0
#define NEW_FRAME(F, V)


#define PRINT_FRAMES_ID(ID)
#define GET_FRAMES_ID(ID)
#define HAS_FRAMES_ID(ID)   0
#define RESET_FRAMES_ID(ID) 0
#define NEW_FRAME_ID(ID, F, V)

#endif

//////////////////////////////////////////////////////////////////////
// API functions
void            printFrame(frame_data_t * frame_data_s, uint32_t index);
void            addFrame(uint64_t ID, frame_data_t * frame_data_s);
void            printFrames(uint64_t ID);
void            freeFrames(void * ptr);
uint32_t        checkFrames(uint64_t ID);
uint32_t        resetFrames(uint64_t ID);
frame_data_t ** getFrames(uint64_t ID);
void            printFrameN(frame_data_t ** frames, uint32_t frame_number);
uint32_t        getNFrames(frame_data_t ** frames);
void *          findFrame(uint64_t ID);
void *          newID_get(uint64_t ID);
#endif
