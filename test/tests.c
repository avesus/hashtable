#include "tests.h"

#define TEST_ALIST

#ifdef TEST_ALIST
#include <datastruct/arrlist.h>
#endif

int32_t verbose = 0;
int32_t rseed   = 0;

// clang-format off
#define Version "0.1"
static ArgOption args[] = {
  // Kind,        Method,		name,	    reqd,  variable,		help
  { KindOption,   Integer, 		"-v", 	    0,     &verbose, 		"Set verbosity level" },
  { KindOption,   Integer, 		"--seed", 	0,     &rseed,  		"Set random number seed" },
  { KindHelp,     Help, 	"-h" },
  { KindEnd }
};
// clang-format on

static ArgDefs argp = { args, "Main", Version, NULL };


#define AL_TEST_SIZE (1 << 25)
static void test_alist();


int
main(int argc, char ** argv) {
    progname = argv[0];

    srand(rseed);
    srandom(rseed);

    INIT_DEBUGGER;

    ArgParser * ap = createArgumentParser(&argp);
    if (parseArguments(ap, argc, argv)) {
        die("Error parsing arguments");
    }
    freeCommandLine();
    freeArgumentParser(ap);

#ifdef TEST_ALIST
    PRINT(LOW_VERBOSE, "Testing Array List with %d Test Size\n", AL_TEST_SIZE);
    test_alist();
#endif

    FREE_DEBUGGER;
    return 0;
}


static void
test_alist() {
    for (int32_t i = 1; i < AL_TEST_SIZE; i += (random() % (AL_TEST_SIZE / 4)) + 1) {
        int64_t      j;
        uint32_t      count;
        idx_node_t * temp_idx;
        arr_node_t * temp;
        
        arr_list_t * alist = init_alist(i);
        assert(alist);

        PRINT(MED_VERBOSE, "[%d / %d]: Testing Add\n", i, AL_TEST_SIZE);
        for (j = 0; j < AL_TEST_SIZE; j++) {
            add_node(alist, (void *)j);
        }

        PRINT(MED_VERBOSE, "[%d / %d]: Testing Get IDX\n", i, AL_TEST_SIZE);
        for (j = 0; j < AL_TEST_SIZE; j++) {
            arr_node_t * n = get_node_idx(alist, j);
            DBG_ASSERT(n, "NULL returned by get_node_idx(%p, %d)\n", alist, j);
            DBG_ASSERT(n->data == (void *)j,
                       "Data does not match: %p != %p\n",
                       n->data,
                       (void *)j);
        }
        DBG_ASSERT(alist->nitems == AL_TEST_SIZE,
                   "Error: Invalid nitems: %d\n",
                   alist->nitems);

        PRINT(MED_VERBOSE, "[%d / %d]: Testing Remove IDX\n", i, AL_TEST_SIZE);
        for (j = 0; j < AL_TEST_SIZE; j += 2) {
            remove_node_idx(alist, j);
        }

        for (j = 0; j < AL_TEST_SIZE; j += 2) {
            assert(!get_node_idx(alist, j));
        }

        PRINT(MED_VERBOSE, "[%d / %d]: Testing Free IDX que\n", i, AL_TEST_SIZE);
        count    = 0;
        temp_idx = alist->free_idx_que;
        while (temp_idx) {
            count++;
            temp_idx = temp_idx->next;
        }
        DBG_ASSERT(count == (AL_TEST_SIZE / 2),
                   "Error: invalid number of free idx nodes (%d vs %d)\n",
                   count,
                   AL_TEST_SIZE / 2);


        PRINT(MED_VERBOSE, "[%d / %d]: Testing List\n", i, AL_TEST_SIZE);
        j    = AL_TEST_SIZE - 1;
        temp = alist->ll;
        while (temp) {
            assert(temp->data == (void *)j);
            j -= 2;
            temp = temp->next;
        }

        DBG_ASSERT(j == (-1), "Error: Invalid j value (%d)\n", j);

        PRINT(MED_VERBOSE, "[%d / %d]: Testing Re-add\n", i, AL_TEST_SIZE);
        for (j = 0; j < AL_TEST_SIZE; j += 2) {
            add_node(alist, (void *)j);
        }

        assert(alist->free_idx_que == NULL);

        DBG_ASSERT(alist->nitems == AL_TEST_SIZE,
                   "Error: Invalid nitems: %d\n",
                   alist->nitems);

        PRINT(MED_VERBOSE, "[%d / %d]: Testing Remove All\n", i, AL_TEST_SIZE);
        count = 0;
        while (alist->ll) {
            remove_node(alist, alist->ll);
            count++;
        }
        
        DBG_ASSERT(alist->nitems == count,
                   "Error: Invalid llcount: %d\n",
                   count);
        free_alist(alist);
    }

}
