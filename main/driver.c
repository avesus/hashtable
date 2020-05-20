#include "driver.h"

#include <datastruct/seq_hashtable.h>

// table that supposedly is fastest hashmap ever....
#include <datastruct/flat_hash_map.hpp>

#define OUR_TABLE   0
#define OTHER_TABLE 1

int32_t  verbose       = 0;
int32_t  rseed         = 0;
uint32_t FHT_TEST_SIZE = (10);
uint32_t Q_PER_INS     = 0;
uint32_t init_size     = 0;
uint32_t which_table   = OUR_TABLE;
// clang-format off
#define Version "0.1"
static ArgOption args[] = {
  // Kind,        Method,		name,	    reqd,  variable,		help
  { KindOption,   Integer, 		"-v", 	    0,     &verbose, 		"Set verbosity level" },
  { KindOption,   Integer, 		"-i",       0,     &init_size,  	"Log_2 for init size of table" },
  { KindOption,   Integer, 		"-s",       0,     &FHT_TEST_SIZE,	"Log 2 for test size" },
  { KindOption,   Integer, 		"-q",       0,     &Q_PER_INS,  	"True value for queries per insert" },
  { KindOption,   Set,   		"-w",       0,     &which_table,  	"dont set for our table, set for other table" },
  { KindOption,   Integer, 		"--seed", 	0,     &rseed,  		"Set random number seed" },
  { KindHelp,     Help, 	"-h" },
  { KindEnd }
};
// clang-format on

static ArgDefs argp = { args, "Main", Version, NULL };

#ifdef FHT_STATS
extern uint64_t nadd;
extern uint64_t success_add;
extern uint64_t fail_add;
extern uint64_t nfind;
extern uint64_t success_find;
extern uint64_t fail_find;

extern uint64_t niter_add;
extern uint64_t natt_add;
extern uint64_t niter_resize;
extern uint64_t natt_resize;
extern uint64_t niter_find;
extern uint64_t natt_find;

extern uint64_t tag_matches;
extern uint64_t false_tag_matches;

extern uint64_t invalid_resize;
extern uint64_t deleted_resize;
extern uint64_t good_resize;
#endif

double
udiv(uint64_t num, uint64_t den) {
    return ((double)num) / ((double)den);
}

static void correct_test();
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

    // code goes here
    struct timespec start, end;
    FHT_TEST_SIZE = (1 << FHT_TEST_SIZE);

#ifdef DEBUG
    correct_test();
#endif

    // init random nodes
    fht_node_t * test_nodes =
        (fht_node_t *)mycalloc(FHT_TEST_SIZE, sizeof(fht_node_t));

    for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
        (test_nodes + i)->key = random();
        (test_nodes + i)->val = i;
    }

    // init random keys (with varying degree of likely hood to be in table)
    uint32_t * test_keys =
        (uint32_t *)mycalloc(FHT_TEST_SIZE * Q_PER_INS, sizeof(uint32_t));
    for (uint32_t i = 0; i < FHT_TEST_SIZE * Q_PER_INS; i++) {
        test_keys[i] = test_nodes[random() % FHT_TEST_SIZE].key;
    }

    if (which_table == OUR_TABLE) {
        flat_hashtable_t * table = fht_init_table(1 << init_size);

        // run perf test
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            fht_add_key(table, (test_nodes + i)->key, (test_nodes + i)->val);
            for (uint32_t j = i * Q_PER_INS; j < (i + 1) * Q_PER_INS; j++) {
                fht_find_key(table, test_keys[j]);
            }
        }
    }
    else {
        ska::flat_hash_map<int32_t, int32_t> table(1 << init_size);
        // run perf test
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            table[test_nodes[i].key] = test_nodes[i].val;
            for (uint32_t j = i * Q_PER_INS; j < (i + 1) * Q_PER_INS; j++) {
                table.find(test_keys[j]);
            }
        }
    }


    clock_gettime(CLOCK_MONOTONIC, &end);
#ifdef FHT_STATS
    fprintf(stderr,
            "Match: %.3lf\n"
            "Nadd : %lu\n\t\t"
            "LAdd : %.3lf\n\t\t"
            "FAdd : %.3lf\n\t\t"
            "SAdd : %.3lf\n"
            "Nfind: %lu\n\t\t"
            "LFind: %.3lf\n\t\t"
            "FFind: %.3lf\n\t\t"
            "SFind: %.3lf\n"
            "Res  : %.3lf\n\t\t"
            "Inv  : %lu\n\t\t"
            "Del  : %lu\n\t\t"
            "Good : %lu\n",
            udiv(false_tag_matches, tag_matches),
            nadd,
            udiv(niter_add, natt_add),
            udiv(fail_add, nadd),
            udiv(success_add, nadd),
            nfind,
            udiv(niter_find, natt_find),
            udiv(fail_find, nfind),
            udiv(success_find, nfind),
            udiv(niter_resize, natt_resize),
            invalid_resize,
            deleted_resize,
            good_resize);
#endif

    fprintf(stderr,
            "S : %lu\nMS: %lu\nUS: %lu\nNS: %lu\n",
            s_diff(end, start),
            ms_diff(end, start),
            us_diff(end, start),
            ns_diff(end, start));


    FREE_DEBUGGER;
    return 0;
}

// basic correctness check. Should put table through enough cases that if there
// is a bug it will catch it
static void
correct_test() {

    uint8_t *          taken = (uint8_t *)mycalloc((1 << 25), sizeof(uint32_t));
    uint32_t           total_unique = 0;
    flat_hashtable_t * table        = fht_init_table(1);
    fht_node_t *       test_nodes =
        (fht_node_t *)mycalloc(2 * FHT_TEST_SIZE, sizeof(fht_node_t));
    for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
        uint32_t rnum         = random() % (1 << 25);
        (test_nodes + i)->key = rnum;
        (test_nodes + i)->val = i;
    }
    for (uint32_t i = FHT_TEST_SIZE; i < 2 * FHT_TEST_SIZE; i++) {
        uint32_t rnum         = (random() % (1 << 25)) + (1 << 25);
        (test_nodes + i)->key = rnum;
        (test_nodes + i)->val = i;
    }

    for (int att = 0; att < 2; att++) {
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(1) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);

            if (taken[test_nodes[i].key] == 0) {
                total_unique += (att == 0);
                assert(fht_delete_key(table, test_nodes[i].key) ==
                       FHT_NOT_DELETED);
                assert(fht_find_key(table, test_nodes[i].key) == FHT_NOT_FOUND);
                assert(fht_add_key(table,
                                   (test_nodes + i)->key,
                                   (test_nodes + i)->val) == FHT_ADDED);
                assert(fht_find_key(table, test_nodes[i].key) == FHT_FOUND);
            }
            else {
                assert(fht_find_key(table, test_nodes[i].key) == FHT_FOUND);
                assert(fht_add_key(table,
                                   (test_nodes + i)->key,
                                   (test_nodes + i)->val) == FHT_NOT_ADDED);
                assert(fht_delete_key(table, test_nodes[i].key) == FHT_DELETED);
                assert(fht_add_key(table,
                                   (test_nodes + i)->key,
                                   (test_nodes + i)->val) == FHT_ADDED);
            }
            taken[test_nodes[i].key] = 1;
        }
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(2) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            assert(fht_find_key(table, test_nodes[i].key) == FHT_FOUND);
            assert(fht_add_key(table,
                               (test_nodes + i)->key,
                               (test_nodes + i)->val) == FHT_NOT_ADDED);
        }
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(3) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            if (taken[test_nodes[i].key] == 1) {
                assert(fht_delete_key(table, test_nodes[i].key) == FHT_DELETED);
            }
            else {
                assert(fht_find_key(table, test_nodes[i].key) == FHT_NOT_FOUND);
            }
            taken[test_nodes[i].key] = 0;
        }
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(4) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            if (taken[test_nodes[i].key] == 0) {
                assert(fht_delete_key(table, test_nodes[i].key) ==
                       FHT_NOT_DELETED);
                assert(fht_find_key(table, test_nodes[i].key) == FHT_NOT_FOUND);
                assert(fht_add_key(table,
                                   (test_nodes + i)->key,
                                   (test_nodes + i)->val) == FHT_ADDED);
            }
            else {
                assert(fht_find_key(table, test_nodes[i].key) == FHT_FOUND);
                assert(fht_add_key(table,
                                   (test_nodes + i)->key,
                                   (test_nodes + i)->val) == FHT_NOT_ADDED);
                assert(fht_delete_key(table, test_nodes[i].key) == FHT_DELETED);
                assert(fht_add_key(table,
                                   (test_nodes + i)->key,
                                   (test_nodes + i)->val) == FHT_ADDED);
            }
            taken[test_nodes[i].key] = 1;
        }

        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(4.5) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            if (taken[test_nodes[i].key] == 1) {
                assert(fht_find_key(table, test_nodes[i].key) == FHT_FOUND);
            }
            else {
                assert(fht_find_key(table, test_nodes[i].key) == FHT_NOT_FOUND);
            }
        }

        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(5) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            if (taken[test_nodes[i].key] == 1) {
                assert(fht_delete_key(table, test_nodes[i].key) == FHT_DELETED);
            }
            else {
                assert(fht_find_key(table, test_nodes[i].key) == FHT_NOT_FOUND);
            }
            taken[test_nodes[i].key] = 0;
        }

        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(6) %d: {%d, %d}\n",
                  i,
                  test_nodes[i + FHT_TEST_SIZE].key,
                  test_nodes[i + FHT_TEST_SIZE].val);

            if (taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] == 0) {
                total_unique += (att == 0);
                assert(
                    fht_delete_key(table, test_nodes[i + FHT_TEST_SIZE].key) ==
                    FHT_NOT_DELETED);
                assert(fht_find_key(table, test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_NOT_FOUND);
                assert(fht_add_key(table,
                                   (test_nodes + i + FHT_TEST_SIZE)->key,
                                   (test_nodes + i + FHT_TEST_SIZE)->val) ==
                       FHT_ADDED);
            }
            else {
                assert(fht_find_key(table, test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_FOUND);
                assert(fht_add_key(table,
                                   (test_nodes + i + FHT_TEST_SIZE)->key,
                                   (test_nodes + i + FHT_TEST_SIZE)->val) ==
                       FHT_NOT_ADDED);
                assert(
                    fht_delete_key(table, test_nodes[i + FHT_TEST_SIZE].key) ==
                    FHT_DELETED);
                assert(fht_add_key(table,
                                   (test_nodes + i + FHT_TEST_SIZE)->key,
                                   (test_nodes + i + FHT_TEST_SIZE)->val) ==
                       FHT_ADDED);
            }
            taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] = 1;
        }
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            if (taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] == 1) {
                assert(fht_find_key(table, test_nodes[i + FHT_TEST_SIZE].key) == FHT_FOUND);
                assert(
                    fht_delete_key(table, test_nodes[i + FHT_TEST_SIZE].key) ==
                    FHT_DELETED);
                taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] = 0;
            }
        }
        for (uint32_t i = 0; i < 2 * FHT_TEST_SIZE; i++) {
            assert(fht_find_key(table, test_nodes[i].key) == FHT_NOT_FOUND);
        }
    }
}
