#include "driver.h"

#define myfree free
#define PRINT(V_LEVEL, ...)                                                    \
    {                                                                          \
        if (verbose >= V_LEVEL) {                                              \
            fprintf(stderr, __VA_ARGS__);                                      \
        }                                                                      \
    }
//////////////////////////////////////////////////////////////////////
// Timing unit conversion stuff
#define unit_change (1000)
#define ns_per_sec  (unit_change * unit_change * unit_change)
uint64_t
to_nsecs(struct timespec t) {
    return (t.tv_sec * ns_per_sec + (uint64_t)t.tv_nsec);
}

uint64_t
ns_diff(struct timespec t1, struct timespec t2) {
    return (to_nsecs(t1) - to_nsecs(t2));
}


uint64_t
to_usecs(struct timespec t) {
    return to_nsecs(t) / unit_change;
}

uint64_t
us_diff(struct timespec t1, struct timespec t2) {
    return ns_diff(t1, t2) / (unit_change);
}


uint64_t
to_msecs(struct timespec t) {
    return to_nsecs(t) / (unit_change * unit_change);
}

uint64_t
ms_diff(struct timespec t1, struct timespec t2) {
    return ns_diff(t1, t2) / (unit_change * unit_change);
}


uint64_t
to_secs(struct timespec t) {
    return to_nsecs(t) / (unit_change * unit_change * unit_change);
}

uint64_t
s_diff(struct timespec t1, struct timespec t2) {
    return ns_diff(t1, t2) / (unit_change * unit_change * unit_change);
}


uint32_t
bitcount_64(uint64_t v) {
    uint64_t c;
    c = v - ((v >> 1) & 0x5555555555555555UL);
    c = ((c >> 2) & 0x3333333333333333UL) + (c & 0x3333333333333333UL);
    c = ((c >> 4) + c) & 0x0F0F0F0F0F0F0F0FUL;
    c = ((c >> 8) + c) & 0x00FF00FF00FF00FFUL;
    c = ((c >> 16) + c) & 0x0000FFFF0000FFFFUL;
    c = ((c >> 32) + c) & 0x00000000FFFFFFFFUL;
    return c;
}

// just a wrapper for getting time with rdtsc
uint64_t
grabTSC() {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (((uint64_t)lo) | (((uint64_t)hi) << 32));
}

typedef struct test_node {
    uint32_t key;
    uint32_t val;
} test_node_t;


#include <datastruct/fht_ht.hpp>

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
uint32_t fun_guess     = 0;
// clang-format off
#define Version "0.1"
static ArgOption args[] = {
  // Kind,        Method,		name,	    reqd,  variable,		help
  { KindOption,   Integer, 		"-v", 	    0,     &verbose, 		"Set verbosity level" },
  { KindOption,   Integer, 		"-i",       0,     &init_size,  	"Log_2 for init size of table" },
  { KindOption,   Integer, 		"-s",       0,     &FHT_TEST_SIZE,	"Log 2 for test size" },
  { KindOption,   Integer, 		"-q",       0,     &Q_PER_INS,  	"True value for queries per insert" },
  { KindOption,   Set,   		"-w",       0,     &which_table,  	"dont set for our table, set for other table" },
  { KindOption,   Integer, 		"-f",       0,     &fun_guess,  	"test possible spread functions" },
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
static void test_spread();


int
main(int argc, char ** argv) {

    srand(rseed);
    srandom(rseed);


    ArgParser * ap = createArgumentParser(&argp);
    if (parseArguments(ap, argc, argv)) {
        assert(0);
    }
    freeCommandLine();
    freeArgumentParser(ap);

    if (fun_guess) {
        test_spread();
        return 0;
    }

    // code goes here
    struct timespec start, end;
    FHT_TEST_SIZE = (1 << FHT_TEST_SIZE);

#ifdef DEBUG
    correct_test();
#endif

    // init random nodes
    test_node_t * test_nodes =
        (test_node_t *)calloc(FHT_TEST_SIZE, sizeof(test_node_t));

    for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
        (test_nodes + i)->key = random();
        (test_nodes + i)->val = i;
    }

    // init random keys (with varying degree of likely hood to be in table)
    uint32_t * test_keys =
        (uint32_t *)calloc(FHT_TEST_SIZE * Q_PER_INS, sizeof(uint32_t));
    for (uint32_t i = 0; i < FHT_TEST_SIZE * Q_PER_INS; i++) {
        test_keys[i] = test_nodes[random() % FHT_TEST_SIZE].key;
    }

    if (which_table == OUR_TABLE) {
        //        fht_table<uint32_t, std::string> table(1 << init_size);
        fht_table<uint32_t, uint32_t> table(1 << init_size);
        // run perf test

        clock_gettime(CLOCK_MONOTONIC, &start);
        uint32_t counter = 0;
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            counter ^= table.add((test_nodes + i)->key, (test_nodes + i)->val);
            // counter ^= table.add((test_nodes + i)->key, "");
            for (uint32_t j = i * Q_PER_INS; j < (i + 1) * Q_PER_INS; j++) {
                counter ^= table.find(test_keys[j]);
            }
        }
        volatile uint32_t sink = counter;
    }
    else {
        ska::flat_hash_map<uint32_t, uint32_t> table(1 << init_size);
        // run perf test
        uint32_t counter = 0;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            table[test_nodes[i].key] = test_nodes[i].val;
            // table[test_nodes[i].key] = "";
            for (uint32_t j = i * Q_PER_INS; j < (i + 1) * Q_PER_INS; j++) {
                volatile auto res = table.find(test_keys[j]);
            }
        }
        volatile uint32_t sink = counter;
    }


    clock_gettime(CLOCK_MONOTONIC, &end);

    fprintf(stderr,
            "S : %lu\nMS: %lu\nUS: %lu\nNS: %lu\n",
            s_diff(end, start),
            ms_diff(end, start),
            us_diff(end, start),
            ns_diff(end, start));


    return 0;
}

// basic correctness check. Should put table through enough cases that if there
// is a bug it will catch it
static void
correct_test() {

    uint8_t * taken        = (uint8_t *)calloc((1 << 25), sizeof(uint32_t));
    uint32_t  total_unique = 0;
    fht_table<uint32_t, uint32_t> table;
    test_node_t *                 test_nodes =
        (test_node_t *)calloc(2 * FHT_TEST_SIZE, sizeof(test_node_t));
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
                assert(table.remove(test_nodes[i].key) == FHT_NOT_DELETED);
                assert(table.find(test_nodes[i].key) == FHT_NOT_FOUND);
                assert(table.add((test_nodes + i)->key,
                                 (test_nodes + i)->val) == FHT_ADDED);
                assert(table.find(test_nodes[i].key) == FHT_FOUND);
            }
            else {
                assert(table.find(test_nodes[i].key) == FHT_FOUND);
                assert(table.add((test_nodes + i)->key,
                                 (test_nodes + i)->val) == FHT_NOT_ADDED);
                assert(table.remove(test_nodes[i].key) == FHT_DELETED);
                assert(table.add((test_nodes + i)->key,
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
            assert(table.find(test_nodes[i].key) == FHT_FOUND);
            assert(table.add((test_nodes + i)->key, (test_nodes + i)->val) ==
                   FHT_NOT_ADDED);
        }
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(3) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            if (taken[test_nodes[i].key] == 1) {
                assert(table.remove(test_nodes[i].key) == FHT_DELETED);
            }
            else {
                assert(table.find(test_nodes[i].key) == FHT_NOT_FOUND);
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
                assert(table.remove(test_nodes[i].key) == FHT_NOT_DELETED);
                assert(table.find(test_nodes[i].key) == FHT_NOT_FOUND);
                assert(table.add((test_nodes + i)->key,
                                 (test_nodes + i)->val) == FHT_ADDED);
            }
            else {
                assert(table.find(test_nodes[i].key) == FHT_FOUND);
                assert(table.add((test_nodes + i)->key,
                                 (test_nodes + i)->val) == FHT_NOT_ADDED);
                assert(table.remove(test_nodes[i].key) == FHT_DELETED);
                assert(table.add((test_nodes + i)->key,
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
                assert(table.find(test_nodes[i].key) == FHT_FOUND);
            }
            else {
                assert(table.find(test_nodes[i].key) == FHT_NOT_FOUND);
            }
        }

        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            PRINT(MED_VERBOSE,
                  "\r(5) %d: {%d, %d}\n",
                  i,
                  test_nodes[i].key,
                  test_nodes[i].val);
            if (taken[test_nodes[i].key] == 1) {
                assert(table.remove(test_nodes[i].key) == FHT_DELETED);
            }
            else {
                assert(table.find(test_nodes[i].key) == FHT_NOT_FOUND);
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
                assert(table.remove(test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_NOT_DELETED);
                assert(table.find(test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_NOT_FOUND);
                assert(table.add((test_nodes + i + FHT_TEST_SIZE)->key,
                                 (test_nodes + i + FHT_TEST_SIZE)->val) ==
                       FHT_ADDED);
            }
            else {
                assert(table.find(test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_FOUND);
                assert(table.add((test_nodes + i + FHT_TEST_SIZE)->key,
                                 (test_nodes + i + FHT_TEST_SIZE)->val) ==
                       FHT_NOT_ADDED);
                assert(table.remove(test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_DELETED);
                assert(table.add((test_nodes + i + FHT_TEST_SIZE)->key,
                                 (test_nodes + i + FHT_TEST_SIZE)->val) ==
                       FHT_ADDED);
            }
            taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] = 1;
        }
        for (uint32_t i = 0; i < FHT_TEST_SIZE; i++) {
            if (taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] == 1) {
                assert(table.find(test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_FOUND);
                assert(table.remove(test_nodes[i + FHT_TEST_SIZE].key) ==
                       FHT_DELETED);
                taken[test_nodes[i + FHT_TEST_SIZE].key - (1 << 25)] = 0;
            }
        }
        for (uint32_t i = 0; i < 2 * FHT_TEST_SIZE; i++) {
            assert(table.find(test_nodes[i].key) == FHT_NOT_FOUND);
        }
    }
}


static void
test_spread() {

    // generate primes up to some reasonable value (basically stopping when * 64
    // would overflow)
    const uint32_t max_prime = 1 << 12;
    uint32_t *     sieve     = (uint32_t *)calloc(max_prime, sizeof(uint32_t));

    uint32_t max_prime_sqrt = (uint32_t)(sqrt((double)max_prime) + 1);
    uint32_t nprimes        = max_prime - 2;
    sieve[0]                = 1;
    sieve[1]                = 1;
    for (uint32_t i = 0; i < max_prime_sqrt; i++) {
        if (sieve[i]) {
            continue;
        }
        for (uint32_t j = i * i; j < max_prime; j += i) {
            sieve[j] = 1;
            nprimes--;
        }
    }
    uint32_t * primes = (uint32_t *)calloc(nprimes, sizeof(uint32_t));

    uint32_t iter = 0;
    for (uint32_t i = 0; i < max_prime; i++) {
        if (sieve[i]) {
            continue;
        }
        primes[iter++] = i;
    }

    myfree(sieve);


    nprimes               = iter;
    const uint32_t target = fun_guess;
    for (uint32_t start = 0; start < 64 - target; start++) {
        // test linear
        for (uint32_t i = 0; i < nprimes && 0; i++) {
            uint64_t mask = 0;
            for (uint32_t j = start; j < 64; j++) {
                if (mask & ((1UL) << ((primes[i] * j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j) & 63));
            }
            if (mask == (~(0UL))) {
                fprintf(stderr, "Func(%d): %d x\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = 0;
            for (uint32_t j = start; j < 64; j++) {
                if (mask & ((1UL) << ((primes[i] ^ j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] ^ j) & 63));
            }
            if (mask == (~(0UL))) {
                fprintf(stderr, "Func(%d): %d XOR x\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = 0;
            for (uint32_t j = start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr, "Func(%d): %d x^2\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = 0;
            for (uint32_t j = start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j + j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j + j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr, "Func(%d): %d x^2 + x\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            for (uint32_t ii = 0; ii < nprimes; ii++) {
                uint64_t mask = 0;
                for (uint32_t j = start; j < target; j++) {
                    if (mask & ((1UL) << ((primes[i] * j * j + primes[ii] * j) &
                                          63))) {
                        break;
                    }
                    mask |=
                        ((1UL) << ((primes[i] * j * j + primes[ii] * j) & 63));
                }
                if (bitcount_64(mask) == target && i) {
                    fprintf(stderr,
                            "Func(%d): %d x^2 + %d x\n",
                            start,
                            primes[i],
                            primes[ii]);
                }
            }
        }
        uint64_t seed_mask = 0;
        for (uint32_t i = start; i < (16u) + start; i++) {
            seed_mask |= (((1UL) << ((i * i) & 63)));
        }

        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = seed_mask;
            for (uint32_t j = 16 + start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr, "Seed Func(%d): %d x^2\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = seed_mask;
            for (uint32_t j = 16 + start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j + j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j + j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr,
                        "Seed Func(%d): %d x^2 + x\n",
                        start,
                        primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            for (uint32_t ii = 0; ii < nprimes; ii++) {
                uint64_t mask = seed_mask;
                for (uint32_t j = 16 + start; j < target; j++) {
                    if (mask & ((1UL) << ((primes[i] * j * j + primes[ii] * j) &
                                          63))) {
                        break;
                    }
                    mask |=
                        ((1UL) << ((primes[i] * j * j + primes[ii] * j) & 63));
                }
                if (bitcount_64(mask) == target && i) {
                    fprintf(stderr,
                            "Seed Func(%d): %d x^2 + %d x\n",
                            start,
                            primes[i],
                            primes[ii]);
                }
            }
        }
        seed_mask = 0;
        for (uint32_t i = start; i < (16u) + start; i++) {
            seed_mask |= 2 * i;
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = seed_mask;
            for (uint32_t j = 16 + start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr, "Seed 2 Func(%d): %d x^2\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = seed_mask;
            for (uint32_t j = 16 + start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j + j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j + j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr,
                        "Seed 2 Func(%d): %d x^2 + x\n",
                        start,
                        primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            for (uint32_t ii = 0; ii < nprimes; ii++) {
                uint64_t mask = seed_mask;
                for (uint32_t j = 16 + start; j < target; j++) {
                    if (mask & ((1UL) << ((primes[i] * j * j + primes[ii] * j) &
                                          63))) {
                        break;
                    }
                    mask |=
                        ((1UL) << ((primes[i] * j * j + primes[ii] * j) & 63));
                }
                if (bitcount_64(mask) == target && i) {
                    fprintf(stderr,
                            "Seed 2 Func(%d): %d x^2 + %d x\n",
                            start,
                            primes[i],
                            primes[ii]);
                }
            }
        }
        seed_mask = 0;
        for (uint32_t i = start; i < (16u) + start; i++) {
            seed_mask |= 3 * i;
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = seed_mask;
            for (uint32_t j = 16 + start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr, "Seed 3 Func(%d): %d x^2\n", start, primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            uint64_t mask = seed_mask;
            for (uint32_t j = 16 + start; j < target; j++) {
                if (mask & ((1UL) << ((primes[i] * j * j + j) & 63))) {
                    break;
                }
                mask |= ((1UL) << ((primes[i] * j * j + j) & 63));
            }
            if (bitcount_64(mask) == target && i) {
                fprintf(stderr,
                        "Seed 3 Func(%d): %d x^2 + x\n",
                        start,
                        primes[i]);
            }
        }
        for (uint32_t i = 0; i < nprimes; i++) {
            for (uint32_t ii = 0; ii < nprimes; ii++) {
                uint64_t mask = seed_mask;
                for (uint32_t j = 16 + start; j < target; j++) {
                    if (mask & ((1UL) << ((primes[i] * j * j + primes[ii] * j) &
                                          63))) {
                        break;
                    }
                    mask |=
                        ((1UL) << ((primes[i] * j * j + primes[ii] * j) & 63));
                }
                if (bitcount_64(mask) == target && i) {
                    fprintf(stderr,
                            "Seed 3 Func(%d): %d x^2 + %d x\n",
                            start,
                            primes[i],
                            primes[ii]);
                }
            }
        }
    }
}
