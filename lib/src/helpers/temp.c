#include <helpers/temp.h>

// two methods of finding temperatures. Via temperature zone structure
// of finding directory containing data for all cores.
#define DIR_TYPE  1
#define ZONE_TYPE 2

// if configs where done compile time this will already by set,
// otherwise its set in createTempbasepath
static int32_t temp_path_type = TEMP_PATH_TYPE;

// number of base paths to check
#define BASE_PATHS 2

// number of sub paths to recurse with
#define SUB_PATHS 2

// number of target files to check
#define TARGET_FILES 1

// target content matches
#define TARGET_CONTENTS 1


// search strategy is to start recursively checking are
// core_temp_base_paths. open the directory and check for all target
// files, and if a target file exists check its content vs
// target_contents. If target_contents is found we found a possible
// directory. Otherwise continue recursively searching directorys that
// match with core_temp_sub_paths. This is for directory search method.

// these are base_paths I know of
const static char core_temp_base_paths[BASE_PATHS][BIG_PATH_LEN] = {
    "/sys/class/hwmon",
    "/sys/devices/platform"
};

// sub paths I know of i.e base_paths/sub_paths*/sub_paths*/.../
const static char core_temp_sub_paths[SUB_PATHS][SMALL_PATH_LEN] = {
    "hwmon",
    "coretemp."
};

// file that contains info about whether its correct dir
const static char target_files[TARGET_FILES][SMALL_PATH_LEN] = { "name" };

// contents we are looking for in that file
const static char target_contents[TARGET_CONTENTS][SMALL_PATH_LEN] = {
    "coretemp"
};


// if cant find directory will use zone. This is not ideal but better
// than nothing. We will not be able to get different temperatures for
// different cores with this approach.

// number of zones currently able to look for
#define ZONE_TARGET_CONTENTS 2

// we are looking through /sys/class/thermal/thermal_zone*/type
const static char zone_format_path[SMALL_PATH_LEN] =
    "/sys/class/thermal/thermal_zone%d/type";

const static char zone_target_contents[ZONE_TARGET_CONTENTS][SMALL_PATH_LEN] = {
    "x86_pkg_temp",  // max temp of all cores x86_64

    "bcm2835_thermal"  // temp on my raspberry pi,
                       // probably not generic
};


static char temp_base_path[BIG_PATH_LEN] = TEMP_PATH;


// checks if a given directory is a valid subpath. Return 1 id d_name
// is a valid subpath, otherwise returns 0
// d_name : directory to test
static int32_t
isSubPath(char * d_name) {

    // check all sub_paths
    for (int32_t i = 0; i < SUB_PATHS; i++) {

        // if match return 1
        if (!strncmp(d_name,
                     core_temp_sub_paths[i],
                     strlen(core_temp_sub_paths[i]))) {
            return 1;
        }
    }

    // if no matches round return 0
    return 0;
}

// checks if a given file contains the expected content. Basically to
// affirm a given directory is correct we look for target file and
// ensure it contains target content.

// In the directory we are checking the file "name" contains
//"coretemp". Returns 1 if a match is found, otherwise returns 0.

// in the zone case we checking that file thermal_zone*/type contains
//"x86_pkg_temp" or "bcm2835_thermal"

static int32_t
isTargetContent(
    char *     content,
    int32_t    ntargets,
    const char tar_content[MAX(TARGET_CONTENTS, ZONE_TARGET_CONTENTS)]
                          [SMALL_PATH_LEN]) {

    for (int32_t i = 0; i < ntargets; i++) {
        if (!strncmp(content, tar_content[i], strlen(tar_content[i]))) {
            return 1;
        }
    }
    return 0;
}


// searches for correct temperature zone
static int32_t
testTempPath_zone() {

    // temp buffer to store formatted zone path
    char possible_zone_path[SMALL_PATH_LEN] = "";

    // buffer to read "name" file int32_to to check for "coretemp"
    char check_buf[SMALL_READ_LEN] = "";

    // will break once no more temp zones or return if proper temp zone
    // is found

    for (int32_t zone_iter = 0;; zone_iter++) {
        sprintf(possible_zone_path, zone_format_path, zone_iter);

        // file doesnt exist, since it goes up sequentially this means we
        // never found it/never will
        if (!myfexists(possible_zone_path)) {
            break;
        }

        int32_t fd = myopen2(possible_zone_path, O_RDONLY);
        if (read(fd, check_buf, SMALL_READ_LEN) <= 0) {
            errdie("Error getting name for: %s\n", possible_zone_path);
        }
        PRINT(HIGH_VERBOSE,
              "Checking zone path:\n\t%s\n\t\t-> %s\n",
              possible_zone_path,
              check_buf);

        if (isTargetContent(check_buf,
                            ZONE_TARGET_CONTENTS,
                            zone_target_contents)) {

            int32_t zone_path_len = strlen(possible_zone_path) - strlen("type");
            strncpy(temp_base_path, possible_zone_path, zone_path_len);
            strncpy(temp_base_path + zone_path_len, "temp", strlen("temp"));
            PRINT(LOW_VERBOSE, "Found temp zone at:\n\t%s\n", temp_base_path);
            close(fd);
            return FOUND_ZONE;
        }
        close(fd);
    }

    // outside break
    return NOT_FOUND;
}

// recursive function for finding temperature path
// test_path: path to test on. This could be a subpath towards final dir or
//            final dir
static int32_t
testTempPath_dir(char * test_path) {

    // for recursively checking
    char possible_temp_path[BIG_PATH_LEN] = "";

    // buffer to read "name" file int32_to to check for "coretemp"
    char check_buf[SMALL_READ_LEN] = "";

    struct dirent * d_ent;
    DIR *           dir = opendir(test_path);
    if (dir == NULL) {
        return NOT_FOUND;
    }

    while ((d_ent = readdir(dir)) != NULL) {
        // we are looking for path so only both with directories or symlinks
        if (isSubPath(d_ent->d_name)) {

            // sanity check. If this assert fails this code is not proper
            // for the system
            DBG_ASSERT(d_ent->d_type == DT_DIR || d_ent->d_type == DT_LNK,
                       "This does not necessarily mean an error"
                       "something has gone wrong but more likely"
                       "than not this code is configured for your OS/version");

            for (int32_t i = 0; i < TARGET_FILES; i++) {
                sprintf(possible_temp_path,
                        "%s/%s/%s",
                        test_path,
                        d_ent->d_name,
                        target_files[i]);

                if (myfexists(possible_temp_path)) {  // name file exists
                    int32_t fd = myopen2(possible_temp_path, O_RDONLY);
                    if (read(fd, check_buf, SMALL_READ_LEN) <= 0) {
                        errdie("Error getting name for: %s\n", test_path);
                    }

                    // this is the temp zone we are trying to find
                    if (isTargetContent(check_buf,
                                        TARGET_CONTENTS,
                                        target_contents)) {
                        strncpy(temp_base_path,
                                possible_temp_path,
                                strlen(possible_temp_path) - strlen("/name"));
                        PRINT(LOW_VERBOSE,
                              "Found temp dir at:\n\t%s\n",
                              temp_base_path);
                        close(fd);
                        closedir(dir);
                        return FOUND_DIR;
                    }
                    // name file exists but incorrect path so not correct
                    // directory and no need to continue searching recursively
                    close(fd);
                }
                else {  // no name file exists
                    // unset attempt for "/name then recursive call
                    memset(possible_temp_path + strlen(possible_temp_path) -
                               strlen("/name"),
                           0,
                           strlen("/name"));
                    if (testTempPath_dir(possible_temp_path) == FOUND_DIR) {
                        // if recursive call found it we are done as finding
                        // above will set temp_base_path properly
                        closedir(dir);
                        return FOUND_DIR;
                    }
                }
            }
        }
    }
    // outside break
    closedir(dir);
    return NOT_FOUND;
}

// finds directory that contains temp data for each core
// that path is set in the global base_path
// outstr: string to write temp_base_path to. This is for compile time config
int32_t
createTempBasePath(char * outstr) {
    // trying all known base_paths
    for (int32_t i = 0; i < BASE_PATHS; i++) {

        // there are potentially multiple ways to find it,
        // core_temp_base_path array is of possible starting paths.
        // variable temp_base_path will be set when found.
        PRINT(HIGH_VERBOSE,
              "Looking for directory path to temp files from start:\n\t%s\n",
              core_temp_base_paths[i]);
        if (testTempPath_dir((char *)core_temp_base_paths[i]) == FOUND_DIR) {
            if (outstr != NULL) {

                /* Feels unnecessary to me, temp_base_path < BIG_PATH_LEN
                 * (512 normally) so just ensure outstr bigger... */
                strcpy(outstr, temp_base_path);  // NOLINT
            }

            temp_path_type = DIR_TYPE;
            return FOUND_DIR;
        }
    }

    PRINT(HIGH_VERBOSE, "Looking for zone file for temp regulation.");
    PRINT(HIGH_VERBOSE,
          "This is not ideal if individual core data is "
          "important and will cause false waiting if other "
          "programs on other cores are running\n");
    if (testTempPath_zone() == FOUND_ZONE) {
        if (outstr != NULL) {
            /* Feels unnecessary to me, temp_base_path < BIG_PATH_LEN
             * (512 normally) so just ensure outstr bigger... */
            strcpy(outstr, temp_base_path);  // NOLINT
        }

        temp_path_type = ZONE_TYPE;
        return FOUND_ZONE;
    }

    return NOT_FOUND;
}

// reads temperature of a given fd. Basically this just reads file,
// resets file offset so it can be read again, and returns the content
// as a float
// fd: fd to file to read temp from
float
getTemp(int32_t fd) {

    // read contents int32_to tmp_buf
    char tmp_buf[SMALL_READ_LEN] = "";
    if (read(fd, tmp_buf, SMALL_READ_LEN) <= 0) {
        errdie("Error reading temperature\n");
    }

    // reset file offset
    if (lseek(fd, 0, SEEK_SET) == -1) {
        errdie("Error reseting temp fd\n");
    }

    // return content as float
    char * end;
    return strtof(tmp_buf, &end);
}

// reads all files that the fds array contains and returns dynamically
// allocated array with the temperatures of each file
// ncores : number of cores/file
// fds    :  pointer to array of fds to temp files
float *
readNStore(int32_t ncores, int32_t * fds) {

    // alocate temperature array
    float * temp_line = (float *)mycalloc(ncores, sizeof(float));
    for (int32_t i = 0; i < ncores; i++) {

        // get temperature from each fd
        temp_line[i] = getTemp(fds[i]);
    }

    // return array
    return temp_line;
}


static int32_t * getTempFiles_zone(int32_t ncores, cpu_set_t * cpus);
static int32_t * getTempFiles_dir(int32_t ncores, cpu_set_t * cpus);

// returns array of file descriptors opening to temp_input file for
// each core set by core_mask. Here was are trying to match
// temp<INT>_label to get core the temp<INT>_input file corresponds to
int32_t *
getTempFiles(int32_t ncores, cpu_set_t * cpus) {
    if (temp_path_type == DIR_TYPE) {
        return getTempFiles_dir(ncores, cpus);
    }
    else if (temp_path_type == ZONE_TYPE) {
        return getTempFiles_zone(ncores, cpus);
    }
    else {
        errdie("Unkown path type\n");
        return NULL;
    }
}


// gets temp files if using temperature directory to find individual
// cpu temperatures
// ncores  : number of cores we want to get temp file for
// cpus    : cpu set of the cores in question
#define REG_MATCH 0
static int32_t *
getTempFiles_dir(int32_t ncores, cpu_set_t * cpus) {

    regex_t regex_name, regex_label;
    int32_t ret;

    // no cflags needed
    const char * expr_name  = "^temp[0-9]+_label$";
    const char * expr_label = "^Core[[:space:]][0-9]+$";
    ret = regcomp(&regex_name, expr_name, REG_EXTENDED | REG_NEWLINE);
    if (ret) {
        errdie("Error compiling regex: %s\n", expr_name);
    }
    ret = regcomp(&regex_label, expr_label, REG_EXTENDED | REG_NEWLINE);
    if (ret) {
        errdie("Error compiling regex: %s\n", expr_label);
    }

    int32_t * fds = (int32_t *)mycalloc(ncores, sizeof(int32_t));

    // this is likely an over allocation but will definetly be enough to
    // store all temperature files
    int32_t opened_fds = 0;  // even though 0 can correspond
    // to a valid index, we cant
    // fall through without opening
    // any fds, another error will
    // be thrown by then

    int32_t * all_fds = (int32_t *)mycalloc(NCORES, sizeof(int32_t));

    char            tmp_path[BIG_PATH_LEN]    = "";
    char            tmp_buf[SMALL_READ_LEN]   = "";
    char            label_buf[SMALL_READ_LEN] = "";
    int32_t         core_num                  = 0;
    struct dirent * d_ent;
    DIR *           dir = opendir(temp_base_path);
    if (dir == NULL) {
        errdie("Couldn't open dir at: %s\n", temp_base_path);
    }

    while ((d_ent = readdir(dir)) != NULL) {
        ret = regexec(&regex_name, d_ent->d_name, 0, NULL, 0);
        if (ret == REG_MATCH) {

            // As with above. If this assert fails this code is not proper
            // for the system
            DBG_ASSERT(d_ent->d_type == DT_REG,
                       "This does not necessarily mean an error"
                       "something has gone wrong but more likely"
                       "than not this code is configured for your OS/version");

            sprintf(tmp_path, "%s/%s", temp_base_path, d_ent->d_name);
            int32_t label_fd = myopen2(tmp_path, O_RDONLY);

            memset(label_buf, 0, SMALL_READ_LEN);
            if (read(label_fd, label_buf, SMALL_READ_LEN) < 0) {
                errdie("Error reading: %s\n", tmp_path);
            }

            // we are looking for "Core <CORE_NUMBER>"
            ret = regexec(&regex_label, label_buf, 0, NULL, 0);
            if (ret == REG_MATCH) {

                DBG_ASSERT(sscanf(label_buf, "%s %d", tmp_buf, &core_num) == 2,
                           "Unable to determine core from label file in: %s\n",
                           tmp_path);
                DBG_ASSERT(all_fds[core_num] == 0,
                           "fd for core(%d) already set\n");

                PRINT(HIGH_VERBOSE,
                      "Found label file for core (%d) at\n\t%s\n",
                      core_num,
                      tmp_path);

                int32_t label_path_len = strlen(tmp_path);
                int32_t core_path_len  = label_path_len - strlen("label");
                strncpy(tmp_path + core_path_len, "input\0", strlen("input\0"));
                all_fds[core_num] = myopen2(tmp_path, O_RDONLY);
                opened_fds |= (1 << core_num);
            }
            else if (ret != REG_NOMATCH) {
                errdie("Error with regex matching: expr(%s), string(%s)\n",
                       expr_label,
                       label_buf);
            }

            // close label file
            close(label_fd);
        }
        else if (ret != REG_NOMATCH) {
            errdie("Error with regex matching: expr(%s), string(%s)\n",
                   expr_name,
                   d_ent->d_name);
        }
    }


    // this is not the worlds most optimized code (runs in physical_cores
    //+ logical_cores) as opposed to physical_cores + log(logical_cores)
    // but given that is off critical path I don't see an issue. It also
    // drastically decreases the amount of logic/work that would need to
    // go int32_to determining hyperthreading on the given cpu
    int32_t fd_index = 0;

    // used for closing fds that are not used. After this next loop all
    // 0 bits correspond to an fd that was opening but will not be used
    // thus needs to be closed
    int32_t used_fds = ~opened_fds;


    uint64_t * cpu_mask = (uint64_t *)cpus->__bits;
    uint64_t * end      = cpu_mask + ((NCORES + (sizeof_bits(uint64_t) - 1)) /
                                 sizeof_bits(uint64_t));

    int32_t iter = 0;
    while (cpu_mask < end) {
        uint64_t tmp_core_mask = *cpu_mask;
        cpu_mask++;
        while (tmp_core_mask) {
            int32_t i = ff1_64_ctz_nz(tmp_core_mask);
            tmp_core_mask ^= ((1UL) << i);
            i += iter * sizeof_bits(uint64_t);

            int32_t core_id = getCoreID(i);

            used_fds |= (1 << core_id);

            // ensure we actually found and opened this file
            DBG_ASSERT(opened_fds & (1 << core_id),
                       "Was unable to find temp file for core: %d\n",
                       core_id);

            // its fine for 2 cores to point32_t to same file as its read only
            // and sequential within barrier.
            fds[fd_index] = all_fds[core_id];
            fd_index++;
            ncores--;
        }
        iter++;
    }

    used_fds = ~used_fds;
    while (used_fds) {
        int32_t to_close = ff1_32_ctz_nz(used_fds);
        used_fds ^= (1 << to_close);
        close(all_fds[to_close]);
    }
    myfree(all_fds);

    regfree(&regex_name);
    regfree(&regex_label);

    // if this assert didn't pass we didn't find temperature file for all
    // cores specified
    DBG_ASSERT(ncores == 0, "Unable to find all temperature files\n");

    closedir(dir);
    return fds;
}


// gets cpu temperature using zone (all cpus use same fd in this case).
// ncores : number of cores we want to get temp file for
// cpus   : cpu set of the cores in question


static int32_t *
getTempFiles_zone(int32_t ncores, cpu_set_t * cpus) {

    int32_t * fds = (int32_t *)mycalloc(ncores, sizeof(int32_t));

    int32_t temp_fd = myopen2(temp_base_path, O_RDONLY);

    for (int32_t i = 0; i < ncores; i++) {
        fds[i] = temp_fd;
    }

    return fds;
}
