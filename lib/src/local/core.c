#include <local/core.h>

static const char * hyperthread_base_format =
    "/sys/devices/system/cpu/cpu%d/topology/core_id";


//gets core_id (physical core) for a logical cpu
int32_t
getCoreID(int32_t core				//logical core number
    ) {
  
    char core_id_path[SMALL_PATH_LEN] = "";	//buffer to store path
    //to core_id file
  
    char tmp_buf[SMALL_READ_LEN] = "";		//buffer to store file
    //contents
  
    //create file path using hyperthread_base_format and core
    sprintf(core_id_path, hyperthread_base_format, core);

    //open the file and read from it
    int32_t fd = myopen2(core_id_path, O_RDONLY);
    if(read(fd, tmp_buf, SMALL_READ_LEN) < 0) {
        errdie("Error reading from: %s\n", core_id_path);
    }

    //convert contents to int32_t
    char * end;
    int32_t core_id = strtol(tmp_buf, &end, 10);
    DBG_ASSERT(end != tmp_buf,
               "Unable to extra file contents as integer\n\t%s -> %s\n",
               core_id_path,
               tmp_buf);
    close(fd);
    PRINT(HIGH_VERBOSE,
          "Found Physical (%d -> %d)\n",
          core,
          core_id);
	
    return core_id;
}
