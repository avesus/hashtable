#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>

//////////////////////////////////////////////////////////////////////
#include <helpers/util.h>
#include <helpers/arg.h>
#include <helpers/temp.h>
//////////////////////////////////////////////////////////////////////



int32_t verbose = 0;
char * compile_time_path = NULL;


#define Version "0.1"

static ArgOption args[] = {
  // Kind, 	  Method,		name,	    reqd,  variable,		help
  { KindOption,   Integer, 		"-v", 	    0,     &verbose, 		"Set verbosity level" },
  { KindOption,   String, 		"-p", 	    0,     &compile_time_path,  "Set path for compile-config.h" },
  { KindHelp,     Help, 	"-h" },
  { KindEnd }
};
static ArgDefs argp = { args, "dev", Version, NULL };


#define BASELINE_LINES 12
const char file_constants[BASELINE_LINES][SMALL_PATH_LEN] =
  {
    "#ifndef _COMPILE_CONFIGS_H_",
    "#define _COMPILE_CONFIGS_H_",
    "",
    "//////////////////////////////////////////////////////////////////////",
    "#define NCORES",
    "#define TEMP_PATH",
    "#define TEMP_PATH_TYPE",
    "#define L1_CACHE_LINE_SIZE",
    "#define L1_LOG_CACHE_LINE_SIZE",
    "//////////////////////////////////////////////////////////////////////",
    "",
    "#endif"
  };

//This should be improved
uint32_t
getCacheLineSize(){
  FILE* fp=myfopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
  char buf[16]="";
  if(!fgets(buf, 8, fp)){
    errdie("Couldn't read cache line size\n");
  }

  char * end;
  uint32_t ret = strtol(buf, &end, 10); //base 10
  DBG_ASSERT(end != buf,
	     "Error reading cache line size\n");
  fclose(fp);
  return ret;
}

int32_t main(int32_t argc, char* argv[]){
  compile_time_path = (char *)mycalloc(BIG_PATH_LEN, sizeof(char));
  void * to_free_path = (void *)compile_time_path;
  
  progname = argv[0];
  ArgParser* ap = createArgumentParser(&argp);
  int32_t result = parseArguments(ap, argc, argv);
  if(result){
    errdie("Error parsing arguments");
  }
  freeCommandLine();
  freeArgumentParser(ap);

  if(!strcmp(compile_time_path, "")) {
      die("No file path provided!\n");
  }
  
  char transfer_path[BIG_PATH_LEN] = "";
  if((result = createTempBasePath(transfer_path)) == NOT_FOUND){
    errdie("Couldn't find directory with core temp info. Please check temp.h params\n");
  }
  
  int32_t ncores = sysconf(_SC_NPROCESSORS_ONLN);
  FILE * config_fp = myfopen(compile_time_path, "w");
  for(int32_t i=0; i<BASELINE_LINES; i++){
    if(!strncmp(file_constants[i],
		"#define NCORES",
		strlen("#define NCORES"))){
      fprintf(config_fp, "%s (%dUL)\n", file_constants[i], ncores);
    }
    
    else if(!strncmp(file_constants[i],
		"#define L1_CACHE_LINE_SIZE",
		strlen("#define L1_CACHE_LINE_SIZE"))){
      fprintf(config_fp, "%s (%dUL)\n", file_constants[i], getCacheLineSize());
    }
    else if(!strncmp(file_constants[i],
		     "#define L1_LOG_CACHE_LINE_SIZE",
		     strlen("#define L1_LOG_CACHE_LINE_SIZE"))){
      fprintf(config_fp, "%s (%dUL)\n", file_constants[i], ulog2_32(getCacheLineSize()));
    }
    
    else if(!strncmp(file_constants[i],
		     "#define TEMP_PATH_TYPE",
		     strlen("#define TEMP_PATH_TYPE"))){
      if(result == FOUND_DIR) {
	fprintf(config_fp, "%s DIR_TYPE\n", file_constants[i]);
      }
      else {
	fprintf(config_fp, "%s ZONE_TYPE\n", file_constants[i]);
      }
    }
    
    else if(!strncmp(file_constants[i],
		     "#define TEMP_PATH",
		     strlen("#define TEMP_PATH"))){
      fprintf(config_fp, "%s \"%s\"\n", file_constants[i], transfer_path);
    }

    else{
      fprintf(config_fp, "%s\n", file_constants[i]);
    }
  }
  fclose(config_fp);
  myfree(to_free_path);
}
