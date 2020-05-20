#ifndef _ARG_H_
#define _ARG_H_

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//////////////////////////////////////////////////////////////////////
#include <general-config.h>
#include <helpers/util.h>
//////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
// simple argument parsing with documentation

// options can be typed as int32_t, char, char*, bool, double, or handled by a
// special function positional parameters are always string and must be listed
// in order a special type Rest means all rest are returned as a point32_ter

// don't change constants unless you fix lots of other stuff
typedef enum {
    Integer   = 1,
    Character = 2,
    String    = 3,
    Boolean   = 4,
    Double    = 5,
    Function,
    Help,
    Increment,
    Set,
    Toggle,
    EndOptions /* use this to allow user to indicate end of positional args,
                  typically it will be for -- */
} ArgType;

// don't renumber these.  We depend on value of KindPositional >
// (KindOption & KindHelp), KindEnd > (all), KindRest > KindPositional
typedef enum {
    KindEnd        = 4,
    KindHelp       = 1,
    KindPositional = 2,
    KindOption     = 0,
    KindRest       = 3
} ArgKind;


// type of function that handles option parsing for type Function.
// gets # of args left and point32_ter to argv.  if argv == NULL return
// description string, else return # of args consumed
typedef const char * (*argOptionParser)(int32_t argc, char ** argv);

#define ArgGetDefault -1
#define ArgGetDesc    -2

typedef struct _argoption ArgOption;
struct _argoption {
    ArgKind      kind;    /* option, positionl, rest, end, help */
    ArgType      type;    /* type of the argument/option */
    const char * longarg; /* also name of positional arg for doc purposes */
    int32_t      required;
    void *       dest;
    const char * desc;
};

typedef struct _argdef ArgDefs;

struct _argdef {
    // public members
    ArgOption *  args;
    const char * progdoc;
    const char * version;
    void (*doneParsing)(void);
};

int32_t parseArgs(int32_t argc, char ** argv, ArgDefs * def);

////////////////////////////////////////////////////////////////
// for more complicated arg parsing, i.e., including parsers of submodules, make
// an ArgParser, then call parse on it

typedef struct _argparserlist ArgParserNode;
struct _argparserlist {
    int32_t         main;
    ArgDefs *       parser;
    ArgParserNode * next;
};

typedef struct _argParser ArgParser;
struct _argParser {
    ArgParserNode * parsers;
    ArgDefs *       mainProg;
};

ArgParser * createArgumentParser(ArgDefs * def);
void        freeArgumentParser(ArgParser * ap);
void        addArgumentParser(ArgParser * ap, ArgDefs * def, int32_t order);
int32_t     parseArguments(ArgParser * ap, int32_t argc, char ** argv);
void        freeCommandLine(void);

#endif
