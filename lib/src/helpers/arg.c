////////////////////////////////////////////////////////////////
// simple argument parsing with documentation

// options can be typed as int32_t, char, char *, bool, double, or handled by a
// special function positional parameters are always string and must be listed
// in order a special type Rest means all rest are returned as a point32_ter

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//////////////////////////////////////////////////////////////////////
#include <general-config.h>
#include <helpers/arg.h>
#include <helpers/util.h>
//////////////////////////////////////////////////////////////////////

#define FALSE 0
#define TRUE  (!FALSE)

static char *       commandLine;
static const char * pname;

// order is based on how enums are defined in ArgType
static const char * type2fmt[] = { "",         "<int32_t>", "<char>",
                                   "<string>", "<bool>",    "<double>",
                                   "",         "",          "" };

static void usage(const char * pname, ArgParser * def);

static void
argdie(ArgParser * def, const char * fmt, ...) {
    va_list ap;

    va_start(ap, fmt);
    fprintf(stderr, "%s: Usage Error: ", pname);
    vfprintf(stderr, fmt, ap);  // NOLINT /* Bug in Clang-Tidy */
    va_end(ap);
    fprintf(stderr, "\n");
    usage(pname, def);
    exit(-1);
}


static const char *
arg2str(ArgOption * desc) {
    static char buffer[128];
    char *      p   = buffer;
    char        sep = (desc->kind == KindPositional) ? ':' : ' ';

    if (desc->kind == (ArgKind)Help) {
        return "[-h]";
    }

    if ((desc->kind == KindPositional) && (desc->required)) {
        *p++ = '<';
    }
    else if (!desc->required) {
        *p++ = '[';
    }

    switch (desc->type) {
        case Integer:
        case Character:
        case String:
        case Boolean:
        case Double:
            sprintf(p, "%s%c%s", desc->longarg, sep, type2fmt[desc->type]);
            p += strlen(p);
            break;

        case Function:
            sprintf(p,
                    "%s%c%s",
                    desc->longarg,
                    sep,
                    (*(argOptionParser)(desc->dest))(ArgGetDesc, NULL));
            p += strlen(p);
            break;

        case Help:
        case Toggle:
        case Set:
        case Increment:
            sprintf(p, "%s", desc->longarg);
            p += strlen(p);
            break;

        default:
            // new type that we didn't implement yet?
            DBG_ASSERT(0, "Unkown argument type: %d\n", desc->type);
    }
    if (desc->kind == KindRest) {
        strncpy(p, "...", strlen("..."));
        p += strlen(p);
    }

    if ((desc->kind == KindPositional) && (desc->required)) {
        *p++ = '>';
    }
    else if (!desc->required) {
        *p++ = ']';
    }
    *p = 0;
    return buffer;
}


static void
usage(const char * pname, ArgParser * ap) {
    fprintf(stderr, "%s: ", pname);
    // print out shorthand for arguments
    for (ArgParserNode * apn = ap->parsers; apn; apn = apn->next) {
        ArgOption * args = apn->parser->args;
        int32_t     i    = 0;
        while (args[i].kind != KindEnd) {
            fprintf(stderr, " %s", arg2str(args + i));
            i++;
        }
        fprintf(stderr, "\n%s\n", apn->parser->progdoc);
    }

    // Now print individual descriptions
    for (ArgParserNode * apn = ap->parsers; apn; apn = apn->next) {
        ArgOption * args = apn->parser->args;
        int32_t     i    = 0;
        while (args[i].kind != KindEnd) {
            fprintf(stderr,
                    "   %20s\t%s\t",
                    arg2str(args + i),
                    args[i].kind == KindHelp ? "Print32_t this message"
                                             : args[i].desc);
            switch (args[i].type) {
                case Increment:
                case Integer:
                    fprintf(stderr,
                            "(default: %d)",
                            *(int32_t *)(args[i].dest));
                    break;

                case Character:
                    fprintf(stderr, "(default: %c)", *(char *)(args[i].dest));
                    break;

                case String:
                    fprintf(stderr, "(default: %s)", (char *)(args[i].dest));
                    break;

                case Toggle:
                case Set:
                case Boolean:
                    fprintf(stderr,
                            "(default: %s)",
                            *(int32_t *)(args[i].dest) ? "true" : "false");
                    break;

                case Double:
                    fprintf(stderr,
                            "(default: %lf)",
                            *(double *)(args[i].dest));
                    break;

                case Function:
                    fprintf(stderr,
                            "(default: %s)",
                            (*(argOptionParser)(args[i].dest))(ArgGetDefault,
                                                               NULL));
                    break;

                case Help:
                    break;

                default:
                    break;
            }
            fprintf(stderr, "\n");
            i++;
        }
    }
}


void
makeCommandline(int32_t argc, char ** argv) {
    int32_t len = 2;
    for (int32_t i = 0; i < argc; i++) {
        len += (1 + strlen(argv[i]));
    }
    char * p = (char *)calloc(len, 1);
    len      = 0;
    for (int32_t i = 0; i < argc; i++) {
        /* Clang-tidy is incorrect in thinking this is unsafe. Maybe
         * rewrite later */
        strcpy(p + len, argv[i]);  // NOLINT
        len += strlen(argv[i]);
        p[len++] = ' ';
    }
    p[len - 1] = 0;
    PRINT(HIGH_VERBOSE, "[%s]\n", p);
    commandLine = p;
}

// offset is 1 for option args (argv[0] point32_ts at option, argv[1] at start
// of data) offset is 0 for positional args (argv[0] point32_ts at actual
// argument)
static int32_t
assignArg(ArgOption * desc,
          int32_t     argc,
          char **     argv,
          ArgParser * ap,
          int32_t     offset) {
    switch (desc->type) {
        case Increment:
            PRINT(HIGH_VERBOSE, "incremement %s\n", desc->longarg);
            {
                int32_t * p = (int32_t *)desc->dest;
                (*p)++;
            }
            break;

        case Function:
            // call function with point32_ter to argv.  Expect # of argv
            // consumed as return result.
            PRINT(HIGH_VERBOSE, "calling function %s\n", desc->longarg);
            {
                argOptionParser aop      = (argOptionParser)desc->dest;
                const char *    ret      = (*aop)(argc, argv);
                long int        consumed = (long int)ret;
                return (int32_t)consumed;
            }
            break;

        case String:
            PRINT(HIGH_VERBOSE,
                  "Saving %s to %s\n",
                  argv[offset],
                  desc->longarg);
            {
                char ** p = (char **)desc->dest;
                *p        = argv[offset];
                return 1;
            }
            break;

        case Set:
            PRINT(HIGH_VERBOSE, "Setting %s to 1\n", desc->longarg);
            {
                int32_t * p = (int32_t *)desc->dest;
                *p          = 1;
            }
            break;

        case Toggle:
            PRINT(HIGH_VERBOSE, "toggeling %s\n", desc->longarg);
            {
                int32_t * p = (int32_t *)desc->dest;
                *p          = !(*p);
            }
            break;

        case Double:
            PRINT(HIGH_VERBOSE,
                  "double -> [%s] = %lf\n",
                  desc->longarg,
                  atof(argv[offset]));
            {
                double * p = (double *)desc->dest;
                *p         = atof(argv[offset]);
                return 1;
            }
            break;

        case Integer:
            PRINT(HIGH_VERBOSE,
                  "int32_t -> [%s] = %d\n",
                  desc->longarg,
                  atoi(argv[offset]));
            {
                int32_t * p = (int32_t *)desc->dest;
                *p          = atoi(argv[offset]);
                return 1;
            }
            break;

        case Help:
            usage(pname, ap);
            exit(-1);

        default:
            fprintf(stderr, "NIY: type\n");
            exit(-1);
    }
    return 0;
}

static const char *
kind2str(ArgKind k) {
    switch (k) {
        case KindEnd:
            return "End";
        case KindHelp:
            return "Help";
        case KindPositional:
            return "Positional";
        case KindOption:
            return "Option";
        case KindRest:
            return "Rest";
        default:
            DBG_ASSERT(0, "Unkown Kind\n");
    }
    return "Unknown";
}


// return true if has help
uint32_t
checkArgDef(ArgParser * ap, ArgDefs * def, uint32_t main) {
    // optional/help come before postional before rest
    int32_t     state   = KindOption;
    ArgOption * desc    = def->args;
    int32_t     hashelp = FALSE;

    for (int32_t i = 0; desc[i].kind != KindEnd; i++) {
        if (desc[i].kind > KindEnd) argdie(ap, "Bad kind - no KindEnd?");
        if (desc[i].kind != state) {
            if ((state == KindOption) && (desc[i].kind == KindHelp) &&
                (desc[i].longarg != NULL)) {
                hashelp = TRUE;
                continue;
            }
            if (desc[i].kind < state)
                argdie(ap,
                       "Bad order of arg defs: %s comes before last of %s",
                       kind2str((ArgKind)state),
                       kind2str(desc[i].kind));
            state = desc[i].kind;
        }
        if (((state == KindPositional) || (state == KindRest)) && !main)
            argdie(ap, "positional args but not main");
    }
    return hashelp;
}

static void
checkArgParser(ArgParser * ap) {
    uint32_t hashelp = FALSE;
    for (ArgParserNode * apn = ap->parsers; apn; apn = apn->next) {
        hashelp |= checkArgDef(ap, apn->parser, apn->main);
    }
    if (!hashelp) argdie(ap, "No help string");
}


int32_t
parseArgs(int32_t argc, char ** argv, ArgDefs * def) {
    ArgParserNode n  = { 1, def, NULL };
    ArgParser     ap = { &n, def };
    return parseArguments(&ap, argc, argv);
}

////////////////////////////////////////////////////////////////
// multiple argument parsers

ArgParser *
createArgumentParser(ArgDefs * def) {
    ArgParser * ap      = (ArgParser *)mycalloc(1, sizeof(ArgParser));
    ap->parsers         = (ArgParserNode *)mycalloc(1, sizeof(ArgParserNode));
    ap->parsers->parser = def;
    ap->parsers->main   = 1;
    ap->mainProg        = def;
    return ap;
}

void
freeArgumentParser(ArgParser * ap) {
    ArgParserNode * next;
    ArgParserNode * p;
    for (p = ap->parsers; p; p = next) {
        next = p->next;
        free(p);
    }
    free(ap);
}

void
addArgumentParser(ArgParser * ap, ArgDefs * def, int32_t order) {
    ArgParserNode * p = (ArgParserNode *)mycalloc(1, sizeof(ArgParserNode));
    p->parser         = def;
    p->next           = NULL;
    p->main           = 0;

    if (order > 0) {
        ArgParserNode * nextp;
        for (nextp = ap->parsers; nextp->next; nextp = nextp->next)
            ;
        nextp->next = p;
    }
    else {
        p->next     = ap->parsers;
        ap->parsers = p;
    }
}

int32_t
parseArguments(ArgParser * ap, int32_t argc, char ** argv) {
    // get program name and commandline as a string
    pname = argv[0];
    makeCommandline(argc, argv);
    argv++;
    argc--;

    checkArgParser(ap);

    // process args
    PRINT(HIGH_VERBOSE, "Processing args for %s: %d\n", pname, argc);
    int32_t  i;
    uint32_t optionsPossible = TRUE;
    for (i = 0; (i < argc) && optionsPossible; i++) {
        char * arg = argv[i];
        PRINT(HIGH_VERBOSE, "%d -> [%s]\n", i, arg);
        if (arg[0] == '-') {
            // Handle options
            uint32_t ok       = FALSE;
            uint32_t notfound = TRUE;
            for (ArgParserNode * apn = ap->parsers; notfound && apn;
                 apn                 = apn->next) {
                ArgOption * desc = apn->parser->args;

                for (int32_t j = 0; notfound && (desc[j].kind != KindEnd);
                     j++) {
                    if (strcmp(desc[j].longarg, arg) == 0) {
                        ok       = TRUE;
                        notfound = FALSE;
                        // see if it is special
                        if (desc[j].type == EndOptions) {
                            optionsPossible = FALSE;
                            break;
                        }
                        // process it
                        int32_t consumed =
                            assignArg(desc + j, argc - i, argv + i, ap, 1);
                        i += consumed;
                    }
                }
            }
            if (!ok) argdie(ap, "Do not understand the flag [%s]\n", arg);
        }
        else {
            // No more options
            break;
        }
    }
    // ok, now we handle positional args, we handle them in the order they are
    // declared only the main parser can define positional args
    ArgOption * desc = NULL;
    for (ArgParserNode * apn = ap->parsers; apn; apn = apn->next) {
        if (apn->main) {
            desc = apn->parser->args;
            break;
        }
    }
    DBG_ASSERT(desc != NULL, "No main parser node\n");
    int32_t baseArg = i;
    int32_t baseDestOffset;
    for (baseDestOffset = 0; desc[baseDestOffset].kind != KindEnd;
         baseDestOffset++) {
        if ((desc[baseDestOffset].kind == KindPositional) ||
            (desc[baseDestOffset].kind == KindRest))
            break;
    }
    // base is first positional arg we are passed, j is first descriptor for
    // positional arg
    PRINT(HIGH_VERBOSE,
          "start pos: j=%d %s kind=%d basearg=%d\n",
          baseDestOffset,
          desc[baseDestOffset].longarg,
          desc[baseDestOffset].type,
          baseArg);
    int32_t j = 0; /* positional offset */
    while ((desc[baseDestOffset + j].kind == KindPositional) &&
           ((baseArg + j) < argc)) {
        PRINT(HIGH_VERBOSE,
              "%d: %s\n",
              j,
              baseArg + desc[baseDestOffset + j].longarg);
        int32_t consumed = assignArg(desc + baseDestOffset + j,
                                     argc - (baseArg + j),
                                     argv + baseArg + j,
                                     ap,
                                     0);
        j += consumed;
    }
    // check that we used all the arguments and don't have any extra
    if (desc[baseDestOffset + j].type == (ArgType)KindPositional)
        argdie(ap, "Expected more arguments, only given %d", j);
    else if ((desc[baseDestOffset + j].type == (ArgType)KindEnd) &&
             ((baseArg + j) < argc)) {
        argdie(ap, "Too many arguments, given %d", j);
    }
    // see if we have a variable number of args at end
    if (desc[baseDestOffset + j].type == (ArgType)KindRest)
        argdie(ap, "Haven't implemented Rest args yets");

    // if user defined a post parsing function, call it - main prog called last
    for (ArgParserNode * apn = ap->parsers; apn; apn = apn->next) {
        if ((apn->main != 1) && (apn->parser->doneParsing != NULL)) {
            (*(apn->parser->doneParsing))();
        }
    }
    if (ap->mainProg->doneParsing) {
        (*(ap->mainProg->doneParsing))();
    }

    return 0;
}

void
freeCommandLine(void) {
    free(commandLine);
}
