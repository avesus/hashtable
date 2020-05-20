#ifndef _CORE_H_
#define _CORE_H_
#include <general-config.h>
#include <helpers/util.h>

// for getting temp of physical core. right now sysfs gurantees core_id
// of a cpu is in core_id file. This also means that don't need to to
// much in way of hyperthreads (i.e track how many logical cores per
// physical core as core_id will map logical -> physical hyperthreads
// or not).

// gets core_id (physical core) for a logical cpu
int32_t
getCoreID(int32_t core);


#endif
