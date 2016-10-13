#if !defined(__QPS_KSPIMPL_H)
#define __QPS_KSPIMPL_H
#include <private/qpsimpl.h>

typedef struct {
  KSP ksp;
  PetscBool setfromoptionscalled;
} QPS_KSP;

#endif
