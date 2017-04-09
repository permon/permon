#if !defined(__QPS_TAOIMPL_H)
#define __QPS_TAOIMPL_H
#include <permon/private/qpsimpl.h>
#include <petsctao.h>

typedef struct {
  Tao tao;
  PetscBool setfromoptionscalled;
  PetscInt ksp_its;
} QPS_Tao;

#endif
