#pragma once

#include <permon/private/qpsimpl.h>
#include <petsctao.h>

typedef struct {
  Tao tao;
  PetscBool setfromoptionscalled;
  PetscInt ksp_its;
} QPS_Tao;
