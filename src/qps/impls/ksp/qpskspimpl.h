#pragma once

#include <permon/private/qpsimpl.h>

typedef struct {
  KSP ksp;
  PetscBool setfromoptionscalled;
} QPS_KSP;
