#pragma once

#include <petscpc.h>
#include "permonmat.h"

/* subclasses */
#define PCDUAL "dual"

PERMON_EXTERN PetscErrorCode PermonPCRegisterAll();
PERMON_EXTERN PetscBool      PermonPCRegisterAllCalled;

/* PCDUAL type-specific functions */
typedef enum {
  PC_DUAL_NONE           = 0,
  PC_DUAL_LUMPED         = 1,
  PC_DUAL_DIRICHLET      = 2,
  PC_DUAL_DIRICHLET_DIAG = 3,
  PC_DUAL_LUMPED_FULL    = 4
} PCDualType;
PERMON_EXTERN const char    *PCDualTypes[];
PERMON_EXTERN PetscErrorCode PCDualSetType(PC pc, PCDualType type);
PERMON_EXTERN PetscErrorCode PCDualGetType(PC pc, PCDualType *type);
