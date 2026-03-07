#pragma once

#include <petscpc.h>
#include "permonmat.h"

/* subclasses */
#define PCDUAL    "dual"
#define PCFREESET "freeset"

PERMON_EXTERN PetscErrorCode PermonPCRegisterAll();
PERMON_EXTERN PetscBool      PermonPCRegisterAllCalled;

/* PCDUAL type-specific functions */
typedef enum {
  PC_DUAL_NONE   = 0,
  PC_DUAL_LUMPED = 1
} PCDualType;
PERMON_EXTERN const char    *PCDualTypes[];
PERMON_EXTERN PetscErrorCode PCDualSetType(PC pc, PCDualType type);
PERMON_EXTERN PetscErrorCode PCDualGetType(PC pc, PCDualType *type);

/* PCFREESET type-specific functions */
typedef enum {
  PC_FREESET_BASIC = 0,
  PC_FREESET_CHEAP = 1,
  PC_FREESET_FIXED = 2,
} PCFreeSetType;
PERMON_EXTERN const char    *PCFreeSetTypes[];
PERMON_EXTERN PetscErrorCode PCFreeSetSetType(PC pc, PCFreeSetType type);
PERMON_EXTERN PetscErrorCode PCFreeSetGetType(PC pc, PCFreeSetType *type);
PERMON_EXTERN PetscErrorCode PCFreeSetSetIS(PC pc, IS is);
PERMON_EXTERN PetscErrorCode PCFreeSetGetIS(PC pc, IS *is);
PERMON_EXTERN PetscErrorCode PCFreeSetSetPC(PC pc, PC innerpc);
PERMON_EXTERN PetscErrorCode PCFreeSetGetPC(PC pc, PC *innerpc);
