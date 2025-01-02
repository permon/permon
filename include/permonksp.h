#pragma once

#include <petscksp.h>
#include <permonqpfeti.h>

#define KSPFETI "feti"

PETSC_EXTERN PetscErrorCode PermonKSPRegisterAll();

PETSC_EXTERN PetscErrorCode KSPViewBriefInfo(KSP ksp, PetscViewer viewer);

PETSC_EXTERN PetscErrorCode KSPFETISetDirichlet(KSP ksp, IS isDir, QPFetiNumberingType numtype, PetscBool enforce_by_B);
