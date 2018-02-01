#if !defined(__FLLOPKSP_H)
#define	__FLLOPKSP_H
#include <petscksp.h>
#include <permonqpfeti.h>

#define KSPFETI             "feti"
#define KSPDCG              "dcg"

PETSC_EXTERN PetscErrorCode PermonKSPRegisterAll();

PETSC_EXTERN PetscErrorCode KSPViewBriefInfo(KSP ksp, PetscViewer viewer);

PETSC_EXTERN PetscErrorCode KSPFETISetDirichlet(KSP ksp,IS isDir,QPFetiNumberingType numtype,PetscBool enforce_by_B);

PETSC_EXTERN PetscErrorCode KSPDCGSetDeflationSpace(KSP ksp,Mat W);

#endif
