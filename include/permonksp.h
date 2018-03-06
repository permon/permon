#if !defined(__FLLOPKSP_H)
#define	__FLLOPKSP_H
#include <petscksp.h>
#include <permonqpfeti.h>

#define KSPFETI             "feti"
#define KSPDCG              "dcg"



typedef enum {
  DCG_SPACE_HAAR,
  DCG_SPACE_JACKET_HAAR,
  DCG_SPACE_DB4,
  DCG_SPACE_DB8,
  DCG_SPACE_DB16,
  DCG_SPACE_BIORTH22,
  DCG_SPACE_AGGREGATION,
  DCG_SPACE_SLEPC,
  DCG_SPACE_USER
} KSPDCGSpaceType;
PETSC_EXTERN const char *const KSPDCGSpaceTypes[];

PETSC_EXTERN PetscErrorCode PermonKSPRegisterAll();

PETSC_EXTERN PetscErrorCode KSPViewBriefInfo(KSP ksp, PetscViewer viewer);

PETSC_EXTERN PetscErrorCode KSPFETISetDirichlet(KSP ksp,IS isDir,QPFetiNumberingType numtype,PetscBool enforce_by_B);

PETSC_EXTERN PetscErrorCode KSPDCGSetDeflationSpace(KSP ksp,Mat W);

#endif
