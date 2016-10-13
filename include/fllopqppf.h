#if !defined(__FLLOPQPPF_H)
#define __FLLOPQPPF_H
#include "fllopmat.h"

typedef struct _p_QPPF* QPPF;

FLLOP_EXTERN PetscClassId QPPF_CLASSID;
#define QPPF_CLASS_NAME  "qppf"

FLLOP_EXTERN PetscErrorCode QPPFInitializePackage();

FLLOP_EXTERN PetscErrorCode QPPFCreate(MPI_Comm comm, QPPF* cp);
FLLOP_EXTERN PetscErrorCode QPPFReset(QPPF cp);
FLLOP_EXTERN PetscErrorCode QPPFView(QPPF cp,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPPFSetUp(QPPF cp);
FLLOP_EXTERN PetscErrorCode QPPFSetFromOptions(QPPF cp);
FLLOP_EXTERN PetscErrorCode QPPFDestroy(QPPF *cp);

FLLOP_EXTERN PetscErrorCode QPPFGetAlphaTilde(QPPF cp, Vec *alpha_tilde);
FLLOP_EXTERN PetscErrorCode QPPFApplyP(QPPF cp, Vec v, Vec Pv);
FLLOP_EXTERN PetscErrorCode QPPFApplyQ(QPPF cp, Vec v, Vec Qv);
FLLOP_EXTERN PetscErrorCode QPPFApplyHalfQ(QPPF cp, Vec x, Vec y);
FLLOP_EXTERN PetscErrorCode QPPFApplyHalfQTranspose(QPPF cp, Vec x, Vec y);
FLLOP_EXTERN PetscErrorCode QPPFApplyCP(QPPF cp, Vec x, Vec y);
FLLOP_EXTERN PetscErrorCode QPPFApplyGtG(QPPF cp, Vec v, Vec GtGv);

FLLOP_EXTERN PetscErrorCode QPPFSetG(QPPF cp, Mat G);
FLLOP_EXTERN PetscErrorCode QPPFSetRedundancy(QPPF cp,PetscInt nred);
FLLOP_EXTERN PetscErrorCode QPPFSetExplicitInv(QPPF cp,PetscBool explicitInv);

FLLOP_EXTERN PetscErrorCode QPPFCreateQ(QPPF cp, Mat *Q);
FLLOP_EXTERN PetscErrorCode QPPFCreateP(QPPF cp, Mat *P);
FLLOP_EXTERN PetscErrorCode QPPFCreateHalfQ(QPPF cp, Mat *HalfQ);
FLLOP_EXTERN PetscErrorCode QPPFCreateGtG(QPPF cp, Mat *newGtG);

FLLOP_EXTERN PetscErrorCode QPPFGetG(QPPF cp, Mat *G);
FLLOP_EXTERN PetscErrorCode QPPFGetGHasOrthonormalRows(QPPF cp, PetscBool *flg);
FLLOP_EXTERN PetscErrorCode QPPFGetGGt(QPPF cp, Mat *GGt);
FLLOP_EXTERN PetscErrorCode QPPFGetGGtinv(QPPF cp, Mat *GGtinv);
FLLOP_EXTERN PetscErrorCode QPPFGetKSP(QPPF cp, KSP *ksp);

#endif
