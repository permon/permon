#pragma once

#include "permonmat.h"

typedef struct _p_QPPF *QPPF;

PERMON_EXTERN PetscClassId QPPF_CLASSID;
#define QPPF_CLASS_NAME "qppf"

PERMON_EXTERN PetscErrorCode QPPFInitializePackage();

PERMON_EXTERN PetscErrorCode QPPFCreate(MPI_Comm comm, QPPF *cp);
PERMON_EXTERN PetscErrorCode QPPFReset(QPPF cp);
PERMON_EXTERN PetscErrorCode QPPFView(QPPF cp, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPPFSetUp(QPPF cp);
PERMON_EXTERN PetscErrorCode QPPFSetFromOptions(QPPF cp);
PERMON_EXTERN PetscErrorCode QPPFDestroy(QPPF *cp);

PERMON_EXTERN PetscErrorCode QPPFGetAlphaTilde(QPPF cp, Vec *alpha_tilde);
PERMON_EXTERN PetscErrorCode QPPFApplyP(QPPF cp, Vec v, Vec Pv);
PERMON_EXTERN PetscErrorCode QPPFApplyQ(QPPF cp, Vec v, Vec Qv);
PERMON_EXTERN PetscErrorCode QPPFApplyHalfQ(QPPF cp, Vec x, Vec y);
PERMON_EXTERN PetscErrorCode QPPFApplyHalfQTranspose(QPPF cp, Vec x, Vec y);
PERMON_EXTERN PetscErrorCode QPPFApplyCP(QPPF cp, Vec x, Vec y);
PERMON_EXTERN PetscErrorCode QPPFApplyGtG(QPPF cp, Vec v, Vec GtGv);

PERMON_EXTERN PetscErrorCode QPPFSetG(QPPF cp, Mat G);
PERMON_EXTERN PetscErrorCode QPPFSetRedundancy(QPPF cp, PetscInt nred);
PERMON_EXTERN PetscErrorCode QPPFSetExplicitInv(QPPF cp, PetscBool explicitInv);

PERMON_EXTERN PetscErrorCode QPPFCreateQ(QPPF cp, Mat *Q);
PERMON_EXTERN PetscErrorCode QPPFCreateP(QPPF cp, Mat *P);
PERMON_EXTERN PetscErrorCode QPPFCreateHalfQ(QPPF cp, Mat *HalfQ);
PERMON_EXTERN PetscErrorCode QPPFCreateGtG(QPPF cp, Mat *newGtG);

PERMON_EXTERN PetscErrorCode QPPFGetG(QPPF cp, Mat *G);
PERMON_EXTERN PetscErrorCode QPPFGetGHasOrthonormalRows(QPPF cp, PetscBool *flg);
PERMON_EXTERN PetscErrorCode QPPFGetGGt(QPPF cp, Mat *GGt);
PERMON_EXTERN PetscErrorCode QPPFGetGGtinv(QPPF cp, Mat *GGtinv);
PERMON_EXTERN PetscErrorCode QPPFGetKSP(QPPF cp, KSP *ksp);
