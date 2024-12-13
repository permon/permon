#pragma once

#include <petscvec.h>
#include "permonsys.h"

PERMON_EXTERN PetscErrorCode VecMergeAndDestroy(MPI_Comm comm, Vec *local, Vec *global);
PERMON_EXTERN PetscErrorCode VecPrintInfo(Vec vec);
PERMON_EXTERN PetscErrorCode VecCreateFromIS(IS is, Vec *vecout);
PERMON_EXTERN PetscErrorCode VecCheckSameLayoutVec(Vec v1,Vec v2);
PERMON_EXTERN PetscErrorCode VecCheckSameLayoutIS(Vec vec,IS is);
PERMON_EXTERN PetscErrorCode VecInvalidate(Vec vec);
PERMON_EXTERN PetscErrorCode VecIsInvalidated(Vec vec,PetscBool *flg);
PERMON_EXTERN PetscErrorCode VecHasValidValues(Vec vec,PetscBool *flg);
PERMON_EXTERN PetscErrorCode VecNestGetMPI(PetscInt N,Vec *vecs[]);
PERMON_EXTERN PetscErrorCode VecNestRestoreMPI(PetscInt N,Vec *vecs[]);
PERMON_EXTERN PetscErrorCode VecGetMPIVector(MPI_Comm comm, PetscInt N, Vec vecs[], Vec *VecOut);
PERMON_EXTERN PetscErrorCode VecRestoreMPIVector(MPI_Comm comm, PetscInt N, Vec vecs[], Vec *VecIn);
PERMON_EXTERN PetscErrorCode VecScaleSkipInf(Vec x,PetscScalar alpha);

PERMON_EXTERN PetscErrorCode ISAdd(IS is,PetscInt value,IS *isnew);
PERMON_EXTERN PetscErrorCode ISCreateFromVec(Vec vec, IS *is);
PERMON_EXTERN PetscErrorCode ISGetVec(IS is, Vec *vec);
PERMON_EXTERN PetscErrorCode ISGetVecBlock(IS is, Vec *vec, PetscInt bs);
