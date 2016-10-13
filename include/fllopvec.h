#if !defined(__FLLOPVEC_H)
#define __FLLOPVEC_H
#include <petscvec.h>
#include "fllopsys.h"

FLLOP_EXTERN PetscErrorCode VecMergeAndDestroy(MPI_Comm comm, Vec *local, Vec *global);
FLLOP_EXTERN PetscErrorCode VecPrintInfo(Vec vec);
FLLOP_EXTERN PetscErrorCode VecCreateFromIS(IS is, Vec *vecout);
FLLOP_EXTERN PetscErrorCode VecCheckSameLayoutVec(Vec v1,Vec v2);
FLLOP_EXTERN PetscErrorCode VecCheckSameLayoutIS(Vec vec,IS is);
FLLOP_EXTERN PetscErrorCode VecInvalidate(Vec vec);
FLLOP_EXTERN PetscErrorCode VecIsInvalidated(Vec vec,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode VecHasValidValues(Vec vec,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode VecNestGetMPI(PetscInt N,Vec *vecs[]);
FLLOP_EXTERN PetscErrorCode VecNestRestoreMPI(PetscInt N,Vec *vecs[]);
FLLOP_EXTERN PetscErrorCode VecGetMPIVector(MPI_Comm comm, PetscInt N, Vec vecs[], Vec *VecOut);
FLLOP_EXTERN PetscErrorCode VecRestoreMPIVector(MPI_Comm comm, PetscInt N, Vec vecs[], Vec *VecIn);
FLLOP_EXTERN PetscErrorCode VecScaleSkipInf(Vec x,PetscScalar alpha);

FLLOP_EXTERN PetscErrorCode ISAdd(IS is,PetscInt value,IS *isnew);
FLLOP_EXTERN PetscErrorCode ISCreateFromVec(Vec vec, IS *is);
FLLOP_EXTERN PetscErrorCode ISGetVec(IS is, Vec *vec);
FLLOP_EXTERN PetscErrorCode ISGetVecBlock(IS is, Vec *vec, PetscInt bs);
FLLOP_EXTERN PetscErrorCode ISViewBlock(IS is,PetscViewer viewer, PetscInt bs);

#endif
