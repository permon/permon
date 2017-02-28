#if !defined(__FLLOPMATIMPL_H)
#define	__FLLOPMATIMPL_H
#include <fllopmat.h>
#include <petsc/private/matimpl.h>
#include <private/fllopimpl.h>

typedef struct {
  Mat               A,R;
  KSP               ksp, innerksp;
  PetscInt          redundancy;
  PetscSubcommType  psubcommType;
  MatInvType        type;
  MatRegularizationType regtype;
  PetscBool         setupcalled,setfromoptionscalled,inner_objects_created;
} Mat_Inv;

typedef struct {
	Mat localBlock;	                  /* local (sequential) blocks of BlockDiag */
	Vec xloc, yloc, yloc1;            /* local work vectors */ 
  Vec *cols_loc;
} Mat_BlockDiag;

typedef struct {
  Mat  A;                               /* the wrapped matrix */
  PetscLogEvent events[256];
} Mat_Timer;

struct _n_MatCompleteCtx {
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  PetscErrorCode (*multadd)(Mat,Vec,Vec,Vec);
  PetscErrorCode (*multtransposeadd)(Mat,Vec,Vec,Vec);
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
  Vec d;
};
typedef struct _n_MatCompleteCtx *MatCompleteCtx;

FLLOP_EXTERN PetscLogEvent Mat_OrthColumns,Mat_Inv_Explicitly,Mat_Inv_SetUp;
FLLOP_EXTERN PetscLogEvent Mat_Regularize,Mat_GetColumnVectors,Mat_RestoreColumnVectors,Mat_MatMultByColumns,Mat_TransposeMatMultByColumns;
FLLOP_EXTERN PetscLogEvent Mat_GetMaxEigenvalue,Mat_FilterZeros,Mat_MergeAndDestroy,FllopMat_GetLocalMat;

#endif
