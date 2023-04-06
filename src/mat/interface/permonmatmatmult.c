
#include <permon/private/permonmatimpl.h>

PetscLogEvent Mat_GetColumnVectors, Mat_RestoreColumnVectors, Mat_MatMultByColumns, Mat_TransposeMatMultByColumns;

static inline PetscErrorCode MatMatMultByColumns_MatMult_Private(Mat A, PetscBool A_transpose, Mat B, Mat C);
static inline PetscErrorCode MatMatMultByColumns_MatFilterZeros_Private(Mat *C,PetscBool filter);
static inline PetscErrorCode MatMatMultByColumns_Private(Mat A, PetscBool A_transpose, Mat B, PetscBool filter, Mat *C_new);
static PetscErrorCode MatMatBlockDiagMultByColumns_Private(Mat B, PetscBool B_transpose, Mat R, PetscBool filter, Mat *Gt_new);

//TODO add an argument specifying whether values should be copied back during Restore
#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors_Default"
static PetscErrorCode MatGetColumnVectors_Default(Mat A, Vec *cols_new[])
{
  PetscInt       i,j,nnz,ilo,ihi,N;
  const PetscInt *nzi;
  const PetscScalar *vals;
  PetscScalar **A_cols_arrs;
  Vec         d,*cols;

  PetscFunctionBegin;
  N = A->cmap->N;
  PetscCall(MatCreateVecs(A, PETSC_IGNORE, &d));
  PetscCall(VecDuplicateVecs(d, N, &cols));
  PetscCall(VecDestroy(&d));

  for (j=0; j<N; j++) {
    PetscCall(VecZeroEntries(cols[j]));
  }

  PetscCall(MatGetOwnershipRange(A, &ilo, &ihi));
  PetscCall(VecGetArrays(cols, N, &A_cols_arrs));
  for (i=ilo; i<ihi; i++) {
    PetscCall(MatGetRow(A, i, &nnz, &nzi, &vals));
    for (j=0; j<nnz; j++) {
      A_cols_arrs[nzi[j]][i-ilo] = vals[j];
    }
    PetscCall(MatRestoreRow(A, i, &nnz, &nzi, &vals));
  }
  PetscCall(VecRestoreArrays(cols, N, &A_cols_arrs));

  *cols_new=cols;
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO add an argument specifying whether values should be copied back during Restore
#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_Default"
static PetscErrorCode MatRestoreColumnVectors_Default(Mat A, Vec *cols[])
{
  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(A->cmap->N,cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO add an argument specifying whether values should be copied back during Restore
#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors"
PetscErrorCode MatGetColumnVectors(Mat A, PetscInt *ncols, Vec *cols_new[])
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(Mat,Vec*[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(cols_new,3);
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatGetColVecs",MAT_CLASSID,&Mat_GetColumnVectors));
    registered = PETSC_TRUE;
  }
  if (ncols) *ncols = A->cmap->N;
  if (!A->cmap->N) {
    *cols_new = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatGetColumnVectors_C",&f));
  if (!f) f = MatGetColumnVectors_Default;

  PetscCall(PetscLogEventBegin(Mat_GetColumnVectors,A,0,0,0));
  PetscCall((*f)(A,cols_new));
  PetscCall(PetscLogEventEnd(  Mat_GetColumnVectors,A,0,0,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO add an argument specifying whether values should be copied back during Restore
#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors"
PetscErrorCode MatRestoreColumnVectors(Mat A, PetscInt *ncols, Vec *cols_new[])
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(Mat,Vec*[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(cols_new,3);
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatResColVecs",MAT_CLASSID,&Mat_RestoreColumnVectors));
    registered = PETSC_TRUE;
  }
  if (ncols) *ncols = 0;
  if (!A->cmap->N) {
    *cols_new = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatRestoreColumnVectors_C",&f));
  if (!f) f = MatRestoreColumnVectors_Default;

  PetscCall(PetscLogEventBegin(Mat_RestoreColumnVectors,A,0,0,0));
  PetscCall((*f)(A,cols_new));
  PetscCall(PetscLogEventEnd(  Mat_RestoreColumnVectors,A,0,0,0));
  *cols_new = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns_MatMult_Private"
static inline PetscErrorCode MatMatMultByColumns_MatMult_Private(Mat A, PetscBool A_transpose, Mat B, Mat C)
{
  PetscInt N,N1,j;
  Vec *B_cols,*C_cols;
  PetscErrorCode (*f)(Mat,Vec,Vec);
  
  PetscFunctionBeginI;
  f = A_transpose ? MatMultTranspose : MatMult;
  N = B->cmap->N;
  
  PetscCall(MatGetColumnVectors(B,&N1,&B_cols)); PERMON_ASSERT(N1==N,"N1==N (%d != %d)",N1,N);
  PetscCall(MatGetColumnVectors(C,&N1,&C_cols)); PERMON_ASSERT(N1==N,"N1==N (%d != %d)",N1,N);
  
  for (j=0; j<N; j++) {
    PetscCall(f(A,B_cols[j],C_cols[j]));
  }

  PetscCall(MatRestoreColumnVectors(B,&N1,&B_cols));
  PetscCall(MatRestoreColumnVectors(C,&N1,&C_cols));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns_MatFilterZeros_Private"
static inline PetscErrorCode MatMatMultByColumns_MatFilterZeros_Private(Mat *C,PetscBool filter)
{
  Mat C_new;

  FllopTracedFunctionBegin;
  if (filter) {
    FllopTraceBegin;
    PetscCall(MatFilterZeros(*C,PETSC_MACHINE_EPSILON,&C_new));
    PetscCall(MatDestroy(C));
    *C = C_new;
    PetscFunctionReturnI(PETSC_SUCCESS);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatBlockDiagMultByColumns_Private"
static PetscErrorCode MatMatBlockDiagMultByColumns_Private(Mat B, PetscBool B_transpose, Mat R, PetscBool filter, Mat *Gt_new)
{
  Mat Bt;
  Mat Bt_loc;
  Mat R_loc;
  Mat Gt_loc;
  Mat G_loc,G;
  Vec lambda;
  
  PetscFunctionBeginI;
  /* get diagonal block */
  PetscCall(MatGetDiagonalBlock(R, &R_loc));

  if (B_transpose) {
    /* B_transpose is true => B is in fact B' */
    Bt = B;
  } else {
    /* transpose matrix B */
    PetscCall(PermonMatTranspose(B, MAT_TRANSPOSE_EXPLICIT, &Bt));
  }
  PetscCall(MatNestPermonGetVecs(Bt,&lambda,NULL));

  /* get "local" part of matrix Bt */
  PetscCall(PermonMatGetLocalMat(Bt, &Bt_loc));
  
  /* multiply Bt_loc' * R_loc = Gt_loc */
  PetscCall(MatMatMultByColumns_Private(Bt_loc, PETSC_TRUE, R_loc, PETSC_FALSE, &Gt_loc));

  PetscCall(MatDestroy(&Bt_loc));
  if (!B_transpose) PetscCall(MatDestroy(&Bt));
  
  /* create global distributed G by vertical concatenating local sequential G_loc=Gt_loc' */
  PetscCall(PermonMatTranspose(Gt_loc, MAT_TRANSPOSE_EXPLICIT, &G_loc));
  PetscCall(MatMatMultByColumns_MatFilterZeros_Private(&G_loc,filter));
  PetscCall(MatMergeAndDestroy(PetscObjectComm((PetscObject)B), &G_loc, lambda, &G));

  /* return Gt as implicit transpose of G */
  PetscCall(PermonMatTranspose(G, MAT_TRANSPOSE_CHEAPEST, Gt_new));
  PetscCall(MatDestroy(&G));

  PetscCall(VecDestroy(&lambda));
  PetscCall(MatDestroy(&Gt_loc));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns_Private"
static inline PetscErrorCode MatMatMultByColumns_Private(Mat A, PetscBool A_transpose, Mat B, PetscBool filter, Mat *C_new)
{
  PetscBool flg;

  PetscFunctionBeginI;
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATBLOCKDIAG,&flg));
  if (flg) {
    PetscCall(MatMatBlockDiagMultByColumns_Private(A,A_transpose,B,filter,C_new));
  } else {
    PetscCall(PermonMatCreateDenseProductMatrix(A,A_transpose,B,C_new));
    PetscCall(MatMatMultByColumns_MatMult_Private(A,A_transpose,B,*C_new));
    PetscCall(MatMatMultByColumns_MatFilterZeros_Private(C_new,filter));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns"
PetscErrorCode MatMatMultByColumns(Mat A, Mat B, PetscBool filter, Mat *C_new)
{
  static PetscBool registered = PETSC_FALSE;
  
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidLogicalCollectiveBool(A,filter,3);
  PetscValidPointer(C_new,4);
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatMatMultByCols",MAT_CLASSID,&Mat_MatMultByColumns));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscLogEventBegin(Mat_MatMultByColumns,A,0,0,0));
  PetscCall(MatMatMultByColumns_Private(A,PETSC_FALSE,B,filter,C_new));
  PetscCall(PetscLogEventEnd(  Mat_MatMultByColumns,A,0,0,0));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultByColumns"
PetscErrorCode MatTransposeMatMultByColumns(Mat A, Mat B, PetscBool filter, Mat *C_new)
{
  static PetscBool registered = PETSC_FALSE;
  
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidLogicalCollectiveBool(A,filter,3);
  PetscValidPointer(C_new,4);
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatTrMatMultByCo",MAT_CLASSID,&Mat_TransposeMatMultByColumns));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscLogEventBegin(Mat_TransposeMatMultByColumns,A,0,0,0));
  PetscCall(MatMatMultByColumns_Private(A,PETSC_TRUE,B,filter,C_new));
  PetscCall(PetscLogEventEnd(  Mat_TransposeMatMultByColumns,A,0,0,0));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

