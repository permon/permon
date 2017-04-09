
#include <permon/private/fllopmatimpl.h>

PetscLogEvent Mat_GetColumnVectors, Mat_RestoreColumnVectors, Mat_MatMultByColumns, Mat_TransposeMatMultByColumns;

PETSC_STATIC_INLINE PetscErrorCode MatMatMultByColumns_MatMult_Private(Mat A, PetscBool A_transpose, Mat B, Mat C);
PETSC_STATIC_INLINE PetscErrorCode MatMatMultByColumns_MatFilterZeros_Private(Mat *C,PetscBool filter);
PETSC_STATIC_INLINE PetscErrorCode MatMatMultByColumns_Private(Mat A, PetscBool A_transpose, Mat B, PetscBool filter, Mat *C_new);
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
  TRY( MatCreateVecs(A, PETSC_IGNORE, &d) );
  TRY( VecDuplicateVecs(d, N, &cols) );
  TRY( VecDestroy(&d) );

  for (j=0; j<N; j++) {
    TRY( VecZeroEntries(cols[j]) );
  }

  TRY( MatGetOwnershipRange(A, &ilo, &ihi) );
  TRY( VecGetArrays(cols, N, &A_cols_arrs) );
  for (i=ilo; i<ihi; i++) {
    TRY( MatGetRow(A, i, &nnz, &nzi, &vals) );
    for (j=0; j<nnz; j++) {
      A_cols_arrs[nzi[j]][i-ilo] = vals[j];
    }
    TRY( MatRestoreRow(A, i, &nnz, &nzi, &vals) );
  }
  TRY( VecRestoreArrays(cols, N, &A_cols_arrs) );

  *cols_new=cols;
  PetscFunctionReturn(0);
}

//TODO add an argument specifying whether values should be copied back during Restore
#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_Default"
static PetscErrorCode MatRestoreColumnVectors_Default(Mat A, Vec *cols[])
{
  PetscFunctionBegin;
  TRY( VecDestroyVecs(A->cmap->N,cols) );
  PetscFunctionReturn(0);
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
    TRY( PetscLogEventRegister("MatGetColVecs",MAT_CLASSID,&Mat_GetColumnVectors) );
    registered = PETSC_TRUE;
  }
  if (ncols) *ncols = A->cmap->N;
  if (!A->cmap->N) {
    *cols_new = NULL;
    PetscFunctionReturn(0);
  }
  TRY( PetscObjectQueryFunction((PetscObject)A,"MatGetColumnVectors_C",&f) );
  if (!f) f = MatGetColumnVectors_Default;

  TRY( PetscLogEventBegin(Mat_GetColumnVectors,A,0,0,0) );
  TRY( (*f)(A,cols_new) );
  TRY( PetscLogEventEnd(  Mat_GetColumnVectors,A,0,0,0) );
  PetscFunctionReturn(0);
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
    TRY( PetscLogEventRegister("MatResColVecs",MAT_CLASSID,&Mat_RestoreColumnVectors) );
    registered = PETSC_TRUE;
  }
  if (ncols) *ncols = 0;
  if (!A->cmap->N) {
    *cols_new = NULL;
    PetscFunctionReturn(0);
  }
  TRY( PetscObjectQueryFunction((PetscObject)A,"MatRestoreColumnVectors_C",&f) );
  if (!f) f = MatRestoreColumnVectors_Default;

  TRY( PetscLogEventBegin(Mat_RestoreColumnVectors,A,0,0,0) );
  TRY( (*f)(A,cols_new) );
  TRY( PetscLogEventEnd(  Mat_RestoreColumnVectors,A,0,0,0) );
  *cols_new = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns_MatMult_Private"
PETSC_STATIC_INLINE PetscErrorCode MatMatMultByColumns_MatMult_Private(Mat A, PetscBool A_transpose, Mat B, Mat C)
{
  PetscInt N,N1,j;
  Vec *B_cols,*C_cols;
  PetscErrorCode (*f)(Mat,Vec,Vec);
  
  PetscFunctionBeginI;
  f = A_transpose ? MatMultTranspose : MatMult;
  N = B->cmap->N;
  
  TRY( MatGetColumnVectors(B,&N1,&B_cols) ); FLLOP_ASSERT2(N1==N,"N1==N (%d != %d)",N1,N);
  TRY( MatGetColumnVectors(C,&N1,&C_cols) ); FLLOP_ASSERT2(N1==N,"N1==N (%d != %d)",N1,N);
  
  for (j=0; j<N; j++) {
    TRY( f(A,B_cols[j],C_cols[j]) );
  }

  TRY( MatRestoreColumnVectors(B,&N1,&B_cols) );
  TRY( MatRestoreColumnVectors(C,&N1,&C_cols) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns_MatFilterZeros_Private"
PETSC_STATIC_INLINE PetscErrorCode MatMatMultByColumns_MatFilterZeros_Private(Mat *C,PetscBool filter)
{
  Mat C_new;

  FllopTracedFunctionBegin;
  if (filter) {
    FllopTraceBegin;
    TRY( MatFilterZeros(*C,PETSC_MACHINE_EPSILON,&C_new) );
    TRY( MatDestroy(C) );
    *C = C_new;
    PetscFunctionReturnI(0);
  }
  PetscFunctionReturn(0);
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
  TRY( MatGetDiagonalBlock(R, &R_loc) );

  if (B_transpose) {
    /* B_transpose is true => B is in fact B' */
    Bt = B;
  } else {
    /* transpose matrix B */
    TRY( FllopMatTranspose(B, MAT_TRANSPOSE_EXPLICIT, &Bt) );
  }
  TRY( MatNestPermonGetVecs(Bt,&lambda,NULL) );

  /* get "local" part of matrix Bt */
  TRY( FllopMatGetLocalMat(Bt, &Bt_loc) );
  
  /* multiply Bt_loc' * R_loc = Gt_loc */
  TRY( MatMatMultByColumns_Private(Bt_loc, PETSC_TRUE, R_loc, PETSC_FALSE, &Gt_loc) );

  TRY( MatDestroy(&Bt_loc) );
  if (!B_transpose) TRY( MatDestroy(&Bt) );
  
  /* create global distributed G by vertical concatenating local sequential G_loc=Gt_loc' */
  TRY( FllopMatTranspose(Gt_loc, MAT_TRANSPOSE_EXPLICIT, &G_loc) );
  TRY( MatMatMultByColumns_MatFilterZeros_Private(&G_loc,filter) );
  TRY( MatMergeAndDestroy(PetscObjectComm((PetscObject)B), &G_loc, lambda, &G) );

  /* return Gt as implicit transpose of G */
  TRY( FllopMatTranspose(G, MAT_TRANSPOSE_CHEAPEST, Gt_new) );
  TRY( MatDestroy(&G) );

  TRY( VecDestroy(&lambda) );
  TRY( MatDestroy(&Gt_loc) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultByColumns_Private"
PETSC_STATIC_INLINE PetscErrorCode MatMatMultByColumns_Private(Mat A, PetscBool A_transpose, Mat B, PetscBool filter, Mat *C_new)
{
  PetscBool flg;

  PetscFunctionBeginI;
  TRY( PetscObjectTypeCompare((PetscObject)B,MATBLOCKDIAG,&flg) );
  if (flg) {
    TRY( MatMatBlockDiagMultByColumns_Private(A,A_transpose,B,filter,C_new) );
  } else {
    TRY( FllopMatCreateDenseProductMatrix(A,A_transpose,B,C_new) );
    TRY( MatMatMultByColumns_MatMult_Private(A,A_transpose,B,*C_new) );
    TRY( MatMatMultByColumns_MatFilterZeros_Private(C_new,filter) );
  }
  PetscFunctionReturnI(0);
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
    TRY( PetscLogEventRegister("MatMatMultByCols",MAT_CLASSID,&Mat_MatMultByColumns) );
    registered = PETSC_TRUE;
  }
  TRY( PetscLogEventBegin(Mat_MatMultByColumns,A,0,0,0) );
  TRY( MatMatMultByColumns_Private(A,PETSC_FALSE,B,filter,C_new) );
  TRY( PetscLogEventEnd(  Mat_MatMultByColumns,A,0,0,0) );
  PetscFunctionReturnI(0);
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
    TRY( PetscLogEventRegister("MatTrMatMultByCo",MAT_CLASSID,&Mat_TransposeMatMultByColumns) );
    registered = PETSC_TRUE;
  }
  TRY( PetscLogEventBegin(Mat_TransposeMatMultByColumns,A,0,0,0) );
  TRY( MatMatMultByColumns_Private(A,PETSC_TRUE,B,filter,C_new) );
  TRY( PetscLogEventEnd(  Mat_TransposeMatMultByColumns,A,0,0,0) );
  PetscFunctionReturnI(0);
}
