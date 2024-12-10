#include <permon/private/permonmatimpl.h>
#include <petscblaslapack.h>

PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
PETSC_EXTERN PetscErrorCode MatConvert_MPIDense_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
PETSC_EXTERN PetscErrorCode MatDestroy_SeqDensePermon(Mat mat);
PETSC_EXTERN PetscErrorCode MatDestroy_MPIDensePermon(Mat mat);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDensePermon(Mat A,Vec xx,Vec yy);
PETSC_INTERN PetscErrorCode MatMult_SeqDensePermon(Mat A,Vec xx,Vec yy);

#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors_DensePermon"
PetscErrorCode MatGetColumnVectors_DensePermon(Mat A, Vec *cols_new[])
{
  PetscScalar *A_arr,*col_arr;
  PetscInt    i,j,m,N;
  Vec         d,*cols;

  PetscFunctionBegin;
  N = A->cmap->N;
  PetscCall(MatCreateVecs(A, PETSC_IGNORE, &d));
  PetscCall(VecDuplicateVecs(d, N, &cols));
  PetscCall(VecDestroy(&d));

  for (i=0; i<N; i++) {
    PetscCall(VecSet(cols[i],0.0));
  }
  m = A->rmap->n;
  PetscCall(MatDenseGetArray(A,&A_arr));
  col_arr = A_arr;
  for (j=0; j<N; j++) {
    PetscCall(VecPlaceArray(cols[j],col_arr));
    col_arr += m;
  }
  *cols_new=cols;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_DensePermon"
PetscErrorCode MatRestoreColumnVectors_DensePermon(Mat A, Vec *cols[])
{
  PetscInt    j,N;

  PetscFunctionBegin;
  N = A->cmap->N;
  for (j=0; j < N; j++) {
    PetscCall(VecResetArray((*cols)[j]));
  }
  PetscCall(VecDestroyVecs(N,cols));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_SeqDensePermon"
PetscErrorCode MatConvertFrom_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(MatConvert(A,MATSEQDENSE,reuse,newmat));
  PetscCall(MatConvert_SeqDense_SeqDensePermon(*newmat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_MPIDensePermon"
PetscErrorCode MatConvertFrom_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(MatConvert(A,MATMPIDENSE,reuse,newmat));
  PetscCall(MatConvert_MPIDense_MPIDensePermon(*newmat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqDense_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSEPERMON));

  B->ops->convertfrom = MatConvertFrom_SeqDensePermon;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqdensepermon",MatConvert_SeqDense_SeqDensePermon));

  /* Replace MatDestroy, stash the old one */
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDestroy_seqdense",B->ops->destroy));
  B->ops->destroy = MatDestroy_SeqDensePermon;

  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIDense_MPIDensePermon"
PETSC_EXTERN PetscErrorCode MatConvert_MPIDense_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSEPERMON));

  B->ops->convertfrom = MatConvertFrom_MPIDensePermon;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidense_mpidensepermon",MatConvert_MPIDense_MPIDensePermon));

  /* Replace MatDestroy, stash the old one */
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDestroy_mpidense",B->ops->destroy));
  B->ops->destroy = MatDestroy_MPIDensePermon;

  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIDensePermon"
PETSC_EXTERN PetscErrorCode MatCreate_MPIDensePermon(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(mat,MATMPIDENSE));
  PetscCall(MatConvert_MPIDense_MPIDensePermon(mat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatCreate_SeqDensePermon(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(mat,MATSEQDENSE));
  PetscCall(MatConvert_SeqDense_SeqDensePermon(mat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatDestroy_SeqDensePermon(Mat mat)
{
  PetscFunctionBegin;
  PetscUseMethod((PetscObject)mat,"MatDestroy_seqdense",(Mat),(mat));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDestroy_seqdense",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatGetColumnVectors_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatRestoreColumnVectors_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_seqdensepermon",NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIDensePermon"
PETSC_EXTERN PetscErrorCode MatDestroy_MPIDensePermon(Mat mat)
{
  PetscFunctionBegin;
  PetscUseMethod((PetscObject)mat,"MatDestroy_mpidense",(Mat),(mat));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDestroy_mpidense",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatGetColumnVectors_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatRestoreColumnVectors_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpidense_mpidensepermon",NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateDensePermon"
PetscErrorCode MatCreateDensePermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    PetscCall(MatSetType(*A,MATMPIDENSEPERMON));
    PetscCall(MatMPIDenseSetPreallocation(*A,data));
  } else {
    PetscCall(MatSetType(*A,MATSEQDENSEPERMON));
    PetscCall(MatSeqDenseSetPreallocation(*A,data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
