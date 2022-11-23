
#include <permon/private/permonmatimpl.h>
#include <petscblaslapack.h>

PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
PETSC_EXTERN PetscErrorCode MatConvert_MPIDense_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDensePermon(Mat A,Vec xx,Vec yy);
PETSC_INTERN PetscErrorCode MatMult_SeqDensePermon(Mat A,Vec xx,Vec yy);

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
  PetscFunctionReturn(0);
}

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
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertFrom_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(MatConvert(A,MATSEQDENSE,reuse,newmat));
  PetscCall(MatConvert_SeqDense_SeqDensePermon(*newmat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertFrom_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(MatConvert(A,MATMPIDENSE,reuse,newmat));
  PetscCall(MatConvert_MPIDense_MPIDensePermon(*newmat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(0);
}

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
  
  *newmat = B;
  PetscFunctionReturn(0);
}

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
  
  *newmat = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIDensePermon(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(mat,MATMPIDENSE));
  PetscCall(MatConvert_MPIDense_MPIDensePermon(mat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqDensePermon(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(mat,MATSEQDENSE));
  PetscCall(MatConvert_SeqDense_SeqDensePermon(mat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(0);
}

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
  PetscFunctionReturn(0);
}
