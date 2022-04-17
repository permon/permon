
#include <permon/private/permonmatimpl.h>
#include <petscblaslapack.h>

PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
PETSC_EXTERN PetscErrorCode MatConvert_MPIDense_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
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
  CHKERRQ(MatCreateVecs(A, PETSC_IGNORE, &d));
  CHKERRQ(VecDuplicateVecs(d, N, &cols));
  CHKERRQ(VecDestroy(&d));

  for (i=0; i<N; i++) {
    CHKERRQ(VecSet(cols[i],0.0));
  }
  m = A->rmap->n;
  CHKERRQ(MatDenseGetArray(A,&A_arr));
  col_arr = A_arr;
  for (j=0; j<N; j++) {
    CHKERRQ(VecPlaceArray(cols[j],col_arr));
    col_arr += m;
  }
  *cols_new=cols;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_DensePermon"
PetscErrorCode MatRestoreColumnVectors_DensePermon(Mat A, Vec *cols[])
{
  PetscInt    j,N;
  
  PetscFunctionBegin;
  N = A->cmap->N;
  for (j=0; j < N; j++) {
    CHKERRQ(VecResetArray((*cols)[j]));
  }
  CHKERRQ(VecDestroyVecs(N,cols));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_SeqDensePermon"
PetscErrorCode MatConvertFrom_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  CHKERRQ(MatConvert(A,MATSEQDENSE,reuse,newmat));
  CHKERRQ(MatConvert_SeqDense_SeqDensePermon(*newmat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_MPIDensePermon"
PetscErrorCode MatConvertFrom_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  CHKERRQ(MatConvert(A,MATMPIDENSE,reuse,newmat));
  CHKERRQ(MatConvert_MPIDense_MPIDensePermon(*newmat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqDense_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }
  
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSEPERMON));

  B->ops->convertfrom = MatConvertFrom_SeqDensePermon;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqdensepermon",MatConvert_SeqDense_SeqDensePermon));
  
  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIDense_MPIDensePermon"
PETSC_EXTERN PetscErrorCode MatConvert_MPIDense_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }
  
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSEPERMON));

  B->ops->convertfrom = MatConvertFrom_MPIDensePermon;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidense_mpidensepermon",MatConvert_MPIDense_MPIDensePermon));
  
  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIDensePermon"
PETSC_EXTERN PetscErrorCode MatCreate_MPIDensePermon(Mat mat)
{
  PetscFunctionBegin;
  CHKERRQ(MatSetType(mat,MATMPIDENSE));
  CHKERRQ(MatConvert_MPIDense_MPIDensePermon(mat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatCreate_SeqDensePermon(Mat mat)
{
  PetscFunctionBegin;
  CHKERRQ(MatSetType(mat,MATSEQDENSE));
  CHKERRQ(MatConvert_SeqDense_SeqDensePermon(mat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateDensePermon"
PetscErrorCode MatCreateDensePermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(comm,A));
  CHKERRQ(MatSetSizes(*A,m,n,M,N));
  CHKERRQ(MPI_Comm_size(comm,&size));
  if (size > 1) {
    CHKERRQ(MatSetType(*A,MATMPIDENSEPERMON));
    CHKERRQ(MatMPIDenseSetPreallocation(*A,data));
  } else {
    CHKERRQ(MatSetType(*A,MATSEQDENSEPERMON));
    CHKERRQ(MatSeqDenseSetPreallocation(*A,data));
  }
  PetscFunctionReturn(0);
}
