
#include <permon/private/permonmatimpl.h>

PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);
PETSC_EXTERN PetscErrorCode MatConvert_MPIDense_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat);

#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors_DensePermon"
PetscErrorCode MatGetColumnVectors_DensePermon(Mat A, Vec *cols_new[])
{
  PetscScalar *A_arr,*col_arr;
  PetscInt    i,j,m,N;
  Vec         d,*cols;
  
  PetscFunctionBegin;
  N = A->cmap->N;
  TRY( MatCreateVecs(A, PETSC_IGNORE, &d) );
  TRY( VecDuplicateVecs(d, N, &cols) );
  TRY( VecDestroy(&d) );

  for (i=0; i<N; i++) {
    TRY( VecSet(cols[i],0.0) );
  }
  m = A->rmap->n;
  TRY( MatDenseGetArray(A,&A_arr) );
  col_arr = A_arr;
  for (j=0; j<N; j++) {
    TRY( VecPlaceArray(cols[j],col_arr) );
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
    TRY( VecResetArray((*cols)[j]) );
  }
  TRY( VecDestroyVecs(N,cols) );
  TRY( MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_SeqDensePermon"
PetscErrorCode MatConvertFrom_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  TRY( MatConvert(A,MATSEQDENSE,reuse,newmat) );
  TRY( MatConvert_SeqDense_SeqDensePermon(*newmat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,newmat) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_MPIDensePermon"
PetscErrorCode MatConvertFrom_MPIDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  PetscFunctionBegin;
  TRY( MatConvert(A,MATMPIDENSE,reuse,newmat) );
  TRY( MatConvert_MPIDense_MPIDensePermon(*newmat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,newmat) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqDense_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatConvert_SeqDense_SeqDensePermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    TRY( MatDuplicate(A,MAT_COPY_VALUES,&B) );
  }
  
  TRY( PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSEPERMON) );

  B->ops->convertfrom = MatConvertFrom_SeqDensePermon;
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqdensepermon",MatConvert_SeqDense_SeqDensePermon) );
  
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
    TRY( MatDuplicate(A,MAT_COPY_VALUES,&B) );
  }
  
  TRY( PetscObjectChangeTypeName((PetscObject)B,MATMPIDENSEPERMON) );

  B->ops->convertfrom = MatConvertFrom_MPIDensePermon;
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpidense_mpidensepermon",MatConvert_MPIDense_MPIDensePermon) );
  
  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIDensePermon"
PETSC_EXTERN PetscErrorCode MatCreate_MPIDensePermon(Mat mat)
{
  PetscFunctionBegin;
  TRY( MatSetType(mat,MATMPIDENSE) );
  TRY( MatConvert_MPIDense_MPIDensePermon(mat,MATMPIDENSEPERMON,MAT_INPLACE_MATRIX,&mat) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqDensePermon"
PETSC_EXTERN PetscErrorCode MatCreate_SeqDensePermon(Mat mat)
{
  PetscFunctionBegin;
  TRY( MatSetType(mat,MATSEQDENSE) );
  TRY( MatConvert_SeqDense_SeqDensePermon(mat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,&mat) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateDensePermon"
PetscErrorCode MatCreateDensePermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIDENSEPERMON);CHKERRQ(ierr);
    ierr = MatMPIDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQDENSEPERMON);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
