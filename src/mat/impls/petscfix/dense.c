
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

/* TODO: remove in 3.8 */
#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMult_SeqDensePermon_SeqDensePermon"
PetscErrorCode MatMatTransposeMult_SeqDensePermon_SeqDensePermon(Mat A,Mat B,MatReuse scall, PetscReal fill,Mat *C)
{
  PetscInt       m=A->rmap->n,n=B->rmap->n;
  Mat            Cmat;
	Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  PetscBLASInt   bm,bn,bk;
  PetscScalar    _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  if (A->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"A->rmap->n %d != B->rmap->n %d\n",A->rmap->n,B->rmap->n);
	if (scall != MAT_INITIAL_MATRIX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Implemented only for MAT_INITIAL_MATRIX \n");

	TRY( MatCreate(PETSC_COMM_SELF,&Cmat) );
  TRY( MatSetSizes(Cmat,m,n,m,n) );
  TRY( MatSetType(Cmat,MATSEQDENSE) );
  TRY( MatSeqDenseSetPreallocation(Cmat,NULL) );
  Cmat->assembled = PETSC_TRUE;
  Mat_SeqDense   *c = (Mat_SeqDense*)Cmat->data;

  TRY( PetscBLASIntCast(A->rmap->n,&bm) );
  TRY( PetscBLASIntCast(B->rmap->n,&bn) );
  TRY( PetscBLASIntCast(A->cmap->n,&bk) );

  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&bm,&bn,&bk,&_DOne,a->v,&a->lda,b->v,&b->lda,&_DZero,c->v,&c->lda));
	MatConvert_SeqDense_SeqDensePermon(Cmat,MATSEQDENSEPERMON,MAT_INPLACE_MATRIX,&Cmat);
  *C = Cmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqDensePermon"
PetscErrorCode MatMultTranspose_SeqDensePermon(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v   = mat->v,*x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n,_One=1;
  PetscScalar       _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) {
    PetscBLASInt i;
    for (i=0; i<n; i++) y[i] = 0.0;
  } else {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&mat->lda,x,&_One,&_DZero,y,&_One));
    ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqDensePermon"
PetscErrorCode MatMult_SeqDensePermon(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *y,_DOne=1.0,_DZero=0.0;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;
  const PetscScalar *v = mat->v,*x;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) {
    PetscBLASInt i;
    for (i=0; i<m; i++) y[i] = 0.0;
  } else {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DZero,y,&_One));
    ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->rmap->n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
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
  #if PETSC_VERSION_MINOR < 8
    B->ops->mult = MatMult_SeqDensePermon;
    B->ops->multtranspose = MatMultTranspose_SeqDensePermon;
  #endif
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_DensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_DensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqdensepermon",MatConvert_SeqDense_SeqDensePermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatMatTransposeMult_seqdensepermon_seqdensepermon_C",MatMatTransposeMult_SeqDensePermon_SeqDensePermon) );
  
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
