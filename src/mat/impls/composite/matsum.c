#include <permonmat.h>
#include <../src/mat/impls/composite/permoncompositeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatSumGetMat_Sum"
static PetscErrorCode MatSumGetMat_Sum(Mat A,PetscInt index,Mat *Ai)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink ilink;
  PetscInt          i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ilink  = shell->head;
  for (i=0; i<index; i++) {
    if (ilink) {
      ilink = ilink->next;
    } else {
      break;
    }
  }
  if (!ilink) FLLOP_SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"partial matrix index out of range: %d",i);
  *Ai = ilink->mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSumGetMat"
PetscErrorCode MatSumGetMat(Mat A,PetscInt i,Mat *Ai)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(A,i,2);
  PetscValidPointer(Ai,3);
  PetscUseMethod(A,"MatSumGetMat_Sum_C",(Mat,PetscInt,Mat*),(A,i,Ai));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Sum"
PetscErrorCode MatMult_Sum(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;
  Vec               in,out;

  PetscFunctionBegin;
  if (!next) FLLOP_SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      ierr = VecDuplicate(shell->right,&shell->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(shell->rightwork,shell->right,in);CHKERRQ(ierr);
    in   = shell->rightwork;
  }
  ierr = MatMult(next->mat,in,y);CHKERRQ(ierr);
  while ((next = next->next)) {
    if (next->mat->ops->multadd) {
      ierr = MatMultAdd(next->mat,in,y,y);CHKERRQ(ierr);
    } else {
      if (!next->work) { /* should reuse previous work if the same size */
        ierr = MatCreateVecs(next->mat,NULL,&next->work);CHKERRQ(ierr);
      }
      out = next->work;
      ierr = MatMult(next->mat,in,out);CHKERRQ(ierr);
      ierr = VecAXPY(y,1.0,out);CHKERRQ(ierr);
    }
  }
  if (shell->left) {
    ierr = VecPointwiseMult(y,shell->left,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Sum"
PetscErrorCode MatMultTranspose_Sum(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next = shell->head;
  PetscErrorCode    ierr;
  Vec               in;

  PetscFunctionBegin;
  if (!next) FLLOP_SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      ierr = VecDuplicate(shell->left,&shell->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(shell->leftwork,shell->left,in);CHKERRQ(ierr);
    in   = shell->leftwork;
  }
  ierr = MatMultTranspose(next->mat,in,y);CHKERRQ(ierr);
  while ((next = next->next)) {
    ierr = MatMultTransposeAdd(next->mat,in,y,y);CHKERRQ(ierr);
  }
  if (shell->right) {
    ierr = VecPointwiseMult(y,shell->right,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,shell->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Sum"
PetscErrorCode MatMultAdd_Sum(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Composite *shell = (Mat_Composite *) A->data;

  PetscFunctionBegin;
  if (y != z) {
    TRY( MatMult_Sum(A,x,z) );
    TRY( VecAXPY(z,1.0,y) );
  } else {
    if (!shell->rightwork) {
      TRY( VecDuplicate(z,&shell->rightwork) );
    }
    TRY( MatMult(A,x,shell->rightwork) );
    TRY( VecAXPY(z,1.0,shell->rightwork) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Sum"
PetscErrorCode MatMultTransposeAdd_Sum(Mat A,Vec x,Vec y,Vec z)
{
  PetscFunctionBegin;
  TRY( MatMultTranspose_Sum(A, x, z) );
  TRY( VecAXPY(z, 1.0, y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Sum"
FLLOP_EXTERN PetscErrorCode  MatCreate_Sum(Mat A)
{
  PetscErrorCode ierr;
  PetscErrorCode (*createComposite)(Mat);
  Mat_Composite  *composite;

  PetscFunctionBegin;
  ierr = PetscFunctionListFind(MatList,MATCOMPOSITE,(void(**)(void))&createComposite);CHKERRQ(ierr);
  ierr = createComposite(A);CHKERRQ(ierr);
  composite = (Mat_Composite*)A->data;

  TRY( PetscObjectComposeFunction((PetscObject)A,"MatSumGetMat_Sum_C",MatSumGetMat_Sum) );

  A->ops->mult              = MatMult_Sum;
  A->ops->multtranspose     = MatMultTranspose_Sum;
  A->ops->multadd           = MatMultAdd_Sum;
  A->ops->multtransposeadd  = MatMultTransposeAdd_Sum;

  composite->type           = MAT_COMPOSITE_ADDITIVE;
  composite->head           = NULL;
  composite->tail           = NULL;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSUM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateSum"
/*@C
   MatCreateSum - Creates a matrix as the implicit sum of zero or more matrices.
   This class is an adoption of MATCOMPOSITE type in PETSc.

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  nmat - number of matrices to put in
-  mats - the matrices

   Output Parameter:
.  A - the matrix

   Level: advanced

   Notes:
     Alternative construction
$       MatCreate(comm,&mat);
$       MatSetType(mat,MATSUM);
$       MatCompositeAddMat(mat,mats[0]);
$       ....
$       MatCompositeAddMat(mat,mats[nmat-1]);
$       MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
$       MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);

.seealso: MatDestroy(), MatMult(), MatCompositeAddMat(), MatCompositeMerge(), MatCompositeSetType(), MatCompositeType

@*/
PetscErrorCode  MatCreateSum(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       m,n,M,N,i;

  PetscFunctionBegin;
  if (nmat < 1) FLLOP_SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,3);

  ierr = MatGetLocalSize(mats[0],PETSC_IGNORE,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = MatGetSize(mats[0],PETSC_IGNORE,&N);CHKERRQ(ierr);
  ierr = MatGetSize(mats[nmat-1],&M,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATSUM);CHKERRQ(ierr);
  for (i=0; i<nmat; i++) {
    ierr = MatCompositeAddMat(*mat,mats[i]);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
