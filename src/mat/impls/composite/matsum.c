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
  if (!ilink) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"partial matrix index out of range: %d",i);
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
  Vec               in,out;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->right) {
    if (!shell->rightwork) {
      CHKERRQ(VecDuplicate(shell->right,&shell->rightwork));
    }
    CHKERRQ(VecPointwiseMult(shell->rightwork,shell->right,in));
    in   = shell->rightwork;
  }
  CHKERRQ(MatMult(next->mat,in,y));
  while ((next = next->next)) {
    if (next->mat->ops->multadd) {
      CHKERRQ(MatMultAdd(next->mat,in,y,y));
    } else {
      if (!next->work) { /* should reuse previous work if the same size */
        CHKERRQ(MatCreateVecs(next->mat,NULL,&next->work));
      }
      out = next->work;
      CHKERRQ(MatMult(next->mat,in,out));
      CHKERRQ(VecAXPY(y,1.0,out));
    }
  }
  if (shell->left) {
    CHKERRQ(VecPointwiseMult(y,shell->left,y));
  }
  CHKERRQ(VecScale(y,shell->scale));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Sum"
PetscErrorCode MatMultTranspose_Sum(Mat A,Vec x,Vec y)
{
  Mat_Composite     *shell = (Mat_Composite*)A->data;
  Mat_CompositeLink next = shell->head;
  Vec               in;

  PetscFunctionBegin;
  if (!next) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide at least one matrix with MatCompositeAddMat()");
  in = x;
  if (shell->left) {
    if (!shell->leftwork) {
      CHKERRQ(VecDuplicate(shell->left,&shell->leftwork));
    }
    CHKERRQ(VecPointwiseMult(shell->leftwork,shell->left,in));
    in   = shell->leftwork;
  }
  CHKERRQ(MatMultTranspose(next->mat,in,y));
  while ((next = next->next)) {
    CHKERRQ(MatMultTransposeAdd(next->mat,in,y,y));
  }
  if (shell->right) {
    CHKERRQ(VecPointwiseMult(y,shell->right,y));
  }
  CHKERRQ(VecScale(y,shell->scale));
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
  PetscErrorCode (*createComposite)(Mat);
  Mat_Composite  *composite;

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListFind(MatList,MATCOMPOSITE,(void(**)(void))&createComposite));
  CHKERRQ(createComposite(A));
  composite = (Mat_Composite*)A->data;

  TRY( PetscObjectComposeFunction((PetscObject)A,"MatSumGetMat_Sum_C",MatSumGetMat_Sum) );

  A->ops->mult              = MatMult_Sum;
  A->ops->multtranspose     = MatMultTranspose_Sum;
  A->ops->multadd           = MatMultAdd_Sum;
  A->ops->multtransposeadd  = MatMultTransposeAdd_Sum;

  composite->type           = MAT_COMPOSITE_ADDITIVE;
  composite->head           = NULL;
  composite->tail           = NULL;

  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,MATSUM));
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
  PetscInt       m,n,M,N,i;

  PetscFunctionBegin;
  if (nmat < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in at least one matrix");
  PetscValidPointer(mat,3);

  CHKERRQ(MatGetLocalSize(mats[0],PETSC_IGNORE,&n));
  CHKERRQ(MatGetLocalSize(mats[nmat-1],&m,PETSC_IGNORE));
  CHKERRQ(MatGetSize(mats[0],PETSC_IGNORE,&N));
  CHKERRQ(MatGetSize(mats[nmat-1],&M,PETSC_IGNORE));
  CHKERRQ(MatCreate(comm,mat));
  CHKERRQ(MatSetSizes(*mat,m,n,M,N));
  CHKERRQ(MatSetType(*mat,MATSUM));
  for (i=0; i<nmat; i++) {
    CHKERRQ(MatCompositeAddMat(*mat,mats[i]));
  }
  CHKERRQ(MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
