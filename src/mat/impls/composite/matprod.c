#include <permonmat.h>
#include <../src/mat/impls/composite/permoncompositeimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatProdGetMat"
PetscErrorCode MatProdGetMat(Mat A,PetscInt i,Mat *Ai)
{
  PetscFunctionBegin;
  PetscCall(MatCompositeGetMat(A,i,Ai));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateProd"
/*@C
   MatCreateProd - Creates a matrix as the implicit product of one or more matrices.
   This is a simple wrapper over MATCOMPOSITE type in PETSc.

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
$       MatCreateComposite(comm,nmat,mats,mat);
$       MatCompositeSetType(*mat,MAT_COMPOSITE_MULTIPLICATIVE);

     For the multiplicative form the product is mat[nmat-1]*mat[nmat-2]*....*mat[0]

.seealso: MatCreateComposite(), MatDestroy(), MatMult(), MatCompositeAddMat(),
          MatCompositeMerge(), MatCompositeSetType(), MatCompositeType

@*/
PetscErrorCode  MatCreateProd(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat)
{
  PetscFunctionBegin;
  PetscCall(MatCreateComposite(comm,nmat,mats,mat));
  PetscCall(MatCompositeSetType(*mat,MAT_COMPOSITE_MULTIPLICATIVE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
