
#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_ShellPermon"
static PetscErrorCode MatMultAdd_ShellPermon(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->right_add_work) CHKERRQ(VecDuplicate(z,&shell->right_add_work));
    CHKERRQ(MatMult(A,x,shell->right_add_work));
    CHKERRQ(VecAXPY(shell->right_add_work,1.0,y));
    CHKERRQ(VecCopy(shell->right_add_work,z));
  } else {
    CHKERRQ(MatMult(A,x,z));
    CHKERRQ(VecAXPY(z,1.0,y));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateShellPermon"
PetscErrorCode MatCreateShellPermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreateShell(comm,m,n,M,N,ctx,A));
  CHKERRQ(MatShellSetOperation(*A,MATOP_MULT_ADD,(void(*)())MatMultAdd_ShellPermon));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateDummy"
PetscErrorCode MatCreateDummy(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreateShell(comm,m,n,M,N,ctx,A));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)*A,MATDUMMY));
  PetscFunctionReturn(0);
}
