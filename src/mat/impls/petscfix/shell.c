
#include <private/fllopmatimpl.h>
#include <private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_ShellPermon"
static PetscErrorCode MatMultAdd_ShellPermon(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Shell      *shell = (Mat_Shell*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->right_add_work) {ierr = VecDuplicate(z,&shell->right_add_work);CHKERRQ(ierr);}
    ierr = MatMult(A,x,shell->right_add_work);CHKERRQ(ierr);
    ierr = VecAXPY(shell->right_add_work,1.0,y);CHKERRQ(ierr);
    ierr = VecCopy(shell->right_add_work,z);CHKERRQ(ierr);
  } else {
    ierr = MatMult(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateShellPermon"
PetscErrorCode MatCreateShellPermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscFunctionBegin;
  TRY( MatCreateShell(comm,m,n,M,N,ctx,A) );
  TRY( MatShellSetOperation(*A,MATOP_MULT_ADD,(void(*)())MatMultAdd_ShellPermon) );
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateDummy"
PetscErrorCode MatCreateDummy(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscFunctionBegin;
  TRY( MatCreateShell(comm,m,n,M,N,ctx,A) );
  TRY( PetscObjectChangeTypeName((PetscObject)*A,MATDUMMY) );
  PetscFunctionReturn(0);
}
