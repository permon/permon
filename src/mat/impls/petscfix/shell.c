#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_ShellPermon"
static PetscErrorCode MatMultAdd_ShellPermon(Mat A, Vec x, Vec y, Vec z)
{
  Mat_Shell *shell = (Mat_Shell *)A->data;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->right_add_work) PetscCall(VecDuplicate(z, &shell->right_add_work));
    PetscCall(MatMult(A, x, shell->right_add_work));
    PetscCall(VecAXPY(shell->right_add_work, 1.0, y));
    PetscCall(VecCopy(shell->right_add_work, z));
  } else {
    PetscCall(MatMult(A, x, z));
    PetscCall(VecAXPY(z, 1.0, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateShellPermon"
PetscErrorCode MatCreateShellPermon(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, void *ctx, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreateShell(comm, m, n, M, N, ctx, A));
  PetscCall(MatShellSetOperation(*A, MATOP_MULT_ADD, (void (*)())MatMultAdd_ShellPermon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateDummy"
PetscErrorCode MatCreateDummy(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, void *ctx, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreateShell(comm, m, n, M, N, ctx, A));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*A, MATDUMMY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
