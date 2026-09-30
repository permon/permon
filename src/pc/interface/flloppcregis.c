#include <permonpc.h>

PERMON_EXTERN PetscErrorCode PCCreate_Dual(PC);

PetscErrorCode PermonPCRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(PCRegister(PCDUAL, PCCreate_Dual));
  PetscFunctionReturn(PETSC_SUCCESS);
}
