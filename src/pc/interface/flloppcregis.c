#include <permonpc.h>

PERMON_EXTERN PetscErrorCode PCCreate_Dual(PC);

#undef __FUNCT__
#define __FUNCT__ "PermonPCRegisterAll"
PetscErrorCode  PermonPCRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(PCRegister(PCDUAL, PCCreate_Dual));
  PetscFunctionReturn(PETSC_SUCCESS);
}
