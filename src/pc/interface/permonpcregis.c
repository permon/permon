#include <permonpc.h>

PERMON_EXTERN PetscErrorCode PCCreate_Dual(PC);
PERMON_EXTERN PetscErrorCode PCCreate_FreeSet(PC);

#undef __FUNCT__
#define __FUNCT__ "PermonPCRegisterAll"
PetscErrorCode PermonPCRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(PCRegister(PCDUAL, PCCreate_Dual));
  PetscCall(PCRegister(PCFREESET, PCCreate_FreeSet));
  PetscFunctionReturn(PETSC_SUCCESS);
}
