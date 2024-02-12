
#include <permonpc.h>

FLLOP_EXTERN PetscErrorCode PCCreate_Dual(PC);
FLLOP_EXTERN PetscErrorCode PCCreate_FreeSet(PC);

#undef __FUNCT__
#define __FUNCT__ "FllopPCRegisterAll"
PetscErrorCode  FllopPCRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(PCRegister(PCDUAL, PCCreate_Dual));
  PetscCall(PCRegister(PCFREESET, PCCreate_FreeSet));
  PetscFunctionReturn(PETSC_SUCCESS);
}
