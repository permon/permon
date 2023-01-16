
#include <permonpc.h>

FLLOP_EXTERN PetscErrorCode PCCreate_Dual(PC);
  
PetscErrorCode  FllopPCRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(PCRegister(PCDUAL, PCCreate_Dual));
  PetscFunctionReturn(0);
}
