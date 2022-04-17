
#include <permonksp.h>

PETSC_EXTERN PetscErrorCode KSPCreate_FETI(KSP);
  
#undef __FUNCT__  
#define __FUNCT__ "PermonKSPRegisterAll"
PetscErrorCode PermonKSPRegisterAll()
{
  PetscFunctionBegin;
  CHKERRQ(KSPRegister(KSPFETI, KSPCreate_FETI));
  PetscFunctionReturn(0);
}
