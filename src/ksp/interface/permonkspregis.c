
#include <permonksp.h>

PETSC_EXTERN PetscErrorCode KSPCreate_FETI(KSP);
  
#undef __FUNCT__  
#define __FUNCT__ "PermonKSPRegisterAll"
PetscErrorCode PermonKSPRegisterAll()
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = KSPRegister(KSPFETI, KSPCreate_FETI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
