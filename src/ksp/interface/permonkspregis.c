
#include <permonksp.h>

PETSC_EXTERN PetscErrorCode KSPCreate_FETI(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_DCG(KSP);
  
#undef __FUNCT__  
#define __FUNCT__ "PermonKSPRegisterAll"
PetscErrorCode PermonKSPRegisterAll()
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = KSPRegister(KSPFETI, KSPCreate_FETI);CHKERRQ(ierr);
  ierr = KSPRegister(KSPDCG, KSPCreate_DCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
