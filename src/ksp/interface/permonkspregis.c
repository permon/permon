#include <permonksp.h>

PETSC_EXTERN PetscErrorCode KSPCreate_FETI(KSP);

#undef __FUNCT__
#define __FUNCT__ "PermonKSPRegisterAll"
PetscErrorCode PermonKSPRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(KSPRegister(KSPFETI, KSPCreate_FETI));
  PetscFunctionReturn(PETSC_SUCCESS);
}
