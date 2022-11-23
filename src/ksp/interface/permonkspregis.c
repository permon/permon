
#include <permonksp.h>

PETSC_EXTERN PetscErrorCode KSPCreate_FETI(KSP);
  
PetscErrorCode PermonKSPRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(KSPRegister(KSPFETI, KSPCreate_FETI));
  PetscFunctionReturn(0);
}
