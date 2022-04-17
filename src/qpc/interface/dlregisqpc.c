#define FLLOPQPC_DLL

#include <permon/private/qpcimpl.h>

static PetscBool QPCPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPCInitializePackage"
PetscErrorCode QPCInitializePackage(void)
{
  PetscFunctionBegin;
  if (QPCPackageInitialized) PetscFunctionReturn(0);
  QPCPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("QP Constraints",&QPC_CLASSID));
  /* Register Constructors */
  PetscCall(QPCRegisterAll());
  /* Process info & summary exclusions */
  PetscCall(FllopProcessInfoExclusions(QPC_CLASSID, QPC_CLASS_NAME));
  PetscCall(PetscRegisterFinalize(QPCFinalizePackage));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCFinalizePackage"
PetscErrorCode QPCFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&QPCList));
  QPCPackageInitialized = PETSC_FALSE;
  QPCRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
