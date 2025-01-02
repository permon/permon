#define PERMONQPC_DLL

#include <permon/private/qpcimpl.h>

static PetscBool QPCPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPCInitializePackage"
PetscErrorCode QPCInitializePackage(void)
{
  PetscFunctionBegin;
  if (QPCPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  QPCPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("QP Constraints", &QPC_CLASSID));
  /* Register Constructors */
  PetscCall(QPCRegisterAll());
  /* Process info & summary exclusions */
  PetscCall(PermonProcessInfoExclusions(QPC_CLASSID, QPC_CLASS_NAME));
  PetscCall(PetscRegisterFinalize(QPCFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCFinalizePackage"
PetscErrorCode QPCFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&QPCList));
  QPCPackageInitialized = PETSC_FALSE;
  QPCRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
