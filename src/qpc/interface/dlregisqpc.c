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
  TRY( PetscClassIdRegister("QP Constraints",&QPC_CLASSID) );
  /* Register Constructors */
  TRY( QPCRegisterAll() );
  /* Process info & summary exclusions */
  TRY( FllopProcessInfoExclusions(QPC_CLASSID, QPC_CLASS_NAME) );
  TRY( PetscRegisterFinalize(QPCFinalizePackage) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCFinalizePackage"
PetscErrorCode QPCFinalizePackage(void)
{
  PetscFunctionBegin;
  TRY( PetscFunctionListDestroy(&QPCList) );
  QPCPackageInitialized = PETSC_FALSE;
  QPCRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
