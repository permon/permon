#define FLLOPQPS_DLL

#include <permon/private/qpsimpl.h>

static PetscBool QPSPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPSInitializePackage"
PetscErrorCode QPSInitializePackage(void)
{
  PetscFunctionBegin;
  if (QPSPackageInitialized) PetscFunctionReturn(0);
  QPSPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("QP Solver",&QPS_CLASSID));
  /* Register Constructors */
  CHKERRQ(QPSRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("QPSSolve",QPS_CLASSID,&QPS_Solve));
  CHKERRQ(PetscLogEventRegister("QPSPostSolve",QPS_CLASSID,&QPS_PostSolve));
  /* Process info & summary exclusions */
  CHKERRQ(FllopProcessInfoExclusions(QPS_CLASSID, QPS_CLASS_NAME));
  CHKERRQ(PetscRegisterFinalize(QPSFinalizePackage));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSFinalizePackage"
PetscErrorCode QPSFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&QPSList));
  QPSPackageInitialized = PETSC_FALSE;
  QPSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
