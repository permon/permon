#define PERMONQPS_DLL

#include <permon/private/qpsimpl.h>

static PetscBool QPSPackageInitialized = PETSC_FALSE;

PetscErrorCode QPSInitializePackage(void)
{
  PetscFunctionBegin;
  if (QPSPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  QPSPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("QP Solver", &QPS_CLASSID));
  /* Register Constructors */
  PetscCall(QPSRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("QPSSolve", QPS_CLASSID, &QPS_Solve));
  PetscCall(PetscLogEventRegister("QPSPostSolve", QPS_CLASSID, &QPS_PostSolve));
  /* Process info & summary exclusions */
  PetscCall(PermonProcessInfoExclusions(QPS_CLASSID, QPS_CLASS_NAME));
  PetscCall(PetscRegisterFinalize(QPSFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode QPSFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&QPSList));
  QPSPackageInitialized = PETSC_FALSE;
  QPSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
