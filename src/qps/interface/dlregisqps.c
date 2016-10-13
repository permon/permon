#define FLLOPQPS_DLL

#include <private/qpsimpl.h>

static PetscBool QPSPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPSInitializePackage"
PetscErrorCode QPSInitializePackage(void)
{
  PetscFunctionBegin;
  if (QPSPackageInitialized) PetscFunctionReturn(0);
  QPSPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  TRY( PetscClassIdRegister("QP Solver",&QPS_CLASSID) );
  /* Register Constructors */
  TRY( QPSRegisterAll() );
  /* Register Events */
  TRY( PetscLogEventRegister("QPSSolve",QPS_CLASSID,&QPS_Solve) );
  TRY( PetscLogEventRegister("QPSPostSolve",QPS_CLASSID,&QPS_PostSolve) );
  /* Process info & summary exclusions */
  TRY( FllopProcessInfoExclusions(QPS_CLASSID, QPS_CLASS_NAME) );
  TRY( PetscRegisterFinalize(QPSFinalizePackage) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSFinalizePackage"
PetscErrorCode QPSFinalizePackage(void)
{
  PetscFunctionBegin;
  TRY( PetscFunctionListDestroy(&QPSList) );
  QPSPackageInitialized = PETSC_FALSE;
  QPSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
