#define FLLOPQPPF_DLL

#include <private/qppfimpl.h>

static PetscBool QPPFPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPPFInitializePackage"
PetscErrorCode QPPFInitializePackage()
{
  PetscFunctionBegin;
  if (QPPFPackageInitialized) PetscFunctionReturn(0);
  QPPFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  TRY( PetscClassIdRegister("QP Projector Factory",&QPPF_CLASSID) );
  /* Register Events */
  TRY( PetscLogEventRegister("QPPFSetUp",       QPPF_CLASSID, &QPPF_SetUp) );
  TRY( PetscLogEventRegister("QPPFSetUp:Gt",    QPPF_CLASSID, &QPPF_SetUp_Gt) );
  TRY( PetscLogEventRegister("QPPFSetUp:GGt",   QPPF_CLASSID, &QPPF_SetUp_GGt) );
  TRY( PetscLogEventRegister("QPPFSetUp:GGtinv",QPPF_CLASSID, &QPPF_SetUp_GGtinv) );
  TRY( PetscLogEventRegister("QPPFApplyCP",     QPPF_CLASSID, &QPPF_ApplyCP) );
  TRY( PetscLogEventRegister("QPPFApplyCP:gt",  QPPF_CLASSID, &QPPF_ApplyCP_gt) );
  TRY( PetscLogEventRegister("QPPFApplyCP:sc",  QPPF_CLASSID, &QPPF_ApplyCP_sc) );
  TRY( PetscLogEventRegister("QPPFApplyP",      QPPF_CLASSID, &QPPF_ApplyP) );
  TRY( PetscLogEventRegister("QPPFApplyQ",      QPPF_CLASSID, &QPPF_ApplyQ) );
  TRY( PetscLogEventRegister("QPPFApplyHalfQ",  QPPF_CLASSID, &QPPF_ApplyHalfQ) );
  TRY( PetscLogEventRegister("QPPFApplyG",      QPPF_CLASSID, &QPPF_ApplyG) );
  TRY( PetscLogEventRegister("QPPFApplyGt",     QPPF_CLASSID, &QPPF_ApplyGt) );
  /* Process info & summary exclusions */
  TRY( FllopProcessInfoExclusions(QPPF_CLASSID, QPPF_CLASS_NAME) );
  PetscFunctionReturn(0);
}
