#define FLLOPQPPF_DLL

#include <permon/private/qppfimpl.h>

static PetscBool QPPFPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPPFInitializePackage"
PetscErrorCode QPPFInitializePackage()
{
  PetscFunctionBegin;
  if (QPPFPackageInitialized) PetscFunctionReturn(0);
  QPPFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("QP Projector Factory",&QPPF_CLASSID));
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("QPPFSetUp",       QPPF_CLASSID, &QPPF_SetUp));
  CHKERRQ(PetscLogEventRegister("QPPFSetUp:Gt",    QPPF_CLASSID, &QPPF_SetUp_Gt));
  CHKERRQ(PetscLogEventRegister("QPPFSetUp:GGt",   QPPF_CLASSID, &QPPF_SetUp_GGt));
  CHKERRQ(PetscLogEventRegister("QPPFSetUp:GGtinv",QPPF_CLASSID, &QPPF_SetUp_GGtinv));
  CHKERRQ(PetscLogEventRegister("QPPFApplyCP",     QPPF_CLASSID, &QPPF_ApplyCP));
  CHKERRQ(PetscLogEventRegister("QPPFApplyCP:gt",  QPPF_CLASSID, &QPPF_ApplyCP_gt));
  CHKERRQ(PetscLogEventRegister("QPPFApplyCP:sc",  QPPF_CLASSID, &QPPF_ApplyCP_sc));
  CHKERRQ(PetscLogEventRegister("QPPFApplyP",      QPPF_CLASSID, &QPPF_ApplyP));
  CHKERRQ(PetscLogEventRegister("QPPFApplyQ",      QPPF_CLASSID, &QPPF_ApplyQ));
  CHKERRQ(PetscLogEventRegister("QPPFApplyHalfQ",  QPPF_CLASSID, &QPPF_ApplyHalfQ));
  CHKERRQ(PetscLogEventRegister("QPPFApplyG",      QPPF_CLASSID, &QPPF_ApplyG));
  CHKERRQ(PetscLogEventRegister("QPPFApplyGt",     QPPF_CLASSID, &QPPF_ApplyGt));
  /* Process info & summary exclusions */
  CHKERRQ(FllopProcessInfoExclusions(QPPF_CLASSID, QPPF_CLASS_NAME));
  PetscFunctionReturn(0);
}
