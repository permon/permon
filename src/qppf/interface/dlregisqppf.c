#define PERMONQPPF_DLL

#include <permon/private/qppfimpl.h>

static PetscBool QPPFPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPPFInitializePackage"
PetscErrorCode QPPFInitializePackage()
{
  PetscFunctionBegin;
  if (QPPFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  QPPFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("QP Projector Factory", &QPPF_CLASSID));
  /* Register Events */
  PetscCall(PetscLogEventRegister("QPPFSetUp", QPPF_CLASSID, &QPPF_SetUp));
  PetscCall(PetscLogEventRegister("QPPFSetUp:Gt", QPPF_CLASSID, &QPPF_SetUp_Gt));
  PetscCall(PetscLogEventRegister("QPPFSetUp:GGt", QPPF_CLASSID, &QPPF_SetUp_GGt));
  PetscCall(PetscLogEventRegister("QPPFSetUp:GGtinv", QPPF_CLASSID, &QPPF_SetUp_GGtinv));
  PetscCall(PetscLogEventRegister("QPPFApplyCP", QPPF_CLASSID, &QPPF_ApplyCP));
  PetscCall(PetscLogEventRegister("QPPFApplyCP:gt", QPPF_CLASSID, &QPPF_ApplyCP_gt));
  PetscCall(PetscLogEventRegister("QPPFApplyCP:sc", QPPF_CLASSID, &QPPF_ApplyCP_sc));
  PetscCall(PetscLogEventRegister("QPPFApplyP", QPPF_CLASSID, &QPPF_ApplyP));
  PetscCall(PetscLogEventRegister("QPPFApplyQ", QPPF_CLASSID, &QPPF_ApplyQ));
  PetscCall(PetscLogEventRegister("QPPFApplyHalfQ", QPPF_CLASSID, &QPPF_ApplyHalfQ));
  PetscCall(PetscLogEventRegister("QPPFApplyG", QPPF_CLASSID, &QPPF_ApplyG));
  PetscCall(PetscLogEventRegister("QPPFApplyGt", QPPF_CLASSID, &QPPF_ApplyGt));
  /* Process info & summary exclusions */
  PetscCall(PermonProcessInfoExclusions(QPPF_CLASSID, QPPF_CLASS_NAME));
  PetscFunctionReturn(PETSC_SUCCESS);
}
