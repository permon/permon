#define FLLOPQP_DLL

#include <permon/private/qpimpl.h>

static PetscBool QPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPInitializePackage"
PetscErrorCode QPInitializePackage()
{
  PetscFunctionBegin;
  if (QPPackageInitialized) PetscFunctionReturn(0);
  QPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("QP Problem",         &QP_CLASSID));
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("QPTHomogenizeEq",   QP_CLASSID, &QPT_HomogenizeEq));
  CHKERRQ(PetscLogEventRegister("QPTEnfEqProject",   QP_CLASSID, &QPT_EnforceEqByProjector));
  CHKERRQ(PetscLogEventRegister("QPTEnfEqPenalty",   QP_CLASSID, &QPT_EnforceEqByPenalty));
  CHKERRQ(PetscLogEventRegister("QPTDualize",        QP_CLASSID, &QPT_Dualize));
  CHKERRQ(PetscLogEventRegister("QPTDualize:G",      QP_CLASSID, &QPT_Dualize_AssembleG));
  CHKERRQ(PetscLogEventRegister("QPTDualize:FactK",  QP_CLASSID, &QPT_Dualize_FactorK));
  CHKERRQ(PetscLogEventRegister("QPTDualize:Bt",     QP_CLASSID, &QPT_Dualize_PrepareBt));
  CHKERRQ(PetscLogEventRegister("QPTFetiPrepare",    QP_CLASSID, &QPT_FetiPrepare));
  CHKERRQ(PetscLogEventRegister("QPTAllInOne",       QP_CLASSID, &QPT_AllInOne));
  CHKERRQ(PetscLogEventRegister("QPTOrthonormEq",    QP_CLASSID, &QPT_OrthonormalizeEq));
  CHKERRQ(PetscLogEventRegister("QPTRemoveGluing",   QP_CLASSID, &QPT_RemoveGluingOfDirichletDofs));
  CHKERRQ(PetscLogEventRegister("QPTSplitBE",        QP_CLASSID, &QPT_SplitBE));
  /* Process info & summary exclusions */
  CHKERRQ(FllopProcessInfoExclusions(QP_CLASSID, QP_CLASS_NAME));
  PetscFunctionReturn(0);
}
