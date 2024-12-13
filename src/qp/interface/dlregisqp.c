#define PERMONQP_DLL

#include <permon/private/qpimpl.h>

static PetscBool QPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPInitializePackage"
PetscErrorCode QPInitializePackage()
{
  PetscFunctionBegin;
  if (QPPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  QPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("QP Problem",         &QP_CLASSID));
  /* Register Events */
  PetscCall(PetscLogEventRegister("QPTHomogenizeEq",   QP_CLASSID, &QPT_HomogenizeEq));
  PetscCall(PetscLogEventRegister("QPTEnfEqProject",   QP_CLASSID, &QPT_EnforceEqByProjector));
  PetscCall(PetscLogEventRegister("QPTEnfEqPenalty",   QP_CLASSID, &QPT_EnforceEqByPenalty));
  PetscCall(PetscLogEventRegister("QPTDualize",        QP_CLASSID, &QPT_Dualize));
  PetscCall(PetscLogEventRegister("QPTDualize:G",      QP_CLASSID, &QPT_Dualize_AssembleG));
  PetscCall(PetscLogEventRegister("QPTDualize:FactK",  QP_CLASSID, &QPT_Dualize_FactorK));
  PetscCall(PetscLogEventRegister("QPTDualize:Bt",     QP_CLASSID, &QPT_Dualize_PrepareBt));
  PetscCall(PetscLogEventRegister("QPTFetiPrepare",    QP_CLASSID, &QPT_FetiPrepare));
  PetscCall(PetscLogEventRegister("QPTAllInOne",       QP_CLASSID, &QPT_AllInOne));
  PetscCall(PetscLogEventRegister("QPTOrthonormEq",    QP_CLASSID, &QPT_OrthonormalizeEq));
  PetscCall(PetscLogEventRegister("QPTRemoveGluing",   QP_CLASSID, &QPT_RemoveGluingOfDirichletDofs));
  PetscCall(PetscLogEventRegister("QPTSplitBE",        QP_CLASSID, &QPT_SplitBE));
  /* Process info & summary exclusions */
  PetscCall(PermonProcessInfoExclusions(QP_CLASSID, QP_CLASS_NAME));
  PetscFunctionReturn(PETSC_SUCCESS);
}
