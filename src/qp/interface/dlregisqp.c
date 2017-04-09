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
  TRY( PetscClassIdRegister("QP Problem",         &QP_CLASSID) );
  /* Register Events */
  TRY( PetscLogEventRegister("QPTHomogenizeEq",   QP_CLASSID, &QPT_HomogenizeEq) );
  TRY( PetscLogEventRegister("QPTEnfEqProject",   QP_CLASSID, &QPT_EnforceEqByProjector) );
  TRY( PetscLogEventRegister("QPTEnfEqPenalty",   QP_CLASSID, &QPT_EnforceEqByPenalty) );
  TRY( PetscLogEventRegister("QPTDualize",        QP_CLASSID, &QPT_Dualize) );
  TRY( PetscLogEventRegister("QPTDualize:G",      QP_CLASSID, &QPT_Dualize_AssembleG) );
  TRY( PetscLogEventRegister("QPTDualize:FactK",  QP_CLASSID, &QPT_Dualize_FactorK) );
  TRY( PetscLogEventRegister("QPTDualize:Bt",     QP_CLASSID, &QPT_Dualize_PrepareBt) );
  TRY( PetscLogEventRegister("QPTAllInOne",       QP_CLASSID, &QPT_AllInOne) );
  TRY( PetscLogEventRegister("QPTOrthonormEq",    QP_CLASSID, &QPT_OrthonormalizeEq) );
  TRY( PetscLogEventRegister("QPTSplitBE",        QP_CLASSID, &QPT_SplitBE) );
  /* Process info & summary exclusions */
  TRY( FllopProcessInfoExclusions(QP_CLASSID, QP_CLASS_NAME) );
  PetscFunctionReturn(0);
}
