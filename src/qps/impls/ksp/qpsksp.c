#include <../src/qps/impls/ksp/qpskspimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPSKSPConverged_KSP"
static PetscErrorCode QPSKSPConverged_KSP(KSP ksp,PetscInt i,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  QPS qps = (QPS) ctx;

  PetscFunctionBegin;
  qps->iteration = i;
  qps->rnorm = rnorm;
  PetscCall((*qps->convergencetest)(qps,reason));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPSynchronize_KSP"
/* synchronize operators and tolerances of the QPS and its underlying KSP */
static PetscErrorCode QPSKSPSynchronize_KSP(QPS qps)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  KSP              ksp = qpsksp->ksp;
  PC               pc_ksp, pc_qp;
  QP               qp;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps, &qp));

  PetscCall(KSPGetPC(ksp, &pc_ksp));
  PetscCall(QPGetPC(qp, &pc_qp));
  if (pc_ksp != pc_qp) {
    PetscCall(KSPSetPC(ksp, pc_qp));
  }

  PetscCall(KSPSetOperators(ksp,qp->A,qp->A));
  PetscCall(KSPSetConvergenceTest(ksp, QPSKSPConverged_KSP, qps, NULL));
  PetscCall(KSPSetTolerances(ksp, qps->rtol, qps->atol, qps->divtol, qps->max_it));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPSetKSP"
PetscErrorCode QPSKSPSetKSP(QPS qps,KSP ksp)
{
  PetscBool flg;
  QPS_KSP *qpsksp;
  const char *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;

  PetscCall(KSPDestroy(&qpsksp->ksp));
  qpsksp->ksp = ksp;
  PetscCall(PetscObjectReference((PetscObject)ksp));

  PetscCall(QPSGetOptionsPrefix(qps,&prefix));
  PetscCall(KSPSetOptionsPrefix(ksp,prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp,"qps_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPGetKSP"
PetscErrorCode QPSKSPGetKSP(QPS qps,KSP *ksp)
{
  PetscBool flg;
  QPS_KSP *qpsksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscAssertPointer(ksp,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  *ksp = qpsksp->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPSetType"
PetscErrorCode QPSKSPSetType(QPS qps,KSPType type)
{
  PetscBool flg;
  QPS_KSP *qpsksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  PetscCall(KSPSetType(qpsksp->ksp,type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPGetType"
PetscErrorCode QPSKSPGetType(QPS qps,KSPType *type)
{
  PetscBool flg;
  QPS_KSP *qpsksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  PetscCall(KSPGetType(qpsksp->ksp,type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetUp_KSP"
PetscErrorCode QPSSetUp_KSP(QPS qps)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  KSP              ksp = qpsksp->ksp;

  PetscFunctionBegin;
  PetscCall(QPSKSPSynchronize_KSP(qps));
  if (qpsksp->setfromoptionscalled) PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolve_KSP"
PetscErrorCode QPSSolve_KSP(QPS qps)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  KSP              ksp = qpsksp->ksp;
  Vec              b,x;
  QP               qp;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPSKSPSynchronize_KSP(qps));
  PetscCall(QPGetRhs(qp,&b));
  PetscCall(QPGetSolutionVector(qp,&x));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp,&qps->iteration));
  PetscCall(KSPGetResidualNorm(   ksp,&qps->rnorm));
  PetscCall(KSPGetConvergedReason(ksp,&qps->reason));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetFromOptions_KSP"
PetscErrorCode QPSSetFromOptions_KSP(QPS qps,PetscOptionItems *PetscOptionsObject)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;

  PetscFunctionBegin;
  qpsksp->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSView_KSP"
PetscErrorCode QPSView_KSP(QPS qps, PetscViewer v)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  KSP              ksp = qpsksp->ksp;

  PetscFunctionBegin;
  PetscCall(KSPView(ksp, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_KSP"
PetscErrorCode QPSViewConvergence_KSP(QPS qps, PetscViewer v)
{
  PetscBool     iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    KSPType ksptype;
    PetscCall(QPSKSPGetType(qps, &ksptype));
    PetscCall(PetscViewerASCIIPrintf(v, "KSPType: %s\n", ksptype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSDestroy_KSP"
PetscErrorCode QPSDestroy_KSP(QPS qps)
{
  QPS_KSP         *qpsksp = (QPS_KSP*)qps->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&qpsksp->ksp));
  PetscCall(QPSDestroyDefault(qps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSIsQPCompatible_KSP"
PetscErrorCode QPSIsQPCompatible_KSP(QPS qps,QP qp,PetscBool *flg)
{
  Mat Beq,Bineq;
  Vec ceq,cineq;
  QPC qpc;

  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  PetscCall(QPGetEq(qp,&Beq,&ceq));
  PetscCall(QPGetIneq(qp,&Bineq,&cineq));
  PetscCall(QPGetQPC(qp,&qpc));
  if (Beq || ceq || Bineq || cineq || qpc)
  {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSCreate_KSP"
FLLOP_EXTERN PetscErrorCode QPSCreate_KSP(QPS qps)
{
  QPS_KSP         *qpsksp;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)qps,&comm));
  PetscCall(PetscNew(&qpsksp));
  qps->data                  = (void*)qpsksp;
  qpsksp->setfromoptionscalled = PETSC_FALSE;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  qps->ops->setup            = QPSSetUp_KSP;
  qps->ops->solve            = QPSSolve_KSP;
  qps->ops->destroy          = QPSDestroy_KSP;
  qps->ops->isqpcompatible   = QPSIsQPCompatible_KSP;
  qps->ops->setfromoptions   = QPSSetFromOptions_KSP;
  qps->ops->view             = QPSView_KSP;
  qps->ops->viewconvergence  = QPSViewConvergence_KSP;

  PetscCall(KSPCreate(comm,&qpsksp->ksp));
  PetscCall(KSPSetOptionsPrefix(qpsksp->ksp,"qps_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)qpsksp->ksp,(PetscObject)qps,1));
  PetscCall(KSPSetType(qpsksp->ksp,KSPCG));
  PetscCall(KSPSetNormType(qpsksp->ksp,KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetInitialGuessNonzero(qpsksp->ksp, PETSC_TRUE));
  {
    PC pc;
    PetscCall(KSPGetPC(qpsksp->ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
