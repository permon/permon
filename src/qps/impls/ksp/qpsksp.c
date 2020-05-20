
#include <../src/qps/impls/ksp/qpskspimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPSKSPConverged_KSP"
static PetscErrorCode QPSKSPConverged_KSP(KSP ksp,PetscInt i,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  QPS qps = (QPS) ctx;

  PetscFunctionBegin;
  qps->iteration = i;
  qps->rnorm = rnorm;
  TRY( (*qps->convergencetest)(qps,reason) );
  PetscFunctionReturn(0);
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
  TRY( QPSGetSolvedQP(qps, &qp) );

  TRY( KSPGetPC(ksp, &pc_ksp) );
  TRY( QPGetPC(qp, &pc_qp) );
  if (pc_ksp != pc_qp) {
    TRY( KSPSetPC(ksp, pc_qp) );
  }

  TRY( KSPSetOperators(ksp,qp->A,qp->A) );
  TRY( KSPSetConvergenceTest(ksp, QPSKSPConverged_KSP, qps, NULL) );
  TRY( KSPSetTolerances(ksp, qps->rtol, qps->atol, qps->divtol, qps->max_it) );  
  PetscFunctionReturn(0);
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
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg) );
  if (!flg) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  
  TRY( KSPDestroy(&qpsksp->ksp) );
  qpsksp->ksp = ksp;
  TRY( PetscObjectReference((PetscObject)ksp) );
  
  TRY( QPSGetOptionsPrefix(qps,&prefix) );
  TRY( KSPSetOptionsPrefix(ksp,prefix) ); 
  TRY( KSPAppendOptionsPrefix(ksp,"qps_") );  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPGetKSP"
PetscErrorCode QPSKSPGetKSP(QPS qps,KSP *ksp)
{
  PetscBool flg;
  QPS_KSP *qpsksp;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(ksp,2);
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg) );
  if (!flg) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  *ksp = qpsksp->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPSetType"
PetscErrorCode QPSKSPSetType(QPS qps,KSPType type)
{
  PetscBool flg;
  QPS_KSP *qpsksp;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg) );
  if (!flg) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  TRY( KSPSetType(qpsksp->ksp,type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSKSPGetType"
PetscErrorCode QPSKSPGetType(QPS qps,KSPType *type)
{
  PetscBool flg;
  QPS_KSP *qpsksp;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSKSP,&flg) );
  if (!flg) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSKSP specific routine!");
  qpsksp = (QPS_KSP*)qps->data;
  TRY( KSPGetType(qpsksp->ksp,type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetUp_KSP"
PetscErrorCode QPSSetUp_KSP(QPS qps)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  KSP              ksp = qpsksp->ksp;
  
  PetscFunctionBegin;
  TRY( QPSKSPSynchronize_KSP(qps) );
  if (qpsksp->setfromoptionscalled) TRY( KSPSetFromOptions(ksp) );
  TRY( KSPSetUp(ksp) );
  PetscFunctionReturn(0);  
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
  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPSKSPSynchronize_KSP(qps) );
  TRY( QPGetRhs(qp,&b) );
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( KSPSolve(ksp, b, x) );
  TRY( KSPGetIterationNumber(ksp,&qps->iteration) );
  TRY( KSPGetResidualNorm(   ksp,&qps->rnorm) );
  TRY( KSPGetConvergedReason(ksp,&qps->reason) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetFromOptions_KSP"
PetscErrorCode QPSSetFromOptions_KSP(PetscOptionItems *PetscOptionsObject,QPS qps)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  
  PetscFunctionBegin;
  qpsksp->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSView_KSP"
PetscErrorCode QPSView_KSP(QPS qps, PetscViewer v)
{
  QPS_KSP          *qpsksp = (QPS_KSP*)qps->data;
  KSP              ksp = qpsksp->ksp;
  
  PetscFunctionBegin;
  TRY( KSPView(ksp, v) );
  PetscFunctionReturn(0);  
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_KSP"
PetscErrorCode QPSViewConvergence_KSP(QPS qps, PetscViewer v)
{
  PetscBool     iascii;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (iascii) {
    KSPType ksptype;
    TRY( QPSKSPGetType(qps, &ksptype) );
    TRY( PetscViewerASCIIPrintf(v, "KSPType: %s\n", ksptype) );
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "QPSDestroy_KSP"
PetscErrorCode QPSDestroy_KSP(QPS qps)
{
  QPS_KSP         *qpsksp = (QPS_KSP*)qps->data;

  PetscFunctionBegin;
  TRY( KSPDestroy(&qpsksp->ksp) );
  TRY( QPSDestroyDefault(qps) );
  PetscFunctionReturn(0);  
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
  TRY( QPGetEq(qp,&Beq,&ceq) );
  TRY( QPGetIneq(qp,&Bineq,&cineq) );
  TRY( QPGetQPC(qp,&qpc) );
  if (Beq || ceq || Bineq || cineq || qpc)
  {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);   
}

#undef __FUNCT__  
#define __FUNCT__ "QPSCreate_KSP"
FLLOP_EXTERN PetscErrorCode QPSCreate_KSP(QPS qps)
{
  QPS_KSP         *qpsksp;
  MPI_Comm        comm;
  
  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)qps,&comm) );
  TRY( PetscNewLog(qps,&qpsksp) );
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
  
  TRY( KSPCreate(comm,&qpsksp->ksp) );
  TRY( KSPSetOptionsPrefix(qpsksp->ksp,"qps_") );
  TRY( PetscLogObjectParent((PetscObject)qps,(PetscObject)qpsksp->ksp) );
  TRY( PetscObjectIncrementTabLevel((PetscObject)qpsksp->ksp,(PetscObject)qps,1) );
  TRY( KSPSetType(qpsksp->ksp,KSPCG) );
  TRY( KSPSetNormType(qpsksp->ksp,KSP_NORM_UNPRECONDITIONED) );
  TRY( KSPSetInitialGuessNonzero(qpsksp->ksp, PETSC_TRUE) );
  {
    PC pc;
    TRY( KSPGetPC(qpsksp->ksp, &pc) );
    TRY( PCSetType(pc, PCNONE) );
  }
  PetscFunctionReturn(0);
}
