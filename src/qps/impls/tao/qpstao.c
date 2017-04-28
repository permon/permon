
#include <../src/qps/impls/tao/qpstaoimpl.h>
#include <petsc/private/taoimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPSTaoConverged_Tao"
static PetscErrorCode QPSTaoConverged_Tao(Tao tao,void *ctx)
{
  QPS qps = (QPS) ctx;

  PetscFunctionBegin;
  //TODO sqrt?
  qps->rnorm = tao->residual;
  qps->iteration = tao->niter;
  TRY( (*qps->convergencetest)(qps,qps->solQP,qps->iteration,qps->rnorm,&qps->reason,qps->cnvctx) );

  //TODO quick&dirty
  if (qps->reason > 0) {
    tao->reason = TAO_CONVERGED_USER;
  } else if (qps->reason < 0) {
    tao->reason = TAO_DIVERGED_USER;
  } else {
    tao->reason = TAO_CONTINUE_ITERATING;
  }
  PetscFunctionReturn(0);
}

/*  FormFunctionGradient - Evaluates f(x) and gradient g(x). 
 
    Input Parameters:
.   tao     - the TaoSolver context
.   X       - input vector
.   userCtx - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()
    
    Output Parameters:
.   fcn     - the function value
.   G       - vector containing the newly evaluated gradient
 */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradientQPS"
static PetscErrorCode FormFunctionGradientQPS(Tao tao, Vec X, PetscReal *fcn, Vec G,void *qps_void)
{
  QP          qp;
  QPS         qps = (QPS) qps_void;

  PetscFunctionBegin; 
  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPComputeObjectiveAndGradient(qp,X,G,fcn) );
  PetscFunctionReturn(0);
  
}

/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao     - the TaoSolver context
.  x       - input vector
.  userCtx - optional user-defined context, as set by TaoSetHessianRoutine()

   Output Parameters:
.  A    - Hessian matrix
.  B    - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
 */
#undef __FUNCT__
#define __FUNCT__ "FormHessianQPS"
static PetscErrorCode FormHessianQPS(Tao tao,Vec X,Mat Hptr, Mat Hpc, void *qps_void)
{
  QPS         qps = (QPS) qps_void;
  QP          qp;
  
  PetscFunctionBegin; 
  TRY( QPSGetSolvedQP(qps,&qp) );
  Hptr = qp->A;   
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSTaoGetTao"
PetscErrorCode QPSTaoGetTao(QPS qps,Tao *tao)
{
  PetscBool flg;
  QPS_Tao *qpstao;
  const char* prefix;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(tao,2);
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg) );
  if (!flg) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  if (!qpstao->tao) {
    TRY( QPSGetOptionsPrefix(qps,&prefix) );
    TRY( TaoCreate(PetscObjectComm((PetscObject)qps),&qpstao->tao) );
    TRY( TaoSetOptionsPrefix(qpstao->tao,prefix) );
    TRY( TaoAppendOptionsPrefix(qpstao->tao,"qps_") );
    TRY( PetscLogObjectParent((PetscObject)qps,(PetscObject)qpstao->tao) );
    TRY( PetscObjectIncrementTabLevel((PetscObject)qpstao->tao,(PetscObject)qps,1) );
    TRY( TaoSetType(qpstao->tao,TAOGPCG) );
    TRY( TaoSetTolerances( qpstao->tao, 1e-50, 1e-5, 1e-50) );
  }
  *tao = qpstao->tao;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSTaoSetType"
PetscErrorCode QPSTaoSetType(QPS qps,TaoType type)
{
  PetscBool flg;
  QPS_Tao *qpstao;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg) );
  if (!flg) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  TRY( QPSTaoGetTao(qps,&qpstao->tao) );
  TRY( TaoSetType(qpstao->tao,type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSTaoGetType"
PetscErrorCode QPSTaoGetType(QPS qps, const TaoType *type)
{
  PetscBool flg;
  QPS_Tao *qpstao;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  TRY( PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg) );
  if (!flg) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  TRY( QPSTaoGetTao(qps,&qpstao->tao) );
  TRY( TaoGetType(qpstao->tao,type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetUp_Tao"
PetscErrorCode QPSSetUp_Tao(QPS qps)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  Tao              tao;
  QP               qp;
  Vec              b,x,lb,ub;
  KSP              ksp;
  PC               pc;
  
  PetscFunctionBegin;
  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPGetRhs(qp,&b) );
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( QPGetBox(qp,&lb,&ub) );
  
  TRY( QPSTaoGetTao(qps,&tao) );
  TRY( TaoSetInitialVector(tao,x) );

  /* Set routines for function, gradient and hessian evaluation */
  TRY( TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradientQPS,qps) );
  TRY( TaoSetHessianRoutine(tao,qp->A,qp->A,FormHessianQPS,qps) );

  /* Set Variable bounds */
  if (!lb) {
    TRY( VecDuplicate(x,&lb) );
    TRY( VecSet(lb,PETSC_NINFINITY) );
  }
  if (!ub) {
    TRY( VecDuplicate(x,&ub) );
    TRY( VecSet(ub,PETSC_INFINITY) );
  }
  TRY( TaoSetVariableBounds(tao,lb,ub) );

  TRY( TaoGetKSP(tao,&ksp) );
  TRY( KSPGetPC(ksp,&pc) );
  TRY( PCSetType(pc,PCNONE) );
  
  TRY( TaoSetConvergenceTest(tao,QPSTaoConverged_Tao,qps) );
  TRY( TaoSetTolerances( tao, qps->atol, qps->rtol, PETSC_DEFAULT ) );
  
  /* Check for any tao command line options */
  if (qpstao->setfromoptionscalled) {
    TRY( TaoSetFromOptions(tao) );
  }

  TRY( TaoSetUp(tao) );
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSolve_Tao"
PetscErrorCode QPSSolve_Tao(QPS qps)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  Tao              tao;
  
  PetscFunctionBegin;
  TRY( QPSTaoGetTao(qps,&tao) );
  TRY( TaoSolve(tao) );
  qpstao->ksp_its = qpstao->ksp_its + tao->ksp_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetFromOptions_Tao"
PetscErrorCode QPSSetFromOptions_Tao(PetscOptionItems *PetscOptionsObject,QPS qps)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  
  PetscFunctionBegin;
  qpstao->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSView_Tao"
PetscErrorCode QPSView_Tao(QPS qps, PetscViewer v)
{
  Tao              tao;
  
  PetscFunctionBegin;
  TRY( QPSTaoGetTao(qps,&tao) );
  TRY( TaoView(tao, v) );
  PetscFunctionReturn(0);  
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_Tao"
PetscErrorCode QPSViewConvergence_Tao(QPS qps, PetscViewer v)
{
  PetscBool     iascii;
  const TaoType taotype;
  QPS_Tao       *qpstao = (QPS_Tao*)qps->data;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (iascii) {
    TRY( QPSTaoGetTao(qps,&qpstao->tao) );
    TRY( TaoGetType(qpstao->tao,&taotype) );
    TRY( PetscViewerASCIIPrintf(v, "TaoType: %s\n", taotype) );
    TRY( PetscViewerASCIIPrintf(v, "Number of KSP iterations in last iteration: %d\n", qpstao->tao->ksp_its) );
    TRY( PetscViewerASCIIPrintf(v, "Total number of KSP iterations: %d\n", qpstao->ksp_its) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSReset_Tao"
PetscErrorCode QPSReset_Tao(QPS qps)
{
  QPS_Tao         *qpstao = (QPS_Tao*)qps->data;

  PetscFunctionBegin;
  TRY( TaoDestroy(&qpstao->tao) );
  qpstao->ksp_its = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSDestroy_Tao"
PetscErrorCode QPSDestroy_Tao(QPS qps)
{
  PetscFunctionBegin;
  TRY( QPSDestroyDefault(qps) );
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSIsQPCompatible_Tao"
PetscErrorCode QPSIsQPCompatible_Tao(QPS qps,QP qp,PetscBool *flg)
{
  Mat Beq,Bineq;
  Vec ceq,cineq,lb,ub;
  
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  TRY( QPGetEq(qp,&Beq,&ceq) );
  TRY( QPGetIneq(qp,&Bineq,&cineq) );
  TRY( QPGetBox(qp,&lb,&ub) );
  if (Beq || ceq || Bineq || cineq) {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);   
}

#undef __FUNCT__  
#define __FUNCT__ "QPSCreate_Tao"
FLLOP_EXTERN PetscErrorCode QPSCreate_Tao(QPS qps)
{
  QPS_Tao         *qpstao;
  MPI_Comm        comm;
  
  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)qps,&comm) );
  TRY( PetscNewLog(qps,&qpstao) );
  qps->data                  = (void*)qpstao;
  qpstao->setfromoptionscalled = PETSC_FALSE;
  qpstao->tao                = NULL;
  
  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  qps->ops->setup            = QPSSetUp_Tao;
  qps->ops->solve            = QPSSolve_Tao;
  qps->ops->destroy          = QPSDestroy_Tao;
  qps->ops->reset            = QPSReset_Tao;
  qps->ops->isqpcompatible   = QPSIsQPCompatible_Tao;
  qps->ops->setfromoptions   = QPSSetFromOptions_Tao;
  qps->ops->view             = QPSView_Tao;
  qps->ops->viewconvergence  = QPSViewConvergence_Tao;
  PetscFunctionReturn(0);
}
