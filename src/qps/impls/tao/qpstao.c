
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
  PetscCall((*qps->convergencetest)(qps,&qps->reason));

  //TODO quick&dirty
  if (qps->reason > 0) {
    tao->reason = TAO_CONVERGED_USER;
  } else if (qps->reason < 0) {
    tao->reason = TAO_DIVERGED_USER;
  } else {
    tao->reason = TAO_CONTINUE_ITERATING;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPComputeObjectiveAndGradient(qp,X,G,fcn));
  PetscFunctionReturn(PETSC_SUCCESS);
  
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
  PetscFunctionBegin; 
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(tao,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg));
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  if (!qpstao->tao) {
    PetscCall(QPSGetOptionsPrefix(qps,&prefix));
    PetscCall(TaoCreate(PetscObjectComm((PetscObject)qps),&qpstao->tao));
    PetscCall(TaoSetOptionsPrefix(qpstao->tao,prefix));
    PetscCall(TaoAppendOptionsPrefix(qpstao->tao,"qps_"));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)qpstao->tao,(PetscObject)qps,1));
    PetscCall(TaoSetType(qpstao->tao,TAOGPCG));
  }
  *tao = qpstao->tao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSTaoSetType"
PetscErrorCode QPSTaoSetType(QPS qps,TaoType type)
{
  PetscBool flg;
  QPS_Tao *qpstao;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  PetscCall(QPSTaoGetTao(qps,&qpstao->tao));
  PetscCall(TaoSetType(qpstao->tao,type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSTaoGetType"
PetscErrorCode QPSTaoGetType(QPS qps,TaoType *type)
{
  PetscBool flg;
  QPS_Tao *qpstao;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  PetscCall(QPSTaoGetTao(qps,&qpstao->tao));
  PetscCall(TaoGetType(qpstao->tao,type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetUp_Tao"
PetscErrorCode QPSSetUp_Tao(QPS qps)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  Tao              tao;
  QP               qp;
  Vec              b,x,lb,ub,lbnew,ubnew;
  IS               is;
  KSP              ksp;
  PC               pc;
  
  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetRhs(qp,&b));
  PetscCall(QPGetSolutionVector(qp,&x));
  PetscCall(QPGetBox(qp,&is,&lb,&ub));
  
  PetscCall(QPSTaoGetTao(qps,&tao));
  PetscCall(TaoSetSolution(tao,x));

  /* Set routines for function, gradient and hessian evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradientQPS,qps));
  PetscCall(TaoSetHessian(tao,qp->A,qp->A,FormHessianQPS,qps));

  /* Set Variable bounds */
  PetscCall(VecDuplicate(x,&lbnew));
  PetscCall(VecDuplicate(x,&ubnew));
  PetscCall(VecSet(lbnew,PETSC_NINFINITY));
  PetscCall(VecSet(ubnew,PETSC_INFINITY));

  if (lb) {
    if (is) {
      PetscCall(VecISCopy(lbnew,is,SCATTER_FORWARD,lb));
    } else {
      PetscCall(VecCopy(lb,lbnew));
    }
  }

  if (ub) {
    if (is) {
      PetscCall(VecISCopy(ubnew,is,SCATTER_FORWARD,ub));
    } else {
      PetscCall(VecCopy(ub,ubnew));
    }
  }

  PetscCall(TaoSetVariableBounds(tao,lbnew,ubnew));
  PetscCall(VecDestroy(&lbnew));
  PetscCall(VecDestroy(&ubnew));

  /* set specific stopping criterion for TAO inside QPSTAO */
  PetscCall(TaoSetConvergenceTest(tao,QPSTaoConverged_Tao,qps));
  PetscCall(TaoSetTolerances( tao, qps->atol, qps->rtol, PETSC_DEFAULT ));

  /* Check for any tao command line options */
  if (qpstao->setfromoptionscalled) {
    PetscCall(TaoSetFromOptions(tao));
  }

  PetscCall(TaoGetKSP(tao,&ksp));
  if (ksp) {
    /* set KSP defaults after TaoSetFromOptions as it can create new KSP instance */
    const char *prefix;
    PetscCall(QPSGetOptionsPrefix(qps,&prefix));
    PetscCall(KSPSetOptionsPrefix(ksp,prefix));
    PetscCall(KSPAppendOptionsPrefix(ksp,"qps_tao_"));
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));
    PetscCall(KSPSetFromOptions(ksp));
  }

  PetscCall(TaoSetUp(tao));
  PetscFunctionReturn(PETSC_SUCCESS);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSolve_Tao"
PetscErrorCode QPSSolve_Tao(QPS qps)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  Tao              tao;
  PetscInt         its;
  
  PetscFunctionBegin;
  PetscCall(QPSTaoGetTao(qps,&tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetLinearSolveIterations(tao,&its));
  qpstao->ksp_its += its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetFromOptions_Tao"
PetscErrorCode QPSSetFromOptions_Tao(QPS qps,PetscOptionItems *PetscOptionsObject)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  
  PetscFunctionBegin;
  qpstao->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSView_Tao"
PetscErrorCode QPSView_Tao(QPS qps, PetscViewer v)
{
  Tao              tao;
  
  PetscFunctionBegin;
  PetscCall(QPSTaoGetTao(qps,&tao));
  PetscCall(TaoView(tao, v));
  PetscFunctionReturn(PETSC_SUCCESS);  
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_Tao"
PetscErrorCode QPSViewConvergence_Tao(QPS qps, PetscViewer v)
{
  PetscBool     iascii;
  TaoType       taotype;
  QPS_Tao       *qpstao = (QPS_Tao*)qps->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(QPSTaoGetTao(qps,&qpstao->tao));
    PetscCall(TaoGetType(qpstao->tao,&taotype));
    PetscCall(PetscViewerASCIIPrintf(v, "TaoType: %s\n", taotype));
    PetscCall(PetscViewerASCIIPrintf(v, "Number of KSP iterations in last iteration: %d\n", qpstao->tao->ksp_its));
    PetscCall(PetscViewerASCIIPrintf(v, "Total number of KSP iterations: %d\n", qpstao->ksp_its));
    PetscCall(PetscViewerASCIIPrintf(v, "Information about last TAOSolve:\n"));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(TaoView(qpstao->tao,v));
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSReset_Tao"
PetscErrorCode QPSReset_Tao(QPS qps)
{
  QPS_Tao         *qpstao = (QPS_Tao*)qps->data;

  PetscFunctionBegin;
  PetscCall(TaoDestroy(&qpstao->tao));
  qpstao->ksp_its = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSDestroy_Tao"
PetscErrorCode QPSDestroy_Tao(QPS qps)
{
  PetscFunctionBegin;
  PetscCall(QPSReset_Tao(qps));
  PetscCall(QPSDestroyDefault(qps));
  PetscFunctionReturn(PETSC_SUCCESS);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSIsQPCompatible_Tao"
PetscErrorCode QPSIsQPCompatible_Tao(QPS qps,QP qp,PetscBool *flg)
{
  Mat Beq,Bineq;
  Vec ceq,cineq;
  QPC qpc;
  
  PetscFunctionBegin;
  PetscCall(QPGetEq(qp,&Beq,&ceq));
  PetscCall(QPGetIneq(qp,&Bineq,&cineq));
  PetscCall(QPGetQPC(qp,&qpc));
  if (Beq || ceq || Bineq || cineq) {
    *flg = PETSC_FALSE;
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)qpc,QPCBOX,flg));
  }
  PetscFunctionReturn(PETSC_SUCCESS);   
}

#undef __FUNCT__  
#define __FUNCT__ "QPSCreate_Tao"
FLLOP_EXTERN PetscErrorCode QPSCreate_Tao(QPS qps)
{
  QPS_Tao         *qpstao;
  MPI_Comm        comm;
  
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)qps,&comm));
  PetscCall(PetscNew(&qpstao));
  qps->data                  = (void*)qpstao;
  qpstao->setfromoptionscalled = PETSC_FALSE;
  qpstao->ksp_its            = 0;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}
