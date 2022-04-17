
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
  CHKERRQ((*qps->convergencetest)(qps,&qps->reason));

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
  CHKERRQ(QPSGetSolvedQP(qps,&qp));
  CHKERRQ(QPComputeObjectiveAndGradient(qp,X,G,fcn));
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
  PetscFunctionBegin; 
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg));
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  if (!qpstao->tao) {
    CHKERRQ(QPSGetOptionsPrefix(qps,&prefix));
    CHKERRQ(TaoCreate(PetscObjectComm((PetscObject)qps),&qpstao->tao));
    CHKERRQ(TaoSetOptionsPrefix(qpstao->tao,prefix));
    CHKERRQ(TaoAppendOptionsPrefix(qpstao->tao,"qps_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)qps,(PetscObject)qpstao->tao));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)qpstao->tao,(PetscObject)qps,1));
    CHKERRQ(TaoSetType(qpstao->tao,TAOGPCG));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  CHKERRQ(QPSTaoGetTao(qps,&qpstao->tao));
  CHKERRQ(TaoSetType(qpstao->tao,type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSTaoGetType"
PetscErrorCode QPSTaoGetType(QPS qps,TaoType *type)
{
  PetscBool flg;
  QPS_Tao *qpstao;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)qps,QPSTAO,&flg));
  if (!flg) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_SUP,"This is a QPSTAO specific routine!");
  qpstao = (QPS_Tao*)qps->data;
  CHKERRQ(QPSTaoGetTao(qps,&qpstao->tao));
  CHKERRQ(TaoGetType(qpstao->tao,type));
  PetscFunctionReturn(0);
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
  CHKERRQ(QPSGetSolvedQP(qps,&qp));
  CHKERRQ(QPGetRhs(qp,&b));
  CHKERRQ(QPGetSolutionVector(qp,&x));
  CHKERRQ(QPGetBox(qp,&is,&lb,&ub));
  
  CHKERRQ(QPSTaoGetTao(qps,&tao));
  CHKERRQ(TaoSetSolution(tao,x));

  /* Set routines for function, gradient and hessian evaluation */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradientQPS,qps));
  CHKERRQ(TaoSetHessian(tao,qp->A,qp->A,FormHessianQPS,qps));

  /* Set Variable bounds */
  CHKERRQ(VecDuplicate(x,&lbnew));
  CHKERRQ(VecDuplicate(x,&ubnew));
  CHKERRQ(VecSet(lbnew,PETSC_NINFINITY));
  CHKERRQ(VecSet(ubnew,PETSC_INFINITY));

  if (lb) {
    if (is) {
      CHKERRQ(VecISCopy(lbnew,is,SCATTER_FORWARD,lb));
    } else {
      CHKERRQ(VecCopy(lb,lbnew));
    }
  }

  if (ub) {
    if (is) {
      CHKERRQ(VecISCopy(ubnew,is,SCATTER_FORWARD,ub));
    } else {
      CHKERRQ(VecCopy(ub,ubnew));
    }
  }

  CHKERRQ(TaoSetVariableBounds(tao,lbnew,ubnew));
  CHKERRQ(VecDestroy(&lbnew));
  CHKERRQ(VecDestroy(&ubnew));

  /* set specific stopping criterion for TAO inside QPSTAO */
  CHKERRQ(TaoSetConvergenceTest(tao,QPSTaoConverged_Tao,qps));
  CHKERRQ(TaoSetTolerances( tao, qps->atol, qps->rtol, PETSC_DEFAULT ));

  /* Check for any tao command line options */
  if (qpstao->setfromoptionscalled) {
    CHKERRQ(TaoSetFromOptions(tao));
  }

  CHKERRQ(TaoGetKSP(tao,&ksp));
  if (ksp) {
    /* set KSP defaults after TaoSetFromOptions as it can create new KSP instance */
    const char *prefix;
    CHKERRQ(QPSGetOptionsPrefix(qps,&prefix));
    CHKERRQ(KSPSetOptionsPrefix(ksp,prefix));
    CHKERRQ(KSPAppendOptionsPrefix(ksp,"qps_tao_"));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));
    CHKERRQ(KSPSetFromOptions(ksp));
  }

  CHKERRQ(TaoSetUp(tao));
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSolve_Tao"
PetscErrorCode QPSSolve_Tao(QPS qps)
{
  QPS_Tao          *qpstao = (QPS_Tao*)qps->data;
  Tao              tao;
  PetscInt         its;
  
  PetscFunctionBegin;
  CHKERRQ(QPSTaoGetTao(qps,&tao));
  CHKERRQ(TaoSolve(tao));
  CHKERRQ(TaoGetLinearSolveIterations(tao,&its));
  qpstao->ksp_its += its;
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
  CHKERRQ(QPSTaoGetTao(qps,&tao));
  CHKERRQ(TaoView(tao, v));
  PetscFunctionReturn(0);  
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_Tao"
PetscErrorCode QPSViewConvergence_Tao(QPS qps, PetscViewer v)
{
  PetscBool     iascii;
  TaoType       taotype;
  QPS_Tao       *qpstao = (QPS_Tao*)qps->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(QPSTaoGetTao(qps,&qpstao->tao));
    CHKERRQ(TaoGetType(qpstao->tao,&taotype));
    CHKERRQ(PetscViewerASCIIPrintf(v, "TaoType: %s\n", taotype));
    CHKERRQ(PetscViewerASCIIPrintf(v, "Number of KSP iterations in last iteration: %d\n", qpstao->tao->ksp_its));
    CHKERRQ(PetscViewerASCIIPrintf(v, "Total number of KSP iterations: %d\n", qpstao->ksp_its));
    CHKERRQ(PetscViewerASCIIPrintf(v, "Information about last TAOSolve:\n"));
    CHKERRQ(PetscViewerASCIIPushTab(v));
    CHKERRQ(TaoView(qpstao->tao,v));
    CHKERRQ(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSReset_Tao"
PetscErrorCode QPSReset_Tao(QPS qps)
{
  QPS_Tao         *qpstao = (QPS_Tao*)qps->data;

  PetscFunctionBegin;
  CHKERRQ(TaoDestroy(&qpstao->tao));
  qpstao->ksp_its = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSDestroy_Tao"
PetscErrorCode QPSDestroy_Tao(QPS qps)
{
  PetscFunctionBegin;
  CHKERRQ(QPSDestroyDefault(qps));
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSIsQPCompatible_Tao"
PetscErrorCode QPSIsQPCompatible_Tao(QPS qps,QP qp,PetscBool *flg)
{
  Mat Beq,Bineq;
  Vec ceq,cineq;
  QPC qpc;
  
  PetscFunctionBegin;
  CHKERRQ(QPGetEq(qp,&Beq,&ceq));
  CHKERRQ(QPGetIneq(qp,&Bineq,&cineq));
  CHKERRQ(QPGetQPC(qp,&qpc));
  if (Beq || ceq || Bineq || cineq) {
    *flg = PETSC_FALSE;
  } else {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)qpc,QPCBOX,flg));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)qps,&comm));
  CHKERRQ(PetscNewLog(qps,&qpstao));
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
  PetscFunctionReturn(0);
}
