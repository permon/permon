#include <permon/private/qpimpl.h>
#include <permonpc.h>
#include <permon/private/qppfimpl.h>
#include <permonqpfeti.h>

PetscLogEvent QPT_HomogenizeEq, QPT_EnforceEqByProjector, QPT_EnforceEqByPenalty, QPT_OrthonormalizeEq, QPT_SplitBE;
PetscLogEvent QPT_Dualize, QPT_Dualize_AssembleG, QPT_Dualize_FactorK, QPT_Dualize_PrepareBt, QPT_FetiPrepare, QPT_AllInOne, QPT_RemoveGluingOfDirichletDofs;

static QPPF QPReusedCP = NULL;

/* common tasks during a QP transform - should be called in the beginning of each transform function */
#undef __FUNCT__
#define __FUNCT__ "QPTransformBegin_Private"
static PetscErrorCode QPTransformBegin_Private(PetscErrorCode(*transform)(QP), const char *trname,
    PetscErrorCode(*postSolve)(QP,QP), PetscErrorCode(*postSolveCtxDestroy)(void*),
    QPDuplicateOption opt, QP *qp_inout, QP *child_new, MPI_Comm *comm)
{
  QP child;
  QP qp = *qp_inout;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)qp,comm));
  PetscCall(QPChainGetLast(qp,&qp));
  PetscCall(QPSetUpInnerObjects(qp));
  PetscCall(QPChainAdd(qp,opt,&child));
  child->transform = transform;
  PetscCall(PetscStrcpy(child->transform_name, trname));
  PetscCall(QPSetPC(child,qp->pc));
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));

  child->postSolve = QPDefaultPostSolve;
  if (postSolve) child->postSolve = postSolve;
  child->postSolveCtxDestroy = postSolveCtxDestroy;

  PetscCall(FllopPetscObjectInheritPrefix((PetscObject)child,(PetscObject)qp,NULL));

  *qp_inout = qp;
  *child_new = child;
  PetscCall(PetscInfo(qp,"QP %p (#%d) transformed by %s to QP %p (#%d)\n",(void*)qp,qp->id,trname,(void*)child,child->id));
  PetscFunctionReturn(0);
}

#define QPTransformBegin(transform,postSolve,postSolveCtxDestroy,opt,qp,child,comm) QPTransformBegin_Private((PetscErrorCode(*)(QP))transform,__FUNCT__,\
    (PetscErrorCode(*)(QP,QP))postSolve, (PetscErrorCode(*)(void*))postSolveCtxDestroy,\
    opt,qp,child,comm)

#undef __FUNCT__
#define __FUNCT__ "QPDefaultPostSolve"
PetscErrorCode QPDefaultPostSolve(QP child,QP parent)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(child,QP_CLASSID,1);
  PetscValidHeaderSpecific(parent,QP_CLASSID,2);
  PetscCall(VecCopy(child->x,parent->x));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTEnforceEqByProjectorPostSolve_Private"
static PetscErrorCode QPTEnforceEqByProjectorPostSolve_Private(QP child,QP parent)
{
  Mat A = parent->A;
  Vec b = parent->b;
  Vec x = parent->x;
  Vec lambda_E  = parent->lambda_E;
  Vec Bt_lambda = parent->Bt_lambda;
  Vec r = parent->xwork;
  PetscBool skip_lambda_E=PETSC_TRUE, skip_Bt_lambda=PETSC_TRUE;
  PetscBool inherit_eq_multipliers=PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(QPDefaultPostSolve(child,parent));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_project_inherit_eq_multipliers",&inherit_eq_multipliers,NULL));

  if (child->BE && inherit_eq_multipliers) {
    PetscCall(VecIsInvalidated(child->lambda_E,&skip_lambda_E));
    PetscCall(VecIsInvalidated(child->Bt_lambda,&skip_Bt_lambda));
  }

  PetscCall(MatMult(A, x, r));                                                      /* r = A*x */
  PetscCall(VecAYPX(r,-1.0,b));                                                     /* r = b - r */

  if (!skip_lambda_E) {
    /* lambda_E1 = lambda_E2 + (B*B')\B*(b-A*x) */
    PetscCall(QPPFApplyHalfQ(parent->pf,r,lambda_E));
    PetscCall(VecAXPY(lambda_E,1.0,child->lambda_E));
  }
  if (!skip_Bt_lambda) {
    /* (B'*lambda)_1 = (B'*lambda)_2 + B'*(B*B')\B*(b-A*x) */
    PetscCall(QPPFApplyQ(parent->pf,r,Bt_lambda) );
    PetscCall(VecAXPY(Bt_lambda,1.0,child->Bt_lambda));
  }
  PetscFunctionReturn(0);
}

typedef struct {
  Mat  P;
  PC   pc;
  Vec  work;
  PetscBool symmetric;
} PC_QPTEnforceEqByProjector;

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_QPTEnforceEqByProjector"
static PetscErrorCode PCDestroy_QPTEnforceEqByProjector(PC pc)
{
  PC_QPTEnforceEqByProjector* ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  PetscCall(MatDestroy(&ctx->P));
  PetscCall(PCDestroy(&ctx->pc));
  PetscCall(VecDestroy(&ctx->work));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_QPTEnforceEqByProjector"
static PetscErrorCode PCApply_QPTEnforceEqByProjector(PC pc,Vec x,Vec y)
{
  PC_QPTEnforceEqByProjector* ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  PetscCall(PCApply(ctx->pc,x,ctx->work));
  PetscCall(MatMult(ctx->P,ctx->work,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_QPTEnforceEqByProjector_Symmetric"
static PetscErrorCode PCApply_QPTEnforceEqByProjector_Symmetric(PC pc,Vec x,Vec y)
{
  PC_QPTEnforceEqByProjector* ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  PetscCall(MatMult(ctx->P,x,y));
  PetscCall(PCApply(ctx->pc,y,ctx->work));
  PetscCall(MatMult(ctx->P,ctx->work,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_QPTEnforceEqByProjector_None"
static PetscErrorCode PCApply_QPTEnforceEqByProjector_None(PC pc,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x,y));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetUp_QPTEnforceEqByProjector"
static PetscErrorCode PCSetUp_QPTEnforceEqByProjector(PC pc)
{
  PC_QPTEnforceEqByProjector* ctx;
  PetscBool flg;
  PCDualType type;
  PetscBool none;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));

  none = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->pc,PCNONE,&none));
  if (!none) {
    PetscCall(PetscObjectTypeCompare((PetscObject)ctx->pc,PCDUAL,&flg));
    if (flg) {
      PetscCall(PCDualGetType(ctx->pc,&type));
      if (type == PC_DUAL_NONE) none = PETSC_TRUE;
    }
  }

  if (none) {
    PetscCall(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector_None));
  } else if (ctx->symmetric) {
    PetscCall(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector_Symmetric));
  } else {
    PetscCall(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCreate_QPTEnforceEqByProjector"
PetscErrorCode PCCreate_QPTEnforceEqByProjector(PC pc_orig,Mat P,PetscBool symmetric,PC *pc_new)
{
  PC_QPTEnforceEqByProjector *ctx;
  PC pc;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->symmetric = symmetric;
  ctx->P = P;
  ctx->pc = pc_orig;
  PetscCall(PetscObjectReference((PetscObject)P));
  PetscCall(PetscObjectReference((PetscObject)pc_orig));
  PetscCall(MatCreateVecs(P,&ctx->work,NULL));

  PetscCall(PCCreate(PetscObjectComm((PetscObject)pc_orig),&pc));
  PetscCall(PCSetType(pc,PCSHELL));
  PetscCall(PCShellSetName(pc,"QPTEnforceEqByProjector"));
  PetscCall(PCShellSetContext(pc,ctx));
  PetscCall(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector));
  PetscCall(PCShellSetDestroy(pc,PCDestroy_QPTEnforceEqByProjector));
  PetscCall(PCShellSetSetUp(pc,PCSetUp_QPTEnforceEqByProjector));

  *pc_new = pc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTEnforceEqByProjector"
PetscErrorCode QPTEnforceEqByProjector(QP qp)
{
  MPI_Comm         comm;
  Mat              P=NULL;
  Mat              newA=NULL, A_arr[3];
  Vec              newb=NULL;
  QP               child;
  PetscBool        eqonly=PETSC_FALSE;
  PetscBool        pc_symmetric=PETSC_FALSE;
  PetscBool        inherit_box_multipliers=PETSC_FALSE;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  PetscCall(QPChainGetLast(qp,&qp));
  if (!qp->BE) {
    PetscCall(PetscInfo(qp, "no lin. eq. con. matrix specified ==> nothing to enforce\n"));
    PetscCall(VecDestroy(&qp->cE));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  if (qp->cE) {
    PetscCall(PetscInfo(qp, "nonzero lin. eq. con. RHS prescribed ==> automatically calling QPTHomogenizeEq\n"));
    PetscCall(QPTHomogenizeEq(qp));
    PetscCall(QPChainGetLast(qp,&qp));
  }

  PetscCall(PetscLogEventBegin(QPT_EnforceEqByProjector,qp,0,0,0));
  PetscCall(QPTransformBegin(QPTEnforceEqByProjector, QPTEnforceEqByProjectorPostSolve_Private,NULL, QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  PetscCall(QPAppendOptionsPrefix(child,"proj_"));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_project_pc_symmetric",&pc_symmetric,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_project_inherit_box_multipliers",&inherit_box_multipliers,NULL));

  eqonly = PetscNot(qp->BI || qp->qpc);
  if (eqonly) {
    PetscCall(PetscInfo(qp, "only lin. eq. con. were prescribed ==> they are now eliminated\n"));
    PetscCall(QPSetEq(  child, NULL, NULL));
  } else {
    PetscCall(PetscInfo(qp, "NOT only lin. eq. con. prescribed ==> lin. eq. con. are NOT eliminated\n"));
    PetscCall(QPSetQPPF(child, qp->pf));
    PetscCall(QPSetEq(  child, qp->BE, qp->cE));
  }
  PetscCall(QPSetIneq(child, qp->BI, qp->cI));
  PetscCall(QPSetRhs( child, qp->b));

  if (inherit_box_multipliers) {
    PetscCall(QPSetQPC(child, qp->qpc));
  } else {
    /* TODO: generalize for other QPC */
    Vec lb,ub;
    IS is;
    PetscCall(QPGetBox(qp,&is,&lb,&ub));
    PetscCall(QPSetBox(child,is,lb,ub));
  }

  PetscCall(QPPFCreateP(qp->pf,&P));
  if (eqonly) {
    /* newA = P*A */
    A_arr[0]=qp->A; A_arr[1]=P;
    PetscCall(MatCreateProd(comm,2,A_arr,&newA));
  } else {
    /* newA = P*A*P */
    A_arr[0]=P; A_arr[1]=qp->A; A_arr[2]=P;
    PetscCall(MatCreateProd(comm,3,A_arr,&newA));
  }
  PetscCall(QPSetOperator(child,newA));
  PetscCall(QPSetOperatorNullSpace(child,qp->R));
  PetscCall(MatDestroy(&newA));

  /* newb = P*b */
  PetscCall(VecDuplicate(qp->b, &newb));
  PetscCall(MatMult(P, qp->b, newb));
  if (FllopDebugEnabled) {
    PetscReal norm1, norm2;
    PetscCall(VecNorm(qp->b, NORM_2, &norm1));
    PetscCall(VecNorm(newb, NORM_2, &norm2));
    PetscCall(FllopDebug2("\n    ||b||\t= %.12e  b = b_bar\n    ||Pb||\t= %.12e\n",norm1,norm2));
  }
  PetscCall(QPSetRhs(child, newb));
  PetscCall(VecDestroy(&newb));

  /* create special preconditioner pc_child = P * pc_parent */
  {
    PC pc_parent,pc_child;

    PetscCall(QPGetPC(qp,&pc_parent));
    PetscCall(PCCreate_QPTEnforceEqByProjector(pc_parent,P,pc_symmetric,&pc_child));
    PetscCall(QPSetPC(child,pc_child));
    PetscCall(PCDestroy(&pc_child));
  }

  PetscCall(QPSetWorkVector(child,qp->xwork));

  PetscCall(MatDestroy(&P));
  PetscCall(PetscLogEventEnd(QPT_EnforceEqByProjector,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTEnforceEqByPenalty_PostSolve_Private"
static PetscErrorCode QPTEnforceEqByPenalty_PostSolve_Private(QP child,QP parent)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

//TODO allow to set multiple of max eigenvalue
#undef __FUNCT__
#define __FUNCT__ "QPTEnforceEqByPenalty"
PetscErrorCode QPTEnforceEqByPenalty(QP qp, PetscReal rho_user, PetscBool rho_direct)
{
  MPI_Comm         comm;
  Mat              newA=NULL;
  Vec              newb=NULL;
  QP               child;
  PetscReal        rho,maxeig;
  PetscReal        maxeig_tol=PETSC_DECIDE;
  PetscInt         maxeig_iter=PETSC_DECIDE;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidLogicalCollectiveReal(qp,rho_user,2);
  PetscValidLogicalCollectiveBool(qp,rho_direct,3);

  if (!rho_user) {
    PetscCall(PetscInfo(qp, "penalty=0.0 ==> no effect, returning...\n"));
    PetscFunctionReturn(0);
  }

  PetscCall(QPChainGetLast(qp,&qp));

  if (!qp->BE) {
    PetscCall(PetscInfo(qp, "no lin. eq. con. matrix specified ==> nothing to enforce\n"));
    PetscCall(VecDestroy(&qp->cE));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  if (qp->cE) {
    PetscBool flg=PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_homogenize_eq_always",&flg,NULL));
    if (flg) {
      PetscCall(PetscInfo(qp, "nonzero lin. eq. con. RHS prescribed and -qpt_homogenize_eq_always set to true ==> automatically calling QPTHomogenizeEq\n"));
      PetscCall(QPTHomogenizeEq(qp));
      PetscCall(QPChainGetLast(qp,&qp));
    }
  }

  PetscCall(PetscOptionsGetReal(NULL,NULL,"-qpt_penalize_maxeig_tol",&maxeig_tol,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-qpt_penalize_maxeig_iter",&maxeig_iter,NULL));

  if (!rho_direct) {
    PetscCall(MatGetMaxEigenvalue(qp->A, NULL, &maxeig, maxeig_tol, maxeig_iter));
    rho = rho_user * maxeig;
  } else {
    rho = rho_user;
  }

  if (rho < 0) {
    SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_WRONG,"rho must be nonnegative");
  }
  PetscCall(PetscInfo(qp, "using penalty = real %.12e\n", rho));

  PetscCall(PetscLogEventBegin(QPT_EnforceEqByPenalty,qp,0,0,0));
  PetscCall(QPTransformBegin(QPTEnforceEqByPenalty, QPTEnforceEqByPenalty_PostSolve_Private,NULL, QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  PetscCall(QPAppendOptionsPrefix(child,"pnlt_"));

  PetscCall(QPSetEq(  child, NULL, NULL));
  PetscCall(QPSetQPC(child, qp->qpc));
  PetscCall(QPSetIneq(child, qp->BI, qp->cI));
  PetscCall(QPSetRhs(child, qp->b));
  PetscCall(QPSetInitialVector(child, qp->x));

  /* newA = A + rho*BE'*BE */
  PetscCall(MatCreatePenalized(qp,rho,&newA));
  PetscCall(QPSetOperator(child,newA));
  PetscCall(QPSetOperatorNullSpace(child,qp->R));
  PetscCall(MatDestroy(&newA));

  /* newb = b + rho*BE'*c */
  if (qp->c) {
    PetscCall(VecDuplicate(qp->b,&newb));
    PetscCall(MatMultTranspose(qp->BE,qp->c,newb));
    PetscCall(VecAYPX(newb,rho,qp->b));
    PetscCall(QPSetRhs(child,newb));
  }
  
  PetscCall(QPSetIneqMultiplier(       child,qp->lambda_I));
  PetscCall(QPSetWorkVector(child,qp->xwork));

  PetscCall(PetscLogEventEnd(QPT_EnforceEqByPenalty,qp,0,0,0));
  PetscFunctionReturnI(0);
}


#undef __FUNCT__
#define __FUNCT__ "QPTHomogenizeEqPostSolve_Private"
static PetscErrorCode QPTHomogenizeEqPostSolve_Private(QP child,QP parent)
{
  Vec xtilde = (Vec) child->postSolveCtx;

  PetscFunctionBegin;
  /* x_parent = x_child + xtilde */
  PetscCall(VecWAXPY(parent->x,1.0,child->x,xtilde));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTHomogenizeEqPostSolveCtxDestroy_Private"
static PetscErrorCode QPTHomogenizeEqPostSolveCtxDestroy_Private(void *ctx)
{
  Vec xtilde = (Vec) ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&xtilde));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTHomogenizeEq"
PetscErrorCode QPTHomogenizeEq(QP qp)
{
  MPI_Comm          comm;
  QP                child;
  Vec               b_bar, cineq, lb, ub, lbnew, ubnew, xtilde, xtilde_sub;
  IS                is;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  lb = NULL; lbnew=NULL;
  ub = NULL; ubnew=NULL;
  is = NULL;

  PetscCall(QPChainGetLast(qp,&qp));
  if (!qp->cE) {
    PetscCall(PetscInfo(qp, "lin. eq. con. already homogenous ==> nothing to homogenize\n"));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  PetscCall(PetscLogEventBegin(QPT_HomogenizeEq,qp,0,0,0));
  PetscCall(QPTransformBegin(QPTHomogenizeEq, QPTHomogenizeEqPostSolve_Private,QPTHomogenizeEqPostSolveCtxDestroy_Private, QP_DUPLICATE_COPY_POINTERS,&qp,&child,&comm));

  /* A, R remain the same */

  PetscCall(VecDuplicate(qp->x,&xtilde));
  PetscCall(QPPFApplyHalfQTranspose(qp->pf,qp->cE,xtilde));                         /* xtilde = BE'*inv(BE*BE')*cE */

  PetscCall(VecDuplicate(qp->b, &b_bar));
  PetscCall(MatMult(qp->A, xtilde, b_bar));
  PetscCall(VecAYPX(b_bar, -1.0, qp->b));                                           /* b_bar = b - A*xtilde */
  PetscCall(QPSetRhs(child, b_bar));
  PetscCall(VecDestroy(&b_bar));

  if (FllopDebugEnabled) {
    PetscReal norm1,norm2,norm3,norm4;
    PetscCall(VecNorm(qp->cE, NORM_2, &norm1));
    PetscCall(VecNorm(xtilde, NORM_2, &norm2));
    PetscCall(VecNorm(qp->b,  NORM_2, &norm3));
    PetscCall(VecNorm(child->b,NORM_2, &norm4));
    PetscCall(FllopDebug4("\n"
        "    ||ceq||\t= %.12e  ceq = e\n"
        "    ||xtilde||\t= %.12e  xtilde = Beq'*inv(Beq*Beq')*ceq\n"
        "    ||b||\t= %.12e  b = d\n"
        "    ||b_bar||\t= %.12e  b_bar = b-A*xtilde\n",norm1,norm2,norm3,norm4));
  }

  PetscCall(QPSetQPPF(child, qp->pf));
  PetscCall(QPSetEq(child,qp->BE,NULL));                                            /* cE is eliminated */

  cineq = NULL;
  if (qp->cI) {
    PetscCall(VecDuplicate(qp->cI, &cineq));
    PetscCall(MatMult(qp->BI, xtilde, cineq));
    PetscCall(VecAYPX(cineq, -1.0, qp->cI));                                        /* cI = cI - BI*xtilde */
  }
  PetscCall(QPSetIneq(child, qp->BI, cineq));
  PetscCall(VecDestroy(&cineq));

  PetscCall(QPGetBox(qp, &is, &lb, &ub));
  if (is) {
    PetscCall(VecGetSubVector(xtilde, is, &xtilde_sub));
  } else {
    xtilde_sub = xtilde;
  }

  if (lb) {
    PetscCall(VecDuplicate(lb, &lbnew));
    PetscCall(VecWAXPY(lbnew, -1.0, xtilde_sub, lb));                                /* lb = lb - xtilde */
  }

  if (ub) {
    PetscCall(VecDuplicate(ub, &ubnew));
    PetscCall(VecWAXPY(ubnew, -1.0, xtilde_sub, ub));                                /* ub = ub - xtilde */
  }

  if (is) {
    PetscCall(VecRestoreSubVector(xtilde, is, &xtilde_sub));
  }

  PetscCall(QPSetBox(child, is, lbnew, ubnew));
  PetscCall(VecDestroy(&lbnew));
  PetscCall(VecDestroy(&ubnew));

  PetscCall(VecDestroy(&child->x));

  child->postSolveCtx = xtilde;
  PetscCall(PetscLogEventEnd(QPT_HomogenizeEq,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTOrthonormalizeEq"
static PetscErrorCode QPTPostSolve_QPTOrthonormalizeEq(QP child,QP parent)
{
  Mat T = (Mat) child->postSolveCtx;
  PetscBool skip_lambda_E=PETSC_TRUE,skip_Bt_lambda=PETSC_TRUE;

  PetscFunctionBegin;
  /* B_p'*lambda_p = B_c'*lambda_c is done implicitly */

  //PetscCall(VecIsInvalidated(child->lambda_E,&skip_lambda_E));
  //PetscCall(VecIsInvalidated(child->Bt_lambda,&skip_Bt_lambda));

  if (!skip_lambda_E && T->ops->multtranspose) {
    /* lambda_E_p = T'*lambda_E_c */
    PetscCall(MatMultTranspose(T,child->lambda_E,parent->lambda_E));
  } else if (!skip_Bt_lambda) {
    //TODO this seems to be inaccurate for explicit orthonormalization
    /* lambda_E_p = (G*G')\G(G'*lambda_c) where G'*lambda_c = child->Bt_lambda */
    PetscCall(QPPFApplyHalfQ(child->pf,child->Bt_lambda,parent->lambda_E));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTOrthonormalizeEq"
static PetscErrorCode QPTPostSolveDestroy_QPTOrthonormalizeEq(void *ctx)
{
  Mat T = (Mat) ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTOrthonormalizeEq"
PetscErrorCode QPTOrthonormalizeEq(QP qp,MatOrthType type,MatOrthForm form)
{
  MPI_Comm          comm;
  QP                child;
  Mat               BE,TBE,T;
  Vec               cE,TcE;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  PetscCall(QPChainGetLast(qp,&qp));
  if (!qp->BE) {
    PetscCall(PetscInfo(qp, "no lin. eq. con. ==> nothing to orthonormalize\n"));
    PetscFunctionReturn(0);
  }
  if (type == MAT_ORTH_NONE) {
    PetscCall(PetscInfo(qp, "MAT_ORTH_NONE ==> skipping orthonormalization\n"));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  if (type == MAT_ORTH_INEXACT && qp->cE) {
    PetscCall(PetscInfo(qp, "MAT_ORTH_INEXACT and nonzero lin. eq. con. RHS prescribed ==> automatically calling QPTHomogenizeEq\n"));
    PetscCall(QPTHomogenizeEq(qp));
    PetscCall(QPChainGetLast(qp,&qp));
  }

  PetscCall(PetscLogEventBegin(QPT_OrthonormalizeEq,qp,0,0,0));
  PetscCall(QPTransformBegin(QPTOrthonormalizeEq, QPTPostSolve_QPTOrthonormalizeEq,QPTPostSolveDestroy_QPTOrthonormalizeEq, QP_DUPLICATE_COPY_POINTERS,&qp,&child,&comm));

  PetscCall(QPGetEq(qp, &BE, &cE));
  PetscCall(MatOrthRows(BE, type, form, &TBE, &T));

  {
    const char *name;
    PetscCall(PetscObjectGetName((PetscObject)BE,&name));
    PetscCall(PetscObjectSetName((PetscObject)TBE,name));
    PetscCall(MatPrintInfo(T));
    PetscCall(MatPrintInfo(TBE));
  }

  TcE=NULL;
  if (cE) {
    if (type == MAT_ORTH_IMPLICIT || type == MAT_ORTH_INEXACT) {
      PetscCall(PetscObjectReference((PetscObject)cE));
      TcE = cE;
    } else {
      PetscCall(VecDuplicate(cE,&TcE));
      PetscCall(MatMult(T,cE,TcE));
    }
  }

  PetscCall(QPSetQPPF(child,NULL));
  PetscCall(QPSetEq(child,TBE,TcE));                /* QPPF re-created in QPSetEq */
  PetscCall(QPSetEqMultiplier(child,NULL));         /* lambda_E will be re-created in QPSetUp */

  if (type == MAT_ORTH_IMPLICIT) {
    /* inherit GGtinv to avoid extra GGt factorization - dirty way */
    PetscCall(QPPFSetUp(qp->pf));
    child->pf->GGtinv = qp->pf->GGtinv;
    child->pf->Gt = qp->pf->Gt;
    PetscCall(PetscObjectReference((PetscObject)qp->pf->GGtinv));
    PetscCall(PetscObjectReference((PetscObject)qp->pf->Gt));
    PetscCall(QPPFSetUp(child->pf));
  } else if (type == MAT_ORTH_INEXACT) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)child->pf,"inexact_"));
    PetscCall(PetscObjectCompose((PetscObject)child->pf,"exact",(PetscObject)qp->pf));
  }

  PetscCall(MatDestroy(&TBE));
  PetscCall(VecDestroy(&TcE));
  child->postSolveCtx = T;
  PetscCall(PetscLogEventEnd(QPT_OrthonormalizeEq,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTOrthonormalizeEqFromOptions"
PetscErrorCode QPTOrthonormalizeEqFromOptions(QP qp)
{
  MatOrthType eq_orth_type=MAT_ORTH_NONE;
  MatOrthForm eq_orth_form=MAT_ORTH_FORM_IMPLICIT;
  QP last;

  PetscFunctionBeginI;
  PetscCall(QPChainGetLast(qp,&last));
  PetscObjectOptionsBegin((PetscObject)last);
  PetscCall(PetscOptionsEnum("-qp_E_orth_type","type of eq. con. orthonormalization","QPTOrthonormalizeEq",MatOrthTypes,(PetscEnum)eq_orth_type,(PetscEnum*)&eq_orth_type,NULL));
  PetscCall(PetscOptionsEnum("-qp_E_orth_form","form of eq. con. orthonormalization","QPTOrthonormalizeEq",MatOrthForms,(PetscEnum)eq_orth_form,(PetscEnum*)&eq_orth_form,NULL));
  PetscCall(PetscInfo(qp, "-qp_E_orth_type %s\n",MatOrthTypes[eq_orth_type]));
  PetscCall(PetscInfo(qp, "-qp_E_orth_form %s\n",MatOrthForms[eq_orth_form]));
  PetscCall(QPTOrthonormalizeEq(last,eq_orth_type,eq_orth_form));
  PetscOptionsEnd();
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTDualizeViewBSpectra_Private"
static PetscErrorCode QPTDualizeViewBSpectra_Private(Mat B)
{
  PetscFunctionBegin;
  PetscInt j,Mn;
  Mat Bt,BBt,Bn;
  PetscReal norm;
  PetscBool flg;
  const char *name;

  PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&flg,MATNEST,MATNESTPERMON,""));
  if (flg) {
    PetscCall(MatNestGetSize(B,&Mn,NULL));
    for (j=0; j<Mn; j++) {
      PetscCall(MatNestGetSubMat(B,j,0,&Bn));
      PetscCall(QPTDualizeViewBSpectra_Private(Bn));
    }
  }
  PetscCall(PetscObjectGetName((PetscObject)B,&name));
  PetscCall(MatCreateTranspose(B,&Bt));
  PetscCall(MatCreateNormal(Bt,&BBt));
  PetscCall(MatGetMaxEigenvalue(BBt,NULL,&norm,1e-6,10));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)B),"||%s * %s'|| = %.12e (10 power method iterations)\n",name,name,norm));
  PetscCall(MatDestroy(&BBt));
  PetscCall(MatDestroy(&Bt));
  PetscFunctionReturn(0);
}

#define QPTDualizeView_Private_SetName(mat,matname) if (mat && !((PetscObject)mat)->name) PetscCall(PetscObjectSetName((PetscObject)mat,matname) )

#undef __FUNCT__
#define __FUNCT__ "QPTDualizeView_Private"
static PetscErrorCode QPTDualizeView_Private(QP qp, QP child)
{
  MPI_Comm comm;
  Mat F,Kplus,K,Kreg,B,Bt,R,G;
  Vec c,d,e,f,lb,lambda,u;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)qp, &comm));

  F = child->A;
  PetscCall(PetscObjectQuery((PetscObject)F,"Kplus",(PetscObject*)&Kplus));
  PetscCall(PetscObjectQuery((PetscObject)F,"B",(PetscObject*)&B));
  PetscCall(PetscObjectQuery((PetscObject)F,"Bt",(PetscObject*)&Bt));
  
  K = qp->A;
  R = qp->R;
  G = child->BE;
  c = qp->c;
  d = child->b;
  e = child->cE;
  f = qp->b;
  PetscCall(QPGetBox(child,NULL,&lb,NULL));
  lambda = child->x;
  u = qp->x;

  if (Kplus) {
    Mat Kplus_inner;
    PetscCall(PetscObjectQuery((PetscObject)Kplus,"Kplus",(PetscObject*)&Kplus_inner));
    if (Kplus_inner) Kplus = Kplus_inner;
    PetscCall(MatInvGetRegularizedMat(Kplus,&Kreg));
  }

  QPTDualizeView_Private_SetName(K,     "K");
  QPTDualizeView_Private_SetName(Kreg,  "Kreg");
  QPTDualizeView_Private_SetName(Kplus, "Kplus");
  QPTDualizeView_Private_SetName(R,     "R");
  QPTDualizeView_Private_SetName(u,     "u");
  QPTDualizeView_Private_SetName(f,     "f");
  QPTDualizeView_Private_SetName(B,     "B");
  QPTDualizeView_Private_SetName(Bt,    "Bt");
  QPTDualizeView_Private_SetName(c,     "c");
  QPTDualizeView_Private_SetName(F,     "F");
  QPTDualizeView_Private_SetName(lambda,"lambda");
  QPTDualizeView_Private_SetName(d,     "d");
  QPTDualizeView_Private_SetName(G,     "G");
  QPTDualizeView_Private_SetName(e,     "e");
  QPTDualizeView_Private_SetName(lb,    "lb");

  if (FllopObjectInfoEnabled && !PetscPreLoadingOn) {
    PetscCall(PetscPrintf(comm, "*** %s:\n",__FUNCT__));
    if (K)      PetscCall(MatPrintInfo(K));
    if (Kreg)   PetscCall(MatPrintInfo(Kreg));
    if (Kplus)  PetscCall(MatPrintInfo(Kplus));
    if (R)      PetscCall(MatPrintInfo(R));
    if (u)      PetscCall(VecPrintInfo(u));
    if (f)      PetscCall(VecPrintInfo(f));
    if (B)      PetscCall(MatPrintInfo(B));
    if (Bt)     PetscCall(MatPrintInfo(Bt));
    if (c)      PetscCall(VecPrintInfo(c));
    if (F)      PetscCall(MatPrintInfo(F));
    if (lambda) PetscCall(VecPrintInfo(lambda));
    if (d)      PetscCall(VecPrintInfo(d));
    if (G)      PetscCall(MatPrintInfo(G));
    if (e)      PetscCall(VecPrintInfo(e));
    if (lb)     PetscCall(VecPrintInfo(lb));

    PetscCall(PetscPrintf(comm, "***\n\n"));
  }

  if (FllopDebugEnabled) {
    PetscReal norm1=0.0, norm2=0.0, norm3=0.0, norm4=0.0;
    if (f) PetscCall(VecNorm(f, NORM_2, &norm1));
    if (c) PetscCall(VecNorm(c, NORM_2, &norm2));
    if (d) PetscCall(VecNorm(d, NORM_2, &norm3));
    if (e) PetscCall(VecNorm(e, NORM_2, &norm4));
    PetscCall(FllopDebug4("\n"
        "    ||f||\t= %.12e\n"
        "    ||c||\t= %.12e\n"
        "    ||d||\t= %.12e  d = B*Kplus*f-c \n"
        "    ||e||\t= %.12e  e = R'*f \n",norm1,norm2,norm3,norm4));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTDualizePostSolve_Private"
static PetscErrorCode QPTDualizePostSolve_Private(QP child,QP parent)
{
    Mat F          = child->A;
    Vec alpha      = child->lambda_E;
    Mat Kplus;
    Vec f          = parent->b;
    Vec u          = parent->x;
    Vec tprim      = parent->xwork;
    PetscBool flg;

    PetscFunctionBegin;
    PetscCall(PetscObjectQuery((PetscObject)F,"Kplus",(PetscObject*)&Kplus));
    PERMON_ASSERT(Kplus,"Kplus != NULL");

    /* copy lambda back to lambda_E and lambda_I */
    if (parent->BE && parent->BI) {
      PetscInt i;
      IS rows[2];
      Vec as[2]={parent->lambda_E,parent->lambda_I};
      Vec a[2];

      PetscCall(MatNestGetISs(parent->B,rows,NULL));
      for (i=0; i<2; i++) {
        if (as[i]) {
          PetscCall(VecGetSubVector(parent->lambda,rows[i],&a[i]));
          PetscCall(VecCopy(a[i],as[i]));
          PetscCall(VecRestoreSubVector(parent->lambda,rows[i],&a[i]));
        }
      }
    }

    /* u = Kplus*(f-B'*lambda) */
    PetscCall(MatMultTranspose(parent->B, parent->lambda, tprim));
    PetscCall(VecAYPX(tprim, -1.0, f));
    PetscCall(MatMult(Kplus, tprim, u));

    if (alpha) {
      PetscCall(VecIsInvalidated(alpha,&flg));
      if (flg) {
        /* compute alpha = (G*G')\G(G'*alpha) where G'*alpha = child->Bt_lambda */
        PetscCall(VecIsInvalidated(child->Bt_lambda,&flg));
        if (flg) {
          PetscCall(MatMultTranspose(child->B,child->lambda,child->Bt_lambda));
        }
        PetscCall(QPPFApplyHalfQ(child->pf,child->Bt_lambda,alpha));
      }

      /* u = u - R*alpha */
      PetscCall(MatMult(parent->R, alpha, tprim));
      PetscCall(VecAXPY(u, -1.0, tprim));
    }
    PetscFunctionReturn(0);
}

//TODO this a prototype, integrate to API
#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_R_Bt"
static PetscErrorCode MatTransposeMatMult_R_Bt(Mat R, Mat Bt, Mat *G_new)
{
  Mat       G,Gt;
  PetscBool flg;
  PetscBool G_explicit = PETSC_TRUE;

  PetscFunctionBeginI;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_G_explicit",&G_explicit,NULL));

  if (!G_explicit) {
    //NOTE doesn't work with redundant GGt_inv & redundancy of R is problematic due to its structure
    Mat G_arr[3];
    Mat Rt;

    PetscCall(PermonMatTranspose(R,MAT_TRANSPOSE_CHEAPEST,&Rt));

    PetscCall(MatCreateTimer(Rt,&G_arr[1]));
    PetscCall(MatCreateTimer(Bt,&G_arr[0]));
    
    PetscCall(MatCreateProd(PetscObjectComm((PetscObject)Bt), 2, G_arr, &G));
    PetscCall(MatCreateTimer(G,&G_arr[2]));
    PetscCall(MatDestroy(&G));
    G = G_arr[2];

    PetscCall(PetscObjectCompose((PetscObject)G,"Bt",(PetscObject)Bt));
    PetscCall(PetscObjectCompose((PetscObject)G,"R",(PetscObject)R));
    PetscCall(PetscObjectSetName((PetscObject)G,"G"));

    PetscCall(MatDestroy(&G_arr[1]));
    PetscCall(MatDestroy(&G_arr[0]));
    PetscCall(MatDestroy(&Rt));
    *G_new = G;
    PetscFunctionReturn(0);
  }

  PetscCall(PetscObjectTypeCompareAny((PetscObject)Bt,&flg,MATNEST,MATNESTPERMON,""));
  if (!flg) {
    PetscCall(PetscObjectTypeCompare((PetscObject)Bt,MATEXTENSION,&flg));
    //PetscCall(MatTransposeMatMultWorks(R,Bt,&flg));
    if (flg) {
      PetscCall(MatTransposeMatMult(R,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&G));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)Bt), "WARNING: MatTransposeMatMult not applicable, falling back to MatMatMultByColumns\n"));
      PetscCall(MatTransposeMatMultByColumns(Bt,R,PETSC_TRUE,&Gt));
      PetscCall(PermonMatTranspose(Gt,MAT_TRANSPOSE_CHEAPEST,&G));
      PetscCall(MatDestroy(&Gt));
    }
  } else {
    //TODO make this MATNESTPERMON's method
    PetscInt Mn,Nn,J;
    Mat BtJ;
    Mat *mats_G;
    PetscCall(MatNestGetSize(Bt,&Mn,&Nn));
    PERMON_ASSERT(Mn==1,"Mn==1");
    PetscCall(PetscMalloc(Nn*sizeof(Mat),&mats_G));
    for (J=0; J<Nn; J++) {
      PetscCall(MatNestGetSubMat(Bt,0,J,&BtJ));
      PetscCall(MatTransposeMatMult_R_Bt(R,BtJ,&mats_G[J]));
    }
    PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)Bt),1,NULL,Nn,NULL,mats_G,&G));
    for (J=0; J<Nn; J++) PetscCall(MatDestroy(&mats_G[J]));
    PetscCall(PetscFree(mats_G));
  }

  PetscCall(PetscObjectSetName((PetscObject)G,"G"));
  *G_new = G;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTDualize"
PetscErrorCode QPTDualize(QP qp,MatInvType invType,MatRegularizationType regType)
{
  MPI_Comm         comm;
  QP               child;
  Mat              R,B,Bt,F,G,K,Kplus,Kplus_orig;
  Vec              c,d,e,f,lb,tprim,lambda;
  PetscBool        B_explicit = PETSC_FALSE, B_extension = PETSC_FALSE, B_view_spectra = PETSC_FALSE;
  PetscBool        B_nest_extension = PETSC_FALSE;
  PetscBool        mp = PETSC_FALSE;
  PetscBool        true_mp = PETSC_FALSE;
  PetscBool        spdset,spd;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscCall(PetscLogEventBegin(QPT_Dualize,qp,0,0,0));
  PetscCall(QPTransformBegin(QPTDualize, QPTDualizePostSolve_Private,NULL, QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  PetscCall(QPAppendOptionsPrefix(child,"dual_"));

  Kplus_orig = NULL;
  B = NULL;
  c = qp->c;
  lambda = qp->lambda;
  tprim = qp->xwork;
  K = qp->A;
  f = qp->b;
  if (!qp->B) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"lin. equality and/or inequality constraint matrix (BE/BI) is needed for dualization");

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_explicit",&B_explicit,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_extension",&B_extension,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_nest_extension",&B_nest_extension,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_view_spectra",&B_view_spectra,NULL));

  if (B_view_spectra) {
    PetscCall(QPTDualizeViewBSpectra_Private(qp->B));
  }

  PetscCall(MatPrintInfo(qp->B));

  PetscCall(PetscLogEventBegin(QPT_Dualize_PrepareBt,qp,0,0,0));
  if (B_extension || B_nest_extension) {
    Mat B_merged;
    MatTransposeType ttype = B_explicit ? MAT_TRANSPOSE_EXPLICIT : MAT_TRANSPOSE_CHEAPEST;
    PetscCall(MatCreateNestPermonVerticalMerge(comm,1,&qp->B,&B_merged));
    PetscCall(PermonMatTranspose(B_merged,MAT_TRANSPOSE_EXPLICIT,&Bt));
    PetscCall(MatDestroy(&B_merged));
    if (B_extension) {
      PetscCall(MatConvert(Bt,MATEXTENSION,MAT_INPLACE_MATRIX,&Bt));
    } else {
      PetscCall(PermonMatConvertBlocks(Bt,MATEXTENSION,MAT_INPLACE_MATRIX,&Bt));
    }
    PetscCall(PermonMatTranspose(Bt,ttype,&B));
  } else {
    PetscCall(PermonMatTranspose(qp->B,MAT_TRANSPOSE_CHEAPEST,&Bt));
    if (B_explicit) {
      PetscCall(PermonMatTranspose(Bt,MAT_TRANSPOSE_EXPLICIT,&B));
    } else {
      /* in this case B remains the same */
      B = qp->B;
      PetscCall(PetscObjectReference((PetscObject)B));
    }
  }
  PetscCall(FllopPetscObjectInheritName((PetscObject)B,(PetscObject)qp->B,NULL));
  PetscCall(FllopPetscObjectInheritName((PetscObject)Bt,(PetscObject)qp->B,"_T"));
  PetscCall(PetscLogEventEnd(  QPT_Dualize_PrepareBt,qp,0,0,0));

  if (FllopObjectInfoEnabled) {
    PetscCall(PetscPrintf(comm, "B and Bt after conversion:\n"));
    PetscCall(MatPrintInfo(B));
    PetscCall(MatPrintInfo(Bt));
  }

  /* create stiffness matrix pseudoinverse */
  PetscCall(MatCreateInv(K, invType, &Kplus));
  Kplus_orig = Kplus;
  PetscCall(MatPrintInfo(K));
  PetscCall(FllopPetscObjectInheritName((PetscObject)Kplus,(PetscObject)K,"_plus"));
  PetscCall(FllopPetscObjectInheritPrefix((PetscObject)Kplus,(PetscObject)child,NULL));

  /* get or compute stiffness matrix kernel (R) */
  R = NULL;
  PetscCall(QPGetOperatorNullSpace(qp,&R));
  PetscCall(MatIsSPDKnown(qp->A,&spdset,&spd));
  if (R) {
    PetscCall(MatInvSetNullSpace(Kplus,R));
    //TODO consider not inheriting R with 0 cols
  } else if (spdset && spd) {
    PetscCall(PetscInfo(qp,"Hessian flagged SPD => not computing null space\n"));
  } else {
    PetscInt Ncols;
    PetscCall(PetscInfo(qp,"null space matrix not set => trying to compute one\n"));
    PetscCall(MatInvComputeNullSpace(Kplus));
    PetscCall(MatInvGetNullSpace(Kplus,&R));
    PetscCall(MatGetSize(R,NULL,&Ncols));
    if (Ncols) {
      PetscCall(MatOrthColumns(R, MAT_ORTH_GS, MAT_ORTH_FORM_EXPLICIT, &R, NULL));
      PetscCall(QPSetOperatorNullSpace(qp,R));
      PetscCall(PetscInfo(qp,"computed null space matrix => using -qpt_dualize_Kplus_left and -regularize 0\n"));
      mp = PETSC_TRUE;
      true_mp = PETSC_FALSE;
      regType = MAT_REG_NONE;
    } else {
      PetscCall(PetscInfo(qp,"computed null space matrix has 0 columns => consider setting SPD flag to Hessian to avoid null space detection\n"));
      R = NULL;
    }
  }
  PetscCall(MatInvSetRegularizationType(Kplus,regType));
  PetscCall(MatSetFromOptions(Kplus));

  PetscCall(PetscLogEventBegin(QPT_Dualize_FactorK,qp,Kplus,0,0));
  PetscCall(MatInvSetUp(Kplus));
  PetscCall(PetscLogEventEnd  (QPT_Dualize_FactorK,qp,Kplus,0,0));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_Kplus_mp",&true_mp,NULL));
  if (!true_mp) {
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_Kplus_left",&mp,NULL));
  }

  /* convert to Moore-Penrose pseudoinverse using projector to image of K (kernel of R') */
  if (true_mp || mp) {
    if (R) {
      QPPF pf_R;
      Mat P_R;
      Mat mats[3];
      Mat Kplus_new;
      Mat Rt;
      PetscInt size=2;

      if (true_mp) {
        PetscCall(PetscInfo(qp,"creating Moore-Penrose inverse\n"));
      } else {
        PetscCall(PetscInfo(qp,"creating left generalized inverse\n"));
      }
      PetscCall(PermonMatTranspose(R,MAT_TRANSPOSE_CHEAPEST,&Rt));
      PetscCall(PetscObjectSetName((PetscObject)Rt,"Rt"));
      PetscCall(QPPFCreate(comm,&pf_R));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)pf_R,"Kplus_"));
      PetscCall(QPPFSetG(pf_R,Rt));
      PetscCall(QPPFCreateP(pf_R,&P_R));
      PetscCall(QPPFSetUp(pf_R));

      if (true_mp) {
        size=3;
        mats[2]=P_R;
      }
      mats[1]=Kplus; mats[0]=P_R;
      PetscCall(MatCreateProd(comm,size,mats,&Kplus_new));
      PetscCall(PetscObjectCompose((PetscObject)Kplus_new,"Kplus",(PetscObject)Kplus));

      PetscCall(MatDestroy(&Kplus));
      PetscCall(MatDestroy(&Rt));
      PetscCall(MatDestroy(&P_R));
      PetscCall(QPPFDestroy(&pf_R));
      Kplus = Kplus_new;

      if (FllopDebugEnabled) {
        /* is Kplus MP? */
        Mat mats2[3];
        Mat prod;
        PetscBool flg;
        mats2[2]=K; mats2[1]=Kplus; mats2[0]=K;
        PetscCall(MatCreateProd(comm,3,mats2,&prod));
        PetscCall(MatMultEqual(prod,K,3,&flg));
        PERMON_ASSERT(flg,"Kplus is left generalized inverse");
        PetscCall(MatDestroy(&prod));
        if (true_mp) {
          mats2[2]=Kplus; mats2[1]=K; mats2[0]=Kplus;
          PetscCall(MatCreateProd(comm,3,mats2,&prod));
          PetscCall(MatMultEqual(prod,Kplus,3,&flg));
          PERMON_ASSERT(flg,"Kplus is Moore-Penrose pseudoinverse");
          PetscCall(MatDestroy(&prod));
        }
      }
    } else {
        PetscCall(PetscInfo(qp,"ignoring requested left generalized or Moore-Penrose inverse, because null space is not set\n"));
    }
  }

  PetscCall(PetscObjectSetName((PetscObject)Kplus,"Kplus"));

  G = NULL;
  e = NULL;
  if (R) {
    /* G = R'*B' */
    PetscCall(PetscLogEventBegin(QPT_Dualize_AssembleG,qp,0,0,0));
    PetscCall(MatTransposeMatMult_R_Bt(R,Bt,&G));
    PetscCall(PetscLogEventEnd(  QPT_Dualize_AssembleG,qp,0,0,0));

    /* e = R'*f */
    PetscCall(MatCreateVecs(R,&e,NULL));
    PetscCall(MatMultTranspose(R,f,e));
  }

  /* F = B*Kplus*Bt (implicitly) */
  {
    Mat F_arr[4];
    
    PetscCall(MatCreateTimer(B,&F_arr[2]));
    PetscCall(MatCreateTimer(Kplus,&F_arr[1]));
    PetscCall(MatCreateTimer(Bt,&F_arr[0]));
  
    PetscCall(MatCreateProd(comm, 3, F_arr, &F));
    PetscCall(PetscObjectSetName((PetscObject) F, "F"));
    PetscCall(MatCreateTimer(F,&F_arr[3]));
    PetscCall(MatDestroy(&F));
    F = F_arr[3];
    
    PetscCall(PetscObjectCompose((PetscObject)F,"B",(PetscObject)B));
    PetscCall(PetscObjectCompose((PetscObject)F,"K",(PetscObject)K));
    PetscCall(PetscObjectCompose((PetscObject)F,"Kplus",(PetscObject)Kplus));
    PetscCall(PetscObjectCompose((PetscObject)F,"Kplus_orig",(PetscObject)Kplus_orig));
    PetscCall(PetscObjectCompose((PetscObject)F,"Bt",(PetscObject)Bt));
    
    PetscCall(MatDestroy(&B));       B     = F_arr[2];
    PetscCall(MatDestroy(&Kplus));   Kplus = F_arr[1];
    PetscCall(MatDestroy(&Bt));      Bt    = F_arr[0];
  }

  /* d = B*Kplus*f - c */
  PetscCall(VecDuplicate(lambda,&d));
  PetscCall(MatMult(Kplus, f, tprim));
  PetscCall(MatMult(B, tprim, d));
  if(c) PetscCall(VecAXPY(d,-1.0,c));

  /* lb(E) = -inf; lb(I) = 0 */
  if (qp->BI) {
    PetscCall(VecDuplicate(lambda,&lb));
    if (qp->BE) {
      IS rows[2];
      Vec lbE,lbI;

      PetscCall(MatNestGetISs(qp->B,rows,NULL));

      /* lb(E) = -inf */
      PetscCall(VecGetSubVector(lb,rows[0],&lbE));
      PetscCall(VecSet(lbE,PETSC_NINFINITY));
      PetscCall(VecRestoreSubVector(lb,rows[0],&lbE));

      /* lb(I) = 0 */
      PetscCall(VecGetSubVector(lb,rows[1],&lbI));
      PetscCall(VecSet(lbI,0.0));
      PetscCall(VecRestoreSubVector(lb,rows[1],&lbI));

      PetscCall(PetscObjectCompose((PetscObject)lb,"is_finite",(PetscObject)rows[1]));
    } else {
      /* lb = 0 */
      PetscCall(VecSet(lb,0.0));
    }
  } else {
    lb = NULL;
  }

  //TODO what if initial x is nonzero
  /* lambda = o */
  PetscCall(VecZeroEntries(lambda));

  /* set data of the new aux. QP */
  PetscCall(QPSetOperator(child, F));
  PetscCall(QPSetRhs(child, d));
  PetscCall(QPSetEq(child, G, e));
  PetscCall(QPSetIneq(child, NULL, NULL));
  PetscCall(QPSetBox(child, NULL, lb, NULL));
  PetscCall(QPSetInitialVector(child,lambda));
  
  /* create special preconditioner for dual formulation */
  {
    PetscCall(PCDestroy(&child->pc));
    PetscCall(QPGetPC(child,&child->pc));
    PetscCall(PCSetType(child->pc,PCDUAL));
    PetscCall(FllopPetscObjectInheritPrefix((PetscObject)child->pc,(PetscObject)child,NULL));
  }

  PetscCall(PetscLogEventEnd(QPT_Dualize,qp,0,0,0));

  PetscCall(QPTDualizeView_Private(qp,child));

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Bt));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&Kplus));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&lb));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFetiPrepare"
PetscErrorCode QPTFetiPrepare(QP qp,PetscBool regularize)
{
  PetscFunctionBeginI;
  PetscCall(PetscLogEventBegin(QPT_FetiPrepare,qp,0,0,0));
  PetscCall(QPTDualize(qp, MAT_INV_BLOCKDIAG, regularize ? MAT_REG_EXPLICIT : MAT_REG_NONE));
  PetscCall(QPTHomogenizeEq(qp));
  PetscCall(QPTEnforceEqByProjector(qp));
  PetscCall(PetscLogEventEnd  (QPT_FetiPrepare,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFetiPrepareReuseCP"
PetscErrorCode QPTFetiPrepareReuseCP(QP qp,PetscBool regularize)
{
  QP dualQP;

  PetscFunctionBeginI;
  PetscCall(QPTDualize(qp, MAT_INV_BLOCKDIAG, regularize ? MAT_REG_EXPLICIT : MAT_REG_NONE));

  PetscCall(QPChainGetLast(qp, &dualQP));
  if (QPReusedCP) {
    Mat G;
    Vec e;

    /* reuse the coarse problem from the 0th iteration */
    PetscCall(QPSetQPPF(dualQP, QPReusedCP));
    PetscCall(QPPFGetG(QPReusedCP, &G));
    PetscCall(QPGetEq(dualQP, NULL, &e));
    PetscCall(QPSetEq(dualQP, G, e));
  } else {
    /* store the coarse problem from the 0th iteration */
    PetscCall(QPGetQPPF(dualQP, &QPReusedCP));
    PetscCall(QPPFSetUp(QPReusedCP));
    PetscCall(PetscObjectReference((PetscObject) QPReusedCP));
  }

  PetscCall(QPTHomogenizeEq(qp));
  PetscCall(QPTEnforceEqByProjector(qp));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFetiPrepareReuseCPReset"
PetscErrorCode QPTFetiPrepareReuseCPReset()
{
  PetscFunctionBegin;
  PetscCall(QPPFDestroy(&QPReusedCP));
  QPReusedCP = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTRemoveGluingOfDirichletDofs"
static PetscErrorCode QPTPostSolve_QPTRemoveGluingOfDirichletDofs(QP child,QP parent)
{
  IS is = (IS) child->postSolveCtx;
  IS *iss_parent, *iss_child;
  Vec *lambda_parent, *lambda_child;
  PetscInt Mn,II;

  PetscFunctionBegin;
  Mn = child->BE_nest_count;
  PERMON_ASSERT(Mn>=1,"child->BE_nest_count >= 2");
  PetscCall(PetscMalloc4(Mn,&iss_parent,Mn,&iss_child,Mn,&lambda_parent,Mn,&lambda_child));

  PetscCall(MatNestGetISs(parent->BE,iss_parent,NULL));
  PetscCall(MatNestGetISs(child->BE,iss_child,NULL));

  /* 0,1 corresponds to gluing and Dirichlet part, respectively */
  for (II=0; II<Mn; II++) {
    PetscCall(VecGetSubVector(parent->lambda_E,iss_parent[II],&lambda_parent[II]));
    PetscCall(VecGetSubVector(child->lambda_E, iss_child[II], &lambda_child[II]));
  }

#if 0
  {
    PetscInt start, end;
    IS is_removed;

    PetscCall(VecGetOwnershipRange(lambda_parent[0],&start,&end));
    PetscCall(ISComplement(is,start,end,&is_removed));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "#### "__FUNCT__": is_removed:\n"));
    PetscCall(ISView(is_removed,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n"));
  }
#endif

  /* copy values from the multiplier related to the restricted Bg to the multiplier
     corresponding to the original Bg, pad with zeros */
  PetscCall(VecZeroEntries(lambda_parent[0]));
  //TODO PETSc 3.5+ VecGetSubVector,VecRestoreSubVector
  {
    VecScatter sc;
    PetscCall(VecScatterCreate(lambda_child[0],NULL,lambda_parent[0],is,&sc));
    PetscCall(VecScatterBegin(sc,lambda_child[0],lambda_parent[0],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(  sc,lambda_child[0],lambda_parent[0],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&sc));
  }

  /* Bd is the same, just copy values */
  for (II=1; II<Mn; II++) {
    PetscCall(VecCopy(lambda_child[II],lambda_parent[II]));
  }

  for (II=0; II<Mn; II++) {
    PetscCall(VecRestoreSubVector(parent->lambda_E,iss_parent[II],&lambda_parent[II]));
    PetscCall(VecRestoreSubVector(child->lambda_E, iss_child[II], &lambda_child[II]));
  }

  PetscCall(PetscFree4(iss_parent,iss_child,lambda_parent,lambda_child));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTRemoveGluingOfDirichletDofs"
static PetscErrorCode QPTPostSolveDestroy_QPTRemoveGluingOfDirichletDofs(void *ctx)
{
  IS is = (IS) ctx;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTRemoveGluingOfDirichletDofs"
PetscErrorCode QPTRemoveGluingOfDirichletDofs(QP qp)
{
  QP child;
  MPI_Comm comm;
  PetscBool flg;
  Mat Bg, Bgt, Bgt_new, Bg_new, Bd, Bdt;
  Mat **mats;
  PetscInt Mn,Nn,II;
  IS is;

  PetscFunctionBeginI;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)qp->BE,&flg,MATNEST,MATNESTPERMON,""));
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_SUP,"only for eq. con. matrix qp->BE of type MATNEST or MATNESTPERMON");

  PetscCall(PetscLogEventBegin(QPT_RemoveGluingOfDirichletDofs,qp,0,0,0));
  PetscCall(MatNestGetSize(qp->BE,&Mn,&Nn));
  PetscCall(MatNestGetSubMats(qp->BE,&Mn,&Nn,&mats));
  PERMON_ASSERT(Mn>=2,"Mn==2");
  PERMON_ASSERT(Nn==1,"Nn==1");
  PERMON_ASSERT(!qp->cE,"!qp->cE");
  Bg = mats[0][0];
  Bd = mats[1][0];

  PetscCall(QPTransformBegin(QPTRemoveGluingOfDirichletDofs,
      QPTPostSolve_QPTRemoveGluingOfDirichletDofs, QPTPostSolveDestroy_QPTRemoveGluingOfDirichletDofs,
      QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));
  PetscCall(VecDestroy(&child->lambda_E));
  PetscCall(MatDestroy(&child->B));
  PetscCall(MatDestroy(&child->BE));
  child->BE_nest_count = 0;

  PetscCall(PermonMatTranspose(Bg,MAT_TRANSPOSE_CHEAPEST,&Bgt));
  PetscCall(PermonMatTranspose(Bd,MAT_TRANSPOSE_CHEAPEST,&Bdt));

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_remove_gluing_dirichlet_old",&flg,NULL));
  if (flg) {
    PetscCall(MatRemoveGluingOfDirichletDofs_old(Bgt,NULL,Bdt,&Bgt_new,NULL,&is));
  } else {
    PetscCall(MatRemoveGluingOfDirichletDofs(Bgt,NULL,Bdt,&Bgt_new,NULL,&is));
  }

  PetscCall(MatDestroy(&Bdt));

  PetscCall(PermonMatTranspose(Bgt_new,MAT_TRANSPOSE_CHEAPEST,&Bg_new));
  PetscCall(QPAddEq(child,Bg_new,NULL));
  for (II=1; II<Mn; II++) {
    PetscCall(QPAddEq(child,mats[II][0],NULL));
  }
  PetscCall(MatDestroy(&Bgt_new));

  //TODO use VecGetSubVector with PETSc 3.5+
  PetscCall(MatCreateVecs(child->BE,NULL,&child->lambda_E));
  PetscCall(VecInvalidate(child->lambda_E));
  child->postSolveCtx = (void*) is;

  PetscCall(MatPrintInfo(Bg));
  PetscCall(VecPrintInfo(qp->lambda_E));
  PetscCall(MatPrintInfo(Bg_new));
  PetscCall(VecPrintInfo(child->lambda_E));

  PetscCall(MatDestroy(&Bg_new));
  PetscCall(PetscLogEventEnd(QPT_RemoveGluingOfDirichletDofs,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTScale"
static PetscErrorCode QPTPostSolve_QPTScale(QP child,QP parent)
{
  QPTScale_Ctx *ctx = (QPTScale_Ctx*) child->postSolveCtx;

  PetscFunctionBegin;
  if (ctx->dE) {
    PetscCall(VecPointwiseMult(parent->lambda_E,ctx->dE,child->lambda_E));
  }
  if (ctx->dI) {
    PetscCall(VecPointwiseMult(parent->lambda_I,ctx->dI,child->lambda_I));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPTPostSolveDestroy_QPTScale"
static PetscErrorCode QPTPostSolveDestroy_QPTScale(void *ctx)
{
  QPTScale_Ctx *cctx = (QPTScale_Ctx*) ctx;  

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cctx->dE));
  PetscCall(VecDestroy(&cctx->dI));
  PetscCall(VecDestroy(&cctx->dO));
  PetscCall(PetscFree(cctx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTScale_Private"
PetscErrorCode QPTScale_Private(Mat A,Vec b,Vec d,Mat *DA,Vec *Db)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,DA));
  PetscCall(FllopPetscObjectInheritName((PetscObject)*DA,(PetscObject)A,NULL));
  PetscCall(MatDiagonalScale(*DA,d,NULL));
  
  if (b) {
    PetscCall(VecDuplicate(b,Db));
    PetscCall(VecPointwiseMult(*Db,d,b));
    PetscCall(FllopPetscObjectInheritName((PetscObject)*Db,(PetscObject)b,NULL));
  } else {
    *Db = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTScale"
PetscErrorCode QPTScale(QP qp)
{
  MPI_Comm comm;
  QPScaleType ScalType;
  MatOrthType R_orth_type=MAT_ORTH_GS;
  MatOrthForm R_orth_form=MAT_ORTH_FORM_EXPLICIT;
  PetscBool remove_gluing_of_dirichlet=PETSC_FALSE;
  PetscBool set;
  QP child;
  Mat A,DA;
  Vec b,d,Db;
  QPTScale_Ctx *ctx;

  PetscFunctionBeginI;
  PetscCall(QPChainGetLast(qp,&qp));

  PetscObjectOptionsBegin((PetscObject)qp);
  PetscCall(PetscOptionsBool("-qp_E_remove_gluing_of_dirichlet","remove gluing of DOFs on Dirichlet boundary","QPTRemoveGluingOfDirichletDofs",remove_gluing_of_dirichlet,&remove_gluing_of_dirichlet,NULL));
  PetscOptionsEnd();
  PetscCall(PetscInfo(qp, "-qp_E_remove_gluing_of_dirichlet %d\n",remove_gluing_of_dirichlet));
  if (remove_gluing_of_dirichlet) {
    PetscCall(QPTRemoveGluingOfDirichletDofs(qp));
  }

  PetscCall(QPTransformBegin(QPTScale,
      QPTPostSolve_QPTScale, QPTPostSolveDestroy_QPTScale,
      QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));
  PetscCall(PetscNew(&ctx));

  PetscObjectOptionsBegin((PetscObject)qp);
  A = qp->A;
  b = qp->b;
  d = NULL;
  ScalType = QP_SCALE_NONE;
  PetscCall(PetscOptionsEnum("-qp_O_scale_type", "", "QPSetSystemScaling", QPScaleTypes, (PetscEnum)ScalType, (PetscEnum*)&ScalType, &set));
  PetscCall(PetscInfo(qp, "-qp_O_scale_type %s\n",QPScaleTypes[ScalType]));
  if (ScalType) {
    if (ScalType == QP_SCALE_ROWS_NORM_2) {
      PetscCall(MatGetRowNormalization(A,&d));
    } else {
      SETERRQ(comm,PETSC_ERR_SUP,"-qp_O_scale_type %s not supported",QPScaleTypes[ScalType]);
    }

    PetscCall(QPTScale_Private(A,b,d,&DA,&Db));
    
    PetscCall(QPSetOperator(child,DA));
    PetscCall(QPSetRhs(child,Db));
    ctx->dO = d;

    PetscCall(MatDestroy(&DA));
    PetscCall(VecDestroy(&Db));
  }

  A = qp->BE;
  b = qp->cE;
  d = NULL;
  ScalType = QP_SCALE_NONE;
  PetscCall(PetscOptionsEnum("-qp_E_scale_type", "", "QPSetEqScaling", QPScaleTypes, (PetscEnum)ScalType, (PetscEnum*)&ScalType, &set));
  PetscCall(PetscInfo(qp, "-qp_E_scale_type %s\n",QPScaleTypes[ScalType]));
  if (ScalType) {
    if (ScalType == QP_SCALE_ROWS_NORM_2) {
      PetscCall(MatGetRowNormalization(A,&d));
    } else if (ScalType == QP_SCALE_DDM_MULTIPLICITY) {
      PetscCall(QPGetEqMultiplicityScaling(qp,&d,&ctx->dI));
    } else {
      SETERRQ(comm,PETSC_ERR_SUP,"-qp_E_scale_type %s not supported",QPScaleTypes[ScalType]);
    }

    PetscCall(QPTScale_Private(A,b,d,&DA,&Db));
    
    PetscCall(QPSetQPPF(child,NULL));
    PetscCall(QPSetEq(child,DA,Db));
    PetscCall(QPSetEqMultiplier(child,NULL));
    ctx->dE = d;

    PetscCall(MatDestroy(&DA));
    PetscCall(VecDestroy(&Db));
  }

  A = qp->BI;
  b = qp->cI;
  d = ctx->dI;
  ScalType = QP_SCALE_NONE;
  PetscCall(PetscOptionsEnum("-qp_I_scale_type", "", "QPSetIneqScaling", QPScaleTypes, (PetscEnum)ScalType, (PetscEnum*)&ScalType, &set));
  PetscCall(PetscInfo(qp, "-qp_I_scale_type %s\n",QPScaleTypes[ScalType]));
  if (ScalType || d) {
    if (ScalType && d) SETERRQ(comm,PETSC_ERR_SUP,"-qp_I_scale_type %s not supported for given eq. con. scaling",QPScaleTypes[ScalType]);

    if (ScalType == QP_SCALE_ROWS_NORM_2) {
      PetscCall(MatGetRowNormalization(A,&d));
    } else if (!d) {
      SETERRQ(comm,PETSC_ERR_SUP,"-qp_I_scale_type %s not supported",QPScaleTypes[ScalType]);
    }

    PetscCall(QPTScale_Private(A,b,d,&DA,&Db));

    PetscCall(QPSetIneq(child,DA,Db));
    PetscCall(QPSetIneqMultiplier(child,NULL));
    ctx->dI = d;

    PetscCall(MatDestroy(&DA));
    PetscCall(VecDestroy(&Db));
  }

  if (qp->R) {
    PetscCall(PetscOptionsEnum("-qp_R_orth_type", "type of nullspace matrix orthonormalization", "", MatOrthTypes, (PetscEnum)R_orth_type, (PetscEnum*)&R_orth_type, NULL));
    PetscCall(PetscOptionsEnum("-qp_R_orth_form", "form of nullspace matrix orthonormalization", "", MatOrthForms, (PetscEnum)R_orth_form, (PetscEnum*)&R_orth_form, NULL));
    PetscCall(PetscInfo(qp, "-qp_R_orth_type %s\n",MatOrthTypes[R_orth_type]));
    PetscCall(PetscInfo(qp, "-qp_R_orth_form %s\n",MatOrthForms[R_orth_form]));
    if (R_orth_type) {
      Mat Rnew;
      PetscCall(MatOrthColumns(qp->R, R_orth_type, R_orth_form, &Rnew, NULL));
      PetscCall(QPSetOperatorNullSpace(child,Rnew));
      PetscCall(MatDestroy(&Rnew));
    }
  }

  PetscOptionsEnd();
  child->postSolveCtx = ctx;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTNormalizeObjective"
PetscErrorCode QPTNormalizeObjective(QP qp)
{
  PetscReal norm_A, norm_b;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscCall(QPChainGetLast(qp,&qp));
  PetscCall(MatGetMaxEigenvalue(qp->A, NULL, &norm_A, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(VecNorm(qp->b,NORM_2,&norm_b));
  PetscCall(PetscInfo(qp,"||A||=%.12e, scale A by 1/||A||=%.12e\n",norm_A,1.0/norm_A));
  PetscCall(PetscInfo(qp,"||b||=%.12e, scale b by 1/||b||=%.12e\n",norm_b,1.0/norm_b));
  PetscCall(QPTScaleObjectiveByScalar(qp, 1.0/norm_A, 1.0/norm_b));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTNormalizeHessian"
PetscErrorCode QPTNormalizeHessian(QP qp)
{
  PetscReal norm_A;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscCall(QPChainGetLast(qp,&qp));
  PetscCall(MatGetMaxEigenvalue(qp->A, NULL, &norm_A, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(PetscInfo(qp,"||A||=%.12e, scale A by 1/||A||=%.12e\n",norm_A,1.0/norm_A));
  PetscCall(QPTScaleObjectiveByScalar(qp, 1.0/norm_A, 1.0/norm_A));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTScaleObjectiveByScalar"
static PetscErrorCode QPTPostSolve_QPTScaleObjectiveByScalar(QP child,QP parent)
{
  QPTScaleObjectiveByScalar_Ctx *psctx = (QPTScaleObjectiveByScalar_Ctx*)child->postSolveCtx;
  PetscReal scale_A = psctx->scale_A;
  PetscReal scale_b = psctx->scale_b;
  Vec lb,ub;
  Vec llb,lub;
  Vec llbnew,lubnew;
  QPC qpc,qpcc;

  PetscFunctionBegin;
  PetscCall(VecCopy(child->x,parent->x));
  PetscCall(VecScale(parent->x,scale_A/scale_b));
  
  if (parent->Bt_lambda) {
    PetscCall(VecCopy(child->Bt_lambda,parent->Bt_lambda));
    PetscCall(VecScale(parent->Bt_lambda,1.0/scale_b));
  }
  if (parent->lambda_E) {
    PetscCall(VecCopy(child->lambda_E,parent->lambda_E));
    PetscCall(VecScale(parent->lambda_E,1.0/scale_b));
  }
  PetscCall(QPGetBox(parent,NULL,&lb,&ub));
  PetscCall(QPGetQPC(parent, &qpc));
  PetscCall(QPGetQPC(child, &qpcc));
  PetscCall(QPCBoxGetMultipliers(qpcc,&llb,&lub));
  PetscCall(QPCBoxGetMultipliers(qpc,&llbnew,&lubnew));
  if (lb) {
    PetscCall(VecCopy(llb,llbnew));
    PetscCall(VecScale(llbnew,1.0/scale_b));
  }
  if (ub) {
    PetscCall(VecCopy(lub,lubnew));
    PetscCall(VecScale(lubnew,1.0/scale_b));
  }
  if (parent->BI) {
    PetscCall(VecCopy(child->lambda_I,parent->lambda_I));
    PetscCall(VecScale(parent->lambda_I,1.0/scale_b));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTScaleObjectiveByScalar"
static PetscErrorCode QPTPostSolveDestroy_QPTScaleObjectiveByScalar(void *ctx)
{
  PetscFunctionBegin;
  PetscFree(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTScaleObjectiveByScalar"
PetscErrorCode QPTScaleObjectiveByScalar(QP qp,PetscScalar scale_A,PetscScalar scale_b)
{
  MPI_Comm comm;
  QP child;
  QPTScaleObjectiveByScalar_Ctx *ctx;
  Mat Anew;
  Vec bnew;
  Vec lb,ub;
  Vec lbnew,ubnew;
  PetscReal norm_A,norm_b;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscCall(QPTransformBegin(QPTScaleObjectiveByScalar,
      QPTPostSolve_QPTScaleObjectiveByScalar, QPTPostSolveDestroy_QPTScaleObjectiveByScalar,
      QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));
  PetscCall(PetscNew(&ctx));
  ctx->scale_A = scale_A;
  ctx->scale_b = scale_b;
  child->postSolveCtx = ctx;

  if (FllopDebugEnabled) {
    PetscCall(MatGetMaxEigenvalue(qp->A, NULL, &norm_A, 1e-5, 50));
    PetscCall(FllopDebug1("||A||=%.12e\n",norm_A));
    PetscCall(VecNorm(qp->b,NORM_2,&norm_b));
    PetscCall(FllopDebug1("||b||=%.12e\n",norm_b));
  }

  if (qp->A->ops->duplicate) {
    PetscCall(MatDuplicate(qp->A,MAT_COPY_VALUES,&Anew));
  } else {
    PetscCall(MatCreateProd(comm,1,&qp->A,&Anew));
  }
  PetscCall(MatScale(Anew,scale_A));
  PetscCall(QPSetOperator(child,Anew));
  PetscCall(MatDestroy(&Anew));

  PetscCall(VecDuplicate(qp->b,&bnew));
  PetscCall(VecCopy(qp->b,bnew));
  PetscCall(VecScale(bnew,scale_b));
  PetscCall(QPSetRhs(child,bnew));
  PetscCall(VecDestroy(&bnew));

  PetscCall(QPGetBox(qp,NULL,&lb,&ub));
  lbnew=NULL;
  if (lb) {
    PetscCall(VecDuplicate(lb,&lbnew));
    PetscCall(VecCopy(lb,lbnew));
    PetscCall(VecScaleSkipInf(lbnew,scale_b/scale_A));
  }
  ubnew=NULL;
  if (ub) {
    PetscCall(VecDuplicate(ub,&ubnew));
    PetscCall(VecCopy(ub,ubnew));
    PetscCall(VecScaleSkipInf(ubnew,scale_b/scale_A));
  }
  PetscCall(QPSetBox(child,NULL,lbnew,ubnew));
  PetscCall(VecDestroy(&lbnew));
  PetscCall(VecDestroy(&ubnew));

  PetscCall(QPSetInitialVector(child,NULL));
  PetscCall(QPSetEqMultiplier(child,NULL));
  PetscCall(QPSetIneqMultiplier(child,NULL));

  if (FllopDebugEnabled) {
    PetscCall(MatGetMaxEigenvalue(child->A, NULL, &norm_A, 1e-5, 50));
    PetscCall(FllopDebug1("||A_new||=%.12e\n",norm_A));
    PetscCall(VecNorm(child->b,NORM_2,&norm_b));
    PetscCall(FllopDebug1("||b_new||=%.12e\n",norm_b));
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTFreezeIneq"
static PetscErrorCode QPTPostSolve_QPTFreezeIneq(QP child,QP parent)
{
  PetscInt Mn;
  IS *iss=NULL,is=NULL;
  Vec lambda_EI;

  PetscFunctionBegin;
  Mn = child->BE_nest_count;
  PERMON_ASSERT(Mn>=1,"child->BE_nest_count >= 1");

  if (Mn>1) {
    PetscCall(PetscMalloc1(Mn,&iss));
    PetscCall(MatNestGetISs(child->BE,iss,NULL));
    is = iss[Mn-1];

    /* copy the corresponding part of child's lambda_E to parent's lambda_I */
    PetscCall(VecGetSubVector(child->lambda_E,is,&lambda_EI));
    PetscCall(VecCopy(lambda_EI,parent->lambda_I));
    PetscCall(VecRestoreSubVector(child->lambda_E,is,&lambda_EI));

    /* copy the rest of child's lambda_E to parent's lambda_E */
    PetscCall(ISConcatenate(PetscObjectComm((PetscObject)child->BE),Mn-1,iss,&is));
    PetscCall(VecGetSubVector(child->lambda_E,is,&lambda_EI));
    PetscCall(VecCopy(lambda_EI,parent->lambda_E));
    PetscCall(VecRestoreSubVector(child->lambda_E,is,&lambda_EI));

    PetscCall(ISDestroy(&is));
    PetscCall(PetscFree(iss));
  } else { /* n==1 => there had been only ineq. con. before */
    PetscCall(VecCopy(child->lambda_E,parent->lambda_I));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFreezeIneq"
PetscErrorCode QPTFreezeIneq(QP qp)
{
  MPI_Comm          comm;
  QP                child;
  Mat               BI;
  Vec               cI;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscCall(QPTransformBegin(QPTFreezeIneq, QPTPostSolve_QPTFreezeIneq,NULL, QP_DUPLICATE_COPY_POINTERS,&qp,&child,&comm));
  PetscCall(QPGetIneq(qp,&BI,&cI));
  PetscCall(QPAddEq(child,BI,cI));
  PetscCall(QPSetEqMultiplier(child,NULL));
  PetscCall(QPSetIneq(child,NULL,NULL));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTSplitBE"
PetscErrorCode QPTSplitBE(QP qp)
{
  QP child;
  MPI_Comm comm;
  Mat Be=NULL, Bd=NULL, Bg=NULL;
  Mat Bet=NULL, Bdt=NULL, Bgt=NULL;
  PetscInt i, ilo, ihi, j, k, ncols, ng=0, nd=0;
  PetscInt *idxg, *idxd;
  const PetscInt *cols;
  const PetscScalar *vals;
  IS isrowg, isrowd;
  PetscBool flg;
  
  PetscFunctionBeginI;
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  PERMON_ASSERT(!qp->cE,"!qp->cE");
  PetscCall(MatIsImplicitTranspose(qp->BE, &flg));
  PERMON_ASSERT(flg,"BE is implicit transpose");
  
  PetscCall(PetscLogEventBegin(QPT_SplitBE,qp,0,0,0));
  PetscCall(QPTransformBegin(QPTSplitBE, NULL, NULL, QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));

  PetscCall(PermonMatTranspose(child->BE, MAT_TRANSPOSE_CHEAPEST, &Bet));
  PetscCall(PermonMatTranspose(Bet, MAT_TRANSPOSE_EXPLICIT, &Be));
  PetscCall(MatDestroy(&Bet));

  PetscCall(MatGetOwnershipRange(Be, &ilo, &ihi));

  PetscCall(PetscMalloc((ihi-ilo)*sizeof(PetscInt), &idxg));
  PetscCall(PetscMalloc((ihi-ilo)*sizeof(PetscInt), &idxd));

  for (i=ilo; i<ihi; i++){
    PetscCall(MatGetRow(Be, i, &ncols, &cols, &vals));
    k = 0;
    for (j=0; j<ncols; j++){
      if (vals[j]) k++;
    }
    if (k==1){
      idxd[nd] = i;
      nd += 1;
    }else if (k>1){
      idxg[ng] = i;
      ng += 1;
    }else{
      SETERRQ(comm, PETSC_ERR_COR, "B columns can't have every element zero");
    }
    PetscCall(MatRestoreRow(Be, i, &ncols, &cols, &vals));
  }
  PetscCall(ISCreateGeneral(comm, ng, idxg, PETSC_OWN_POINTER, &isrowg));
  PetscCall(ISCreateGeneral(comm, nd, idxd, PETSC_OWN_POINTER, &isrowd));
  
  PetscCall(MatCreateSubMatrix(Be, isrowg, NULL, MAT_INITIAL_MATRIX, &Bg));
  PetscCall(MatCreateSubMatrix(Be, isrowd, NULL, MAT_INITIAL_MATRIX, &Bd));
  PetscCall(MatDestroy(&Be));

  PetscCall(PermonMatTranspose(Bg, MAT_TRANSPOSE_EXPLICIT, &Bgt));
  PetscCall(MatDestroy(&Bg));
  PetscCall(PermonMatTranspose(Bgt, MAT_TRANSPOSE_CHEAPEST, &Bg));
  
  PetscCall(PermonMatTranspose(Bd, MAT_TRANSPOSE_EXPLICIT, &Bdt));
  PetscCall(MatDestroy(&Bd));
  PetscCall(PermonMatTranspose(Bdt, MAT_TRANSPOSE_CHEAPEST, &Bd));

  PetscCall(MatDestroy(&child->BE));
  PetscCall(QPAddEq(child, Bg, NULL));
  PetscCall(QPAddEq(child, Bd, NULL));
  
  PetscCall(PetscLogEventEnd(QPT_SplitBE,qp,0,0,0));

  PetscCall(ISDestroy(&isrowg));
  PetscCall(ISDestroy(&isrowd));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTMatISToBlockDiag"
static PetscErrorCode QPTPostSolve_QPTMatISToBlockDiag(QP child,QP parent)
{
  Mat AsubCopy;
  Vec dir,b_adjust,b_adjustU,resid;
  Mat_IS *matis  = (Mat_IS*)parent->A->data;
  QPTMatISToBlockDiag_Ctx *ctx = (QPTMatISToBlockDiag_Ctx*)child->postSolveCtx;
  PetscReal norm=0.0,normb=0.0;
  PetscBool computeNorm = PETSC_FALSE;
  PetscInt lock;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_matis_to_diag_norm",&computeNorm,NULL));
  if (ctx->isDir) {
    if (computeNorm) {
      /* TODO: change implementation for submat copies for PETSc>=3.8 */
      MatDuplicate(child->A,MAT_COPY_VALUES,&AsubCopy);
      //PetscCall(MatGetLocalToGlobalMapping(child->A,&l2g,NULL));
      //PetscCall(ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,ctx->isDir,&isDirLoc));
      //PetscCall(MatGetLocalSubMatrix(parent->A,isDirLoc,isDirLoc,&Asub));
      //PetscCall(MatDuplicate(Asub,MAT_COPY_VALUES,&AsubCopy));
      PetscCall(VecDuplicate(child->x,&dir));
      PetscCall(VecDuplicate(child->b,&b_adjustU));
      PetscCall(VecCopy(child->b,b_adjustU));
      PetscCall(VecGetLocalVector(dir,matis->y));
      PetscCall(VecScatterBegin(matis->cctx,parent->x,matis->y,INSERT_VALUES,SCATTER_FORWARD)); /* set local vec */
      PetscCall(VecScatterEnd(matis->cctx,parent->x,matis->y,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecRestoreLocalVector(dir,matis->y));
      PetscCall(MatZeroRowsColumnsIS(child->A,ctx->isDir,1.0,dir,b_adjustU));
    }
  } else {
    /* propagate changed RHS */
    /* TODO: flag for pure Neumann? */
    PetscCall(VecGetLocalVector(child->b,matis->y));
    PetscCall(VecLockGet(parent->b,&lock));
    if (lock) PetscCall(VecLockReadPop(parent->b)); /* TODO: safe? */
    PetscCall(VecSet(parent->b,0.0));
    PetscCall(VecScatterBegin(matis->rctx,matis->y,parent->b,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx,matis->y,parent->b,ADD_VALUES,SCATTER_REVERSE));
    if (lock) PetscCall(VecLockReadPush(parent->b));
    PetscCall(VecRestoreLocalVector(child->b,matis->y));
  }

  /* assemble solution */
  PetscCall(VecGetLocalVector(child->x,matis->x));
  PetscCall(VecScatterBegin(matis->rctx,matis->x,parent->x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(matis->rctx,matis->x,parent->x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecRestoreLocalVector(child->x,matis->x));
  
  if (computeNorm) {
    /* compute norm */
    PetscCall(VecDuplicate(parent->b,&resid));
    if (ctx->isDir) {
      PetscCall(VecDuplicate(parent->b,&b_adjust));
      PetscCall(VecSet(b_adjust,.0));
      PetscCall(VecGetLocalVector(b_adjustU,matis->y));
      PetscCall(VecScatterBegin(matis->rctx,matis->y,b_adjust,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(matis->rctx,matis->y,b_adjust,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecRestoreLocalVector(b_adjustU,matis->y));
      PetscCall(MatMult(parent->A,parent->x,resid));
      PetscCall(VecAXPY(resid,-1.0,b_adjust)); /* Ax-b */ 
      PetscCall(VecNorm(b_adjust,NORM_2,&normb));
      PetscCall(MatCopy(AsubCopy,child->A,SAME_NONZERO_PATTERN));
      //PetscCall(MatCopy(AsubCopy,Asub,SAME_NONZERO_PATTERN));
      PetscCall(VecDestroy(&b_adjustU));
      PetscCall(VecDestroy(&b_adjust));
    } else {
      PetscCall(MatMult(parent->A,parent->x,resid));
      PetscCall(VecAXPY(resid,-1.0,parent->b)); /* Ax-b */ 
      PetscCall(VecNorm(parent->b,NORM_2,&normb));
    } 
    PetscCall(VecNorm(resid,NORM_2,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Dirichlet in Hess: %d, r = ||Ax-b|| = %e, r/||b|| = %e\n",!ctx->isDir,norm,norm/normb));
    PetscCall(VecDestroy(&resid));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTMatISToBlockDiag"
static PetscErrorCode QPTPostSolveDestroy_QPTMatISToBlockDiag(void *ctx)
{
  QPTMatISToBlockDiag_Ctx *cctx = (QPTMatISToBlockDiag_Ctx*)ctx;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&cctx->isDir));
  PetscCall(PetscFree(cctx));
  PetscFunctionReturn(0);
}

/*@
   QPTMatISToBlockDiag - Transforms system matrix from MATIS format to BlockDiag.

   Collective on QP

   Input Parameter:
.  qp   - the QP

   Level: developer
@*/
#undef __FUNCT__
#define __FUNCT__ "QPTMatISToBlockDiag"
PetscErrorCode QPTMatISToBlockDiag(QP qp)
{
  QP child;
  Mat A;
  IS l2g,i2g;
  const PetscInt *idx_l2g;
  QPTMatISToBlockDiag_Ctx *ctx;
  ISLocalToGlobalMapping mapping;
  PetscInt n;         /* number of nodes (interior+interface) in this subdomain */
  PetscInt n_B;       /* number of interface nodes in this subdomain */
  PetscInt n_I;
  PetscInt n_neigh;   /* number of neighbours this subdomain has (by now, INCLUDING OR NOT the subdomain itself). */
                      /* Once this is definitively decided, the code can be simplifies and some if's eliminated.  */
  PetscInt *neigh;    /* list of neighbouring subdomains                                                          */
  PetscInt *n_shared; /* n_shared[j] is the number of nodes shared with subdomain neigh[j]                        */
  PetscInt **shared;  /* shared[j][i] is the local index of the i-th node shared with subdomain neigh[j]          */
  PetscInt *idx_I_local,*idx_B_local,*idx_I_global,*idx_B_global;
  PetscInt i,j;
  PetscBT  bt;
  IS is_B_local,is_I_local,is_B_global, is_I_global; /* local (seq) index sets for interface (B) and interior (I) nodes */
  VecScatter N_to_B;      /* scattering context from all local nodes to local interface nodes */
  VecScatter global_to_B; /* scattering context from global to local interface nodes */
  Vec b,D,vec1_B;
  MPI_Comm comm;
  
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  PetscCall(QPTransformBegin(QPTMatISToBlockDiag,QPTPostSolve_QPTMatISToBlockDiag,QPTPostSolveDestroy_QPTMatISToBlockDiag,QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  PetscCall(PetscNew(&ctx));
  child->postSolveCtx = ctx;
  ctx->isDir = NULL; /* set by QPFetiSetDirichlet() */
  PetscCall(PCDestroy(&child->pc));
  PetscCall(QPGetPC(child,&child->pc));

  /* create block diag */
  Mat_IS *matis  = (Mat_IS*)qp->A->data;
  PetscCall(MatCreateBlockDiag(comm,matis->A,&A));
  PetscCall(QPSetOperator(child,A));
  PetscCall(QPSetEq(child,qp->BE,NULL));
  PetscCall(QPSetOperatorNullSpace(child,qp->R));
  PetscCall(MatDestroy(&qp->BE));

  /* get mappings for RHS decomposition
  *  create interface mappings
  *  adapted from PCISSetUp */
  /* get info on mapping */
  mapping = matis->rmapping;
  PetscCall(ISLocalToGlobalMappingGetSize(mapping,&n));
  PetscCall(ISLocalToGlobalMappingGetInfo(mapping,&n_neigh,&neigh,&n_shared,&shared));

  /* Identifying interior and interface nodes, in local numbering */
  PetscCall(PetscBTCreate(n,&bt));
  for (i=0;i<n_neigh;i++) {
    for (j=0;j<n_shared[i];j++) {
        PetscCall(PetscBTSet(bt,shared[i][j]));
    }
  }
  /* Creating local and global index sets for interior and inteface nodes. */
  PetscCall(PetscMalloc1(n,&idx_I_local));
  PetscCall(PetscMalloc1(n,&idx_B_local));
  for (i=0, n_B=0, n_I=0; i<n; i++) {
    if (!PetscBTLookup(bt,i)) {
      idx_I_local[n_I] = i;
      n_I++;
    } else {
      idx_B_local[n_B] = i;
      n_B++;
    }
  }
  /* Getting the global numbering */
  idx_B_global = idx_I_local + n_I; /* Just avoiding allocating extra memory, since we have vacant space */
  idx_I_global = idx_B_local + n_B;
  PetscCall(ISLocalToGlobalMappingApply(mapping,n_B,idx_B_local,idx_B_global));
  PetscCall(ISLocalToGlobalMappingApply(mapping,n_I,idx_I_local,idx_I_global));

  /* Creating the index sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n_B,idx_B_local,PETSC_COPY_VALUES, &is_B_local));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n_B,idx_B_global,PETSC_COPY_VALUES,&is_B_global));
  /* TODO remove interior idx sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_local,PETSC_COPY_VALUES, &is_I_local));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_global,PETSC_COPY_VALUES,&is_I_global));

    /* Creating work vectors and arrays */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n_B,&vec1_B));
  PetscCall(VecDuplicate(vec1_B, &D));

  /* Creating the scatter contexts */
  PetscCall(VecScatterCreate(matis->x,is_B_local,vec1_B,NULL,&N_to_B));
  PetscCall(VecScatterCreate(qp->x,is_B_global,vec1_B,NULL,&global_to_B));

  /* Creating scaling "matrix" D */
  PetscCall(VecSet(D,1.0));
  PetscCall(VecScatterBegin(N_to_B,matis->counter,vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(N_to_B,matis->counter,vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecPointwiseDivide(D,D,vec1_B));

  /* decompose assembled vecs */
  PetscCall(MatCreateVecs(A,&child->x,&child->b));
  /* assemble b */
  PetscCall(VecGetLocalVector(child->b,matis->y));
  PetscCall(VecDuplicate(qp->b,&b));
  PetscCall(VecCopy(qp->b,b));
  PetscCall(VecScatterBegin(global_to_B,qp->b,vec1_B,INSERT_VALUES,SCATTER_FORWARD)); /* get interface DOFs */
  PetscCall(VecScatterEnd(global_to_B,qp->b,vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecPointwiseMult(vec1_B,D,vec1_B)); /* DOF/(number of subdomains it belongs to) */
  PetscCall(VecScatterBegin(global_to_B,vec1_B,b,INSERT_VALUES,SCATTER_REVERSE)); /* replace values in RHS */
  PetscCall(VecScatterEnd(global_to_B,vec1_B,b,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterBegin(matis->cctx,b,matis->y,INSERT_VALUES,SCATTER_FORWARD)); /* set local vec */
  PetscCall(VecScatterEnd(matis->cctx,b,matis->y,INSERT_VALUES,SCATTER_FORWARD));
  /* assemble x */
  PetscCall(VecGetLocalVector(child->x,matis->x));
  PetscCall(VecScatterBegin(matis->cctx,qp->x,matis->x,INSERT_VALUES,SCATTER_FORWARD)); /* set local vec */
  PetscCall(VecScatterEnd(matis->cctx,qp->x,matis->x,INSERT_VALUES,SCATTER_FORWARD));

  /* inherit l2g and i2g */
  PetscCall(ISLocalToGlobalMappingGetIndices(mapping,&idx_l2g));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)qp),n,idx_l2g,PETSC_COPY_VALUES,&l2g));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(mapping,&idx_l2g));
  PetscCall(QPFetiSetLocalToGlobalMapping(child,l2g));
  PetscCall(ISOnComm(is_B_global,PETSC_COMM_WORLD,PETSC_COPY_VALUES,&i2g));
  PetscCall(ISSort(i2g));
  PetscCall(QPFetiSetInterfaceToGlobalMapping(child,i2g));

  PetscCall(ISDestroy(&i2g));
  PetscCall(ISDestroy(&l2g));
  PetscCall(VecRestoreLocalVector(child->x,matis->x));
  PetscCall(VecRestoreLocalVector(child->b,matis->y));
  PetscCall(ISLocalToGlobalMappingRestoreInfo(mapping,&n_neigh,&neigh,&n_shared,&shared));
  PetscCall(ISDestroy(&is_B_local));
  PetscCall(ISDestroy(&is_B_global));
  PetscCall(ISDestroy(&is_I_local));
  PetscCall(ISDestroy(&is_I_global));
  PetscCall(VecScatterDestroy(&N_to_B));
  PetscCall(VecScatterDestroy(&global_to_B));
  PetscCall(PetscFree(idx_B_local));
  PetscCall(PetscFree(idx_I_local));
  PetscCall(PetscBTDestroy(&bt));
  PetscCall(VecDestroy(&D));
  PetscCall(VecDestroy(&vec1_B));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTAllInOne"
PetscErrorCode QPTAllInOne(QP qp,MatInvType invType,PetscBool dual,PetscBool project,PetscReal penalty,PetscBool penalty_direct,PetscBool regularize)
{
  MatRegularizationType regularize_e;
  PetscBool freeze=PETSC_FALSE, normalize=PETSC_FALSE, normalize_hessian=PETSC_FALSE;
  QP last;

  PetscFunctionBeginI;
  regularize_e = regularize ? MAT_REG_EXPLICIT : MAT_REG_NONE;

  PetscCall(PetscLogEventBegin(QPT_AllInOne,qp,0,0,0));
  PetscObjectOptionsBegin((PetscObject)qp);
  PetscCall(PetscOptionsBool("-qp_I_freeze","perform QPTFreezeIneq","QPTFreezeIneq",freeze,&freeze,NULL));
  PetscCall(PetscOptionsBoolGroupBegin("-qp_O_normalize","perform QPTNormalizeObjective","QPTNormalizeObjective",&normalize));
  PetscCall(PetscOptionsBoolGroupEnd("-qp_O_normalize_hessian","perform QPTNormalizeHessian","QPTNormalizeHessian",&normalize_hessian));
  PetscOptionsEnd();

  //TODO do this until QPTFromOptions supports chain updates
  PetscCall(QPRemoveChild(qp));

  if (normalize) {
    PetscCall(QPTNormalizeObjective(qp));
  } else if (normalize_hessian) {
    PetscCall(QPTNormalizeHessian(qp));
  }

  PetscCall(QPTScale(qp));
  PetscCall(QPTOrthonormalizeEqFromOptions(qp));
  if (freeze) PetscCall(QPTFreezeIneq(qp));
  if (dual) {
    PetscCall(QPTDualize(qp,invType,regularize_e));
    PetscCall(QPTScale(qp));
    PetscCall(QPTOrthonormalizeEqFromOptions(qp));
  }
  if (project) {
    PetscCall(QPTEnforceEqByProjector(qp));
  }
  
  if (dual || project) {
    normalize = PETSC_FALSE;
    normalize_hessian = PETSC_FALSE;
    PetscCall(QPChainGetLast(qp,&last));
    PetscObjectOptionsBegin((PetscObject)last);
    PetscCall(PetscOptionsBoolGroupBegin("-qp_O_normalize","perform QPTNormalizeObjective","QPTNormalizeObjective",&normalize));
    PetscCall(PetscOptionsBoolGroupEnd("-qp_O_normalize_hessian","perform QPTNormalizeHessian","QPTNormalizeHessian",&normalize_hessian));
    PetscOptionsEnd();
    if (normalize) {
      PetscCall(QPTNormalizeObjective(qp));
    } else if (normalize_hessian) {
      PetscCall(QPTNormalizeHessian(qp));
    }
  }

  PetscCall(QPTEnforceEqByPenalty(qp,penalty,penalty_direct));
  PetscCall(PetscLogEventEnd  (QPT_AllInOne,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFromOptions"
PetscErrorCode QPTFromOptions(QP qp)
{
  MatInvType invType=MAT_INV_MONOLITHIC;
  PetscBool ddm=PETSC_FALSE, dual=PETSC_FALSE, feti=PETSC_FALSE, project=PETSC_FALSE;
  PetscReal penalty=0.0;
  PetscBool penalty_direct=PETSC_FALSE;
  PetscBool regularize=PETSC_TRUE;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)qp),NULL,"QP transforms options","QP");
  {
    PetscCall(PetscOptionsBool("-feti","perform FETI DDM combination of transforms","QPTAllInOne",feti,&feti,NULL));
    if (feti) {
      ddm             = PETSC_TRUE;
      dual            = PETSC_TRUE;
      project         = PETSC_TRUE;
    }
    PetscCall(PetscOptionsReal("-penalty","QPTEnforceEqByPenalty penalty parameter","QPTEnforceEqByPenalty",penalty,&penalty,NULL));
    PetscCall(PetscOptionsBool("-penalty_direct","","QPTEnforceEqByPenalty",penalty_direct,&penalty_direct,NULL));
    PetscCall(PetscOptionsBool("-project","perform QPTEnforceEqByProjector","QPTEnforceEqByProjector",project,&project,NULL));
    PetscCall(PetscOptionsBool("-dual","perform QPTDualize","QPTDualize",dual,&dual,NULL));
    PetscCall(PetscOptionsBool("-ddm","domain decomposed data","QPTDualize",ddm,&ddm,NULL));
    PetscCall(PetscOptionsBool("-regularize","perform stiffness matrix regularization (for singular TFETI matrices)","QPTDualize",regularize,&regularize,NULL));
    if (ddm) {
      invType = MAT_INV_BLOCKDIAG;
    }
  }
  PetscOptionsEnd();
  PetscCall(QPTAllInOne(qp, invType, dual, project, penalty, penalty_direct, regularize));
  PetscFunctionReturn(0);
}
#undef QPTransformBegin
