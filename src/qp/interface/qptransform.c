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
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,comm));
  CHKERRQ(QPChainGetLast(qp,&qp));
  CHKERRQ(QPSetUpInnerObjects(qp));
  CHKERRQ(QPChainAdd(qp,opt,&child));
  child->transform = transform;
  CHKERRQ(PetscStrcpy(child->transform_name, trname));
  CHKERRQ(QPSetPC(child,qp->pc));
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));

  child->postSolve = QPDefaultPostSolve;
  if (postSolve) child->postSolve = postSolve;
  child->postSolveCtxDestroy = postSolveCtxDestroy;

  CHKERRQ(FllopPetscObjectInheritPrefix((PetscObject)child,(PetscObject)qp,NULL));

  *qp_inout = qp;
  *child_new = child;
  CHKERRQ(PetscInfo(qp,"QP %x (#%d) transformed by %s to QP %x (#%d)\n",qp,qp->id,trname,child,child->id));
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
  CHKERRQ(VecCopy(child->x,parent->x));
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
  CHKERRQ(QPDefaultPostSolve(child,parent));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_project_inherit_eq_multipliers",&inherit_eq_multipliers,NULL));

  if (child->BE && inherit_eq_multipliers) {
    CHKERRQ(VecIsInvalidated(child->lambda_E,&skip_lambda_E));
    CHKERRQ(VecIsInvalidated(child->Bt_lambda,&skip_Bt_lambda));
  }

  CHKERRQ(MatMult(A, x, r));                                                      /* r = A*x */
  CHKERRQ(VecAYPX(r,-1.0,b));                                                     /* r = b - r */

  if (!skip_lambda_E) {
    /* lambda_E1 = lambda_E2 + (B*B')\B*(b-A*x) */
    CHKERRQ(QPPFApplyHalfQ(parent->pf,r,lambda_E));
    CHKERRQ(VecAXPY(lambda_E,1.0,child->lambda_E));
  }
  if (!skip_Bt_lambda) {
    /* (B'*lambda)_1 = (B'*lambda)_2 + B'*(B*B')\B*(b-A*x) */
    CHKERRQ(QPPFApplyQ(parent->pf,r,Bt_lambda) );
    CHKERRQ(VecAXPY(Bt_lambda,1.0,child->Bt_lambda));
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
  CHKERRQ(PCShellGetContext(pc,(void**)&ctx));
  CHKERRQ(MatDestroy(&ctx->P));
  CHKERRQ(PCDestroy(&ctx->pc));
  CHKERRQ(VecDestroy(&ctx->work));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_QPTEnforceEqByProjector"
static PetscErrorCode PCApply_QPTEnforceEqByProjector(PC pc,Vec x,Vec y)
{
  PC_QPTEnforceEqByProjector* ctx;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,(void**)&ctx));
  CHKERRQ(PCApply(ctx->pc,x,ctx->work));
  CHKERRQ(MatMult(ctx->P,ctx->work,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_QPTEnforceEqByProjector_Symmetric"
static PetscErrorCode PCApply_QPTEnforceEqByProjector_Symmetric(PC pc,Vec x,Vec y)
{
  PC_QPTEnforceEqByProjector* ctx;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,(void**)&ctx));
  CHKERRQ(MatMult(ctx->P,x,y));
  CHKERRQ(PCApply(ctx->pc,y,ctx->work));
  CHKERRQ(MatMult(ctx->P,ctx->work,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_QPTEnforceEqByProjector_None"
static PetscErrorCode PCApply_QPTEnforceEqByProjector_None(PC pc,Vec x,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(VecCopy(x,y));
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
  CHKERRQ(PCShellGetContext(pc,(void**)&ctx));

  none = PETSC_FALSE;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ctx->pc,PCNONE,&none));
  if (!none) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ctx->pc,PCDUAL,&flg));
    if (flg) {
      CHKERRQ(PCDualGetType(ctx->pc,&type));
      if (type == PC_DUAL_NONE) none = PETSC_TRUE;
    }
  }

  if (none) {
    CHKERRQ(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector_None));
  } else if (ctx->symmetric) {
    CHKERRQ(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector_Symmetric));
  } else {
    CHKERRQ(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector));
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
  CHKERRQ(PetscNew(&ctx));
  ctx->symmetric = symmetric;
  ctx->P = P;
  ctx->pc = pc_orig;
  CHKERRQ(PetscObjectReference((PetscObject)P));
  CHKERRQ(PetscObjectReference((PetscObject)pc_orig));
  CHKERRQ(MatCreateVecs(P,&ctx->work,NULL));

  CHKERRQ(PCCreate(PetscObjectComm((PetscObject)pc_orig),&pc));
  CHKERRQ(PCSetType(pc,PCSHELL));
  CHKERRQ(PCShellSetName(pc,"QPTEnforceEqByProjector"));
  CHKERRQ(PCShellSetContext(pc,ctx));
  CHKERRQ(PCShellSetApply(pc,PCApply_QPTEnforceEqByProjector));
  CHKERRQ(PCShellSetDestroy(pc,PCDestroy_QPTEnforceEqByProjector));
  CHKERRQ(PCShellSetSetUp(pc,PCSetUp_QPTEnforceEqByProjector));

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

  CHKERRQ(QPChainGetLast(qp,&qp));
  if (!qp->BE) {
    CHKERRQ(PetscInfo(qp, "no lin. eq. con. matrix specified ==> nothing to enforce\n"));
    CHKERRQ(VecDestroy(&qp->cE));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  if (qp->cE) {
    CHKERRQ(PetscInfo(qp, "nonzero lin. eq. con. RHS prescribed ==> automatically calling QPTHomogenizeEq\n"));
    CHKERRQ(QPTHomogenizeEq(qp));
    CHKERRQ(QPChainGetLast(qp,&qp));
  }

  CHKERRQ(PetscLogEventBegin(QPT_EnforceEqByProjector,qp,0,0,0));
  CHKERRQ(QPTransformBegin(QPTEnforceEqByProjector, QPTEnforceEqByProjectorPostSolve_Private,NULL, QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  CHKERRQ(QPAppendOptionsPrefix(child,"proj_"));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_project_pc_symmetric",&pc_symmetric,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_project_inherit_box_multipliers",&inherit_box_multipliers,NULL));

  eqonly = PetscNot(qp->BI || qp->qpc);
  if (eqonly) {
    CHKERRQ(PetscInfo(qp, "only lin. eq. con. were prescribed ==> they are now eliminated\n"));
    CHKERRQ(QPSetEq(  child, NULL, NULL));
  } else {
    CHKERRQ(PetscInfo(qp, "NOT only lin. eq. con. prescribed ==> lin. eq. con. are NOT eliminated\n"));
    CHKERRQ(QPSetQPPF(child, qp->pf));
    CHKERRQ(QPSetEq(  child, qp->BE, qp->cE));
  }
  CHKERRQ(QPSetIneq(child, qp->BI, qp->cI));
  CHKERRQ(QPSetRhs( child, qp->b));

  if (inherit_box_multipliers) {
    CHKERRQ(QPSetQPC(child, qp->qpc));
  } else {
    /* TODO: generalize for other QPC */
    Vec lb,ub;
    IS is;
    CHKERRQ(QPGetBox(qp,&is,&lb,&ub));
    CHKERRQ(QPSetBox(child,is,lb,ub));
  }

  CHKERRQ(QPPFCreateP(qp->pf,&P));
  if (eqonly) {
    /* newA = P*A */
    A_arr[0]=qp->A; A_arr[1]=P;
    CHKERRQ(MatCreateProd(comm,2,A_arr,&newA));
  } else {
    /* newA = P*A*P */
    A_arr[0]=P; A_arr[1]=qp->A; A_arr[2]=P;
    CHKERRQ(MatCreateProd(comm,3,A_arr,&newA));
  }
  CHKERRQ(QPSetOperator(child,newA));
  CHKERRQ(QPSetOperatorNullSpace(child,qp->R));
  CHKERRQ(MatDestroy(&newA));

  /* newb = P*b */
  CHKERRQ(VecDuplicate(qp->b, &newb));
  CHKERRQ(MatMult(P, qp->b, newb));
  if (FllopDebugEnabled) {
    PetscReal norm1, norm2;
    CHKERRQ(VecNorm(qp->b, NORM_2, &norm1));
    CHKERRQ(VecNorm(newb, NORM_2, &norm2));
    CHKERRQ(FllopDebug2("\n    ||b||\t= %.12e  b = b_bar\n    ||Pb||\t= %.12e\n",norm1,norm2));
  }
  CHKERRQ(QPSetRhs(child, newb));
  CHKERRQ(VecDestroy(&newb));

  /* create special preconditioner pc_child = P * pc_parent */
  {
    PC pc_parent,pc_child;

    CHKERRQ(QPGetPC(qp,&pc_parent));
    CHKERRQ(PCCreate_QPTEnforceEqByProjector(pc_parent,P,pc_symmetric,&pc_child));
    CHKERRQ(QPSetPC(child,pc_child));
    CHKERRQ(PCDestroy(&pc_child));
  }

  CHKERRQ(QPSetWorkVector(child,qp->xwork));

  CHKERRQ(MatDestroy(&P));
  CHKERRQ(PetscLogEventEnd(QPT_EnforceEqByProjector,qp,0,0,0));
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
    CHKERRQ(PetscInfo(qp, "penalty=0.0 ==> no effect, returning...\n"));
    PetscFunctionReturn(0);
  }

  CHKERRQ(QPChainGetLast(qp,&qp));

  if (!qp->BE) {
    CHKERRQ(PetscInfo(qp, "no lin. eq. con. matrix specified ==> nothing to enforce\n"));
    CHKERRQ(VecDestroy(&qp->cE));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  if (qp->cE) {
    PetscBool flg=PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_homogenize_eq_always",&flg,NULL));
    if (flg) {
      CHKERRQ(PetscInfo(qp, "nonzero lin. eq. con. RHS prescribed and -qpt_homogenize_eq_always set to true ==> automatically calling QPTHomogenizeEq\n"));
      CHKERRQ(QPTHomogenizeEq(qp));
      CHKERRQ(QPChainGetLast(qp,&qp));
    }
  }

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-qpt_penalize_maxeig_tol",&maxeig_tol,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-qpt_penalize_maxeig_iter",&maxeig_iter,NULL));

  if (!rho_direct) {
    CHKERRQ(MatGetMaxEigenvalue(qp->A, NULL, &maxeig, maxeig_tol, maxeig_iter));
    rho = rho_user * maxeig;
  } else {
    rho = rho_user;
  }

  if (rho < 0) {
    SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_WRONG,"rho must be nonnegative");
  }
  CHKERRQ(PetscInfo(qp, "using penalty = real %.12e\n", rho));

  CHKERRQ(PetscLogEventBegin(QPT_EnforceEqByPenalty,qp,0,0,0));
  CHKERRQ(QPTransformBegin(QPTEnforceEqByPenalty, QPTEnforceEqByPenalty_PostSolve_Private,NULL, QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  CHKERRQ(QPAppendOptionsPrefix(child,"pnlt_"));

  CHKERRQ(QPSetEq(  child, NULL, NULL));
  CHKERRQ(QPSetQPC(child, qp->qpc));
  CHKERRQ(QPSetIneq(child, qp->BI, qp->cI));
  CHKERRQ(QPSetRhs(child, qp->b));
  CHKERRQ(QPSetInitialVector(child, qp->x));

  /* newA = A + rho*BE'*BE */
  CHKERRQ(MatCreatePenalized(qp,rho,&newA));
  CHKERRQ(QPSetOperator(child,newA));
  CHKERRQ(QPSetOperatorNullSpace(child,qp->R));
  CHKERRQ(MatDestroy(&newA));

  /* newb = b + rho*BE'*c */
  if (qp->c) {
    CHKERRQ(VecDuplicate(qp->b,&newb));
    CHKERRQ(MatMultTranspose(qp->BE,qp->c,newb));
    CHKERRQ(VecAYPX(newb,rho,qp->b));
    CHKERRQ(QPSetRhs(child,newb));
  }
  
  CHKERRQ(QPSetIneqMultiplier(       child,qp->lambda_I));
  CHKERRQ(QPSetWorkVector(child,qp->xwork));

  CHKERRQ(PetscLogEventEnd(QPT_EnforceEqByPenalty,qp,0,0,0));
  PetscFunctionReturnI(0);
}


#undef __FUNCT__
#define __FUNCT__ "QPTHomogenizeEqPostSolve_Private"
static PetscErrorCode QPTHomogenizeEqPostSolve_Private(QP child,QP parent)
{
  Vec xtilde = (Vec) child->postSolveCtx;

  PetscFunctionBegin;
  /* x_parent = x_child + xtilde */
  CHKERRQ(VecWAXPY(parent->x,1.0,child->x,xtilde));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTHomogenizeEqPostSolveCtxDestroy_Private"
static PetscErrorCode QPTHomogenizeEqPostSolveCtxDestroy_Private(void *ctx)
{
  Vec xtilde = (Vec) ctx;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&xtilde));
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

  CHKERRQ(QPChainGetLast(qp,&qp));
  if (!qp->cE) {
    CHKERRQ(PetscInfo(qp, "lin. eq. con. already homogenous ==> nothing to homogenize\n"));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  CHKERRQ(PetscLogEventBegin(QPT_HomogenizeEq,qp,0,0,0));
  CHKERRQ(QPTransformBegin(QPTHomogenizeEq, QPTHomogenizeEqPostSolve_Private,QPTHomogenizeEqPostSolveCtxDestroy_Private, QP_DUPLICATE_COPY_POINTERS,&qp,&child,&comm));

  /* A, R remain the same */

  CHKERRQ(VecDuplicate(qp->x,&xtilde));
  CHKERRQ(QPPFApplyHalfQTranspose(qp->pf,qp->cE,xtilde));                         /* xtilde = BE'*inv(BE*BE')*cE */

  CHKERRQ(VecDuplicate(qp->b, &b_bar));
  CHKERRQ(MatMult(qp->A, xtilde, b_bar));
  CHKERRQ(VecAYPX(b_bar, -1.0, qp->b));                                           /* b_bar = b - A*xtilde */
  CHKERRQ(QPSetRhs(child, b_bar));
  CHKERRQ(VecDestroy(&b_bar));

  if (FllopDebugEnabled) {
    PetscReal norm1,norm2,norm3,norm4;
    CHKERRQ(VecNorm(qp->cE, NORM_2, &norm1));
    CHKERRQ(VecNorm(xtilde, NORM_2, &norm2));
    CHKERRQ(VecNorm(qp->b,  NORM_2, &norm3));
    CHKERRQ(VecNorm(child->b,NORM_2, &norm4));
    CHKERRQ(FllopDebug4("\n"
        "    ||ceq||\t= %.12e  ceq = e\n"
        "    ||xtilde||\t= %.12e  xtilde = Beq'*inv(Beq*Beq')*ceq\n"
        "    ||b||\t= %.12e  b = d\n"
        "    ||b_bar||\t= %.12e  b_bar = b-A*xtilde\n",norm1,norm2,norm3,norm4));
  }

  CHKERRQ(QPSetQPPF(child, qp->pf));
  CHKERRQ(QPSetEq(child,qp->BE,NULL));                                            /* cE is eliminated */

  cineq = NULL;
  if (qp->cI) {
    CHKERRQ(VecDuplicate(qp->cI, &cineq));
    CHKERRQ(MatMult(qp->BI, xtilde, cineq));
    CHKERRQ(VecAYPX(cineq, -1.0, qp->cI));                                        /* cI = cI - BI*xtilde */
  }
  CHKERRQ(QPSetIneq(child, qp->BI, cineq));
  CHKERRQ(VecDestroy(&cineq));

  CHKERRQ(QPGetBox(qp, &is, &lb, &ub));
  if (is) {
    CHKERRQ(VecGetSubVector(xtilde, is, &xtilde_sub));
  } else {
    xtilde_sub = xtilde;
  }

  if (lb) {
    CHKERRQ(VecDuplicate(lb, &lbnew));
    CHKERRQ(VecWAXPY(lbnew, -1.0, xtilde_sub, lb));                                /* lb = lb - xtilde */
  }

  if (ub) {
    CHKERRQ(VecDuplicate(ub, &ubnew));
    CHKERRQ(VecWAXPY(ubnew, -1.0, xtilde_sub, ub));                                /* ub = ub - xtilde */
  }

  if (is) {
    CHKERRQ(VecRestoreSubVector(xtilde, is, &xtilde_sub));
  }

  CHKERRQ(QPSetBox(child, is, lbnew, ubnew));
  CHKERRQ(VecDestroy(&lbnew));
  CHKERRQ(VecDestroy(&ubnew));

  CHKERRQ(VecDestroy(&child->x));

  child->postSolveCtx = xtilde;
  CHKERRQ(PetscLogEventEnd(QPT_HomogenizeEq,qp,0,0,0));
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

  //CHKERRQ(VecIsInvalidated(child->lambda_E,&skip_lambda_E));
  //CHKERRQ(VecIsInvalidated(child->Bt_lambda,&skip_Bt_lambda));

  if (!skip_lambda_E && T->ops->multtranspose) {
    /* lambda_E_p = T'*lambda_E_c */
    CHKERRQ(MatMultTranspose(T,child->lambda_E,parent->lambda_E));
  } else if (!skip_Bt_lambda) {
    //TODO this seems to be inaccurate for explicit orthonormalization
    /* lambda_E_p = (G*G')\G(G'*lambda_c) where G'*lambda_c = child->Bt_lambda */
    CHKERRQ(QPPFApplyHalfQ(child->pf,child->Bt_lambda,parent->lambda_E));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTOrthonormalizeEq"
static PetscErrorCode QPTPostSolveDestroy_QPTOrthonormalizeEq(void *ctx)
{
  Mat T = (Mat) ctx;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&T));
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

  CHKERRQ(QPChainGetLast(qp,&qp));
  if (!qp->BE) {
    CHKERRQ(PetscInfo(qp, "no lin. eq. con. ==> nothing to orthonormalize\n"));
    PetscFunctionReturn(0);
  }
  if (type == MAT_ORTH_NONE) {
    CHKERRQ(PetscInfo(qp, "MAT_ORTH_NONE ==> skipping orthonormalization\n"));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  if (type == MAT_ORTH_INEXACT && qp->cE) {
    CHKERRQ(PetscInfo(qp, "MAT_ORTH_INEXACT and nonzero lin. eq. con. RHS prescribed ==> automatically calling QPTHomogenizeEq\n"));
    CHKERRQ(QPTHomogenizeEq(qp));
    CHKERRQ(QPChainGetLast(qp,&qp));
  }

  CHKERRQ(PetscLogEventBegin(QPT_OrthonormalizeEq,qp,0,0,0));
  CHKERRQ(QPTransformBegin(QPTOrthonormalizeEq, QPTPostSolve_QPTOrthonormalizeEq,QPTPostSolveDestroy_QPTOrthonormalizeEq, QP_DUPLICATE_COPY_POINTERS,&qp,&child,&comm));

  CHKERRQ(QPGetEq(qp, &BE, &cE));
  CHKERRQ(MatOrthRows(BE, type, form, &TBE, &T));

  {
    const char *name;
    CHKERRQ(PetscObjectGetName((PetscObject)BE,&name));
    CHKERRQ(PetscObjectSetName((PetscObject)TBE,name));
    CHKERRQ(MatPrintInfo(T));
    CHKERRQ(MatPrintInfo(TBE));
  }

  TcE=NULL;
  if (cE) {
    if (type == MAT_ORTH_IMPLICIT || type == MAT_ORTH_INEXACT) {
      CHKERRQ(PetscObjectReference((PetscObject)cE));
      TcE = cE;
    } else {
      CHKERRQ(VecDuplicate(cE,&TcE));
      CHKERRQ(MatMult(T,cE,TcE));
    }
  }

  CHKERRQ(QPSetQPPF(child,NULL));
  CHKERRQ(QPSetEq(child,TBE,TcE));                /* QPPF re-created in QPSetEq */
  CHKERRQ(QPSetEqMultiplier(child,NULL));         /* lambda_E will be re-created in QPSetUp */

  if (type == MAT_ORTH_IMPLICIT) {
    /* inherit GGtinv to avoid extra GGt factorization - dirty way */
    CHKERRQ(QPPFSetUp(qp->pf));
    child->pf->GGtinv = qp->pf->GGtinv;
    child->pf->Gt = qp->pf->Gt;
    CHKERRQ(PetscObjectReference((PetscObject)qp->pf->GGtinv));
    CHKERRQ(PetscObjectReference((PetscObject)qp->pf->Gt));
    CHKERRQ(QPPFSetUp(child->pf));
  } else if (type == MAT_ORTH_INEXACT) {
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)child->pf,"inexact_"));
    CHKERRQ(PetscObjectCompose((PetscObject)child->pf,"exact",(PetscObject)qp->pf));
  }

  CHKERRQ(MatDestroy(&TBE));
  CHKERRQ(VecDestroy(&TcE));
  child->postSolveCtx = T;
  CHKERRQ(PetscLogEventEnd(QPT_OrthonormalizeEq,qp,0,0,0));
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
  CHKERRQ(QPChainGetLast(qp,&last));
  PetscObjectOptionsBegin((PetscObject)last);
  CHKERRQ(PetscOptionsEnum("-qp_E_orth_type","type of eq. con. orthonormalization","QPTOrthonormalizeEq",MatOrthTypes,(PetscEnum)eq_orth_type,(PetscEnum*)&eq_orth_type,NULL));
  CHKERRQ(PetscOptionsEnum("-qp_E_orth_form","form of eq. con. orthonormalization","QPTOrthonormalizeEq",MatOrthForms,(PetscEnum)eq_orth_form,(PetscEnum*)&eq_orth_form,NULL));
  CHKERRQ(PetscInfo(qp, "-qp_E_orth_type %s\n",MatOrthTypes[eq_orth_type]));
  CHKERRQ(PetscInfo(qp, "-qp_E_orth_form %s\n",MatOrthForms[eq_orth_form]));
  CHKERRQ(QPTOrthonormalizeEq(last,eq_orth_type,eq_orth_form));
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

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)B,&flg,MATNEST,MATNESTPERMON,""));
  if (flg) {
    CHKERRQ(MatNestGetSize(B,&Mn,NULL));
    for (j=0; j<Mn; j++) {
      CHKERRQ(MatNestGetSubMat(B,j,0,&Bn));
      CHKERRQ(QPTDualizeViewBSpectra_Private(Bn));
    }
  }
  CHKERRQ(PetscObjectGetName((PetscObject)B,&name));
  CHKERRQ(MatCreateTranspose(B,&Bt));
  CHKERRQ(MatCreateNormal(Bt,&BBt));
  CHKERRQ(MatGetMaxEigenvalue(BBt,NULL,&norm,1e-6,10));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)B),"||%s * %s'|| = %.12e (10 power method iterations)\n",name,name,norm));
  CHKERRQ(MatDestroy(&BBt));
  CHKERRQ(MatDestroy(&Bt));
  PetscFunctionReturn(0);
}

#define QPTDualizeView_Private_SetName(mat,matname) if (mat && !((PetscObject)mat)->name) CHKERRQ(PetscObjectSetName((PetscObject)mat,matname) )

#undef __FUNCT__
#define __FUNCT__ "QPTDualizeView_Private"
static PetscErrorCode QPTDualizeView_Private(QP qp, QP child)
{
  MPI_Comm comm;
  Mat F,Kplus,K,Kreg,B,Bt,R,G;
  Vec c,d,e,f,lb,lambda,u;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)qp, &comm));

  F = child->A;
  CHKERRQ(PetscObjectQuery((PetscObject)F,"Kplus",(PetscObject*)&Kplus));
  CHKERRQ(PetscObjectQuery((PetscObject)F,"B",(PetscObject*)&B));
  CHKERRQ(PetscObjectQuery((PetscObject)F,"Bt",(PetscObject*)&Bt));
  
  K = qp->A;
  R = qp->R;
  G = child->BE;
  c = qp->c;
  d = child->b;
  e = child->cE;
  f = qp->b;
  CHKERRQ(QPGetBox(child,NULL,&lb,NULL));
  lambda = child->x;
  u = qp->x;

  if (Kplus) {
    Mat Kplus_inner;
    CHKERRQ(PetscObjectQuery((PetscObject)Kplus,"Kplus",(PetscObject*)&Kplus_inner));
    if (Kplus_inner) Kplus = Kplus_inner;
    CHKERRQ(MatInvGetRegularizedMat(Kplus,&Kreg));
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
    CHKERRQ(PetscPrintf(comm, "*** %s:\n",__FUNCT__));
    if (K)      CHKERRQ(MatPrintInfo(K));
    if (Kreg)   CHKERRQ(MatPrintInfo(Kreg));
    if (Kplus)  CHKERRQ(MatPrintInfo(Kplus));
    if (R)      CHKERRQ(MatPrintInfo(R));
    if (u)      CHKERRQ(VecPrintInfo(u));
    if (f)      CHKERRQ(VecPrintInfo(f));
    if (B)      CHKERRQ(MatPrintInfo(B));
    if (Bt)     CHKERRQ(MatPrintInfo(Bt));
    if (c)      CHKERRQ(VecPrintInfo(c));
    if (F)      CHKERRQ(MatPrintInfo(F));
    if (lambda) CHKERRQ(VecPrintInfo(lambda));
    if (d)      CHKERRQ(VecPrintInfo(d));
    if (G)      CHKERRQ(MatPrintInfo(G));
    if (e)      CHKERRQ(VecPrintInfo(e));
    if (lb)     CHKERRQ(VecPrintInfo(lb));

    CHKERRQ(PetscPrintf(comm, "***\n\n"));
  }

  if (FllopDebugEnabled) {
    PetscReal norm1=0.0, norm2=0.0, norm3=0.0, norm4=0.0;
    if (f) CHKERRQ(VecNorm(f, NORM_2, &norm1));
    if (c) CHKERRQ(VecNorm(c, NORM_2, &norm2));
    if (d) CHKERRQ(VecNorm(d, NORM_2, &norm3));
    if (e) CHKERRQ(VecNorm(e, NORM_2, &norm4));
    CHKERRQ(FllopDebug4("\n"
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
    CHKERRQ(PetscObjectQuery((PetscObject)F,"Kplus",(PetscObject*)&Kplus));
    PERMON_ASSERT(Kplus,"Kplus != NULL");

    /* copy lambda back to lambda_E and lambda_I */
    if (parent->BE && parent->BI) {
      PetscInt i;
      IS rows[2];
      Vec as[2]={parent->lambda_E,parent->lambda_I};
      Vec a[2];

      CHKERRQ(MatNestGetISs(parent->B,rows,NULL));
      for (i=0; i<2; i++) {
        if (as[i]) {
          CHKERRQ(VecGetSubVector(parent->lambda,rows[i],&a[i]));
          CHKERRQ(VecCopy(a[i],as[i]));
          CHKERRQ(VecRestoreSubVector(parent->lambda,rows[i],&a[i]));
        }
      }
    }

    /* u = Kplus*(f-B'*lambda) */
    CHKERRQ(MatMultTranspose(parent->B, parent->lambda, tprim));
    CHKERRQ(VecAYPX(tprim, -1.0, f));
    CHKERRQ(MatMult(Kplus, tprim, u));

    if (alpha) {
      CHKERRQ(VecIsInvalidated(alpha,&flg));
      if (flg) {
        /* compute alpha = (G*G')\G(G'*alpha) where G'*alpha = child->Bt_lambda */
        CHKERRQ(VecIsInvalidated(child->Bt_lambda,&flg));
        if (flg) {
          CHKERRQ(MatMultTranspose(child->B,child->lambda,child->Bt_lambda));
        }
        CHKERRQ(QPPFApplyHalfQ(child->pf,child->Bt_lambda,alpha));
      }

      /* u = u - R*alpha */
      CHKERRQ(MatMult(parent->R, alpha, tprim));
      CHKERRQ(VecAXPY(u, -1.0, tprim));
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
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_G_explicit",&G_explicit,NULL));

  if (!G_explicit) {
    //NOTE doesn't work with redundant GGt_inv & redundancy of R is problematic due to its structure
    Mat G_arr[3];
    Mat Rt;

    CHKERRQ(PermonMatTranspose(R,MAT_TRANSPOSE_CHEAPEST,&Rt));

    CHKERRQ(MatCreateTimer(Rt,&G_arr[1]));
    CHKERRQ(MatCreateTimer(Bt,&G_arr[0]));
    
    CHKERRQ(MatCreateProd(PetscObjectComm((PetscObject)Bt), 2, G_arr, &G));
    CHKERRQ(MatCreateTimer(G,&G_arr[2]));
    CHKERRQ(MatDestroy(&G));
    G = G_arr[2];

    CHKERRQ(PetscObjectCompose((PetscObject)G,"Bt",(PetscObject)Bt));
    CHKERRQ(PetscObjectCompose((PetscObject)G,"R",(PetscObject)R));
    CHKERRQ(PetscObjectSetName((PetscObject)G,"G"));

    CHKERRQ(MatDestroy(&G_arr[1]));
    CHKERRQ(MatDestroy(&G_arr[0]));
    CHKERRQ(MatDestroy(&Rt));
    *G_new = G;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)Bt,&flg,MATNEST,MATNESTPERMON,""));
  if (!flg) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)Bt,MATEXTENSION,&flg));
    //CHKERRQ(MatTransposeMatMultWorks(R,Bt,&flg));
    if (flg) {
      CHKERRQ(MatTransposeMatMult(R,Bt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&G));
    } else {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)Bt), "WARNING: MatTransposeMatMult not applicable, falling back to MatMatMultByColumns\n"));
      CHKERRQ(MatTransposeMatMultByColumns(Bt,R,PETSC_TRUE,&Gt));
      CHKERRQ(PermonMatTranspose(Gt,MAT_TRANSPOSE_CHEAPEST,&G));
      CHKERRQ(MatDestroy(&Gt));
    }
  } else {
    //TODO make this MATNESTPERMON's method
    PetscInt Mn,Nn,J;
    Mat BtJ;
    Mat *mats_G;
    CHKERRQ(MatNestGetSize(Bt,&Mn,&Nn));
    PERMON_ASSERT(Mn==1,"Mn==1");
    CHKERRQ(PetscMalloc(Nn*sizeof(Mat),&mats_G));
    for (J=0; J<Nn; J++) {
      CHKERRQ(MatNestGetSubMat(Bt,0,J,&BtJ));
      CHKERRQ(MatTransposeMatMult_R_Bt(R,BtJ,&mats_G[J]));
    }
    CHKERRQ(MatCreateNestPermon(PetscObjectComm((PetscObject)Bt),1,NULL,Nn,NULL,mats_G,&G));
    for (J=0; J<Nn; J++) CHKERRQ(MatDestroy(&mats_G[J]));
    CHKERRQ(PetscFree(mats_G));
  }

  CHKERRQ(PetscObjectSetName((PetscObject)G,"G"));
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
  PetscBool        spd;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscLogEventBegin(QPT_Dualize,qp,0,0,0));
  CHKERRQ(QPTransformBegin(QPTDualize, QPTDualizePostSolve_Private,NULL, QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  CHKERRQ(QPAppendOptionsPrefix(child,"dual_"));

  Kplus_orig = NULL;
  B = NULL;
  c = qp->c;
  lambda = qp->lambda;
  tprim = qp->xwork;
  K = qp->A;
  f = qp->b;
  if (!qp->B) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"lin. equality and/or inequality constraint matrix (BE/BI) is needed for dualization");

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_explicit",&B_explicit,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_extension",&B_extension,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_nest_extension",&B_nest_extension,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_B_view_spectra",&B_view_spectra,NULL));

  if (B_view_spectra) {
    CHKERRQ(QPTDualizeViewBSpectra_Private(qp->B));
  }

  CHKERRQ(MatPrintInfo(qp->B));

  CHKERRQ(PetscLogEventBegin(QPT_Dualize_PrepareBt,qp,0,0,0));
  if (B_extension || B_nest_extension) {
    Mat B_merged;
    MatTransposeType ttype = B_explicit ? MAT_TRANSPOSE_EXPLICIT : MAT_TRANSPOSE_CHEAPEST;
    CHKERRQ(MatCreateNestPermonVerticalMerge(comm,1,&qp->B,&B_merged));
    CHKERRQ(PermonMatTranspose(B_merged,MAT_TRANSPOSE_EXPLICIT,&Bt));
    CHKERRQ(MatDestroy(&B_merged));
    if (B_extension) {
      CHKERRQ(MatConvert(Bt,MATEXTENSION,MAT_INPLACE_MATRIX,&Bt));
    } else {
      CHKERRQ(PermonMatConvertBlocks(Bt,MATEXTENSION,MAT_INPLACE_MATRIX,&Bt));
    }
    CHKERRQ(PermonMatTranspose(Bt,ttype,&B));
  } else {
    CHKERRQ(PermonMatTranspose(qp->B,MAT_TRANSPOSE_CHEAPEST,&Bt));
    if (B_explicit) {
      CHKERRQ(PermonMatTranspose(Bt,MAT_TRANSPOSE_EXPLICIT,&B));
    } else {
      /* in this case B remains the same */
      B = qp->B;
      CHKERRQ(PetscObjectReference((PetscObject)B));
    }
  }
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)B,(PetscObject)qp->B,NULL));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)Bt,(PetscObject)qp->B,"_T"));
  CHKERRQ(PetscLogEventEnd(  QPT_Dualize_PrepareBt,qp,0,0,0));

  if (FllopObjectInfoEnabled) {
    CHKERRQ(PetscPrintf(comm, "B and Bt after conversion:\n"));
    CHKERRQ(MatPrintInfo(B));
    CHKERRQ(MatPrintInfo(Bt));
  }

  /* create stiffness matrix pseudoinverse */
  CHKERRQ(MatCreateInv(K, invType, &Kplus));
  Kplus_orig = Kplus;
  CHKERRQ(MatPrintInfo(K));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)Kplus,(PetscObject)K,"_plus"));
  CHKERRQ(FllopPetscObjectInheritPrefix((PetscObject)Kplus,(PetscObject)child,NULL));

  /* get or compute stiffness matrix kernel (R) */
  R = NULL;
  CHKERRQ(QPGetOperatorNullSpace(qp,&R));
  CHKERRQ(MatGetOption(qp->A,MAT_SPD,&spd));
  if (R) {
    CHKERRQ(MatInvSetNullSpace(Kplus,R));
    //TODO consider not inheriting R with 0 cols
  } else if (spd) {
    CHKERRQ(PetscInfo(qp,"Hessian flagged SPD => not computing null space\n"));
  } else {
    PetscInt Ncols;
    CHKERRQ(PetscInfo(qp,"null space matrix not set => trying to compute one\n"));
    CHKERRQ(MatInvComputeNullSpace(Kplus));
    CHKERRQ(MatInvGetNullSpace(Kplus,&R));
    CHKERRQ(MatGetSize(R,NULL,&Ncols));
    if (Ncols) {
      CHKERRQ(MatOrthColumns(R, MAT_ORTH_GS, MAT_ORTH_FORM_EXPLICIT, &R, NULL));
      CHKERRQ(QPSetOperatorNullSpace(qp,R));
      CHKERRQ(PetscInfo(qp,"computed null space matrix => using -qpt_dualize_Kplus_left and -regularize 0\n"));
      mp = PETSC_TRUE;
      true_mp = PETSC_FALSE;
      regType = MAT_REG_NONE;
    } else {
      CHKERRQ(PetscInfo(qp,"computed null space matrix has 0 columns => consider setting SPD flag to Hessian to avoid null space detection\n"));
      R = NULL;
    }
  }
  CHKERRQ(MatInvSetRegularizationType(Kplus,regType));
  CHKERRQ(MatSetFromOptions(Kplus));

  CHKERRQ(PetscLogEventBegin(QPT_Dualize_FactorK,qp,Kplus,0,0));
  CHKERRQ(MatInvSetUp(Kplus));
  CHKERRQ(PetscLogEventEnd  (QPT_Dualize_FactorK,qp,Kplus,0,0));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_Kplus_mp",&true_mp,NULL));
  if (!true_mp) {
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_Kplus_left",&mp,NULL));
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
        CHKERRQ(PetscInfo(qp,"creating Moore-Penrose inverse\n"));
      } else {
        CHKERRQ(PetscInfo(qp,"creating left generalized inverse\n"));
      }
      CHKERRQ(PermonMatTranspose(R,MAT_TRANSPOSE_CHEAPEST,&Rt));
      CHKERRQ(PetscObjectSetName((PetscObject)Rt,"Rt"));
      CHKERRQ(QPPFCreate(comm,&pf_R));
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)pf_R,"Kplus_"));
      CHKERRQ(QPPFSetG(pf_R,Rt));
      CHKERRQ(QPPFCreateP(pf_R,&P_R));
      CHKERRQ(QPPFSetUp(pf_R));

      if (true_mp) {
        size=3;
        mats[2]=P_R;
      }
      mats[1]=Kplus; mats[0]=P_R;
      CHKERRQ(MatCreateProd(comm,size,mats,&Kplus_new));
      CHKERRQ(PetscObjectCompose((PetscObject)Kplus_new,"Kplus",(PetscObject)Kplus));

      CHKERRQ(MatDestroy(&Kplus));
      CHKERRQ(MatDestroy(&Rt));
      CHKERRQ(MatDestroy(&P_R));
      CHKERRQ(QPPFDestroy(&pf_R));
      Kplus = Kplus_new;

      if (FllopDebugEnabled) {
        /* is Kplus MP? */
        Mat mats2[3];
        Mat prod;
        PetscBool flg;
        mats2[2]=K; mats2[1]=Kplus; mats2[0]=K;
        CHKERRQ(MatCreateProd(comm,3,mats2,&prod));
        CHKERRQ(MatMultEqual(prod,K,3,&flg));
        PERMON_ASSERT(flg,"Kplus is left generalized inverse");
        CHKERRQ(MatDestroy(&prod));
        if (true_mp) {
          mats2[2]=Kplus; mats2[1]=K; mats2[0]=Kplus;
          CHKERRQ(MatCreateProd(comm,3,mats2,&prod));
          CHKERRQ(MatMultEqual(prod,Kplus,3,&flg));
          PERMON_ASSERT(flg,"Kplus is Moore-Penrose pseudoinverse");
          CHKERRQ(MatDestroy(&prod));
        }
      }
    } else {
        CHKERRQ(PetscInfo(qp,"ignoring requested left generalized or Moore-Penrose inverse, because null space is not set\n"));
    }
  }

  CHKERRQ(PetscObjectSetName((PetscObject)Kplus,"Kplus"));

  G = NULL;
  e = NULL;
  if (R) {
    /* G = R'*B' */
    CHKERRQ(PetscLogEventBegin(QPT_Dualize_AssembleG,qp,0,0,0));
    CHKERRQ(MatTransposeMatMult_R_Bt(R,Bt,&G));
    CHKERRQ(PetscLogEventEnd(  QPT_Dualize_AssembleG,qp,0,0,0));

    /* e = R'*f */
    CHKERRQ(MatCreateVecs(R,&e,NULL));
    CHKERRQ(MatMultTranspose(R,f,e));
  }

  /* F = B*Kplus*Bt (implicitly) */
  {
    Mat F_arr[4];
    
    CHKERRQ(MatCreateTimer(B,&F_arr[2]));
    CHKERRQ(MatCreateTimer(Kplus,&F_arr[1]));
    CHKERRQ(MatCreateTimer(Bt,&F_arr[0]));
  
    CHKERRQ(MatCreateProd(comm, 3, F_arr, &F));
    CHKERRQ(PetscObjectSetName((PetscObject) F, "F"));
    CHKERRQ(MatCreateTimer(F,&F_arr[3]));
    CHKERRQ(MatDestroy(&F));
    F = F_arr[3];
    
    CHKERRQ(PetscObjectCompose((PetscObject)F,"B",(PetscObject)B));
    CHKERRQ(PetscObjectCompose((PetscObject)F,"K",(PetscObject)K));
    CHKERRQ(PetscObjectCompose((PetscObject)F,"Kplus",(PetscObject)Kplus));
    CHKERRQ(PetscObjectCompose((PetscObject)F,"Kplus_orig",(PetscObject)Kplus_orig));
    CHKERRQ(PetscObjectCompose((PetscObject)F,"Bt",(PetscObject)Bt));
    
    CHKERRQ(MatDestroy(&B));       B     = F_arr[2];
    CHKERRQ(MatDestroy(&Kplus));   Kplus = F_arr[1];
    CHKERRQ(MatDestroy(&Bt));      Bt    = F_arr[0];
  }

  /* d = B*Kplus*f - c */
  CHKERRQ(VecDuplicate(lambda,&d));
  CHKERRQ(MatMult(Kplus, f, tprim));
  CHKERRQ(MatMult(B, tprim, d));
  if(c) CHKERRQ(VecAXPY(d,-1.0,c));

  /* lb(E) = -inf; lb(I) = 0 */
  if (qp->BI) {
    CHKERRQ(VecDuplicate(lambda,&lb));
    if (qp->BE) {
      IS rows[2];
      Vec lbE,lbI;

      CHKERRQ(MatNestGetISs(qp->B,rows,NULL));

      /* lb(E) = -inf */
      CHKERRQ(VecGetSubVector(lb,rows[0],&lbE));
      CHKERRQ(VecSet(lbE,PETSC_NINFINITY));
      CHKERRQ(VecRestoreSubVector(lb,rows[0],&lbE));

      /* lb(I) = 0 */
      CHKERRQ(VecGetSubVector(lb,rows[1],&lbI));
      CHKERRQ(VecSet(lbI,0.0));
      CHKERRQ(VecRestoreSubVector(lb,rows[1],&lbI));

      CHKERRQ(PetscObjectCompose((PetscObject)lb,"is_finite",(PetscObject)rows[1]));
    } else {
      /* lb = 0 */
      CHKERRQ(VecSet(lb,0.0));
    }
  } else {
    lb = NULL;
  }

  //TODO what if initial x is nonzero
  /* lambda = o */
  CHKERRQ(VecZeroEntries(lambda));

  /* set data of the new aux. QP */
  CHKERRQ(QPSetOperator(child, F));
  CHKERRQ(QPSetRhs(child, d));
  CHKERRQ(QPSetEq(child, G, e));
  CHKERRQ(QPSetIneq(child, NULL, NULL));
  CHKERRQ(QPSetBox(child, NULL, lb, NULL));
  CHKERRQ(QPSetInitialVector(child,lambda));
  
  /* create special preconditioner for dual formulation */
  {
    CHKERRQ(PCDestroy(&child->pc));
    CHKERRQ(QPGetPC(child,&child->pc));
    CHKERRQ(PCSetType(child->pc,PCDUAL));
    CHKERRQ(FllopPetscObjectInheritPrefix((PetscObject)child->pc,(PetscObject)child,NULL));
  }

  CHKERRQ(PetscLogEventEnd(QPT_Dualize,qp,0,0,0));

  CHKERRQ(QPTDualizeView_Private(qp,child));

  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&Bt));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&G));
  CHKERRQ(MatDestroy(&Kplus));
  CHKERRQ(VecDestroy(&d));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(VecDestroy(&lb));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFetiPrepare"
PetscErrorCode QPTFetiPrepare(QP qp,PetscBool regularize)
{
  PetscFunctionBeginI;
  CHKERRQ(PetscLogEventBegin(QPT_FetiPrepare,qp,0,0,0));
  CHKERRQ(QPTDualize(qp, MAT_INV_BLOCKDIAG, regularize ? MAT_REG_EXPLICIT : MAT_REG_NONE));
  CHKERRQ(QPTHomogenizeEq(qp));
  CHKERRQ(QPTEnforceEqByProjector(qp));
  CHKERRQ(PetscLogEventEnd  (QPT_FetiPrepare,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFetiPrepareReuseCP"
PetscErrorCode QPTFetiPrepareReuseCP(QP qp,PetscBool regularize)
{
  QP dualQP;

  PetscFunctionBeginI;
  CHKERRQ(QPTDualize(qp, MAT_INV_BLOCKDIAG, regularize ? MAT_REG_EXPLICIT : MAT_REG_NONE));

  CHKERRQ(QPChainGetLast(qp, &dualQP));
  if (QPReusedCP) {
    Mat G;
    Vec e;

    /* reuse the coarse problem from the 0th iteration */
    CHKERRQ(QPSetQPPF(dualQP, QPReusedCP));
    CHKERRQ(QPPFGetG(QPReusedCP, &G));
    CHKERRQ(QPGetEq(dualQP, NULL, &e));
    CHKERRQ(QPSetEq(dualQP, G, e));
  } else {
    /* store the coarse problem from the 0th iteration */
    CHKERRQ(QPGetQPPF(dualQP, &QPReusedCP));
    CHKERRQ(QPPFSetUp(QPReusedCP));
    CHKERRQ(PetscObjectReference((PetscObject) QPReusedCP));
  }

  CHKERRQ(QPTHomogenizeEq(qp));
  CHKERRQ(QPTEnforceEqByProjector(qp));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTFetiPrepareReuseCPReset"
PetscErrorCode QPTFetiPrepareReuseCPReset()
{
  PetscFunctionBegin;
  CHKERRQ(QPPFDestroy(&QPReusedCP));
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
  CHKERRQ(PetscMalloc4(Mn,&iss_parent,Mn,&iss_child,Mn,&lambda_parent,Mn,&lambda_child));

  CHKERRQ(MatNestGetISs(parent->BE,iss_parent,NULL));
  CHKERRQ(MatNestGetISs(child->BE,iss_child,NULL));

  /* 0,1 corresponds to gluing and Dirichlet part, respectively */
  for (II=0; II<Mn; II++) {
    CHKERRQ(VecGetSubVector(parent->lambda_E,iss_parent[II],&lambda_parent[II]));
    CHKERRQ(VecGetSubVector(child->lambda_E, iss_child[II], &lambda_child[II]));
  }

#if 0
  {
    PetscInt start, end;
    IS is_removed;

    CHKERRQ(VecGetOwnershipRange(lambda_parent[0],&start,&end));
    CHKERRQ(ISComplement(is,start,end,&is_removed));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "#### "__FUNCT__": is_removed:\n"));
    CHKERRQ(ISView(is_removed,PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n\n"));
  }
#endif

  /* copy values from the multiplier related to the restricted Bg to the multiplier
     corresponding to the original Bg, pad with zeros */
  CHKERRQ(VecZeroEntries(lambda_parent[0]));
  //TODO PETSc 3.5+ VecGetSubVector,VecRestoreSubVector
  {
    VecScatter sc;
    CHKERRQ(VecScatterCreate(lambda_child[0],NULL,lambda_parent[0],is,&sc));
    CHKERRQ(VecScatterBegin(sc,lambda_child[0],lambda_parent[0],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(  sc,lambda_child[0],lambda_parent[0],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&sc));
  }

  /* Bd is the same, just copy values */
  for (II=1; II<Mn; II++) {
    CHKERRQ(VecCopy(lambda_child[II],lambda_parent[II]));
  }

  for (II=0; II<Mn; II++) {
    CHKERRQ(VecRestoreSubVector(parent->lambda_E,iss_parent[II],&lambda_parent[II]));
    CHKERRQ(VecRestoreSubVector(child->lambda_E, iss_child[II], &lambda_child[II]));
  }

  CHKERRQ(PetscFree4(iss_parent,iss_child,lambda_parent,lambda_child));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTRemoveGluingOfDirichletDofs"
static PetscErrorCode QPTPostSolveDestroy_QPTRemoveGluingOfDirichletDofs(void *ctx)
{
  IS is = (IS) ctx;

  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&is));
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
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)qp->BE,&flg,MATNEST,MATNESTPERMON,""));
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_SUP,"only for eq. con. matrix qp->BE of type MATNEST or MATNESTPERMON");

  CHKERRQ(PetscLogEventBegin(QPT_RemoveGluingOfDirichletDofs,qp,0,0,0));
  CHKERRQ(MatNestGetSize(qp->BE,&Mn,&Nn));
  CHKERRQ(MatNestGetSubMats(qp->BE,&Mn,&Nn,&mats));
  PERMON_ASSERT(Mn>=2,"Mn==2");
  PERMON_ASSERT(Nn==1,"Nn==1");
  PERMON_ASSERT(!qp->cE,"!qp->cE");
  Bg = mats[0][0];
  Bd = mats[1][0];

  CHKERRQ(QPTransformBegin(QPTRemoveGluingOfDirichletDofs,
      QPTPostSolve_QPTRemoveGluingOfDirichletDofs, QPTPostSolveDestroy_QPTRemoveGluingOfDirichletDofs,
      QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));
  CHKERRQ(VecDestroy(&child->lambda_E));
  CHKERRQ(MatDestroy(&child->B));
  CHKERRQ(MatDestroy(&child->BE));
  child->BE_nest_count = 0;

  CHKERRQ(PermonMatTranspose(Bg,MAT_TRANSPOSE_CHEAPEST,&Bgt));
  CHKERRQ(PermonMatTranspose(Bd,MAT_TRANSPOSE_CHEAPEST,&Bdt));

  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_remove_gluing_dirichlet_old",&flg,NULL));
  if (flg) {
    CHKERRQ(MatRemoveGluingOfDirichletDofs_old(Bgt,NULL,Bdt,&Bgt_new,NULL,&is));
  } else {
    CHKERRQ(MatRemoveGluingOfDirichletDofs(Bgt,NULL,Bdt,&Bgt_new,NULL,&is));
  }

  CHKERRQ(MatDestroy(&Bdt));

  CHKERRQ(PermonMatTranspose(Bgt_new,MAT_TRANSPOSE_CHEAPEST,&Bg_new));
  CHKERRQ(QPAddEq(child,Bg_new,NULL));
  for (II=1; II<Mn; II++) {
    CHKERRQ(QPAddEq(child,mats[II][0],NULL));
  }
  CHKERRQ(MatDestroy(&Bgt_new));

  //TODO use VecGetSubVector with PETSc 3.5+
  CHKERRQ(MatCreateVecs(child->BE,NULL,&child->lambda_E));
  CHKERRQ(VecInvalidate(child->lambda_E));
  child->postSolveCtx = (void*) is;

  CHKERRQ(MatPrintInfo(Bg));
  CHKERRQ(VecPrintInfo(qp->lambda_E));
  CHKERRQ(MatPrintInfo(Bg_new));
  CHKERRQ(VecPrintInfo(child->lambda_E));

  CHKERRQ(MatDestroy(&Bg_new));
  CHKERRQ(PetscLogEventEnd(QPT_RemoveGluingOfDirichletDofs,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolve_QPTScale"
static PetscErrorCode QPTPostSolve_QPTScale(QP child,QP parent)
{
  QPTScale_Ctx *ctx = (QPTScale_Ctx*) child->postSolveCtx;

  PetscFunctionBegin;
  if (ctx->dE) {
    CHKERRQ(VecPointwiseMult(parent->lambda_E,ctx->dE,child->lambda_E));
  }
  if (ctx->dI) {
    CHKERRQ(VecPointwiseMult(parent->lambda_I,ctx->dI,child->lambda_I));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPTPostSolveDestroy_QPTScale"
static PetscErrorCode QPTPostSolveDestroy_QPTScale(void *ctx)
{
  QPTScale_Ctx *cctx = (QPTScale_Ctx*) ctx;  

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&cctx->dE));
  CHKERRQ(VecDestroy(&cctx->dI));
  CHKERRQ(VecDestroy(&cctx->dO));
  CHKERRQ(PetscFree(cctx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTScale_Private"
PetscErrorCode QPTScale_Private(Mat A,Vec b,Vec d,Mat *DA,Vec *Db)
{
  PetscFunctionBegin;
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,DA));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)*DA,(PetscObject)A,NULL));
  CHKERRQ(MatDiagonalScale(*DA,d,NULL));
  
  if (b) {
    CHKERRQ(VecDuplicate(b,Db));
    CHKERRQ(VecPointwiseMult(*Db,d,b));
    CHKERRQ(FllopPetscObjectInheritName((PetscObject)*Db,(PetscObject)b,NULL));
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
  CHKERRQ(QPChainGetLast(qp,&qp));

  PetscObjectOptionsBegin((PetscObject)qp);
  CHKERRQ(PetscOptionsBool("-qp_E_remove_gluing_of_dirichlet","remove gluing of DOFs on Dirichlet boundary","QPTRemoveGluingOfDirichletDofs",remove_gluing_of_dirichlet,&remove_gluing_of_dirichlet,NULL));
  PetscOptionsEnd();
  CHKERRQ(PetscInfo(qp, "-qp_E_remove_gluing_of_dirichlet %d\n",remove_gluing_of_dirichlet));
  if (remove_gluing_of_dirichlet) {
    CHKERRQ(QPTRemoveGluingOfDirichletDofs(qp));
  }

  CHKERRQ(QPTransformBegin(QPTScale,
      QPTPostSolve_QPTScale, QPTPostSolveDestroy_QPTScale,
      QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));
  CHKERRQ(PetscNew(&ctx));

  PetscObjectOptionsBegin((PetscObject)qp);
  A = qp->A;
  b = qp->b;
  d = NULL;
  ScalType = QP_SCALE_NONE;
  CHKERRQ(PetscOptionsEnum("-qp_O_scale_type", "", "QPSetSystemScaling", QPScaleTypes, (PetscEnum)ScalType, (PetscEnum*)&ScalType, &set));
  CHKERRQ(PetscInfo(qp, "-qp_O_scale_type %s\n",QPScaleTypes[ScalType]));
  if (ScalType) {
    if (ScalType == QP_SCALE_ROWS_NORM_2) {
      CHKERRQ(MatGetRowNormalization(A,&d));
    } else {
      SETERRQ(comm,PETSC_ERR_SUP,"-qp_O_scale_type %s not supported",QPScaleTypes[ScalType]);
    }

    CHKERRQ(QPTScale_Private(A,b,d,&DA,&Db));
    
    CHKERRQ(QPSetOperator(child,DA));
    CHKERRQ(QPSetRhs(child,Db));
    ctx->dO = d;

    CHKERRQ(MatDestroy(&DA));
    CHKERRQ(VecDestroy(&Db));
  }

  A = qp->BE;
  b = qp->cE;
  d = NULL;
  ScalType = QP_SCALE_NONE;
  CHKERRQ(PetscOptionsEnum("-qp_E_scale_type", "", "QPSetEqScaling", QPScaleTypes, (PetscEnum)ScalType, (PetscEnum*)&ScalType, &set));
  CHKERRQ(PetscInfo(qp, "-qp_E_scale_type %s\n",QPScaleTypes[ScalType]));
  if (ScalType) {
    if (ScalType == QP_SCALE_ROWS_NORM_2) {
      CHKERRQ(MatGetRowNormalization(A,&d));
    } else if (ScalType == QP_SCALE_DDM_MULTIPLICITY) {
      CHKERRQ(QPGetEqMultiplicityScaling(qp,&d,&ctx->dI));
    } else {
      SETERRQ(comm,PETSC_ERR_SUP,"-qp_E_scale_type %s not supported",QPScaleTypes[ScalType]);
    }

    CHKERRQ(QPTScale_Private(A,b,d,&DA,&Db));
    
    CHKERRQ(QPSetQPPF(child,NULL));
    CHKERRQ(QPSetEq(child,DA,Db));
    CHKERRQ(QPSetEqMultiplier(child,NULL));
    ctx->dE = d;

    CHKERRQ(MatDestroy(&DA));
    CHKERRQ(VecDestroy(&Db));
  }

  A = qp->BI;
  b = qp->cI;
  d = ctx->dI;
  ScalType = QP_SCALE_NONE;
  CHKERRQ(PetscOptionsEnum("-qp_I_scale_type", "", "QPSetIneqScaling", QPScaleTypes, (PetscEnum)ScalType, (PetscEnum*)&ScalType, &set));
  CHKERRQ(PetscInfo(qp, "-qp_I_scale_type %s\n",QPScaleTypes[ScalType]));
  if (ScalType || d) {
    if (ScalType && d) SETERRQ(comm,PETSC_ERR_SUP,"-qp_I_scale_type %s not supported for given eq. con. scaling",QPScaleTypes[ScalType]);

    if (ScalType == QP_SCALE_ROWS_NORM_2) {
      CHKERRQ(MatGetRowNormalization(A,&d));
    } else if (!d) {
      SETERRQ(comm,PETSC_ERR_SUP,"-qp_I_scale_type %s not supported",QPScaleTypes[ScalType]);
    }

    CHKERRQ(QPTScale_Private(A,b,d,&DA,&Db));

    CHKERRQ(QPSetIneq(child,DA,Db));
    CHKERRQ(QPSetIneqMultiplier(child,NULL));
    ctx->dI = d;

    CHKERRQ(MatDestroy(&DA));
    CHKERRQ(VecDestroy(&Db));
  }

  if (qp->R) {
    CHKERRQ(PetscOptionsEnum("-qp_R_orth_type", "type of nullspace matrix orthonormalization", "", MatOrthTypes, (PetscEnum)R_orth_type, (PetscEnum*)&R_orth_type, NULL));
    CHKERRQ(PetscOptionsEnum("-qp_R_orth_form", "form of nullspace matrix orthonormalization", "", MatOrthForms, (PetscEnum)R_orth_form, (PetscEnum*)&R_orth_form, NULL));
    CHKERRQ(PetscInfo(qp, "-qp_R_orth_type %s\n",MatOrthTypes[R_orth_type]));
    CHKERRQ(PetscInfo(qp, "-qp_R_orth_form %s\n",MatOrthForms[R_orth_form]));
    if (R_orth_type) {
      Mat Rnew;
      CHKERRQ(MatOrthColumns(qp->R, R_orth_type, R_orth_form, &Rnew, NULL));
      CHKERRQ(QPSetOperatorNullSpace(child,Rnew));
      CHKERRQ(MatDestroy(&Rnew));
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
  CHKERRQ(QPChainGetLast(qp,&qp));
  CHKERRQ(MatGetMaxEigenvalue(qp->A, NULL, &norm_A, PETSC_DECIDE, PETSC_DECIDE));
  CHKERRQ(VecNorm(qp->b,NORM_2,&norm_b));
  CHKERRQ(PetscInfo(qp,"||A||=%.12e, scale A by 1/||A||=%.12e\n",norm_A,1.0/norm_A));
  CHKERRQ(PetscInfo(qp,"||b||=%.12e, scale b by 1/||b||=%.12e\n",norm_b,1.0/norm_b));
  CHKERRQ(QPTScaleObjectiveByScalar(qp, 1.0/norm_A, 1.0/norm_b));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTNormalizeHessian"
PetscErrorCode QPTNormalizeHessian(QP qp)
{
  PetscReal norm_A;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(QPChainGetLast(qp,&qp));
  CHKERRQ(MatGetMaxEigenvalue(qp->A, NULL, &norm_A, PETSC_DECIDE, PETSC_DECIDE));
  CHKERRQ(PetscInfo(qp,"||A||=%.12e, scale A by 1/||A||=%.12e\n",norm_A,1.0/norm_A));
  CHKERRQ(QPTScaleObjectiveByScalar(qp, 1.0/norm_A, 1.0/norm_A));
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
  CHKERRQ(VecCopy(child->x,parent->x));
  CHKERRQ(VecScale(parent->x,scale_A/scale_b));
  
  if (parent->Bt_lambda) {
    CHKERRQ(VecCopy(child->Bt_lambda,parent->Bt_lambda));
    CHKERRQ(VecScale(parent->Bt_lambda,1.0/scale_b));
  }
  if (parent->lambda_E) {
    CHKERRQ(VecCopy(child->lambda_E,parent->lambda_E));
    CHKERRQ(VecScale(parent->lambda_E,1.0/scale_b));
  }
  CHKERRQ(QPGetBox(parent,NULL,&lb,&ub));
  CHKERRQ(QPGetQPC(parent, &qpc));
  CHKERRQ(QPGetQPC(child, &qpcc));
  CHKERRQ(QPCBoxGetMultipliers(qpcc,&llb,&lub));
  CHKERRQ(QPCBoxGetMultipliers(qpc,&llbnew,&lubnew));
  if (lb) {
    CHKERRQ(VecCopy(llb,llbnew));
    CHKERRQ(VecScale(llbnew,1.0/scale_b));
  }
  if (ub) {
    CHKERRQ(VecCopy(lub,lubnew));
    CHKERRQ(VecScale(lubnew,1.0/scale_b));
  }
  if (parent->BI) {
    CHKERRQ(VecCopy(child->lambda_I,parent->lambda_I));
    CHKERRQ(VecScale(parent->lambda_I,1.0/scale_b));
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
  CHKERRQ(QPTransformBegin(QPTScaleObjectiveByScalar,
      QPTPostSolve_QPTScaleObjectiveByScalar, QPTPostSolveDestroy_QPTScaleObjectiveByScalar,
      QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));
  CHKERRQ(PetscNew(&ctx));
  ctx->scale_A = scale_A;
  ctx->scale_b = scale_b;
  child->postSolveCtx = ctx;

  if (FllopDebugEnabled) {
    CHKERRQ(MatGetMaxEigenvalue(qp->A, NULL, &norm_A, 1e-5, 50));
    CHKERRQ(FllopDebug1("||A||=%.12e\n",norm_A));
    CHKERRQ(VecNorm(qp->b,NORM_2,&norm_b));
    CHKERRQ(FllopDebug1("||b||=%.12e\n",norm_b));
  }

  if (qp->A->ops->duplicate) {
    CHKERRQ(MatDuplicate(qp->A,MAT_COPY_VALUES,&Anew));
  } else {
    CHKERRQ(MatCreateProd(comm,1,&qp->A,&Anew));
  }
  CHKERRQ(MatScale(Anew,scale_A));
  CHKERRQ(QPSetOperator(child,Anew));
  CHKERRQ(MatDestroy(&Anew));

  CHKERRQ(VecDuplicate(qp->b,&bnew));
  CHKERRQ(VecCopy(qp->b,bnew));
  CHKERRQ(VecScale(bnew,scale_b));
  CHKERRQ(QPSetRhs(child,bnew));
  CHKERRQ(VecDestroy(&bnew));

  CHKERRQ(QPGetBox(qp,NULL,&lb,&ub));
  lbnew=NULL;
  if (lb) {
    CHKERRQ(VecDuplicate(lb,&lbnew));
    CHKERRQ(VecCopy(lb,lbnew));
    CHKERRQ(VecScaleSkipInf(lbnew,scale_b/scale_A));
  }
  ubnew=NULL;
  if (ub) {
    CHKERRQ(VecDuplicate(ub,&ubnew));
    CHKERRQ(VecCopy(ub,ubnew));
    CHKERRQ(VecScaleSkipInf(ubnew,scale_b/scale_A));
  }
  CHKERRQ(QPSetBox(child,NULL,lbnew,ubnew));
  CHKERRQ(VecDestroy(&lbnew));
  CHKERRQ(VecDestroy(&ubnew));

  CHKERRQ(QPSetInitialVector(child,NULL));
  CHKERRQ(QPSetEqMultiplier(child,NULL));
  CHKERRQ(QPSetIneqMultiplier(child,NULL));

  if (FllopDebugEnabled) {
    CHKERRQ(MatGetMaxEigenvalue(child->A, NULL, &norm_A, 1e-5, 50));
    CHKERRQ(FllopDebug1("||A_new||=%.12e\n",norm_A));
    CHKERRQ(VecNorm(child->b,NORM_2,&norm_b));
    CHKERRQ(FllopDebug1("||b_new||=%.12e\n",norm_b));
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
    CHKERRQ(PetscMalloc1(Mn,&iss));
    CHKERRQ(MatNestGetISs(child->BE,iss,NULL));
    is = iss[Mn-1];

    /* copy the corresponding part of child's lambda_E to parent's lambda_I */
    CHKERRQ(VecGetSubVector(child->lambda_E,is,&lambda_EI));
    CHKERRQ(VecCopy(lambda_EI,parent->lambda_I));
    CHKERRQ(VecRestoreSubVector(child->lambda_E,is,&lambda_EI));

    /* copy the rest of child's lambda_E to parent's lambda_E */
    CHKERRQ(ISConcatenate(PetscObjectComm((PetscObject)child->BE),Mn-1,iss,&is));
    CHKERRQ(VecGetSubVector(child->lambda_E,is,&lambda_EI));
    CHKERRQ(VecCopy(lambda_EI,parent->lambda_E));
    CHKERRQ(VecRestoreSubVector(child->lambda_E,is,&lambda_EI));

    CHKERRQ(ISDestroy(&is));
    CHKERRQ(PetscFree(iss));
  } else { /* n==1 => there had been only ineq. con. before */
    CHKERRQ(VecCopy(child->lambda_E,parent->lambda_I));
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
  CHKERRQ(QPTransformBegin(QPTFreezeIneq, QPTPostSolve_QPTFreezeIneq,NULL, QP_DUPLICATE_COPY_POINTERS,&qp,&child,&comm));
  CHKERRQ(QPGetIneq(qp,&BI,&cI));
  CHKERRQ(QPAddEq(child,BI,cI));
  CHKERRQ(QPSetEqMultiplier(child,NULL));
  CHKERRQ(QPSetIneq(child,NULL,NULL));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  PERMON_ASSERT(!qp->cE,"!qp->cE");
  CHKERRQ(MatIsImplicitTranspose(qp->BE, &flg));
  PERMON_ASSERT(flg,"BE is implicit transpose");
  
  CHKERRQ(PetscLogEventBegin(QPT_SplitBE,qp,0,0,0));
  CHKERRQ(QPTransformBegin(QPTSplitBE, NULL, NULL, QP_DUPLICATE_COPY_POINTERS, &qp, &child, &comm));

  CHKERRQ(PermonMatTranspose(child->BE, MAT_TRANSPOSE_CHEAPEST, &Bet));
  CHKERRQ(PermonMatTranspose(Bet, MAT_TRANSPOSE_EXPLICIT, &Be));
  CHKERRQ(MatDestroy(&Bet));

  CHKERRQ(MatGetOwnershipRange(Be, &ilo, &ihi));

  CHKERRQ(PetscMalloc((ihi-ilo)*sizeof(PetscInt), &idxg));
  CHKERRQ(PetscMalloc((ihi-ilo)*sizeof(PetscInt), &idxd));

  for (i=ilo; i<ihi; i++){
    CHKERRQ(MatGetRow(Be, i, &ncols, &cols, &vals));
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
    CHKERRQ(MatRestoreRow(Be, i, &ncols, &cols, &vals));
  }
  CHKERRQ(ISCreateGeneral(comm, ng, idxg, PETSC_OWN_POINTER, &isrowg));
  CHKERRQ(ISCreateGeneral(comm, nd, idxd, PETSC_OWN_POINTER, &isrowd));
  
  CHKERRQ(MatCreateSubMatrix(Be, isrowg, NULL, MAT_INITIAL_MATRIX, &Bg));
  CHKERRQ(MatCreateSubMatrix(Be, isrowd, NULL, MAT_INITIAL_MATRIX, &Bd));
  CHKERRQ(MatDestroy(&Be));

  CHKERRQ(PermonMatTranspose(Bg, MAT_TRANSPOSE_EXPLICIT, &Bgt));
  CHKERRQ(MatDestroy(&Bg));
  CHKERRQ(PermonMatTranspose(Bgt, MAT_TRANSPOSE_CHEAPEST, &Bg));
  
  CHKERRQ(PermonMatTranspose(Bd, MAT_TRANSPOSE_EXPLICIT, &Bdt));
  CHKERRQ(MatDestroy(&Bd));
  CHKERRQ(PermonMatTranspose(Bdt, MAT_TRANSPOSE_CHEAPEST, &Bd));

  CHKERRQ(MatDestroy(&child->BE));
  CHKERRQ(QPAddEq(child, Bg, NULL));
  CHKERRQ(QPAddEq(child, Bd, NULL));
  
  CHKERRQ(PetscLogEventEnd(QPT_SplitBE,qp,0,0,0));

  CHKERRQ(ISDestroy(&isrowg));
  CHKERRQ(ISDestroy(&isrowd));
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
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qpt_matis_to_diag_norm",&computeNorm,NULL));
  if (ctx->isDir) {
    if (computeNorm) {
      /* TODO: change implementation for submat copies for PETSc>=3.8 */
      MatDuplicate(child->A,MAT_COPY_VALUES,&AsubCopy);
      //CHKERRQ(MatGetLocalToGlobalMapping(child->A,&l2g,NULL));
      //CHKERRQ(ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,ctx->isDir,&isDirLoc));
      //CHKERRQ(MatGetLocalSubMatrix(parent->A,isDirLoc,isDirLoc,&Asub));
      //CHKERRQ(MatDuplicate(Asub,MAT_COPY_VALUES,&AsubCopy));
      CHKERRQ(VecDuplicate(child->x,&dir));
      CHKERRQ(VecDuplicate(child->b,&b_adjustU));
      CHKERRQ(VecCopy(child->b,b_adjustU));
      CHKERRQ(VecGetLocalVector(dir,matis->y));
      CHKERRQ(VecScatterBegin(matis->cctx,parent->x,matis->y,INSERT_VALUES,SCATTER_FORWARD)); /* set local vec */
      CHKERRQ(VecScatterEnd(matis->cctx,parent->x,matis->y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecRestoreLocalVector(dir,matis->y));
      CHKERRQ(MatZeroRowsColumnsIS(child->A,ctx->isDir,1.0,dir,b_adjustU));
    }
  } else {
    /* propagate changed RHS */
    /* TODO: flag for pure Neumann? */
    CHKERRQ(VecGetLocalVector(child->b,matis->y));
    CHKERRQ(VecLockGet(parent->b,&lock));
    if (lock) CHKERRQ(VecLockReadPop(parent->b)); /* TODO: safe? */
    CHKERRQ(VecSet(parent->b,0.0));
    CHKERRQ(VecScatterBegin(matis->rctx,matis->y,parent->b,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,matis->y,parent->b,ADD_VALUES,SCATTER_REVERSE));
    if (lock) CHKERRQ(VecLockReadPush(parent->b));
    CHKERRQ(VecRestoreLocalVector(child->b,matis->y));
  }

  /* assemble solution */
  CHKERRQ(VecGetLocalVector(child->x,matis->x));
  CHKERRQ(VecScatterBegin(matis->rctx,matis->x,parent->x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(matis->rctx,matis->x,parent->x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecRestoreLocalVector(child->x,matis->x));
  
  if (computeNorm) {
    /* compute norm */
    CHKERRQ(VecDuplicate(parent->b,&resid));
    if (ctx->isDir) {
      CHKERRQ(VecDuplicate(parent->b,&b_adjust));
      CHKERRQ(VecSet(b_adjust,.0));
      CHKERRQ(VecGetLocalVector(b_adjustU,matis->y));
      CHKERRQ(VecScatterBegin(matis->rctx,matis->y,b_adjust,ADD_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(matis->rctx,matis->y,b_adjust,ADD_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecRestoreLocalVector(b_adjustU,matis->y));
      CHKERRQ(MatMult(parent->A,parent->x,resid));
      CHKERRQ(VecAXPY(resid,-1.0,b_adjust)); /* Ax-b */ 
      CHKERRQ(VecNorm(b_adjust,NORM_2,&normb));
      CHKERRQ(MatCopy(AsubCopy,child->A,SAME_NONZERO_PATTERN));
      //CHKERRQ(MatCopy(AsubCopy,Asub,SAME_NONZERO_PATTERN));
      CHKERRQ(VecDestroy(&b_adjustU));
      CHKERRQ(VecDestroy(&b_adjust));
    } else {
      CHKERRQ(MatMult(parent->A,parent->x,resid));
      CHKERRQ(VecAXPY(resid,-1.0,parent->b)); /* Ax-b */ 
      CHKERRQ(VecNorm(parent->b,NORM_2,&normb));
    } 
    CHKERRQ(VecNorm(resid,NORM_2,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Dirichlet in Hess: %d, r = ||Ax-b|| = %e, r/||b|| = %e\n",!ctx->isDir,norm,norm/normb));
    CHKERRQ(VecDestroy(&resid));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPTPostSolveDestroy_QPTMatISToBlockDiag"
static PetscErrorCode QPTPostSolveDestroy_QPTMatISToBlockDiag(void *ctx)
{
  QPTMatISToBlockDiag_Ctx *cctx = (QPTMatISToBlockDiag_Ctx*)ctx;

  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&cctx->isDir));
  CHKERRQ(PetscFree(cctx));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  CHKERRQ(QPTransformBegin(QPTMatISToBlockDiag,QPTPostSolve_QPTMatISToBlockDiag,QPTPostSolveDestroy_QPTMatISToBlockDiag,QP_DUPLICATE_DO_NOT_COPY,&qp,&child,&comm));
  CHKERRQ(PetscNew(&ctx));
  child->postSolveCtx = ctx;
  ctx->isDir = NULL; /* set by QPFetiSetDirichlet() */
  CHKERRQ(PCDestroy(&child->pc));
  CHKERRQ(QPGetPC(child,&child->pc));

  /* create block diag */
  Mat_IS *matis  = (Mat_IS*)qp->A->data;
  CHKERRQ(MatCreateBlockDiag(comm,matis->A,&A));
  CHKERRQ(QPSetOperator(child,A));
  CHKERRQ(QPSetEq(child,qp->BE,NULL));
  CHKERRQ(QPSetOperatorNullSpace(child,qp->R));
  CHKERRQ(MatDestroy(&qp->BE));

  /* get mappings for RHS decomposition
  *  create interface mappings
  *  adapted from PCISSetUp */
  /* get info on mapping */
  mapping = matis->rmapping;
  CHKERRQ(ISLocalToGlobalMappingGetSize(mapping,&n));
  CHKERRQ(ISLocalToGlobalMappingGetInfo(mapping,&n_neigh,&neigh,&n_shared,&shared));

  /* Identifying interior and interface nodes, in local numbering */
  CHKERRQ(PetscBTCreate(n,&bt));
  for (i=0;i<n_neigh;i++) {
    for (j=0;j<n_shared[i];j++) {
        CHKERRQ(PetscBTSet(bt,shared[i][j]));
    }
  }
  /* Creating local and global index sets for interior and inteface nodes. */
  CHKERRQ(PetscMalloc1(n,&idx_I_local));
  CHKERRQ(PetscMalloc1(n,&idx_B_local));
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
  CHKERRQ(ISLocalToGlobalMappingApply(mapping,n_B,idx_B_local,idx_B_global));
  CHKERRQ(ISLocalToGlobalMappingApply(mapping,n_I,idx_I_local,idx_I_global));

  /* Creating the index sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n_B,idx_B_local,PETSC_COPY_VALUES, &is_B_local));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n_B,idx_B_global,PETSC_COPY_VALUES,&is_B_global));
  /* TODO remove interior idx sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_local,PETSC_COPY_VALUES, &is_I_local));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_global,PETSC_COPY_VALUES,&is_I_global));

    /* Creating work vectors and arrays */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n_B,&vec1_B));
  CHKERRQ(VecDuplicate(vec1_B, &D));

  /* Creating the scatter contexts */
  CHKERRQ(VecScatterCreate(matis->x,is_B_local,vec1_B,NULL,&N_to_B));
  CHKERRQ(VecScatterCreate(qp->x,is_B_global,vec1_B,NULL,&global_to_B));

  /* Creating scaling "matrix" D */
  CHKERRQ(VecSet(D,1.0));
  CHKERRQ(VecScatterBegin(N_to_B,matis->counter,vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(N_to_B,matis->counter,vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecPointwiseDivide(D,D,vec1_B));

  /* decompose assembled vecs */
  CHKERRQ(MatCreateVecs(A,&child->x,&child->b));
  /* assemble b */
  CHKERRQ(VecGetLocalVector(child->b,matis->y));
  CHKERRQ(VecDuplicate(qp->b,&b));
  CHKERRQ(VecCopy(qp->b,b));
  CHKERRQ(VecScatterBegin(global_to_B,qp->b,vec1_B,INSERT_VALUES,SCATTER_FORWARD)); /* get interface DOFs */
  CHKERRQ(VecScatterEnd(global_to_B,qp->b,vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecPointwiseMult(vec1_B,D,vec1_B)); /* DOF/(number of subdomains it belongs to) */
  CHKERRQ(VecScatterBegin(global_to_B,vec1_B,b,INSERT_VALUES,SCATTER_REVERSE)); /* replace values in RHS */
  CHKERRQ(VecScatterEnd(global_to_B,vec1_B,b,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(matis->cctx,b,matis->y,INSERT_VALUES,SCATTER_FORWARD)); /* set local vec */
  CHKERRQ(VecScatterEnd(matis->cctx,b,matis->y,INSERT_VALUES,SCATTER_FORWARD));
  /* assemble x */
  CHKERRQ(VecGetLocalVector(child->x,matis->x));
  CHKERRQ(VecScatterBegin(matis->cctx,qp->x,matis->x,INSERT_VALUES,SCATTER_FORWARD)); /* set local vec */
  CHKERRQ(VecScatterEnd(matis->cctx,qp->x,matis->x,INSERT_VALUES,SCATTER_FORWARD));

  /* inherit l2g and i2g */
  CHKERRQ(ISLocalToGlobalMappingGetIndices(mapping,&idx_l2g));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)qp),n,idx_l2g,PETSC_COPY_VALUES,&l2g));
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(mapping,&idx_l2g));
  CHKERRQ(QPFetiSetLocalToGlobalMapping(child,l2g));
  CHKERRQ(ISOnComm(is_B_global,PETSC_COMM_WORLD,PETSC_COPY_VALUES,&i2g));
  CHKERRQ(ISSort(i2g));
  CHKERRQ(QPFetiSetInterfaceToGlobalMapping(child,i2g));

  CHKERRQ(ISDestroy(&i2g));
  CHKERRQ(ISDestroy(&l2g));
  CHKERRQ(VecRestoreLocalVector(child->x,matis->x));
  CHKERRQ(VecRestoreLocalVector(child->b,matis->y));
  CHKERRQ(ISLocalToGlobalMappingRestoreInfo(mapping,&n_neigh,&neigh,&n_shared,&shared));
  CHKERRQ(ISDestroy(&is_B_local));
  CHKERRQ(ISDestroy(&is_B_global));
  CHKERRQ(ISDestroy(&is_I_local));
  CHKERRQ(ISDestroy(&is_I_global));
  CHKERRQ(VecScatterDestroy(&N_to_B));
  CHKERRQ(VecScatterDestroy(&global_to_B));
  CHKERRQ(PetscFree(idx_B_local));
  CHKERRQ(PetscFree(idx_I_local));
  CHKERRQ(PetscBTDestroy(&bt));
  CHKERRQ(VecDestroy(&D));
  CHKERRQ(VecDestroy(&vec1_B));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  
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

  CHKERRQ(PetscLogEventBegin(QPT_AllInOne,qp,0,0,0));
  PetscObjectOptionsBegin((PetscObject)qp);
  CHKERRQ(PetscOptionsBool("-qp_I_freeze","perform QPTFreezeIneq","QPTFreezeIneq",freeze,&freeze,NULL));
  CHKERRQ(PetscOptionsBoolGroupBegin("-qp_O_normalize","perform QPTNormalizeObjective","QPTNormalizeObjective",&normalize));
  CHKERRQ(PetscOptionsBoolGroupEnd("-qp_O_normalize_hessian","perform QPTNormalizeHessian","QPTNormalizeHessian",&normalize_hessian));
  PetscOptionsEnd();

  //TODO do this until QPTFromOptions supports chain updates
  CHKERRQ(QPRemoveChild(qp));

  if (normalize) {
    CHKERRQ(QPTNormalizeObjective(qp));
  } else if (normalize_hessian) {
    CHKERRQ(QPTNormalizeHessian(qp));
  }

  CHKERRQ(QPTScale(qp));
  CHKERRQ(QPTOrthonormalizeEqFromOptions(qp));
  if (freeze) CHKERRQ(QPTFreezeIneq(qp));
  if (dual) {
    CHKERRQ(QPTDualize(qp,invType,regularize_e));
    CHKERRQ(QPTScale(qp));
    CHKERRQ(QPTOrthonormalizeEqFromOptions(qp));
  }
  if (project) {
    CHKERRQ(QPTEnforceEqByProjector(qp));
  }
  
  if (dual || project) {
    normalize = PETSC_FALSE;
    normalize_hessian = PETSC_FALSE;
    CHKERRQ(QPChainGetLast(qp,&last));
    PetscObjectOptionsBegin((PetscObject)last);
    CHKERRQ(PetscOptionsBoolGroupBegin("-qp_O_normalize","perform QPTNormalizeObjective","QPTNormalizeObjective",&normalize));
    CHKERRQ(PetscOptionsBoolGroupEnd("-qp_O_normalize_hessian","perform QPTNormalizeHessian","QPTNormalizeHessian",&normalize_hessian));
    PetscOptionsEnd();
    if (normalize) {
      CHKERRQ(QPTNormalizeObjective(qp));
    } else if (normalize_hessian) {
      CHKERRQ(QPTNormalizeHessian(qp));
    }
  }

  CHKERRQ(QPTEnforceEqByPenalty(qp,penalty,penalty_direct));
  CHKERRQ(PetscLogEventEnd  (QPT_AllInOne,qp,0,0,0));
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
    CHKERRQ(PetscOptionsBool("-feti","perform FETI DDM combination of transforms","QPTAllInOne",feti,&feti,NULL));
    if (feti) {
      ddm             = PETSC_TRUE;
      dual            = PETSC_TRUE;
      project         = PETSC_TRUE;
    }
    CHKERRQ(PetscOptionsReal("-penalty","QPTEnforceEqByPenalty penalty parameter","QPTEnforceEqByPenalty",penalty,&penalty,NULL));
    CHKERRQ(PetscOptionsBool("-penalty_direct","","QPTEnforceEqByPenalty",penalty_direct,&penalty_direct,NULL));
    CHKERRQ(PetscOptionsBool("-project","perform QPTEnforceEqByProjector","QPTEnforceEqByProjector",project,&project,NULL));
    CHKERRQ(PetscOptionsBool("-dual","perform QPTDualize","QPTDualize",dual,&dual,NULL));
    CHKERRQ(PetscOptionsBool("-ddm","domain decomposed data","QPTDualize",ddm,&ddm,NULL));
    CHKERRQ(PetscOptionsBool("-regularize","perform stiffness matrix regularization (for singular TFETI matrices)","QPTDualize",regularize,&regularize,NULL));
    if (ddm) {
      invType = MAT_INV_BLOCKDIAG;
    }
  }
  PetscOptionsEnd();
  CHKERRQ(QPTAllInOne(qp, invType, dual, project, penalty, penalty_direct, regularize));
  PetscFunctionReturn(0);
}
#undef QPTransformBegin
