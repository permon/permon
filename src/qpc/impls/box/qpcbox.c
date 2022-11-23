#include <../src/qpc/impls/box/qpcboximpl.h>

PetscErrorCode QPCSetUp_Box(QPC qpc)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;
  Vec                   lb;

  PetscFunctionBegin;

  /* prepare lambdawork vector based on the layout of lb */
  lb = ctx->lb;
  PetscCall(VecDuplicate(lb,&(qpc->lambdawork)));

  // TODO: verify layout of ub somewhere in setup or in create function

  PetscFunctionReturn(0);
}

static PetscErrorCode QPCGrads_Box(QPC qpc, Vec x, Vec g, Vec gf, Vec gc)
{
  Vec                   lb,ub;
  QPC_Box               *ctx = (QPC_Box*)qpc->data;

  PetscScalar           *x_a, *lb_a, *ub_a, *g_a, *gf_a, *gc_a;
  PetscInt              n_local, i;

  PetscFunctionBegin;
  lb = ctx->lb;
  ub = ctx->ub;
  PetscCall(VecGetLocalSize(x,&n_local));
  PetscCall(VecGetArray(x, &x_a));
  if (lb) PetscCall(VecGetArray(lb, &lb_a));
  if (ub) PetscCall(VecGetArray(ub, &ub_a));
  PetscCall(VecGetArray(g, &g_a));
  PetscCall(VecGetArray(gf, &gf_a));
  PetscCall(VecGetArray(gc, &gc_a));

  /* TODO create free/active IS? */
  for (i = 0; i < n_local; i++){
    if (lb && PetscAbsScalar(x_a[i] - lb_a[i]) <= qpc->astol) {
      /* active lower bound */
      gf_a[i] = 0.0;
      gc_a[i]= PetscMin(g_a[i],0.0);
    //} else if (ub && x_a[i] >= ub_a[i]) {
    } else if (ub && PetscAbsScalar(x_a[i] -  ub_a[i]) <= qpc->astol) {
      /* active upper bound */
      gf_a[i] = 0.0;
      gc_a[i]= PetscMax(g_a[i],0.0);
    } else {
      /* index of this component is in FREE SET */
      gf_a[i] = g_a[i];
    }
  }

  PetscCall(VecRestoreArray(x, &x_a));
  if (lb) PetscCall(VecRestoreArray(lb, &lb_a));
  if (ub) PetscCall(VecRestoreArray(ub, &ub_a));
  PetscCall(VecRestoreArray(g, &g_a));
  PetscCall(VecRestoreArray(gf, &gf_a));
  PetscCall(VecRestoreArray(gc, &gc_a));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPCGradReduced_Box(QPC qpc, Vec x, Vec gf, PetscReal alpha, Vec gr)
{
  Vec                   lb,ub;
  QPC_Box               *ctx = (QPC_Box*)qpc->data;

  PetscScalar           *x_a, *lb_a, *ub_a, *gf_a, *gr_a;
  PetscInt              n_local, i;

  PetscFunctionBegin;
  lb = ctx->lb;
  ub = ctx->ub;
  PetscCall(VecGetLocalSize(x,&n_local));
  PetscCall(VecGetArray(x, &x_a));
  if (lb) PetscCall(VecGetArray(lb, &lb_a));
  if (ub) PetscCall(VecGetArray(ub, &ub_a));
  PetscCall(VecGetArray(gf, &gf_a));
  PetscCall(VecGetArray(gr, &gr_a));

  for (i = 0; i < n_local; i++){
    if (lb && gf_a[i] > 0.0) {
      gr_a[i] = PetscMin(gf_a[i],(x_a[i]-lb_a[i])/alpha);
    } else if (ub && gf_a[i] < 0.0) {
      gr_a[i] = PetscMax(gf_a[i],(x_a[i]-ub_a[i])/alpha);
    }
  }

  PetscCall(VecRestoreArray(x, &x_a));
  if (lb) PetscCall(VecRestoreArray(lb, &lb_a));
  if (ub) PetscCall(VecRestoreArray(ub, &ub_a));
  PetscCall(VecRestoreArray(gf, &gf_a));
  PetscCall(VecRestoreArray(gr, &gr_a));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPCFeas_Box(QPC qpc,Vec x, Vec d, PetscScalar *alpha)
{
  Vec                   lb,ub;
  QPC_Box               *ctx = (QPC_Box*)qpc->data;
  PetscScalar           alpha_temp, alpha_i;

  PetscScalar           *x_a, *d_a, *lb_a, *ub_a;
  PetscInt              n_local, i;

  PetscFunctionBegin;
  lb = ctx->lb;
  ub = ctx->ub;

  alpha_temp = PETSC_INFINITY;

  PetscCall(VecGetLocalSize(x,&n_local));
  PetscCall(VecGetArray(x,&x_a));
  PetscCall(VecGetArray(d,&d_a));
  if (lb) PetscCall(VecGetArray(lb,&lb_a));
  if (ub) PetscCall(VecGetArray(ub,&ub_a));

  for(i=0;i < n_local;i++){
    if(d_a[i] > 0 && lb && lb_a[i] > PETSC_NINFINITY) {
      alpha_i = x_a[i]-lb_a[i];
      alpha_i = alpha_i/d_a[i];
      if (alpha_i < alpha_temp){
          alpha_temp = alpha_i;
      }
    }

    if(d_a[i] < 0 && ub && ub_a[i] < PETSC_INFINITY) {
      alpha_i = (x_a[i]-ub_a[i]);
      alpha_i = alpha_i/d_a[i];

      if (alpha_i < alpha_temp){
          alpha_temp = alpha_i;
      }
    }
  }
  *alpha = alpha_temp;

  PetscCall(VecRestoreArray(x,&x_a));
  PetscCall(VecRestoreArray(d,&d_a));
  if (lb) PetscCall(VecRestoreArray(lb,&lb_a));
  if (ub) PetscCall(VecRestoreArray(ub,&ub_a));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPCBoxSet_Box(QPC qpc,Vec lb, Vec ub)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ctx->lb));
  PetscCall(VecDestroy(&ctx->ub));
  PetscCall(VecDestroy(&ctx->llb));
  PetscCall(VecDestroy(&ctx->lub));
  ctx->lb = lb;
  ctx->ub = ub;
  if (lb) {
    PetscCall(VecDuplicate(lb,&ctx->llb));
    PetscCall(VecInvalidate(ctx->llb));
  }
  if (ub) {
    PetscCall(VecDuplicate(ub,&ctx->lub));
    PetscCall(VecInvalidate(ctx->lub));
  }
  PetscCall(PetscObjectReference((PetscObject)lb));
  PetscCall(PetscObjectReference((PetscObject)ub));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPCBoxGet_Box(QPC qpc,Vec *lb,Vec *ub)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;

  PetscFunctionBegin;
  if (lb) *lb = ctx->lb;
  if (ub) *ub = ctx->ub;
  PetscFunctionReturn(0);
}

static PetscErrorCode QPCBoxGetMultipliers_Box(QPC qpc,Vec *llb,Vec *lub)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;

  PetscFunctionBegin;
  if (llb) *llb = ctx->llb;
  if (lub) *lub = ctx->lub;
  PetscFunctionReturn(0);
}

PetscErrorCode QPCIsLinear_Box(QPC qpc,PetscBool *linear)
{
  PetscFunctionBegin;
  *linear = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode QPCIsSubsymmetric_Box(QPC qpc,PetscBool *subsymmetric)
{
  PetscFunctionBegin;
  *subsymmetric = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode QPCGetBlockSize_Box(QPC qpc,PetscInt *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode QPCGetNumberOfConstraints_Box(QPC qpc, PetscInt *num)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;
  PetscInt          size,bs;

  PetscFunctionBegin;

  if(qpc->is){
        /* IS is present, return the size of IS */
        PetscCall(ISGetSize(qpc->is,&size));
        PetscCall(QPCGetBlockSize(qpc,&bs));
        *num = size/bs;
  } else {
        /* IS is not present, all components are constrained */
        PetscCall(VecGetSize(ctx->lb,&size)); /* = size of ub */
        *num = size;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode QPCGetConstraintFunction_Box(QPC qpc, Vec x_sub, Vec *hx_out)
{
  QPC_Box               *ctx = (QPC_Box*)qpc->data;
  Vec                   lb,ub;
  PetscScalar           *lb_a, *ub_a, *x_a, *hx_a; /* local arrays */
  PetscInt              m; /* local size */
  PetscInt              i; /* iterator */

  PetscFunctionBegin;
  lb = ctx->lb;
  ub = ctx->ub;
  PetscCall(VecGetLocalSize(qpc->lambdawork,&m));
  PetscCall(VecGetArray(x_sub,&x_a));
  PetscCall(VecGetArray(qpc->lambdawork,&hx_a));
  if (lb) PetscCall(VecGetArray(lb,&lb_a));
  if (ub) PetscCall(VecGetArray(ub,&ub_a));

  for(i = 0; i< m; i++){
      // TODO: verify the access to arrays, with correct layout, it should be fine..
      if (lb) {
        if (ub) {
          hx_a[i] = PetscAbsScalar(x_a[i] - (ub_a[i] + lb_a[i])/2) - (ub_a[i] - lb_a[i])/2;
        } else {
          hx_a[i] = lb_a[i] - x_a[i];
        }
      } else {
        hx_a[i] = x_a[i] - ub_a[i];
      }
  }

  /* restore local arrays */
  PetscCall(VecRestoreArray(x_sub, &x_a));
  PetscCall(VecRestoreArray(qpc->lambdawork,&hx_a));
  if (lb) PetscCall(VecRestoreArray(lb,&lb_a));
  if (ub) PetscCall(VecRestoreArray(ub,&ub_a));

  *hx_out = qpc->lambdawork;
  PetscFunctionReturn(0);
}

PetscErrorCode QPCProject_Box(QPC qpc, Vec x, Vec Px)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;
  Vec             lb,ub;

  PetscFunctionBegin;
  lb = ctx->lb;
  ub = ctx->ub;
  if (lb) {
    PetscCall(VecPointwiseMax(Px,x,lb));
    if (ub) PetscCall(VecPointwiseMin(Px,Px,ub));
  } else {
    PetscCall(VecPointwiseMin(Px,x,ub));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QPCView_Box(QPC qpc, PetscViewer viewer)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;

  PetscFunctionBegin;
  /* print lb */
  if (ctx->lb) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "lb:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(VecView(ctx->lb,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  /* print ub */
  if (ctx->ub) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "ub:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(VecView(ctx->ub,viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QPCViewKKT_Box(QPC qpc, Vec x, PetscReal normb, PetscViewer v)
{
  QPC_Box         *ctx = (QPC_Box*)qpc->data;
  Vec             lb,ub,llb,lub,r,o;
  PetscScalar     norm,dot;

  PetscFunctionBegin;
  lb = ctx->lb;
  ub = ctx->ub;
  llb = ctx->llb;
  lub = ctx->lub;
  if (lb) {
    PetscCall(VecDuplicate(x,&o));
    PetscCall(VecDuplicate(x,&r));

    /* rI = norm(min(x-lb,0)) */
    PetscCall(VecSet(o,0.0));                                   /* o = zeros(size(r)) */
    PetscCall(VecWAXPY(r, -1.0, lb, x));                        /* r = x - lb       */
    PetscCall(VecPointwiseMin(r,r,o));                          /* r = min(r,o)     */
    PetscCall(VecNorm(r,NORM_2,&norm));                         /* norm = norm(r)     */
    PetscCall(PetscViewerASCIIPrintf(v,"r = ||min(x-lb,0)||      = %.2e    r/||b|| = %.2e\n",norm,norm/normb));

    /* lambda >= o  =>  examine min(lambda,o) */
    PetscCall(VecSet(o,0.0));                                   /* o = zeros(size(r)) */
    PetscCall(VecPointwiseMin(r,llb,o));
    PetscCall(VecNorm(r,NORM_2,&norm));                         /* norm = ||min(lambda,o)|| */
    PetscCall(PetscViewerASCIIPrintf(v,"r = ||min(lambda_lb,0)|| = %.2e    r/||b|| = %.2e\n",norm,norm/normb));

    /* lambda'*(lb-x) = 0 */
    PetscCall(VecCopy(lb,r));
    PetscCall(VecAXPY(r,-1.0,x));
    {
      PetscInt i,n;
      PetscScalar *rarr;
      const PetscScalar *larr;
      PetscCall(VecGetLocalSize(r,&n));
      PetscCall(VecGetArray(r,&rarr));
      PetscCall(VecGetArrayRead(lb,&larr));
      for (i=0; i<n; i++) if (larr[i]<=PETSC_NINFINITY) rarr[i]=-1.0;
      PetscCall(VecRestoreArray(r,&rarr));
      PetscCall(VecRestoreArrayRead(lb,&larr));
    }
    PetscCall(VecDot(llb,r,&dot));
    dot = PetscAbs(dot);
    PetscCall(PetscViewerASCIIPrintf(v,"r = |lambda_lb'*(lb-x)|  = %.2e    r/||b|| = %.2e\n",dot,dot/normb));

    PetscCall(VecDestroy(&o));
    PetscCall(VecDestroy(&r));
  }

  if (ub) {
    PetscCall(VecDuplicate(x,&o));
    PetscCall(VecDuplicate(x,&r));

    /* rI = norm(max(x-ub,0)) */
    PetscCall(VecDuplicate(x,&r));
    PetscCall(VecDuplicate(x,&o));
    PetscCall(VecSet(o,0.0));                                   /* o = zeros(size(r)) */
    PetscCall(VecWAXPY(r, -1.0, ub, x));                        /* r = x - ub       */
    PetscCall(VecPointwiseMax(r,r,o));                          /* r = max(r,o)     */
    PetscCall(VecNorm(r,NORM_2,&norm));                         /* norm = norm(r)     */
    PetscCall(PetscViewerASCIIPrintf(v,"r = ||max(x-ub,0)||      = %.2e    r/||b|| = %.2e\n",norm,norm/normb));

    /* lambda >= o  =>  examine min(lambda,o) */
    PetscCall(VecSet(o,0.0));                                   /* o = zeros(size(r)) */
    PetscCall(VecPointwiseMin(r,lub,o));
    PetscCall(VecNorm(r,NORM_2,&norm));                         /* norm = ||min(lambda,o)|| */
    PetscCall(PetscViewerASCIIPrintf(v,"r = ||min(lambda_ub,0)|| = %.2e    r/||b|| = %.2e\n",norm,norm/normb));

    /* lambda'*(x-ub) = 0 */
    PetscCall(VecCopy(ub,r));
    PetscCall(VecAYPX(r,-1.0,x));
    {
      PetscInt i,n;
      PetscScalar *rarr;
      const PetscScalar *uarr;
      PetscCall(VecGetLocalSize(r,&n));
      PetscCall(VecGetArray(r,&rarr));
      PetscCall(VecGetArrayRead(ub,&uarr));
      for (i=0; i<n; i++) if (uarr[i]>=PETSC_INFINITY) rarr[i]=1.0;
      PetscCall(VecRestoreArray(r,&rarr));
      PetscCall(VecRestoreArrayRead(ub,&uarr));
    }
    PetscCall(VecDot(lub,r,&dot));
    dot = PetscAbs(dot);
    PetscCall(PetscViewerASCIIPrintf(v,"r = |lambda_ub'*(x-ub)|  = %.2e    r/||b|| = %.2e\n",dot,dot/normb));

    PetscCall(VecDestroy(&o));
    PetscCall(VecDestroy(&r));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QPCDestroy_Box(QPC qpc)
{
  QPC_Box      *ctx = (QPC_Box*)qpc->data;

  PetscFunctionBegin;
  if (ctx->lb) {
    PetscCall(VecDestroy(&ctx->lb));
    PetscCall(VecDestroy(&ctx->llb));
  }
  if (ctx->ub) {
    PetscCall(VecDestroy(&ctx->ub));
    PetscCall(VecDestroy(&ctx->lub));
  }
  PetscFunctionReturn(0);
}

FLLOP_EXTERN PetscErrorCode QPCCreate_Box(QPC qpc)
{
  QPC_Box      *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  qpc->data = (void*)ctx;

  /* set general QPC functions already implemented for this QPC type */
  qpc->ops->destroy                     = QPCDestroy_Box;
  qpc->ops->setup                       = QPCSetUp_Box;
  qpc->ops->view                        = QPCView_Box;
  qpc->ops->viewkkt                     = QPCViewKKT_Box;
  qpc->ops->getblocksize                = QPCGetBlockSize_Box;
  qpc->ops->getconstraintfunction       = QPCGetConstraintFunction_Box;
  qpc->ops->getnumberofconstraints      = QPCGetNumberOfConstraints_Box;
  qpc->ops->project                     = QPCProject_Box;
  qpc->ops->islinear                    = QPCIsLinear_Box;
  qpc->ops->issubsymmetric              = QPCIsSubsymmetric_Box;
  qpc->ops->feas                        = QPCFeas_Box;
  qpc->ops->grads                       = QPCGrads_Box;
  qpc->ops->gradreduced                 = QPCGradReduced_Box;

  /* set type-specific functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)qpc,"QPCBoxSet_Box_C",QPCBoxSet_Box));
  PetscCall(PetscObjectComposeFunction((PetscObject)qpc,"QPCBoxGet_Box_C",QPCBoxGet_Box));
  PetscCall(PetscObjectComposeFunction((PetscObject)qpc,"QPCBoxGetMultipliers_Box_C",QPCBoxGetMultipliers_Box));

  /* initialize type-specific inner data */
  ctx->lb                         = NULL;
  ctx->ub                         = NULL;
  ctx->llb                        = NULL;
  ctx->lub                        = NULL;
  PetscFunctionReturn(0);
}


PetscErrorCode QPCBoxSet(QPC qpc,Vec lb, Vec ub)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  if (lb) PetscValidHeaderSpecific(lb,VEC_CLASSID,2);
  if (ub) PetscValidHeaderSpecific(ub,VEC_CLASSID,3);
  if (!lb && !ub) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_ARG_NULL,"lb and ub cannot be both NULL");
#if defined(PETSC_USE_DEBUG)
  if (lb && ub) {
    Vec diff;
    PetscReal min;

    PetscCall(VecDuplicate(lb,&diff));
    PetscCall(VecWAXPY(diff,-1.0,lb,ub));
    PetscCall(VecMin(diff,NULL,&min));
    /* TODO verify that algorithms work with min = 0 */
    if (min < 0.0) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_ARG_INCOMP,"lb components must be smaller than ub components");
    PetscCall(VecDestroy(&diff));
  }
#endif

  PetscUseMethod(qpc,"QPCBoxSet_Box_C",(QPC,Vec,Vec),(qpc,lb,ub));
  PetscFunctionReturn(0);
}

PetscErrorCode QPCBoxGet(QPC qpc,Vec *lb, Vec *ub)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  if (lb) PetscValidPointer(lb,2);
  if (ub) PetscValidPointer(ub,3);

  PetscUseMethod(qpc,"QPCBoxGet_Box_C",(QPC,Vec*,Vec*),(qpc,lb,ub));
  PetscFunctionReturn(0);
}

PetscErrorCode QPCBoxGetMultipliers(QPC qpc,Vec *llb,Vec *lub)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  if (llb) PetscValidPointer(llb,2);
  if (lub) PetscValidPointer(lub,3);

  PetscUseMethod(qpc,"QPCBoxGetMultipliers_Box_C",(QPC,Vec*,Vec*),(qpc,llb,lub));
  PetscFunctionReturn(0);
}

/*@
QPCCreateBox - create QPC Box instance; set the type of QPC to Box, set vector of variables, set index set of constrained components, set the value of lower and upper bounds

Parameters:
+ comm - MPI comm
. x - vector of variables
. is - index set of constrained variables, if NULL then all compoments are constrained
. lb - the vector with lower bounds values
. ub - the vector with upper bounds values
- qpc_out - pointer to QPC
@*/
PetscErrorCode QPCCreateBox(MPI_Comm comm,IS is,Vec lb,Vec ub,QPC *qpc_out)
{
  QPC qpc;

  PetscFunctionBegin;

  /* verify input data */
  if (is) PetscValidHeaderSpecific(is,IS_CLASSID,2);
  if (lb) PetscValidHeaderSpecific(lb,VEC_CLASSID,3);
  if (ub) PetscValidHeaderSpecific(ub,VEC_CLASSID,4);
  PetscValidPointer(qpc_out,5);

  PetscCall(QPCCreate(comm,&qpc));
  PetscCall(QPCSetIS(qpc,is));
  PetscCall(QPCSetType(qpc,QPCBOX));
  PetscCall(QPCBoxSet(qpc,lb,ub));
  *qpc_out = qpc;
  PetscFunctionReturn(0);
}
