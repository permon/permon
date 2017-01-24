
#include <../src/qps/impls/smalxe/smalxeimpl.h>

FLLOP_EXTERN PetscErrorCode QPSReset_SMALXE(QPS qps);

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetOperatorMaxEigenvalue_SMALXE"
static PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalue_SMALXE(QPS qps,PetscReal *maxeig)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *maxeig = smalxe->maxeig;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetOperatorMaxEigenvalue_SMALXE"
static PetscErrorCode QPSSMALXESetOperatorMaxEigenvalue_SMALXE(QPS qps,PetscReal maxeig)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->maxeig = maxeig;
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetM1Initial_SMALXE"
static PetscErrorCode QPSSMALXEGetM1Initial_SMALXE(QPS qps,PetscReal *M1_initial,QPSScalarArgType *argtype)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  if (M1_initial) *M1_initial = smalxe->M1_user;
  if (argtype) *argtype = smalxe->M1_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetM1Initial_SMALXE"
static PetscErrorCode QPSSMALXESetM1Initial_SMALXE(QPS qps,PetscReal M1_initial,QPSScalarArgType argtype)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->M1_user = M1_initial;
  smalxe->M1_type = argtype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetEta_SMALXE"
static PetscErrorCode QPSSMALXEGetEta_SMALXE(QPS qps,PetscReal *eta,QPSScalarArgType *argtype)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  if (eta) {
    *eta = smalxe->eta_user;
  }
  if (argtype) {
    *argtype = smalxe->eta_type;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetEta_SMALXE"
static PetscErrorCode QPSSMALXESetEta_SMALXE(QPS qps,PetscReal eta,QPSScalarArgType argtype)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->eta_user = eta;
  smalxe->eta_type = argtype;
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetRhoInitial_SMALXE"
static PetscErrorCode QPSSMALXEGetRhoInitial_SMALXE(QPS qps,PetscReal *rho_initial,QPSScalarArgType *argtype)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  if (rho_initial) {
    *rho_initial = smalxe->rho_user;
  }
  if (argtype) {
    *argtype = smalxe->rho_type;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetRhoInitial_SMALXE"
static PetscErrorCode QPSSMALXESetRhoInitial_SMALXE(QPS qps,PetscReal rho_initial,QPSScalarArgType argtype)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->rho_user = rho_initial;
  smalxe->rho_type = argtype;
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetM1Update_SMALXE"
static PetscErrorCode QPSSMALXEGetM1Update_SMALXE(QPS qps,PetscReal *M1_update)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *M1_update = smalxe->M1_update;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetM1Update_SMALXE"
static PetscErrorCode QPSSMALXESetM1Update_SMALXE(QPS qps,PetscReal M1_update)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->M1_update = M1_update;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetRhoUpdate_SMALXE"
static PetscErrorCode QPSSMALXEGetRhoUpdate_SMALXE(QPS qps,PetscReal *rho_update)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *rho_update = smalxe->rho_update;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetRhoUpdate_SMALXE"
static PetscErrorCode QPSSMALXESetRhoUpdate_SMALXE(QPS qps,PetscReal rho_update)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->rho_update = rho_update;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetRhoUpdateLate_SMALXE"
static PetscErrorCode QPSSMALXEGetRhoUpdateLate_SMALXE(QPS qps,PetscReal *rho_update_late)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *rho_update_late = smalxe->rho_update_late;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetRhoUpdateLate_SMALXE"
static PetscErrorCode QPSSMALXESetRhoUpdateLate_SMALXE(QPS qps,PetscReal rho_update_late)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->rho_update_late = rho_update_late;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetOperatorMaxEigenvalueIterations_SMALXE"
static PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueIterations_SMALXE(QPS qps,PetscInt *numit)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *numit = smalxe->maxeig_iter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetOperatorMaxEigenvalueIterations_SMALXE"
static PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueIterations_SMALXE(QPS qps,PetscInt numit)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->maxeig_iter = numit;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetOperatorMaxEigenvalueTolerance_SMALXE"
static PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueTolerance_SMALXE(QPS qps,PetscReal *tol)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *tol = smalxe->maxeig_tol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetOperatorMaxEigenvalueTolerance_SMALXE"
static PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueTolerance_SMALXE(QPS qps,PetscReal tol)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->maxeig_tol = tol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetInjectOperatorMaxEigenvalue_SMALXE"
static PetscErrorCode QPSSMALXESetInjectOperatorMaxEigenvalue_SMALXE(QPS qps,PetscBool flg)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->inject_maxeig = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetInjectOperatorMaxEigenvalue_SMALXE"
static PetscErrorCode QPSSMALXEGetInjectOperatorMaxEigenvalue_SMALXE(QPS qps,PetscBool *flg)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  *flg = smalxe->inject_maxeig;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetMonitor_SMALXE"
static PetscErrorCode QPSSMALXESetMonitor_SMALXE(QPS qps,PetscBool flg)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  
  PetscFunctionBegin;
  smalxe->monitor = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEUpdateNormBu_SMALXE"
PetscErrorCode QPSSMALXEUpdateNormBu_SMALXE(QPS qps,Vec u,PetscReal *normBu,PetscReal *enorm)
{
  QP qp_outer = qps->solQP;
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  Mat BE = qp_outer->BE;
  Vec cE = qp_outer->cE;
  Vec rE = smalxe->Bu;

  PetscFunctionBegin;
  TRY( MatMult(BE, u, rE) );                                /* Bu = B*u */
  if (cE) TRY( VecAXPY(rE, -1.0, cE) );                     /* Bu = Bu - c */
  TRY( VecNorm(rE, NORM_2, normBu) );                       /* normBu = norm(Bu) */
  *enorm = *normBu / smalxe->rtol_E;                        /* enorm = norm(Bu)/rtol_E */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEUpdateNormBu_SMALXEON"
static PetscErrorCode QPSSMALXEUpdateNormBu_SMALXEON(QPS qps,Vec u,PetscReal *normBu,PetscReal *enorm)
{
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  QPS qps_inner = smalxe->inner;
  QP qp_inner = qps_inner->solQP;
  Mat BtB;
  Vec BtBu=qps->work[0];

  PetscFunctionBegin;
  TRY( MatPenalizedGetPenalizedTerm(qp_inner->A,&BtB) );

  TRY( MatMult(BtB,u,BtBu) );                           /* BtBu = B'*B*u */
  TRY( QPSWorkVecStateUpdate(qps,0) );
  TRY( QPSSolutionVecStateUpdate(qps) );

  TRY( VecDot(u,BtBu,normBu) );                         /* normBu = u'*B'*B*u */
  *normBu = PetscSqrtReal(*normBu);                     /* normBu = sqrt(u'*B'*B*u) */
  *enorm = *normBu / smalxe->rtol_E;                    /* enorm = norm(Bu)/rtol_E */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEUpdateNormBu_Lag_SMALXEON"
static PetscErrorCode QPSSMALXEUpdateNormBu_Lag_SMALXEON(QPS qps,Vec u,PetscReal *normBu,PetscReal *enorm)
{
  static PetscReal normBu0=0.0;
  static PetscInt II=0,J=0;
  static PetscInt neval=0,niter=0;

  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps->data;
  QPS qps_inner = smalxe->inner;
  PetscReal normBu_approx;
  PetscReal normBu_exact,enorm_exact;
  PetscReal rdiff;
  PetscBool eval=PETSC_FALSE;

  PetscBool lag_monitor = smalxe->lag_monitor;
  PetscBool lag_compare = smalxe->lag_compare;
  PetscInt  offset = smalxe->norm_update_lag_offset;
  PetscInt  Jstart = smalxe->Jstart;
  PetscInt  Jstep = smalxe->Jstep;
  PetscInt  Jend = smalxe->Jend;
  PetscReal lower = smalxe->lower;
  PetscReal upper = smalxe->upper;

  PetscFunctionBegin;
  if (qps_inner->iteration <= offset) {
    TRY( QPSSMALXEUpdateNormBu_SMALXEON(qps,u,&normBu_exact,&enorm_exact) );
    eval=PETSC_TRUE;
    neval++;
    normBu0         = normBu_exact;
    normBu_approx   = normBu0;
    J               = Jstart;
    II               = 0;
  } else {
    if (II == 0) {
      TRY( QPSSMALXEUpdateNormBu_SMALXEON(qps,u,&normBu_exact,&enorm_exact) );
      eval=PETSC_TRUE;
      neval++;
      rdiff = PetscAbs(normBu_exact/normBu0);
      if (rdiff >= upper) {
        TRY( PetscInfo4(qps,"rdiff = |%.3e / %.3e| = %.3e >= %.3e, ||B*u|| will be recalculated\n",normBu_exact,normBu0,rdiff,upper) );
        II=0;
        J = Jstart;
      } else if (rdiff < lower) {
        TRY( PetscInfo4(qps,"rdiff = |%.3e / %.3e| = %.3e < %.3e, ||B*u|| will be recalculated\n",normBu_exact,normBu0,rdiff,lower) );
        II=0;
        J = Jstart;
      } else {
        II++;
      }
      normBu0 = normBu_exact;
    } else {
      II++;
    }
    normBu_approx  = normBu0;
  }
  niter++;

  if (II==J) {
    II=0;
    if (J < Jend) J+=Jstep;
  }

  if (lag_compare) {
    char sign;
    if (!eval) {
      TRY( QPSSMALXEUpdateNormBu_SMALXEON(qps,u,&normBu_exact,&enorm_exact) );
    }
    rdiff = PetscAbs(normBu_approx-normBu_exact)/normBu_exact;
    sign = (PetscBool)((normBu_exact > normBu_approx) ? '>' : '<');
    if (normBu_exact > normBu_approx) {
      sign = '>';
    } else if (normBu_exact < normBu_approx) {
      sign = '<';
    } else {
      sign = '=';
    }
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps), __FUNCT__": out %3d in %4d   II=%2d J=%2d niter=%4d neval=%4d   ||Bu||=%.4e  %c  %.4e=~||Bu|| relative_difference=%.4e %c\n",qps->iteration,qps_inner->iteration,II,J,niter,neval, normBu_exact, sign, normBu_approx, rdiff, rdiff > 10 ? sign : ' ') );
  } else if (lag_monitor) {
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps), __FUNCT__": out %3d in %4d   II=%2d J=%2d niter=%4d neval=%4d\n",qps->iteration,qps_inner->iteration,II,J,niter,neval) );
  }

  *normBu = normBu_approx;
  *enorm = *normBu / smalxe->rtol_E;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEUpdateRho_SMALXE"
static PetscErrorCode QPSSMALXEUpdateRho_SMALXE(QPS qps, PetscBool Lagrangian_flag)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  Mat           A_inner = smalxe->qp_penalized->A;
  PetscReal     rho_update;

  PetscFunctionBegin;
  switch (smalxe->state) {
    case 1: rho_update = smalxe->rho_update; break;
    case 3: rho_update = smalxe->rho_update_late; Lagrangian_flag = PETSC_TRUE; break;
  }
  if (!Lagrangian_flag || rho_update == 1.0) PetscFunctionReturn(0);
  
  TRY( PetscInfo2(qps,"updating rho, multiply by rho_update%d = %.4e\n",smalxe->state,rho_update) );
  TRY( MatPenalizedUpdatePenalty(A_inner, rho_update) );
  TRY( QPSMPGPUpdateMaxEigenvalue(smalxe->inner, rho_update) );
  smalxe->rho_updates++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEUpdateLambda_SMALXE"
static PetscErrorCode QPSSMALXEUpdateLambda_SMALXE(QPS qps,PetscReal rho)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  QP            qp, qp_inner;
  Mat           A_inner, BtB;
  Vec           u,BtBu,Btmu;
  PetscBool     flg1,flg2;

  PetscFunctionBegin;
  qp            = qps->solQP;
  qp_inner      = smalxe->qp_penalized;
  BtBu          = qps->work[0];
  Btmu          = qp->Bt_lambda;

  TRY( QPGetSolutionVector(qp, &u) );
  TRY( QPGetOperator(qp_inner, &A_inner) );
  TRY( MatPenalizedGetPenalizedTerm(A_inner, &BtB) );

  /* check if BtBu is up-to-date; if not, recompute it */
  TRY( QPSWorkVecStateChanged(qps,0,&flg1) );
  TRY( QPSSolutionVecStateChanged(qps,&flg2) );
  if (flg1 || flg2) {
    TRY( MatMult(BtB,u,BtBu) );                         /* BtBu = B'*B*u */
    TRY( QPSWorkVecStateUpdate(qps,0) );
    TRY( QPSSolutionVecStateUpdate(qps) );
    TRY( PetscInfo(qps,"BtBu recomputed\n") );
  } else {
    TRY( PetscInfo(qps,"BtBu reused\n") );
  }

  /* Update Btmu (eq. con. multiplier pre-multiplied by eq. con. matrix transpose) */
  TRY( VecAXPY(Btmu,rho,BtBu) );                      /* Btmu = Btmu + rho*BtBu = Btmu + rho*B'*B*u  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEUpdate_SMALXE"
PetscErrorCode QPSSMALXEUpdate_SMALXE(QPS qps, PetscReal Lag_old, PetscReal Lag, PetscReal rho)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  PetscReal     M1_new, M1_update=smalxe->M1_update;
  PetscReal     t,t2;
  PetscBool     flag;

  PetscFunctionBegin;
  {
    t = 0.5*rho*smalxe->normBu*smalxe->normBu;
  }
  t2 = Lag - (Lag_old + t);
  flag = (PetscBool)(t2 < smalxe->update_threshold);

  if (smalxe->monitor_outer) {
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps), "out %3d:    Lagrangian L       L-L_old      L-(L_old+1/2*rho*||Bu||^2)   threshold    1/2*rho*||Bu||^2\n",qps->iteration) );
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps), "out %3d: L: %+.10e  %+.3e                   %+.3e %c %+.3e   %.3e\n\n",
        qps->iteration, Lag, Lag-Lag_old, t2, flag?'<':'>', smalxe->update_threshold, t) );
  }
  if (flag && M1_update != 1.0) {
    if (smalxe->inner->reason != KSP_CONVERGED_ATOL) {
      TRY( PetscInfo(qps,"not updating M1 as the inner solver has not returned due to M1\n") );
    } else {
      M1_new = smalxe->M1 / M1_update;
      {
        TRY( PetscInfo3(qps,"updating M1 := M1/M1_update = %.4e/%.4e = %.4e\n",smalxe->M1,M1_update,M1_new) );
        smalxe->M1 = M1_new;                                /* M1 = M1 / M1_update       */
        smalxe->M1_updates++;
      }
    }
  }

  if (smalxe->inner->rnorm > smalxe->enorm) {
    TRY( PetscInfo2(qps,"not updating rho because G = %.8e > %.8e = E\n",smalxe->inner->rnorm,smalxe->enorm) );
    PetscFunctionReturn(0);
  }

  TRY( QPSSMALXEUpdateRho_SMALXE(qps,flag) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSConvergedDestroy_Inner_SMALXE"
PetscErrorCode QPSConvergedDestroy_Inner_SMALXE(void *ctx)
{
  QPSConvergedCtx_Inner_SMALXE *cctx = (QPSConvergedCtx_Inner_SMALXE*) ctx;

  PetscFunctionBegin;
  TRY( PetscFree(cctx) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSConvergedCreate_Inner_SMALXE"
PetscErrorCode QPSConvergedCreate_Inner_SMALXE(QPS qps_outer, void **ctx)
{
  QPSConvergedCtx_Inner_SMALXE *cctx;

  PetscFunctionBegin;
  TRY( PetscNew(&cctx) );
  cctx->gtol = NAN;
  cctx->ttol_outer = NAN;
  cctx->qps_outer = qps_outer;
  TRY( QPSGetSolvedQP(qps_outer, &cctx->qp_outer) );
  *ctx = cctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConvergedSetUp_Inner_SMALXE"
PetscErrorCode QPSConvergedSetUp_Inner_SMALXE(QPS qps_inner)
{
  QPSConvergedCtx_Inner_SMALXE *cctx = (QPSConvergedCtx_Inner_SMALXE*) qps_inner->cnvctx;
  QPS qps_outer = cctx->qps_outer;
  Vec b_inner = qps_inner->solQP->b;
  Vec b_outer = qps_outer->solQP->b;

  PetscFunctionBegin;
  TRY( PetscInfo(qps_inner,"inner QP solver convergence criterion initialized.\n") );

  TRY( VecNorm(b_outer, NORM_2, &cctx->norm_rhs_outer) );
  cctx->gtol = qps_outer->rtol*cctx->norm_rhs_outer;
  TRY( PetscInfo3(qps_inner,"  gtol = rtol * norm_rhs_outer= %.4e * %.4e = %.4e\n",qps_outer->rtol,cctx->norm_rhs_outer,cctx->gtol) );
  cctx->ttol_outer = PetscMax(qps_outer->rtol*cctx->norm_rhs_outer, qps_outer->atol);
  TRY( PetscInfo4(qps_outer,"  ttol_outer = max(rtol_outer*norm_rhs_outer, atol_outer) = max(%.4e * %.4e, %.4e) = %.4e\n",qps_outer->rtol,cctx->norm_rhs_outer,qps_outer->atol,cctx->ttol_outer) );

  //TODO this is just a quick&dirty solution
  /* use inner b for divergence criterion of outer solver */
  TRY( QPSConvergedDefaultSetRhsForDivergence(cctx->qps_outer->cnvctx, b_inner) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConverged_Inner_SMALXE_Monitor_Outer"
PETSC_STATIC_INLINE PetscErrorCode QPSConverged_Inner_SMALXE_Monitor_Outer(QPS qps_inner,QP qp_inner,PetscInt i,PetscReal gnorm,KSPConvergedReason *reason,QPSConvergedCtx_Inner_SMALXE *cctx,PetscBool header) 
{
  QPS qps_outer = cctx->qps_outer;
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps_outer->data;
  PetscReal rho;

  PetscFunctionBegin;
  if (i && !(*reason))  PetscFunctionReturn(0);
  if (smalxe->monitor_outer) {
    TRY( MatPenalizedGetPenalty(qp_inner->A, &rho) );
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps_outer),"out %3d:   onorm = %.8e,   M1 = %.3e,   rho = %.3e,   eta = %.3e,   rtol_E = %.3e\n", qps_outer->iteration, qps_outer->rnorm, smalxe->M1, rho, smalxe->eta, smalxe->rtol_E) );
  }
  if (header) {
    if (smalxe->monitor) {
      TRY( PetscPrintf(PetscObjectComm((PetscObject)qps_outer),"    in   G=||g||    E=||Bx||/rtol_E     max(G,E)         ttol_outer       G                min(M1||Bx||,eta)          M1||Bx||   eta        gtol\n") );
    } else if (smalxe->monitor_excel && !smalxe->inner_iter_accu) {
      TRY( PetscPrintf(PetscObjectComm((PetscObject)qps_outer),"in_ac    in        ||g||       ||Bx||    out           M           rho               Lag\n") );
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConverged_Inner_SMALXE_Monitor_Inner"
PETSC_STATIC_INLINE PetscErrorCode QPSConverged_Inner_SMALXE_Monitor_Inner(QPS qps_inner,QP qp_inner,PetscInt i,PetscReal gnorm,KSPConvergedReason *reason,QPSConvergedCtx_Inner_SMALXE *cctx)  
{
  QPS qps_outer = cctx->qps_outer;
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps_outer->data;
  char stepType;

  PetscFunctionBegin;
  TRY( QPSMPGPGetCurrentStepType(qps_inner,&stepType) );
  if (smalxe->monitor) {
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps_outer),"  %4d %c %.4e   %.4e    %c = %.8e %c %.8e   %.8e %c %.8e = %-8s  %.3e  %.3e  %.3e\n",
        i, stepType, gnorm, smalxe->enorm, (gnorm > smalxe->enorm)?'G':'E', qps_outer->rnorm, (qps_outer->rnorm < cctx->ttol_outer)?'<':'>', cctx->ttol_outer,
        gnorm, (gnorm<qps_inner->atol)?'<':'>', qps_inner->atol, (cctx->MNormBu<smalxe->eta)?"M1||Bu||":"eta", cctx->MNormBu, smalxe->eta, cctx->gtol) );
  } else if (smalxe->monitor_excel) {
    PetscReal rho,Lag;
    TRY( MatPenalizedGetPenalty(qp_inner->A,&rho) );
    TRY( QPComputeObjective(qps_outer->solQP,qps_outer->solQP->x,&Lag) );
    TRY( PetscPrintf(PetscObjectComm((PetscObject)qps_outer),"%5d  %4d   %.4e   %.4e   %4d   %.4e   %.4e   %.8e\n",
        smalxe->inner_iter_accu+i, i, gnorm, smalxe->normBu, qps_outer->iteration, smalxe->M1, rho, Lag) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConverged_Inner_SMALXE"
PetscErrorCode QPSConverged_Inner_SMALXE(QPS qps_inner,QP qp_inner,PetscInt i,PetscReal gnorm,KSPConvergedReason *reason,void *ctx)
{
  QPSConvergedCtx_Inner_SMALXE *cctx = (QPSConvergedCtx_Inner_SMALXE*) ctx;
  QPS qps_outer = cctx->qps_outer;
  QPS_SMALXE *smalxe = (QPS_SMALXE*)qps_outer->data;
  QP qp_outer = cctx->qp_outer;
  Vec u = qp_inner->x;
  MPI_Comm comm;
  
  PetscFunctionBegin;
  FLLOP_ASSERT(qp_inner->parent == qp_outer,"qp_inner->parent == qp_outer");
  FLLOP_ASSERT(qp_inner->x == qp_outer->x, "qp_inner->x == qp_outer->x");
  TRY( PetscObjectGetComm((PetscObject)qps_inner,&comm) );
  *reason = KSP_CONVERGED_ITERATING;

  TRY( smalxe->updateNormBu(qps_outer,u,&smalxe->normBu,&smalxe->enorm) );
  qps_outer->rnorm = PetscMax(smalxe->enorm,gnorm);
  cctx->MNormBu = smalxe->M1 * smalxe->normBu;
  qps_inner->atol = PetscMin(cctx->MNormBu, smalxe->eta);
  
  TRY( QPSConverged_Inner_SMALXE_Monitor_Outer(qps_inner,qp_inner,i,gnorm,reason,cctx,PETSC_TRUE) );
  TRY( QPSConverged_Inner_SMALXE_Monitor_Inner(qps_inner,qp_inner,i,gnorm,reason,cctx) );
  
  if (i > qps_inner->max_it - smalxe->inner_iter_accu) {
    *reason = KSP_DIVERGED_ITS;
    qps_outer->reason = KSP_DIVERGED_BREAKDOWN;
    TRY( PetscInfo(qps_inner,"Inner QP solver is diverging (iteration count reached the maximum).\n") );
    TRY( PetscInfo2(qps_inner,"Current residual norm %14.12e at inner iteration %D\n",(double)gnorm,i) );
    TRY( QPSConverged_Inner_SMALXE_Monitor_Outer(qps_inner,qp_inner,i,gnorm,reason,cctx,PETSC_FALSE) );
    PetscFunctionReturn(0);
  }

  if (PetscIsInfOrNanScalar(gnorm)) {
    *reason = KSP_DIVERGED_NANORINF;
    qps_outer->reason = KSP_DIVERGED_BREAKDOWN;
    TRY( PetscInfo(qps_inner,"Inner QP solver has created a not a number (NaN) as the residual norm, declaring divergence.\n") );
    TRY( QPSConverged_Inner_SMALXE_Monitor_Outer(qps_inner,qp_inner,i,gnorm,reason,cctx,PETSC_FALSE) );
    PetscFunctionReturn(0);
  }

  
  TRY( (*qps_outer->convergencetest)(qps_outer,qp_outer,qps_outer->iteration,qps_outer->rnorm,&qps_outer->reason,qps_outer->cnvctx) );

  if (qps_outer->reason) {
    if (qps_outer->reason > 0) {
      *reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      TRY( PetscInfo(qps_inner,"Inner QP solver has converged due to convergence of the outer solver.\n") );
    } else {
      *reason = KSP_DIVERGED_BREAKDOWN;
      TRY( PetscInfo(qps_inner,"Inner QP solver has diverged due to divergence of the outer solver.\n") );
    }
    TRY( QPSConverged_Inner_SMALXE_Monitor_Outer(qps_inner,qp_inner,i,gnorm,reason,cctx,PETSC_FALSE) );
    PetscFunctionReturn(0);
  }

  if (gnorm < qps_inner->atol) {
    TRY( PetscInfo4(qps_inner,"Inner QP solver has converged. Residual norm gnorm=%.8e is less than atol = min(M1*||Bu||),eta) = %s = %.8e at iteration %D.\n",gnorm,(cctx->MNormBu<smalxe->eta)?"M1||Bu||":"eta",qps_inner->atol,i) );
    *reason = KSP_CONVERGED_ATOL;
    if (cctx->MNormBu < smalxe->eta) {
      smalxe->M1_hits++;
    } else {
      smalxe->eta_hits++;
    }
    PetscFunctionReturn(0);
  }

  if (smalxe->state == 3 && (i < smalxe->inner_iter_min || smalxe->inner_no_gtol_stop)) {
    PetscFunctionReturn(0);
  }

  if (gnorm <= cctx->gtol) {
    if (smalxe->inner->rnorm > smalxe->enorm) {
      TRY( PetscInfo2(qps_inner,"skipping gtol criterion because G = %.8e > %.8e = E\n",smalxe->inner->rnorm,smalxe->enorm) );
    } else {
      if (smalxe->inner_no_gtol_stop < 2) {
        TRY( PetscInfo3(qps_inner,"Inner QP solver has converged. Residual norm gnorm=%.8e is less than gtol = %.8e at iteration %D\n",gnorm,cctx->gtol,i) );
        *reason = KSP_CONVERGED_RTOL;
      } else {
        TRY( PetscInfo3(qps_inner,"Gradient tolerance has been reached. Residual norm gnorm=%.8e is less than gtol = %.8e at iteration %D\n",gnorm,cctx->gtol,i) );
      }
      if (smalxe->state != 3) {
        TRY( PetscInfo( qps_inner,"changing rho_update_type to 3\n") );
        smalxe->state = 3;
      }
    }
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "QPSSetFromOptions_SMALXE"
#if PETSC_VERSION_MINOR<6
PetscErrorCode QPSSetFromOptions_SMALXE(QPS qps)
#else
PetscErrorCode QPSSetFromOptions_SMALXE(PetscOptionItems *PetscOptionsObject,QPS qps)
#endif
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  PetscBool flg1,flg2,eta_direct,rho_direct,M1_direct;
  PetscReal maxeig,maxeig_tol,eta,rho,M1,M1_update,rho_update,rho_update_late;
  PetscInt maxeig_iter;

  PetscFunctionBegin;
  TRY( PetscOptionsHead(PetscOptionsObject,"QPSSMALXE options") );
  TRY( PetscOptionsReal("-qps_smalxe_maxeig","Approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means this is automatically computed.","QPSSMALXESetOperatorMaxEigenValue",smalxe->maxeig,&maxeig,&flg1) );
  if (flg1) TRY( QPSSMALXESetOperatorMaxEigenvalue(qps,maxeig) );
  TRY( PetscOptionsReal("-qps_smalxe_maxeig_tol","Relative tolerance to find approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means QPS rtol*100","QPSSMALXESetOperatorMaxEigenvalueTolerance",smalxe->maxeig_tol,&maxeig_tol,&flg1) );
  if (flg1) TRY( QPSSMALXESetOperatorMaxEigenvalueTolerance(qps,maxeig_tol) );
  TRY( PetscOptionsInt("-qps_smalxe_maxeig_iter","Number of iterations to find an approximate maximum eigenvalue of the Hessian","QPSSMALXESetOperatorMaxEigenValueIterations",smalxe->maxeig_iter,&maxeig_iter,&flg1) );
  if (flg1) TRY( QPSSMALXESetOperatorMaxEigenvalueIterations(qps,maxeig_iter) );
  TRY( PetscOptionsBool("-qps_smalxe_maxeig_inject","","QPSSMALXESetInjectOperatorMaxEigenvalue",smalxe->inject_maxeig,&flg2,&smalxe->inject_maxeig_set) );
  if (smalxe->inject_maxeig_set) TRY( QPSSMALXESetInjectOperatorMaxEigenvalue(qps,flg2) );

  eta = smalxe->eta_user;
  eta_direct = PETSC_FALSE;
  TRY( PetscOptionsBool("-qps_smalxe_eta_direct","","QPSSMALXESetEta",(PetscBool) smalxe->eta_type,&eta_direct,&flg1) );
  TRY( PetscOptionsReal("-qps_smalxe_eta","","QPSSMALXESetEta",smalxe->eta_user,&eta,&flg2) );
  if (flg1 || flg2) TRY( QPSSMALXESetEta(qps,eta,(QPSScalarArgType) eta_direct) );

  rho = smalxe->rho_user;
  rho_direct = PETSC_FALSE;
  TRY( PetscOptionsBool("-qps_smalxe_rho_direct","","QPSSMALXESetRhoInitial",(PetscBool) smalxe->rho_type,&rho_direct,&flg1) );
  TRY( PetscOptionsReal("-qps_smalxe_rho","","QPSSMALXESetRhoInitial",smalxe->rho_user,&rho,&flg2) );
  if (flg1 || flg2) TRY( QPSSMALXESetRhoInitial(qps,rho,(QPSScalarArgType) rho_direct) );
  TRY( PetscOptionsReal("-qps_smalxe_rho_update","","QPSSMALXESetRhoUpdate",smalxe->rho_update,&rho_update,&flg1) );
  if (flg1) TRY( QPSSMALXESetRhoUpdate(qps,rho_update) );
  TRY( PetscOptionsReal("-qps_smalxe_rho_update_late","","QPSSMALXESetRhoUpdateLate",smalxe->rho_update_late,&rho_update_late,&flg1) );
  if (flg1) TRY( QPSSMALXESetRhoUpdateLate(qps,rho_update_late) );

  M1 = smalxe->M1_user;
  M1_direct = PETSC_FALSE;
  TRY( PetscOptionsBool("-qps_smalxe_M1_direct","","QPSSMALXESetM1Initial",smalxe->M1_type,&M1_direct,&flg1) );
  TRY( PetscOptionsReal("-qps_smalxe_M1","","QPSSMALXESetM1Initial",smalxe->M1_user,&M1,&flg2) );
  if (flg1 || flg2) TRY( QPSSMALXESetM1Initial(qps,M1,M1_direct) );
  TRY( PetscOptionsReal("-qps_smalxe_M1_update","","QPSSMALXESetM1Update",smalxe->M1_update,&M1_update,&flg1) );
  if (flg1) TRY( QPSSMALXESetM1Update(qps,M1_update) );

  //TODO impl. setter function
  TRY( PetscOptionsReal("-qps_smalxe_rtol_E","Ratio between desired ||B*x|| and ||g|| (norm of projected gradient of inner problem)","",smalxe->rtol_E,&smalxe->rtol_E,NULL) );

  TRY( PetscOptionsBool("-qps_smalxe_get_lambda","","",smalxe->get_lambda,&smalxe->get_lambda,NULL) );
  TRY( PetscOptionsBool("-qps_smalxe_get_Bt_lambda","","",smalxe->get_Bt_lambda,&smalxe->get_Bt_lambda,NULL) );

  //TODO temporary
  TRY( PetscOptionsBoolGroupBegin("-qps_smalxe_monitor","","QPSSMALXESetMonitor",&smalxe->monitor) );
  TRY( PetscOptionsBoolGroupEnd(  "-qps_smalxe_monitor_excel","","",&smalxe->monitor_excel) );
  if (smalxe->monitor || smalxe->monitor_excel) smalxe->monitor_outer = PETSC_TRUE;
  TRY( PetscOptionsBool("-qps_smalxe_monitor_outer","","QPSSMALXESetMonitor",smalxe->monitor_outer,&smalxe->monitor_outer,NULL) );
  TRY( PetscOptionsInt( "-qps_smalxe_inner_iter_min","","",smalxe->inner_iter_min,&smalxe->inner_iter_min,NULL) );
  TRY( PetscOptionsInt( "-qps_smalxe_inner_no_gtol_stop","","",smalxe->inner_no_gtol_stop,&smalxe->inner_no_gtol_stop,NULL) );
  TRY( PetscOptionsReal("-qps_smalxe_update_threshold","","",smalxe->update_threshold,&smalxe->update_threshold,NULL) );
  TRY( PetscOptionsInt( "-qps_smalxe_offset","","",smalxe->offset,&smalxe->offset,NULL) );
  //
  TRY( PetscOptionsBool("-qps_smalxe_norm_update_lag","","",smalxe->lag_enabled,&smalxe->lag_enabled,NULL) );
  TRY( PetscOptionsBoolGroupBegin("-qps_smalxe_norm_update_lag_monitor","","",&smalxe->lag_monitor) );
  TRY( PetscOptionsBoolGroupEnd("-qps_smalxe_norm_update_lag_compare","","",&smalxe->lag_compare ) );
  TRY( PetscOptionsInt("-qps_smalxe_norm_update_lag_offset","","",smalxe->norm_update_lag_offset,&smalxe->norm_update_lag_offset,NULL) );
  TRY( PetscOptionsInt("-qps_smalxe_norm_update_lag_start","","",smalxe->Jstart,&smalxe->Jstart,NULL) );
  TRY( PetscOptionsInt("-qps_smalxe_norm_update_lag_step","","",smalxe->Jstep,&smalxe->Jstep,NULL) );
  TRY( PetscOptionsInt("-qps_smalxe_norm_update_lag_end","","",smalxe->Jend,&smalxe->Jend,NULL) );
  TRY( PetscOptionsReal("-qps_smalxe_norm_update_lag_lower","","",smalxe->lower,&smalxe->lower,NULL) );
  TRY( PetscOptionsReal("-qps_smalxe_norm_update_lag_upper","","",smalxe->upper,&smalxe->upper,NULL) );

  smalxe->setfromoptionscalled = PETSC_TRUE;
  TRY( PetscOptionsTail() );
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QPSSetUp_SMALXE"
PetscErrorCode QPSSetUp_SMALXE(QPS qps)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  QP            qp,qp_inner;
  Mat           A,B,A_inner;
  Vec           c,b_inner;
  PetscReal     maxeig_inner, rho;

  PetscFunctionBegin;
  TRY( QPSReset_SMALXE(qps) );

  qp = qps->solQP;
  if (qp->cE) {
    TRY( PetscInfo(qps, "nonzero lin. eq. con. RHS prescribed ==> automatically calling QPTHomogenizeEq\n") );
    TRY( QPTHomogenizeEq(qp) );
    TRY( QPChainGetLast(qp,&qps->solQP) );
    qp = qps->solQP;
  }

  /* get the original Hessian */
  TRY( QPGetOperator(qp, &A) );

  /* get the linear equality constraints */
  TRY( QPGetEq(qp, &B, &c) );

  /* initialize work vectors */
  TRY( QPSSetWorkVecs(qps,1) );
  TRY( MatCreateVecs(B, NULL, &smalxe->Bu) );
  TRY( VecZeroEntries(qp->lambda_E) );
  TRY( VecZeroEntries(qp->Bt_lambda) );
  
  /* initialize parameter eta */
  smalxe->eta = smalxe->eta_user;
  if (smalxe->eta_type == QPS_ARG_MULTIPLE) {
    PetscReal normb;
    TRY( VecNorm(qp->b, NORM_2, &normb) );
    smalxe->eta *= normb;
  }
  
  /* initialize parameter M1 */
  smalxe->M1_initial = smalxe->M1_user;
  if (smalxe->M1_type == QPS_ARG_MULTIPLE) {
    if (smalxe->maxeig == PETSC_DECIDE) {
      TRY( MatGetMaxEigenvalue(A, NULL, &smalxe->maxeig, smalxe->maxeig_tol, smalxe->maxeig_iter) );
    }
    smalxe->M1_initial *= smalxe->maxeig;
  }

  /* initialize penalty rho */
  if (smalxe->rho_type == QPS_ARG_MULTIPLE) {
    if (smalxe->maxeig == PETSC_DECIDE) {
      TRY( MatGetMaxEigenvalue(A, NULL, &smalxe->maxeig, smalxe->maxeig_tol, smalxe->maxeig_iter) );
    }
    rho = smalxe->rho_user * smalxe->maxeig;
  } else {
    rho = smalxe->rho_user;
  }

  TRY( PetscInfo3(qps,"   eta=%.8e eta_user=%.8e eta_type=%c\n",smalxe->eta,smalxe->eta_user,smalxe->eta_type==QPS_ARG_DIRECT?'D':'M') );
  TRY( PetscInfo1(qps,"maxeig=%.8e\n",smalxe->maxeig) );
  TRY( PetscInfo3(qps,"    M1=%.8e  M1_user=%.8e  M1_type=%c\n",smalxe->M1_initial,smalxe->M1_user,smalxe->M1_type==QPS_ARG_DIRECT?'D':'M') );
  TRY( PetscInfo3(qps,"   rho=%.8e rho_user=%.8e rho_type=%c\n",rho,smalxe->rho_user,smalxe->rho_type==QPS_ARG_DIRECT?'D':'M') );

  /* explicitly setup projector factory, e.g. to set its inner G_has_orthonormal_rows flag */
  TRY( QPPFSetUp(qp->pf) );

  /* setup QP with eq. constraints eliminated for inner loop */
  TRY( QPTEnforceEqByPenalty(qp, rho) );
  TRY( QPChainGetLast(qp,&smalxe->qp_penalized) );
  qp_inner = smalxe->qp_penalized;

  {
    PetscErrorCode(*transform)(QP);
    TRY( QPGetTransform(qp_inner,&transform) );
    FLLOP_ASSERT(qp_inner->parent == qp,"qp_inner->parent == qp");
    FLLOP_ASSERT(transform == (PetscErrorCode(*)(QP))QPTEnforceEqByPenalty,"transform == QPTEnforceEqByRho");
  }

  /* make independent copy b_inner of the original rhs b to allow updates of b_inner without touching b */
  TRY( VecDuplicate(qp->b, &b_inner) );
  TRY( VecCopy(qp->b, b_inner) );
  TRY( QPSetRhs(qp_inner, b_inner) );
  TRY( VecDestroy(&b_inner) );

  /* inject the QP with penalized Hessian into inner solver */
  TRY( QPSSetQP(smalxe->inner, qp_inner) );

  if (smalxe->setfromoptionscalled) {
    TRY( QPSSetFromOptions(smalxe->inner) );
  } else {
    TRY( QPSSetDefaultTypeIfNotSpecified(smalxe->inner) );
  }
  
  /* if the inner solver is MPGP, it inherits maximum operator eigenvalue maxeig_inner */
  maxeig_inner = PetscMax(rho, smalxe->maxeig);
  if (!smalxe->inject_maxeig_set) {
    TRY( QPPFGetGHasOrthonormalRows(qp->pf,&smalxe->inject_maxeig) );
  }
  TRY( PetscInfo2(qps,"maximum operator eigenvalue estimate %.8e is %sinjected to the inner solver\n",maxeig_inner,smalxe->inject_maxeig?"":"NOT ") );
  if (smalxe->inject_maxeig) TRY( QPSMPGPSetOperatorMaxEigenvalue(smalxe->inner, maxeig_inner) );
  
  TRY( QPSSetUp(smalxe->inner) );
  
  /* inject the special stopping criterion to the inner loop solver */
  TRY( QPSConvergedCreate_Inner_SMALXE(qps, (void**)&smalxe->cctx_inner) );
  TRY( QPSSetConvergenceTest(smalxe->inner, QPSConverged_Inner_SMALXE, smalxe->cctx_inner, QPSConvergedDestroy_Inner_SMALXE) );

  /* choose function updating BtBu and normBu */
  if (qp->BE->ops->mult) {
    smalxe->updateNormBu = QPSSMALXEUpdateNormBu_SMALXE;
  } else {
    TRY( VecInvalidate(qps->solQP->lambda_E) );
    if (smalxe->lag_enabled) {
      smalxe->updateNormBu = QPSSMALXEUpdateNormBu_Lag_SMALXEON;
    } else {
      smalxe->updateNormBu = QPSSMALXEUpdateNormBu_SMALXEON;
    }
  }

  TRY( QPGetOperator(qp_inner, &A_inner) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolve_SMALXE"
/* qps->rtol is eta */
PetscErrorCode QPSSolve_SMALXE(QPS qps)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  QPS           qps_inner = smalxe->inner;
  QP            qp, qp_inner;
  Mat           A_inner;
  Vec           b,b_inner,u,Btmu;
  PetscReal     Lag, Lag_old, rho;
  PetscInt      i,it_inner,maxits;
  PetscErrorCode(*transform)(QP);

  PetscFunctionBegin;
  qp            = qps->solQP;
  qp_inner      = smalxe->qp_penalized;
  smalxe->M1    = smalxe->M1_initial;
  Btmu          = qp->Bt_lambda;
  maxits        = qps->max_it;
  it_inner      = 0;

  FLLOP_ASSERT(qp_inner->parent == qp,"qp_inner->parent == qp");
  TRY( QPGetTransform(qp_inner,&transform) );
  if (transform != (PetscErrorCode(*)(QP))QPTEnforceEqByPenalty) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_ARG_WRONGSTATE,"last QP transform must be QPTEnforceEqByPenalty");

  TRY( QPGetRhs(qp, &b) );
  TRY( QPGetSolutionVector(qp, &u) );
  TRY( QPGetOperator(qp_inner, &A_inner) );
  TRY( QPGetRhs(qp_inner, &b_inner) );

  /* store initial value of penalty */
  TRY( MatPenalizedGetPenalty(A_inner, &rho) );

  /* initialize Btmu as zero vector */
  TRY( VecZeroEntries(Btmu) );

  /* compute initial value of Lagrangian */
  TRY( QPComputeObjective(qp_inner,u,&Lag_old) );

  /* update BtBu and normBu */
  TRY( smalxe->updateNormBu(qps,u,&smalxe->normBu_old,&smalxe->enorm) );
  smalxe->normBu_prev = smalxe->normBu_old;

  qps->iteration = 0;
  smalxe->inner_iter_accu = 0;

  for (i=0; i<maxits; i++) {
    qps->iteration = i+1;

    /* update Btmu (eq. con. multiplier pre-multiplied by eq. con. matrix transpose) */
    TRY( QPSSMALXEUpdateLambda_SMALXE(qps,rho) );

    /* inner solver can set the convergence reason of the outer solver so check it */
    if (qps->reason) break;

    /* update the inner RHS b_inner=b-Btmu */
    TRY( VecWAXPY(b_inner, -1.0, Btmu, b) );

    /* call inner solver with custom stopping criterion */
    qps_inner->divtol = qps->divtol;
    TRY( QPSConvergedSetUp_Inner_SMALXE(qps_inner) );
    TRY( QPSSolve(qps_inner) );
    TRY( QPSGetIterationNumber(qps_inner, &it_inner) );
    smalxe->inner_iter_accu += it_inner+1;

    /* update BtBu and normBu */
    TRY( smalxe->updateNormBu(qps,u,&smalxe->normBu,&smalxe->enorm) );

    /* store rho used in inner solve before update */
    TRY( MatPenalizedGetPenalty(A_inner, &rho) );

    /* compute current value of Lagrangian */
    TRY( QPComputeObjective(qp_inner,u,&Lag) );

    /* update M1, rho if needed */
    TRY( QPSSMALXEUpdate_SMALXE(qps,Lag_old,Lag,rho) );
    Lag_old = Lag;
    smalxe->normBu_old = smalxe->normBu;
  }
  if (i == maxits) {
    TRY( PetscInfo1(qps,"Maximum number of iterations has been reached: %D\n",maxits) );
    if (!qps->reason) qps->reason = KSP_DIVERGED_ITS;
  }

  if (smalxe->get_lambda) {
    TRY( QPPFApplyHalfQ(qp->pf,qp->Bt_lambda,qp->lambda) );
  }
  if (!smalxe->get_Bt_lambda) {
    TRY( VecInvalidate(qp->Bt_lambda) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_SMALXE"
PetscErrorCode QPSViewConvergence_SMALXE(QPS qps, PetscViewer v)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;
  PetscBool     iascii;
  const QPSType qpstype;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (iascii) {
    TRY( PetscViewerASCIIPrintf(v,"Total number of inner iterations %d\n",smalxe->inner_iter_accu) );
    TRY( PetscViewerASCIIPrintf(v,"#hits    of M1, eta: %3d, %3d\n",smalxe->M1_hits,smalxe->eta_hits) );
    TRY( PetscViewerASCIIPrintf(v,"#updates of M1, rho: %3d, %3d\n",smalxe->M1_updates,smalxe->rho_updates) );

    TRY( QPSGetType(smalxe->inner, &qpstype) );
    TRY( PetscViewerASCIIPrintf(v,"inner ") );
    TRY( QPSViewConvergence(smalxe->inner,v) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSReset_SMALXE"
PetscErrorCode QPSReset_SMALXE(QPS qps)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;

  PetscFunctionBegin;
  if (smalxe->qp_penalized) TRY( QPRemoveChild(smalxe->qp_penalized->parent) );
  smalxe->qp_penalized = NULL;
  smalxe->normBu                = NAN;
  smalxe->enorm                 = NAN;
  smalxe->state                 = 1;
  smalxe->inner_iter_accu       = 0;
  smalxe->M1_updates            = 0;
  smalxe->M1_hits               = 0;
  smalxe->eta_hits              = 0;
  smalxe->rho_updates           = 0;
  TRY( VecDestroy(&smalxe->Bu) );
  TRY( QPSReset(smalxe->inner) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSDestroy_SMALXE"
PetscErrorCode QPSDestroy_SMALXE(QPS qps)
{
  QPS_SMALXE    *smalxe = (QPS_SMALXE*)qps->data;

  PetscFunctionBegin;
  TRY( QPSReset_SMALXE(qps) );
  TRY( QPSDestroy(&smalxe->inner) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetOperatorMaxEigenvalue_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetOperatorMaxEigenvalue_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetM1Initial_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetM1Initial_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetM1Update_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetM1Update_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetEta_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetEta_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoInitial_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoInitial_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoUpdate_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoUpdate_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoUpdate2_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoUpdate2_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoUpdateLate_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoUpdateLate_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetInjectOperatorMaxEigenvalue_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetInjectOperatorMaxEigenvalue_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetOperatorMaxEigenvalueIterations_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetOperatorMaxEigenvalueIterations_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetOperatorMaxEigenvalueTolerance_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetOperatorMaxEigenvalueTolerance_SMALXE_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetMonitor_SMALXE_C",NULL) );
  TRY( QPSDestroyDefault(qps) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSIsQPCompatible_SMALXE"
PetscErrorCode QPSIsQPCompatible_SMALXE(QPS qps,QP qp,PetscBool *flg)
{
  Mat Beq,Bineq;
  Vec ceq;
  
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  TRY( QPGetEq(qp,&Beq,&ceq) );
  TRY( QPGetIneq(qp,&Bineq,NULL) );
  if (!Beq || Bineq) {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSCreate_SMALXE"
FLLOP_EXTERN PetscErrorCode QPSCreate_SMALXE(QPS qps)
{
  QPS_SMALXE      *smalxe;
  MPI_Comm        comm;
  
  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)qps,&comm) );
  TRY( PetscNewLog(qps,&smalxe) );
  qps->data                  = (void*)smalxe;
  
  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  qps->ops->reset            = QPSReset_SMALXE;
  qps->ops->setup            = QPSSetUp_SMALXE;
  qps->ops->solve            = QPSSolve_SMALXE;
  qps->ops->destroy          = QPSDestroy_SMALXE;
  qps->ops->isqpcompatible   = QPSIsQPCompatible_SMALXE;
  qps->ops->setfromoptions   = QPSSetFromOptions_SMALXE;
  qps->ops->viewconvergence  = QPSViewConvergence_SMALXE;

  smalxe->updateNormBu       = QPSSMALXEUpdateNormBu_SMALXE;

  /* set type-specific functions */
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetOperatorMaxEigenvalue_SMALXE_C",QPSSMALXEGetOperatorMaxEigenvalue_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetOperatorMaxEigenvalue_SMALXE_C",QPSSMALXESetOperatorMaxEigenvalue_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetM1Initial_SMALXE_C",QPSSMALXEGetM1Initial_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetM1Initial_SMALXE_C",QPSSMALXESetM1Initial_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetM1Update_SMALXE_C",QPSSMALXEGetM1Update_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetM1Update_SMALXE_C",QPSSMALXESetM1Update_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetEta_SMALXE_C",QPSSMALXEGetEta_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetEta_SMALXE_C",QPSSMALXESetEta_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoInitial_SMALXE_C",QPSSMALXEGetRhoInitial_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoInitial_SMALXE_C",QPSSMALXESetRhoInitial_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoUpdate_SMALXE_C",QPSSMALXEGetRhoUpdate_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoUpdate_SMALXE_C",QPSSMALXESetRhoUpdate_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetRhoUpdateLate_SMALXE_C",QPSSMALXEGetRhoUpdateLate_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetRhoUpdateLate_SMALXE_C",QPSSMALXESetRhoUpdateLate_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetInjectOperatorMaxEigenvalue_SMALXE_C",QPSSMALXEGetInjectOperatorMaxEigenvalue_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetInjectOperatorMaxEigenvalue_SMALXE_C",QPSSMALXESetInjectOperatorMaxEigenvalue_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetOperatorMaxEigenvalueIterations_SMALXE_C",QPSSMALXEGetOperatorMaxEigenvalueIterations_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetOperatorMaxEigenvalueIterations_SMALXE_C",QPSSMALXESetOperatorMaxEigenvalueIterations_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetOperatorMaxEigenvalueTolerance_SMALXE_C",QPSSMALXESetOperatorMaxEigenvalueTolerance_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXEGetOperatorMaxEigenvalueTolerance_SMALXE_C",QPSSMALXEGetOperatorMaxEigenvalueTolerance_SMALXE) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSSMALXESetMonitor_SMALXE_C",QPSSMALXESetMonitor_SMALXE) );

  /* initialize inner data */
  smalxe->inner = NULL;
  smalxe->cctx_inner = NULL;
  smalxe->setfromoptionscalled  = PETSC_FALSE;
  /* inner data that should be reinitialized in QPSReset */
  smalxe->Bu = NULL;
  smalxe->qp_penalized          = NULL;
  smalxe->normBu                = NAN;
  smalxe->normBu_old            = NAN;
  smalxe->normBu_prev           = NAN;
  smalxe->enorm                 = NAN;
  smalxe->state                 = 1;
  smalxe->inner_iter_accu       = 0;
  smalxe->rho_updates           = 0;
  
  /* set default values of parameters */
  smalxe->M1_user     = 1e2;
  smalxe->M1_type     = QPS_ARG_MULTIPLE;
  smalxe->M1_update   = 2.0;
  smalxe->M1_updates  = 0;
  smalxe->M1_hits     = 0;

  smalxe->rtol_E      = 1e-0;

  smalxe->rho_user    = 1.1;
  smalxe->rho_type    = QPS_ARG_MULTIPLE;
  smalxe->rho_update = 1.0;
  smalxe->rho_update_late = 2.0;

  smalxe->eta_user    = 1e-1;
  smalxe->eta_type    = QPS_ARG_MULTIPLE;
  smalxe->eta_hits    = 0;

  smalxe->update_threshold    = 0.0;

  smalxe->maxeig              = PETSC_DECIDE;
  smalxe->maxeig_tol          = PETSC_DECIDE;
  smalxe->maxeig_iter         = PETSC_DECIDE;
  smalxe->inject_maxeig       = PETSC_FALSE;
  smalxe->inject_maxeig_set   = PETSC_FALSE;

  smalxe->monitor             = PETSC_FALSE;
  smalxe->monitor_outer       = PETSC_FALSE;
  smalxe->monitor_excel       = PETSC_FALSE;

  smalxe->get_lambda          = PETSC_FALSE;
  smalxe->get_Bt_lambda       = PETSC_TRUE;
  smalxe->offset              = 5;

  smalxe->lag_enabled = PETSC_FALSE;
  smalxe->lag_monitor = PETSC_FALSE;
  smalxe->lag_compare = PETSC_FALSE;
  smalxe->offset      = 2;
  smalxe->Jstart      = 10;
  smalxe->Jstep       = 5;
  smalxe->Jend        = 20;
  smalxe->lower       = 0.1;
  smalxe->upper       = 1.1;

  /* set SMALXE-specific default maximum number of outer iterations */
  qps->max_it = 100;

  TRY( QPSCreate(comm, &smalxe->inner) );
  TRY( QPSSetOptionsPrefix(smalxe->inner, "smalxe_") );
  TRY( QPSSetAutoPostSolve(smalxe->inner, PETSC_FALSE) );
  smalxe->inner_iter_min      = 1;
  smalxe->inner_no_gtol_stop  = PETSC_FALSE;
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetOperatorMaxEigenvalue"
PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalue(QPS qps,PetscReal *maxeig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(maxeig,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetOperatorMaxEigenvalue_SMALXE_C",(QPS,PetscReal*),(qps,maxeig)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetOperatorMaxEigenvalue"
PetscErrorCode QPSSMALXESetOperatorMaxEigenvalue(QPS qps,PetscReal maxeig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,maxeig,2);
  if (maxeig <= 0 && maxeig != PETSC_DECIDE) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  TRY( PetscTryMethod(qps,"QPSSMALXESetOperatorMaxEigenvalue_SMALXE_C",(QPS,PetscReal),(qps,maxeig)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetM1Initial"
PetscErrorCode QPSSMALXEGetM1Initial(QPS qps,PetscReal *M1_initial,QPSScalarArgType *argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  if (M1_initial) PetscValidPointer(M1_initial,2);
  if (argtype) PetscValidPointer(argtype,3);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetM1Initial_SMALXE_C",(QPS,PetscReal*,QPSScalarArgType*),(qps,M1_initial,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetM1Initial"
PetscErrorCode QPSSMALXESetM1Initial(QPS qps,PetscReal M1_initial,QPSScalarArgType argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,M1_initial,2);
  PetscValidLogicalCollectiveEnum(qps,argtype,3);
  if (M1_initial <= 0) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  TRY( PetscTryMethod(qps,"QPSSMALXESetM1Initial_SMALXE_C",(QPS,PetscReal,QPSScalarArgType),(qps,M1_initial,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetEta"
PetscErrorCode QPSSMALXEGetEta(QPS qps,PetscReal *eta,QPSScalarArgType *argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  if (eta) PetscValidPointer(eta,2);
  if (argtype) PetscValidPointer(argtype,3);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetEta_SMALXE_C",(QPS,PetscReal*,QPSScalarArgType*),(qps,eta,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetEta"
PetscErrorCode QPSSMALXESetEta(QPS qps,PetscReal eta,QPSScalarArgType argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,eta,2);
  PetscValidLogicalCollectiveEnum(qps,argtype,3);
  if (eta <= 0) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  TRY( PetscTryMethod(qps,"QPSSMALXESetEta_SMALXE_C",(QPS,PetscReal,QPSScalarArgType),(qps,eta,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetRhoInitial"
PetscErrorCode QPSSMALXEGetRhoInitial(QPS qps,PetscReal *rho_initial,QPSScalarArgType *argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  if (rho_initial) PetscValidPointer(rho_initial,2);
  if (argtype) PetscValidPointer(argtype,3);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetRhoInitial_SMALXE_C",(QPS,PetscReal*,QPSScalarArgType*),(qps,rho_initial,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetRhoInitial"
PetscErrorCode QPSSMALXESetRhoInitial(QPS qps,PetscReal rho_initial,QPSScalarArgType argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,rho_initial,2);
  PetscValidLogicalCollectiveEnum(qps,argtype,3);
  if (rho_initial <= 0) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be positive");
  TRY( PetscTryMethod(qps,"QPSSMALXESetRhoInitial_SMALXE_C",(QPS,PetscReal,QPSScalarArgType),(qps,rho_initial,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetM1Update"
PetscErrorCode QPSSMALXEGetM1Update(QPS qps,PetscReal *M1_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(M1_update,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetM1Update_SMALXE_C",(QPS,PetscReal*),(qps,M1_update)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetM1Update"
PetscErrorCode QPSSMALXESetM1Update(QPS qps,PetscReal M1_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,M1_update,2);
  //if (M1_update < 1) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be >= 1");
  TRY( PetscTryMethod(qps,"QPSSMALXESetM1Update_SMALXE_C",(QPS,PetscReal),(qps,M1_update)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetRhoUpdate"
PetscErrorCode QPSSMALXEGetRhoUpdate(QPS qps,PetscReal *rho_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(rho_update,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetRhoUpdate_SMALXE_C",(QPS,PetscReal*),(qps,rho_update)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetRhoUpdate"
PetscErrorCode QPSSMALXESetRhoUpdate(QPS qps,PetscReal rho_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,rho_update,2);
  if (rho_update < 1) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be >= 1");
  TRY( PetscTryMethod(qps,"QPSSMALXESetRhoUpdate_SMALXE_C",(QPS,PetscReal),(qps,rho_update)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetRhoUpdateLate"
PetscErrorCode QPSSMALXEGetRhoUpdateLate(QPS qps,PetscReal *rho_update_late)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(rho_update_late,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetRhoUpdateLate_SMALXE_C",(QPS,PetscReal*),(qps,rho_update_late)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetRhoUpdateLate"
PetscErrorCode QPSSMALXESetRhoUpdateLate(QPS qps,PetscReal rho_update_late)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,rho_update_late,2);
  if (rho_update_late < 1) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be >= 1");
  TRY( PetscTryMethod(qps,"QPSSMALXESetRhoUpdateLate_SMALXE_C",(QPS,PetscReal),(qps,rho_update_late)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetOperatorMaxEigenvalueIterations"
PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueIterations(QPS qps,PetscInt *numit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(numit,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetOperatorMaxEigenvalueIterations_SMALXE_C",(QPS,PetscInt*),(qps,numit)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetOperatorMaxEigenvalueIterations"
PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueIterations(QPS qps,PetscInt numit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(qps,numit,2);
  if (numit <= 1) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be > 1");
  TRY( PetscTryMethod(qps,"QPSSMALXESetOperatorMaxEigenvalueIterations_SMALXE_C",(QPS,PetscInt),(qps,numit)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetInjectOperatorMaxEigenvalue"
PetscErrorCode QPSSMALXESetInjectOperatorMaxEigenvalue(QPS qps,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(qps,flg,2);
  TRY( PetscTryMethod(qps,"QPSSMALXESetInjectOperatorMaxEigenvalue_SMALXE_C",(QPS,PetscBool),(qps,flg)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetInjectOperatorMaxEigenvalue"
PetscErrorCode QPSSMALXEGetInjectOperatorMaxEigenvalue(QPS qps,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(flg,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetInjectOperatorMaxEigenvalue_SMALXE_C",(QPS,PetscBool*),(qps,flg)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetOperatorMaxEigenvalueTolerance"
PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,tol,2);
  TRY( PetscTryMethod(qps,"QPSSMALXESetOperatorMaxEigenvalueTolerance_SMALXE_C",(QPS,PetscReal),(qps,tol)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSMALXEGetOperatorMaxEigenvalueTolerance"
PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal *tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(tol,2);
  TRY( PetscUseMethod(qps,"QPSSMALXEGetOperatorMaxEigenvalueTolerance_SMALXE_C",(QPS,PetscReal*),(qps,tol)) );
  PetscFunctionReturn(0);
}

//TODO temporary solution, monitors should be implemented more generally
#undef __FUNCT__
#define __FUNCT__ "QPSSMALXESetMonitor"
PetscErrorCode QPSSMALXESetMonitor(QPS qps,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(qps,flg,2);
  TRY( PetscTryMethod(qps,"QPSSMALXESetMonitor_SMALXE_C",(QPS,PetscBool),(qps,flg)) );
  PetscFunctionReturn(0);
}
