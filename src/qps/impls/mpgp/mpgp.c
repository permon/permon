
#include <../src/qps/impls/mpgp/mpgpimpl.h>

/* 
  WORK VECTORS:

  gP = qps->work[0];
  phi = qps->work[1];
  beta = qps->work[2];

  g  = qps->work[3];
  p  = qps->work[4];
  Ap = qps->work[5];
  gr = qps->work[6];
*/

#undef __FUNCT__
#define __FUNCT__ "QPSMonitorDefault_MPGP"
PetscErrorCode QPSMonitorDefault_MPGP(QPS qps,PetscInt n,PetscViewer viewer)
{
   QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

   PetscFunctionBegin;  
   if (n == 0 && ((PetscObject)qps)->prefix) {
     TRY( PetscViewerASCIIPrintf(viewer,"  Projected gradient norms for %s solve.\n",((PetscObject)qps)->prefix) );
   }

   TRY( PetscViewerASCIIPrintf(viewer,"%3D MPGP ||gp||=%.10e",n,(double)qps->rnorm) );
   TRY( PetscViewerASCIIPrintf(viewer,",\t||phi||=%.10e",(double)mpgp->phinorm) );
   TRY( PetscViewerASCIIPrintf(viewer,",\t||beta||=%.10e",(double)mpgp->betanorm) );
   TRY( PetscViewerASCIIPrintf(viewer,"\n") );
   PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetCurrentStepType_MPGP"
PetscErrorCode QPSMPGPGetCurrentStepType_MPGP(QPS qps,char *stepType)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  *stepType = mpgp->currentStepType;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetAlpha_MPGP"
static PetscErrorCode QPSMPGPGetAlpha_MPGP(QPS qps,PetscReal *alpha,QPSScalarArgType *argtype)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  if (alpha) *alpha = mpgp->alpha_user;
  if (argtype) *argtype = mpgp->alpha_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetAlpha_MPGP"
static PetscErrorCode QPSMPGPSetAlpha_MPGP(QPS qps,PetscReal alpha,QPSScalarArgType argtype)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  mpgp->alpha_user = alpha;
  mpgp->alpha_type = argtype;
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetGamma_MPGP"
static PetscErrorCode QPSMPGPGetGamma_MPGP(QPS qps,PetscReal *gamma)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  *gamma = mpgp->gamma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetGamma_MPGP"
static PetscErrorCode QPSMPGPSetGamma_MPGP(QPS qps,PetscReal gamma)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  mpgp->gamma = gamma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalue_MPGP"
static PetscErrorCode QPSMPGPGetOperatorMaxEigenvalue_MPGP(QPS qps,PetscReal *maxeig)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  *maxeig = mpgp->maxeig;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalue_MPGP"
static PetscErrorCode QPSMPGPSetOperatorMaxEigenvalue_MPGP(QPS qps,PetscReal maxeig)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  mpgp->maxeig = maxeig;
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPUpdateMaxEigenvalue_MPGP"
static PetscErrorCode  QPSMPGPUpdateMaxEigenvalue_MPGP(QPS qps, PetscReal maxeig_update)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  PetscReal maxeig_old = mpgp->maxeig;
  PetscReal alpha_old = mpgp->alpha;
  
  PetscFunctionBegin;
  if (!qps->setupcalled) FLLOP_SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_ARG_WRONGSTATE,"this routine is intended to be called after QPSSetUp");

  mpgp->maxeig = maxeig_old*maxeig_update;
  TRY( PetscInfo3(qps,"updating maxeig := %.8e = %.8e * %.8e = maxeig * maxeig_update\n",mpgp->maxeig,maxeig_old,maxeig_update) );

  if (mpgp->alpha_type == QPS_ARG_MULTIPLE) {
    mpgp->alpha = alpha_old/maxeig_update;
    TRY( PetscInfo3(qps,"updating alpha := %.8e = %.8e / %.8e = alpha / maxeig_update\n",mpgp->alpha,alpha_old,maxeig_update) );
  }

  //TODO temporary
  if (FllopDebugEnabled) {
    PetscReal lambda;
    TRY( MatGetMaxEigenvalue(qps->solQP->A,NULL,&lambda,mpgp->maxeig_tol,mpgp->maxeig_iter) );
    TRY( FllopDebug1("|maxeig_from_power_method - mpgp->maxeig| = %8e\n",PetscAbs(lambda-mpgp->maxeig)) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP"
static PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP(QPS qps,PetscReal *tol)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  *tol = mpgp->maxeig_tol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP"
static PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP(QPS qps,PetscReal tol)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  mpgp->maxeig_tol = tol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP"
static PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP(QPS qps,PetscInt *numit)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  *numit = mpgp->maxeig_iter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP"
static PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP(QPS qps,PetscInt numit)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  
  PetscFunctionBegin;
  mpgp->maxeig_iter = numit;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MPGPProj"
/*
MPGPProj - realize projection to feasible set

Parameters:
+ x - vector of variables
. lb - vector of lower bounds
- ub - vector of upper bounds 
*/
static PetscErrorCode MPGPProj(Vec x, Vec lb, Vec ub)
{
  PetscFunctionBegin;
  if (ub) TRY( VecPointwiseMin(x,x,ub) );
  if (lb) TRY( VecPointwiseMax(x,x,lb) );
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MPGPGrads"
/*
MPGPGrads - compute projected, chopped, and free gradient

Parameters:
+ qps - QP solver
- g - gradient  
*/
static PetscErrorCode MPGPGrads(QPS qps, Vec x, Vec lb, Vec ub, Vec g)
{
  PetscInt          i;                  /* ... iterator                         */
  PetscInt          m;                  /* ... local size                       */

  Vec               gP;                 /* ... projected gradient               */
  Vec               gr;                 /* ... reduced free gradient            */
  Vec               beta;               /* ... chopped gradient                 */
  Vec               phi;                /* ... free gradient                    */

  /* vector arrays */
  PetscScalar       *x_a, *lb_a, *ub_a, *g_a, *phi_a, *beta_a, *gP_a, *gr_a; 

  QPS_MPGP          *mpgp = (QPS_MPGP*)qps->data;
  PetscReal         alpha;
  
  PetscFunctionBegin;
  TRY( VecGetLocalSize(g, &m) );

  gP                = qps->work[0];
  gr                = qps->work[6];
  phi               = qps->work[1];
  beta              = qps->work[2];
  alpha             = mpgp->alpha;

  TRY( VecWAXPY(gr,-1.0,lb,x) ); /* gr = x-lb */
  TRY( VecWAXPY(gP,-1.0,ub,x) ); /* gP = x-ub */
  /* get local arrays */
  TRY( VecGetArray(x, &x_a) );
  TRY( VecGetArray(lb, &lb_a) );
  TRY( VecGetArray(ub, &ub_a) );
  TRY( VecGetArray(g, &g_a) );
  TRY( VecGetArray(phi, &phi_a) );
  TRY( VecGetArray(beta, &beta_a) );
  TRY( VecGetArray(gP, &gP_a) );
  TRY( VecGetArray(gr, &gr_a) );

  /* go through the local array and fill the gradients */
  for (i = 0; i < m; i++){
      if (PetscAbsScalar(gr_a[i]) < mpgp->btol) {
          /* index of this component is in ACTIVE SET */
          phi_a[i] = 0.0;
          gr_a[i] = 0.0;
          /* active lower bound */
          beta_a[i]= PetscMin(g_a[i],0.0);
      } else if (PetscAbsScalar(gP_a[i]) < mpgp->btol) {
         /* index of this component is in ACTIVE SET */
          phi_a[i] = 0.0;
          gr_a[i] = 0.0;
          /* active upper bound */
          beta_a[i]= PetscMax(g_a[i],0.0);
      } else {
          /* index of this component is in FREE SET */
          phi_a[i] = g_a[i];
          beta_a[i] = 0.0;
          gr_a[i] = alpha*gr_a[i];
          if (g_a[i] > 0 && g_a[i] < gr_a[i]) {
            gr_a[i] = g_a[i];
          } else {
            gr_a[i] = alpha*gP_a[i];
            gr_a[i] = PetscMax(gr_a[i],g_a[i]);
          }
      }
      
      /* compute projected gradient */
      gP_a[i] = phi_a[i] + beta_a[i];
      
  }
  
  /* restore local arrays */
  TRY( VecRestoreArray(x, &x_a) );
  TRY( VecRestoreArray(lb, &lb_a) );
  TRY( VecRestoreArray(ub, &ub_a) );
  TRY( VecRestoreArray(g, &g_a) );
  TRY( VecRestoreArray(phi, &phi_a) );
  TRY( VecRestoreArray(beta, &beta_a) );
  TRY( VecRestoreArray(gP, &gP_a) );
  TRY( VecRestoreArray(gr, &gr_a) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MPGPFeas"
/*
MPGPFeas - compute the largest feasible step alpha,
  i.e. lb <= x-alpha*p <= ub

  MATLAB:
  i = find(p > 0);
  alpha_f1 = inf;
  if min(size(i)) > 0
      alpha_f1 = min((x(i)-l1(i))./p(i));
  end
  j = find(p < 0);
  alpha_f2 = inf;
  if min(size(j)) > 0
      alpha_f2 = min((x(j)-l2(j))./p(j));
  end
  alpha_f = min(alpha_f1,alpha_f2);

Parameters:
+ x - vector of actual iteration
. lb - vector of lower bounds
. ub - vector of upper bounds
. p - vector of direction
- *alpha - pointer to return value  
*/
static PetscErrorCode MPGPFeas(Vec x, Vec lb, Vec ub, Vec p, PetscScalar *alpha)
{
  PetscInt          i, m;
  PetscScalar       *x_a, *lb_a, *ub_a, *p_a;
  PetscScalar       alpha_temp, alpha_loc;

  PetscFunctionBegin;
  
  /* get local arrays */
  TRY( VecGetLocalSize(x,&m) );
  TRY( VecGetArray(x,    &x_a) );     
  TRY( VecGetArray(lb,   &lb_a) );     
  TRY( VecGetArray(ub,   &ub_a) );     
  TRY( VecGetArray(p,    &p_a) );     

  /* the initial value of max feasible step-size */
  alpha_loc = PETSC_INFINITY;

  /* go through local arrays */
  for (i=0; i<m; i++) {
    /* max step to lower bound */
    if (p_a[i] > 0.0) {
      alpha_temp=x_a[i]-lb_a[i];
      alpha_temp=alpha_temp/p_a[i];
      if (alpha_temp < alpha_loc) alpha_loc=alpha_temp;
    }

    /* max step to upper bound */
    if (p_a[i] < 0.0) {
      alpha_temp=x_a[i]-ub_a[i];
      alpha_temp=alpha_temp/p_a[i];
      if (alpha_temp < alpha_loc) alpha_loc=alpha_temp;
    }
  }
  
  /* compute the minimal alpha from all computed maximal steps */
  TRY( MPI_Allreduce(&alpha_loc, alpha, 1, MPIU_SCALAR, MPIU_MIN, PetscObjectComm((PetscObject)x)) );

  if (*alpha >= PETSC_INFINITY) *alpha = 0.0;

  /* restore vectors */
  TRY( VecRestoreArray(x, &x_a) );
  TRY( VecRestoreArray(lb,&lb_a) );
  TRY( VecRestoreArray(ub,&ub_a) );
  TRY( VecRestoreArray(p, &p_a) );
  PetscFunctionReturn(0);
}

//TODO move to Constraints
#undef __FUNCT__
#define __FUNCT__ "QPGetScaledProjectedGradient"
PetscErrorCode QPGetScaledProjectedGradient(QP qp, PetscReal alpha, Vec galpha)
{
  Vec               x,b,lb,ub;
  Mat               A;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidLogicalCollectiveReal(qp,alpha,2);
  PetscValidHeaderSpecific(galpha,VEC_CLASSID,3);
  
  A                 = qp->A;
  x                 = qp->x;
  b                 = qp->b;
  lb                = qp->lb;
  ub                = qp->ub;
  TRY( MatMult(A, x, galpha) );          /* g=A*x         */
  TRY( VecAXPY(galpha, -1.0, b) );       /* g=g-b         */
  TRY( VecAYPX(galpha,-alpha,x) );       /* galpha = x - alpha*g */
  TRY( MPGPProj(galpha,lb,ub) );         /* galpha = P(galpha) */
  TRY( VecAYPX(galpha,-1.0,x) );         /* galpha = x - galpha */
  TRY( VecScale(galpha,1.0/alpha) );     /* galpha = 1/alpha * galpha */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSetup_MPGP"
/*
QPSSetup_MPGP - the setup function of MPGP algorithm; initialize constant step-size, check the constraints

Parameters:
. qps - QP solver
*/
PetscErrorCode QPSSetup_MPGP(QPS qps)
{
  QPS_MPGP          *mpgp = (QPS_MPGP*)qps->data;
  Vec               lb,ub,y;
  PetscScalar       min;

  PetscFunctionBegin;
  
  /* set the number of working vectors */
  TRY( QPSSetWorkVecs(qps,7) );

  lb                = qps->solQP->lb;
  ub                = qps->solQP->ub;
  y                 = qps->work[5];     /* here is used temporarily Ap vector */
  
  /* initialize alpha */
  if (mpgp->alpha_type == QPS_ARG_MULTIPLE) {
    if (mpgp->maxeig == PETSC_DECIDE) {
      TRY( MatGetMaxEigenvalue(qps->solQP->A, NULL, &mpgp->maxeig, mpgp->maxeig_tol, mpgp->maxeig_iter) );
    }
    if (mpgp->alpha_user == PETSC_DECIDE) {
      mpgp->alpha_user = 2.0;
    }
    TRY( PetscInfo1(qps,"maxeig     = %.8e\n", mpgp->maxeig) );
    TRY( PetscInfo1(qps,"alpha_user = %.8e\n", mpgp->alpha_user) );
    mpgp->alpha = mpgp->alpha_user/mpgp->maxeig;
  } else {
    mpgp->alpha = mpgp->alpha_user;
  }
  TRY( PetscInfo1(qps,  "alpha      = %.8e\n", mpgp->alpha) );

  if (ub) {
    /* check that lb < ub */
    TRY( VecWAXPY(y,-1.0,lb,ub) );
    TRY( VecMin(y,NULL,&min) );
    if (min <= 0.0) {
      FLLOP_SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_ARG_INCOMP,"lower bounds must be less than upper bounds");
    }
  } else {
    //TODO make MPGP compatible with ub = NULL
    /* create ub=inf(size(lb)) */
    TRY( VecDuplicate(lb,&ub) );
    TRY( VecDuplicate(ub,&qps->solQP->lambda_ub) );
    TRY( VecSet(ub,PETSC_INFINITY) );
    TRY( VecSet(qps->solQP->lambda_ub,0.0) );
    qps->solQP->ub = ub;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSSolve_MPGP"
/*
QPSSolve_MPGP - the solver; solve the problem using MPGP algorithm

Parameters:
+ qps - QP solver
*/
PetscErrorCode QPSSolve_MPGP(QPS qps)
{
  QPS_MPGP          *mpgp = (QPS_MPGP*)qps->data;
  QP                qp;
  Mat               A;                  /* ... hessian matrix                   */
  Vec               b;                  /* ... right-hand side vector           */
  Vec               x;                  /* ... vector of variables              */
  Vec               lb, ub;             /* ... lower and upper bound            */
  Vec               gP;                 /* ... projected gradient               */
  Vec               beta;               /* ... chopped gradient                 */
  Vec               phi;                /* ... free gradient                    */
  Vec               g;                  /* ... gradient                         */
  Vec               p;                  /* ... conjugate gradient               */
  Vec               Ap;                 /* ... multiplicated vector             */
  Vec               galpha;             /* ... scaled projected gradient        */
  
  PetscReal         alpha, gamma;       /* ... algorithm constants              */
  PetscReal         acg;                /* ... conjugate gradient step-size     */
  PetscReal         bcg;                /* ... cg ortogonalization parameter    */
  PetscReal         afeas;              /* ... maximum feasible step-size       */
  PetscReal         pAp, betaTbeta, phiTphi;  /* ... results of dot products    */

  PetscInt          nmv=0;              /* ... matrix-vector mult. counter      */
  PetscInt          ncg=0;              /* ... cg step counter                  */ 
  PetscInt          nprop=0;            /* ... proportional step counter        */
  PetscInt          nexp=0;             /* ... expansion step counter           */

  PetscFunctionBegin;
  /* set working vectors */
  gP                = qps->work[0];
  phi               = qps->work[1];
  beta              = qps->work[2];

  g                 = qps->work[3];
  p                 = qps->work[4];
  Ap                = qps->work[5];
  galpha            = qps->work[5];     /* uses the same work vector as Ap */

  /* set constants of algorithm */
  gamma             = mpgp->gamma;

  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPGetSolutionVector(qp, &x) );             /* get the solution vector */
  TRY( QPGetOperator(qp, &A ) );                  /* get hessian matrix */
  TRY( QPGetRhs(qp, &b  ) );                      /* get right-hand side vector */
  TRY( QPGetBox(qp, &lb, &ub ) );                 /* get lower and upper bounds */

  //TODO make MPGP compatible with ub = NULL
  if (!ub) {
    /* create ub=inf(size(lb)) */
    TRY( VecDuplicate(lb,&ub) );
    TRY( VecDuplicate(ub,&qp->lambda_ub) );
    TRY( VecSet(ub,PETSC_INFINITY) );
    TRY( VecSet(qp->lambda_ub,0.0) );
    qp->ub = ub;
  }

  TRY( MPGPProj(x, lb, ub) );                     /* project x initial guess to feasible set */

  /* compute gradient */
  TRY( MatMult(A, x, g) );                        /* g=A*x */
  nmv++;                                          /* matrix multiplication counter */
  TRY( VecAXPY(g, -1.0, b) );                     /* g=g-b */

  TRY( MPGPGrads(qps, x, lb, ub, g) );            /* grad. splitting  gP,phi,beta */

  /* initiate CG method */
  TRY( VecCopy(phi, p) );                         /* p=phi */

  alpha = mpgp->alpha;
  qps->iteration = 0;                             /* main iteration counter */
  while (1)                                       /* main cycle */
  {
    /* compute the norm of projected gradient - stopping criterion */
    TRY( VecNorm(gP, NORM_2, &qps->rnorm) );      /* qps->rnorm=norm(gP)*/

    /* compute dot products to control the proportionality */
    TRY( VecDot(beta, beta, &betaTbeta) );        /* betaTbeta=beta'*beta   */
    TRY( VecDot(phi, phi, &phiTphi) );            /* phiTphi=phi'*phi   */

    /* compute norm of phi, beta from computed dot products */
    if (qps->numbermonitors) {
      mpgp->phinorm =  PetscSqrtScalar(phiTphi);
      mpgp->betanorm =  PetscSqrtScalar(betaTbeta);
      TRY( QPSMonitor(qps,qps->iteration,qps->rnorm)) ;
    }

    /* test the convergence of algorithm */
    TRY( (*qps->convergencetest)(qps,qp,qps->iteration,qps->rnorm,&qps->reason,qps->cnvctx) ); /* test for convergence */
    if (qps->reason != KSP_CONVERGED_ITERATING) break;
    
    /* proportional condition */
    if (betaTbeta <= gamma*gamma*phiTphi)         /* u is proportional */
    {
      TRY( MatMult(A, p, Ap) );                   /* Ap=A*p */
      nmv++;                                      /* matrix multiplication counter */
      
      /* compute step-sizes */
      TRY( VecDot(p, Ap, &pAp) );                 /* pAp=p'*Ap      */
      TRY( VecDot(g,  p, &acg) );                 /* acg=g'*p       */
      acg  = acg/pAp;                             /* acg=acg/pAp    */
      TRY( MPGPFeas(x, lb, ub, p, &afeas) );      /* finds max.feas.steplength */

      /* decide if it is able to do full CG step */
      if (acg <= afeas)
      {
        /* CONJUGATE GRADIENT STEP */
        ncg++;                                    /* increase CG step counter */
        mpgp->currentStepType = 'c';

        /* make CG step */
        TRY( VecAXPY(x, -acg, p) );               /* x=x-acg*p      */
        TRY( VecAXPY(g, -acg, Ap) );              /* g=g-acg*Ap      */

        TRY( MPGPGrads(qps, x, lb, ub, g) );      /* grad. splitting  gP,phi,beta */
        
        /* compute orthogonalization parameter and next orthogonal vector */
        TRY( VecDot(Ap, phi, &bcg) );             /* bcg=Ap'*phi     */
        bcg  = bcg/pAp;                           /* bcg=bcg/pAp     */
        TRY( VecAYPX(p, -bcg, phi) );             /* p=phi-bcg*p     */
      }
      else                                        /* expansion step  */
      {
        /* EXPANSION STEP */
        /* make maximal feasible step */
        TRY( VecAXPY(x, -afeas, p) );             /* x=x-afeas*p*/
        TRY( VecAXPY(g, -afeas, Ap) );            /* g=g-afeas*Ap    */

        TRY( MPGPGrads(qps, x, lb, ub, g) );      /* grad. splitting  gP,phi,beta,gr */

        /* make one more projected gradient step with constant step-length */
        alpha = mpgp->alpha;
        TRY( VecAXPY(x, -alpha, phi) );           /* x=x-abar*phi */
        TRY( MPGPProj(x, lb, ub) );               /* project x to feas.set */

        /* compute new gradient */
        TRY( MatMult(A, x, g) );                  /* g=A*x */
        nmv++;                                    /* matrix multiplication counter */
        TRY( VecAXPY(g, -1.0, b) );               /* g=g-b           */

        TRY( MPGPGrads(qps, x, lb, ub, g) );      /* grad. splitting  gP,phi,beta */

        /* restart CG method */
        TRY( VecCopy(phi, p) );                   /* p=phi           */

        nexp++;                                   /* increase expansion step counter */
        mpgp->currentStepType = 'e';
      }
    }
    else                                          /* proportioning step  */
    {
      /* PROPORTIONING STEP */
      nprop++;                                    /* increase proportioning step counter */
      mpgp->currentStepType = 'p';
 
      TRY( VecCopy(beta, p) );                    /* p=beta           */
      TRY( MatMult(A, p, Ap) );                   /* Ap=A*p */
      nmv++;                                      /* matrix multiplication counter */

      /* compute step-size */
      TRY( VecDot(p, Ap, &pAp) );                 /* pAp=p'*Ap       */
      TRY( VecDot(g,  p, &acg) );                 /* acg=g'*p        */
      acg  = acg/pAp;                             /* acg=acg/pAp     */

      /* make a step */
      TRY( VecAXPY(x, -acg, p) );                 /* x=x-acg*p       */
      TRY( VecAXPY(g, -acg, Ap) );                /* g=g-acg*Ap      */

      TRY( MPGPGrads(qps, x, lb, ub, g) );        /* grad. splitting  gP,phi,beta */
    
      /* restart CG method */
      TRY( VecCopy(phi, p) );                     /* p=phi           */
    }
    qps->iteration++;
  };

  if (FllopDebugEnabled) {
    PetscReal norm;

    TRY( QPGetScaledProjectedGradient(qp,alpha,galpha) );
    TRY( VecAXPY(galpha,-1.0,gP) );
    TRY( VecNorm(galpha,NORM_2,&norm) );
    TRY( FllopDebug1("||g_alpha - g_P|| = %.8e\n",norm) );
  }

  mpgp->ncg     += ncg;
  mpgp->nexp    += nexp;
  mpgp->nmv     += nmv;
  mpgp->nprop   += nprop;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSDestroy_MPGP"
/*
QPSDestroy_MPGP - MPGP afterparty

Parameters:
. qps - QP solver
*/
PetscErrorCode QPSDestroy_MPGP(QPS qps)
{
  PetscFunctionBegin;
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetAlpha_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetAlpha_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetGamma_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetGamma_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalue_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalue_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP_C",NULL) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP_C",NULL) );
  TRY( QPSDestroyDefault(qps) );
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "QPSIsQPCompatible_MPGP"
/*
QPSIsQPCompatible_MPGP - verify if the algorithm is able to solve given QP problem

Parameters:
+ qps - QP solver
. qp - quadratic programming problem
- flg - the pointer to result
*/
PetscErrorCode QPSIsQPCompatible_MPGP(QPS qps,QP qp,PetscBool *flg)
{
  Vec lb,ub;
  Mat Beq,Bineq;
  
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  TRY( QPGetBox(qp,&lb,&ub) );
  TRY( QPGetEq(qp,&Beq,NULL) );
  TRY( QPGetIneq(qp,&Bineq,NULL) );
  if (!(lb || (lb && ub))) {
    *flg = PETSC_FALSE;
  }
  if (Beq || Bineq) {
    *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetFromOptions_MPGP"
PetscErrorCode QPSSetFromOptions_MPGP(PetscOptionItems *PetscOptionsObject,QPS qps)
{
  QPS_MPGP    *mpgp = (QPS_MPGP*)qps->data;
  PetscBool flg1,flg2,alpha_direct;
  PetscReal maxeig,maxeig_tol,alpha,gamma;
  PetscInt maxeig_iter;

  PetscFunctionBegin;
  TRY( PetscOptionsHead(PetscOptionsObject,"QPS MPGP options") );

  alpha_direct = PETSC_FALSE;
  TRY( PetscOptionsBool("-qps_mpgp_alpha_direct","","QPSMPGPSetAlpha",(PetscBool) mpgp->alpha_type,&alpha_direct,&flg1) );
  TRY( PetscOptionsReal("-qps_mpgp_alpha","","QPSMPGPSetAlpha",mpgp->alpha_user,&alpha,&flg2) );
  if (flg1 || flg2) TRY( QPSMPGPSetAlpha(qps,alpha,(QPSScalarArgType) alpha_direct) );
  TRY( PetscOptionsReal("-qps_mpgp_gamma","","QPSMPGPSetGamma",mpgp->gamma,&gamma,&flg1) );
  if (flg1) TRY( QPSMPGPSetGamma(qps,gamma) );
  TRY( PetscOptionsReal("-qps_mpgp_maxeig","Approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means this is automatically computed.","QPSMPGPSetOperatorMaxEigenvalue",mpgp->maxeig,&maxeig,&flg1) );
  if (flg1) TRY( QPSMPGPSetOperatorMaxEigenvalue(qps,maxeig) );
  TRY( PetscOptionsReal("-qps_mpgp_maxeig_tol","Relative tolerance to find approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means QPS rtol","QPSMPGPSetOperatorMaxEigenvalueTolerance",mpgp->maxeig_tol,&maxeig_tol,&flg1) );
  if (flg1) TRY( QPSMPGPSetOperatorMaxEigenvalueTolerance(qps,maxeig_tol) );
  TRY( PetscOptionsInt("-qps_mpgp_maxeig_iter","Number of iterations to find an approximate maximum eigenvalue of the Hessian","QPSMPGPSetOperatorMaxEigenvalueIterations",mpgp->maxeig_iter,&maxeig_iter,&flg1) );
  if (flg1) TRY( QPSMPGPSetOperatorMaxEigenvalueIterations(qps,maxeig_iter) );
  TRY( PetscOptionsReal("-qps_mpgp_btol","Boundary overshoot tolerance; default: 10*PETSC_MACHINE_EPSILON","",mpgp->btol,&mpgp->btol,&flg1) );
  TRY( PetscOptionsTail() );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_MPGP"
PetscErrorCode QPSViewConvergence_MPGP(QPS qps, PetscViewer v)
{
  QPS_MPGP      *mpgp = (QPS_MPGP*)qps->data;
  PetscBool     iascii;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (iascii) {
    TRY( PetscViewerASCIIPrintf(v,"from the last QPSReset:\n") );
    TRY( PetscViewerASCIIPrintf(v,"number of Hessian multiplications %d\n",mpgp->nmv) );
    TRY( PetscViewerASCIIPrintf(v,"number of CG steps %d\n",mpgp->ncg) );
    TRY( PetscViewerASCIIPrintf(v,"number of expansion steps %d\n",mpgp->nexp) );
    TRY( PetscViewerASCIIPrintf(v,"number of proportioning steps %d\n",mpgp->nprop) );
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "QPSCreate_MPGP"
FLLOP_EXTERN PetscErrorCode QPSCreate_MPGP(QPS qps)
{
  QPS_MPGP         *mpgp;
  
  PetscFunctionBegin;
  TRY( PetscNewLog(qps,&mpgp) );
  qps->data                  = (void*)mpgp;

  mpgp->alpha_user           = PETSC_DECIDE;
  mpgp->alpha_type           = QPS_ARG_MULTIPLE;
  mpgp->gamma                = 1.0;
  mpgp->maxeig               = PETSC_DECIDE;
  mpgp->maxeig_tol           = PETSC_DECIDE;
  mpgp->maxeig_iter          = PETSC_DECIDE;
  mpgp->btol                 = 10*PETSC_MACHINE_EPSILON; /* boundary tol */

  /* set the computed norms of gradients */
  //TODO: set from options/command line
  mpgp->ncg                  = 0;
  mpgp->nexp                 = 0;
  mpgp->nmv                  = 0;
  mpgp->nprop                = 0;
  mpgp->currentStepType      = ' ';
  
  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  qps->ops->setup            = QPSSetup_MPGP;
  qps->ops->solve            = QPSSolve_MPGP;
  qps->ops->destroy          = QPSDestroy_MPGP;
  qps->ops->isqpcompatible   = QPSIsQPCompatible_MPGP;
  qps->ops->setfromoptions   = QPSSetFromOptions_MPGP;
  qps->ops->monitor          = QPSMonitorDefault_MPGP;
  qps->ops->viewconvergence  = QPSViewConvergence_MPGP;

  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetCurrentStepType_MPGP_C",QPSMPGPGetCurrentStepType_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetAlpha_MPGP_C",QPSMPGPGetAlpha_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetAlpha_MPGP_C",QPSMPGPSetAlpha_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetGamma_MPGP_C",QPSMPGPGetGamma_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetGamma_MPGP_C",QPSMPGPSetGamma_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalue_MPGP_C",QPSMPGPGetOperatorMaxEigenvalue_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalue_MPGP_C",QPSMPGPSetOperatorMaxEigenvalue_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP_C",QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP_C",QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP_C",QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP_C",QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP) );
  TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPUpdateMaxEigenvalue_MPGP_C",QPSMPGPUpdateMaxEigenvalue_MPGP) );
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetCurrentStepType"
PetscErrorCode QPSMPGPGetCurrentStepType(QPS qps,char *stepType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  if (stepType) PetscValidRealPointer(stepType,2);
  *stepType = ' ';
  TRY( PetscTryMethod(qps,"QPSMPGPGetCurrentStepType_MPGP_C",(QPS,char*),(qps,stepType)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetAlpha"
/*@
QPSMPGPGetAlpha - get the constant step-size used in algorithm based on spectral properties of Hessian matrix

Parameters:
+ qps - QP solver
. alpha - pointer to store the value
- argtype -  

Level: advanced
@*/
PetscErrorCode QPSMPGPGetAlpha(QPS qps,PetscReal *alpha,QPSScalarArgType *argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  if (alpha) PetscValidPointer(alpha,2);
  if (argtype) PetscValidPointer(argtype,3);
  TRY( PetscUseMethod(qps,"QPSMPGPGetAlpha_MPGP_C",(QPS,PetscReal*,QPSScalarArgType*),(qps,alpha,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetAlpha"
/*@
QPSMPGPSetAlpha - set the constant step-size used in algorithm based on spectral properties of Hessian matrix

Parameters:
+ qps - QP solver
. alpha - new value of parameter
- argtype -  

Level: intermediate
@*/
PetscErrorCode QPSMPGPSetAlpha(QPS qps,PetscReal alpha,QPSScalarArgType argtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,alpha,2);
  TRY( PetscTryMethod(qps,"QPSMPGPSetAlpha_MPGP_C",(QPS,PetscReal,QPSScalarArgType),(qps,alpha,argtype)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetGamma"
/*@
QPSMPGPGetGamma - get the proportioning parameter used in algorithm

Parameters:
+ qps - QP solver
- gamma - pointer to store the value  

Level: advanced
@*/
PetscErrorCode QPSMPGPGetGamma(QPS qps,PetscReal *gamma)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(gamma,2);
  TRY( PetscUseMethod(qps,"QPSMPGPGetGamma_MPGP_C",(QPS,PetscReal*),(qps,gamma)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetGamma"
/*@
QPSMPGPSetGamma - set the proportioning parameter used in algorithm

Parameters:
+ qps - QP solver
- gamma - new value of parameter  

Level: intermediate
@*/
PetscErrorCode QPSMPGPSetGamma(QPS qps,PetscReal gamma)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,gamma,2);
  TRY( PetscTryMethod(qps,"QPSMPGPSetGamma_MPGP_C",(QPS,PetscReal),(qps,gamma)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalue"
PetscErrorCode QPSMPGPGetOperatorMaxEigenvalue(QPS qps,PetscReal *maxeig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(maxeig,2);
  TRY( PetscUseMethod(qps,"QPSMPGPGetOperatorMaxEigenvalue_MPGP_C",(QPS,PetscReal*),(qps,maxeig)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalue"
/*@
QPSMPGPSetOperatorMaxEigenvalue - set the estimation of largest eigenvalue

Parameters:
+ qps - QP solver
- maxeig - new value  

Level: intermediate
@*/
PetscErrorCode QPSMPGPSetOperatorMaxEigenvalue(QPS qps,PetscReal maxeig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,maxeig,2);
  if (maxeig < 0 && maxeig != PETSC_DECIDE) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be nonnegative");
  TRY( PetscTryMethod(qps,"QPSMPGPSetOperatorMaxEigenvalue_MPGP_C",(QPS,PetscReal),(qps,maxeig)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPUpdateMaxEigenvalue"
PetscErrorCode  QPSMPGPUpdateMaxEigenvalue(QPS qps, PetscReal maxeig_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,maxeig_update,2);
  if (maxeig_update == 1.0) PetscFunctionReturn(0);
  TRY( PetscTryMethod(qps,"QPSMPGPUpdateMaxEigenvalue_MPGP_C",(QPS,PetscReal),(qps,maxeig_update)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalueTolerance"
/*@
QPSMPGPSetOperatorMaxEigenvalueTolerance - set the tolerance of the largest eigenvalue computation

Parameters:
+ qps - QP solver
- tol - new value 

Level: intermediate
@*/
PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,tol,2);
  TRY( PetscTryMethod(qps,"QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP_C",(QPS,PetscReal),(qps,tol)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalueTolerance"
/*@
QPSMPGPGetOperatorMaxEigenvalueTolerance - get the tolerance of the largest eigenvalue computation

Parameters:
+ qps - QP solver
- tol - pointer to returned value 

Level: advanced
@*/
PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal *tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(tol,2);
  TRY( PetscTryMethod(qps,"QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP_C",(QPS,PetscReal*),(qps,tol)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalueIterations"
/*@
QPSMPGPGetOperatorMaxEigenvalueIterations - get the maximum number of iterations to obtain the largest eigenvalue computation

Parameters:
+ qps - QP solver
- numit - pointer to returned value 

Level: advanced
@*/
PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueIterations(QPS qps,PetscInt *numit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidPointer(numit,2);
  TRY( PetscUseMethod(qps,"QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP_C",(QPS,PetscInt*),(qps,numit)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalueIterations"
/*@
QPSMPGPSetOperatorMaxEigenvalueIterations - set the maximum number of iterations to obtain the largest eigenvalue computation

Parameters:
+ qps - QP solver
- numit - new value

Level: intermediate
@*/
PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueIterations(QPS qps,PetscInt numit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(qps,numit,2);
  if (numit <= 1) FLLOP_SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be > 1");
  TRY( PetscTryMethod(qps,"QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP_C",(QPS,PetscInt),(qps,numit)) );
  PetscFunctionReturn(0);
}
