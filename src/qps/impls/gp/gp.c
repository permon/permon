
#include <../src/qps/impls/gp/gpimpl.h>

/*
  WORK VECTORS:

  gProj = qps->work[0];
  gf = qps->work[1];
  gc = qps->work[2];

  g  = qps->work[3];
  d  = qps->work[4];
  Ad = qps->work[5];
  gr = qps->work[6];
*/

PetscLogEvent QPS_GP_LineSearch, QPS_GP_StepLength;

#undef __FUNCT__
#define __FUNCT__ "QPSMonitorDefault_GP"
PetscErrorCode QPSMonitorDefault_GP(QPS qps,PetscInt n,PetscViewer viewer)
{
   QPS_GP *gp = (QPS_GP*)qps->data;

   PetscFunctionBegin;
   if (n == 0 && ((PetscObject)qps)->prefix) {
     TRY( PetscViewerASCIIPrintf(viewer,"  Projected gradient norms for %s solve.\n",((PetscObject)qps)->prefix) );
   }
   TRY( VecNorm(qps->work[1],NORM_2,&gp->gfnorm) );
   TRY( VecNorm(qps->work[2],NORM_2,&gp->gcnorm) );

   TRY( PetscViewerASCIIPrintf(viewer,"%3D GP ||gp||=%.10e",n,(double)qps->rnorm) );
   TRY( PetscViewerASCIIPrintf(viewer,",\t||gf||=%.10e",(double)gp->gfnorm) );
   TRY( PetscViewerASCIIPrintf(viewer,",\t||gc||=%.10e",(double)gp->gcnorm) );
   TRY( PetscViewerASCIIPrintf(viewer,",\tsl_alpha=%.10e",(double)gp->sl_alpha) );
   TRY( PetscViewerASCIIPrintf(viewer,",\tls_alpha=%.10e",(double)gp->ls_alpha) );
   TRY( PetscViewerASCIIPrintf(viewer,"\n") );
   PetscFunctionReturn(0);
}

//#undef __FUNCT__
//#define __FUNCT__ "QPSGPGetStepLengthAlpha_GP"
//static PetscErrorCode QPSMPGPGetStepLengthAlpha_GP(QPS qps,PetscReal *alphamin,PetscReal *alphamax)
//{
//  QPS_GP *gp = (QPS_GP*)qps->data;
//
//  PetscFunctionBegin;
//  if (alphamin) *alphamin = gp->sl_alphamin;
//  if (alphamax) *alphamax = gp->sl_alphamax;
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPSetAlpha_GP"
//static PetscErrorCode QPSMPGPSetAlpha_GP(QPS qps,PetscReal alpha,QPSScalarArgType argtype)
//{
//  QPS_GP *mpgp = (QPS_GP*)qps->data;
//
//  PetscFunctionBegin;
//  mpgp->alpha_user = alpha;
//  mpgp->alpha_type = argtype;
//  qps->setupcalled = PETSC_FALSE;
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPGetGamma_GP"
//static PetscErrorCode QPSMPGPGetGamma_GP(QPS qps,PetscReal *gamma)
//{
//  QPS_GP *mpgp = (QPS_GP*)qps->data;
//
//  PetscFunctionBegin;
//  *gamma = mpgp->gamma;
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPSetGamma_GP"
//static PetscErrorCode QPSMPGPSetGamma_GP(QPS qps,PetscReal gamma)
//{
//  QPS_GP *mpgp = (QPS_GP*)qps->data;
//
//  PetscFunctionBegin;
//  mpgp->gamma = gamma;
//  PetscFunctionReturn(0);
//}

#undef __FUNCT__
#define __FUNCT__ "GPGrads"
/*
GPGrads - compute projected, chopped, and free gradient

Parameters:
+ qps - QP solver
- g - gradient
*/
static PetscErrorCode GPGrads(QPS qps, Vec x, Vec g)
{
  QP                qp;
  QPC               qpc;

  Vec               gProj;                 /* ... projected gradient               */
  Vec               gc;                 /* ... chopped gradient                 */
  Vec               gf;                 /* ... free gradient                    */

  PetscFunctionBegin;
  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPGetQPC(qp,&qpc) );

  gProj             = qps->work[0];
  gf                = qps->work[1];
  gc                = qps->work[2];

  TRY( QPCGrads(qpc,x,g,gf,gc) );
  TRY( VecWAXPY(gProj,1.0,gf,gc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GPLineSearch_Default"
/*
GPLineSearch - line search with the feasible direction d
Default is backtracking GLL (Grippo-Lampariello-Lucidi) line-search that
satisfies the (nonmonotone) Armijo condition:
alpha = 1; g = Ax - b;
while f(x + alpha*d) > fmax + gamma*alpha*g'*d:
  alpha = beta*alpha
where fmax is the maximum from the last M functional values.

Parameters:
+ M      - maximum size of array holding previous cost function values
. alpha  - step length in d direction
. beta   - multiplier for alpha increment
- gamma  - Armijo rule parameter
*/
static PetscErrorCode GPLineSearch_Default(QPS qps)
{
  QP                qp;
  Mat               A;
  Vec               Ad,d,g,x;
  PetscReal         fmax,f,alpha_old,delta;
  PetscReal         alpha = 1.;
  PetscInt          i,M;
  QPS_GP            *gp = (QPS_GP*)qps->data;

  PetscFunctionBegin;
  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( QPGetOperator(qp,&A) );
  g  = qps->work[3];
  d  = qps->work[4];
  Ad = qps->work[5];

  /* find fmax */
  if (!qps->iteration) {
    TRY( QPComputeObjectiveFromGradient(qp,x,g,&gp->ls_f[0]) );
  }
  M  = PetscMax(0,PetscMin(gp->ls_M-1,qps->iteration));
  fmax = gp->ls_f[M];
  for (i = M; i>=0; --i) {
    if (gp->ls_f[i] > fmax) fmax = gp->ls_f[i];
  } 

  TRY( MatMult(A,d,Ad) );
  gp->nmv++;
  TRY( VecAXPY(x,alpha,d) );               /* x = x-d      */
  TRY( VecAXPY(g,alpha,Ad) );              /* g = x-Ad     */
  TRY( QPComputeObjectiveFromGradient(qp,x,g,&f) );
  TRY( VecDot(g,d,&delta) );
  while (f > fmax + gp->ls_gamma*delta*alpha && i < gp->ls_maxit) {
//printf("%e %e %e %e %e\n",fmax,f,gp->ls_gamma,alpha,delta);
    alpha_old = alpha;
    alpha *= gp->ls_beta;
    TRY( VecAXPY(x,alpha-alpha_old,d) );               /* x = x-d      */
    TRY( VecAXPY(g,alpha-alpha_old,Ad) );              /* g = x-Ad     */
    TRY( QPComputeObjectiveFromGradient(qp,x,g,&f) );
    i++;
  }

  /* write the computed values */
  gp->ls_alpha = alpha;
  if (gp->ls_M) {
    i = (qps->iteration+1)%gp->ls_M;
  } else {
    i = 0;
  }
  gp->ls_f[i] = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GPStepLength_Default"
/*
GPStepLength - computes step length to determine the feasible descent direction
Default is BoxABBVAmin 

Parameters:
+ M      - maximum size of array holding previous cost function values
. alpha  - step length in d direction
. beta   - multiplier for alpha increment
- gamma  - Armijo rule parameter
*/
static PetscErrorCode GPStepLength_Default(QPS qps)
{
  QP                qp;
  QPC               qpc;
  Mat               A;
  Vec               g,x,s,y,g_old;
  PetscReal         dots[3],bb1,bb2;
  PetscInt          i,M;
  IS                is;
  QPS_GP            *gp = (QPS_GP*)qps->data;

  PetscFunctionBegin;
  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPGetSolutionVector(qp,&x) );
  TRY( QPGetOperator(qp,&A) );
  TRY( QPGetQPC(qp,&qpc) );
  Vec lb,ub;
  TRY( QPGetBox(qps->solQP,NULL,&lb,&ub) );
  g     = qps->work[3];
  s     = qps->work[4]; /* replace d by  s = x - x_old */
  y     = qps->work[5]; /* replace Ad by y = g - g_old */
  g_old = qps->work[6];

  TRY( VecScale(s,gp->ls_alpha) ); /* s = ls_alpha*d = x - x_old */
  TRY( VecWAXPY(y,-1.,g_old,g) );  /* y = g - g_old              */
  TRY( VecDot(s,y,&dots[0]) );
  TRY( VecDot(s,s,&dots[1]) );
  //TODO Mdot dot, split reductiond
  if (qps->iteration) {
    TRY( ISDestroy(&gp->isactive_old) );
    TRY( ISDuplicate(gp->isactive,&gp->isactive_old) );
    TRY( ISDestroy(&gp->isactive) );
  }

  //IS is2,is3;
  //PetscInt ilo,ihi;
  //VecGetOwnershipRange(x,&ilo,&ihi);
  //PetscBool flg;
  //VecWhichBetween(lb,x,ub,&is2);
  //ISComplement(is2,ilo,ihi,&is3);
  //TRY( QPCGetActiveSet(qpc,NULL,&gp->isactive) );
  //ISEqual(is3,gp->isactive,&flg);
  //printf ("%d same\n",flg);
  if (!qps->iteration) {
    TRY( VecCopy(g,g_old) );
    PetscFunctionReturn(0);
  }

    TRY( PetscLogEventBegin(QPS_GP_LineSearch,qps,0,0,0) );
    TRY( ISSetInfo(gp->isactive,IS_SORTED,IS_LOCAL,PETSC_TRUE,PETSC_TRUE) );
    TRY( ISSetInfo(gp->isactive_old,IS_SORTED,IS_LOCAL,PETSC_TRUE,PETSC_TRUE) );
  TRY( ISIntersect(gp->isactive,gp->isactive_old,&is) );
    TRY( PetscLogEventEnd(QPS_GP_LineSearch,qps,0,0,0) );
  TRY( VecISSet(y,is,0.0) );
   TRY( ISDestroy(&is) );
  TRY( VecDot(y,y,&dots[2]) );
  //printf("sy %e ss %e yy %e\n",dots[0],dots[1],dots[2]);
  TRY( VecCopy(g,g_old) );

  bb1 = dots[1]/dots[0];
  bb1 = PetscMax(gp->sl_alphamin,PetscMin(gp->sl_alphamax,bb1));
  bb2 = dots[0]/dots[2];
  bb2 = PetscMax(gp->sl_alphamin,PetscMin(gp->sl_alphamax,bb2));

  //printf("%e %e \n",bb1,bb2);
  i = (qps->iteration-1) % gp->sl_M;
  gp->sl_bb2[i] = bb2;

  if (bb2/bb1 < gp->sl_tau) {
    M = PetscMax(0,PetscMin(gp->sl_M-1,qps->iteration-1));
    gp->sl_alpha = gp->sl_bb2[M];
    for (i = M; i>=0; --i) {
      if (gp->sl_bb2[i] < gp->sl_alpha) gp->sl_alpha = gp->sl_bb2[i];
    }
    gp->sl_tau /= gp->sl_fact;
  } else {
    gp->sl_alpha = bb1;
    gp->sl_tau *= gp->sl_fact;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetup_GP"
/*
QPSSetup_GP - the setup function of MPGP algorithm; initialize constant step-size, check the constraints

Parameters:
. qps - QP solver
*/
PetscErrorCode QPSSetup_GP(QPS qps)
{
  QPS_GP          *gp = (QPS_GP*)qps->data;
  QPC             qpc;
  Vec               lb,ub;

  PetscFunctionBegin;
  /* set the number of working vectors */
  //if (mpgp->explengthtype != QPS_GP_EXPANSION_LENGTH_BB) {
    TRY( QPSSetWorkVecs(qps,7) );
  //} else {
  //  TRY( QPSSetWorkVecs(qps,11) );
  //}
  
  TRY( PetscMalloc1(gp->ls_M,&gp->ls_f) );
  TRY( PetscMalloc1(gp->sl_M,&gp->sl_bb2) );

  TRY( QPGetBox(qps->solQP,NULL,&lb,&ub) );
  if (gp->bchop_tol) {
    if (lb) TRY( VecChop(lb,gp->bchop_tol) );
    if (ub) TRY( VecChop(ub,gp->bchop_tol) );
  }

  TRY( QPGetQPC(qps->solQP,&qpc) );
  //TRY( QPCGetStoreActiveSet(qpc,gp->qpcstore) );

  //switch (mpgp->exptype) {
  //  case QPS_GP_EXPANSION_STD:
  //    mpgp->expdirection = qps->work[6];     /* gr */
  //    mpgp->explengthvec = qps->work[6];
  //    if (mpgp->explengthtype == QPS_GP_EXPANSION_LENGTH_FIXED) {
  //      mpgp->expproject   = PETSC_FALSE;
  //    }
  //    break;
  //  case QPS_GP_EXPANSION_GF:
  //    mpgp->expdirection = qps->work[1];     /* gf */
  //    mpgp->explengthvec = qps->work[1];
  //    break;
  //  case QPS_GP_EXPANSION_G:
  //    mpgp->expdirection = qps->work[3];     /* g  */
  //    mpgp->explengthvec = qps->work[3];
  //    break;
  //  case QPS_GP_EXPANSION_GFGR:
  //    mpgp->expdirection = qps->work[1];     /* gf */
  //    mpgp->explengthvec = qps->work[6];
  //    break;
  //  case QPS_GP_EXPANSION_GGR:
  //    mpgp->expdirection = qps->work[3];     /* g  */
  //    mpgp->explengthvec = qps->work[6];
  //    break;
  //  case QPS_GP_EXPANSION_PROJCG:
  //    mpgp->expansion = MPGPExpansion_ProjCG;
  //    //falback
  //    mpgp->expdirection = qps->work[1];     /* gf */
  //    mpgp->explengthvec = qps->work[1];
  //    break;
  //  default: SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_PLIB,"Unknown MPGP expansion type");
  //}
  
  /* initialize alpha */
  //if (mpgp->alpha_type == QPS_ARG_MULTIPLE) {
  //  if (mpgp->maxeig == PETSC_DECIDE) {
  //    TRY( MatGetMaxEigenvalue(qps->solQP->A, NULL, &mpgp->maxeig, mpgp->maxeig_tol, mpgp->maxeig_iter) );
  //  }
  //  if (mpgp->alpha_user == PETSC_DECIDE) {
  //    mpgp->alpha_user = 2.0;
  //  }
  //  TRY( PetscInfo1(qps,"maxeig     = %.8e\n", mpgp->maxeig) );
  //  TRY( PetscInfo1(qps,"alpha_user = %.8e\n", mpgp->alpha_user) );
  //  mpgp->alpha = mpgp->alpha_user/mpgp->maxeig;
  //} else {
  //  mpgp->alpha = mpgp->alpha_user;
  //}
  //TRY( PetscInfo1(qps,  "alpha      = %.8e\n", mpgp->alpha) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolve_GP"
/*
QPSSolve_GP - the solver; solve the problem using GP algorithm

Parameters:
+ qps - QP solver
*/
PetscErrorCode QPSSolve_GP(QPS qps)
{
  QPS_GP          *gp = (QPS_GP*)qps->data;
  QP                qp;
  QPC               qpc;
  Mat               A;                  /* ... hessian matrix                   */
  Vec               b;                  /* ... right-hand side vector           */
  Vec               x;                  /* ... vector of variables              */
  Vec               gProj;                 /* ... projected gradient               */
  Vec               gc;                 /* ... chopped gradient                 */
  Vec               gf;                 /* ... free gradient                    */
  Vec               g;                  /* ... gradient                         */
  Vec               d;                  /* ... direction vector                 */
  Vec               Ad;                 /* ... multiplicated vector             */


  PetscInt          nmv=0;              /* ... matrix-vector mult. counter      */

  PetscFunctionBegin;
  /* set working vectors */
  gProj = qps->work[0];
  gf = qps->work[1];
  gc = qps->work[2];

  g  = qps->work[3];
  d  = qps->work[4];
  Ad = qps->work[5];

    Vec oldx,oldg;
    VecDuplicate(g,&oldx);
    VecDuplicate(g,&oldg);

    //if (mpgp->explengthtype == QPS_GP_EXPANSION_LENGTH_BB) {
    //  mpgp->explengthvecold = qps->work[7];
    //  mpgp->xold            = qps->work[8];
    //}


  TRY( QPSGetSolvedQP(qps,&qp) );
  TRY( QPGetQPC(qp,&qpc) );                       /* get constraints */
  TRY( QPGetSolutionVector(qp, &x) );             /* get the solution vector */
  TRY( QPGetOperator(qp, &A) );                   /* get hessian matrix */
  TRY( QPGetRhs(qp, &b) );                        /* get right-hand side vector */

  TRY( QPCProject(qpc,x,x) );                     /* project x initial guess to feasible set */

  /* compute gradient */
  TRY( MatMult(A, x, g) );                        /* g=A*x */
  nmv++;                                          /* matrix multiplication counter */
  TRY( VecAXPY(g, -1.0, b) );                     /* g=g-b */
  qps->iteration = 0;                             /* main iteration counter */
  while (1)                                       /* main cycle */
  {
    TRY( GPGrads(qps, x, g) );                   /* grad. splitting  gProj,gf,gc */
    /* compute the norm of projected gradient - stopping criterion */
    TRY( VecNorm(gProj, NORM_2, &qps->rnorm) );      /* qps->rnorm=norm(gProj)*/
    if (qps->numbermonitors) {
      TRY( QPSMonitor(qps,qps->iteration,qps->rnorm)) ;
    }

    /* test the convergence of algorithm */
    TRY( (*qps->convergencetest)(qps,qp,qps->iteration,qps->rnorm,&qps->reason,qps->cnvctx) );
    if (qps->reason != KSP_CONVERGED_ITERATING) break;

    /* compute steplength for descent direction */
    TRY( PetscLogEventBegin(QPS_GP_StepLength,qps,0,0,0) );
    TRY( gp->steplength(qps) );
    TRY( PetscLogEventEnd(QPS_GP_StepLength,qps,0,0,0) );

    /* descent direction */
    TRY( VecWAXPY(d,-gp->sl_alpha,g,x) );                /* d = x - alpha*g */
    TRY( QPCProject(qpc,d,d) );                   /* project d to feasible set */
    TRY( VecAXPY(d,-1.0,x) );                     /* d = d - x */

    TRY( gp->linesearch(qps) );
    qps->iteration++;
  };

  gp->nmv += nmv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSResetStatistics_GP"
PetscErrorCode QPSResetStatistics_GP(QPS qps)
{
  QPS_GP *mpgp = (QPS_GP*)qps->data;
  PetscFunctionBegin;
  mpgp->ncg   = 0;
  mpgp->nexp  = 0;
  mpgp->nmv   = 0;
  mpgp->nprop = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSDestroy_GP"
/*
QPSDestroy_GP - MPGP afterparty

Parameters:
. qps - QP solver
*/
PetscErrorCode QPSDestroy_GP(QPS qps)
{
  PetscFunctionBegin;
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetAlpha_GP_C",NULL) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetAlpha_GP_C",NULL) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetGamma_GP_C",NULL) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetGamma_GP_C",NULL) );
  //TODO restore active set computation
  TRY( QPSDestroyDefault(qps) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSIsQPCompatible_GP"
PetscErrorCode QPSIsQPCompatible_GP(QPS qps,QP qp,PetscBool *flg)
{
  Mat Beq,Bineq;
  Vec ceq,cineq;
  QPC qpc;

  PetscFunctionBegin;
  TRY( QPGetEq(qp,&Beq,&ceq) );
  TRY( QPGetIneq(qp,&Bineq,&cineq) );
  TRY( QPGetQPC(qp,&qpc) );
  if (Beq || ceq || Bineq || cineq) {
    *flg = PETSC_FALSE;
  } else {
    TRY( PetscObjectTypeCompare((PetscObject)qpc,QPCBOX,flg) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetFromOptions_GP"
PetscErrorCode QPSSetFromOptions_GP(PetscOptionItems *PetscOptionsObject,QPS qps)
{
  QPS_GP    *gp = (QPS_GP*)qps->data;
  //PetscBool flg1,flg2,alpha_direct;
  //PetscReal maxeig,maxeig_tol,alpha,gamma;
  //PetscInt maxeig_iter;

  PetscFunctionBegin;
  TRY( PetscOptionsHead(PetscOptionsObject,"QPS GP options") );

  //TRY( PetscOptionsBool("-qps_mpgp_alpha_direct","","QPSMPGPSetAlpha",(PetscBool) mpgp->alpha_type,&alpha_direct,&flg0) );
  //TRY( PetscOptionsReal("-qps_mpgp_alpha","","QPSMPGPSetAlpha",mpgp->alpha_user,&alpha,&flg2) );
  //if (flg1 || flg2) TRY( QPSMPGPSetAlpha(qps,alpha,(QPSScalarArgType) alpha_direct) );
  //TRY( PetscOptionsReal("-qps_mpgp_gamma","","QPSMPGPSetGamma",mpgp->gamma,&gamma,&flg1) );
  //if (flg1) TRY( QPSMPGPSetGamma(qps,gamma) );
  //TRY( PetscOptionsReal("-qps_mpgp_maxeig","Approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means this is automatically computed.","QPSMPGPSetOperatorMaxEigenvalue",mpgp->maxeig,&maxeig,&flg1) );
  //if (flg1) TRY( QPSMPGPSetOperatorMaxEigenvalue(qps,maxeig) );
  //TRY( PetscOptionsReal("-qps_mpgp_maxeig_tol","Relative tolerance to find approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means QPS rtol","QPSMPGPSetOperatorMaxEigenvalueTolerance",mpgp->maxeig_tol,&maxeig_tol,&flg1) );
  //if (flg1) TRY( QPSMPGPSetOperatorMaxEigenvalueTolerance(qps,maxeig_tol) );
  //TRY( PetscOptionsInt("-qps_mpgp_maxeig_iter","Number of iterations to find an approximate maximum eigenvalue of the Hessian","QPSMPGPSetOperatorMaxEigenvalueIterations",mpgp->maxeig_iter,&maxeig_iter,&flg1) );
  //if (flg1) TRY( QPSMPGPSetOperatorMaxEigenvalueIterations(qps,maxeig_iter) );
  //TRY( PetscOptionsReal("-qps_mpgp_btol","Boundary overshoot tolerance; default: 10*PETSC_MACHINE_EPSILON","",mpgp->btol,&mpgp->btol,&flg1) );
  //TRY( PetscOptionsReal("-qps_mpgp_bound_chop_tol","Sets boundary to 0 for |boundary|<tol ; default: 0","",mpgp->bchop_tol,&mpgp->bchop_tol,NULL) );
  //TRY( PetscOptionsEnum("-qps_mpgp_expansion_type","Set expansion step type","",QPSMPGPExpansionTypes,(PetscEnum)mpgp->exptype,(PetscEnum*)&mpgp->exptype,NULL) );
  //TRY( PetscOptionsEnum("-qps_mpgp_expansion_length_type","Set expansion step length type","",QPSMPGPExpansionLengthTypes,(PetscEnum)mpgp->explengthtype,(PetscEnum*)&mpgp->explengthtype,NULL) );
  //TRY( PetscOptionsBool("-qps_mpgp_alpha_reset","If alpha=Nan reset to initial value, otherwise keep last alpaha","QPSMPGPSetAlpha",(PetscBool) mpgp->resetalpha,&mpgp->resetalpha,NULL) );
  //TRY( PetscOptionsBool("-qps_mpgp_fallback","","",(PetscBool) mpgp->fallback,&mpgp->fallback,NULL) );
  //TRY( PetscOptionsBool("-qps_mpgp_fallback2","","",(PetscBool) mpgp->fallback2,&mpgp->fallback2,NULL) );
  //if (mpgp->fallback2) mpgp->fallback = PETSC_FALSE;
  TRY( PetscOptionsTail() );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_GP"
PetscErrorCode QPSViewConvergence_GP(QPS qps, PetscViewer v)
{
  QPS_GP      *gp = (QPS_GP*)qps->data;
  PetscBool   iascii;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (iascii) {
    TRY( PetscViewerASCIIPrintf(v,"from the last QPSReset:\n") );
    TRY( PetscViewerASCIIPrintf(v,"number of Hessian multiplications %d\n",gp->nmv) );
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "QPSCreate_GP"
FLLOP_EXTERN PetscErrorCode QPSCreate_GP(QPS qps)
{
  QPS_GP         *gp;
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  TRY( PetscNewLog(qps,&gp) );
  qps->data                  = (void*)gp;

  gp->btol                 = 10*PETSC_MACHINE_EPSILON; /* boundary tol */
  gp->bchop_tol            = 0.0; /* chop of bounds */

  /* line search */
  gp->linesearch  = GPLineSearch_Default;
  gp->ls_beta     = 0.5;
  gp->ls_gamma    = 1e-4;
  gp->ls_M        = 10;
  gp->ls_maxit    = 1000;

  /* step length */
  gp->steplength  = GPStepLength_Default;
  gp->sl_alphamin = 1e-10;
  gp->sl_alphamax = 1e6;
  gp->sl_alpha    = 1.0;
  gp->sl_tau      = 0.5;
  gp->sl_fact     = 1.1;
  gp->sl_M        = 2;

  qps->ops->setup            = QPSSetup_GP;
  qps->ops->solve            = QPSSolve_GP;
  qps->ops->resetstatistics  = QPSResetStatistics_GP;
  qps->ops->destroy          = QPSDestroy_GP;
  qps->ops->isqpcompatible   = QPSIsQPCompatible_GP;
  qps->ops->setfromoptions   = QPSSetFromOptions_GP;
  qps->ops->monitor          = QPSMonitorDefault_GP;
  qps->ops->viewconvergence  = QPSViewConvergence_GP;

  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetCurrentStepType_GP_C",QPSMPGPGetCurrentStepType_GP) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetAlpha_GP_C",QPSMPGPGetAlpha_GP) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetAlpha_GP_C",QPSMPGPSetAlpha_GP) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetGamma_GP_C",QPSMPGPGetGamma_GP) );
  //TRY( PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetGamma_GP_C",QPSMPGPSetGamma_GP) );
  
  if (!registered) {
    TRY( PetscLogEventRegister("QPSGP:LineSearch", QPS_CLASSID, &QPS_GP_LineSearch) );
    TRY( PetscLogEventRegister("QPSGP:StepLength", QPS_CLASSID, &QPS_GP_StepLength) );
    registered = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}


//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPGetCurrentStepType"
//PetscErrorCode QPSMPGPGetCurrentStepType(QPS qps,char *stepType)
//{
//  PetscFunctionBegin;
//  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
//  if (stepType) PetscValidRealPointer(stepType,2);
//  *stepType = ' ';
//  TRY( PetscTryMethod(qps,"QPSMPGPGetCurrentStepType_GP_C",(QPS,char*),(qps,stepType)) );
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPGetAlpha"
///*@
//QPSMPGPGetAlpha - get the constant step-size used in algorithm based on spectral properties of Hessian matrix
//
//Parameters:
//+ qps - QP solver
//. alpha - pointer to store the value
//- argtype -
//
//Level: advanced
//@*/
//PetscErrorCode QPSMPGPGetAlpha(QPS qps,PetscReal *alpha,QPSScalarArgType *argtype)
//{
//  PetscFunctionBegin;
//  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
//  if (alpha) PetscValidPointer(alpha,2);
//  if (argtype) PetscValidPointer(argtype,3);
//  TRY( PetscUseMethod(qps,"QPSMPGPGetAlpha_GP_C",(QPS,PetscReal*,QPSScalarArgType*),(qps,alpha,argtype)) );
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPSetAlpha"
///*@
//QPSMPGPSetAlpha - set the constant step-size used in algorithm based on spectral properties of Hessian matrix
//
//Parameters:
//+ qps - QP solver
//. alpha - new value of parameter
//- argtype -
//
//Level: intermediate
//@*/
//PetscErrorCode QPSMPGPSetAlpha(QPS qps,PetscReal alpha,QPSScalarArgType argtype)
//{
//  PetscFunctionBegin;
//  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
//  PetscValidLogicalCollectiveReal(qps,alpha,2);
//  TRY( PetscTryMethod(qps,"QPSMPGPSetAlpha_GP_C",(QPS,PetscReal,QPSScalarArgType),(qps,alpha,argtype)) );
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPGetGamma"
///*@
//QPSMPGPGetGamma - get the proportioning parameter used in algorithm
//
//Parameters:
//+ qps - QP solver
//- gamma - pointer to store the value
//
//Level: advanced
//@*/
//PetscErrorCode QPSMPGPGetGamma(QPS qps,PetscReal *gamma)
//{
//  PetscFunctionBegin;
//  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
//  PetscValidPointer(gamma,2);
//  TRY( PetscUseMethod(qps,"QPSMPGPGetGamma_GP_C",(QPS,PetscReal*),(qps,gamma)) );
//  PetscFunctionReturn(0);
//}
//
//#undef __FUNCT__
//#define __FUNCT__ "QPSMPGPSetGamma"
///*@
//QPSMPGPSetGamma - set the proportioning parameter used in algorithm
//
//Parameters:
//+ qps - QP solver
//- gamma - new value of parameter
//
//Level: intermediate
//@*/
//PetscErrorCode QPSMPGPSetGamma(QPS qps,PetscReal gamma)
//{
//  PetscFunctionBegin;
//  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
//  PetscValidLogicalCollectiveReal(qps,gamma,2);
//  TRY( PetscTryMethod(qps,"QPSMPGPSetGamma_GP_C",(QPS,PetscReal),(qps,gamma)) );
//  PetscFunctionReturn(0);
//}

