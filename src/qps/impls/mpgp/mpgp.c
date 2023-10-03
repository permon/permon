
#include <../src/qps/impls/mpgp/mpgpimpl.h>

const char *const QPSMPGPExpansionTypes[] = {"std","projcg","gf","g","gfgr","ggr","QPSMPGPExpansionType","QPS_MPGP_EXPANSION_",0};
const char *const QPSMPGPExpansionLengthTypes[] = {"fixed","opt","optapprox","bb","QPSMPGPExpansionLengthType","QPS_MPGP_EXPANSION_LENGTH_",0};

/*
  WORK VECTORS:

  gP = qps->work[0];
  gf = qps->work[1];
  gc = qps->work[2];

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
     PetscCall(PetscViewerASCIIPrintf(viewer,"  Projected gradient norms for %s solve.\n",((PetscObject)qps)->prefix));
   }

   PetscCall(PetscViewerASCIIPrintf(viewer,"%3" PetscInt_FMT " MPGP [%c] ||gp||=%.10e",n,mpgp->currentStepType,(double)qps->rnorm));
   PetscCall(PetscViewerASCIIPrintf(viewer,",\t||gf||=%.10e",(double)mpgp->gfnorm));
   PetscCall(PetscViewerASCIIPrintf(viewer,",\t||gc||=%.10e",(double)mpgp->gcnorm));
   PetscCall(PetscViewerASCIIPrintf(viewer,",\talpha=%.10e",(double)mpgp->alpha));
   PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
   PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetCurrentStepType_MPGP"
PetscErrorCode QPSMPGPGetCurrentStepType_MPGP(QPS qps,char *stepType)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  *stepType = mpgp->currentStepType;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetAlpha_MPGP"
static PetscErrorCode QPSMPGPGetAlpha_MPGP(QPS qps,PetscReal *alpha,QPSScalarArgType *argtype)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  if (alpha) *alpha = mpgp->alpha_user;
  if (argtype) *argtype = mpgp->alpha_type;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetGamma_MPGP"
static PetscErrorCode QPSMPGPGetGamma_MPGP(QPS qps,PetscReal *gamma)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  *gamma = mpgp->gamma;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetGamma_MPGP"
static PetscErrorCode QPSMPGPSetGamma_MPGP(QPS qps,PetscReal gamma)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  mpgp->gamma = gamma;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalue_MPGP"
static PetscErrorCode QPSMPGPGetOperatorMaxEigenvalue_MPGP(QPS qps,PetscReal *maxeig)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  *maxeig = mpgp->maxeig;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalue_MPGP"
static PetscErrorCode QPSMPGPSetOperatorMaxEigenvalue_MPGP(QPS qps,PetscReal maxeig)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  mpgp->maxeig = maxeig;
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPUpdateMaxEigenvalue_MPGP"
static PetscErrorCode  QPSMPGPUpdateMaxEigenvalue_MPGP(QPS qps, PetscReal maxeig_update)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  PetscReal maxeig_old = mpgp->maxeig;
  PetscReal alpha_old = mpgp->alpha;

  PetscFunctionBegin;
  if (!qps->setupcalled) SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_ARG_WRONGSTATE,"this routine is intended to be called after QPSSetUp");

  mpgp->maxeig = maxeig_old*maxeig_update;
  PetscCall(PetscInfo(qps,"updating maxeig := %.8e = %.8e * %.8e = maxeig * maxeig_update\n",mpgp->maxeig,maxeig_old,maxeig_update));

  if (mpgp->alpha_type == QPS_ARG_MULTIPLE) {
    mpgp->alpha = alpha_old/maxeig_update;
    PetscCall(PetscInfo(qps,"updating alpha := %.8e = %.8e / %.8e = alpha / maxeig_update\n",mpgp->alpha,alpha_old,maxeig_update));
  }

  //TODO temporary
  if (FllopDebugEnabled) {
    PetscReal lambda;
    PetscCall(MatGetMaxEigenvalue(qps->solQP->A,NULL,&lambda,mpgp->maxeig_tol,mpgp->maxeig_iter));
    PetscCall(FllopDebug1("|maxeig_from_power_method - mpgp->maxeig| = %8e\n",PetscAbs(lambda-mpgp->maxeig)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP"
static PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP(QPS qps,PetscReal *tol)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  *tol = mpgp->maxeig_tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP"
static PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP(QPS qps,PetscReal tol)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  mpgp->maxeig_tol = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP"
static PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP(QPS qps,PetscInt *numit)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  *numit = mpgp->maxeig_iter;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP"
static PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP(QPS qps,PetscInt numit)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  mpgp->maxeig_iter = numit;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MPGPGrads"
/*
MPGPGrads - compute projected, chopped, and free gradient

Parameters:
+ qps - QP solver
- g - gradient
*/
static PetscErrorCode MPGPGrads(QPS qps, Vec x, Vec g)
{
  QP                qp;
  QPC               qpc;

  Vec               gP;                 /* ... projected gradient               */
  Vec               gr;                 /* ... reduced free gradient            */
  Vec               gc;                 /* ... chopped gradient                 */
  Vec               gf;                 /* ... free gradient                    */

  QPS_MPGP          *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetQPC(qp,&qpc));

  gP                = qps->work[0];
  gr                = qps->work[6];
  gf                = qps->work[1];
  gc                = qps->work[2];

  PetscCall(QPCGrads(qpc,x,g,gf,gc));
  PetscCall(QPCGradReduced(qpc,x,gf,mpgp->alpha,gr));
  PetscCall(VecWAXPY(gP,1.0,gf,gc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MPGPExpansionLength"
/*
MPGPExpansionLength - compute expanson step length type

Parameters:
. qps   - QP solver
*/
static PetscErrorCode MPGPExpansionLength(QPS qps)
{
  QP                qp;
  Mat               A;
  Vec               x,vecs[2];
  PetscReal         dots[2];
  QPS_MPGP          *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetOperator(qp, &A));                   /* get hessian matrix */
  switch (mpgp->explengthtype) {
    case QPS_MPGP_EXPANSION_LENGTH_FIXED:
      break;
    case QPS_MPGP_EXPANSION_LENGTH_OPT:
      vecs[0] = qps->work[3]; /* g */
      vecs[1] = qps->work[5]; /* Ap  */
      PetscCall(MatMult(A,mpgp->explengthvec,vecs[1]));
      mpgp->nmv++;
      PetscCall(VecMDot(mpgp->explengthvec,2,vecs,dots));
      if (dots[1] == .0 && mpgp->resetalpha) {  /* TODO dots[1] is tiny? */
        mpgp->alpha = mpgp->alpha/mpgp->maxeig;
      } else {
        mpgp->alpha = mpgp->alpha_user*dots[0]/dots[1];
      }
      break;
    case QPS_MPGP_EXPANSION_LENGTH_OPTAPPROX:
      vecs[0] = qps->work[3]; /* g */
      vecs[1] = mpgp->explengthvec;
      PetscCall(VecMDot(mpgp->explengthvec,2,vecs,dots));
      mpgp->alpha = mpgp->alpha_user*dots[0]/dots[1];
      mpgp->alpha = mpgp->alpha/mpgp->maxeig;
      break;
    case QPS_MPGP_EXPANSION_LENGTH_BB:
      PetscCall(QPGetSolutionVector(qp, &x));
      vecs[0] = mpgp->explengthvecold;
      vecs[1] = mpgp->xold;
      PetscCall(VecAYPX(vecs[0],-1.0,mpgp->explengthvec)); /* s_k = x_k - x_{k-1} */
      PetscCall(VecAYPX(vecs[1],-1.0,x)); /* y+k = d_k - d_{k-1} */
      PetscCall(VecMDot(vecs[0],2,vecs,dots));
      if (dots[1] == .0 && mpgp->resetalpha) {  /* TODO dots[1] is tiny? can be skipped?*/
        mpgp->alpha = mpgp->alpha/mpgp->maxeig;
      } else {
        mpgp->alpha = mpgp->alpha_user*dots[0]/dots[1];
      }
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_PLIB,"Unknown MPGP expansion length type");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MPGPExpansion_Std"
/*
MPGPExpansion - expand active set

Parameters:
+ qps   - QP solver
. afeas - feasible step length in p direction
- acg   - cg step length in p direction
*/
static PetscErrorCode MPGPExpansion_Std(QPS qps, PetscReal afeas, PetscReal acg)
{
  QP                qp;
  Vec               g;                  /* ... gradient                         */
  Vec               p;                  /* ... conjugate gradient               */
  Vec               Ap;                 /* ... multiplicated vector             */
  Vec               x;
  QPS_MPGP          *mpgp = (QPS_MPGP*)qps->data;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetSolutionVector(qp, &x));
  g                 = qps->work[3];
  p                 = qps->work[4];
  Ap                = qps->work[5];

  /* make maximal feasible step */
  PetscCall(VecAXPY(x, -afeas, p));             /* x=x-afeas*p*/
  PetscCall(VecAXPY(g, -afeas, Ap));            /* g=g-afeas*Ap    */
  PetscCall(MPGPGrads(qps, x, g));              /* grad. splitting  gP,gf,gc,gr */

  PetscCall(MPGPExpansionLength(qps));
  PetscCall(VecAXPY(x, -mpgp->alpha, mpgp->expdirection));      /* x=x-abar*direction */
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MPGPExpansion_ProjCG"
/*
MPGPExpansion - expand active set

Parameters:
+ qps   - QP solver
. afeas - feasible step length in p direction
- acg   - cg step length in p direction
*/
static PetscErrorCode MPGPExpansion_ProjCG(QPS qps, PetscReal afeas, PetscReal acg)
{
  QP                qp;
  Vec               p;                  /* ... conjugate gradient               */
  Vec               x;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetSolutionVector(qp, &x));
  p                 = qps->work[4];

  /* make projected CG step */
  PetscCall(VecAXPY(x, -acg, p));               /* x=x-acg*p      */
  PetscFunctionReturn(PETSC_SUCCESS);
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
  Vec               lb,ub;

  PetscFunctionBegin;
  /* set the number of working vectors */
  if (mpgp->fallback || mpgp->fallback2) {
    if (mpgp->explengthtype != QPS_MPGP_EXPANSION_LENGTH_BB) {
      PetscCall(QPSSetWorkVecs(qps,9));
    } else {
      PetscCall(QPSSetWorkVecs(qps,10));
    }
  } else if (mpgp->explengthtype == QPS_MPGP_EXPANSION_LENGTH_BB) {
      PetscCall(QPSSetWorkVecs(qps,9));
  } else {
    PetscCall(QPSSetWorkVecs(qps,7));
  }

  PetscCall(QPGetBox(qps->solQP,NULL,&lb,&ub));
  if (mpgp->bchop_tol) {
    if (lb) PetscCall(VecChop(lb,mpgp->bchop_tol));
    if (ub) PetscCall(VecChop(ub,mpgp->bchop_tol));
  }

  switch (mpgp->exptype) {
    case QPS_MPGP_EXPANSION_STD:
      mpgp->expdirection = qps->work[6];     /* gr */
      mpgp->explengthvec = qps->work[6];
      if (mpgp->explengthtype == QPS_MPGP_EXPANSION_LENGTH_FIXED) {
        mpgp->expproject   = PETSC_FALSE;
      }
      break;
    case QPS_MPGP_EXPANSION_GF:
      mpgp->expdirection = qps->work[1];     /* gf */
      mpgp->explengthvec = qps->work[1];
      break;
    case QPS_MPGP_EXPANSION_G:
      mpgp->expdirection = qps->work[3];     /* g  */
      mpgp->explengthvec = qps->work[3];
      break;
    case QPS_MPGP_EXPANSION_GFGR:
      mpgp->expdirection = qps->work[1];     /* gf */
      mpgp->explengthvec = qps->work[6];
      break;
    case QPS_MPGP_EXPANSION_GGR:
      mpgp->expdirection = qps->work[3];     /* g  */
      mpgp->explengthvec = qps->work[6];
      break;
    case QPS_MPGP_EXPANSION_PROJCG:
      mpgp->expansion = MPGPExpansion_ProjCG;
      //falback
      mpgp->expdirection = qps->work[1];     /* gf */
      mpgp->explengthvec = qps->work[1];
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)qps),PETSC_ERR_PLIB,"Unknown MPGP expansion type");
  }

  /* initialize alpha */
  if (mpgp->alpha_type == QPS_ARG_MULTIPLE) {
    if (mpgp->maxeig == PETSC_DECIDE) {
      PetscCall(MatGetMaxEigenvalue(qps->solQP->A, NULL, &mpgp->maxeig, mpgp->maxeig_tol, mpgp->maxeig_iter));
    }
    if (mpgp->alpha_user == PETSC_DECIDE) {
      mpgp->alpha_user = 2.0;
    }
    PetscCall(PetscInfo(qps,"maxeig     = %.8e\n", mpgp->maxeig));
    PetscCall(PetscInfo(qps,"alpha_user = %.8e\n", mpgp->alpha_user));
    mpgp->alpha = mpgp->alpha_user/mpgp->maxeig;
  } else {
    mpgp->alpha = mpgp->alpha_user;
  }
  PetscCall(PetscInfo(qps,  "alpha      = %.8e\n", mpgp->alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  QPC               qpc;
  Mat               A;                  /* ... hessian matrix                   */
  Vec               b;                  /* ... right-hand side vector           */
  Vec               x;                  /* ... vector of variables              */
  Vec               gP;                 /* ... projected gradient               */
  Vec               gc;                 /* ... chopped gradient                 */
  Vec               gf;                 /* ... free gradient                    */
  Vec               g;                  /* ... gradient                         */
  Vec               p;                  /* ... conjugate gradient               */
  Vec               Ap;                 /* ... multiplicated vector             */
  Vec               gold;               /* ... old gradient for fallback        */

  PetscReal         gamma2;             /* ... algorithm constants              */
  PetscReal         acg;                /* ... conjugate gradient step-size     */
  PetscReal         bcg;                /* ... cg ortogonalization parameter    */
  PetscReal         afeas;              /* ... maximum feasible step-size       */
  PetscReal         pAp, gcTgc, gfTgf;  /* ... results of dot products          */
  PetscReal         f,fold;             /* ... cost function value              */

  PetscInt          nmv=0;              /* ... matrix-vector mult. counter      */
  PetscInt          ncg=0;              /* ... cg step counter                  */
  PetscInt          nprop=0;            /* ... proportional step counter        */
  PetscInt          nexp=0;             /* ... expansion step counter           */

  PetscInt          nfinc=0;            /* ... functional increase counter      */
  PetscInt          nfall=0;            /* ... fallback step counter            */

  PetscFunctionBegin;
  /* set working vectors */
  gP                = qps->work[0];
  gf                = qps->work[1];
  gc                = qps->work[2];

  g                 = qps->work[3];
  p                 = qps->work[4];
  Ap                = qps->work[5];

  if (mpgp->explengthtype == QPS_MPGP_EXPANSION_LENGTH_BB) {
    mpgp->explengthvecold = qps->work[7];
    mpgp->xold            = qps->work[8];
    if (mpgp->fallback || mpgp->fallback2) {
      gold                = qps->work[9];
    }
  } else if (mpgp->fallback || mpgp->fallback2) {
      mpgp->xold            = qps->work[7];
      gold                  = qps->work[8];
  }

  /* set constants of algorithm */
  gamma2            = mpgp->gamma*mpgp->gamma;

  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetQPC(qp,&qpc));                       /* get constraints */
  PetscCall(QPGetSolutionVector(qp, &x));             /* get the solution vector */
  PetscCall(QPGetOperator(qp, &A));                   /* get hessian matrix */
  PetscCall(QPGetRhs(qp, &b));                        /* get right-hand side vector */

  PetscCall(QPCProject(qpc,x,x));                     /* project x initial guess to feasible set */

  /* compute gradient */
  PetscCall(MatMult(A, x, g));                        /* g=A*x */
  nmv++;                                          /* matrix multiplication counter */
  PetscCall(VecAXPY(g, -1.0, b));                     /* g=g-b */

  PetscCall(MPGPGrads(qps, x, g));                    /* grad. splitting  gP,gf,gc */

  /* initiate CG method */
  PetscCall(VecCopy(gf, p));                          /* p=gf */

  mpgp->currentStepType = ' ';
  qps->iteration = 0;                             /* main iteration counter */
  while (1)                                       /* main cycle */
  {
    /* compute the norm of projected gradient - stopping criterion */
    PetscCall(VecNorm(gP, NORM_2, &qps->rnorm));      /* qps->rnorm=norm(gP)*/

    /* compute dot products to control the proportionality */
    PetscCall(VecDot(gc, gc, &gcTgc));               /* gcTgc=gc'*gc   */
    /* NOTE: using gf'*gf for proportiong rule instead of gr'*gf
    *  which can lead to more agressive proportioning as
    *  sqrt(g_reduced^T * g_free) <= ||g_free||                    */
    PetscCall(VecDot(gf, gf, &gfTgf));               /* gfTgf=gr'*gf   */

    /* compute norm of gf, gc from computed dot products */
    if (qps->numbermonitors) {
      mpgp->gfnorm =  PetscSqrtScalar(gfTgf);
      mpgp->gcnorm =  PetscSqrtScalar(gcTgc);
      PetscCall(QPSMonitor(qps,qps->iteration,qps->rnorm)) ;
    }

    /* test the convergence of algorithm */
    PetscCall((*qps->convergencetest)(qps,&qps->reason)); /* test for convergence */
    if (qps->reason != KSP_CONVERGED_ITERATING) break;

    /* proportional condition */
    if (gcTgc <= gamma2*gfTgf)                    /* u is proportional */
    {
      PetscCall(MatMult(A, p, Ap));                   /* Ap=A*p */
      nmv++;                                      /* matrix multiplication counter */

      /* compute step-sizes */
      PetscCall(VecDot(p, Ap, &pAp));                 /* pAp=p'*Ap      */
      PetscCall(VecDot(g,  p, &acg));                 /* acg=g'*p       */
      acg  = acg/pAp;                             /* acg=acg/pAp    */
      PetscCall(QPCFeas(qpc, x, p, &afeas));          /* finds max.feas.steplength */

      /* decide if it is able to do full CG step */
      if (acg <= afeas)
      {
        /* CONJUGATE GRADIENT STEP */
        ncg++;                                    /* increase CG step counter */
        mpgp->currentStepType = 'c';

        /* make CG step */
        PetscCall(VecAXPY(x, -acg, p));               /* x=x-acg*p      */
        PetscCall(VecAXPY(g, -acg, Ap));              /* g=g-acg*Ap      */
        PetscCall(MPGPGrads(qps, x, g));              /* grad. splitting  gP,gf,gc */

        /* compute orthogonalization parameter and next orthogonal vector */
        PetscCall(VecDot(Ap, gf, &bcg));              /* bcg=Ap'*gf     */
        bcg  = bcg/pAp;                           /* bcg=bcg/pAp     */
        PetscCall(VecAYPX(p, -bcg, gf));              /* p=gf-bcg*p     */
      }
      else                                        /* expansion step  */
      {
        /* EXPANSION STEP */
        nexp++;                                   /* increase expansion step counter */
        mpgp->currentStepType = 'e';

        /* save old direction vec for BB expansion step length */
        if (mpgp->explengthtype == QPS_MPGP_EXPANSION_LENGTH_BB || mpgp->fallback || mpgp->fallback2) {
          PetscCall(VecCopy(x,mpgp->xold));
          if (mpgp->explengthtype == QPS_MPGP_EXPANSION_LENGTH_BB) {
            PetscCall(VecCopy(mpgp->explengthvec,mpgp->explengthvecold));
          }
        }

        PetscCall(mpgp->expansion(qps,afeas,acg));
        if (mpgp->expproject) {
          PetscCall(QPCProject(qpc, x, x));             /* project x to feas.set */
        }

        /* compute new gradient */
        if (mpgp->fallback || mpgp->fallback2) {
          PetscCall(VecCopy(g,gold));
        }
        PetscCall(MatMult(A, x, g));                  /* g=A*x */
        nmv++;                                    /* matrix multiplication counter */
        PetscCall(VecAXPY(g, -1.0, b));               /* g=g-b           */

        if (mpgp->fallback || mpgp->fallback2) {
          PetscCall(QPComputeObjectiveFromGradient(qp, mpgp->xold, gold, &fold));
          PetscCall(QPComputeObjectiveFromGradient(qp, x, g, &f));
          if (f>fold) {
            nfinc++;
            if (mpgp->fallback2) {
              PetscCall(MPGPGrads(qps, x, g));              /* grad. splitting  gP,gf,gc */
              PetscCall(VecDot(gc, gc, &gcTgc));               /* gcTgc=gc'*gc   */
              PetscCall(VecDot(gf, gf, &gfTgf));               /* gfTgf=gr'*gf   */
              if (gcTgc <= gamma2*gfTgf) {                   /* u is proportional */
                mpgp->fallback = PETSC_FALSE;
              } else {
                mpgp->fallback = PETSC_TRUE;
              }
            }

            if (mpgp->fallback){
              nfall++;
              mpgp->currentStepType = 'f';
              PetscCall(VecCopy(mpgp->xold, x));
              PetscCall(VecCopy(gold, g));
              if (mpgp->fallback2) {
                PetscCall(MPGPGrads(qps, mpgp->xold, gold));              /* grad. splitting  gP,gf,gc */
              }
              PetscCall(MPGPExpansion_Std(qps, afeas, acg));
              PetscCall(QPCProject(qpc, x, x));             /* project x to feas.set */
              PetscCall(MatMult(A, x, g));                  /* g=A*x */
              nmv++;                                    /* matrix multiplication counter */
              PetscCall(VecAXPY(g, -1.0, b));               /* g=g-b           */
            }
          }
        }

        PetscCall(MPGPGrads(qps, x, g));              /* grad. splitting  gP,gf,gc */
        /* restart CG method */
        PetscCall(VecCopy(gf, p));                    /* p=gf           */
      }
    }
    else                                          /* proportioning step  */
    {
      /* PROPORTIONING STEP */
      nprop++;                                    /* increase proportioning step counter */
      mpgp->currentStepType = 'p';

      PetscCall(VecCopy(gc, p));                      /* p=gc           */
      PetscCall(MatMult(A, p, Ap));                   /* Ap=A*p */
      nmv++;                                      /* matrix multiplication counter */

      /* compute step-size */
      PetscCall(VecDot(p, Ap, &pAp));                 /* pAp=p'*Ap       */
      PetscCall(VecDot(g,  p, &acg));                 /* acg=g'*p        */
      acg  = acg/pAp;                             /* acg=acg/pAp     */

      /* make a step */
      PetscCall(VecAXPY(x, -acg, p));                 /* x=x-acg*p       */
      PetscCall(VecAXPY(g, -acg, Ap));                /* g=g-acg*Ap      */
      PetscCall(MPGPGrads(qps, x, g));                /* grad. splitting  gP,gf,gc */

      /* restart CG method */
      PetscCall(VecCopy(gf, p));                      /* p=gf           */
    }
    qps->iteration++;
  };

  mpgp->ncg     += ncg;
  mpgp->nexp    += nexp;
  mpgp->nmv     += nmv;
  mpgp->nprop   += nprop;
  mpgp->nfinc    += nfinc;
  mpgp->nfall   += nfall;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSResetStatistics_MPGP"
PetscErrorCode QPSResetStatistics_MPGP(QPS qps)
{
  QPS_MPGP *mpgp = (QPS_MPGP*)qps->data;
  PetscFunctionBegin;
  mpgp->ncg   = 0;
  mpgp->nexp  = 0;
  mpgp->nmv   = 0;
  mpgp->nprop = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetCurrentStepType_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetAlpha_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetAlpha_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetGamma_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetGamma_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalue_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalue_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPUpdateMaxEigenvalue_MPGP_C",NULL));
  PetscCall(QPSDestroyDefault(qps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSIsQPCompatible_MPGP"
PetscErrorCode QPSIsQPCompatible_MPGP(QPS qps,QP qp,PetscBool *flg)
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
#define __FUNCT__ "QPSSetFromOptions_MPGP"
PetscErrorCode QPSSetFromOptions_MPGP(QPS qps,PetscOptionItems *PetscOptionsObject)
{
  QPS_MPGP    *mpgp = (QPS_MPGP*)qps->data;
  PetscBool flg1,flg2,alpha_direct;
  PetscReal maxeig,maxeig_tol,alpha,gamma;
  PetscInt maxeig_iter;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"QPS MPGP options");

  alpha_direct = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-qps_mpgp_alpha_direct","","QPSMPGPSetAlpha",(PetscBool) mpgp->alpha_type,&alpha_direct,&flg1));
  PetscCall(PetscOptionsReal("-qps_mpgp_alpha","","QPSMPGPSetAlpha",mpgp->alpha_user,&alpha,&flg2));
  if (flg1 || flg2) PetscCall(QPSMPGPSetAlpha(qps,alpha,(QPSScalarArgType) alpha_direct));
  PetscCall(PetscOptionsReal("-qps_mpgp_gamma","","QPSMPGPSetGamma",mpgp->gamma,&gamma,&flg1));
  if (flg1) PetscCall(QPSMPGPSetGamma(qps,gamma));
  PetscCall(PetscOptionsReal("-qps_mpgp_maxeig","Approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means this is automatically computed.","QPSMPGPSetOperatorMaxEigenvalue",mpgp->maxeig,&maxeig,&flg1));
  if (flg1) PetscCall(QPSMPGPSetOperatorMaxEigenvalue(qps,maxeig));
  PetscCall(PetscOptionsReal("-qps_mpgp_maxeig_tol","Relative tolerance to find approximate maximum eigenvalue of the Hessian, PETSC_DECIDE means QPS rtol","QPSMPGPSetOperatorMaxEigenvalueTolerance",mpgp->maxeig_tol,&maxeig_tol,&flg1));
  if (flg1) PetscCall(QPSMPGPSetOperatorMaxEigenvalueTolerance(qps,maxeig_tol));
  PetscCall(PetscOptionsInt("-qps_mpgp_maxeig_iter","Number of iterations to find an approximate maximum eigenvalue of the Hessian","QPSMPGPSetOperatorMaxEigenvalueIterations",mpgp->maxeig_iter,&maxeig_iter,&flg1));
  if (flg1) PetscCall(QPSMPGPSetOperatorMaxEigenvalueIterations(qps,maxeig_iter));
  PetscCall(PetscOptionsReal("-qps_mpgp_btol","Boundary overshoot tolerance; default: 10*PETSC_MACHINE_EPSILON","",mpgp->btol,&mpgp->btol,&flg1));
  PetscCall(PetscOptionsReal("-qps_mpgp_bound_chop_tol","Sets boundary to 0 for |boundary|<tol ; default: 0","",mpgp->bchop_tol,&mpgp->bchop_tol,NULL));
  PetscCall(PetscOptionsEnum("-qps_mpgp_expansion_type","Set expansion step type","",QPSMPGPExpansionTypes,(PetscEnum)mpgp->exptype,(PetscEnum*)&mpgp->exptype,NULL));
  PetscCall(PetscOptionsEnum("-qps_mpgp_expansion_length_type","Set expansion step length type","",QPSMPGPExpansionLengthTypes,(PetscEnum)mpgp->explengthtype,(PetscEnum*)&mpgp->explengthtype,NULL));
  PetscCall(PetscOptionsBool("-qps_mpgp_alpha_reset","If alpha=Nan reset to initial value, otherwise keep last alpaha","QPSMPGPSetAlpha",(PetscBool) mpgp->resetalpha,&mpgp->resetalpha,NULL));
  PetscCall(PetscOptionsBool("-qps_mpgp_fallback","Throw away expansion step if cost function increased and do a std expansion step.","",(PetscBool) mpgp->fallback,&mpgp->fallback,NULL));
  PetscCall(PetscOptionsBool("-qps_mpgp_fallback2","Same as fallback which is done only if the next step is proportioning","",(PetscBool) mpgp->fallback2,&mpgp->fallback2,NULL));
  if (mpgp->fallback2) mpgp->fallback = PETSC_FALSE;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence_MPGP"
PetscErrorCode QPSViewConvergence_MPGP(QPS qps, PetscViewer v)
{
  QPS_MPGP      *mpgp = (QPS_MPGP*)qps->data;
  PetscBool     iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(v,"from the last QPSReset:\n"));
    PetscCall(PetscViewerASCIIPrintf(v,"number of Hessian multiplications %d\n",mpgp->nmv));
    PetscCall(PetscViewerASCIIPrintf(v,"number of CG steps %d\n",mpgp->ncg));
    PetscCall(PetscViewerASCIIPrintf(v,"number of expansion steps %d\n",mpgp->nexp));
    PetscCall(PetscViewerASCIIPrintf(v,"number of proportioning steps %d\n",mpgp->nprop));
    if (mpgp->fallback || mpgp->fallback2) {
      PetscCall(PetscViewerASCIIPrintf(v,"number of cost function value increases: %d\n",mpgp->nfinc));
      PetscCall(PetscViewerASCIIPrintf(v,"number of fallbacks: %d\n",mpgp->nfall));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   QPSMPGP - Modified proportioning with (reduced) gradient projections type algorithm

   This method does three types of steps, unconstrained minimization if feasible, expansion step to expand the active set, and proportioning step to reduce the active set.

   Options Database Keys:
+  -qps_mpgp_alpha_direct - true sets expansion step length value directly, false (default) multiplier (typical between (0,2]) for step length equal to reciprocal of maximal eigenvalue
.  -qps_mpgp_alpha - fixed step length value for expansion, default: 2.0
.  -qps_mpgp_gamma - proportionality constant
.  -qps_mpgp_maxeig - approximate maximum eigenvalue of the Hessian (automatically computed by default)
.  -qps_mpgp_maxeig_tol - relative tolerance for power method to find approximate maximum eigenvalue of the Hessian
.  -qps_mpgp_maxeig_iter - number of iterations of power method to find an approximate maximum eigenvalue of the Hessian
.  -qps_mpgp_btol - boundary overshoot tolerance, default: 10*PETSC_MACHINE_EPSILON"
.  -qps_mpgp_bound_chop_tol - sets boundary to 0 for |boundary|<tol, default: 0
.  -qps_mpgp_expansion_type - set expansion step type, default: "std"
.  -qps_mpgp_expansion_length_type - set expansion step length type, default: "fixed"
.  -qps_mpgp_alpha_reset - if alpha=Nan reset to initial value, otherwise keep last alpaha, default: true
.  -qps_mpgp_fallback - throw away expansion step if cost function increased and do a std expansion step, default false
-  -qps_mpgp_fallback2 - same as fallback which is done only if the next step is proportioning

   Available expansion types:
+  "std" - standard expansion
.  "projcg" - unconstrained CG step projected back to feasible set
.  "gf" - free gradient for both step length computation and expansion direction
.  "g" - normal gradient for both step length computation and expansion direction
.  "gfgr" - expansion in free gradient direction, steplength using reduced gradient
-  "ggr" - expansion in normal gradient direction, steplength using reduced gradient

   Available step lengths types:
+  "fixed" - standard fixed step length
-  "opt" - optimal step length
-  "optapprox" - (usually poor) approximation of optimal step length
.  "bb" - Barzilai-Borwein step length
   Level: intermediate

   Reference:
   . J. Kružík, D. Horák, M. Čermák, L. Pospíšil, M. Pecha, "Active set expansion strategies in MPRGP algorithm", Advances in Engineering Software, Volume 149, 2020.

.seealso:  QPSCreate(), QPSSetType(), QPSType (for list of available types), QPS,
           QPSMPGPSetAlpha(), QPSMPGPGetAlpha(), QPSMPGPSetGamma(), QPSMPGPGetGamma(),
           QPSMPGPGetOperatorMaxEigenvalue(), QPSMPGPSetOperatorMaxEigenvalue(),
           QPSMPGPUpdateMaxEigenvalue(), QPSMPGPSetOperatorMaxEigenvalueTolerance(),
           QPSMPGPGetOperatorMaxEigenvalueTolerance(), QPSMPGPGetOperatorMaxEigenvalueIterations(),
           QPSMPGPSetOperatorMaxEigenvalueIterations(), QPSMPGPGetCurrentStepType()
M*/
#undef __FUNCT__
#define __FUNCT__ "QPSCreate_MPGP"
FLLOP_EXTERN PetscErrorCode QPSCreate_MPGP(QPS qps)
{
  QPS_MPGP         *mpgp;

  PetscFunctionBegin;
  PetscCall(PetscNew(&mpgp));
  qps->data                  = (void*)mpgp;

  mpgp->alpha_user           = PETSC_DECIDE;
  mpgp->alpha_type           = QPS_ARG_MULTIPLE;
  mpgp->gamma                = 1.0;
  mpgp->maxeig               = PETSC_DECIDE;
  mpgp->maxeig_tol           = PETSC_DECIDE;
  mpgp->maxeig_iter          = PETSC_DECIDE;
  mpgp->btol                 = 10*PETSC_MACHINE_EPSILON; /* boundary tol */
  mpgp->bchop_tol            = 0.0; /* chop of bounds */

  mpgp->exptype              = QPS_MPGP_EXPANSION_STD;
  mpgp->explengthtype        = QPS_MPGP_EXPANSION_LENGTH_FIXED;
  mpgp->expansion            = MPGPExpansion_Std;
  mpgp->expproject           = PETSC_TRUE;
  mpgp->resetalpha           = PETSC_FALSE;

  mpgp->fallback              = PETSC_FALSE;
  mpgp->fallback2              = PETSC_FALSE;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  qps->ops->setup            = QPSSetup_MPGP;
  qps->ops->solve            = QPSSolve_MPGP;
  qps->ops->resetstatistics  = QPSResetStatistics_MPGP;
  qps->ops->destroy          = QPSDestroy_MPGP;
  qps->ops->isqpcompatible   = QPSIsQPCompatible_MPGP;
  qps->ops->setfromoptions   = QPSSetFromOptions_MPGP;
  qps->ops->monitor          = QPSMonitorDefault_MPGP;
  qps->ops->viewconvergence  = QPSViewConvergence_MPGP;

  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetCurrentStepType_MPGP_C",QPSMPGPGetCurrentStepType_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetAlpha_MPGP_C",QPSMPGPGetAlpha_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetAlpha_MPGP_C",QPSMPGPSetAlpha_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetGamma_MPGP_C",QPSMPGPGetGamma_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetGamma_MPGP_C",QPSMPGPSetGamma_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalue_MPGP_C",QPSMPGPGetOperatorMaxEigenvalue_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalue_MPGP_C",QPSMPGPSetOperatorMaxEigenvalue_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP_C",QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP_C",QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP_C",QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP_C",QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP));
  PetscCall(PetscObjectComposeFunction((PetscObject)qps,"QPSMPGPUpdateMaxEigenvalue_MPGP_C",QPSMPGPUpdateMaxEigenvalue_MPGP));
  PetscFunctionReturn(PETSC_SUCCESS);
}


#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetCurrentStepType"
PetscErrorCode QPSMPGPGetCurrentStepType(QPS qps,char *stepType)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  if (stepType) PetscAssertPointer(stepType,2);
  *stepType = ' ';
  PetscTryMethod(qps,"QPSMPGPGetCurrentStepType_MPGP_C",(QPS,char*),(qps,stepType));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (alpha) PetscAssertPointer(alpha,2);
  if (argtype) PetscAssertPointer(argtype,3);
  PetscUseMethod(qps,"QPSMPGPGetAlpha_MPGP_C",(QPS,PetscReal*,QPSScalarArgType*),(qps,alpha,argtype));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscTryMethod(qps,"QPSMPGPSetAlpha_MPGP_C",(QPS,PetscReal,QPSScalarArgType),(qps,alpha,argtype));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(gamma,2);
  PetscUseMethod(qps,"QPSMPGPGetGamma_MPGP_C",(QPS,PetscReal*),(qps,gamma));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscTryMethod(qps,"QPSMPGPSetGamma_MPGP_C",(QPS,PetscReal),(qps,gamma));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPGetOperatorMaxEigenvalue"
PetscErrorCode QPSMPGPGetOperatorMaxEigenvalue(QPS qps,PetscReal *maxeig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscAssertPointer(maxeig,2);
  PetscUseMethod(qps,"QPSMPGPGetOperatorMaxEigenvalue_MPGP_C",(QPS,PetscReal*),(qps,maxeig));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (maxeig < 0 && maxeig != PETSC_DECIDE) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be nonnegative");
  PetscTryMethod(qps,"QPSMPGPSetOperatorMaxEigenvalue_MPGP_C",(QPS,PetscReal),(qps,maxeig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMPGPUpdateMaxEigenvalue"
PetscErrorCode  QPSMPGPUpdateMaxEigenvalue(QPS qps, PetscReal maxeig_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps,QPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(qps,maxeig_update,2);
  if (maxeig_update == 1.0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscTryMethod(qps,"QPSMPGPUpdateMaxEigenvalue_MPGP_C",(QPS,PetscReal),(qps,maxeig_update));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscTryMethod(qps,"QPSMPGPSetOperatorMaxEigenvalueTolerance_MPGP_C",(QPS,PetscReal),(qps,tol));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(tol,2);
  PetscTryMethod(qps,"QPSMPGPGetOperatorMaxEigenvalueTolerance_MPGP_C",(QPS,PetscReal*),(qps,tol));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(numit,2);
  PetscUseMethod(qps,"QPSMPGPGetOperatorMaxEigenvalueIterations_MPGP_C",(QPS,PetscInt*),(qps,numit));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (numit <= 1) SETERRQ(((PetscObject)qps)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Argument must be > 1");
  PetscTryMethod(qps,"QPSMPGPSetOperatorMaxEigenvalueIterations_MPGP_C",(QPS,PetscInt),(qps,numit));
  PetscFunctionReturn(PETSC_SUCCESS);
}

