#include <permon/private/qpsimpl.h>

PetscClassId  QPS_CLASSID;
PetscLogEvent QPS_Solve, QPS_PostSolve;

#undef __FUNCT__
#define __FUNCT__ "QPSQPChangeListener_Private"
static PetscErrorCode QPSQPChangeListener_Private(QP qp)
{
  QPS sol;

  PetscFunctionBegin;
  PetscCall(QPGetChangeListenerContext(qp, (QPS *)&sol));
  sol->setupcalled = PETSC_FALSE;
  qp->solved       = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSAttachQP_Private"
static PetscErrorCode QPSAttachQP_Private(QPS qps, QP qp)
{
  PetscFunctionBegin;
  qps->topQP = qp;
  PetscCall(QPSetChangeListener(qp, QPSQPChangeListener_Private));
  PetscCall(QPSetChangeListenerContext(qp, qps));
  PetscCall(PetscObjectReference((PetscObject)qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSDetachQP_Private"
static PetscErrorCode QPSDetachQP_Private(QPS qps)
{
  PetscFunctionBegin;
  if (!qps->topQP) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(QPSetChangeListener(qps->topQP, NULL));
  PetscCall(QPSetChangeListenerContext(qps->topQP, NULL));
  PetscCall(QPDestroy(&qps->topQP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSCreate"
/*@
   QPSCreate - create QP Solver instance

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI comm

   Output Parameters:
.  qps_new - pointer to created QPS

   Level: beginner

.seealso QPSDestroy()
@*/
PetscErrorCode QPSCreate(MPI_Comm comm, QPS *qps_new)
{
  QPS   qps;
  void *ctx;

  PetscFunctionBegin;
  PetscAssertPointer(qps_new, 2);
  *qps_new = 0;
  PetscCall(QPSInitializePackage());

  PetscCall(PetscHeaderCreate(qps, QPS_CLASSID, "QPS", "Quadratic Programming Solver", "QPS", comm, QPSDestroy, QPSView));

  qps->rtol                   = 1e-5;
  qps->atol                   = 1e-50;
  qps->divtol                 = 1e4;
  qps->max_it                 = 10000;
  qps->autoPostSolve          = PETSC_TRUE;
  qps->topQP                  = NULL;
  qps->solQP                  = NULL;
  qps->setupcalled            = PETSC_FALSE;
  qps->postsolvecalled        = PETSC_FALSE;
  qps->user_type              = PETSC_FALSE;
  qps->iteration              = 0;
  qps->iterations_accumulated = 0;
  qps->nsolves                = 0;

  /* monitor */
  qps->res_hist       = NULL;
  qps->res_hist_alloc = NULL;
  qps->res_hist_len   = 0;
  qps->res_hist_max   = 0;
  qps->res_hist_reset = PETSC_TRUE;
  qps->numbermonitors = 0;

  PetscCall(QPSConvergedDefaultCreate(&ctx));
  PetscCall(QPSSetConvergenceTest(qps, QPSConvergedDefault, ctx, QPSConvergedDefaultDestroy));

  *qps_new = qps;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetQP"
/*@
   QPSGetQP - Return the user's QP set by QPSSetQP, or a new one if none has been set.

   Not Collective

   Parameters:
+  qps - instance of QPS
-  qp - pointer to returning QP

   Level: beginner

.seealso QPSSetQP(), QPSGetSolvedQP(), QPSSolve()
@*/
PetscErrorCode QPSGetQP(QPS qps, QP *qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(qp, 2);
  if (!qps->topQP) {
    QP qp_;
    PetscCall(QPCreate(PetscObjectComm((PetscObject)qps), &qp_));
    PetscCall(QPSAttachQP_Private(qps, qp_));
    PetscCall(QPDestroy(&qp_));
  }
  *qp = qps->topQP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetSolvedQP"
/*@
   QPSGetSolvedQP - return the actually solved QP from QPS (typically the most derived - last in QP chain)

   Not Collective

   Input Parameter:
.  qps - instance of QPS

   Output Parameter:
.  qp - pointer to returning QP

   Level: advanced

.seealso QPSGetQP(), QPSSolve()
@*/
PetscErrorCode QPSGetSolvedQP(QPS qps, QP *qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(qp, 2);
  *qp = qps->solQP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetQP"
/*@
   QPSSetQP - set the user's QP to QPS

   Not Collective, but the QPS and QP objects must live on the same MPI_Comm

   Input parameters:
+  qps - instance of QPS
-  qp - instance of QP

   Level: beginner
@*/
PetscErrorCode QPSSetQP(QPS qps, QP qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscValidHeaderSpecific(qp, QP_CLASSID, 2);
  PetscCheckSameComm(qps, 1, qp, 2);
  if (qps->topQP == qp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(QPSDetachQP_Private(qps));
  PetscCall(QPSAttachQP_Private(qps, qp));
  qps->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetUp"
/*@
   QPSSetUp - set up the QPS; prepare QP chain, check up the QPS compatibility, set up the PC preconditioner

   Collective on QPS

   Input parameter:
.  qps - instance of QPS

   Level: advanced

.seealso QPSSolve(), QPSReset(), QPChainSetUp(), QPSIsQPCompatible()
@*/
PetscErrorCode QPSSetUp(QPS qps)
{
  QP        solqp;
  PetscBool flg;

  PermonTracedFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  if (qps->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PermonTraceBegin;
  PetscCall(QPChainSetUp(qps->topQP));
  if (!qps->solQP) { PetscCall(QPChainGetLast(qps->topQP, &qps->solQP)); }
  solqp = qps->solQP;
  PetscCall(PetscObjectReference((PetscObject)qps->solQP));

  PetscCall(QPSSetDefaultTypeIfNotSpecified(qps));
  PetscCall(QPSIsQPCompatible(qps, solqp, &flg));
  PetscCheck(flg, ((PetscObject)qps)->comm, PETSC_ERR_ARG_INCOMP, "QPS solver %s is not compatible with its attached QP", ((PetscObject)qps)->type_name);

  PetscTryTypeMethod(qps, setup);
  PetscCall(QPChainSetUp(solqp));
  qps->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSReset"
/*@
   QPSReset - reset QPS; prepare the instance to new data

   Collective on QPS

   Input parameter:
.  qps - instance of QPS

   Level: beginner

.seealso QPSSolve(), QPSSetUp()
@*/
PetscErrorCode QPSReset(QPS qps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscTryTypeMethod(qps, reset);
  if (qps->topQP) PetscCall(QPDestroy(&qps->topQP));
  PetscCall(QPDestroy(&qps->solQP));
  PetscCall(VecDestroyVecs(qps->nwork, &qps->work));
  PetscCall(PetscFree(qps->work_state));
  qps->setupcalled = PETSC_FALSE;
  PetscCall(QPSResetStatistics(qps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSResetStatistics"
/*@
   QPSResetStatistics - Reset QPSSolve statistics. Called in QPSReset().

   Collective on QPS

   Input parameter:
.  qps - instance of QPS

   Level: beginner

.seealso QPSSolve(), QPSReset()
@*/
PetscErrorCode QPSResetStatistics(QPS qps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  qps->iteration              = 0;
  qps->iterations_accumulated = 0;
  qps->nsolves                = 0;

  PetscTryTypeMethod(qps, resetstatistics);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSView"
/*@
   QPSView - view the basic properties of QPS

   Collective on QPS

   Input parameters:
+  qps - instance of QPS
-  v - viewer

   Level: beginner

.seealso QPSViewConvergence()
@*/
PetscErrorCode QPSView(QPS qps, PetscViewer v)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)qps, v));
  PetscCall(PetscViewerASCIIPushTab(v));
  if (*qps->ops->view) {
    PetscUseTypeMethod(qps, view, v);
  } else {
    const QPSType type;
    PetscCall(QPSGetType(qps, &type));
    PetscCall(PetscInfo(qps, "Warning: QPSView not implemented yet for type %s\n", type));
  }
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSDestroyDefault"
/*
   QPSDestroyDefault - destroy the QPS content

   Input parameter:
.  qps - instance of QPS

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin QPS implementations

   Level: developer
*/
PetscErrorCode QPSDestroyDefault(QPS qps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscCall(PetscFree(qps->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSDestroy"
/*@
   QPSDestroy - destroy the QPS instance

   Input parameter:
.  qps - pointer to instance of QPS

   Level: beginner

.seealso QPSCreate()
@*/
PetscErrorCode QPSDestroy(QPS *qps)
{
  PetscFunctionBegin;
  if (!*qps) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*qps, QPS_CLASSID, 1);
  if (--((PetscObject)(*qps))->refct > 0) {
    *qps = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(QPSReset(*qps));
  PetscTryTypeMethod(*qps, destroy);

  if ((*qps)->convergencetestdestroy) { PetscCall((*(*qps)->convergencetestdestroy)((*qps)->cnvctx)); }

  PetscCall(QPDestroy(&(*qps)->topQP));
  PetscCall(PetscFree((*qps)->data));

  PetscCall(QPSMonitorCancel((*qps)));

  PetscCall(PetscHeaderDestroy(qps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetType"
/*@
   QPSSetType - set the type of solver

   Collective on QPS

   Input parameters:
+  qps - instance of QPS
-  type - type of the solver (QPSKSP, QPSMPGP, QPSSMALXE)

   Level: intermediate

.seealso QPSCreate()
@*/
PetscErrorCode QPSSetType(QPS qps, const QPSType type)
{
  PetscErrorCode (*create_xxx)(QPS);
  PetscBool issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)qps, type, &issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(QPSList, type, (void (**)(void))&create_xxx));
  PetscCheck(create_xxx, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested QPS type %s", type);

  /* Destroy the pre-existing private QPS context */
  PetscTryTypeMethod(qps, destroy);

  /* Reinitialize function pointers in QPSOps structure */
  PetscCall(PetscMemzero(qps->ops, sizeof(struct _QPSOps)));

  qps->setupcalled = PETSC_FALSE;
  qps->user_type   = PETSC_TRUE;

  PetscCall((*create_xxx)(qps));
  PetscCall(PetscObjectChangeTypeName((PetscObject)qps, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetDefaultType"
/*@
   QPSSetDefaultType - set the type of solver to the default one corresponding to prescribed constraints

   Collective on QPS

   Input parameters:
.  qps - instance of QPS

   Level: advanced

.seealso QPSSetType()
@*/
PetscErrorCode QPSSetDefaultType(QPS qps)
{
  QP  qp;
  Mat Beq, Bineq;
  Vec ceq, cineq;
  QPC qpc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscCheck(qps->topQP, ((PetscObject)qps)->comm, PETSC_ERR_ORDER, "QPS needs QP to be set in order to find a default type");

  PetscCall(QPChainGetLast(qps->topQP, &qp));

  PetscCall(QPGetEq(qp, &Beq, &ceq));
  PetscCall(QPGetIneq(qp, &Bineq, &cineq));
  PetscCall(QPGetQPC(qp, &qpc));

  /* general linear inequality constraints Bx <= c */
  PetscCheck(!Bineq, ((PetscObject)qps)->comm, PETSC_ERR_SUP, "There is currently no QPS type implemented that can solve QP s.t. linear inequality constraints without any preprocessing. Try to use QPDualize");

  /* problem with linear equality constraints Bx = c */
  if (Beq) {
    PetscCall(QPSSetType(qps, QPSSMALXE));
  } else if (qpc) {
    /* problem without equality constraints but with box constraints */
    PetscCall(QPSSetType(qps, QPSMPGP));
  } else {
    /* without constraints */
    PetscCall(QPSSetType(qps, QPSKSP));
  }

  qps->user_type = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetDefaultTypeIfNotSpecified"
PetscErrorCode QPSSetDefaultTypeIfNotSpecified(QPS qps)
{
  PetscFunctionBegin;
  if (!((PetscObject)qps)->type_name) {
    PetscCall(QPSSetDefaultType(qps));
    qps->user_type = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetType"
/*@
   QPSGetType - return the type of solver

   Not Collective

   Input parameter:
.  qps - instance of QPS

   Output parameter:
.  type - type of the solver

   Level: developer

.seealso QPSSetType()
@*/
PetscErrorCode QPSGetType(QPS qps, const QPSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)qps)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSIsQPCompatible"
/*@
   QPSIsQPCompatible - check that given solver is able to solve given QP

   Not Collective

   Input parameters:
+  qps - instance of QPS
-  qp - QP

   Output parameter:
.  flg - return value

   Level: developer

.seealso QPSSolve(), QPSetQP()
@*/
PetscErrorCode QPSIsQPCompatible(QPS qps, QP qp, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscValidHeaderSpecific(qp, QP_CLASSID, 2);
  *flg = PETSC_FALSE;
  PetscTryTypeMethod(qps, isqpcompatible, qp, flg);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolve"
/*@
   QPSSolve - solve the QP using QPS; initiate the solver

   Collective on QPS

   Input parameter:
.  qps - instance of QPS

   Level: beginner

.seealso QPSSetQP(), QPSSetUp(), QPIsSolved(), QPGetSolutionVector()
@*/
PetscErrorCode QPSSolve(QPS qps)
{
  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscCall(QPSSetUp(qps));

  PetscCall(PetscLogEventBegin(QPS_Solve, qps, 0, 0, 0));
  PetscUseTypeMethod(qps, solve);
  PetscCall(PetscLogEventEnd(QPS_Solve, qps, 0, 0, 0));

  qps->iterations_accumulated += qps->iteration;
  qps->nsolves++;

  qps->postsolvecalled = PETSC_FALSE;
  qps->solQP->solved   = (PetscBool)(qps->reason > 0);

  if (qps->autoPostSolve) { PetscCall(QPSPostSolve(qps)); }
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSPostSolve"
/*@
   QPSPostSolve - Apply post solve functions and optionally view.

   Collective on QP

   Input Parameter:
.  qp   - the QP

   Options Database Keys:
+  -qps_view             - view information about QPS
.  -qps_view_convergence - view information about QPS type and convergence
.  -qp_view              - view information about QP
.  -qp_chain_view        - view information about all QPs in the chain
.  -qp_chain_view_qppf   - view information about all QPPFs in the chain
-  -qp_chain_view_kkt    - view how well are KKT conditions satisfied for each QP in the chain

   Level: advanced

.seealso QPSSolve(), QPChainPostSolve(), QPSView(), QPSViewConvergence(), QPChainView(), QPChainViewKKT(), QPChainViewQPPF()
@*/
PetscErrorCode QPSPostSolve(QPS qps)
{
  QP                qp;
  PetscBool         view;
  PetscViewer       v;
  PetscViewerFormat format;

  PermonTracedFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  if (qps->postsolvecalled) PetscFunctionReturn(PETSC_SUCCESS);

  PermonTraceBegin;
  PetscCall(PetscLogEventBegin(QPS_PostSolve, qps, 0, 0, 0));
  PetscCall(PetscOptionsCreateViewer(((PetscObject)qps)->comm, NULL, ((PetscObject)qps)->prefix, "-qps_view", &v, &format, &view));
  if (view && !PetscPreLoadingOn) {
    PetscCall(PetscViewerPushFormat(v, format));
    PetscCall(QPSView(qps, v));
    PetscCall(PetscViewerPopFormat(v));
  }
  PetscCall(PetscViewerDestroy(&v));

  PetscCall(PetscOptionsCreateViewer(((PetscObject)qps)->comm, NULL, ((PetscObject)qps)->prefix, "-qps_view_convergence", &v, &format, &view));
  if (view && !PetscPreLoadingOn) {
    PetscCall(PetscViewerPushFormat(v, format));
    PetscCall(QPSViewConvergence(qps, v));
    PetscCall(PetscViewerPopFormat(v));
  }
  PetscCall(PetscViewerDestroy(&v));

  PetscCall(QPSGetQP(qps, &qp));
  PetscCall(QPChainPostSolve(qp));
  qps->postsolvecalled = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(QPS_PostSolve, qps, 0, 0, 0));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetConvergenceTest"
PetscErrorCode QPSSetConvergenceTest(QPS qps, PetscErrorCode (*converge)(QPS, KSPConvergedReason *), void *cctx, PetscErrorCode (*destroy)(void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  if (qps->convergencetestdestroy) { PetscCall((*qps->convergencetestdestroy)(qps->cnvctx)); }
  qps->convergencetest        = converge;
  qps->convergencetestdestroy = destroy;
  qps->cnvctx                 = cctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetConvergenceContext"
PetscErrorCode QPSGetConvergenceContext(QPS qps, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  *ctx = qps->cnvctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConvergedDefault"
/*@C
   QPSConvergedDefault - Determines convergence of the QPS iterative solvers (default code).

   Collective on QPS

   Input Parameters:
.  qps   - iterative context

   Reason is set to:
+  positive - if the iteration has converged;
.  negative - if residual norm exceeds divergence threshold;
-  0 - otherwise.

   Notes:
   QPSConvergedDefault() reaches convergence when
$      rnorm < MAX (rtol * rnorm_0, abstol);
   Divergence is detected if
$      rnorm > dtol * rnorm_0,
   where
+  rtol - relative tolerance,
.  abstol - absolute tolerance,
.  dtol - divergence tolerance,
-  rnorm_0 - the 2-norm of the right hand side.
   Use QPSSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.keywords: QPS, default, convergence, residual

.seealso: QPSSetConvergenceTest(), QPSSetTolerances(), QPSConvergedSkip(), KSPConvergedReason, QPSGetConvergedReason(),
          QPSConvergedDefaultCreate(), QPSConvergedDefaultDestroy()
@*/
PetscErrorCode QPSConvergedDefault(QPS qps, KSPConvergedReason *reason)
{
  QPSConvergedDefaultCtx *cctx  = (QPSConvergedDefaultCtx *)qps->cnvctx;
  PetscInt                i     = qps->iteration;
  PetscReal               rnorm = qps->rnorm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  *reason = KSP_CONVERGED_ITERATING;

  PetscCheck(cctx, ((PetscObject)qps)->comm, PETSC_ERR_ARG_NULL, "Convergence context must have been created with QPSConvergedDefaultCreate()");
  if (!cctx->setup_called) { PetscCall(QPSConvergedDefaultSetUp(qps)); }

  if (i > qps->max_it) {
    *reason = KSP_DIVERGED_ITS;
    PetscCall(PetscInfo(qps, "QP solver is diverging (iteration count reached the maximum). Initial right hand size norm %14.12e, current residual norm %14.12e at iteration %" PetscInt_FMT "\n", (double)cctx->norm_rhs, (double)rnorm, i));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (i != -1) PetscCall(PermonDebug2("iteration %5d  rnorm %.10e \n", i, rnorm));

  if (PetscIsInfOrNanScalar(rnorm)) {
    PetscCall(PetscInfo(qps, "QP solver has created a not a number (NaN) as the residual norm, declaring divergence \n"));
    *reason = KSP_DIVERGED_NANORINF;
  } else if (rnorm <= cctx->ttol) {
    if (rnorm < qps->atol) {
      PetscCall(PetscInfo(qps, "QP solver has converged. Residual norm %14.12e is less than absolute tolerance %14.12e at iteration %" PetscInt_FMT "\n", (double)rnorm, (double)qps->atol, i));
      *reason = KSP_CONVERGED_ATOL;
    } else {
      PetscCall(
        PetscInfo(qps, "QP solver has converged. Residual norm %14.12e is less than rtol*||b|| =  %14.12e * %14.12e = %14.12e at iteration %" PetscInt_FMT "\n", (double)rnorm, (double)qps->rtol, (double)cctx->norm_rhs, (double)qps->rtol * cctx->norm_rhs, i));
      *reason = KSP_CONVERGED_RTOL;
    }
  } else if (rnorm >= qps->divtol * cctx->norm_rhs_div) {
    PetscCall(PetscInfo(qps, "QP solver is diverging. Residual norm %14.12e exceeded the divergence tolerance divtol * ||b|| = %14.12e * %14.12e = %14.12e at iteration %" PetscInt_FMT "\n", (double)rnorm, (double)qps->divtol, (double)cctx->norm_rhs,
                        (double)qps->divtol * cctx->norm_rhs_div, i));
    *reason = KSP_DIVERGED_DTOL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConvergedDefaultSetUp"
PetscErrorCode QPSConvergedDefaultSetUp(QPS qps)
{
  QPSConvergedDefaultCtx *cctx = (QPSConvergedDefaultCtx *)qps->cnvctx;

  PetscFunctionBegin;
  if (cctx->setup_called) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(qps->setupcalled, ((PetscObject)qps)->comm, PETSC_ERR_ARG_WRONGSTATE, "QPSSetUp() not yet called");
  PetscCall(VecNorm(qps->solQP->b, NORM_2, &cctx->norm_rhs));
  cctx->ttol         = PetscMax(qps->rtol * cctx->norm_rhs, qps->atol);
  cctx->norm_rhs_div = cctx->norm_rhs;
  cctx->setup_called = PETSC_TRUE;
  PetscCall(PetscInfo(qps, "QP solver convergence criterion initialized: ttol = max(rtol*norm(b),atol) = max(%.4e * %.4e, %.4e) = %.4e\n", qps->rtol, cctx->norm_rhs, qps->atol, cctx->ttol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO this is just a quick&dirty solution
#undef __FUNCT__
#define __FUNCT__ "QPSConvergedDefaultSetRhsForDivergence"
PetscErrorCode QPSConvergedDefaultSetRhsForDivergence(void *ctx, Vec b)
{
  QPSConvergedDefaultCtx *cctx = (QPSConvergedDefaultCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(VecNorm(b, NORM_2, &cctx->norm_rhs_div));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConvergedDefaultDestroy"
PetscErrorCode QPSConvergedDefaultDestroy(void *ctx)
{
  QPSConvergedDefaultCtx *cctx = (QPSConvergedDefaultCtx *)ctx;

  PetscFunctionBegin;
  //PetscCall(VecDestroy(&cctx->work));
  PetscCall(PetscFree(cctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConvergedDefaultCreate"
PetscErrorCode QPSConvergedDefaultCreate(void **ctx)
{
  QPSConvergedDefaultCtx *cctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cctx));
  *ctx               = cctx;
  cctx->norm_rhs     = NAN;
  cctx->ttol         = NAN;
  cctx->setup_called = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSConvergedSkip"
PetscErrorCode QPSConvergedSkip(QPS qps, KSPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  *reason = KSP_CONVERGED_ITERATING;
  if (qps->iteration >= qps->max_it) *reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetConvergedReason"
PetscErrorCode QPSGetConvergedReason(QPS qps, KSPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(reason, 2);
  *reason = qps->reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetResidualNorm"
PetscErrorCode QPSGetResidualNorm(QPS qps, PetscReal *rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(rnorm, 2);
  *rnorm = qps->rnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetIterationNumber"
PetscErrorCode QPSGetIterationNumber(QPS qps, PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(its, 2);
  *its = qps->iteration;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetAccumulatedIterationNumber"
PetscErrorCode QPSGetAccumulatedIterationNumber(QPS qps, PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(its, 2);
  *its = qps->iterations_accumulated;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetOptionsPrefix"
PetscErrorCode QPSSetOptionsPrefix(QPS qps, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)qps, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSAppendOptionsPrefix"
PetscErrorCode QPSAppendOptionsPrefix(QPS qps, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)qps, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetOptionsPrefix"
PetscErrorCode QPSGetOptionsPrefix(QPS qps, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)qps, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetFromOptions"
PetscErrorCode QPSSetFromOptions(QPS qps)
{
  PetscBool flg;
  PetscReal rtol, atol, dtol;
  PetscInt  maxit;
  char      type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)qps);
  PetscCall(PetscOptionsFList("-qps_type", "QP solution method", "QPSSetType", QPSList, (char *)(((PetscObject)qps)->type_name), type, 256, &flg));
  if (flg) PetscCall(QPSSetType(qps, type));
  PetscCall(QPSSetDefaultTypeIfNotSpecified(qps));

  PetscCall(PetscOptionsInt("-qps_max_it", "Maximum number of iterations", "QPSSetTolerances", qps->max_it, &maxit, &flg));
  if (!flg) maxit = qps->max_it;
  PetscCall(PetscOptionsReal("-qps_rtol", "Relative decrease in residual norm", "QPSSetTolerances", qps->rtol, &rtol, &flg));
  if (!flg) rtol = qps->rtol;
  PetscCall(PetscOptionsReal("-qps_atol", "Absolute value of residual norm", "QPSSetTolerances", qps->atol, &atol, &flg));
  if (!flg) atol = qps->atol;
  PetscCall(PetscOptionsReal("-qps_divtol", "Residual norm increase cause divergence", "QPSSetTolerances", qps->divtol, &dtol, &flg));
  if (!flg) dtol = qps->divtol;
  PetscCall(QPSSetTolerances(qps, rtol, atol, dtol, maxit));
  PetscCall(PetscOptionsBool("-qps_auto_post_solve", "QPSSolve automatically triggers PostSolve", "QPSSetAutoPostSolve", qps->autoPostSolve, &qps->autoPostSolve, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-qps_monitor_cancel", "Turn off all QPS monitors", "QPSMonitorCancel", flg, &flg, NULL));
  if (flg) PetscCall(QPSMonitorCancel(qps));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-qps_monitor", "Turn on default QPS monitor", "QPSMonitorSet", flg, &flg, NULL));
  if (flg) PetscCall(QPSMonitorSet(qps, QPSMonitorDefault, NULL, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-qps_monitor_cost", "Switches QPS monitor", "QPSMonitorSet", flg, &flg, NULL));
  if (flg) PetscCall(QPSMonitorSet(qps, QPSMonitorCostFunction, NULL, NULL));
  /* actually checked in setup - this is just here to go into help message */
  PetscCall(PetscOptionsName("-qps_view", "print the QPS parameters at the end of a QPSSolve call", "QPSView", &flg));
  PetscCall(PetscOptionsName("-qps_view_convergence", "print the QPS convergence info at the end of a QPSSolve call", "QPSViewConvergence", &flg));
  PetscTryTypeMethod(qps, setfromoptions, PetscOptionsObject);
  if (qps->topQP) PetscCall(QPChainSetFromOptions(qps->topQP));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetTolerances"
PetscErrorCode QPSSetTolerances(QPS qps, PetscReal rtol, PetscReal atol, PetscReal divtol, PetscInt max_it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(qps, rtol, 2);
  PetscValidLogicalCollectiveReal(qps, atol, 3);
  PetscValidLogicalCollectiveReal(qps, divtol, 4);
  PetscValidLogicalCollectiveInt(qps, max_it, 5);

  if (rtol != PETSC_DEFAULT) {
    PetscCheck(rtol >= 0.0 && 1.0 > rtol, ((PetscObject)qps)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Relative tolerance %g must be non-negative and less than 1.0", rtol);
    qps->rtol = rtol;
  }
  if (atol != PETSC_DEFAULT) {
    PetscCheck(atol >= 0.0, ((PetscObject)qps)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Absolute tolerance %g must be non-negative", atol);
    qps->atol = atol;
  }
  if (divtol != PETSC_DEFAULT) {
    PetscCheck(divtol >= 0.0, ((PetscObject)qps)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Divergence tolerance %g must be larger than 1.0", divtol);
    qps->divtol = divtol;
  }
  if (max_it != PETSC_DEFAULT) {
    PetscCheck(max_it >= 0, ((PetscObject)qps)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Maximum number of iterations %" PetscInt_FMT " must be non-negative", max_it);
    qps->max_it = max_it;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetTolerances"
PetscErrorCode QPSGetTolerances(QPS qps, PetscReal *rtol, PetscReal *atol, PetscReal *divtol, PetscInt *max_it)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  if (atol) *atol = qps->atol;
  if (rtol) *rtol = qps->rtol;
  if (divtol) *divtol = qps->divtol;
  if (max_it) *max_it = qps->max_it;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetAutoPostSolve"
PetscErrorCode QPSSetAutoPostSolve(QPS qps, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscValidLogicalCollectiveBool(qps, flg, 2);
  qps->autoPostSolve = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetAutoPostSolve"
PetscErrorCode QPSGetAutoPostSolve(QPS qps, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qps, QPS_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  *flg = qps->autoPostSolve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSViewConvergence"
PetscErrorCode QPSViewConvergence(QPS qps, PetscViewer v)
{
  MPI_Comm           comm;
  PetscReal          rnorm, rtol, abstol, dtol;
  PetscInt           its, maxits;
  QP                 topqp;
  const QPSType      qpstype;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)qps, &comm));
  PetscCall(QPSGetQP(qps, &topqp));

  PetscCall(QPSGetConvergedReason(qps, &reason));
  PetscCall(QPSGetIterationNumber(qps, &its));
  PetscCall(QPSGetResidualNorm(qps, &rnorm));
  PetscCall(QPSGetTolerances(qps, &rtol, &abstol, &dtol, &maxits));
  PetscCall(QPSGetType(qps, &qpstype));

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)qps, v));
  PetscCall(PetscViewerASCIIPushTab(v));
  PetscCall(PetscViewerASCIIPrintf(v, "last QPSSolve %s due to %s, KSPReason=%d, required %d iterations\n", (reason > 0) ? "CONVERGED" : "DIVERGED", KSPConvergedReasons[reason], reason, its));
  PetscCall(PetscViewerASCIIPrintf(v, "all %d QPSSolves from last QPSReset/QPSResetStatistics have required %d iterations\n", qps->nsolves, qps->iterations_accumulated));
  PetscCall(PetscViewerASCIIPrintf(v, "tolerances: rtol=%.1e, abstol=%.1e, dtol=%.1e, maxits=%d\n", rtol, abstol, dtol, maxits));
  if (*qps->ops->viewconvergence) {
    PetscCall(PetscViewerASCIIPrintf(v, "%s specific:\n", qpstype));
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscUseTypeMethod(qps, viewconvergence, v);
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetVecs"
/*@C
   QPSGetVecs - Gets a number of work vectors.

   Input Parameters:
+  qps  - iterative context
.  rightn  - number of right work vectors
-  leftn   - number of left work vectors to allocate

   Output Parameters:
+  right - the array of vectors created
-  left - the array of left vectors

   Note: The right vector has as many elements as the matrix has columns. The left
     vector has as many elements as the matrix has rows.

   Level: advanced

.seealso: MatCreateVecs()
@*/
PetscErrorCode QPSGetVecs(QPS qps, PetscInt rightn, Vec **right, PetscInt leftn, Vec **left)
{
  Vec vecr, vecl;

  PetscFunctionBegin;
  if (rightn) {
    PetscCheck(right, PetscObjectComm((PetscObject)qps), PETSC_ERR_ARG_INCOMP, "You asked for right vectors but did not pass a pointer to hold them");
    PetscCall(QPGetVecs(qps->solQP, &vecr, NULL));
    PetscCall(VecDuplicateVecs(vecr, rightn, right));
    PetscCall(VecDestroy(&vecr));
  }
  if (leftn) {
    PetscCheck(left, PetscObjectComm((PetscObject)qps), PETSC_ERR_ARG_INCOMP, "You asked for left vectors but did not pass a pointer to hold them");
    PetscCall(QPGetVecs(qps->solQP, NULL, &vecl));
    PetscCall(VecDuplicateVecs(vecl, leftn, left));
    PetscCall(VecDestroy(&vecl));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetWorkVecs"
/*
   QPSSetWorkVecs - Sets a number of work vectors into a QPS object

   Input Parameters:
+  qps  - iterative context
-  nw   - number of work vectors to allocate

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin QPS implementations

   Level: developer
*/
PetscErrorCode QPSSetWorkVecs(QPS qps, PetscInt nw)
{
  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(qps->nwork, &qps->work));
  PetscCall(PetscFree(qps->work_state));

  qps->nwork = nw;
  PetscCall(QPSGetVecs(qps, nw, &qps->work, 0, NULL));
  PetscCall(PetscMalloc1(nw, &qps->work_state));
  PetscCall(PetscMemzero(qps->work_state, nw * sizeof(PetscObjectState)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSWorkVecStateUpdate"
PetscErrorCode QPSWorkVecStateUpdate(QPS qps, PetscInt idx)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectStateGet((PetscObject)qps->work[idx], &qps->work_state[idx]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolutionVecStateUpdate"
PetscErrorCode QPSSolutionVecStateUpdate(QPS qps)
{
  PetscFunctionBegin;
  PERMON_ASSERT(qps->solQP, "qps->solQP initialized");
  PetscCall(PetscObjectStateGet((PetscObject)qps->solQP->x, &qps->xstate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSWorkVecStateChanged"
PetscErrorCode QPSWorkVecStateChanged(QPS qps, PetscInt idx, PetscBool *flg)
{
  PetscObjectState state_saved, state_current;

  PetscFunctionBegin;
  state_saved = qps->work_state[idx];
  PetscCall(PetscObjectStateGet((PetscObject)qps->work[idx], &state_current));
  *flg = (PetscBool)(state_current != state_saved);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolutionVecStateChanged"
PetscErrorCode QPSSolutionVecStateChanged(QPS qps, PetscBool *flg)
{
  PetscObjectState state_saved, state_current;

  PetscFunctionBegin;
  PERMON_ASSERT(qps->solQP, "qps->solQP initialized");
  state_saved = qps->xstate;
  PetscCall(PetscObjectStateGet((PetscObject)qps->solQP->x, &state_current));
  *flg = (PetscBool)(state_current != state_saved);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMonitor"
/*@
   QPSMonitor - runs the user provided monitor routines, if they exist

   Collective on QPS

   Input Parameters:
+  qps - iterative context obtained from QPSCreate()
.  it - iteration number
-  rnorm - relative norm of the residual

   Notes:
   This routine is called by the QPS implementations.
   It does not typically need to be called by the user.

   Level: developer

.seealso: QPSMonitorSet()
@*/
PetscErrorCode QPSMonitor(QPS qps, PetscInt it, PetscReal rnorm)
{
  PetscInt i, n = qps->numbermonitors;

  PetscFunctionBegin;
  for (i = 0; i < n; i++) { (*qps->monitor[i])(qps, it, rnorm, qps->monitorcontext[i]); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMonitorSet"
/*@
   QPSMonitorSet - Sets an ADDITIONAL function to be called at every iteration to monitor the residual/error etc.

   Logically Collective on QPS

   Input Parameters:
+  qps - iterative context obtained from QPSCreate()
.  monitor - pointer to function
.  mctx    - [optional] context for private data for the monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context (may be NULL)

   Calling Sequence of monitor:
$     monitor (QPS qps, PetscInt it, PetscReal rnorm, void *mctx)

+  qps - iterative context obtained from QPSCreate()
.  it - iteration number
.  rnorm - 2-norm of the residual (more general, a vector whose norm determines convergence, such as a projected gradient for QPSMPGP)
-  mctx  - optional monitoring context, as set by QPSMonitorSet()

   Options Database Keys:
+  -qps_monitor - sets QPSMonitorDefault()
.  -qps_monitor_cost    - sets QPSMonitorCostFunction()
-  -qps_monitor_cancel - cancels all monitors that have been hardwired into a code by
    calls to QPSMonitorSet(), but does not cancel those set via the options database.

   Notes:
   The default is to do nothing. To print the residual, use
   QPSMonitorDefault() as the monitoring routine, with a null monitoring context.

   Several different monitoring routines may be set by calling
   QPSMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Level: beginner

.seealso: QPSMonitorDefault(), QPSMonitorCancel()
@*/
PetscErrorCode QPSMonitorSet(QPS qps, PetscErrorCode (*monitor)(QPS, PetscInt, PetscReal, void *), void *mctx, PetscCtxDestroyFn *monitordestroy)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCheck(monitor, PetscObjectComm((PetscObject)qps), PETSC_ERR_ARG_NULL, "Monitor function must be specified");

  /* verify the number of monitors */
  PetscCheck(qps->numbermonitors < MAXQPSMONITORS, PetscObjectComm((PetscObject)qps), PETSC_ERR_ARG_OUTOFRANGE, "Too many QPS monitors set");

  /* don't add exactly the same monitor twice */
  //TODO we could use PetscMonitorCompare() once it gets fixed
  for (i = 0; i < qps->numbermonitors; i++) {
    if (monitor == qps->monitor[i] && monitordestroy == qps->monitordestroy[i] && mctx == qps->monitorcontext[i]) { PetscFunctionReturn(PETSC_SUCCESS); }
  }

  /* set new QPS monitor */
  qps->monitor[qps->numbermonitors]        = monitor;
  qps->monitordestroy[qps->numbermonitors] = monitordestroy;
  qps->monitorcontext[qps->numbermonitors] = (void *)mctx;
  qps->numbermonitors++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMonitorCancel"
/*@
   QPSMonitorCancel - Clears all monitors for a QPS object.

   Logically Collective on QPS

   Input Parameters:
.  qps - iterative context obtained from QPSCreate()

   Options Database Key:
.  -qps_monitor_cancel - cancels all monitors that have been hardwired into a code by
    calls to QPSMonitorSet(), but does not cancel those set via the options database.

   Level: intermediate

.seealso: QPSMonitorDefault(), QPSMonitorSet()
@*/
PetscErrorCode QPSMonitorCancel(QPS qps)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < qps->numbermonitors; i++) {
    if (qps->monitordestroy[i]) { (*qps->monitordestroy[i])(&qps->monitorcontext[i]); }
  }
  qps->numbermonitors = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetMonitorContext"
/*@
   QPSGetMonitorContext - Gets the monitoring context, as set by
     QPSMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  qps - iterative context obtained from QPSCreate()

   Output Parameter:
.  qps - monitoring context

   Level: intermediate

.keywords: QPS, get, monitor, context

.seealso: QPSMonitorDefault()
@*/
PetscErrorCode QPSGetMonitorContext(QPS qps, void **ctx)
{
  PetscFunctionBegin;
  *ctx = (qps->monitorcontext[0]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSetResidualHistory"
/*@
   QPSSetResidualHistory - Sets the array used to hold the residual history.
     If set, this array will contain the residual norms computed at each
     iteration of the solver.

   Not Collective

   Input Parameters:
+  qps - iterative context obtained from QPSCreate()
.  a   - array to hold history
.  na  - size of a
-  reset - PETSC_TRUE indicates the history counter is reset to zero
           for each new linear solve

   Level: advanced

   Notes:
   Array is NOT freed by PETSc so the user needs to keep track of
    and destroy once the QPS object is destroyed.

    'a' is NULL then space is allocated for the history.
    If 'na' PETSC_DECIDE or PETSC_DEFAULT then an array of length 10000 is allocated.

.keywords: QPS, set, residual, history, norm

.seealso: QPSGetResidualHistory()
@*/
PetscErrorCode QPSSetResidualHistory(QPS qps, PetscReal a[], PetscInt na, PetscBool reset)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(qps->res_hist_alloc));
  if (na != PETSC_DECIDE && na != PETSC_DEFAULT && a) {
    qps->res_hist     = a;
    qps->res_hist_max = na;
  } else {
    if (na != PETSC_DECIDE && na != PETSC_DEFAULT) qps->res_hist_max = na;
    else qps->res_hist_max = 10000; /* like default ksp->max_it */
    PetscCall(PetscMalloc(qps->res_hist_max, &qps->res_hist_alloc));

    qps->res_hist = qps->res_hist_alloc;
  }
  qps->res_hist_len   = 0;
  qps->res_hist_reset = reset;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSGetResidualHistory"
/*@
   QPSGetResidualHistory - Gets the array used to hold the residual history
       and the number of residuals it contains.

   Not Collective

   Input Parameter:
.  qps - iterative context obtained from QPSCreate()

   Output Parameters:
+  a   - pointer to array to hold history (or NULL)
-  na  - number of used entries in a (or NULL)

   Level: advanced

   Notes:
     Can only be called after a QPSSetResidualHistory() otherwise a and na are set to zero

.keywords: QPS, get, residual, history, norm

.seealso: QPSGetResidualHistory()
@*/
PetscErrorCode QPSGetResidualHistory(QPS qps, PetscReal *a[], PetscInt *na)
{
  PetscFunctionBegin;
  if (a) *a = qps->res_hist;
  if (na) *na = qps->res_hist_len;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMonitorDefault"
/*@
   QPSMonitorDefault - Print the projected gradient norm at each iteration of an
                       iterative solver.

   Collective on QPS

   Input Parameters:
+  qps   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated).
-  ctx   - the PetscViewer

   Level: intermediate

.keywords: QPS, default, monitor, residual

.seealso: QPSMonitorSet(), QPSMonitorCostFunction(), QPSMonitorCancel()
@*/
PetscErrorCode QPSMonitorDefault(QPS qps, PetscInt n, PetscReal rnorm, void *ctx)
{
  PetscViewer viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  if (!viewer) { PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)qps), &viewer)); }
  PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)qps)->tablevel));

  if (qps->ops->monitor) {
    /* this algorithm has own monitor */
    PetscUseTypeMethod(qps, monitor, n, viewer);
  } else {
    /* use default QPS monitor */
    if (n == 0 && ((PetscObject)qps)->prefix) { PetscCall(PetscViewerASCIIPrintf(viewer, "  Projected gradient norms for %s solve.\n", ((PetscObject)qps)->prefix)); }
    PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " QPS Projected gradient norm %14.12e \n", n, (double)rnorm));
    PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)qps)->tablevel));
  }

  PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)qps)->tablevel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSMonitorCostFunction"
/*@
   QPSMonitorDefault - Print the value of the cost function at each iteration of an
                       iterative solver.

   Collective on QPS

   Input Parameters:
+  qps   - iterative context
.  n     - iteration number
.  rnorm - unused
-  dummy - unused monitor context

   Level: intermediate

.keywords: QPS, monitor, cost function

.seealso: QPSMonitorSet(), QPSMonitorDefault()
@*/
PetscErrorCode QPSMonitorCostFunction(QPS qps, PetscInt n, PetscReal rnorm, void *dummy)
{
  PetscViewer viewer = (PetscViewer)dummy;
  QP          qp;
  Vec         x;
  PetscReal   f;

  PetscFunctionBegin;
  if (!viewer) { PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)qps), &viewer)); }
  PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)qps)->tablevel));

  if (qps->ops->monitorcostfunction) {
    /* this algorithm has own monitor */
    PetscUseTypeMethod(qps, monitorcostfunction, n, viewer);
  } else {
    /* use default QPS monitor */
    if (n == 0 && ((PetscObject)qps)->prefix) { PetscCall(PetscViewerASCIIPrintf(viewer, "  Cost function values for %s solve.\n", ((PetscObject)qps)->prefix)); }
    PetscCall(QPSGetSolvedQP(qps, &qp));
    PetscCall(QPGetSolutionVector(qp, &x));
    PetscCall(QPComputeObjective(qp, x, &f));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " QPS Cost function value %14.12e \n", n, (double)f));
    PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)qps)->tablevel));
  }

  PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)qps)->tablevel));
  PetscFunctionReturn(PETSC_SUCCESS);
}
