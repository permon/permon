#include <permon/private/qpimpl.h>

PetscClassId QP_CLASSID;

const char *QPScaleTypes[] = {"none", "norm2", "multiplicity", "QPScalType", "QPPF_", 0};

static PetscErrorCode QPSetFromOptions_Private(QP qp);

#define QPView_PrintObjectLoaded(v, obj, description) PetscViewerASCIIPrintf(v, "    %-32s %-16s %c\n", description, obj ? ((PetscObject)obj)->name : "", obj ? 'Y' : 'N')
#define QPView_Vec(v, x, iname) \
  { \
    PetscReal   max, min, norm; \
    PetscInt    imax, imin; \
    const char *name = (iname); \
    PetscCall(VecNorm(x, NORM_2, &norm)); \
    PetscCall(VecMax(x, &imax, &max)); \
    PetscCall(VecMin(x, &imin, &min)); \
    PetscCall(PetscViewerASCIIPrintf(v, "||%2s|| = %.8e    max(%2s) = %.2e = %2s(%d)    min(%2s) = %.2e = %2s(%d)    %p\n", name, norm, name, max, name, imax, name, min, name, imin, (void *)x)); \
  }

#undef __FUNCT__
#define __FUNCT__ "QPInitializeInitialVector_Private"
static PetscErrorCode QPInitializeInitialVector_Private(QP qp)
{
  Vec xp, xc;

  PetscFunctionBegin;
  if (qp->x) PetscFunctionReturn(PETSC_SUCCESS);
  if (!qp->parent) {
    /* if no initial guess exists, just set it to a zero vector */
    PetscCall(MatCreateVecs(qp->A, &qp->x, NULL));
    PetscCall(VecZeroEntries(qp->x)); // TODO: is it in the feasible set?
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(QPGetSolutionVector(qp->parent, &xp));
  if (xp) {
    PetscCall(VecDuplicate(xp, &xc));
    PetscCall(VecCopy(xp, xc));
    PetscCall(QPSetInitialVector(qp, xc));
    PetscCall(VecDestroy(&xc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPAddChild"
PetscErrorCode QPAddChild(QP qp, QPDuplicateOption opt, QP *newchild)
{
  QP child;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(newchild, 2);
  PetscCall(QPDuplicate(qp, opt, &child));
  qp->child     = child;
  child->parent = qp;
  child->id     = qp->id + 1;
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  if (newchild) *newchild = child;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPRemoveChild"
PetscErrorCode QPRemoveChild(QP qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (!qp->child) PetscFunctionReturn(PETSC_SUCCESS);
  qp->child->parent    = NULL;
  qp->child->postSolve = NULL;
  PetscCall(QPDestroy(&qp->child));
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCreate"
/*@
   QPCreate - Creates a quadratic programming problem (QP) object.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI comm

   Output Parameter:
-  qp_new - the new QP

   Level: beginner

.seealso: QPDestroy()
@*/
PetscErrorCode QPCreate(MPI_Comm comm, QP *qp_new)
{
  QP qp;

  PetscFunctionBegin;
  PetscAssertPointer(qp_new, 2);
  *qp_new = 0;
  PetscCall(QPInitializePackage());

  PetscCall(PetscHeaderCreate(qp, QP_CLASSID, "QP", "Quadratic Programming Problem", "QP", comm, QPDestroy, QPView));
  PetscCall(PetscObjectChangeTypeName((PetscObject)qp, "QP"));
  qp->A                    = NULL;
  qp->R                    = NULL;
  qp->b                    = NULL;
  qp->b_plus               = PETSC_FALSE;
  qp->BE                   = NULL;
  qp->BE_nest_count        = 0;
  qp->cE                   = NULL;
  qp->lambda_E             = NULL;
  qp->Bt_lambda            = NULL;
  qp->BI                   = NULL;
  qp->cI                   = NULL;
  qp->lambda_I             = NULL;
  qp->B                    = NULL;
  qp->c                    = NULL;
  qp->lambda               = NULL;
  qp->x                    = NULL;
  qp->xwork                = NULL;
  qp->pc                   = NULL;
  qp->pf                   = NULL;
  qp->child                = NULL;
  qp->parent               = NULL;
  qp->solved               = PETSC_FALSE;
  qp->setupcalled          = PETSC_FALSE;
  qp->setfromoptionscalled = 0;

  /* set the initial constraints */
  qp->qpc = NULL;

  qp->changeListener      = NULL;
  qp->changeListenerCtx   = NULL;
  qp->postSolve           = NULL;
  qp->postSolveCtx        = NULL;
  qp->postSolveCtxDestroy = NULL;

  qp->id                = 0;
  qp->transform         = NULL;
  qp->transform_name[0] = 0;

  /* initialize preconditioner */
  PetscCall(QPGetPC(qp, &qp->pc));

  *qp_new = qp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPDuplicate"
/*@
   QPDuplicate - Duplicate QP object.

   Collective on QP

   Input Parameters:
+  qp1 - QP to duplicate
-  opt - either QP_DUPLICATE_DO_NOT_COPY or QP_DUPLICATE_COPY_POINTERS

   Output Parameter:
.  qp2 - duplicated QP

   Level: advanced

.seealso: QPCreate()
@*/
PetscErrorCode QPDuplicate(QP qp1, QPDuplicateOption opt, QP *qp2)
{
  QP qp2_;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp1, QP_CLASSID, 1);
  PetscAssertPointer(qp2, 2);
  PetscCall(QPCreate(PetscObjectComm((PetscObject)qp1), &qp2_));

  if (opt == QP_DUPLICATE_DO_NOT_COPY) {
    *qp2 = qp2_;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(QPSetQPC(qp2_, qp1->qpc));
  PetscCall(QPSetEq(qp2_, qp1->BE, qp1->cE));
  PetscCall(QPSetEqMultiplier(qp2_, qp1->lambda_E));
  qp2_->BE_nest_count = qp1->BE_nest_count;
  PetscCall(QPSetIneq(qp2_, qp1->BI, qp1->cI));
  PetscCall(QPSetIneqMultiplier(qp2_, qp1->lambda_I));
  PetscCall(QPSetInitialVector(qp2_, qp1->x));
  PetscCall(QPSetOperator(qp2_, qp1->A));
  PetscCall(QPSetOperatorNullSpace(qp2_, qp1->R));
  if (qp1->pc) PetscCall(QPSetPC(qp2_, qp1->pc));
  PetscCall(QPSetQPPF(qp2_, qp1->pf));
  PetscCall(QPSetRhs(qp2_, qp1->b));
  PetscCall(QPSetWorkVector(qp2_, qp1->xwork));

  if (qp1->lambda) PetscCall(PetscObjectReference((PetscObject)(qp2_->lambda = qp1->lambda)));
  if (qp1->Bt_lambda) PetscCall(PetscObjectReference((PetscObject)(qp2_->Bt_lambda = qp1->Bt_lambda)));
  *qp2 = qp2_;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCompareEqMultiplierWithLeastSquare"
PetscErrorCode QPCompareEqMultiplierWithLeastSquare(QP qp, PetscReal *norm)
{
  Vec BEt_lambda = NULL;
  Vec BEt_lambda_LS;
  QP  qp2;

  PetscFunctionBegin;
  if (!qp->BE) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(QPCompute_BEt_lambda(qp, &BEt_lambda));

  PetscCall(QPDuplicate(qp, QP_DUPLICATE_COPY_POINTERS, &qp2));
  PetscCall(QPSetEqMultiplier(qp2, NULL));
  PetscCall(QPComputeMissingEqMultiplier(qp2));
  PetscCall(QPCompute_BEt_lambda(qp2, &BEt_lambda_LS));

  /* compare lambda_E with least-square lambda_E */
  PetscCall(VecAXPY(BEt_lambda_LS, -1.0, BEt_lambda));
  PetscCall(VecNorm(BEt_lambda_LS, NORM_2, norm));

  PetscCall(VecDestroy(&BEt_lambda));
  PetscCall(VecDestroy(&BEt_lambda_LS));
  PetscCall(QPDestroy(&qp2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPViewKKT"
/*@
   QPViewKKT - Print how well are KKT conditions satisfied with the computed minimizer and Lagrange multipliers.

   Collective on QP

   Input Parameters:
+  qp - the QP
-  v - visualization context

   Level: intermediate

.seealso: QPChainViewKKT(), QPSSolve()
@*/
PetscErrorCode QPViewKKT(QP qp, PetscViewer v)
{
  PetscReal normb = 0.0, norm = 0.0, dot = 0.0;
  Vec       x, b, cE, cI, r, o, t;
  Mat       A, BE, BI;
  PetscBool flg = PETSC_FALSE, compare_lambda_E = PETSC_FALSE, notavail;
  MPI_Comm  comm;
  char     *kkt_name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)qp, &comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(qp, 1, v, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &flg));
  PetscCheck(flg, comm, PETSC_ERR_SUP, "Viewer type %s not supported for QP", ((PetscObject)v)->type_name);

  PetscCall(PetscOptionsGetBool(((PetscObject)qp)->options, NULL, "-qp_view_kkt_compare_lambda_E", &compare_lambda_E, NULL));

  PetscCall(PetscObjectName((PetscObject)qp));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)qp, v));
  PetscCall(PetscViewerASCIIPrintf(v, "  #%d in chain, derived by %s\n", qp->id, qp->transform_name));
  if (!qp->solved) { PetscCall(PetscViewerASCIIPrintf(v, "*** WARNING: QP is not solved. ***\n")); }

  PetscCall(QPGetOperator(qp, &A));
  PetscCall(QPGetRhs(qp, &b));
  PetscCall(QPGetEq(qp, &BE, &cE));
  PetscCall(QPGetIneq(qp, &BI, &cI));
  PetscCall(QPGetSolutionVector(qp, &x));
  PetscCall(VecNorm(b, NORM_2, &normb));

  QPView_Vec(v, x, "x");
  QPView_Vec(v, b, "b");
  if (cE) QPView_Vec(v, cE, "cE");
  if (BE && !cE) PetscCall(PetscViewerASCIIPrintf(v, "||cE|| = 0.00e-00    max(cE) = 0.00e-00 = cE(0)    min(cE) = 0.00e-00 = cE(0)\n"));
  if (cI) QPView_Vec(v, cI, "cI");
  if (BI && !cI) PetscCall(PetscViewerASCIIPrintf(v, "||cI|| = 0.00e-00    max(cI) = 0.00e-00 = cI(0)    min(cI) = 0.00e-00 = cI(0)\n"));

  PetscCall(VecDuplicate(b, &r));
  PetscCall(QPComputeLagrangianGradient(qp, x, r, &kkt_name));
  PetscCall(VecIsInvalidated(r, &notavail));

  if (!notavail) {
    if (compare_lambda_E) {
      PetscCall(QPCompareEqMultiplierWithLeastSquare(qp, &norm));
      PetscCall(PetscViewerASCIIPrintf(v, "||BE'*lambda_E - BE'*lambda_E_LS|| = %.4e\n", norm));
    }
    PetscCall(VecNorm(r, NORM_2, &norm));
    PetscCall(PetscViewerASCIIPrintf(v, "r = ||%s|| = %.2e    rO/||b|| = %.2e\n", kkt_name, norm, norm / normb));
  } else {
    PetscCall(PetscViewerASCIIPrintf(v, "r = ||%s|| not available\n", kkt_name));
  }
  PetscCall(VecDestroy(&r));
  PetscCall(PetscFree(kkt_name));

  if (BE) {
    if (BE->ops->mult) {
      PetscCall(MatCreateVecs(BE, NULL, &r));
      PetscCall(MatMult(BE, x, r));
      if (cE) PetscCall(VecAXPY(r, -1.0, cE));
      PetscCall(VecNorm(r, NORM_2, &norm));
      if (cE) {
        PetscCall(PetscViewerASCIIPrintf(v, "r = ||BE*x-cE||          = %.2e    r/||b|| = %.2e\n", norm, norm / normb));
      } else {
        PetscCall(PetscViewerASCIIPrintf(v, "r = ||BE*x||             = %.2e    r/||b|| = %.2e\n", norm, norm / normb));
      }
      PetscCall(VecDestroy(&r));
    } else {
      if (cE) {
        PetscCall(PetscViewerASCIIPrintf(v, "r = ||BE*x-cE||         not available\n"));
      } else {
        Vec t = qp->xwork;
        PetscCall(QPPFApplyGtG(qp->pf, x, t)); /* BEtBEx = BE'*BE*x */
        PetscCall(VecDot(x, t, &norm));        /* norm = x'*BE'*BE*x */
        norm = PetscSqrtReal(norm);
        PetscCall(PetscViewerASCIIPrintf(v, "r = ||BE*x||             = %.2e    r/||b|| = %.2e\n", norm, norm / normb));
      }
    }
  }

  if (BI) {
    PetscCall(VecDuplicate(qp->lambda_I, &r));
    PetscCall(VecDuplicate(r, &o));
    PetscCall(VecDuplicate(r, &t));

    PetscCall(VecSet(o, 0.0)); /* o = zeros(size(r)) */

    /* r = BI*x - cI */
    PetscCall(MatMult(BI, x, r));            /* r = BI*x         */
    if (cI) PetscCall(VecAXPY(r, -1.0, cI)); /* r = r - cI       */

    /* rI = norm(max(BI*x-cI,0)) */
    PetscCall(VecPointwiseMax(t, r, o));  /* t = max(r,o)     */
    PetscCall(VecNorm(t, NORM_2, &norm)); /* norm = norm(t)     */
    if (cI) {
      PetscCall(PetscViewerASCIIPrintf(v, "r = ||max(BI*x-cI,0)||   = %.2e    r/||b|| = %.2e\n", norm, norm / normb));
    } else {
      PetscCall(PetscViewerASCIIPrintf(v, "r = ||max(BI*x,0)||      = %.2e    r/||b|| = %.2e\n", norm, norm / normb));
    }

    /* lambda >= o  =>  examine min(lambda,o) */
    PetscCall(VecSet(o, 0.0)); /* o = zeros(size(r)) */
    PetscCall(VecPointwiseMin(t, qp->lambda_I, o));
    PetscCall(VecNorm(t, NORM_2, &norm)); /* norm = ||min(lambda,o)|| */
    PetscCall(PetscViewerASCIIPrintf(v, "r = ||min(lambda_I,0)||  = %.2e    r/||b|| = %.2e\n", norm, norm / normb));

    /* lambda'*(BI*x-cI) = 0 */
    PetscCall(VecDot(qp->lambda_I, r, &dot));
    dot = PetscAbs(dot);
    if (cI) {
      PetscCall(PetscViewerASCIIPrintf(v, "r = |lambda_I'*(BI*x-cI)|= %.2e    r/||b|| = %.2e\n", dot, dot / normb));
    } else {
      PetscCall(PetscViewerASCIIPrintf(v, "r = |lambda_I'*(BI*x)|= %.2e       r/||b|| = %.2e\n", dot, dot / normb));
    }

    PetscCall(VecDestroy(&o));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&t));
  }

  if (qp->qpc) PetscCall(QPCViewKKT(qp->qpc, x, normb, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPView"
/*@
   QPView - Print information about the QP.

   Collective on QP

   Input Parameters:
+  qp - the QP
-  v - visualization context

  Level: beginner

.seealso QPChainView()
@*/
PetscErrorCode QPView(QP qp, PetscViewer v)
{
  Vec       b, cE, cI, lb, ub;
  Mat       A, R, BE, BI;
  QPC       qpc;
  PetscBool iascii;
  MPI_Comm  comm;
  QP        childDual;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)qp, &comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(qp, 1, v, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &iascii));
  PetscCheck(iascii, comm, PETSC_ERR_SUP, "Viewer type %s not supported for QP", ((PetscObject)v)->type_name);
  PetscCall(PetscObjectName((PetscObject)qp));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)qp, v));
  PetscCall(PetscViewerASCIIPrintf(v, "#%d in chain, derived by %s\n", qp->id, qp->transform_name));

  PetscCall(QPGetOperator(qp, &A));
  PetscCall(QPGetOperatorNullSpace(qp, &R));
  PetscCall(QPGetRhs(qp, &b));
  PetscCall(QPGetBox(qp, NULL, &lb, &ub));
  PetscCall(QPGetEq(qp, &BE, &cE));
  PetscCall(QPGetIneq(qp, &BI, &cI));
  PetscCall(QPGetQPC(qp, &qpc));
  PetscCall(QPChainFind(qp, (PetscErrorCode (*)(QP))QPTDualize, &childDual));

  PetscCall(PetscViewerASCIIPrintf(v, "  LOADED OBJECTS:\n"));
  PetscCall(PetscViewerASCIIPrintf(v, "    %-32s %-16s %s\n", "what", "name", "present"));
  PetscCall(QPView_PrintObjectLoaded(v, A, "Hessian"));
  PetscCall(QPView_PrintObjectLoaded(v, b, "linear term (right-hand-side)"));
  PetscCall(QPView_PrintObjectLoaded(v, R, "R (kernel of K)"));
  PetscCall(QPView_PrintObjectLoaded(v, lb, "lower bounds"));
  PetscCall(QPView_PrintObjectLoaded(v, ub, "upper bounds"));
  PetscCall(QPView_PrintObjectLoaded(v, BE, "linear eq. constraint matrix"));
  PetscCall(QPView_PrintObjectLoaded(v, cE, "linear eq. constraint RHS"));
  PetscCall(QPView_PrintObjectLoaded(v, BI, "linear ineq. constraint"));
  PetscCall(QPView_PrintObjectLoaded(v, cI, "linear ineq. constraint RHS"));
  PetscCall(QPView_PrintObjectLoaded(v, qpc, "QPC"));

  if (A) PetscCall(MatPrintInfo(A));
  if (b) PetscCall(VecPrintInfo(b));
  if (R) PetscCall(MatPrintInfo(R));
  if (lb) PetscCall(VecPrintInfo(lb));
  if (ub) PetscCall(VecPrintInfo(ub));
  if (BE) PetscCall(MatPrintInfo(BE));
  if (cE) PetscCall(VecPrintInfo(cE));
  if (BI) PetscCall(MatPrintInfo(BI));
  if (cI) PetscCall(VecPrintInfo(cI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPReset"
/*@
   QPReset - Resets a QP context to the setupcalled = 0 and solved = 0 state,
     and destroys the Vecs and Mats describing the data as well as PC and child QP.

   Collective on QP

   Input Parameter:
.  qp - the QP

   Level: beginner

.seealso QPCreate(), QPSetUp(), QPISSolved()
@*/
PetscErrorCode QPReset(QP qp)
{
  PetscFunctionBegin;
  PetscCall(QPDestroy(&qp->child));

  PetscCall(MatDestroy(&qp->A));
  PetscCall(MatDestroy(&qp->R));
  PetscCall(MatDestroy(&qp->BE));
  PetscCall(MatDestroy(&qp->BI));
  PetscCall(MatDestroy(&qp->B));

  PetscCall(VecDestroy(&qp->b));
  PetscCall(VecDestroy(&qp->x));
  PetscCall(VecDestroy(&qp->xwork));
  PetscCall(VecDestroy(&qp->cE));
  PetscCall(VecDestroy(&qp->lambda_E));
  PetscCall(VecDestroy(&qp->Bt_lambda));
  PetscCall(VecDestroy(&qp->cI));
  PetscCall(VecDestroy(&qp->lambda_I));
  PetscCall(VecDestroy(&qp->c));
  PetscCall(VecDestroy(&qp->lambda));

  PetscCall(PCDestroy(&qp->pc));

  PetscCall(QPCDestroy(&qp->qpc));

  PetscCall(QPPFDestroy(&qp->pf));
  qp->setupcalled = PETSC_FALSE;
  qp->solved      = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetUpInnerObjects"
PetscErrorCode QPSetUpInnerObjects(QP qp)
{
  MPI_Comm comm;
  PetscInt i;
  Mat      Bs[2];
  IS       rows[2];
  Vec      cs[2], c[2];
  Vec      lambdas[2];

  PermonTracedFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);

  PetscCall(PetscObjectGetComm((PetscObject)qp, &comm));
  PetscCheck(qp->A, comm, PETSC_ERR_ORDER, "Hessian must be set before " __FUNCT__);
  PetscCheck(qp->b, comm, PETSC_ERR_ORDER, "linear term must be set before " __FUNCT__);

  PermonTraceBegin;
  PetscCall(PetscInfo(qp, "setup inner objects for QP #%d\n", qp->id));

  if (!qp->pc) PetscCall(QPGetPC(qp, &qp->pc));
  PetscCall(PCSetOperators(qp->pc, qp->A, qp->A));

  PetscCall(QPInitializeInitialVector_Private(qp));

  if (!qp->xwork) PetscCall(VecDuplicate(qp->x, &qp->xwork));

  if (qp->BE && !qp->lambda_E) {
    PetscCall(MatCreateVecs(qp->BE, NULL, &qp->lambda_E));
    PetscCall(VecInvalidate(qp->lambda_E));
  }
  if (!qp->BE) { PetscCall(VecDestroy(&qp->lambda_E)); }

  if (qp->BI && !qp->lambda_I) {
    PetscCall(MatCreateVecs(qp->BI, NULL, &qp->lambda_I));
    PetscCall(VecInvalidate(qp->lambda_I));
  }
  if (!qp->BI) { PetscCall(VecDestroy(&qp->lambda_I)); }

  if ((qp->BE || qp->BI) && !qp->B) {
    PetscCall(VecDestroy(&qp->c));

    if (qp->BE && !qp->BI) {
      PetscCall(PetscObjectReference((PetscObject)(qp->B = qp->BE)));
      PetscCall(PetscObjectReference((PetscObject)(qp->lambda = qp->lambda_E)));
      PetscCall(PetscObjectReference((PetscObject)(qp->c = qp->cE)));
    } else if (!qp->BE && qp->BI) {
      PetscCall(PetscObjectReference((PetscObject)(qp->B = qp->BI)));
      PetscCall(PetscObjectReference((PetscObject)(qp->lambda = qp->lambda_I)));
      PetscCall(PetscObjectReference((PetscObject)(qp->c = qp->cI)));
    } else {
      PetscCall(PetscObjectReference((PetscObject)(Bs[0] = qp->BE)));
      PetscCall(PetscObjectReference((PetscObject)(lambdas[0] = qp->lambda_E)));
      if (qp->cE) {
        PetscCall(PetscObjectReference((PetscObject)(cs[0] = qp->cE)));
      } else {
        PetscCall(MatCreateVecs(Bs[0], NULL, &cs[0]));
        PetscCall(VecSet(cs[0], 0.0));
      }

      PetscCall(PetscObjectReference((PetscObject)(Bs[1] = qp->BI)));
      PetscCall(PetscObjectReference((PetscObject)(lambdas[1] = qp->lambda_I)));
      if (qp->cI) {
        PetscCall(PetscObjectReference((PetscObject)(cs[1] = qp->cI)));
      } else {
        PetscCall(MatCreateVecs(Bs[1], NULL, &cs[1]));
        PetscCall(VecSet(cs[1], 0.0));
      }

      PetscCall(MatCreateNestPermon(comm, 2, NULL, 1, NULL, Bs, &qp->B));
      PetscCall(MatCreateVecs(qp->B, NULL, &qp->c));
      PetscCall(PetscObjectSetName((PetscObject)qp->B, "B"));
      PetscCall(PetscObjectSetName((PetscObject)qp->c, "c"));

      /* copy cE,cI to c */
      PetscCall(MatNestGetISs(qp->B, rows, NULL));
      for (i = 0; i < 2; i++) {
        PetscCall(VecGetSubVector(qp->c, rows[i], &c[i]));
        PetscCall(VecCopy(cs[i], c[i]));
        PetscCall(VecRestoreSubVector(qp->c, rows[i], &c[i]));
      }

      for (i = 0; i < 2; i++) {
        PetscCall(MatDestroy(&Bs[i]));
        PetscCall(VecDestroy(&cs[i]));
        PetscCall(VecDestroy(&lambdas[i]));
      }
    }
  }

  if (qp->B && !qp->lambda) {
    PetscCall(MatCreateVecs(qp->B, NULL, &qp->lambda));
    PetscCall(PetscObjectSetName((PetscObject)qp->lambda, "lambda"));
    PetscCall(VecInvalidate(qp->lambda));
  }

  if (qp->B && !qp->Bt_lambda) {
    PetscCall(MatCreateVecs(qp->B, &qp->Bt_lambda, NULL));
    PetscCall(PetscObjectSetName((PetscObject)qp->lambda, "Bt_lambda"));
    PetscCall(VecInvalidate(qp->Bt_lambda));
  }

  if (!qp->B) {
    PetscCall(VecDestroy(&qp->lambda));
    PetscCall(VecDestroy(&qp->Bt_lambda));
  }
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetUp"
/*@
   QPSetUp - Sets up the internal data structures for the QP.

   Collective on QP

   Input Parameter:
.  qp   - the QP

   Level: advanced

.seealso QPChainSetUp(), QPCreate(), QPReset(), QPSSetUp()
@*/
PetscErrorCode QPSetUp(QP qp)
{
  MPI_Comm comm;

  PermonTracedFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (qp->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscObjectGetComm((PetscObject)qp, &comm));
  PetscCheck(qp->A, comm, PETSC_ERR_ORDER, "Hessian must be set before " __FUNCT__);
  PetscCheck(qp->b, comm, PETSC_ERR_ORDER, "linear term must be set before " __FUNCT__);

  PermonTraceBegin;
  PetscCall(PetscInfo(qp, "setup QP #%d\n", qp->id));
  PetscCall(QPSetUpInnerObjects(qp));
  PetscCall(QPSetFromOptions_Private(qp));
  PetscCall(PCSetUp(qp->pc));
  qp->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCompute_BEt_lambda"
PetscErrorCode QPCompute_BEt_lambda(QP qp, Vec *BEt_lambda)
{
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  *BEt_lambda = NULL;
  if (!qp->BE) PetscFunctionReturn(PETSC_SUCCESS);

  if (!qp->BI) {
    PetscCall(VecIsInvalidated(qp->Bt_lambda, &flg));
    if (!flg) {
      PetscCall(VecDuplicate(qp->Bt_lambda, BEt_lambda));
      PetscCall(VecCopy(qp->Bt_lambda, *BEt_lambda)); /* BEt_lambda = Bt_lambda */
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(VecIsInvalidated(qp->lambda, &flg));
    if (!flg && qp->B->ops->multtranspose) {
      PetscCall(MatCreateVecs(qp->B, BEt_lambda, NULL));
      PetscCall(MatMultTranspose(qp->B, qp->lambda, *BEt_lambda)); /* BEt_lambda = B'*lambda */
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  PetscCall(VecIsInvalidated(qp->lambda_E, &flg));
  if (!flg || !qp->BE->ops->multtranspose) {
    PetscCall(MatCreateVecs(qp->BE, BEt_lambda, NULL));
    PetscCall(MatMultTranspose(qp->BE, qp->lambda_E, *BEt_lambda)); /* Bt_lambda = BE'*lambda_E */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeLagrangianGradient"
PetscErrorCode QPComputeLagrangianGradient(QP qp, Vec x, Vec r, char *kkt_name_[])
{
  Vec       b, cE, cI, Bt_lambda = NULL;
  Mat       A, BE, BI;
  PetscBool flg = PETSC_FALSE, avail = PETSC_TRUE;
  char      kkt_name[256] = "A*x - b";

  PetscFunctionBegin;
  PetscCall(QPSetUp(qp));
  PetscCall(QPGetOperator(qp, &A));
  PetscCall(QPGetRhs(qp, &b));
  PetscCall(QPGetEq(qp, &BE, &cE));
  PetscCall(QPGetIneq(qp, &BI, &cI));

  PetscCall(MatMult(A, x, r));
  PetscCall(VecAXPY(r, -1.0, b)); /* r = A*x - b */

  /* TODO: replace with QPC function */
  {
    Vec lb, ub;
    Vec llb, lub;
    IS  is;
    QPC qpc;

    PetscCall(QPGetBox(qp, &is, &lb, &ub));
    PetscCall(QPGetQPC(qp, &qpc));
    if (qpc) PetscCall(QPCBoxGetMultipliers(qpc, &llb, &lub));
    if (lb) {
      if (is) {
        PetscCall(VecISAXPY(r, is, -1.0, llb));
      } else {
        PetscCall(VecAXPY(r, -1.0, llb));
      }
    }
    if (ub) {
      if (is) {
        PetscCall(VecISAXPY(r, is, 1.0, lub));
      } else {
        PetscCall(VecAXPY(r, 1.0, lub));
      }
    }
  }

  PetscCall(VecDuplicate(r, &Bt_lambda));
  if (qp->B) {
    PetscCall(VecIsInvalidated(qp->Bt_lambda, &flg));
    if (!flg) {
      PetscCall(VecCopy(qp->Bt_lambda, Bt_lambda)); /* Bt_lambda = (B'*lambda) */
      if (kkt_name_) PetscCall(PetscStrlcat(kkt_name, " + (B'*lambda)", sizeof(kkt_name)));
      goto endif;
    }

    PetscCall(VecIsInvalidated(qp->lambda, &flg));
    if (!flg && qp->B->ops->multtranspose) {
      PetscCall(MatMultTranspose(qp->B, qp->lambda, Bt_lambda)); /* Bt_lambda = B'*lambda */
      if (kkt_name_) PetscCall(PetscStrlcat(kkt_name, " + B'*lambda", sizeof(kkt_name)));
      goto endif;
    }

    if (qp->BE) {
      if (kkt_name_) PetscCall(PetscStrlcat(kkt_name, " + BE'*lambda_E", sizeof(kkt_name)));
    }
    if (qp->BI) {
      if (kkt_name_) PetscCall(PetscStrlcat(kkt_name, " + BI'*lambda_I", sizeof(kkt_name)));
    }

    if (qp->BE) {
      PetscCall(VecIsInvalidated(qp->lambda_E, &flg));
      if (flg || !qp->BE->ops->multtranspose) {
        avail = PETSC_FALSE;
        goto endif;
      }
      PetscCall(MatMultTranspose(BE, qp->lambda_E, Bt_lambda)); /* Bt_lambda = BE'*lambda_E */
    }

    if (qp->BI) {
      PetscCall(VecIsInvalidated(qp->lambda_I, &flg));
      if (flg || !qp->BI->ops->multtransposeadd) {
        avail = PETSC_FALSE;
        goto endif;
      }
      PetscCall(MatMultTransposeAdd(BI, qp->lambda_I, Bt_lambda, Bt_lambda)); /* Bt_lambda = Bt_lambda + BI'*lambda_I */
    }
  }
endif:
  if (avail) {
    Vec lb, ub;

    /* TODO: replace with QPC function */
    PetscCall(QPGetBox(qp, NULL, &lb, &ub));
    PetscCall(VecAXPY(r, 1.0, Bt_lambda)); /* r = r + Bt_lambda */
    if (lb) {
      if (kkt_name_) PetscCall(PetscStrlcat(kkt_name, " - lambda_lb", sizeof(kkt_name)));
    }
    if (ub) {
      if (kkt_name_) PetscCall(PetscStrlcat(kkt_name, " + lambda_ub", sizeof(kkt_name)));
    }
  } else {
    PetscCall(VecInvalidate(r));
  }
  if (kkt_name_) PetscCall(PetscStrallocpy(kkt_name, kkt_name_));
  PetscCall(VecDestroy(&Bt_lambda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeMissingEqMultiplier"
PetscErrorCode QPComputeMissingEqMultiplier(QP qp)
{
  Vec         r = qp->xwork;
  PetscBool   flg;
  const char *name;
  QP          qp_I;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(QPSetUp(qp));

  if (!qp->BE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecIsInvalidated(qp->lambda_E, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  if (qp->Bt_lambda) {
    PetscCall(VecIsInvalidated(qp->Bt_lambda, &flg));
    if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(QPDuplicate(qp, QP_DUPLICATE_COPY_POINTERS, &qp_I));
  PetscCall(QPSetEq(qp_I, NULL, NULL));
  PetscCall(QPComputeLagrangianGradient(qp_I, qp->x, r, NULL));
  PetscCall(QPDestroy(&qp_I));

  //TODO Should we add qp->Bt_lambda_E ?
  if (qp->BE == qp->B) {
    PetscCall(VecCopy(r, qp->Bt_lambda));
    PetscCall(VecScale(qp->Bt_lambda, -1.0));
  } else {
    PetscCall(QPPFApplyHalfQ(qp->pf, r, qp->lambda_E));
    PetscCall(VecScale(qp->lambda_E, -1.0)); /* lambda_E_LS = -(BE*BE')\\BE*r */
  }

  if (PermonDebugEnabled) {
    PetscReal norm;
    if (qp->BE->ops->multtranspose) {
      PetscCall(MatMultTransposeAdd(qp->BE, qp->lambda_E, r, r));
    } else {
      PetscCall(VecAXPY(r, 1.0, qp->Bt_lambda));
    }
    PetscCall(VecNorm(r, NORM_2, &norm));
    PetscCall(PermonDebug1("||r||=%.2e\n", norm));
  }

  PetscCall(PetscObjectGetName((PetscObject)qp, &name));
  PetscCall(PetscInfo(qp, "missing eq. con. multiplier computed for QP Object %s (#%d in chain, derived by %s)\n", name, qp->id, qp->transform_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeMissingBoxMultipliers"
PetscErrorCode QPComputeMissingBoxMultipliers(QP qp)
{
  Vec       lb, ub;
  Vec       llb, lub;
  IS        is;
  Vec       r   = qp->xwork;
  PetscBool flg = PETSC_FALSE, flg2 = PETSC_FALSE;
  QP        qp_E;
  QPC       qpc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(QPSetUp(qp));

  PetscCall(QPGetBox(qp, &is, &lb, &ub));
  PetscCall(QPGetQPC(qp, &qpc));
  if (qpc) PetscCall(QPCBoxGetMultipliers(qpc, &llb, &lub));

  if (lb) { PetscCall(VecIsInvalidated(llb, &flg)); }
  if (ub) { PetscCall(VecIsInvalidated(lub, &flg2)); }
  if (!lb && !ub) PetscFunctionReturn(PETSC_SUCCESS);
  if (!flg && !flg2) PetscFunctionReturn(PETSC_SUCCESS);

  /* currently cannot handle this situation, leave multipliers untouched */
  if (qp->BI) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(QPDuplicate(qp, QP_DUPLICATE_COPY_POINTERS, &qp_E));
  PetscCall(QPCDestroy(&qp_E->qpc));
  PetscCall(QPComputeLagrangianGradient(qp_E, qp->x, r, NULL));
  PetscCall(QPDestroy(&qp_E));

  if (lb) {
    if (is) {
      PetscCall(VecISCopy(r, is, SCATTER_REVERSE, llb));
    } else {
      PetscCall(VecCopy(r, llb));
    }
  }
  if (ub) {
    if (is) {
      PetscCall(VecISCopy(r, is, SCATTER_REVERSE, lub));
    } else {
      PetscCall(VecCopy(r, lub));
    }
    PetscCall(VecScale(lub, -1.0));
  }
  if (lb && ub) {
    Vec lambdawork;
    PetscCall(VecDuplicate(llb, &lambdawork));
    PetscCall(VecZeroEntries(lambdawork));
    PetscCall(VecPointwiseMax(llb, llb, lambdawork));
    PetscCall(VecPointwiseMax(lub, lub, lambdawork));
    PetscCall(VecDestroy(&lambdawork));
  }

  {
    const char *name_qp;
    PetscCall(PetscObjectGetName((PetscObject)qp, &name_qp));
    PetscCall(PetscInfo(qp, "missing lower bound con. multiplier computed for QP Object %s (#%d in chain, derived by %s)\n", name_qp, qp->id, qp->transform_name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjective"
/*@
   QPComputeObjective - Evaluate the quadratic objective function f(x) = 1/2*x'*A*x - x'*b.

   Collective on QP

   Input Parameters:
+  qp   - the QP
-  x    - the state vector

   Output Parameter:
.  f    - the objective value

   Notes:
   Computes f(x) as -x'(b - 1/2*A*x).

   Level: beginner

.seealso: QPComputeObjectiveGradient(), QPComputeObjectiveFromGradient(), QPComputeObjectiveAndGradient()
@*/
PetscErrorCode QPComputeObjective(QP qp, Vec x, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscCheck(qp->setupcalled, PetscObjectComm((PetscObject)qp), PETSC_ERR_ORDER, "QPSetUp must be called first.");
  PetscCall(MatMult(qp->A, x, qp->xwork));
  PetscCall(VecAYPX(qp->xwork, -0.5, qp->b));
  PetscCall(VecDot(x, qp->xwork, f));
  *f = -*f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjectiveGradient"
/*@
   QPComputeObjectiveGradient - Computes the gradient of the quadratic objective function g(x) = Ax - b.

   Collective on QP

   Input Parameters:
+  qp   - the QP
-  x    - the state vector

   Output Parameter:
.  g    - the gradient value

   Level: intermediate

.seealso: QPComputeObjective(), QPComputeObjectiveFromGradient(), QPComputeObjectiveAndGradient()
@*/
PetscErrorCode QPComputeObjectiveGradient(QP qp, Vec x, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheck(qp->setupcalled, PetscObjectComm((PetscObject)qp), PETSC_ERR_ORDER, "QPSetUp must be called first.");
  PetscCall(MatMult(qp->A, x, g));
  PetscCall(VecAXPY(g, -1.0, qp->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjectiveFromGradient"
/*@
   QPComputeObjectiveFromGradient - Evaluate the quadratic objective function f(x) = 1/2*x'*A*x - x'*b using known gradient.

   Collective on QP

   Input Parameters:
+  qp   - the QP
.  x    - the state vector
-  g    - the gradient

   Output Parameter:
.  f    - the objective value

   Notes:
   Computes f(x) as x'*(g - b)/2

   Level: intermediate

.seealso: QPComputeObjective(), QPComputeObjectiveGradient(), QPComputeObjectiveAndGradient()
@*/
PetscErrorCode QPComputeObjectiveFromGradient(QP qp, Vec x, Vec g, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 4);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheck(qp->setupcalled, PetscObjectComm((PetscObject)qp), PETSC_ERR_ORDER, "QPSetUp must be called first.");

  PetscCall(VecWAXPY(qp->xwork, -1.0, qp->b, g));
  PetscCall(VecDot(x, qp->xwork, f));
  *f /= 2.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeObjectiveAndGradient"
/*@
   QPComputeObjectiveAndGradient - Computes the objective and gradient at once.

   Collective on QP

   Input Parameters:
+  qp   - the QP
-  x    - the state vector

   Output Parameters:
+  g    - the gradient
-  f    - the objective value

   Notes:
   Computes g(x) = A*x - b and f(x) = x'*(g - b)/2

   Level: intermediate

.seealso: QPComputeObjective(), QPComputeObjectiveGradient(), QPComputeObjectiveFromGradient()
@*/
PetscErrorCode QPComputeObjectiveAndGradient(QP qp, Vec x, Vec g, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (f) PetscAssertPointer(f, 4);
  if (g) PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheck(qp->setupcalled, PetscObjectComm((PetscObject)qp), PETSC_ERR_ORDER, "QPSetUp must be called first.");

  if (!g) {
    PetscCall(QPComputeObjective(qp, x, f));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(QPComputeObjectiveGradient(qp, x, g));
  if (!f) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(QPComputeObjectiveFromGradient(qp, x, g, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPDestroy"
/*@
   QPDestroy - Destroys the QP object.

   Collective on QP

   Input Parameter:
.  qp - QP context

   Level: beginner

.seealso QPCreate()
@*/
PetscErrorCode QPDestroy(QP *qp)
{
  PetscFunctionBegin;
  if (!*qp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*qp, QP_CLASSID, 1);
  if (--((PetscObject)(*qp))->refct > 0) {
    *qp = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*qp)->postSolveCtxDestroy) PetscCall((*qp)->postSolveCtxDestroy((*qp)->postSolveCtx));
  PetscCall(QPReset(*qp));
  PetscCall(QPPFDestroy(&(*qp)->pf));
  PetscCall(PetscHeaderDestroy(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetOperator"
/*@
   QPSetOperator - Sets the Hessian matrix

   Collective on QP

   Input Parameters:
+  qp  - the QP
-  A   - the Hessian matrix

   Level: beginner

.seealso QPGetOperator(), QPSetRhs(), QPSetEq(), QPAddEq(), QPSetIneq(), QPSetBox(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPSetOperator(QP qp, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 2);
  PetscCheckSameComm(qp, 1, A, 2);
  if (A == qp->A) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatDestroy(&qp->A));
  qp->A = A;
  PetscCall(PetscObjectReference((PetscObject)A));

  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetPC"
/*@
   QPSetPC - Sets preconditioner context.

   Collective on QP

   Input Parameters:
+  qp - the QP
-  pc - the preconditioner context

   Level: developer
@*/
PetscErrorCode QPSetPC(QP qp, PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(pc, PC_CLASSID, 2);
  PetscCheckSameComm(qp, 1, pc, 2);
  if (pc == qp->pc) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCDestroy(&qp->pc));
  qp->pc = pc;
  PetscCall(PetscObjectReference((PetscObject)pc));
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetOperator"
/*@
   QPGetOperator - Get the Hessian matrix.

   Not Collective

   Input Parameter:
.  qp  - the QP

   Output Parameter:
.  A   - the Hessian matrix

   Level: intermediate

.seealso QPSetOperator(), QPGetRhs(), QPGetEq(), QPGetIneq(), QPGetBox(), QPGetSolutionVector()
@*/
PetscErrorCode QPGetOperator(QP qp, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(A, 2);
  *A = qp->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetPC"
/*@
   QPGetPC - Get preconditioner context.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  pc - the preconditioner context

   Level: advanced
@*/
PetscErrorCode QPGetPC(QP qp, PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(pc, 2);
  if (!qp->pc) {
    PetscCall(PCCreate(PetscObjectComm((PetscObject)qp), &qp->pc));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)qp->pc, (PetscObject)qp, 0));
    PetscCall(PCSetType(qp->pc, PCNONE));
    PetscCall(PCSetOperators(qp->pc, qp->A, qp->A));
  }
  *pc = qp->pc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetOperatorNullSpace"
/*@
   QPSetOperatorNullSpace - Sets matrix with columns representing the null space of the Hessian operator.

   Collective on QP

   Input Parameters:
+  qp - the QP
-  R - the null space matrix

   Level: intermediate

.seealso QPGetOperatorNullSpace(), QPSetOperator(), QPSSolve()
@*/
PetscErrorCode QPSetOperatorNullSpace(QP qp, Mat R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (R) PetscValidHeaderSpecific(R, MAT_CLASSID, 2);
  if (R == qp->R) PetscFunctionReturn(PETSC_SUCCESS);
  if (R) {
#if defined(PETSC_USE_DEBUG)
    PetscCall(MatCheckNullSpace(qp->A, R, PETSC_SMALL));
#endif
    PetscCall(PetscObjectReference((PetscObject)R));
  }
  PetscCall(MatDestroy(&qp->R));
  qp->R = R;
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetOperatorNullSpace"
/*@
   QPGetOperatorNullSpace - Get matrix with columns representing the null space of the Hessian operator.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  R - the null space matrix

   Level: advanced

.seealso QPSetOperatorNullSpace(), QPGetOperator()
@*/
PetscErrorCode QPGetOperatorNullSpace(QP qp, Mat *R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(R, 2);
  *R = qp->R;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetRhs"
/*@
   QPSetRhs - Set the QP right hand side (linear term) b with '-' sign, i.e. objective function 1/2*x'*A'*x - x'*b.

   Collective on QP

   Input Parameters:
+  qp - the QP
-  b - right hand side

   Level: beginner

.seealso QPSetRhsPlus(), QPGetRhs(), QPSetOperator(), QPSetEq(), QPAddEq(), QPSetIneq(), QPSetBox(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPSetRhs(QP qp, Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(b, VEC_CLASSID, 2);
  PetscCheckSameComm(qp, 1, b, 2);
  if (b == qp->b && qp->b_plus == PETSC_FALSE) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecDestroy(&qp->b));
  qp->b = b;
  PetscCall(PetscObjectReference((PetscObject)b));
  qp->b_plus = PETSC_FALSE;

  if (PermonDebugEnabled) {
    PetscReal norm;
    PetscCall(VecNorm(b, NORM_2, &norm));
    PetscCall(PermonDebug1("||b|| = %0.2e\n", norm));
  }

  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetRhsPlus"
/*@
   QPSetRhsPlus - Set the QP right hand side (linear term) b with '+' sign, i.e. objective function 1/2*x'*A'*x + x'*b.

   Collective on QP

   Input Parameters:
+  qp - the QP
-  b - right hand side

   Level: beginner

.seealso QPSetRhs(), QPGetRhs(), QPSetOperator(), QPSetEq(), QPAddEq(), QPSetIneq(), QPSetBox(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPSetRhsPlus(QP qp, Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(b, VEC_CLASSID, 2);
  PetscCheckSameComm(qp, 1, b, 2);
  if (b == qp->b && qp->b_plus == PETSC_TRUE) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecDuplicate(b, &qp->b));
  PetscCall(VecCopy(b, qp->b));
  PetscCall(VecScale(qp->b, -1.0));
  qp->b_plus = PETSC_TRUE;

  if (PermonDebugEnabled) {
    PetscReal norm;
    PetscCall(VecNorm(b, NORM_2, &norm));
    PetscCall(PermonDebug1("||b|| = %0.2e\n", norm));
  }

  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetRhs"
/*@
   QPGetRhs - Get the QPs right hand side.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  b - right hand side

   Level: intermediate

.seealso QPSetRhs(), QPGetOperator(), QPGetEq(), QPGetIneq(), QPGetBox(), QPGetSolutionVector()
@*/
PetscErrorCode QPGetRhs(QP qp, Vec *b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(b, 2);
  *b = qp->b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetIneq"
/*@
   QPSetIneq - Sets the inequality constraints.

   Collective on QP

   Input Parameters:
+  qp - the QP
.  Bineq - boolean matrix representing the inequality constraints placement
-  cineq - vector prescribing inequality constraints

   Level: beginner

.seealso QPGetIneq(), QPSetOperator(), QPSetRhs(), QPSetEq(), QPAddEq(), QPSetBox(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPSetIneq(QP qp, Mat Bineq, Vec cineq)
{
  PetscReal norm;
  PetscBool change = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);

  if (Bineq) {
    PetscValidHeaderSpecific(Bineq, MAT_CLASSID, 2);
    PetscCheckSameComm(qp, 1, Bineq, 2);
  }

  if (Bineq != qp->BI) {
    if (Bineq) PetscCall(PetscObjectReference((PetscObject)Bineq));
    PetscCall(MatDestroy(&qp->BI));
    PetscCall(MatDestroy(&qp->B));
    PetscCall(QPSetIneqMultiplier(qp, NULL));
    qp->BI = Bineq;
    change = PETSC_TRUE;
  }

  if (cineq) {
    if (!Bineq) {
      cineq = NULL;
      PetscCall(PetscInfo(qp, "null inequality constraint matrix specified, the constraint RHS vector will be ignored\n"));
    } else {
      PetscValidHeaderSpecific(cineq, VEC_CLASSID, 3);
      PetscCheckSameComm(qp, 1, cineq, 3);
      PetscCall(VecNorm(cineq, NORM_2, &norm));
      PetscCall(PermonDebug1("||cineq|| = %0.2e\n", norm));
      if (norm < PETSC_MACHINE_EPSILON) {
        PetscCall(PetscInfo(qp, "zero inequality constraint RHS vector detected\n"));
        cineq = NULL;
      }
    }
  } else if (Bineq) {
    PetscCall(PetscInfo(qp, "null inequality constraint RHS vector handled as zero vector\n"));
  }

  if (cineq != qp->cI) {
    if (cineq) PetscCall(PetscObjectReference((PetscObject)cineq));
    PetscCall(VecDestroy(&qp->cI));
    PetscCall(VecDestroy(&qp->c));
    qp->cI = cineq;
    change = PETSC_TRUE;
  }

  if (!Bineq) { PetscCall(VecDestroy(&qp->lambda_I)); }

  if (change) {
    if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetIneq"
/*@
   QPGetIneq - Get the inequality constraints.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
+  Bineq - boolean matrix representing the inequality constraints placement
-  cineq - vector prescribing inequality constraints

   Level: intermediate

.seealso QPSetIneq(), QPGetOperator(), QPGetRhs(), QPSetEq(), QPAddEq(), QPSetBox(), QPSetInitialVector()
@*/
PetscErrorCode QPGetIneq(QP qp, Mat *Bineq, Vec *cineq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (Bineq) {
    PetscAssertPointer(Bineq, 2);
    *Bineq = qp->BI;
  }
  if (cineq) {
    PetscAssertPointer(cineq, 3);
    *cineq = qp->cI;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetEq"
/*@
   QPSetEq - Sets the equality constraints.

   Collective on QP

   Input Parameter:
+  qp - the QP
.  Beq - boolean matrix representing the equality constraints placement
-  ceq - vector prescribing equality constraints

   Level: beginner

.seealso QPGetEq(), QPAddEq(), QPSetOperator(), QPSetRhs(), QPSetIneq(), QPSetBox(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPSetEq(QP qp, Mat Beq, Vec ceq)
{
  PetscReal norm;
  PetscBool change = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);

  if (Beq) {
    PetscValidHeaderSpecific(Beq, MAT_CLASSID, 2);
    PetscCheckSameComm(qp, 1, Beq, 2);
  }

  if (Beq != qp->BE) {
    if (Beq) {
      PetscCall(QPGetQPPF(qp, &qp->pf));
      PetscCall(QPPFSetG(qp->pf, Beq));
      PetscCall(PetscObjectReference((PetscObject)Beq));
    }
    PetscCall(MatDestroy(&qp->BE));
    PetscCall(MatDestroy(&qp->B));
    PetscCall(QPSetEqMultiplier(qp, NULL));
    qp->BE            = Beq;
    qp->BE_nest_count = 0;
    change            = PETSC_TRUE;
  }

  if (ceq) {
    if (!Beq) {
      ceq = NULL;
      PetscCall(PetscInfo(qp, "null equality constraint matrix specified, the constraint RHS vector will be ignored\n"));
    } else {
      PetscValidHeaderSpecific(ceq, VEC_CLASSID, 3);
      PetscCheckSameComm(qp, 1, ceq, 3);
      PetscCall(VecNorm(ceq, NORM_2, &norm));
      PetscCall(PermonDebug1("||ceq|| = %0.2e\n", norm));
      if (norm < PETSC_MACHINE_EPSILON) {
        PetscCall(PetscInfo(qp, "zero equality constraint RHS vector detected\n"));
        ceq = NULL;
      }
    }
  } else if (Beq) {
    PetscCall(PetscInfo(qp, "null equality constraint RHS vector handled as zero vector\n"));
  }

  if (ceq != qp->cE) {
    if (ceq) PetscCall(PetscObjectReference((PetscObject)ceq));
    PetscCall(VecDestroy(&qp->cE));
    PetscCall(VecDestroy(&qp->c));
    qp->cE = ceq;
    change = PETSC_TRUE;
  }

  if (!Beq) { PetscCall(VecDestroy(&qp->lambda_E)); }

  if (change) {
    if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPAddEq"
/*@
   QPAddEq - Add the equality constraints.

   Collective on QP

   Input Parameter:
+  qp - the QP
.  Beq - boolean matrix representing the equality constraints placement
-  ceq - vector prescribing equality constraints

   Level: beginner

.seealso QPGetEq(), QPSetEq(), QPSetOperator(), QPSetRhs(), QPSetIneq(), QPSetBox(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPAddEq(QP qp, Mat Beq, Vec ceq)
{
  PetscReal norm;
  Mat      *subBE;
  PetscInt  M, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(Beq, MAT_CLASSID, 2);
  PetscCheckSameComm(qp, 1, Beq, 2);

  /* handle BE that has been set by QPSetEq */
  if (qp->BE && !qp->BE_nest_count) {
    Mat BE_orig = qp->BE;
    Vec cE_orig = qp->cE;

    PetscCall(PetscObjectReference((PetscObject)BE_orig));
    if (cE_orig) PetscCall(PetscObjectReference((PetscObject)cE_orig));
    PetscCall(QPSetEq(qp, NULL, NULL));
    PetscCall(QPAddEq(qp, BE_orig, cE_orig));
    PetscCall(MatDestroy(&BE_orig));
    PetscCall(VecDestroy(&cE_orig));
    PERMON_ASSERT(qp->BE_nest_count == 1, "qp->BE_nest_count==1");
  }

  M = qp->BE_nest_count++;

  PetscCall(PetscMalloc((M + 1) * sizeof(Mat), &subBE)); //Mat subBE[M+1];
  for (i = 0; i < M; i++) {
    PetscCall(MatNestGetSubMat(qp->BE, i, 0, &subBE[i]));
    PetscCall(PetscObjectReference((PetscObject)subBE[i]));
  }
  subBE[M] = Beq;

  PetscCall(MatDestroy(&qp->BE));
  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)qp), M + 1, NULL, 1, NULL, subBE, &qp->BE));
  PetscCall(PetscObjectSetName((PetscObject)qp->BE, "BE"));

  PetscCall(QPGetQPPF(qp, &qp->pf));
  PetscCall(QPPFSetG(qp->pf, qp->BE));

  if (ceq) {
    PetscValidHeaderSpecific(ceq, VEC_CLASSID, 3);
    PetscCheckSameComm(qp, 1, ceq, 3);
    PetscCall(VecNorm(ceq, NORM_2, &norm));
    PetscCall(PermonDebug1("||ceq|| = %0.2e\n", norm));
    if (norm < PETSC_MACHINE_EPSILON) {
      PetscCall(PetscInfo(qp, "zero equality constraint RHS vector detected\n"));
      ceq = NULL;
    }
  } else {
    PetscCall(PetscInfo(qp, "null equality constraint RHS vector handled as zero vector\n"));
  }

  if (ceq || qp->cE) {
    Vec *subCE;
    PetscCall(PetscMalloc((M + 1) * sizeof(Vec), &subCE));
    if (qp->cE) {
      for (i = 0; i < M; i++) {
        PetscCall(VecNestGetSubVec(qp->cE, i, &subCE[i]));
        PetscCall(PetscObjectReference((PetscObject)subCE[i]));
      }
      PetscCall(VecDestroy(&qp->cE));
    } else {
      for (i = 0; i < M; i++) {
        PetscCall(MatCreateVecs(subBE[i], NULL, &subCE[i]));
        PetscCall(VecSet(subCE[i], 0.0));
      }
    }
    if (!ceq) {
      PetscCall(MatCreateVecs(Beq, NULL, &ceq));
      PetscCall(VecSet(ceq, 0.0));
    }
    subCE[M] = ceq;

    PetscCall(VecCreateNest(PetscObjectComm((PetscObject)qp), M + 1, NULL, subCE, &qp->cE));
    for (i = 0; i < M; i++) { PetscCall(VecDestroy(&subCE[i])); }
    PetscCall(PetscFree(subCE));
  }

  for (i = 0; i < M; i++) { PetscCall(MatDestroy(&subBE[i])); }
  PetscCall(PetscFree(subBE));
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetEqMultiplicityScaling"
PetscErrorCode QPGetEqMultiplicityScaling(QP qp, Vec *dE_new, Vec *dI_new)
{
  MPI_Comm           comm;
  PetscInt           i, ilo, ihi, j, k, ncols;
  PetscScalar        multiplicity;
  Mat                Bc = NULL, Bd = NULL, Bg = NULL;
  Mat                Bct = NULL, Bdt = NULL, Bgt = NULL;
  PetscBool          flg, scale_Bd = PETSC_TRUE, scale_Bc = PETSC_TRUE, count_Bd = PETSC_TRUE, count_Bc = PETSC_TRUE;
  Vec                dof_multiplicities = NULL, edge_multiplicities_g = NULL, edge_multiplicities_d = NULL, edge_multiplicities_c = NULL;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBeginI;
  PetscCall(PetscObjectGetComm((PetscObject)qp, &comm));
  //TODO we now assume fully redundant case
  if (!qp->BE_nest_count) {
    Bg = qp->BE;
  } else {
    PetscCall(MatNestGetSubMat(qp->BE, 0, 0, &Bg));
    if (qp->BE_nest_count >= 2) { PetscCall(MatNestGetSubMat(qp->BE, 1, 0, &Bd)); }
  }
  Bc = qp->BI;
  PERMON_ASSERT(Bg, "Bg");

  if (!Bc) {
    scale_Bc = PETSC_FALSE;
    count_Bc = PETSC_FALSE;
  }
  if (!Bd) {
    scale_Bd = PETSC_FALSE;
    count_Bd = PETSC_FALSE;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-qp_E_scale_Bd", &scale_Bd, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-qp_E_scale_Bc", &scale_Bc, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-qp_E_count_Bd", &count_Bd, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-qp_E_count_Bc", &count_Bc, NULL));
  //PetscCheck(!scale_Bc || count_Bc,PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_INCOMP,"-qp_E_scale_Bc implies -qp_E_count_Bc");
  //PetscCheck(!scale_Bd || count_Bd,PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_INCOMP,"-qp_E_scale_Bd implies -qp_E_count_Bd");

  PetscCall(MatGetOwnershipRangeColumn(Bg, &ilo, &ihi));

  {
    PetscCall(MatCreateVecs(Bg, &dof_multiplicities, &edge_multiplicities_g));
    PetscCall(VecSet(dof_multiplicities, 1.0));
  }
  if (scale_Bd) {
    PetscCall(MatCreateVecs(Bd, NULL, &edge_multiplicities_d));
    PetscCall(VecSet(edge_multiplicities_d, 1.0));
  }
  if (scale_Bc) {
    PetscCall(MatCreateVecs(Bc, NULL, &edge_multiplicities_c));
    PetscCall(VecSet(edge_multiplicities_c, 1.0));
  }

  {
    PetscCall(MatIsImplicitTranspose(Bg, &flg));
    PERMON_ASSERT(flg, "Bg must be implicit transpose");
    PetscCall(PermonMatTranspose(Bg, MAT_TRANSPOSE_EXPLICIT, &Bgt));
    for (i = ilo; i < ihi; i++) {
      PetscCall(MatGetRow(Bgt, i, &ncols, &cols, &vals));
      k = 0;
      for (j = 0; j < ncols; j++) {
        if (vals[j]) k++;
      }
      PetscCall(MatRestoreRow(Bgt, i, &ncols, &cols, &vals));
      if (k) {
        multiplicity = k + 1;
        PetscCall(VecSetValue(dof_multiplicities, i, multiplicity, INSERT_VALUES));
      }
    }
  }

  if (count_Bd) {
    PetscCall(MatIsImplicitTranspose(Bd, &flg));
    PERMON_ASSERT(flg, "Bd must be implicit transpose");
    PetscCall(PermonMatTranspose(Bd, MAT_TRANSPOSE_EXPLICIT, &Bdt));
    for (i = ilo; i < ihi; i++) {
      PetscCall(MatGetRow(Bdt, i, &ncols, &cols, &vals));
      k = 0;
      for (j = 0; j < ncols; j++) {
        if (vals[j]) k++;
        PetscCheck(k <= 1, comm, PETSC_ERR_PLIB, "more than one nonzero in Bd row %d", i);
      }
      PetscCall(MatRestoreRow(Bdt, i, &ncols, &cols, &vals));
      if (k) {
        PetscCall(VecGetValues(dof_multiplicities, 1, &i, &multiplicity));
        multiplicity++;
        PetscCall(VecSetValue(dof_multiplicities, i, multiplicity, INSERT_VALUES));
      }
    }
  }

  if (count_Bc) {
    PetscCall(MatIsImplicitTranspose(Bc, &flg));
    PERMON_ASSERT(flg, "Bc must be implicit transpose");
    PetscCall(PermonMatTranspose(Bc, MAT_TRANSPOSE_EXPLICIT, &Bct));
    for (i = ilo; i < ihi; i++) {
      PetscCall(MatGetRow(Bct, i, &ncols, &cols, &vals));
      k = 0;
      for (j = 0; j < ncols; j++) {
        if (vals[j]) k++;
      }
      PetscCall(MatRestoreRow(Bct, i, &ncols, &cols, &vals));
      if (k > 1) PetscCall(PetscPrintf(comm, "WARNING: more than one nonzero in Bc row %d\n", i));
      if (k) {
        PetscCall(VecGetValues(dof_multiplicities, 1, &i, &multiplicity));
        multiplicity++;
        PetscCall(VecSetValue(dof_multiplicities, i, multiplicity, INSERT_VALUES));
      }
    }
  }

  PetscCall(VecAssemblyBegin(dof_multiplicities));
  PetscCall(VecAssemblyEnd(dof_multiplicities));
  PetscCall(VecSqrtAbs(dof_multiplicities));
  PetscCall(VecReciprocal(dof_multiplicities));

  {
    for (i = ilo; i < ihi; i++) {
      PetscCall(MatGetRow(Bgt, i, &ncols, &cols, NULL));
      if (ncols) {
        PetscCall(VecGetValues(dof_multiplicities, 1, &i, &multiplicity));
        for (j = 0; j < ncols; j++) { PetscCall(VecSetValue(edge_multiplicities_g, cols[j], multiplicity, INSERT_VALUES)); }
      }
      PetscCall(MatRestoreRow(Bgt, i, &ncols, &cols, NULL));
    }
    PetscCall(VecAssemblyBegin(edge_multiplicities_g));
    PetscCall(VecAssemblyEnd(edge_multiplicities_g));
    PetscCall(MatDestroy(&Bgt));
  }

  if (scale_Bd) {
    PetscCall(PermonMatTranspose(Bd, MAT_TRANSPOSE_EXPLICIT, &Bdt));
    for (i = ilo; i < ihi; i++) {
      PetscCall(MatGetRow(Bdt, i, &ncols, &cols, NULL));
      if (ncols) {
        PetscCall(VecGetValues(dof_multiplicities, 1, &i, &multiplicity));
        for (j = 0; j < ncols; j++) { PetscCall(VecSetValue(edge_multiplicities_d, cols[j], multiplicity, INSERT_VALUES)); }
      }
      PetscCall(MatRestoreRow(Bdt, i, &ncols, &cols, NULL));
    }
    PetscCall(VecAssemblyBegin(edge_multiplicities_d));
    PetscCall(VecAssemblyEnd(edge_multiplicities_d));
    PetscCall(MatDestroy(&Bdt));
  }

  if (scale_Bc) {
    PetscCall(PermonMatTranspose(Bc, MAT_TRANSPOSE_EXPLICIT, &Bct));
    for (i = ilo; i < ihi; i++) {
      PetscCall(MatGetRow(Bct, i, &ncols, &cols, NULL));
      if (ncols) {
        PetscCall(VecGetValues(dof_multiplicities, 1, &i, &multiplicity));
        for (j = 0; j < ncols; j++) { PetscCall(VecSetValue(edge_multiplicities_c, cols[j], multiplicity, INSERT_VALUES)); }
      }
      PetscCall(MatRestoreRow(Bct, i, &ncols, &cols, NULL));
    }
    PetscCall(VecAssemblyBegin(edge_multiplicities_c));
    PetscCall(VecAssemblyEnd(edge_multiplicities_c));
    PetscCall(MatDestroy(&Bct));
  }

  if (edge_multiplicities_d) {
    Vec dE_vecs[2] = {edge_multiplicities_g, edge_multiplicities_d};
    PetscCall(VecCreateNest(PetscObjectComm((PetscObject)qp), 2, NULL, dE_vecs, dE_new));
    PetscCall(VecDestroy(&edge_multiplicities_d));
    PetscCall(VecDestroy(&edge_multiplicities_g));
  } else {
    *dE_new = edge_multiplicities_g;
  }

  *dI_new = edge_multiplicities_c;

  PetscCall(VecDestroy(&dof_multiplicities));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetEq"
/*@
   QPGetEq - Get the equality constraints.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
+  Beq - boolean matrix representing the equality constraints placement
-  ceq - vector prescribing equality constraints

   Level: intermediate

.seealso QPSetEq(), QPAddEq(), QPGetOperator(), QPGetRhs(), QPGetIneq(), QPGetBox(), QPGetSolutionVector()
@*/
PetscErrorCode QPGetEq(QP qp, Mat *Beq, Vec *ceq)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (Beq) {
    PetscAssertPointer(Beq, 2);
    *Beq = qp->BE;
  }
  if (ceq) {
    PetscAssertPointer(ceq, 3);
    *ceq = qp->cE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetBox"
/*@
   QPSetBox - Sets the box constraints.

   Collective on QP

   Input Parameter:
+  qp - the QP
.  is - index set of constrained variables; if NULL then all unknowns are constrained
.  lb - lower bound
-  ub - upper bound

   Level: beginner

.seealso QPGetBox(), QPSetOperator(), QPSetRhs(), QPSetEq(), QPAddEq(), QPSetIneq(), QPSetInitialVector(), QPSSolve()
@*/
PetscErrorCode QPSetBox(QP qp, IS is, Vec lb, Vec ub)
{
  QPC qpc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (lb) {
    PetscValidHeaderSpecific(lb, VEC_CLASSID, 2);
    PetscCheckSameComm(qp, 1, lb, 2);
  }
  if (ub) {
    PetscValidHeaderSpecific(ub, VEC_CLASSID, 3);
    PetscCheckSameComm(qp, 1, ub, 3);
  }

  if (lb || ub) {
    PetscCall(QPCCreateBox(PetscObjectComm((PetscObject)qp), is, lb, ub, &qpc));
    PetscCall(QPSetQPC(qp, qpc));
    PetscCall(QPCDestroy(&qpc));
  }

  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetBox"
/*@
   QPGetBox - Get the box constraints.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
+  is - index set of constrained variables; NULL means all unknowns are constrained
.  lb - lower bound
-  ub - upper bound

   Level: advanced

.seealso QPSetBox(), QPGetOperator(), QPGetRhs(), QPGetEq(), QPGetIneq(), QPGetSolutionVector()
@*/
PetscErrorCode QPGetBox(QP qp, IS *is, Vec *lb, Vec *ub)
{
  QPC       qpc;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(QPGetQPC(qp, &qpc));
  if (qpc) {
    PetscCall(PetscObjectTypeCompare((PetscObject)qpc, QPCBOX, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)qp), PETSC_ERR_SUP, "QPC type %s", ((PetscObject)qp->qpc)->type_name);
    PetscCall(QPCBoxGet(qpc, lb, ub));
    if (is) PetscCall(QPCGetIS(qpc, is));
  } else {
    if (is) *is = NULL;
    if (lb) *lb = NULL;
    if (ub) *ub = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetEqMultiplier"
PetscErrorCode QPSetEqMultiplier(QP qp, Vec lambda_E)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (lambda_E == qp->lambda_E) PetscFunctionReturn(PETSC_SUCCESS);
  if (lambda_E) {
    PetscValidHeaderSpecific(lambda_E, VEC_CLASSID, 2);
    PetscCheckSameComm(qp, 1, lambda_E, 2);
    PetscCall(PetscObjectReference((PetscObject)lambda_E));
  }
  PetscCall(VecDestroy(&qp->lambda_E));
  PetscCall(VecDestroy(&qp->lambda));
  PetscCall(VecDestroy(&qp->Bt_lambda));
  qp->lambda_E = lambda_E;
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetIneqMultiplier"
PetscErrorCode QPSetIneqMultiplier(QP qp, Vec lambda_I)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (lambda_I == qp->lambda_I) PetscFunctionReturn(PETSC_SUCCESS);
  if (lambda_I) {
    PetscValidHeaderSpecific(lambda_I, VEC_CLASSID, 2);
    PetscCheckSameComm(qp, 1, lambda_I, 2);
    PetscCall(PetscObjectReference((PetscObject)lambda_I));
  }
  PetscCall(VecDestroy(&qp->lambda_I));
  PetscCall(VecDestroy(&qp->lambda));
  PetscCall(VecDestroy(&qp->Bt_lambda));
  qp->lambda_I = lambda_I;
  if (qp->changeListener) PetscCall((*qp->changeListener)(qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetInitialVector"
/*@
   QPSetInitialVector - Sets the inital guess of the solution.

   Collective on QP

   Input Parameter:
.  qp - the QP
-  x  - initial guess

   Level: beginner

.seealso QPGetSolutionVector(), QPSetOperator(), QPSetRhs(), QPSetEq(), QPAddEq(), QPSetIneq(), QPSetBox(), QPSSolve()
@*/
PetscErrorCode QPSetInitialVector(QP qp, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (x) {
    PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
    PetscCheckSameComm(qp, 1, x, 2);
  }

  PetscCall(VecDestroy(&qp->x));
  qp->x = x;

  if (x) {
    PetscCall(PetscObjectReference((PetscObject)x));
    if (PermonDebugEnabled) {
      PetscReal norm;
      PetscCall(VecNorm(x, NORM_2, &norm));
      PetscCall(PermonDebug1("||x|| = %0.2e\n", norm));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetSolutionVector"
/*@
   QPGetSolutionVector - Get the solution vector.

   Collective on QP

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  x - solution vector

   Level: beginner

.seealso QPGetSolutionVector(), QPGetOperator(), QPGetRhs(), QPGetEq(), QPGetIneq(), QPGetBox(), QPSSolve()
@*/
PetscErrorCode QPGetSolutionVector(QP qp, Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(x, 2);
  PetscCall(QPInitializeInitialVector_Private(qp));
  *x = qp->x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetWorkVector"
/*@
   QPSetWorkVector - Set work vector.

   Collective on QP

   Input Parameter:
+  qp    - the QP
-  xwork - work vector

   Level: developer
@*/
PetscErrorCode QPSetWorkVector(QP qp, Vec xwork)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (xwork == qp->xwork) PetscFunctionReturn(PETSC_SUCCESS);
  if (xwork) {
    PetscValidHeaderSpecific(xwork, VEC_CLASSID, 2);
    PetscCheckSameComm(qp, 1, xwork, 2);
    PetscCall(PetscObjectReference((PetscObject)xwork));
  }
  PetscCall(VecDestroy(&qp->xwork));
  qp->xwork = xwork;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetVecs"
/*@
   QPGetVecs - Get vector(s) compatible with the QP operator matrix, i.e. with the same
     parallel layout

   Collective on QP

   Input Parameter:
.  qp - the QP

   Output Parameter:
+   right - (optional) vector that the matrix can be multiplied against
-   left - (optional) vector that the matrix vector product can be stored in

   Notes: These are new vectors which are not owned by the QP, they should be destroyed by VecDestroy() when no longer needed

   Level: intermediate

.seealso MatCreateVecs()
@*/
PetscErrorCode QPGetVecs(QP qp, Vec *right, Vec *left)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidType(qp, 1);
  if (qp->A) {
    PetscCall(MatCreateVecs(qp->A, right, left));
  } else {
    SETERRQ(((PetscObject)qp)->comm, PETSC_ERR_ORDER, "system operator not set yet");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetChangeListener"
PetscErrorCode QPSetChangeListener(QP qp, PetscErrorCode (*f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  qp->changeListener = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChangeListener"
PetscErrorCode QPGetChangeListener(QP qp, PetscErrorCode (**f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  *f = qp->changeListener;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetChangeListenerContext"
PetscErrorCode QPSetChangeListenerContext(QP qp, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  qp->changeListenerCtx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChangeListenerContext"
PetscErrorCode QPGetChangeListenerContext(QP qp, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  *(void **)ctx = qp->changeListenerCtx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChild"
/*@
   QPGetChild - Get QP child within QP chain.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  child - QP child

   Level: advanced
@*/
PetscErrorCode QPGetChild(QP qp, QP *child)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(child, 2);
  *child = qp->child;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetParent"
/*@
   QPGetParent - Get QP parent within QP chain.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  parent - QP parent

   Level: advanced
@*/
PetscErrorCode QPGetParent(QP qp, QP *parent)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(parent, 2);
  *parent = qp->parent;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetPostSolve"
/*@
   QPGetPostSolve - Get QP post solve function.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  f - post solve function

   Level: developer

.seealso QPChainPostSolve()
@*/
PetscErrorCode QPGetPostSolve(QP qp, PetscErrorCode (**f)(QP, QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  *f = qp->postSolve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetTransform"
/*@
   QPGetTransform - Get QP transform which derived this QP.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  f - transform function

   Level: developer
@*/
PetscErrorCode QPGetTransform(QP qp, PetscErrorCode (**f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  *f = qp->transform;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetQPPF"
/*@
   QPGetQPPF - Get QPPF associated with QP.

   Not Collective

   Input Parameter:
.  qp - the QP

   Output Parameter:
.  pf - the QPPF

   Level: developer
@*/
PetscErrorCode QPGetQPPF(QP qp, QPPF *pf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(pf, 2);
  if (!qp->pf) {
    PetscCall(QPPFCreate(PetscObjectComm((PetscObject)qp), &qp->pf));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)qp->pf, (PetscObject)qp, 1));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)qp->pf, ((PetscObject)qp)->prefix));
    //TODO dirty that we call it unconditionally
    PetscCall(QPPFSetFromOptions(qp->pf));
  }
  *pf = qp->pf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetQPPF"
/*@
   QPSetQPPF - Set QPPF into QP.

   Not Collective, but the QP and QPPF objects must live on the same MPI_Comm

   Input Parameter:
+  qp - the QP
-  pf - the QPPF

   Level: developer
@*/
PetscErrorCode QPSetQPPF(QP qp, QPPF pf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (pf) {
    PetscValidHeaderSpecific(pf, QPPF_CLASSID, 2);
    PetscCheckSameComm(qp, 1, pf, 2);
    PetscCall(PetscObjectReference((PetscObject)pf));
  }
  PetscCall(QPPFDestroy(&qp->pf));
  qp->pf = pf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPIsSolved"
/*@
   QPIsSolved - Is the QP solved, i.e. is the vector returned by QPGetSolutionVector() the sought-after minimizer?
     Set to PETSC_TRUE if QPSSolve() converges (last QP in chain) or the post-solve function has been called.

   Not Collective

   Input Parameter:
.  qp  - the QP

   Output Parameter
.  flg - true if solved

   Level: beginner

.seealso QPGetSolutionVector(), QPSSolve(), QPChainPostSolve()
@*/
PetscErrorCode QPIsSolved(QP qp, PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = qp->solved;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetQPC"
/*
   QPSetQPC - add constraints to QP problem

   Collective on QP

   Parameters:
+  qp - quadratic programming problem
-  qpc - constraints

   Level: intermediate
*/
PetscErrorCode QPSetQPC(QP qp, QPC qpc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  if (qpc) PetscValidHeaderSpecific(qpc, QPC_CLASSID, 2);
  PetscCall(QPCDestroy(&qp->qpc));
  qp->qpc = qpc;
  PetscCall(PetscObjectReference((PetscObject)qpc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetQPC"
/*
   QPGetQPC - return constraints from QP

   Not Collective

   Parameters:
   + qp - quadratic programming problem
   - qpc - pointer to constraints

   Level: developer
*/
PetscErrorCode QPGetQPC(QP qp, QPC *qpc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscAssertPointer(qpc, 2);
  *qpc = qp->qpc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetOptionsPrefix"
/*@
   QPSetOptionsPrefix - Sets the prefix used for searching for all
   QP options in the database.

   Logically Collective on QP

   Input Parameters:
+  qp - the QP
-  prefix - the prefix string to prepend to all QP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: QPAppendOptionsPrefix(), QPGetOptionsPrefix()
@*/
PetscErrorCode QPSetOptionsPrefix(QP qp, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)qp, prefix));
  if (qp->pf) {
    PetscCall(QPGetQPPF(qp, &qp->pf));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)qp->pf, prefix));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPAppendOptionsPrefix"
/*@
   QPAppendOptionsPrefix - Appends to the prefix used for searching for all
   QP options in the database.

   Logically Collective on QP

   Input Parameters:
+  QP - the QP
-  prefix - the prefix string to prepend to all QP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: QPSetOptionsPrefix(), QPGetOptionsPrefix()
@*/
PetscErrorCode QPAppendOptionsPrefix(QP qp, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)qp, prefix));
  if (qp->pf) {
    PetscCall(QPGetQPPF(qp, &qp->pf));
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)qp->pf, prefix));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetOptionsPrefix"
/*@
   QPGetOptionsPrefix - Gets the prefix used for searching for all
   QP options in the database.

   Not Collective

   Input Parameters:
.  qp - the Krylov context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Level: advanced

.seealso: QPSetOptionsPrefix(), QPAppendOptionsPrefix()
@*/
PetscErrorCode QPGetOptionsPrefix(QP qp, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)qp, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetFromOptions_Private"
static PetscErrorCode QPSetFromOptions_Private(QP qp)
{
  PetscFunctionBegin;
  if (!qp->setfromoptionscalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscObjectOptionsBegin((PetscObject)qp);
  if (!qp->pc) PetscCall(QPGetPC(qp, &qp->pc));

  if (qp->pf) { PetscCall(PermonPetscObjectInheritPrefixIfNotSet((PetscObject)qp->pf, (PetscObject)qp, NULL)); }
  PetscCall(PermonPetscObjectInheritPrefixIfNotSet((PetscObject)qp->pc, (PetscObject)qp, NULL));

  if (qp->pf) { PetscCall(QPPFSetFromOptions(qp->pf)); }
  PetscCall(PCSetFromOptions(qp->pc));

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetFromOptions"
/*@
   QPSetFromOptions - Sets QP options from the options database.

   Collective on QP

   Input Parameters:
.  qp - the QP context

   Options Database Keys:
.  -qp_view            - view information about QP
.  -qp_chain_view      - view information about all QPs in the chain
.  -qp_chain_view_kkt  - view how well are satisfied KKT conditions for each QP in the chain
-  -qp_chain_view_qppf - view information about all QPPFs in the chain

   Notes:
   To see all options, run your program with the -help option
   or consult Users-Manual: ch_qp

   Level: beginner
@*/
PetscErrorCode QPSetFromOptions(QP qp)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)qp);

  /* options processed elsewhere */
  PetscCall(PetscOptionsName("-qp_view", "print the QP info at the end of a QPSSolve call", "QPView", &flg));

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)qp, PetscOptionsObject));
  PetscOptionsEnd();
  qp->setfromoptionscalled++;
  PetscFunctionReturn(PETSC_SUCCESS);
}
