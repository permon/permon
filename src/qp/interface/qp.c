
#include <permon/private/qpimpl.h>

PetscClassId  QP_CLASSID;

const char *QPScaleTypes[]={"none","norm2","multiplicity","QPScalType","QPPF_",0};

static PetscErrorCode QPSetFromOptions_Private(QP qp);

#define QPView_PrintObjectLoaded(v,obj,description) PetscViewerASCIIPrintf(v,"    %-32s %-16s %c\n", description, obj?((PetscObject)obj)->name:"", obj?'Y':'N')
#define QPView_Vec(v,x,iname) \
{\
  PetscReal max,min,norm;\
  PetscInt  imax,imin;\
  const char *name = (iname);\
  CHKERRQ(VecNorm(x,NORM_2,&norm));\
  CHKERRQ(VecMax(x,&imax,&max));\
  CHKERRQ(VecMin(x,&imin,&min));\
  CHKERRQ(PetscViewerASCIIPrintf(v, "||%2s|| = %.8e    max(%2s) = %.2e = %2s(%d)    min(%2s) = %.2e = %2s(%d)    %x\n",name,norm,name,max,name,imax,name,min,name,imin,x));\
}

#undef __FUNCT__
#define __FUNCT__ "QPInitializeInitialVector_Private"
static PetscErrorCode QPInitializeInitialVector_Private(QP qp)
{
  Vec xp, xc;

  PetscFunctionBegin;
  if (qp->x) PetscFunctionReturn(0);
  if (!qp->parent) {
    /* if no initial guess exists, just set it to a zero vector */
    CHKERRQ(MatCreateVecs(qp->A,&qp->x,NULL));
    CHKERRQ(VecZeroEntries(qp->x)); // TODO: is it in the feasible set?
    PetscFunctionReturn(0);
  }
  CHKERRQ(QPGetSolutionVector(qp->parent, &xp));
  if (xp) {
    CHKERRQ(VecDuplicate(xp, &xc));
    CHKERRQ(VecCopy(xp, xc));
    CHKERRQ(QPSetInitialVector(qp, xc));
    CHKERRQ(VecDestroy(&xc));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPAddChild"
PetscErrorCode QPAddChild(QP qp, QPDuplicateOption opt, QP *newchild)
{
  QP child;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(newchild,2);
  CHKERRQ(QPDuplicate(qp,opt,&child));
  qp->child = child;
  child->parent = qp;
  child->id = qp->id+1;
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  if (newchild) *newchild = child;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPRemoveChild"
PetscErrorCode QPRemoveChild(QP qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (!qp->child) PetscFunctionReturn(0);
  qp->child->parent = NULL;
  qp->child->postSolve = NULL;
  CHKERRQ(QPDestroy(&qp->child));
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
  QP               qp;

  PetscFunctionBegin;
  PetscValidPointer(qp_new,2);
  *qp_new = 0;
  CHKERRQ(QPInitializePackage());

  CHKERRQ(PetscHeaderCreate(qp,QP_CLASSID,"QP","Quadratic Programming Problem","QP",comm,QPDestroy,QPView));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)qp,"QP"));
  qp->A            = NULL;
  qp->R            = NULL;
  qp->b            = NULL;
  qp->b_plus       = PETSC_FALSE;
  qp->BE           = NULL;
  qp->BE_nest_count= 0;
  qp->cE           = NULL;
  qp->lambda_E     = NULL;
  qp->Bt_lambda    = NULL;
  qp->BI           = NULL;
  qp->cI           = NULL;
  qp->lambda_I     = NULL;
  qp->B            = NULL;
  qp->c            = NULL;
  qp->lambda       = NULL;
  qp->x            = NULL;
  qp->xwork        = NULL;
  qp->pc           = NULL;
  qp->pf           = NULL;
  qp->child        = NULL;
  qp->parent       = NULL;
  qp->solved       = PETSC_FALSE;
  qp->setupcalled  = PETSC_FALSE;
  qp->setfromoptionscalled = 0;

  /* set the initial constraints */
  qp->qpc                    = NULL;
  
  qp->changeListener         = NULL;
  qp->changeListenerCtx      = NULL;
  qp->postSolve              = NULL;
  qp->postSolveCtx           = NULL;
  qp->postSolveCtxDestroy    = NULL;

  qp->id = 0;
  qp->transform = NULL;
  qp->transform_name[0] = 0;

  /* initialize preconditioner */
  CHKERRQ(QPGetPC(qp,&qp->pc));

  *qp_new = qp;
  PetscFunctionReturn(0);
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
PetscErrorCode QPDuplicate(QP qp1,QPDuplicateOption opt,QP *qp2)
{
  QP qp2_;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp1,QP_CLASSID,1);
  PetscValidPointer(qp2,2);
  CHKERRQ(QPCreate(PetscObjectComm((PetscObject)qp1),&qp2_));

  if (opt==QP_DUPLICATE_DO_NOT_COPY) {
    *qp2 = qp2_;
    PetscFunctionReturn(0);
  }

  CHKERRQ(QPSetQPC(qp2_,qp1->qpc));
  CHKERRQ(QPSetEq(qp2_,qp1->BE,qp1->cE));
  CHKERRQ(QPSetEqMultiplier(qp2_,qp1->lambda_E));
  qp2_->BE_nest_count = qp1->BE_nest_count;
  CHKERRQ(QPSetIneq(qp2_,qp1->BI,qp1->cI));
  CHKERRQ(QPSetIneqMultiplier(qp2_,qp1->lambda_I));
  CHKERRQ(QPSetInitialVector(qp2_,qp1->x));
  CHKERRQ(QPSetOperator(qp2_,qp1->A));
  CHKERRQ(QPSetOperatorNullSpace(qp2_,qp1->R));
  if (qp1->pc) CHKERRQ(QPSetPC(qp2_,qp1->pc));
  CHKERRQ(QPSetQPPF(qp2_,qp1->pf));
  CHKERRQ(QPSetRhs(qp2_,qp1->b));
  CHKERRQ(QPSetWorkVector(qp2_,qp1->xwork));

  if (qp1->lambda)    CHKERRQ(PetscObjectReference((PetscObject)(qp2_->lambda = qp1->lambda)));
  if (qp1->Bt_lambda) CHKERRQ(PetscObjectReference((PetscObject)(qp2_->Bt_lambda = qp1->Bt_lambda)));
  *qp2 = qp2_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCompareEqMultiplierWithLeastSquare"
PetscErrorCode QPCompareEqMultiplierWithLeastSquare(QP qp,PetscReal *norm)
{
  Vec BEt_lambda = NULL;
  Vec BEt_lambda_LS;
  QP qp2;

  PetscFunctionBegin;
  if (!qp->BE) PetscFunctionReturn(0);

  CHKERRQ(QPCompute_BEt_lambda(qp,&BEt_lambda));

  CHKERRQ(QPDuplicate(qp,QP_DUPLICATE_COPY_POINTERS,&qp2));
  CHKERRQ(QPSetEqMultiplier(qp2,NULL));
  CHKERRQ(QPComputeMissingEqMultiplier(qp2));
  CHKERRQ(QPCompute_BEt_lambda(qp2,&BEt_lambda_LS));

  /* compare lambda_E with least-square lambda_E */
  CHKERRQ(VecAXPY(BEt_lambda_LS,-1.0,BEt_lambda));
  CHKERRQ(VecNorm(BEt_lambda_LS,NORM_2,norm));

  CHKERRQ(VecDestroy(&BEt_lambda));
  CHKERRQ(VecDestroy(&BEt_lambda_LS));
  CHKERRQ(QPDestroy(&qp2));
  PetscFunctionReturn(0);
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
PetscErrorCode QPViewKKT(QP qp,PetscViewer v)
{
  PetscReal   normb=0.0,norm=0.0,dot=0.0;
  Vec         x,b,cE,cI,r,o,t;
  Mat         A,BE,BI;
  PetscBool   flg=PETSC_FALSE,compare_lambda_E=PETSC_FALSE,notavail;
  MPI_Comm    comm;
  char        *kkt_name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&flg));
  if (!flg) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for QP",((PetscObject)v)->type_name);

  CHKERRQ(PetscOptionsGetBool(((PetscObject)qp)->options,NULL,"-qp_view_kkt_compare_lambda_E",&compare_lambda_E,NULL));

  CHKERRQ(PetscObjectName((PetscObject)qp));
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)qp,v));
  CHKERRQ(PetscViewerASCIIPrintf(v, "  #%d in chain, derived by %s\n",qp->id,qp->transform_name));
  if (!qp->solved) {
    CHKERRQ(PetscViewerASCIIPrintf(v, "*** WARNING: QP is not solved. ***\n"));
  }

  CHKERRQ(QPGetOperator(qp, &A));
  CHKERRQ(QPGetRhs(qp, &b));
  CHKERRQ(QPGetEq(qp, &BE, &cE));
  CHKERRQ(QPGetIneq(qp, &BI, &cI));
  CHKERRQ(QPGetSolutionVector(qp, &x));
  CHKERRQ(VecNorm(b,NORM_2,&normb));

  QPView_Vec(v,x,"x");
  QPView_Vec(v,b,"b");
  if (cE) QPView_Vec(v,cE,"cE");
  if (BE && !cE) CHKERRQ(PetscViewerASCIIPrintf(v, "||cE|| = 0.00e-00    max(cE) = 0.00e-00 = cE(0)    min(cE) = 0.00e-00 = cE(0)\n"));
  if (cI) QPView_Vec(v,cI,"cI");
  if (BI && !cI) CHKERRQ(PetscViewerASCIIPrintf(v, "||cI|| = 0.00e-00    max(cI) = 0.00e-00 = cI(0)    min(cI) = 0.00e-00 = cI(0)\n"));
  
  CHKERRQ(VecDuplicate(b, &r));
  CHKERRQ(QPComputeLagrangianGradient(qp,x,r,&kkt_name));
  CHKERRQ(VecIsInvalidated(r,&notavail));

  if (!notavail) {
    if (compare_lambda_E) {
      CHKERRQ(QPCompareEqMultiplierWithLeastSquare(qp,&norm));
      CHKERRQ(PetscViewerASCIIPrintf(v,"||BE'*lambda_E - BE'*lambda_E_LS|| = %.4e\n",norm));
    }
    CHKERRQ(VecNorm(r, NORM_2, &norm));
    CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||%s|| = %.2e    rO/||b|| = %.2e\n",kkt_name,norm,norm/normb));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||%s|| not available\n",kkt_name));
  }
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(PetscFree(kkt_name));

  if (BE) {
    if (BE->ops->mult) {
      CHKERRQ(MatCreateVecs(BE, NULL, &r));
      CHKERRQ(MatMult(BE, x, r));
      if (cE) CHKERRQ(VecAXPY(r, -1.0, cE));
      CHKERRQ(VecNorm(r, NORM_2, &norm));
      if (cE) {
        CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||BE*x-cE||          = %.2e    r/||b|| = %.2e\n",norm,norm/normb));
      } else {
        CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||BE*x||             = %.2e    r/||b|| = %.2e\n",norm,norm/normb));
      }
      CHKERRQ(VecDestroy(&r));
    } else {
      if (cE) {
        CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||BE*x-cE||         not available\n"));
      } else {
        Vec t = qp->xwork;
        CHKERRQ(QPPFApplyGtG(qp->pf,x,t));                    /* BEtBEx = BE'*BE*x */
        CHKERRQ(VecDot(x,t,&norm));                           /* norm = x'*BE'*BE*x */
        norm = PetscSqrtReal(norm);
        CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||BE*x||             = %.2e    r/||b|| = %.2e\n",norm,norm/normb));
      }
    }
  }

  if (BI) {
    CHKERRQ(VecDuplicate(qp->lambda_I,&r));
    CHKERRQ(VecDuplicate(r,&o));
    CHKERRQ(VecDuplicate(r,&t));

    CHKERRQ(VecSet(o,0.0));                                   /* o = zeros(size(r)) */

    /* r = BI*x - cI */
    CHKERRQ(MatMult(BI, x, r));                               /* r = BI*x         */
    if (cI) CHKERRQ(VecAXPY(r, -1.0, cI));                    /* r = r - cI       */

    /* rI = norm(max(BI*x-cI,0)) */
    CHKERRQ(VecPointwiseMax(t,r,o));                          /* t = max(r,o)     */
    CHKERRQ(VecNorm(t,NORM_2,&norm));                         /* norm = norm(t)     */
    if (cI) {
      CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||max(BI*x-cI,0)||   = %.2e    r/||b|| = %.2e\n",norm,norm/normb));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||max(BI*x,0)||      = %.2e    r/||b|| = %.2e\n",norm,norm/normb));
    }

    /* lambda >= o  =>  examine min(lambda,o) */
    CHKERRQ(VecSet(o,0.0));                                   /* o = zeros(size(r)) */
    CHKERRQ(VecPointwiseMin(t,qp->lambda_I,o));
    CHKERRQ(VecNorm(t,NORM_2,&norm));                         /* norm = ||min(lambda,o)|| */
    CHKERRQ(PetscViewerASCIIPrintf(v,"r = ||min(lambda_I,0)||  = %.2e    r/||b|| = %.2e\n",norm,norm/normb));

    /* lambda'*(BI*x-cI) = 0 */
    CHKERRQ(VecDot(qp->lambda_I,r,&dot));
    dot = PetscAbs(dot);
    if (cI) {
      CHKERRQ(PetscViewerASCIIPrintf(v,"r = |lambda_I'*(BI*x-cI)|= %.2e    r/||b|| = %.2e\n",dot,dot/normb));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(v,"r = |lambda_I'*(BI*x)|= %.2e       r/||b|| = %.2e\n",dot,dot/normb));
    }

    CHKERRQ(VecDestroy(&o));
    CHKERRQ(VecDestroy(&r));
    CHKERRQ(VecDestroy(&t));
  }

  if (qp->qpc) CHKERRQ(QPCViewKKT(qp->qpc,x,normb,v));
  PetscFunctionReturn(0);
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
PetscErrorCode QPView(QP qp,PetscViewer v)
{
  Vec         b,cE,cI,lb,ub;
  Mat         A,R,BE,BI;
  QPC         qpc;
  PetscBool   iascii;
  MPI_Comm    comm;
  QP          childDual;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for QP",((PetscObject)v)->type_name);
  CHKERRQ(PetscObjectName((PetscObject)qp));
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)qp,v));
  CHKERRQ(PetscViewerASCIIPrintf(v, "#%d in chain, derived by %s\n",qp->id,qp->transform_name));

  CHKERRQ(QPGetOperator(qp, &A));
  CHKERRQ(QPGetOperatorNullSpace(qp, &R));
  CHKERRQ(QPGetRhs(qp, &b));
  CHKERRQ(QPGetBox(qp, NULL, &lb, &ub));
  CHKERRQ(QPGetEq(qp, &BE, &cE));
  CHKERRQ(QPGetIneq(qp, &BI, &cI));
  CHKERRQ(QPGetQPC(qp, &qpc));
  CHKERRQ(QPChainFind(qp, (PetscErrorCode(*)(QP))QPTDualize, &childDual));

  CHKERRQ(PetscViewerASCIIPrintf(v,"  LOADED OBJECTS:\n"));
  CHKERRQ(PetscViewerASCIIPrintf(v,"    %-32s %-16s %s\n", "what", "name", "present"));
  CHKERRQ(QPView_PrintObjectLoaded(v, A,   "Hessian"));
  CHKERRQ(QPView_PrintObjectLoaded(v, b,   "linear term (right-hand-side)"));
  CHKERRQ(QPView_PrintObjectLoaded(v, R,   "R (kernel of K)"));
  CHKERRQ(QPView_PrintObjectLoaded(v, lb,  "lower bounds"));
  CHKERRQ(QPView_PrintObjectLoaded(v, ub,  "upper bounds"));
  CHKERRQ(QPView_PrintObjectLoaded(v, BE,  "linear eq. constraint matrix"));
  CHKERRQ(QPView_PrintObjectLoaded(v, cE,  "linear eq. constraint RHS"));
  CHKERRQ(QPView_PrintObjectLoaded(v, BI,  "linear ineq. constraint"));
  CHKERRQ(QPView_PrintObjectLoaded(v, cI,  "linear ineq. constraint RHS"));
  CHKERRQ(QPView_PrintObjectLoaded(v, qpc, "QPC"));

  if (A)   CHKERRQ(MatPrintInfo(A));
  if (b)   CHKERRQ(VecPrintInfo(b));
  if (R)   CHKERRQ(MatPrintInfo(R));
  if (lb)  CHKERRQ(VecPrintInfo(lb));
  if (ub)  CHKERRQ(VecPrintInfo(ub));
  if (BE)  CHKERRQ(MatPrintInfo(BE));
  if (cE)  CHKERRQ(VecPrintInfo(cE));
  if (BI)  CHKERRQ(MatPrintInfo(BI));
  if (cI)  CHKERRQ(VecPrintInfo(cI));
  PetscFunctionReturn(0);
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
  CHKERRQ(QPDestroy( &qp->child));

  CHKERRQ(MatDestroy(&qp->A));
  CHKERRQ(MatDestroy(&qp->R));
  CHKERRQ(MatDestroy(&qp->BE));
  CHKERRQ(MatDestroy(&qp->BI));
  CHKERRQ(MatDestroy(&qp->B));

  CHKERRQ(VecDestroy(&qp->b));
  CHKERRQ(VecDestroy(&qp->x));
  CHKERRQ(VecDestroy(&qp->xwork));
  CHKERRQ(VecDestroy(&qp->cE));
  CHKERRQ(VecDestroy(&qp->lambda_E));
  CHKERRQ(VecDestroy(&qp->Bt_lambda));
  CHKERRQ(VecDestroy(&qp->cI));
  CHKERRQ(VecDestroy(&qp->lambda_I));
  CHKERRQ(VecDestroy(&qp->c));
  CHKERRQ(VecDestroy(&qp->lambda));
  
  CHKERRQ(PCDestroy( &qp->pc));

  CHKERRQ(QPCDestroy(&qp->qpc));
  
  CHKERRQ(QPPFDestroy(&qp->pf));
  qp->setupcalled = PETSC_FALSE;
  qp->solved = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetUpInnerObjects"
PetscErrorCode QPSetUpInnerObjects(QP qp)
{
  MPI_Comm comm;
  PetscInt i;
  Mat Bs[2];
  IS rows[2];
  Vec cs[2],c[2];
  Vec lambdas[2];

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!qp->A) SETERRQ(comm,PETSC_ERR_ORDER,"Hessian must be set before " __FUNCT__);
  if (!qp->b) SETERRQ(comm,PETSC_ERR_ORDER,"linear term must be set before " __FUNCT__);

  FllopTraceBegin;
  CHKERRQ(PetscInfo(qp,"setup inner objects for QP #%d\n",qp->id));

  if (!qp->pc) CHKERRQ(QPGetPC(qp,&qp->pc));
  CHKERRQ(PCSetOperators(qp->pc,qp->A,qp->A));

  CHKERRQ(QPInitializeInitialVector_Private(qp));

  if (!qp->xwork) CHKERRQ(VecDuplicate(qp->x,&qp->xwork));

  if (qp->BE && !qp->lambda_E) {
    CHKERRQ(MatCreateVecs(qp->BE,NULL,&qp->lambda_E));
    CHKERRQ(VecInvalidate(qp->lambda_E));
  }
  if (!qp->BE) {
    CHKERRQ(VecDestroy(&qp->lambda_E));
  }

  if (qp->BI && !qp->lambda_I) {
    CHKERRQ(MatCreateVecs(qp->BI,NULL,&qp->lambda_I));
    CHKERRQ(VecInvalidate(qp->lambda_I));
  }
  if (!qp->BI) {
    CHKERRQ(VecDestroy(&qp->lambda_I));
  }

  if ((qp->BE || qp->BI) && !qp->B)
  {
  CHKERRQ(VecDestroy(&qp->c));

  if (qp->BE && !qp->BI) {
    CHKERRQ(PetscObjectReference((PetscObject)(qp->B       = qp->BE)));
    CHKERRQ(PetscObjectReference((PetscObject)(qp->lambda  = qp->lambda_E)));
    CHKERRQ(PetscObjectReference((PetscObject)(qp->c       = qp->cE)));
  } else if (!qp->BE && qp->BI) {
    CHKERRQ(PetscObjectReference((PetscObject)(qp->B       = qp->BI)));
    CHKERRQ(PetscObjectReference((PetscObject)(qp->lambda  = qp->lambda_I)));
    CHKERRQ(PetscObjectReference((PetscObject)(qp->c       = qp->cI)));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)(Bs[0]       = qp->BE)));
    CHKERRQ(PetscObjectReference((PetscObject)(lambdas[0]  = qp->lambda_E)));
    if (qp->cE) {
      CHKERRQ(PetscObjectReference((PetscObject)(cs[0]     = qp->cE)));
    } else {
      CHKERRQ(MatCreateVecs(Bs[0],NULL,&cs[0]));
      CHKERRQ(VecSet(cs[0],0.0));
    }
    
    CHKERRQ(PetscObjectReference((PetscObject)(Bs[1]       = qp->BI)));
    CHKERRQ(PetscObjectReference((PetscObject)(lambdas[1]  = qp->lambda_I)));
    if (qp->cI) {
      CHKERRQ(PetscObjectReference((PetscObject)(cs[1]     = qp->cI)));
    } else {
      CHKERRQ(MatCreateVecs(Bs[1],NULL,&cs[1]));
      CHKERRQ(VecSet(cs[1],0.0));
    }
    
    CHKERRQ(MatCreateNestPermon(comm,2,NULL,1,NULL,Bs,&qp->B));
    CHKERRQ(MatCreateVecs(qp->B,NULL,&qp->c));
    CHKERRQ(PetscObjectSetName((PetscObject)qp->B,"B"));
    CHKERRQ(PetscObjectSetName((PetscObject)qp->c,"c"));
    
    /* copy cE,cI to c */
    CHKERRQ(MatNestGetISs(qp->B,rows,NULL));
    for (i=0; i<2; i++) {
      CHKERRQ(VecGetSubVector(qp->c,rows[i],&c[i]));
      CHKERRQ(VecCopy(cs[i],c[i]));
      CHKERRQ(VecRestoreSubVector(qp->c,rows[i],&c[i]));
    }
    
    for (i=0; i<2; i++) {
      CHKERRQ(MatDestroy(&Bs[i]));
      CHKERRQ(VecDestroy(&cs[i]));
      CHKERRQ(VecDestroy(&lambdas[i]));
    }
  }
  }

  if (qp->B && !qp->lambda) {
    CHKERRQ(MatCreateVecs(qp->B,NULL,&qp->lambda));
    CHKERRQ(PetscObjectSetName((PetscObject)qp->lambda,"lambda"));
    CHKERRQ(VecInvalidate(qp->lambda));
  }

  if (qp->B && !qp->Bt_lambda) {
    CHKERRQ(MatCreateVecs(qp->B,&qp->Bt_lambda,NULL));
    CHKERRQ(PetscObjectSetName((PetscObject)qp->lambda,"Bt_lambda"));
    CHKERRQ(VecInvalidate(qp->Bt_lambda));
  }

  if (!qp->B) {
    CHKERRQ(VecDestroy(&qp->lambda));
    CHKERRQ(VecDestroy(&qp->Bt_lambda));
  }
  PetscFunctionReturnI(0);
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

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (qp->setupcalled) PetscFunctionReturn(0);

  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!qp->A) SETERRQ(comm,PETSC_ERR_ORDER,"Hessian must be set before " __FUNCT__);
  if (!qp->b) SETERRQ(comm,PETSC_ERR_ORDER,"linear term must be set before " __FUNCT__);

  FllopTraceBegin;
  CHKERRQ(PetscInfo(qp,"setup QP #%d\n",qp->id));
  CHKERRQ(QPSetUpInnerObjects(qp));
  CHKERRQ(QPSetFromOptions_Private(qp));
  CHKERRQ(PCSetUp(qp->pc));
  qp->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCompute_BEt_lambda"
PetscErrorCode QPCompute_BEt_lambda(QP qp,Vec *BEt_lambda)
{
  PetscBool   flg=PETSC_FALSE;

  PetscFunctionBegin;
  *BEt_lambda = NULL;
  if (!qp->BE) PetscFunctionReturn(0);

  if (!qp->BI) {
    CHKERRQ(VecIsInvalidated(qp->Bt_lambda, &flg));
    if (!flg) {
      CHKERRQ(VecDuplicate(qp->Bt_lambda, BEt_lambda));
      CHKERRQ(VecCopy(qp->Bt_lambda, *BEt_lambda));                               /* BEt_lambda = Bt_lambda */
      PetscFunctionReturn(0);
    }

    CHKERRQ(VecIsInvalidated(qp->lambda,&flg));
    if (!flg && qp->B->ops->multtranspose) {
      CHKERRQ(MatCreateVecs(qp->B, BEt_lambda, NULL));
      CHKERRQ(MatMultTranspose(qp->B, qp->lambda, *BEt_lambda));                  /* BEt_lambda = B'*lambda */
      PetscFunctionReturn(0);
    }
  }

  CHKERRQ(VecIsInvalidated(qp->lambda_E,&flg));
  if (!flg || !qp->BE->ops->multtranspose) {
    CHKERRQ(MatCreateVecs(qp->BE, BEt_lambda, NULL));
    CHKERRQ(MatMultTranspose(qp->BE, qp->lambda_E, *BEt_lambda));                 /* Bt_lambda = BE'*lambda_E */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeLagrangianGradient"
PetscErrorCode QPComputeLagrangianGradient(QP qp, Vec x, Vec r, char *kkt_name_[])
{
  Vec         b,cE,cI,Bt_lambda=NULL;
  Mat         A,BE,BI;
  PetscBool   flg=PETSC_FALSE,avail=PETSC_TRUE;
  char        kkt_name[256]="A*x - b";

  PetscFunctionBegin;
  CHKERRQ(QPSetUp(qp));
  CHKERRQ(QPGetOperator(qp, &A));
  CHKERRQ(QPGetRhs(qp, &b));
  CHKERRQ(QPGetEq(qp, &BE, &cE));
  CHKERRQ(QPGetIneq(qp, &BI, &cI));

  CHKERRQ(MatMult(A, x, r));
  CHKERRQ(VecAXPY(r, -1.0, b));                                                   /* r = A*x - b */

  /* TODO: replace with QPC function */
  {
    Vec lb,ub;
    Vec llb,lub;
    IS  is;
    QPC qpc;

    CHKERRQ(QPGetBox(qp,&is,&lb,&ub));
    CHKERRQ(QPGetQPC(qp,&qpc));
    if (qpc) CHKERRQ(QPCBoxGetMultipliers(qpc,&llb,&lub));
    if (lb) {
      if (is) {
        CHKERRQ(VecISAXPY(r,is,-1.0,llb));
      } else {
        CHKERRQ(VecAXPY(r,-1.0,llb));
      }
    } if (ub) {
      if (is) {
        CHKERRQ(VecISAXPY(r,is,1.0,lub));
      } else {
        CHKERRQ(VecAXPY(r,1.0,lub));
      }
    }
  }

  CHKERRQ(VecDuplicate(r,&Bt_lambda));
  if (qp->B) {
    CHKERRQ(VecIsInvalidated(qp->Bt_lambda,&flg));
    if (!flg) {
      CHKERRQ(VecCopy(qp->Bt_lambda,Bt_lambda));                                  /* Bt_lambda = (B'*lambda) */
      if (kkt_name_) CHKERRQ(PetscStrcat(kkt_name," + (B'*lambda)"));
      goto endif;
    }

    CHKERRQ(VecIsInvalidated(qp->lambda,&flg));
    if (!flg && qp->B->ops->multtranspose) {
      CHKERRQ(MatMultTranspose(qp->B, qp->lambda, Bt_lambda));                    /* Bt_lambda = B'*lambda */
      if (kkt_name_) CHKERRQ(PetscStrcat(kkt_name," + B'*lambda"));
      goto endif;
    }

    if (qp->BE) {
      if (kkt_name_) CHKERRQ(PetscStrcat(kkt_name," + BE'*lambda_E"));
    }
    if (qp->BI) {
      if (kkt_name_) CHKERRQ(PetscStrcat(kkt_name," + BI'*lambda_I"));
    }

    if (qp->BE) {
      CHKERRQ(VecIsInvalidated(qp->lambda_E,&flg));
      if (flg || !qp->BE->ops->multtranspose) {
        avail = PETSC_FALSE;
        goto endif;
      }
      CHKERRQ(MatMultTranspose(BE, qp->lambda_E, Bt_lambda));                     /* Bt_lambda = BE'*lambda_E */
    }

    if (qp->BI) {
      CHKERRQ(VecIsInvalidated(qp->lambda_I,&flg));
      if (flg || !qp->BI->ops->multtransposeadd) {
        avail = PETSC_FALSE;
        goto endif;
      }
      CHKERRQ(MatMultTransposeAdd(BI, qp->lambda_I, Bt_lambda, Bt_lambda));       /* Bt_lambda = Bt_lambda + BI'*lambda_I */
    }
  }
  endif:
  if (avail) {
    Vec lb,ub;

    /* TODO: replace with QPC function */
    CHKERRQ(QPGetBox(qp,NULL,&lb,&ub));
    CHKERRQ(VecAXPY(r,1.0,Bt_lambda));                                            /* r = r + Bt_lambda */
    if (lb) {
      if (kkt_name_) CHKERRQ(PetscStrcat(kkt_name," - lambda_lb"));
    }
    if (ub) {
      if (kkt_name_) CHKERRQ(PetscStrcat(kkt_name," + lambda_ub"));
    }
  } else {
    CHKERRQ(VecInvalidate(r));
  }
  if (kkt_name_) CHKERRQ(PetscStrallocpy(kkt_name,kkt_name_));
  CHKERRQ(VecDestroy(&Bt_lambda));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeMissingEqMultiplier"
PetscErrorCode QPComputeMissingEqMultiplier(QP qp)
{
  Vec r = qp->xwork;
  PetscBool flg;
  const char *name;
  QP qp_I;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(QPSetUp(qp));

  if (!qp->BE) PetscFunctionReturn(0);
  CHKERRQ(VecIsInvalidated(qp->lambda_E,&flg));
  if (!flg) PetscFunctionReturn(0);
  if (qp->Bt_lambda) {
    CHKERRQ(VecIsInvalidated(qp->Bt_lambda,&flg));
    if (!flg) PetscFunctionReturn(0);
  }

  CHKERRQ(QPDuplicate(qp,QP_DUPLICATE_COPY_POINTERS,&qp_I));
  CHKERRQ(QPSetEq(qp_I,NULL,NULL));
  CHKERRQ(QPComputeLagrangianGradient(qp_I,qp->x,r,NULL));
  CHKERRQ(QPDestroy(&qp_I));

  //TODO Should we add qp->Bt_lambda_E ?
  if (qp->BE == qp->B) {
    CHKERRQ(VecCopy(r,qp->Bt_lambda));
    CHKERRQ(VecScale(qp->Bt_lambda,-1.0));
  } else {
    CHKERRQ(QPPFApplyHalfQ(qp->pf,r,qp->lambda_E));
    CHKERRQ(VecScale(qp->lambda_E,-1.0));                                               /* lambda_E_LS = -(BE*BE')\\BE*r */
  }

  if (FllopDebugEnabled) {
    PetscReal norm;
    if (qp->BE->ops->multtranspose) {
      CHKERRQ(MatMultTransposeAdd(qp->BE,qp->lambda_E,r,r));
    } else {
      CHKERRQ(VecAXPY(r,1.0,qp->Bt_lambda));
    }
    CHKERRQ(VecNorm(r,NORM_2,&norm));
    CHKERRQ(FllopDebug1("||r||=%.2e\n",norm));
  }

  CHKERRQ(PetscObjectGetName((PetscObject)qp,&name));
  CHKERRQ(PetscInfo(qp,"missing eq. con. multiplier computed for QP Object %s (#%d in chain, derived by %s)\n",name,qp->id,qp->transform_name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeMissingBoxMultipliers"
PetscErrorCode QPComputeMissingBoxMultipliers(QP qp)
{
  Vec lb,ub;
  Vec llb,lub;
  IS  is;
  Vec r = qp->xwork;
  PetscBool flg=PETSC_FALSE,flg2=PETSC_FALSE;
  QP qp_E;
  QPC qpc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(QPSetUp(qp));

  CHKERRQ(QPGetBox(qp,&is,&lb,&ub));
  CHKERRQ(QPGetQPC(qp,&qpc));
  if (qpc) CHKERRQ(QPCBoxGetMultipliers(qpc,&llb,&lub));

  if (lb) {
    CHKERRQ(VecIsInvalidated(llb,&flg));
  }
  if (ub) {
    CHKERRQ(VecIsInvalidated(lub,&flg2));
  }
  if (!lb && !ub) PetscFunctionReturn(0);
  if (!flg && !flg2) PetscFunctionReturn(0);

  /* currently cannot handle this situation, leave multipliers untouched */
  if (qp->BI) PetscFunctionReturn(0);

  CHKERRQ(QPDuplicate(qp,QP_DUPLICATE_COPY_POINTERS,&qp_E));
  CHKERRQ(QPCDestroy(&qp_E->qpc));
  CHKERRQ(QPComputeLagrangianGradient(qp_E,qp->x,r,NULL));
  CHKERRQ(QPDestroy(&qp_E));

  if (lb) {
    if (is) {
      CHKERRQ(VecISCopy(r,is,SCATTER_REVERSE,llb));
    } else {
      CHKERRQ(VecCopy(r,llb));
    }
  }
  if (ub) {
    if (is) {
      CHKERRQ(VecISCopy(r,is,SCATTER_REVERSE,lub));
    } else {
      CHKERRQ(VecCopy(r,lub));
    }
    CHKERRQ(VecScale(lub,-1.0));
  }
  if (lb && ub) {
    Vec lambdawork;
    CHKERRQ(VecDuplicate(llb,&lambdawork));
    CHKERRQ(VecZeroEntries(lambdawork));
    CHKERRQ(VecPointwiseMax(llb,llb,lambdawork));
    CHKERRQ(VecPointwiseMax(lub,lub,lambdawork));
    CHKERRQ(VecDestroy(&lambdawork));
  }

  {
    const char *name_qp;
    CHKERRQ(PetscObjectGetName((PetscObject)qp,&name_qp));
    CHKERRQ(PetscInfo(qp,"missing lower bound con. multiplier computed for QP Object %s (#%d in chain, derived by %s)\n",name_qp,qp->id,qp->transform_name));
  }
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidRealPointer(f,3);
  if (!qp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");
  CHKERRQ(MatMult(qp->A,x,qp->xwork));
  CHKERRQ(VecAYPX(qp->xwork,-0.5,qp->b));
  CHKERRQ(VecDot(x,qp->xwork,f));
  *f = -*f;
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (!qp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");
  CHKERRQ(MatMult(qp->A,x,g));
  CHKERRQ(VecAXPY(g,-1.0,qp->b));
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidRealPointer(f,4);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (!qp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");

  CHKERRQ(VecWAXPY(qp->xwork,-1.0,qp->b,g));
  CHKERRQ(VecDot(x,qp->xwork,f));
  *f /= 2.0;
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  if (f) PetscValidRealPointer(f,4);
  if (g) PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  if (!qp->setupcalled) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ORDER,"QPSetUp must be called first.");

  if (!g) {
    CHKERRQ(QPComputeObjective(qp,x,f));
    PetscFunctionReturn(0);
  }

  CHKERRQ(QPComputeObjectiveGradient(qp,x,g));
  if (!f) PetscFunctionReturn(0);

  CHKERRQ(QPComputeObjectiveFromGradient(qp,x,g,f));
  PetscFunctionReturn(0);
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
  if (!*qp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*qp,QP_CLASSID,1);
  if (--((PetscObject)(*qp))->refct > 0) {
    *qp = 0;
    PetscFunctionReturn(0);
  }
  if ((*qp)->postSolveCtxDestroy) CHKERRQ((*qp)->postSolveCtxDestroy((*qp)->postSolveCtx));
  CHKERRQ(QPReset(*qp));
  CHKERRQ(QPPFDestroy(&(*qp)->pf));
  CHKERRQ(PetscHeaderDestroy(qp));
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(qp,1,A,2);
  if (A == qp->A) PetscFunctionReturn(0);

  CHKERRQ(MatDestroy(&qp->A));
  qp->A = A;
  CHKERRQ(PetscObjectReference((PetscObject)A));

  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  PetscCheckSameComm(qp,1,pc,2);
  if (pc == qp->pc) PetscFunctionReturn(0);
  CHKERRQ(PCDestroy(&qp->pc));
  qp->pc = pc;
  CHKERRQ(PetscObjectReference((PetscObject)pc));
  CHKERRQ(PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pc));
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetOperator(QP qp,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(A,2);
  *A = qp->A;
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetPC(QP qp,PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(pc,2);
  if (!qp->pc) {
    CHKERRQ(PCCreate(PetscObjectComm((PetscObject)qp),&qp->pc));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)qp->pc,(PetscObject)qp,0));
    CHKERRQ(PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pc));
    CHKERRQ(PCSetType(qp->pc,PCNONE));
    CHKERRQ(PCSetOperators(qp->pc,qp->A,qp->A));
  }
  *pc = qp->pc;
  PetscFunctionReturn(0);
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
PetscErrorCode QPSetOperatorNullSpace(QP qp,Mat R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (R) PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  if (R == qp->R) PetscFunctionReturn(0);
  if (R) {
#if defined(PETSC_USE_DEBUG)
    CHKERRQ(MatCheckNullSpace(qp->A, R, PETSC_SMALL));
#endif
    CHKERRQ(PetscObjectReference((PetscObject)R));
  }
  CHKERRQ(MatDestroy(&qp->R));
  qp->R = R;
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetOperatorNullSpace(QP qp,Mat *R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(R,2);
  *R = qp->R;
  PetscFunctionReturn(0);
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
PetscErrorCode QPSetRhs(QP qp,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscCheckSameComm(qp,1,b,2);
  if (b == qp->b && qp->b_plus == PETSC_FALSE) PetscFunctionReturn(0);

  CHKERRQ(VecDestroy(&qp->b));
  qp->b = b;
  CHKERRQ(PetscObjectReference((PetscObject)b));
  qp->b_plus = PETSC_FALSE;

  if (FllopDebugEnabled) {
    PetscReal norm;
    CHKERRQ(VecNorm(b,NORM_2,&norm));
    CHKERRQ(FllopDebug1("||b|| = %0.2e\n", norm));
  }

  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
PetscErrorCode QPSetRhsPlus(QP qp,Vec b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscCheckSameComm(qp,1,b,2);
  if (b == qp->b && qp->b_plus == PETSC_TRUE) PetscFunctionReturn(0);
  
  CHKERRQ(VecDuplicate(b,&qp->b));
  CHKERRQ(VecCopy(b,qp->b));
  CHKERRQ(VecScale(qp->b,-1.0));
  qp->b_plus = PETSC_TRUE;
  
  if (FllopDebugEnabled) {
    PetscReal norm;
    CHKERRQ(VecNorm(b,NORM_2,&norm));
    CHKERRQ(FllopDebug1("||b|| = %0.2e\n", norm));
  }

  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetRhs(QP qp,Vec *b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(b,2);
  *b = qp->b;
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  if (Bineq) {
    PetscValidHeaderSpecific(Bineq,MAT_CLASSID,2);
    PetscCheckSameComm(qp,1,Bineq,2);
  }

  if (Bineq != qp->BI) {
    if (Bineq) CHKERRQ(PetscObjectReference((PetscObject)Bineq));
    CHKERRQ(MatDestroy(&qp->BI));
    CHKERRQ(MatDestroy(&qp->B));
    CHKERRQ(QPSetIneqMultiplier(qp,NULL));
    qp->BI = Bineq;
    change = PETSC_TRUE;
  }

  if (cineq) {
    if (!Bineq) {
      cineq = NULL;
      CHKERRQ(PetscInfo(qp, "null inequality constraint matrix specified, the constraint RHS vector will be ignored\n"));
    } else {
      PetscValidHeaderSpecific(cineq,VEC_CLASSID,3);
      PetscCheckSameComm(qp,1,cineq,3);
      CHKERRQ(VecNorm(cineq,NORM_2,&norm));
      CHKERRQ(FllopDebug1("||cineq|| = %0.2e\n", norm));
      if (norm < PETSC_MACHINE_EPSILON) {
        CHKERRQ(PetscInfo(qp, "zero inequality constraint RHS vector detected\n"));
        cineq = NULL;
      }
    }
  } else if (Bineq) {
    CHKERRQ(PetscInfo(qp, "null inequality constraint RHS vector handled as zero vector\n"));
  }

  if (cineq != qp->cI) {
    if (cineq) CHKERRQ(PetscObjectReference((PetscObject)cineq));
    CHKERRQ(VecDestroy(&qp->cI));
    CHKERRQ(VecDestroy(&qp->c));
    qp->cI = cineq;
    change = PETSC_TRUE;
  }

  if (!Bineq) {
    CHKERRQ(VecDestroy(&qp->lambda_I));
  }

  if (change) {
    if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  }
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (Bineq) {
    PetscValidPointer(Bineq, 2);
    *Bineq = qp->BI;
  }
  if (cineq) {
    PetscValidPointer(cineq, 3);
    *cineq = qp->cI;
  }
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);

  if (Beq) {
    PetscValidHeaderSpecific(Beq,MAT_CLASSID,2);
    PetscCheckSameComm(qp,1,Beq,2);
  }

  if (Beq != qp->BE) {
    if (Beq) {
      CHKERRQ(QPGetQPPF(qp, &qp->pf));
      CHKERRQ(QPPFSetG(qp->pf, Beq));
      CHKERRQ(PetscObjectReference((PetscObject)Beq));
    }
    CHKERRQ(MatDestroy(&qp->BE));
    CHKERRQ(MatDestroy(&qp->B));
    CHKERRQ(QPSetEqMultiplier(qp,NULL));
    qp->BE = Beq;
    qp->BE_nest_count = 0;
    change = PETSC_TRUE;
  }

  if (ceq) {
    if (!Beq) {
      ceq = NULL;
      CHKERRQ(PetscInfo(qp, "null equality constraint matrix specified, the constraint RHS vector will be ignored\n"));
    } else {
      PetscValidHeaderSpecific(ceq,VEC_CLASSID,3);
      PetscCheckSameComm(qp,1,ceq,3);
      CHKERRQ(VecNorm(ceq,NORM_2,&norm));
      CHKERRQ(FllopDebug1("||ceq|| = %0.2e\n", norm));
      if (norm < PETSC_MACHINE_EPSILON) {
        CHKERRQ(PetscInfo(qp, "zero equality constraint RHS vector detected\n"));
        ceq = NULL;
      }
    }
  } else if (Beq) {
    CHKERRQ(PetscInfo(qp, "null equality constraint RHS vector handled as zero vector\n"));
  }

  if (ceq != qp->cE) {
    if (ceq) CHKERRQ(PetscObjectReference((PetscObject)ceq));
    CHKERRQ(VecDestroy(&qp->cE));
    CHKERRQ(VecDestroy(&qp->c));
    qp->cE = ceq;
    change = PETSC_TRUE;
  }

  if (!Beq) {
    CHKERRQ(VecDestroy(&qp->lambda_E));
  }

  if (change) {
    if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  }
  PetscFunctionReturn(0);
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
  Mat *subBE;
  PetscInt M, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidHeaderSpecific(Beq, MAT_CLASSID, 2);
  PetscCheckSameComm(qp, 1, Beq, 2);

  /* handle BE that has been set by QPSetEq */
  if (qp->BE && !qp->BE_nest_count) {
    Mat BE_orig=qp->BE;
    Vec cE_orig=qp->cE;

    CHKERRQ(PetscObjectReference((PetscObject)BE_orig));
    if (cE_orig) CHKERRQ(PetscObjectReference((PetscObject)cE_orig));
    CHKERRQ(QPSetEq(qp,NULL,NULL));
    CHKERRQ(QPAddEq(qp,BE_orig,cE_orig));
    CHKERRQ(MatDestroy(&BE_orig));
    CHKERRQ(VecDestroy(&cE_orig));
    PERMON_ASSERT(qp->BE_nest_count==1,"qp->BE_nest_count==1");
  }
  
  M = qp->BE_nest_count++;

  CHKERRQ(PetscMalloc((M+1)*sizeof(Mat), &subBE));   //Mat subBE[M+1];
  for (i = 0; i<M; i++) {
    CHKERRQ(MatNestGetSubMat(qp->BE, i, 0, &subBE[i]));
    CHKERRQ(PetscObjectReference((PetscObject) subBE[i]));
  }
  subBE[M] = Beq;

  CHKERRQ(MatDestroy(&qp->BE));
  CHKERRQ(MatCreateNestPermon(PetscObjectComm((PetscObject) qp), M+1, NULL, 1, NULL, subBE, &qp->BE));
  CHKERRQ(PetscObjectSetName((PetscObject)qp->BE,"BE"));

  CHKERRQ(QPGetQPPF(qp, &qp->pf));
  CHKERRQ(QPPFSetG(qp->pf, qp->BE));

  if (ceq) {
    PetscValidHeaderSpecific(ceq, VEC_CLASSID, 3);
    PetscCheckSameComm(qp, 1, ceq, 3);
    CHKERRQ(VecNorm(ceq, NORM_2, &norm));
    CHKERRQ(FllopDebug1("||ceq|| = %0.2e\n", norm));
    if (norm < PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscInfo(qp, "zero equality constraint RHS vector detected\n"));
      ceq = NULL;
    }
  } else {
    CHKERRQ(PetscInfo(qp, "null equality constraint RHS vector handled as zero vector\n"));
  }

  if (ceq || qp->cE) {
    Vec *subCE;
    CHKERRQ(PetscMalloc((M+1)*sizeof(Vec), &subCE));
    if (qp->cE) {
      for (i = 0; i<M; i++) {
        CHKERRQ(VecNestGetSubVec(qp->cE, i, &subCE[i]));
        CHKERRQ(PetscObjectReference((PetscObject) subCE[i]));
      }
      CHKERRQ(VecDestroy(&qp->cE));
    } else {
      for (i = 0; i<M; i++) {
        CHKERRQ(MatCreateVecs(subBE[i], NULL, &subCE[i]));
        CHKERRQ(VecSet(subCE[i], 0.0));
      }
    }
    if (!ceq) {
      CHKERRQ(MatCreateVecs(Beq, NULL, &ceq));
      CHKERRQ(VecSet(ceq, 0.0));
    }
    subCE[M] = ceq;

    CHKERRQ(VecCreateNest(PetscObjectComm((PetscObject) qp), M+1, NULL, subCE, &qp->cE));
    for (i = 0; i<M; i++) {
      CHKERRQ(VecDestroy( &subCE[i]));
    }
    CHKERRQ(PetscFree(subCE));
  }

  for (i = 0; i<M; i++) {
    CHKERRQ(MatDestroy(   &subBE[i]));
  }
  CHKERRQ(PetscFree(subBE));
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetEqMultiplicityScaling"
PetscErrorCode QPGetEqMultiplicityScaling(QP qp, Vec *dE_new, Vec *dI_new)
{
  MPI_Comm comm;
  PetscInt i,ilo,ihi,j,k,ncols;
  PetscScalar multiplicity;
  Mat Bc=NULL, Bd=NULL, Bg=NULL;
  Mat Bct=NULL, Bdt=NULL, Bgt=NULL;
  PetscBool flg, scale_Bd=PETSC_TRUE, scale_Bc=PETSC_TRUE, count_Bd=PETSC_TRUE, count_Bc=PETSC_TRUE;
  Vec dof_multiplicities=NULL, edge_multiplicities_g=NULL, edge_multiplicities_d=NULL, edge_multiplicities_c=NULL;
  const PetscInt *cols;
  const PetscScalar *vals;
  
  PetscFunctionBeginI;
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  //TODO we now assume fully redundant case
  if (!qp->BE_nest_count) {
    Bg = qp->BE;
  } else {
    CHKERRQ(MatNestGetSubMat(qp->BE,0,0,&Bg));
    if (qp->BE_nest_count >= 2) {
      CHKERRQ(MatNestGetSubMat(qp->BE,1,0,&Bd));
    }
  }
  Bc = qp->BI;
  PERMON_ASSERT(Bg,"Bg");
  
  if (!Bc) { scale_Bc = PETSC_FALSE; count_Bc = PETSC_FALSE; }
  if (!Bd) { scale_Bd = PETSC_FALSE; count_Bd = PETSC_FALSE; }
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qp_E_scale_Bd",&scale_Bd,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qp_E_scale_Bc",&scale_Bc,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qp_E_count_Bd",&count_Bd,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qp_E_count_Bc",&count_Bc,NULL));
  //if (scale_Bc && !count_Bc) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_INCOMP,"-qp_E_scale_Bc implies -qp_E_count_Bc");
  //if (scale_Bd && !count_Bd) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_INCOMP,"-qp_E_scale_Bd implies -qp_E_count_Bd");

  CHKERRQ(MatGetOwnershipRangeColumn(Bg,&ilo,&ihi));

  {
    CHKERRQ(MatCreateVecs(Bg,&dof_multiplicities,&edge_multiplicities_g));
    CHKERRQ(VecSet(dof_multiplicities,   1.0));
  }
  if (scale_Bd) {
    CHKERRQ(MatCreateVecs(Bd,NULL,&edge_multiplicities_d));
    CHKERRQ(VecSet(edge_multiplicities_d,1.0));
  }
  if (scale_Bc) {
    CHKERRQ(MatCreateVecs(Bc,NULL,&edge_multiplicities_c));
    CHKERRQ(VecSet(edge_multiplicities_c,1.0));
  }

  {
    CHKERRQ(MatIsImplicitTranspose(Bg,&flg));
    PERMON_ASSERT(flg,"Bg must be implicit transpose");
    CHKERRQ(PermonMatTranspose(Bg,MAT_TRANSPOSE_EXPLICIT,&Bgt));
    for (i=ilo; i<ihi; i++) {
      CHKERRQ(MatGetRow(Bgt,i,&ncols,&cols,&vals));
      k=0;
      for (j=0; j<ncols; j++) {
        if (vals[j]) k++;
      }
      CHKERRQ(MatRestoreRow(Bgt,i,&ncols,&cols,&vals));
      if (k) {
        multiplicity = k+1;
        CHKERRQ(VecSetValue(dof_multiplicities,i,multiplicity,INSERT_VALUES));
      }
    }
  }

  if (count_Bd) {
    CHKERRQ(MatIsImplicitTranspose(Bd,&flg));
    PERMON_ASSERT(flg,"Bd must be implicit transpose");
    CHKERRQ(PermonMatTranspose(Bd,MAT_TRANSPOSE_EXPLICIT,&Bdt));
    for (i=ilo; i<ihi; i++) {
      CHKERRQ(MatGetRow(Bdt,i,&ncols,&cols,&vals));
      k=0;
      for (j=0; j<ncols; j++) {
        if (vals[j]) k++;
        if (k>1) SETERRQ(comm,PETSC_ERR_PLIB,"more than one nonzero in Bd row %d",i);
      }
      CHKERRQ(MatRestoreRow(Bdt,i,&ncols,&cols,&vals));
      if (k) {
        CHKERRQ(VecGetValues(dof_multiplicities,1,&i,&multiplicity));
        multiplicity++;
        CHKERRQ(VecSetValue(dof_multiplicities,i,multiplicity,INSERT_VALUES));
      }
    }
  }

  if (count_Bc) {
    CHKERRQ(MatIsImplicitTranspose(Bc,&flg));
    PERMON_ASSERT(flg,"Bc must be implicit transpose");
    CHKERRQ(PermonMatTranspose(Bc,MAT_TRANSPOSE_EXPLICIT,&Bct));
    for (i=ilo; i<ihi; i++) {
      CHKERRQ(MatGetRow(Bct,i,&ncols,&cols,&vals));
      k=0;
      for (j=0; j<ncols; j++) {
        if (vals[j]) k++;
      }
      CHKERRQ(MatRestoreRow(Bct,i,&ncols,&cols,&vals));
      if (k>1) CHKERRQ(PetscPrintf(comm,"WARNING: more than one nonzero in Bc row %d\n",i));
      if (k) {
        CHKERRQ(VecGetValues(dof_multiplicities,1,&i,&multiplicity));
        multiplicity++;
        CHKERRQ(VecSetValue(dof_multiplicities,i,multiplicity,INSERT_VALUES));
      }
    }
  }

  CHKERRQ(VecAssemblyBegin(dof_multiplicities));
  CHKERRQ(VecAssemblyEnd(dof_multiplicities));
  CHKERRQ(VecSqrtAbs(dof_multiplicities));
  CHKERRQ(VecReciprocal(dof_multiplicities));

  {
    for (i=ilo; i<ihi; i++) {
      CHKERRQ(MatGetRow(Bgt,i,&ncols,&cols,NULL));
      if (ncols) {
        CHKERRQ(VecGetValues(dof_multiplicities,1,&i,&multiplicity));
        for (j=0; j<ncols; j++) {
          CHKERRQ(VecSetValue(edge_multiplicities_g,cols[j],multiplicity,INSERT_VALUES));
        }
      }
      CHKERRQ(MatRestoreRow(Bgt,i,&ncols,&cols,NULL));
    }
    CHKERRQ(VecAssemblyBegin(edge_multiplicities_g));
    CHKERRQ(VecAssemblyEnd(  edge_multiplicities_g));
    CHKERRQ(MatDestroy(&Bgt));
  }

  if (scale_Bd) {
    CHKERRQ(PermonMatTranspose(Bd,MAT_TRANSPOSE_EXPLICIT,&Bdt));
    for (i=ilo; i<ihi; i++) {
      CHKERRQ(MatGetRow(Bdt,i,&ncols,&cols,NULL));
      if (ncols) {
        CHKERRQ(VecGetValues(dof_multiplicities,1,&i,&multiplicity));
        for (j=0; j<ncols; j++) {
          CHKERRQ(VecSetValue(edge_multiplicities_d,cols[j],multiplicity,INSERT_VALUES));
        }
      }
      CHKERRQ(MatRestoreRow(Bdt,i,&ncols,&cols,NULL));
    }
    CHKERRQ(VecAssemblyBegin(edge_multiplicities_d));
    CHKERRQ(VecAssemblyEnd(  edge_multiplicities_d));
    CHKERRQ(MatDestroy(&Bdt));
  }

  if (scale_Bc) {
    CHKERRQ(PermonMatTranspose(Bc,MAT_TRANSPOSE_EXPLICIT,&Bct));
    for (i=ilo; i<ihi; i++) {
      CHKERRQ(MatGetRow(Bct,i,&ncols,&cols,NULL));
      if (ncols) {
        CHKERRQ(VecGetValues(dof_multiplicities,1,&i,&multiplicity));
        for (j=0; j<ncols; j++) {
          CHKERRQ(VecSetValue(edge_multiplicities_c,cols[j],multiplicity,INSERT_VALUES));
        }
      }
      CHKERRQ(MatRestoreRow(Bct,i,&ncols,&cols,NULL));
    }
    CHKERRQ(VecAssemblyBegin(edge_multiplicities_c));
    CHKERRQ(VecAssemblyEnd(  edge_multiplicities_c));
    CHKERRQ(MatDestroy(&Bct));
  }

  if (edge_multiplicities_d) {
    Vec dE_vecs[2]={edge_multiplicities_g,edge_multiplicities_d};
    CHKERRQ(VecCreateNest(PetscObjectComm((PetscObject)qp),2,NULL,dE_vecs,dE_new));
    CHKERRQ(VecDestroy(&edge_multiplicities_d));
    CHKERRQ(VecDestroy(&edge_multiplicities_g));
  } else {
    *dE_new = edge_multiplicities_g;
  }

  *dI_new = edge_multiplicities_c;

  CHKERRQ(VecDestroy(&dof_multiplicities));
  PetscFunctionReturnI(0);
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
    PetscValidPointer(Beq, 2);
    *Beq = qp->BE;
  }
  if (ceq) {
    PetscValidPointer(ceq, 3);
    *ceq = qp->cE;
  }
  PetscFunctionReturn(0);
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
    PetscValidHeaderSpecific(lb,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lb,2);
  }
  if (ub) {
    PetscValidHeaderSpecific(ub,VEC_CLASSID,3);
    PetscCheckSameComm(qp,1,ub,3);
  }

  if (lb || ub) {
    CHKERRQ(QPCCreateBox(PetscObjectComm((PetscObject)qp),is,lb,ub,&qpc));
    CHKERRQ(QPSetQPC(qp, qpc));
    CHKERRQ(QPCDestroy(&qpc));
  }

  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
  QPC qpc;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(QPGetQPC(qp,&qpc));
  if (qpc) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)qpc,QPCBOX,&flg));
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qp->qpc)->type_name);
    CHKERRQ(QPCBoxGet(qpc,lb,ub));
    if (is) CHKERRQ(QPCGetIS(qpc,is));
  } else {
    if (is) *is = NULL;
    if (lb) *lb = NULL;
    if (ub) *ub = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetEqMultiplier"
PetscErrorCode QPSetEqMultiplier(QP qp, Vec lambda_E)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lambda_E == qp->lambda_E) PetscFunctionReturn(0);
  if (lambda_E) {
    PetscValidHeaderSpecific(lambda_E,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lambda_E,2);
    CHKERRQ(PetscObjectReference((PetscObject)lambda_E));
  }
  CHKERRQ(VecDestroy(&qp->lambda_E));
  CHKERRQ(VecDestroy(&qp->lambda));
  CHKERRQ(VecDestroy(&qp->Bt_lambda));
  qp->lambda_E = lambda_E;
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetIneqMultiplier"
PetscErrorCode QPSetIneqMultiplier(QP qp, Vec lambda_I)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (lambda_I == qp->lambda_I) PetscFunctionReturn(0);
  if (lambda_I) {
    PetscValidHeaderSpecific(lambda_I,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,lambda_I,2);
    CHKERRQ(PetscObjectReference((PetscObject)lambda_I));
  }
  CHKERRQ(VecDestroy(&qp->lambda_I));
  CHKERRQ(VecDestroy(&qp->lambda));
  CHKERRQ(VecDestroy(&qp->Bt_lambda));
  qp->lambda_I = lambda_I;
  if (qp->changeListener) CHKERRQ((*qp->changeListener)(qp));
  PetscFunctionReturn(0);
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
PetscErrorCode QPSetInitialVector(QP qp,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (x) {
    PetscValidHeaderSpecific(x,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,x,2);
  }

  CHKERRQ(VecDestroy(&qp->x));
  qp->x = x;

  if (x) {
    CHKERRQ(PetscObjectReference((PetscObject)x));
    if (FllopDebugEnabled) {
      PetscReal norm;
      CHKERRQ(VecNorm(x,NORM_2,&norm));
      CHKERRQ(FllopDebug1("||x|| = %0.2e\n", norm));
    }
  }
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetSolutionVector(QP qp,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(x,2);
  CHKERRQ(QPInitializeInitialVector_Private(qp));
  *x = qp->x;
  PetscFunctionReturn(0);
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
PetscErrorCode QPSetWorkVector(QP qp,Vec xwork)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (xwork == qp->xwork) PetscFunctionReturn(0);
  if (xwork) {
    PetscValidHeaderSpecific(xwork,VEC_CLASSID,2);
    PetscCheckSameComm(qp,1,xwork,2);
    CHKERRQ(PetscObjectReference((PetscObject)xwork));
  }
  CHKERRQ(VecDestroy(&qp->xwork));
  qp->xwork = xwork;
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetVecs(QP qp,Vec *right,Vec *left)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidType(qp,1);
  if (qp->A) {
    CHKERRQ(MatCreateVecs(qp->A,right,left));
  } else {
    SETERRQ(((PetscObject)qp)->comm,PETSC_ERR_ORDER,"system operator not set yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetChangeListener"
PetscErrorCode QPSetChangeListener(QP qp,PetscErrorCode (*f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  qp->changeListener = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChangeListener"
PetscErrorCode QPGetChangeListener(QP qp,PetscErrorCode (**f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  *f = qp->changeListener;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetChangeListenerContext"
PetscErrorCode QPSetChangeListenerContext(QP qp,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  qp->changeListenerCtx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPGetChangeListenerContext"
PetscErrorCode QPGetChangeListenerContext(QP qp,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(ctx,2);
  *(void**)ctx = qp->changeListenerCtx;
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetChild(QP qp,QP *child)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(child,2);
  *child = qp->child;
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetParent(QP qp,QP *parent)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(parent,2);
  *parent = qp->parent;
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetPostSolve(QP qp, PetscErrorCode (**f)(QP,QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  *f = qp->postSolve;
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetTransform(QP qp,PetscErrorCode(**f)(QP))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  *f = qp->transform;
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(pf,2);
  if (!qp->pf) {
    CHKERRQ(QPPFCreate(PetscObjectComm((PetscObject)qp),&qp->pf));
    CHKERRQ(PetscLogObjectParent((PetscObject)qp,(PetscObject)qp->pf));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)qp->pf,(PetscObject)qp,1));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)qp->pf,((PetscObject)qp)->prefix));
    //TODO dirty that we call it unconditionally
    CHKERRQ(QPPFSetFromOptions(qp->pf));
  }
  *pf = qp->pf;
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (pf) {
    PetscValidHeaderSpecific(pf,QPPF_CLASSID,2);
    PetscCheckSameComm(qp,1,pf,2);
    CHKERRQ(PetscObjectReference((PetscObject)pf));
  }
  CHKERRQ(QPPFDestroy(&qp->pf));
  qp->pf = pf;
  PetscFunctionReturn(0);
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
PetscErrorCode QPIsSolved(QP qp,PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = qp->solved;
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  if (qpc) PetscValidHeaderSpecific(qpc,QPC_CLASSID,2);
  CHKERRQ(QPCDestroy(&qp->qpc));
  qp->qpc = qpc;
  CHKERRQ(PetscObjectReference((PetscObject)qpc));
  PetscFunctionReturn(0);
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

  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(qpc,2);
  *qpc = qp->qpc;
  PetscFunctionReturn(0);
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
PetscErrorCode QPSetOptionsPrefix(QP qp,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)qp,prefix));
  if (qp->pf) {
    CHKERRQ(QPGetQPPF(qp, &qp->pf));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)qp->pf,prefix));
  }
  PetscFunctionReturn(0);
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
PetscErrorCode QPAppendOptionsPrefix(QP qp,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)qp,prefix));
  if (qp->pf) {
    CHKERRQ(QPGetQPPF(qp, &qp->pf));
    CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)qp->pf,prefix));
  }
  PetscFunctionReturn(0);
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
PetscErrorCode QPGetOptionsPrefix(QP qp,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)qp,prefix));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPSetFromOptions_Private"
static PetscErrorCode QPSetFromOptions_Private(QP qp)
{
  PetscFunctionBegin;
  if (!qp->setfromoptionscalled) PetscFunctionReturn(0);

  PetscObjectOptionsBegin((PetscObject)qp);
  if (!qp->pc) CHKERRQ(QPGetPC(qp,&qp->pc));

  if (qp->pf) {
    CHKERRQ(FllopPetscObjectInheritPrefixIfNotSet((PetscObject)qp->pf,(PetscObject)qp,NULL));
  }
  CHKERRQ(FllopPetscObjectInheritPrefixIfNotSet((PetscObject)qp->pc,(PetscObject)qp,NULL));

  if (qp->pf) {
    CHKERRQ(QPPFSetFromOptions(qp->pf));
  }
  CHKERRQ(PCSetFromOptions(qp->pc));

  PetscOptionsEnd();
  PetscFunctionReturn(0);
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
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscObjectOptionsBegin((PetscObject)qp);

  /* options processed elsewhere */
  CHKERRQ(PetscOptionsName("-qp_view","print the QP info at the end of a QPSSolve call","QPView",&flg));

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  CHKERRQ(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)qp));
  PetscOptionsEnd();
  qp->setfromoptionscalled++;
  PetscFunctionReturn(0);
}
