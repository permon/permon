
#include <permon/private/qpimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPChainAdd"
/*@
   QPChainAdd - Append QP into the QP chain.

   Collective on QP

   Input Parameters:
+  qp - a QP that is member of the chain to which we want to append
.  opt - either QP_DUPLICATE_DO_NOT_COPY or QP_DUPLICATE_COPY_POINTERS
-  newchild - a pointer to QP that should be appended

   Level: developer
@*/
PetscErrorCode QPChainAdd(QP qp, QPDuplicateOption opt, QP *newchild)
{
  QP last;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(newchild,2);
  CHKERRQ(QPChainGetLast(qp, &last));
  CHKERRQ(QPAddChild(last,opt,newchild));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainPop"
/*@
   QPChainPop - Delete the last QP of the chain.

   Input Parameters:
.  qp - a QP that is member of the chain from which we want to delete the last QP

   Level: developer
@*/
PetscErrorCode QPChainPop(QP qp)
{
  QP last;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(QPChainGetLast(qp, &last));
  if (last->parent) CHKERRQ(QPRemoveChild(last->parent));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainFind"
/*@
   QPChainFind - Find QP in the chain.

   Not Collective

   Input Parameters:
+  qp - a QP that is member of the chain; specifies the starting point of search
-  transform - a tranformation function that was used to create QP we search for

   Output Parameters:
.  child - first QP in chain that was transformed by transform; NULL if such QP does not exist

   Level: advanced
@*/
PetscErrorCode QPChainFind(QP qp,PetscErrorCode(*transform)(QP),QP *child)
{
  QP cchild;
  PetscErrorCode (*ctransform)(QP);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp, QP_CLASSID, 1);
  PetscValidPointer(child, 3);

  *child = NULL;
  ctransform = NULL;

  CHKERRQ(QPGetChild(qp, &cchild));
  while (cchild) {
    CHKERRQ(QPGetTransform(cchild, &ctransform));
    if (ctransform == transform) {
      *child = cchild;
      break;
    }
    CHKERRQ(QPGetChild(cchild, &cchild));
  };
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainGetLast"
/*@
   QPChainGetLast - Get last QP in the chain.

   Not Collective

   Input Parameters:
.  qp - a QP that is member of the chain

   Output Parameters:
.  child - last QP in the chain

   Level: advanced
@*/
PetscErrorCode QPChainGetLast(QP qp,QP *last)
{
  QP tchild;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(last,2);

  tchild = qp;
  do {
    qp = tchild;
    CHKERRQ(QPGetChild(qp,&tchild));
  } while (tchild);

  *last = qp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainSetUp"
/*@
   QPChainSetUp - Calls QPSetUP() on QP and its descendants in the chain.

   Collective on QP

   Input Parameters:
.  qp - a first QP to call QPSetUp() on

   Level: developer
@*/
PetscErrorCode QPChainSetUp(QP qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  do {
    CHKERRQ(QPSetUp(qp));
    CHKERRQ(QPGetChild(qp,&qp));
  } while (qp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainSetFromOptions"
/*@
   QPChainSetFromOptions - Calls QPSetFromOptions() on QP and its descendants in the chain.

   Collective on QP

   Input Parameters:
.  qp - a first QP to call QPSetFromOptions on

   Level: advanced
@*/
PetscErrorCode QPChainSetFromOptions(QP qp)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscOptionsBegin(PetscObjectComm((PetscObject)qp),NULL,"QP chain options","QP");  /* options processed elsewhere */
  CHKERRQ(PetscOptionsName("-qp_chain_view","print the info about all QPs in the chain at the end of a QPSSolve call","QPChainView",&flg));
  CHKERRQ(PetscOptionsName("-qp_chain_view_kkt","print detailed post-solve KKT satisfaction information","QPChainViewKKT",&flg));
  CHKERRQ(PetscOptionsName("-qp_chain_view_qppf","print info about QPPF instances in the QP chain","QPChainViewQPPF",&flg));

  do {
    CHKERRQ(QPSetFromOptions(qp));
    CHKERRQ(QPGetChild(qp,&qp));
  } while (qp);
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainPostSolve"
/*@
   QPChainPostSolve - Apply post solve functions and optionally view. 

   Collective on QP

   Input Parameter:
.  qp   - the QP
  
   Options Database Keys:
+  -qp_view            - view information about QP
.  -qp_chain_view      - view information about all QPs in the chain
.  -qp_chain_view_qppf - view information about all QPPFs in the chain
-  -qp_chain_view_kkt  - view how well are KKT conditions satisfied for each QP in the chain 

   Notes:
   This is called automatically by QPSPostSolve.

   Level: developer

.seealso QPSSolve(), QPSPostSolve(), QPChainView(), QPChainViewKKT(), QPChainViewQPPF()
@*/
PetscErrorCode QPChainPostSolve(QP qp)
{
  PetscErrorCode (*postSolve)(QP,QP);
  QP parent, cqp;
  PetscBool flg, solved, view, first;
  PetscViewer v=NULL;
  PetscViewerFormat format;
  MPI_Comm comm;
  const char *prefix;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)qp,&prefix));

  CHKERRQ(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_view",&v,&format,&view));
  if (view & !PetscPreLoadingOn) {
    CHKERRQ(PetscViewerPushFormat(v,format));
    CHKERRQ(QPView(qp,v));
    CHKERRQ(PetscViewerPopFormat(v));
    CHKERRQ(PetscViewerDestroy(&v));
  }

  CHKERRQ(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_chain_view",&v,&format,&view));
  if (view & !PetscPreLoadingOn) {
    CHKERRQ(PetscViewerPushFormat(v,format));
    CHKERRQ(QPChainView(qp,v));
    CHKERRQ(PetscViewerPopFormat(v));
    CHKERRQ(PetscViewerDestroy(&v));
  }

  CHKERRQ(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_chain_view_qppf",&v,&format,&view));
  if (view & !PetscPreLoadingOn) {
    CHKERRQ(PetscViewerPushFormat(v,format));
    CHKERRQ(QPChainViewQPPF(qp,v));
    CHKERRQ(PetscViewerPopFormat(v));
    CHKERRQ(PetscViewerDestroy(&v));
  }

  CHKERRQ(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_chain_view_kkt",&v,&format,&view));
  view = (PetscBool)(view && !PetscPreLoadingOn);
  if (view) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&flg));
    if (!flg) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);
    CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  }

  CHKERRQ(QPChainGetLast(qp,&cqp));
  solved = cqp->solved;
  first = PETSC_TRUE;
  while (1) {
    CHKERRQ(QPComputeMissingBoxMultipliers(cqp));
    CHKERRQ(QPComputeMissingEqMultiplier(cqp));
    parent = cqp->parent;
    postSolve = cqp->postSolve;
    if (postSolve) CHKERRQ((*postSolve)(cqp,parent));

    if (view) {
      if (first) {
        first = PETSC_FALSE;
      } else {
        CHKERRQ(PetscViewerASCIIPrintf(v, "-------------------\n"));
      }
      CHKERRQ(QPViewKKT(cqp,v));
    }

    if (!parent) break;
    parent->solved = solved;
    if (cqp == qp) break;
    cqp = parent;
  }

  if (view) {
    CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
    CHKERRQ(PetscViewerDestroy(&v));
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainViewKKT"
/*@
   QPChainViewKKT - Calls QPViewKKT() on each QP in the chain.

   Collective on QP

   Input Parameters:
+  qp - a QP specifying the chain
-  v - viewer

   Level: advanced
@*/
PetscErrorCode QPChainViewKKT(QP qp, PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool iascii, first=PETSC_TRUE;
  QP cqp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  CHKERRQ(QPChainGetLast(qp,&cqp));
  while (1) {
    if (first) {
      first = PETSC_FALSE;
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(v, "-------------------\n"));
    }
    CHKERRQ(QPViewKKT(cqp,v));
    cqp = cqp->parent;
    if (cqp == qp) break;
  }
  CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainView"
/*@
   QPChainView - Calls QPView() on each QP in the chain.

   Collective on QP

   Input Parameters:
+  qp - a QP specifying the chain
-  v - viewer

   Level: advanced
@*/
PetscErrorCode QPChainView(QP qp, PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  CHKERRQ(PetscViewerASCIIPrintf(v,__FUNCT__" output follows\n"));
  CHKERRQ(QPView(qp, v));
  CHKERRQ(QPGetChild(qp, &qp));
  while (qp) {
    CHKERRQ(PetscViewerASCIIPrintf(v, "-------------------\n"));
    CHKERRQ(QPView(qp, v));
    CHKERRQ(QPGetChild(qp, &qp));
  }
  CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainViewQPPF"
/*@
   QPChainViewQPPF - Calls QPViewQPPF() on each QP in the chain.

   Collective on QP

   Input Parameters:
+  qp - a QP specifying the chain
-  v - viewer

   Level: advanced
@*/
PetscErrorCode QPChainViewQPPF(QP qp,PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool iascii;
  QPPF      pf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  CHKERRQ(PetscViewerASCIIPrintf(v,__FUNCT__" output follows\n"));
  CHKERRQ(QPGetChild(qp, &qp));
  CHKERRQ(PetscViewerASCIIPushTab(v));
  while (qp) {
    CHKERRQ(QPGetQPPF(qp,&pf));
    if (pf) {
      CHKERRQ(PetscViewerASCIIPrintf(v, "-------------------\n"));
      CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)qp,v));
      CHKERRQ(PetscViewerASCIIPrintf(v, "  #%d in chain, derived by %s\n",qp->id,qp->transform_name));
      CHKERRQ(PetscViewerASCIIPushTab(v));
      CHKERRQ(QPPFView(pf,v));
      CHKERRQ(PetscViewerASCIIPopTab(v));
    }
    CHKERRQ(QPGetChild(qp, &qp));
  }
  CHKERRQ(PetscViewerASCIIPopTab(v));
  CHKERRQ(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscFunctionReturn(0);
}
