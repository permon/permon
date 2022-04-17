
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
  PetscCall(QPChainGetLast(qp, &last));
  PetscCall(QPAddChild(last,opt,newchild));
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
  PetscCall(QPChainGetLast(qp, &last));
  if (last->parent) PetscCall(QPRemoveChild(last->parent));
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

  PetscCall(QPGetChild(qp, &cchild));
  while (cchild) {
    PetscCall(QPGetTransform(cchild, &ctransform));
    if (ctransform == transform) {
      *child = cchild;
      break;
    }
    PetscCall(QPGetChild(cchild, &cchild));
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
    PetscCall(QPGetChild(qp,&tchild));
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
    PetscCall(QPSetUp(qp));
    PetscCall(QPGetChild(qp,&qp));
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
  PetscCall(PetscOptionsName("-qp_chain_view","print the info about all QPs in the chain at the end of a QPSSolve call","QPChainView",&flg));
  PetscCall(PetscOptionsName("-qp_chain_view_kkt","print detailed post-solve KKT satisfaction information","QPChainViewKKT",&flg));
  PetscCall(PetscOptionsName("-qp_chain_view_qppf","print info about QPPF instances in the QP chain","QPChainViewQPPF",&flg));

  do {
    PetscCall(QPSetFromOptions(qp));
    PetscCall(QPGetChild(qp,&qp));
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
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)qp,&prefix));

  PetscCall(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_view",&v,&format,&view));
  if (view & !PetscPreLoadingOn) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(QPView(qp,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }

  PetscCall(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_chain_view",&v,&format,&view));
  if (view & !PetscPreLoadingOn) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(QPChainView(qp,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }

  PetscCall(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_chain_view_qppf",&v,&format,&view));
  if (view & !PetscPreLoadingOn) {
    PetscCall(PetscViewerPushFormat(v,format));
    PetscCall(QPChainViewQPPF(qp,v));
    PetscCall(PetscViewerPopFormat(v));
    PetscCall(PetscViewerDestroy(&v));
  }

  PetscCall(PetscOptionsGetViewer(comm,NULL,prefix,"-qp_chain_view_kkt",&v,&format,&view));
  view = (PetscBool)(view && !PetscPreLoadingOn);
  if (view) {
    PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&flg));
    if (!flg) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);
    PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
  }

  PetscCall(QPChainGetLast(qp,&cqp));
  solved = cqp->solved;
  first = PETSC_TRUE;
  while (1) {
    PetscCall(QPComputeMissingBoxMultipliers(cqp));
    PetscCall(QPComputeMissingEqMultiplier(cqp));
    parent = cqp->parent;
    postSolve = cqp->postSolve;
    if (postSolve) PetscCall((*postSolve)(cqp,parent));

    if (view) {
      if (first) {
        first = PETSC_FALSE;
      } else {
        PetscCall(PetscViewerASCIIPrintf(v, "-------------------\n"));
      }
      PetscCall(QPViewKKT(cqp,v));
    }

    if (!parent) break;
    parent->solved = solved;
    if (cqp == qp) break;
    cqp = parent;
  }

  if (view) {
    PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
    PetscCall(PetscViewerDestroy(&v));
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
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscCall(QPChainGetLast(qp,&cqp));
  while (1) {
    if (first) {
      first = PETSC_FALSE;
    } else {
      PetscCall(PetscViewerASCIIPrintf(v, "-------------------\n"));
    }
    PetscCall(QPViewKKT(cqp,v));
    cqp = cqp->parent;
    if (cqp == qp) break;
  }
  PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
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
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscCall(PetscViewerASCIIPrintf(v,__FUNCT__" output follows\n"));
  PetscCall(QPView(qp, v));
  PetscCall(QPGetChild(qp, &qp));
  while (qp) {
    PetscCall(PetscViewerASCIIPrintf(v, "-------------------\n"));
    PetscCall(QPView(qp, v));
    PetscCall(QPGetChild(qp, &qp));
  }
  PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
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
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscCall(PetscViewerASCIIPrintf(v,__FUNCT__" output follows\n"));
  PetscCall(QPGetChild(qp, &qp));
  PetscCall(PetscViewerASCIIPushTab(v));
  while (qp) {
    PetscCall(QPGetQPPF(qp,&pf));
    if (pf) {
      PetscCall(PetscViewerASCIIPrintf(v, "-------------------\n"));
      PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)qp,v));
      PetscCall(PetscViewerASCIIPrintf(v, "  #%d in chain, derived by %s\n",qp->id,qp->transform_name));
      PetscCall(PetscViewerASCIIPushTab(v));
      PetscCall(QPPFView(pf,v));
      PetscCall(PetscViewerASCIIPopTab(v));
    }
    PetscCall(QPGetChild(qp, &qp));
  }
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscCall(PetscViewerASCIIPrintf(v,"=====================\n"));
  PetscFunctionReturn(0);
}
