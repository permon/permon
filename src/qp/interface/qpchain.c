
#include <private/qpimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPChainAdd"
/*@
   QPChainAdd - Append QP into the QP chain.

   Input Parameters:
+  qp - a QP that is member of the chain to which we want to append
.  opt - either QP_DUPLICATE_DO_NOT_COPY or QP_DUPLICATE_COPY_POINTERS
-  newchild - a pointer to QP that should be appended
@*/
PetscErrorCode QPChainAdd(QP qp, QPDuplicateOption opt, QP *newchild)
{
  QP last;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidPointer(newchild,2);
  TRY( QPChainGetLast(qp, &last) );
  TRY( QPAddChild(last,opt,newchild) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainPop"
/*@
   QPChainPop - Delete the last QP of the chain.

   Input Parameters:
.  qp - a QP that is member of the chain from which we want to delete the last QP
@*/
PetscErrorCode QPChainPop(QP qp)
{
  QP last;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( QPChainGetLast(qp, &last) );
  if (last->parent) TRY( QPRemoveChild(last->parent) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainFind"
/*@
   QPChainFind - Find QP in the chain.

   Input Parameters:
+  qp - a QP that is member of the chain; specifies the starting point of search
-  transform - a tranformation function that was used to create QP we search for

   Output Parameters:
.  child - first QP in chain that was transformed by transform; NULL if such QP does not exist
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

  TRY( QPGetChild(qp, &cchild) );
  while (cchild) {
    TRY( QPGetTransform(cchild, &ctransform) );
    if (ctransform == transform) {
      *child = cchild;
      break;
    }
    TRY( QPGetChild(cchild, &cchild) );
  };
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainGetLast"
/*@
   QPChainGetLast - Get last QP in the chain.

   Input Parameters:
.  qp - a QP that is member of the chain

   Output Parameters:
.  child - last QP in the chain
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
    TRY( QPGetChild(qp,&tchild) );
  } while (tchild);

  *last = qp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainSetUp"
/*@
   QPChainSetUp - Calls QPSetUP() on QP and its descendants in the chain.

   Input Parameters:
.  qp - a first QP to call QPSetUp() on
@*/
PetscErrorCode QPChainSetUp(QP qp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  do {
    TRY( QPSetUp(qp) );
    TRY( QPGetChild(qp,&qp) );
  } while (qp);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainSetFromOptions"
/*@
   QPChainSetFromOptions - Calls QPSetFromOptions() on QP and its descendants in the chain.

   Input Parameters:
.  qp - a first QP to call QPSetFromOptions on
@*/
PetscErrorCode QPChainSetFromOptions(QP qp)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  _fllop_ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)qp),NULL,"QP chain options","QP");CHKERRQ(_fllop_ierr);
  /* options processed elsewhere */
  TRY( PetscOptionsName("-qp_chain_view","print the info about all QPs in the chain at the end of a QPSSolve call","QPChainView",&flg) );
  TRY( PetscOptionsName("-qp_chain_view_kkt","print detailed post-solve KKT satisfaction information","QPChainViewKKT",&flg) );
  TRY( PetscOptionsName("-qp_chain_view_qppf","print info about QPPF instances in the QP chain","QPChainViewQPPF",&flg) );

  do {
    TRY( QPSetFromOptions(qp) );
    TRY( QPGetChild(qp,&qp) );
  } while (qp);
  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainViewKKT"
/*@
   QPChainViewKKT - Calls QPViewKKT() on each QP in the chain.

   Input Parameters:
+  qp - a QP specifying the chain
-  v - viewer
@*/
PetscErrorCode QPChainViewKKT(QP qp, PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool iascii, first=PETSC_TRUE;
  QP cqp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (!iascii) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  TRY( QPChainGetLast(qp,&cqp) );
  while (1) {
    if (first) {
      first = PETSC_FALSE;
    } else {
      TRY( PetscViewerASCIIPrintf(v, "-------------------\n") );
    }
    TRY( QPViewKKT(cqp,v) );
    cqp = cqp->parent;
    if (cqp == qp) break;
  }
  TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainView"
/*@
   QPChainView - Calls QPView() on each QP in the chain.

   Input Parameters:
+  qp - a QP specifying the chain
-  v - viewer
@*/
PetscErrorCode QPChainView(QP qp, PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (!iascii) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  TRY( PetscViewerASCIIPrintf(v,__FUNCT__" output follows\n") );
  TRY( QPView(qp, v) );
  TRY( QPGetChild(qp, &qp) );
  while (qp) {
    TRY( PetscViewerASCIIPrintf(v, "-------------------\n") );
    TRY( QPView(qp, v) );
    TRY( QPGetChild(qp, &qp) );
  }
  TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPChainViewQPPF"
/*@
   QPChainViewQPPF - Calls QPViewQPPF() on each QP in the chain.

   Input Parameters:
+  qp - a QP specifying the chain
-  v - viewer
@*/
PetscErrorCode QPChainViewQPPF(QP qp,PetscViewer v)
{
  MPI_Comm  comm;
  PetscBool iascii;
  QPPF      pf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)qp,&comm) );
  if (!v) v = PETSC_VIEWER_STDOUT_(comm);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(qp,1,v,2);

  TRY( PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&iascii) );
  if (!iascii) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);

  TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  TRY( PetscViewerASCIIPrintf(v,__FUNCT__" output follows\n") );
  TRY( QPGetChild(qp, &qp) );
  TRY( PetscViewerASCIIPushTab(v) );
  while (qp) {
    TRY( QPGetQPPF(qp,&pf) );
    if (pf) {
      TRY( PetscViewerASCIIPrintf(v, "-------------------\n") );
      TRY( PetscObjectPrintClassNamePrefixType((PetscObject)qp,v) );
      TRY( PetscViewerASCIIPrintf(v, "  #%d in chain, derived by %s\n",qp->id,qp->transform_name) );
      TRY( PetscViewerASCIIPushTab(v) );
      TRY( QPPFView(pf,v) );
      TRY( PetscViewerASCIIPopTab(v) );
    }
    TRY( QPGetChild(qp, &qp) );
  }
  TRY( PetscViewerASCIIPopTab(v) );
  TRY( PetscViewerASCIIPrintf(v,"=====================\n") );
  PetscFunctionReturn(0);
}
