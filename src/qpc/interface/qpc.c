
#include <permon/private/qpcimpl.h>

PetscClassId  QPC_CLASSID;

#undef __FUNCT__
#define __FUNCT__ "QPCCreate"
/*@
QPCCreate - create qpc instance

Parameters:
+ comm - MPI comm
- qpc_out - pointer to created QPC
@*/
PetscErrorCode QPCCreate(MPI_Comm comm,QPC *qpc_new)
{
  QPC              qpc;

  PetscFunctionBegin;
  PetscValidPointer(qpc_new,2);
  *qpc_new = 0;
  TRY( QPCInitializePackage() );

  TRY( PetscHeaderCreate(qpc,QPC_CLASSID,"QPC","Quadratic Programming Constraints","QPC",comm,QPCDestroy,QPCView) );

  qpc->lambdawork   = NULL;
  qpc->is           = NULL;
  qpc->setupcalled  = PETSC_FALSE; /* the setup was not called yet */

  *qpc_new = qpc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCSetUp"
PetscErrorCode QPCSetUp(QPC qpc)
{
  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);

  /* if the setup was already called, then ignore this calling */
  if (qpc->setupcalled) PetscFunctionReturn(0);

  FllopTraceBegin;

  /* it is necessary to set up lambdawork vectors based on given constraint data */
  if (!qpc->ops->setup) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* prepare lambdawork vector based on the layout of given constraint data */
  /* this method is independent on layout of IS (so it is the same for IS=null) */
  TRY( (*qpc->ops->setup)(qpc) );

  if (qpc->is) {
    /* IS is given */
    // TODO verify: layout of IS == layout of constraint function data
  }

  /* set values of working vectors */
  TRY( VecSet(qpc->lambdawork,0.0) );

  /* the setup was now called, do not call it again */
  qpc->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCReset"
PetscErrorCode QPCReset(QPC qpc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  TRY( VecDestroy(&qpc->lambdawork) );
  TRY( ISDestroy(&qpc->is) );
  if (qpc->ops->reset) TRY( (*qpc->ops->reset)(qpc) );
  qpc->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCView"
PetscErrorCode QPCView(QPC qpc,PetscViewer v)
{
  PetscInt nmb_of_constraints;
  PetscInt block_size;
  PetscBool islinear, issubsymmetric;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,2);
  TRY( PetscObjectPrintClassNamePrefixType((PetscObject) qpc, v) );

  /* get and view general properties of QPC */
  TRY( PetscViewerASCIIPushTab(v) );

  /* linearity */
  TRY( QPCIsLinear(qpc,&islinear));
  if(islinear){
     TRY( PetscViewerASCIIPrintf(v, "linear: yes\n") );
  } else {
     TRY( PetscViewerASCIIPrintf(v, "linear: no\n") );
  }

  /* subsymmetricity */
  TRY( QPCIsSubsymmetric(qpc,&issubsymmetric));
  if(issubsymmetric){
     TRY( PetscViewerASCIIPrintf(v, "subsymmetric: yes\n") );
  } else {
     TRY( PetscViewerASCIIPrintf(v, "subsymmetric: no\n") );
  }

  /* get block size */
  TRY( QPCGetBlockSize(qpc,&block_size));
  TRY( PetscViewerASCIIPrintf(v, "block size: %d\n", block_size) );

  /* get number of constraints */
  TRY( QPCGetNumberOfConstraints(qpc,&nmb_of_constraints));
  TRY( PetscViewerASCIIPrintf(v, "nmb of constraints: %d\n", nmb_of_constraints) );

  /* print IS */
  TRY( PetscViewerASCIIPrintf(v, "index set:\n") );
  PetscViewerASCIIPushTab(v);
  if(qpc->is){
    TRY( ISView(qpc->is,v) );
    // TODO: make ISViewBlock to view IS in blocks
    // TRY( ISViewBlock(qpc->is,v,block_size) );
  } else {
    TRY( PetscViewerASCIIPrintf(v, "not present; all components are constrained or QPC is composite\n") );
  }

  PetscViewerASCIIPopTab(v);

  if (*qpc->ops->view) {
    TRY( (*qpc->ops->view)(qpc,v) );
  } else {
    const QPCType type;
    TRY( QPCGetType(qpc, &type) );
    TRY( PetscInfo1(qpc,"Warning: QPCView not implemented yet for type %s\n",type) );
  }
  TRY( PetscViewerASCIIPopTab(v) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCViewKKT"
PetscErrorCode QPCViewKKT(QPC qpc, Vec x, PetscReal normb, PetscViewer v)
{
  Vec x_sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidLogicalCollectiveReal(qpc,normb,3);
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,4);

  TRY( QPCGetSubvector( qpc, x, &x_sub) );
  if (!qpc->ops->viewkkt) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  TRY( (*qpc->ops->viewkkt)(qpc, x_sub, normb, v) );
  TRY( QPCRestoreSubvector( qpc, x, &x_sub) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCDestroy"
/*@
QPCDestroy - destroy qpc instance

Parameters:
. qpc_out - pointer of QPC to destroy
@*/
PetscErrorCode QPCDestroy(QPC *qpc)
{
  PetscFunctionBegin;
  if (!*qpc) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*qpc), QPC_CLASSID, 1);
  if (--((PetscObject) (*qpc))->refct > 0) {
    *qpc = 0;
    PetscFunctionReturn(0);
  }

  TRY( QPCReset(*qpc) );

  if ((*qpc)->ops->destroy) {
    TRY( (*(*qpc)->ops->destroy)(*qpc) );
  }

  TRY( PetscFree((*qpc)->data) );
  TRY( PetscHeaderDestroy(qpc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCSetType"
/*@
QPCSetType - set type of constraint

Parameters:
+ qpc - instance of QPC
- type - type of constraint
@*/
PetscErrorCode QPCSetType(QPC qpc, const QPCType type)
{
    PetscErrorCode (*create)(QPC);
    PetscBool  issame;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
    PetscValidCharPointer(type,2);

    TRY( PetscObjectTypeCompare((PetscObject)qpc,type,&issame) );
    if (issame) PetscFunctionReturn(0);

    TRY( PetscFunctionListFind(QPCList,type,(void(**)(void))&create) );
    if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested QPC type %s",type);

    /* Destroy the pre-existing private QPC context */
    if (qpc->ops->destroy) TRY( (*qpc->ops->destroy)(qpc) );

    /* Reinitialize function pointers in QPCOps structure */
    TRY( PetscMemzero(qpc->ops,sizeof(struct _QPCOps)) );

    qpc->setupcalled = PETSC_FALSE;

    TRY( (*create)(qpc) );
    TRY( PetscObjectChangeTypeName((PetscObject)qpc,type) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetType"
/*@
QPCGetType - get the type of the constraint

Parameters:
+ qpc - QPC
- type - pointer of returning type
@*/
PetscErrorCode QPCGetType(QPC qpc,const QPCType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)qpc)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCSetIS"
PetscErrorCode QPCSetIS(QPC qpc,IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  if (is) {
    PetscValidHeaderSpecific(is,IS_CLASSID,2);
    PetscCheckSameComm(qpc,1,is,2);
    TRY( PetscObjectReference((PetscObject)is) );
  }
  TRY( ISDestroy(&qpc->is) );
  qpc->is = is;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetIS"
PetscErrorCode QPCGetIS(QPC qpc,IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidPointer(is,2);
  *is = qpc->is;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetBlockSize"
/*@
QPCGetBlockSize - get the number of constrained unknowns by each constraint, depends on the type of constraint

Parameters:
+ qpc - QPC instance
- bs - pointer to returning block size
@*/
PetscErrorCode QPCGetBlockSize(QPC qpc,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidIntPointer(bs,2);
  if (!qpc->ops->getblocksize) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  TRY( (*qpc->ops->getblocksize)(qpc,bs) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCIsLinear"
/*@
QPCIsLinear - returns the boolean function of linear property of the QPC type; this property is used in MPRGP and MPGP algorithms

Parameters:
+ qpc - QPC instance
- linear - return value
@*/
PetscErrorCode QPCIsLinear(QPC qpc,PetscBool *linear)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidPointer(linear,2);
  if (!qpc->ops->islinear){
      /* default value */
      *linear = PETSC_FALSE;
  } else {
      TRY( (*qpc->ops->islinear)(qpc,linear) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCIsSubsymmetric"
/*@
QPCIsSubsymmetric - returns the boolean function of subsymmetricity property of the QPC type; this property is used in MPRGP and MPGP algorithms

Parameters:
+ qpc - QPC instance
- subsym - return value
@*/
PetscErrorCode QPCIsSubsymmetric(QPC qpc,PetscBool *subsym)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidPointer(subsym,2);
  if (!qpc->ops->issubsymmetric){
      /* default value */
      *subsym = PETSC_FALSE;
  } else {
      TRY( (*qpc->ops->issubsymmetric)(qpc,subsym) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetNumberOfConstraints"
/*@
QPCGetNumberOfConstraints - get the number of constraints

Parameters:
+ qpc - QPC instance
- num - pointer to returning number
@*/
PetscErrorCode QPCGetNumberOfConstraints(QPC qpc,PetscInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidIntPointer(num,2);

  /* verify if there is a corresponding function of QPC type */
  if (!qpc->ops->getnumberofconstraints) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* QPC of this type has own implementation of getnumberofconstraints*/
  TRY( (*qpc->ops->getnumberofconstraints)(qpc,num) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetConstraintFunction"
/*@
QPCGetConstraintFunction - get the function value of constraint functions, afterwards call QPCRestoreConstraintFunction

Parameters:
+ qpc - QPC instance
. x - vector of variables
- hx - pointer to returning values hx = h(x)
@*/
PetscErrorCode QPCGetConstraintFunction(QPC qpc, Vec x, Vec *hx)
{
  Vec               x_sub, hx_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(hx,3);

  /* verify if there is a corresponding function of QPC type */
  if (!qpc->ops->getconstraintfunction) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* set up QPC (if not already set) */
  TRY( QPCSetUp(qpc) );

  /* get the sub vector of variables, copy them to xcwork */
  TRY( QPCGetSubvector(qpc,x,&x_sub) );

  /* call corresponding function of QPC type */
  TRY( (*qpc->ops->getconstraintfunction)(qpc,x_sub,&hx_type) );

  /* hx has the same layout as lambda, in lambdawork I have computed constraint function values */
  TRY( QPCRestoreSubvector(qpc,x,&x_sub) );

  *hx = hx_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCRestoreConstraintFunction"
/*@
QPCRestoreConstraintFunction - restore function values, has to be called after QPCGetConstraintFunction

Parameters:
+ qpc - QPC instance
- hx - pointer to returning values
@*/
PetscErrorCode QPCRestoreConstraintFunction(QPC qpc,Vec x, Vec *hx)
{
  PetscInt bs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(hx,3);

  /* get block size */
  TRY(QPCGetBlockSize(qpc,&bs));

  if (qpc->ops->restoreconstraintfunction) {
    TRY( (*qpc->ops->restoreconstraintfunction)(qpc,x,hx) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetSubvector"
PetscErrorCode QPCGetSubvector(QPC qpc,Vec x,Vec *xc)
{
  Vec xc_out;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(xc,3);
  TRY( QPCSetUp(qpc) );

  if (qpc->is) {
    /* IS is present, scatter the vector subject to IS */
    TRY( VecGetSubVector(x,qpc->is,&xc_out) );
    TRY( PetscObjectReference((PetscObject)x) );
  } else {
    /* IS is not present, return whole vector of variables */
    xc_out = x;
    TRY( PetscObjectReference((PetscObject)x) );
  }

  *xc = xc_out;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCRestoreSubvector"
PetscErrorCode QPCRestoreSubvector(QPC qpc,Vec x,Vec *xc)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(xc,3);

  if (qpc->is) {
    TRY( VecRestoreSubVector(x,qpc->is,xc) );
  } else {
    TRY( VecDestroy(xc) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCProject"
/*@
QPCProject - project the input vector to feasible set

Parameters:
+ qpc - QPC instance
. x - vector of variables
- Px - vector, if NULL then projecting qpc->x
@*/
PetscErrorCode QPCProject(QPC qpc,Vec x, Vec Px)
{
  Vec x_sub,Px_sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Px,VEC_CLASSID,3);
  if (!qpc->ops->project) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  TRY( QPCSetUp(qpc) );

  /* at first, copy the values of x to be sure that also unconstrained components will be set */
  TRY( VecCopy(x,Px) );

  /* get the subvectors to send them to appropriate QPC type function */
  TRY( QPCGetSubvector(qpc,Px,&Px_sub) );
  TRY( QPCGetSubvector(qpc,x,&x_sub) );

  /* make projection */
  TRY( (*qpc->ops->project)(qpc,x_sub,Px_sub) );

  /* restore subvectors */
  TRY( QPCRestoreSubvector(qpc,Px,&Px_sub) );
  TRY( QPCRestoreSubvector(qpc,x,&x_sub) );

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCFeas"
/*@
QPCFeas - compute maximum step-size

Parameters:
+ qpc - QPC instance
. d - minus value of direction
- alpha - pointer to return value
@*/
PetscErrorCode QPCFeas(QPC qpc, Vec x, Vec d, PetscScalar *alpha)
{
  Vec x_sub, d_sub;
  PetscScalar alpha_temp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(d,VEC_CLASSID,3);
  PetscValidScalarPointer(alpha,4);
  if (!qpc->ops->feas) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* scatter the gradients */
  TRY(QPCGetSubvector( qpc, x, &x_sub));
  TRY(QPCGetSubvector( qpc, d, &d_sub));

  /* compute largest step-size for the given QPC type */
  TRY((*qpc->ops->feas)(qpc, x_sub, d_sub, &alpha_temp));

  /* restore the gradients */
  TRY(QPCRestoreSubvector( qpc, x, &x_sub));
  TRY(QPCRestoreSubvector( qpc, d, &d_sub));

  *alpha = alpha_temp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGrads"
/*@
QPCGrads - compute free and chopped gradient

Parameters:
+ qpc - QPC instance
. g - gradient
. phi - free gradient
- beta - chopped gradient
@*/
PetscErrorCode QPCGrads(QPC qpc, Vec x, Vec g, Vec phi, Vec beta)
{
  Vec g_sub, phi_sub, beta_sub, x_sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  PetscValidHeaderSpecific(phi,VEC_CLASSID,4);
  PetscValidHeaderSpecific(beta,VEC_CLASSID,5);

  /* scatter the gradients */
  TRY(QPCGetSubvector( qpc, x, &x_sub));
  TRY(QPCGetSubvector( qpc, g, &g_sub));
  TRY(QPCGetSubvector( qpc, phi, &phi_sub));
  TRY(QPCGetSubvector( qpc, beta, &beta_sub));

  if (!qpc->ops->grads) SETERRQ1(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  TRY((*qpc->ops->grads)(qpc, x_sub, g_sub, phi_sub, beta_sub));

  /* restore the gradients */
  TRY(QPCRestoreSubvector( qpc, x, &x_sub));
  TRY(QPCRestoreSubvector( qpc, g, &g_sub));
  TRY(QPCRestoreSubvector( qpc, phi, &phi_sub));
  TRY(QPCRestoreSubvector( qpc, beta, &beta_sub));
  PetscFunctionReturn(0);
}

