
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
  CHKERRQ(QPCInitializePackage());

  CHKERRQ(PetscHeaderCreate(qpc,QPC_CLASSID,"QPC","Quadratic Programming Constraints","QPC",comm,QPCDestroy,QPCView));

  qpc->lambdawork   = NULL;
  qpc->is           = NULL;
  /* TODO QPCSetFromOptions */
  qpc->astol        = 10*PETSC_MACHINE_EPSILON;
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
  if (!qpc->ops->setup) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* prepare lambdawork vector based on the layout of given constraint data */
  /* this method is independent on layout of IS (so it is the same for IS=null) */
  CHKERRQ((*qpc->ops->setup)(qpc));

  if (qpc->is) {
    /* IS is given */
    // TODO verify: layout of IS == layout of constraint function data
  }

  /* set values of working vectors */
  CHKERRQ(VecSet(qpc->lambdawork,0.0));

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
  CHKERRQ(VecDestroy(&qpc->lambdawork));
  CHKERRQ(ISDestroy(&qpc->is));
  if (qpc->ops->reset) CHKERRQ((*qpc->ops->reset)(qpc));
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
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject) qpc, v));

  /* get and view general properties of QPC */
  CHKERRQ(PetscViewerASCIIPushTab(v));

  /* linearity */
  CHKERRQ(QPCIsLinear(qpc,&islinear));
  if(islinear){
     CHKERRQ(PetscViewerASCIIPrintf(v, "linear: yes\n"));
  } else {
     CHKERRQ(PetscViewerASCIIPrintf(v, "linear: no\n"));
  }

  /* subsymmetricity */
  CHKERRQ(QPCIsSubsymmetric(qpc,&issubsymmetric));
  if(issubsymmetric){
     CHKERRQ(PetscViewerASCIIPrintf(v, "subsymmetric: yes\n"));
  } else {
     CHKERRQ(PetscViewerASCIIPrintf(v, "subsymmetric: no\n"));
  }

  /* get block size */
  CHKERRQ(QPCGetBlockSize(qpc,&block_size));
  CHKERRQ(PetscViewerASCIIPrintf(v, "block size: %d\n", block_size));

  /* get number of constraints */
  CHKERRQ(QPCGetNumberOfConstraints(qpc,&nmb_of_constraints));
  CHKERRQ(PetscViewerASCIIPrintf(v, "nmb of constraints: %d\n", nmb_of_constraints));

  /* print IS */
  CHKERRQ(PetscViewerASCIIPrintf(v, "index set:\n"));
  PetscViewerASCIIPushTab(v);
  if(qpc->is){
    CHKERRQ(ISView(qpc->is,v));
    // TODO: make ISViewBlock to view IS in blocks
    // CHKERRQ(ISViewBlock(qpc->is,v,block_size));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(v, "not present; all components are constrained or QPC is composite\n"));
  }

  PetscViewerASCIIPopTab(v);

  if (*qpc->ops->view) {
    CHKERRQ((*qpc->ops->view)(qpc,v));
  } else {
    const QPCType type;
    CHKERRQ(QPCGetType(qpc, &type));
    CHKERRQ(PetscInfo(qpc,"Warning: QPCView not implemented yet for type %s\n",type));
  }
  CHKERRQ(PetscViewerASCIIPopTab(v));
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

  CHKERRQ(QPCGetSubvector( qpc, x, &x_sub));
  if (!qpc->ops->viewkkt) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  CHKERRQ((*qpc->ops->viewkkt)(qpc, x_sub, normb, v));
  CHKERRQ(QPCRestoreSubvector( qpc, x, &x_sub));
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

  CHKERRQ(QPCReset(*qpc));

  if ((*qpc)->ops->destroy) {
    CHKERRQ((*(*qpc)->ops->destroy)(*qpc));
  }

  CHKERRQ(PetscFree((*qpc)->data));
  CHKERRQ(PetscHeaderDestroy(qpc));
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

    CHKERRQ(PetscObjectTypeCompare((PetscObject)qpc,type,&issame));
    if (issame) PetscFunctionReturn(0);

    CHKERRQ(PetscFunctionListFind(QPCList,type,(void(**)(void))&create));
    if (!create) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested QPC type %s",type);

    /* Destroy the pre-existing private QPC context */
    if (qpc->ops->destroy) CHKERRQ((*qpc->ops->destroy)(qpc));

    /* Reinitialize function pointers in QPCOps structure */
    CHKERRQ(PetscMemzero(qpc->ops,sizeof(struct _QPCOps)));

    qpc->setupcalled = PETSC_FALSE;

    CHKERRQ((*create)(qpc));
    CHKERRQ(PetscObjectChangeTypeName((PetscObject)qpc,type));
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
    CHKERRQ(PetscObjectReference((PetscObject)is));
  }
  CHKERRQ(ISDestroy(&qpc->is));
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
  if (!qpc->ops->getblocksize) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  CHKERRQ((*qpc->ops->getblocksize)(qpc,bs));
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
      CHKERRQ((*qpc->ops->islinear)(qpc,linear));
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
      CHKERRQ((*qpc->ops->issubsymmetric)(qpc,subsym));
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
  if (!qpc->ops->getnumberofconstraints) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* QPC of this type has own implementation of getnumberofconstraints*/
  CHKERRQ((*qpc->ops->getnumberofconstraints)(qpc,num));
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
  if (!qpc->ops->getconstraintfunction) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* set up QPC (if not already set) */
  CHKERRQ(QPCSetUp(qpc));

  /* get the sub vector of variables, copy them to xcwork */
  CHKERRQ(QPCGetSubvector(qpc,x,&x_sub));

  /* call corresponding function of QPC type */
  CHKERRQ((*qpc->ops->getconstraintfunction)(qpc,x_sub,&hx_type));

  /* hx has the same layout as lambda, in lambdawork I have computed constraint function values */
  CHKERRQ(QPCRestoreSubvector(qpc,x,&x_sub));

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
  CHKERRQ(QPCGetBlockSize(qpc,&bs));

  if (qpc->ops->restoreconstraintfunction) {
    CHKERRQ((*qpc->ops->restoreconstraintfunction)(qpc,x,hx));
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
  CHKERRQ(QPCSetUp(qpc));

  if (qpc->is) {
    /* IS is present, scatter the vector subject to IS */
    CHKERRQ(VecGetSubVector(x,qpc->is,&xc_out));
  } else {
    /* IS is not present, return whole vector of variables */
    xc_out = x;
    CHKERRQ(PetscObjectReference((PetscObject)x));
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
    CHKERRQ(VecRestoreSubVector(x,qpc->is,xc));
  } else {
    CHKERRQ(VecDestroy(xc));
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
  if (!qpc->ops->project) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  CHKERRQ(QPCSetUp(qpc));

  /* at first, copy the values of x to be sure that also unconstrained components will be set */
  CHKERRQ(VecCopy(x,Px));

  /* get the subvectors to send them to appropriate QPC type function */
  CHKERRQ(QPCGetSubvector(qpc,Px,&Px_sub));
  CHKERRQ(QPCGetSubvector(qpc,x,&x_sub));

  /* make projection */
  CHKERRQ((*qpc->ops->project)(qpc,x_sub,Px_sub));

  /* restore subvectors */
  CHKERRQ(QPCRestoreSubvector(qpc,Px,&Px_sub));
  CHKERRQ(QPCRestoreSubvector(qpc,x,&x_sub));

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
  if (!qpc->ops->feas) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* scatter the gradients */
  CHKERRQ(QPCGetSubvector( qpc, x, &x_sub));
  CHKERRQ(QPCGetSubvector( qpc, d, &d_sub));

  /* compute largest step-size for the given QPC type */
  CHKERRQ((*qpc->ops->feas)(qpc, x_sub, d_sub, &alpha_temp));
  CHKERRQ(MPI_Allreduce(&alpha_temp, alpha, 1, MPIU_SCALAR, MPIU_MIN, PetscObjectComm((PetscObject)qpc)));

  /* restore the gradients */
  CHKERRQ(QPCRestoreSubvector( qpc, x, &x_sub));
  CHKERRQ(QPCRestoreSubvector( qpc, d, &d_sub));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGrads"
/*@
QPCGrads - compute free and chopped gradient

Parameters:
+ qpc - QPC instance
. g  - gradient
. gf - free gradient
- gc - chopped gradient
@*/
PetscErrorCode QPCGrads(QPC qpc, Vec x, Vec g, Vec gf, Vec gc)
{
  Vec g_sub, gf_sub, gc_sub, x_sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  PetscValidHeaderSpecific(gf,VEC_CLASSID,4);
  PetscValidHeaderSpecific(gc,VEC_CLASSID,5);

  /* initial values */
  CHKERRQ(VecCopy(g,gf));
  CHKERRQ(VecSet(gc,0.0));

  /* scatter the gradients */
  CHKERRQ(QPCGetSubvector( qpc, x, &x_sub));
  CHKERRQ(QPCGetSubvector( qpc, g, &g_sub));
  CHKERRQ(QPCGetSubvector( qpc, gf, &gf_sub));
  CHKERRQ(QPCGetSubvector( qpc, gc, &gc_sub));

  if (!qpc->ops->grads) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  CHKERRQ((*qpc->ops->grads)(qpc, x_sub, g_sub, gf_sub, gc_sub));

  /* restore the gradients */
  CHKERRQ(QPCRestoreSubvector( qpc, x, &x_sub));
  CHKERRQ(QPCRestoreSubvector( qpc, g, &g_sub));
  CHKERRQ(QPCRestoreSubvector( qpc, gf, &gf_sub));
  CHKERRQ(QPCRestoreSubvector( qpc, gc, &gc_sub));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGradReduced"
/*@
  QPCGradReduced - compute reduced free gradient
  
  Given the step size alpha, the reduce free gradient is defined component wise such that
  x + alpha*gr is a step in the direction of gf if it doesn't violate constraint, otherwise it is gf component shortened
  so that the component of x + alpha*gr will be in active set. E.g., with only lower bound constraint gr=min(gf,(x-lb)/alpha).
  
  Input Parameters:
  + qpc   - QPC instance
  . x     - solution vector
  . gf    - free gradient
  - alpha - step length

  Output Parameters:
  . gr  -  reduced free gradient
@*/
PetscErrorCode QPCGradReduced(QPC qpc, Vec x, Vec gf, PetscReal alpha, Vec gr)
{
  Vec gf_sub, gr_sub, x_sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gf,VEC_CLASSID,3);
  PetscValidLogicalCollectiveScalar(x,alpha,4);
  PetscValidHeaderSpecific(gr,VEC_CLASSID,5);

  /* initial values */
  CHKERRQ(VecCopy(gf,gr));

  /* scatter the gradients */
  CHKERRQ(QPCGetSubvector( qpc, x, &x_sub));
  CHKERRQ(QPCGetSubvector( qpc, gf, &gf_sub));
  CHKERRQ(QPCGetSubvector( qpc, gr, &gr_sub));

  if (!qpc->ops->gradreduced) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);
  CHKERRQ((*qpc->ops->gradreduced)(qpc, x_sub, gf_sub, alpha, gr_sub));

  /* restore the gradients */
  CHKERRQ(QPCRestoreSubvector( qpc, x, &x_sub));
  CHKERRQ(QPCRestoreSubvector( qpc, gf, &gf_sub));
  CHKERRQ(QPCRestoreSubvector( qpc, gr, &gr_sub));
  PetscFunctionReturn(0);
}

