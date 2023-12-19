
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
  PetscAssertPointer(qpc_new,2);
  *qpc_new = 0;
  PetscCall(QPCInitializePackage());

  PetscCall(PetscHeaderCreate(qpc,QPC_CLASSID,"QPC","Quadratic Programming Constraints","QPC",comm,QPCDestroy,QPCView));

  qpc->lambdawork   = NULL;
  qpc->is           = NULL;
  /* TODO QPCSetFromOptions */
  qpc->astol        = 100*PETSC_MACHINE_EPSILON;
  qpc->setupcalled  = PETSC_FALSE; /* the setup was not called yet */

  *qpc_new = qpc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCSetUp"
PetscErrorCode QPCSetUp(QPC qpc)
{
  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);

  /* if the setup was already called, then ignore this calling */
  if (qpc->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  FllopTraceBegin;

  /* prepare lambdawork vector based on the layout of given constraint data */
  /* this method is independent on layout of IS (so it is the same for IS=null) */
  PetscUseTypeMethod(qpc,setup);

  if (qpc->is) {
    /* IS is given */
    // TODO verify: layout of IS == layout of constraint function data
  }

  /* set values of working vectors */
  PetscCall(VecSet(qpc->lambdawork,0.0));

  /* the setup was now called, do not call it again */
  qpc->setupcalled = PETSC_TRUE;
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCReset"
PetscErrorCode QPCReset(QPC qpc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscCall(VecDestroy(&qpc->lambdawork));
  PetscCall(ISDestroy(&qpc->is));
  PetscTryTypeMethod(qpc,reset);
  qpc->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) qpc, v));

  /* get and view general properties of QPC */
  PetscCall(PetscViewerASCIIPushTab(v));

  /* linearity */
  PetscCall(QPCIsLinear(qpc,&islinear));
  if(islinear){
     PetscCall(PetscViewerASCIIPrintf(v, "linear: yes\n"));
  } else {
     PetscCall(PetscViewerASCIIPrintf(v, "linear: no\n"));
  }

  /* subsymmetricity */
  PetscCall(QPCIsSubsymmetric(qpc,&issubsymmetric));
  if(issubsymmetric){
     PetscCall(PetscViewerASCIIPrintf(v, "subsymmetric: yes\n"));
  } else {
     PetscCall(PetscViewerASCIIPrintf(v, "subsymmetric: no\n"));
  }

  /* get block size */
  PetscCall(QPCGetBlockSize(qpc,&block_size));
  PetscCall(PetscViewerASCIIPrintf(v, "block size: %d\n", block_size));

  /* get number of constraints */
  PetscCall(QPCGetNumberOfConstraints(qpc,&nmb_of_constraints));
  PetscCall(PetscViewerASCIIPrintf(v, "nmb of constraints: %d\n", nmb_of_constraints));

  /* print IS */
  PetscCall(PetscViewerASCIIPrintf(v, "index set:\n"));
  PetscCall(PetscViewerASCIIPushTab(v));
  if(qpc->is){
    PetscCall(ISView(qpc->is,v));
    // TODO: make ISViewBlock to view IS in blocks
    // PetscCall(ISViewBlock(qpc->is,v,block_size));
  } else {
    PetscCall(PetscViewerASCIIPrintf(v, "not present; all components are constrained or QPC is composite\n"));
  }

  PetscCall(PetscViewerASCIIPopTab(v));

  if (*qpc->ops->view) {
    PetscUseTypeMethod(qpc,view,v);
  } else {
    const QPCType type;
    PetscCall(QPCGetType(qpc, &type));
    PetscCall(PetscInfo(qpc,"Warning: QPCView not implemented yet for type %s\n",type));
  }
  PetscCall(PetscViewerASCIIPopTab(v));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscCall(QPCGetSubvector(qpc,x,&x_sub));
  PetscUseTypeMethod(qpc,viewkkt,x_sub,normb,v);
  PetscCall(QPCRestoreSubvector(qpc,x,&x_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (!*qpc) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*qpc), QPC_CLASSID, 1);
  if (--((PetscObject) (*qpc))->refct > 0) {
    *qpc = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(QPCReset(*qpc));

  PetscTryTypeMethod((*qpc),destroy);

  PetscCall(PetscFree((*qpc)->data));
  PetscCall(PetscHeaderDestroy(qpc));
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscAssertPointer(type,2);

    PetscCall(PetscObjectTypeCompare((PetscObject)qpc,type,&issame));
    if (issame) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(PetscFunctionListFind(QPCList,type,(void(**)(void))&create));
    if (!create) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested QPC type %s",type);

    /* Destroy the pre-existing private QPC context */
    PetscTryTypeMethod(qpc,destroy);

    /* Reinitialize function pointers in QPCOps structure */
    PetscCall(PetscMemzero(qpc->ops,sizeof(struct _QPCOps)));

    qpc->setupcalled = PETSC_FALSE;

    PetscCall((*create)(qpc));
    PetscCall(PetscObjectChangeTypeName((PetscObject)qpc,type));
    PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(type,2);
  *type = ((PetscObject)qpc)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
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
    PetscCall(PetscObjectReference((PetscObject)is));
  }
  PetscCall(ISDestroy(&qpc->is));
  qpc->is = is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetIS"
PetscErrorCode QPCGetIS(QPC qpc,IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscAssertPointer(is,2);
  *is = qpc->is;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(bs,2);
  PetscUseTypeMethod(qpc,getblocksize,bs);
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(linear,2);
  if (!qpc->ops->islinear){
      /* default value */
      *linear = PETSC_FALSE;
  } else {
      PetscUseTypeMethod(qpc,islinear,linear);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(subsym,2);
  if (!qpc->ops->issubsymmetric){
      /* default value */
      *subsym = PETSC_FALSE;
  } else {
      PetscUseTypeMethod(qpc,issubsymmetric,subsym);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(num,2);

  PetscUseTypeMethod(qpc,getnumberofconstraints,num);
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(hx,3);

  /* verify if there is a corresponding function of QPC type */
  if (!qpc->ops->getconstraintfunction) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* set up QPC (if not already set) */
  PetscCall(QPCSetUp(qpc));

  /* get the sub vector of variables, copy them to xcwork */
  PetscCall(QPCGetSubvector(qpc,x,&x_sub));

  /* call corresponding function of QPC type */
  PetscUseTypeMethod(qpc,getconstraintfunction,x_sub,&hx_type);

  /* hx has the same layout as lambda, in lambdawork I have computed constraint function values */
  PetscCall(QPCRestoreSubvector(qpc,x,&x_sub));

  *hx = hx_type;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(hx,3);

  /* get block size */
  PetscCall(QPCGetBlockSize(qpc,&bs));

  PetscTryTypeMethod(qpc,restoreconstraintfunction,x,hx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCGetSubvector"
PetscErrorCode QPCGetSubvector(QPC qpc,Vec x,Vec *xc)
{
  Vec xc_out;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscAssertPointer(xc,3);
  PetscCall(QPCSetUp(qpc));

  if (qpc->is) {
    /* IS is present, scatter the vector subject to IS */
    PetscCall(VecGetSubVector(x,qpc->is,&xc_out));
  } else {
    /* IS is not present, return whole vector of variables */
    xc_out = x;
    PetscCall(PetscObjectReference((PetscObject)x));
  }

  *xc = xc_out;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCRestoreSubvector"
PetscErrorCode QPCRestoreSubvector(QPC qpc,Vec x,Vec *xc)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscAssertPointer(xc,3);

  if (qpc->is) {
    PetscCall(VecRestoreSubVector(x,qpc->is,xc));
  } else {
    PetscCall(VecDestroy(xc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(QPCSetUp(qpc));

  /* at first, copy the values of x to be sure that also unconstrained components will be set */
  PetscCall(VecCopy(x,Px));

  /* get the subvectors to send them to appropriate QPC type function */
  PetscCall(QPCGetSubvector(qpc,Px,&Px_sub));
  PetscCall(QPCGetSubvector(qpc,x,&x_sub));

  /* make projection */
  PetscUseTypeMethod(qpc,project,x_sub,Px_sub);

  /* restore subvectors */
  PetscCall(QPCRestoreSubvector(qpc,Px,&Px_sub));
  PetscCall(QPCRestoreSubvector(qpc,x,&x_sub));

  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(alpha,4);
  if (!qpc->ops->feas) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* scatter the gradients */
  PetscCall(QPCGetSubvector( qpc, x, &x_sub));
  PetscCall(QPCGetSubvector( qpc, d, &d_sub));

  /* compute largest step-size for the given QPC type */
  PetscUseTypeMethod(qpc, feas, x_sub, d_sub, &alpha_temp);
  PetscCallMPI(MPI_Allreduce(&alpha_temp, alpha, 1, MPIU_SCALAR, MPIU_MIN, PetscObjectComm((PetscObject)qpc)));

  /* restore the gradients */
  PetscCall(QPCRestoreSubvector( qpc, x, &x_sub));
  PetscCall(QPCRestoreSubvector( qpc, d, &d_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCInfeas"
/*@
QPCInfeas - TODO compute maximum step-size

Parameters:
+ qpc - QPC instance
. d - minus value of direction
- alpha - pointer to return value
@*/
PetscErrorCode QPCInfeas(QPC qpc, Vec x, Vec d, PetscScalar *alpha)
{
  Vec x_sub, d_sub;
  PetscScalar alpha_temp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qpc,QPC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(d,VEC_CLASSID,3);
  PetscAssertPointer(alpha,4);
  if (!qpc->ops->infeas) SETERRQ(PetscObjectComm((PetscObject)qpc),PETSC_ERR_SUP,"QPC type %s",((PetscObject)qpc)->type_name);

  /* scatter the gradients */
  PetscCall(QPCGetSubvector( qpc, x, &x_sub));
  PetscCall(QPCGetSubvector( qpc, d, &d_sub));

  /* compute largest step-size for the given QPC type */
  PetscUseTypeMethod(qpc, infeas, x_sub, d_sub, &alpha_temp);
  PetscCallMPI(MPI_Allreduce(&alpha_temp, alpha, 1, MPIU_SCALAR, MPIU_MAX, PetscObjectComm((PetscObject)qpc)));

  /* restore the gradients */
  PetscCall(QPCRestoreSubvector( qpc, x, &x_sub));
  PetscCall(QPCRestoreSubvector( qpc, d, &d_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(VecCopy(g,gf));
  PetscCall(VecSet(gc,0.0));

  /* scatter the gradients */
  PetscCall(QPCGetSubvector( qpc, x, &x_sub));
  PetscCall(QPCGetSubvector( qpc, g, &g_sub));
  PetscCall(QPCGetSubvector( qpc, gf, &gf_sub));
  PetscCall(QPCGetSubvector( qpc, gc, &gc_sub));

  PetscUseTypeMethod(qpc,grads, x_sub, g_sub, gf_sub, gc_sub);

  /* restore the gradients */
  PetscCall(QPCRestoreSubvector( qpc, x, &x_sub));
  PetscCall(QPCRestoreSubvector( qpc, g, &g_sub));
  PetscCall(QPCRestoreSubvector( qpc, gf, &gf_sub));
  PetscCall(QPCRestoreSubvector( qpc, gc, &gc_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(VecCopy(gf,gr));

  /* scatter the gradients */
  PetscCall(QPCGetSubvector( qpc, x, &x_sub));
  PetscCall(QPCGetSubvector( qpc, gf, &gf_sub));
  PetscCall(QPCGetSubvector( qpc, gr, &gr_sub));

  PetscUseTypeMethod(qpc, gradreduced, x_sub, gf_sub, alpha, gr_sub);

  /* restore the gradients */
  PetscCall(QPCRestoreSubvector( qpc, x, &x_sub));
  PetscCall(QPCRestoreSubvector( qpc, gf, &gf_sub));
  PetscCall(QPCRestoreSubvector( qpc, gr, &gr_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

