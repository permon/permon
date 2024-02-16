
#include <permon/private/permonpcimpl.h>
#include <petscmat.h>
#include <permonqps.h>

const char *PCFreeSetTypes[]={"basic","cheap","PCFreeSetType","PC_FREESET_",0};

PetscLogEvent PC_FreeSet_Apply;

static PetscErrorCode PCApply_FreeSet_Cheap(PC pc,Vec x,Vec y);

/* Private context (data structure) for the FreeSet preconditioner. */
typedef struct {
  PC pc;
  IS is;
  PetscObjectState ISState;
  Vec xlayout,xwork,ywork;
  PCFreeSetType type;
  PetscBool setfromoptionscalled;
} PC_FreeSet;

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetSetIS_FreeSet"
static PetscErrorCode PCFreeSetSetIS_FreeSet(PC pc,IS is)
{
  PC_FreeSet *data = (PC_FreeSet*) pc->data;

  PetscFunctionBegin;
  data->is = is;
  //PetscCall(PCReset(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetSetIS"
PetscErrorCode PCFreeSetSetType(PC pc,IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscTryMethod(pc,"PCFreeSetSetIS_FreeSet_C",(PC,IS),(pc,is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetGetIS_FreeSet"
static PetscErrorCode PCFreeSetGetIS_FreeSet(PC pc,IS *is)
{
  PC_FreeSet *data = (PC_FreeSet*) pc->data;

  PetscFunctionBegin;
  *is = data->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetGetIS"
PetscErrorCode PCFreeSetGetIS(PC pc,IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscAssertPointer(is,2);
  PetscTryMethod(pc,"PCFreeSetGetIS_FreeSet_C",(PC,IS*),(pc,is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUpInnerPC_FreeSet"
static PetscErrorCode PCSetUpInnerPC_FreeSet(PC pc)
{
  PC_FreeSet *ctx = (PC_FreeSet*)pc->data;
  Mat innerpmat;

  PetscFunctionBegin;
  if (ctx->type == PC_FREESET_BASIC) {
    //TODO add Virtual submat option
    PetscCall(MatCreateSubMatrix(pc->pmat,ctx->is,ctx->is,MAT_INITIAL_MATRIX,&innerpmat));
    PetscCall(PCSetOperators(ctx->pc,innerpmat,innerpmat));
    PetscCall(MatCreateVecs(innerpmat,&ctx->xwork,&ctx->ywork));
    PetscCall(MatDestroy(&innerpmat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_FreeSet"
static PetscErrorCode PCSetUp_FreeSet(PC pc)
{
  PC_FreeSet *ctx = (PC_FreeSet*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(pc->pmat,&ctx->xlayout,NULL));
  if (!ctx->pc) PetscCall(PCCreate(PetscObjectComm((PetscObject)pc),&ctx->pc));
  if (ctx->type == PC_FREESET_CHEAP) {
    pc->ops->apply = PCApply_FreeSet_Cheap;
    PetscCall(PCSetOperators(ctx->pc,pc->pmat,pc->pmat));
  }
  //PetscCall(PCSetUpInnerPC_FreeSet(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_FreeSet"
static PetscErrorCode PCReset_FreeSet(PC pc)
{
  PC_FreeSet         *ctx = (PC_FreeSet*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset(ctx->pc));
  if (ctx->type == PC_FREESET_BASIC) {
    PetscCall(VecDestroy(&ctx->xwork));
    PetscCall(VecDestroy(&ctx->ywork));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCUpdateFromQPS_FreeSet"
static PetscErrorCode PCUpdateFromQPS_FreeSet(PC pc,QPS qps)
{
  PC_FreeSet *ctx = (PC_FreeSet*) pc->data;
  QP  qp;
  QPC qpc;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));
  PetscCall(QPGetQPC(qp,&qpc));
  /* TODO only if freeset changed */
  if (ctx->type == PC_FREESET_BASIC) {
    PetscCall(QPCGetFreeSet(qpc,PETSC_TRUE,ctx->xlayout,&ctx->is));
    PetscCall(PCReset_FreeSet(pc));
    PetscCall(PCSetUpInnerPC_FreeSet(pc));
  } else { /* CHEAP variant */
    PetscCall(QPCGetActiveSet(qpc,PETSC_TRUE,&ctx->is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_FreeSet"
static PetscErrorCode PCApply_FreeSet(PC pc,Vec x,Vec y)
{
  PC_FreeSet         *ctx = (PC_FreeSet*)pc->data;

  PetscFunctionBegin;
  //PetscCall(PCFreeSetSetUpInnerPC(pc));
  PetscCall(PetscLogEventBegin(PC_FreeSet_Apply,pc,x,y,0));
  PetscCall(VecSet(y,0.));
  PetscCall(VecGetSubVector(x,ctx->is,&ctx->xwork));
  PetscCall(VecGetSubVector(y,ctx->is,&ctx->ywork));
  PetscCall(PCApply(ctx->pc,ctx->xwork,ctx->ywork));
  PetscCall(VecRestoreSubVector(x,ctx->is,&ctx->xwork));
  PetscCall(VecRestoreSubVector(y,ctx->is,&ctx->ywork));
  PetscCall(PetscLogEventEnd(PC_FreeSet_Apply,pc,x,y,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_FreeSet_Cheap"
static PetscErrorCode PCApply_FreeSet_Cheap(PC pc,Vec x,Vec y)
{
  PC_FreeSet         *ctx = (PC_FreeSet*)pc->data;

  PetscFunctionBegin;
  //PetscCall(PCFreeSetSetUpInnerPC(pc));
  PetscCall(PetscLogEventBegin(PC_FreeSet_Apply,pc,x,y,0));
  PetscCall(PCApply(ctx->pc,x,y));
  PetscCall(VecISSet(y,ctx->is,0.));
  PetscCall(PetscLogEventEnd(PC_FreeSet_Apply,pc,x,y,0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_FreeSet"
static PetscErrorCode PCView_FreeSet(PC pc,PetscViewer viewer)
{
  PC_FreeSet         *ctx = (PC_FreeSet*)pc->data;
  PetscBool       iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (!iascii) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCView(ctx->pc,viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_FreeSet"
static PetscErrorCode PCDestroy_FreeSet(PC pc)
{
  //PC_FreeSet         *ctx = (PC_FreeSet*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_FreeSet(pc));
  PetscCall(PetscFree(pc->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFreeSetSetIS_FreeSet_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFreeSetGetIS_FreeSet_C",NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_FreeSet"
PetscErrorCode PCSetFromOptions_FreeSet(PC pc,PetscOptionItems *PetscOptionsObject)
{
  PC_FreeSet         *ctx = (PC_FreeSet*)pc->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"PCFREESET options");
  PetscCall(PetscOptionsEnum("-pc_freeset_type", "PCFREESET type", "PCFreeSetSetType", PCFreeSetTypes, (PetscEnum)ctx->type, (PetscEnum*)&ctx->type, NULL));
  if(!ctx->pc) {
    PetscCall(PCCreate(PetscObjectComm((PetscObject)pc),&ctx->pc));
    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(PCSetOptionsPrefix(ctx->pc,prefix));
    PetscCall(PCAppendOptionsPrefix(ctx->pc,"pc_inner_"));
  }
  PetscCall(PCSetFromOptions(ctx->pc));
  PetscOptionsHeadEnd();
  ctx->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCCreate_FreeSet - Creates a FreeSet preconditioner context, PC_FreeSet,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCJACOBI - FreeSet (i.e. diagonal scaling preconditioning)

   Options Database Keys:
+    -pc_jacobi_type <diagonal,rowmax,rowsum> - approach for forming the preconditioner
.    -pc_jacobi_abs - use the absolute value of the diagonal entry
-    -pc_jacobi_fixdiag - fix for zero diagonal terms by placing 1.0 in those locations

   Level: beginner

  Notes:
    By using `KSPSetPCSide`(ksp,`PC_SYMMETRIC`) or -ksp_pc_side symmetric
    can scale each side of the matrix by the square root of the diagonal entries.

    Zero entries along the diagonal are replaced with the value 1.0

    See `PCPBJACOBI` for fixed-size point block, `PCVPBJACOBI` for variable-sized point block, and `PCBJACOBI` for large size blocks

.seealso:  `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
           `PCFreeSetSetType()`, `PCFreeSetSetUseAbs()`, `PCFreeSetGetUseAbs()`, `PCASM`,
           `PCFreeSetSetFixDiagonal()`, `PCFreeSetGetFixDiagonal()`
           `PCFreeSetSetType()`, `PCFreeSetSetUseAbs()`, `PCFreeSetGetUseAbs()`, `PCPBJACOBI`, `PCBJACOBI`, `PCVPBJACOBI`
M*/
#undef __FUNCT__
#define __FUNCT__ "PCCreate_FreeSet"
FLLOP_EXTERN PetscErrorCode PCCreate_FreeSet(PC pc)
{
  PC_FreeSet      *ctx;
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  /* Create the private data structure for this preconditioner and
     attach it to the PC object.  */
  PetscCall(PetscNew(&ctx));
  pc->data = (void*)ctx;

  ctx->setfromoptionscalled = PETSC_FALSE;

  /* set general PC functions already implemented for this PC type */
  pc->ops->apply               = PCApply_FreeSet;
  pc->ops->destroy             = PCDestroy_FreeSet;
  pc->ops->reset               = PCReset_FreeSet;
  pc->ops->setfromoptions      = PCSetFromOptions_FreeSet;
  pc->ops->setup               = PCSetUp_FreeSet;
  pc->ops->view                = PCView_FreeSet;

  /* set type-specific functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFreeSetSetIS_FreeSet_C",PCFreeSetSetIS_FreeSet));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFreeSetGetIS_FreeSet_C",PCFreeSetGetIS_FreeSet));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCUpdateFromQPS_C",PCUpdateFromQPS_FreeSet));

  /* prepare log events*/
  if (!registered) {
    PetscCall(PetscLogEventRegister("PCFreeSet:Apply", PC_CLASSID, &PC_FreeSet_Apply));
    registered = PETSC_TRUE;
  }

  /* initialize inner data */
  PetscFunctionReturn(PETSC_SUCCESS);
}

