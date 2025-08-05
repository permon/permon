#include <permon/private/permonpcimpl.h>
#include <permon/private/qpcimpl.h>
#include <permonqps.h>

const char *PCFreeSetTypes[] = {"basic", "cheap", "fixed", "PCFreeSetType", "PC_FREESET_", 0};

PetscLogEvent PC_FreeSet_Apply;

static PetscErrorCode PCApply_FreeSet_Cheap(PC pc, Vec x, Vec y);
static PetscErrorCode PCApply_FreeSet_Fixed(PC pc, Vec x, Vec y);

/* Private context (data structure) for the FreeSet preconditioner. */
typedef struct {
  PC               pc;
  IS               is;
  PetscObjectState ISState;
  Vec              xlayout, xwork, ywork;
  PCFreeSetType    type;
  PetscBool        setfromoptionscalled;
} PC_FreeSet;

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetType_FreeSet"
static PetscErrorCode PCFreeSetSetType_FreeSet(PC pc, PCFreeSetType type)
{
  PC_FreeSet *data = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  data->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetSetType"
/*@
   PCFreeSetSetType -  Set type of the preconditioner

   Logically Collective

   Input Parameters:
+  pc - instance of PC
-  type - the type of preconditioner

   Level: intermediate

.seealso `PCFREESET`, `PCFreeSetGetType()`
@*/
PetscErrorCode PCFreeSetSetType(PC pc, PCFreeSetType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCFreeSetSetType_C", (PC, PCFreeSetType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeGetType_FreeSet"
static PetscErrorCode PCFreeSetGetType_FreeSet(PC pc, PCFreeSetType *type)
{
  PC_FreeSet *data = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  *type = data->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetGetType"
/*@
   PCFreeSetGetType -  Get type of the preconditioner

   Logically Collective

   Input Parameters:
+  pc - instance of PC
-  type - the type of preconditioner

   Level: intermediate

.seealso `PCFREESET`, `PCFreeSetSetType()`
@*/
PetscErrorCode PCFreeSetGetType(PC pc, PCFreeSetType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCFreeSetGetType_C", (PC, PCFreeSetType *), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetSetIS_FreeSet"
static PetscErrorCode PCFreeSetSetIS_FreeSet(PC pc, IS is)
{
  PC_FreeSet *data = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  data->is = is;
  PetscCall(PetscObjectReference((PetscObject)is));
  //PetscCall(PCReset(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetSetIS"
/*@
   PCFreeSetSetIS -  Set free/active set `IS` (see Notes)

   Logically Collective

   Input Parameters:
+  PC - instance of PC
-  is - free/active set IS

   Notes:
   The `IS` corresponds either to free set (basic variant) or active set (cheap variant).

   Level: advanced

.seealso `PCFREESET`, `PCFreeSetGetIS()`, `PCFreeSetSetType()`, `PCFreeSetGetType()`
@*/
PetscErrorCode PCFreeSetSetIS(PC pc, IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscTryMethod(pc, "PCFreeSetSetIS_C", (PC, IS), (pc, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetGetIS_FreeSet"
static PetscErrorCode PCFreeSetGetIS_FreeSet(PC pc, IS *is)
{
  PC_FreeSet *data = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  *is = data->is;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCFreeSetGetIS"
/*@
   PCFreeSetGetIS -  Get free/active set (see Notes)

   Not Collective

   Input Parameter:
.  PC - instance of PC

   Output Parameter:
.  is - free/active set IS

   Notes:
   The `IS` corresponds either to free set (basic variant) or active set (cheap variant).

   Level: advanced

.seealso `PCFREESET`, `PCFreeSetSetIS()`, `PCFreeSetSetType()`, `PCFreeSetGetType()`
@*/
PetscErrorCode PCFreeSetGetIS(PC pc, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(is, 2);
  PetscTryMethod(pc, "PCFreeSetGetIS_C", (PC, IS *), (pc, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUpInnerPC_FreeSet"
static PetscErrorCode PCSetUpInnerPC_FreeSet(PC pc)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;
  Mat         innerpmat;

  PetscFunctionBegin;
  if (ctx->type == PC_FREESET_BASIC || ctx->type == PC_FREESET_FIXED) {
    //TODO add Virtual submat option
    PetscCall(MatCreateSubMatrix(pc->pmat, ctx->is, ctx->is, MAT_INITIAL_MATRIX, &innerpmat));
    PetscCall(PCSetOperators(ctx->pc, innerpmat, innerpmat));
    PetscCall(MatCreateVecs(innerpmat, &ctx->xwork, &ctx->ywork));
    PetscCall(MatDestroy(&innerpmat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_FreeSet"
static PetscErrorCode PCSetUp_FreeSet(PC pc)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(pc->pmat, &ctx->xlayout, NULL));
  if (!ctx->pc) PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &ctx->pc));
  if (ctx->type == PC_FREESET_CHEAP) {
    pc->ops->apply = PCApply_FreeSet_Cheap;
    PetscCall(PCSetOperators(ctx->pc, pc->pmat, pc->pmat));
  } else if (ctx->type == PC_FREESET_FIXED) {
    pc->ops->apply = PCApply_FreeSet_Fixed;
    if (ctx->is) { PetscCall(PCSetUpInnerPC_FreeSet(pc)); } // defer the PCSetUp to PCUpdateFromQPS to attempt to grab IS from QPC
  }
  //PetscCall(PCSetUpInnerPC_FreeSet(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_FreeSet"
static PetscErrorCode PCReset_FreeSet(PC pc)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  if (ctx->pc) PetscCall(PCReset(ctx->pc));
  if (ctx->type == PC_FREESET_BASIC) {
    PetscCall(VecDestroy(&ctx->xwork));
    PetscCall(VecDestroy(&ctx->ywork));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCUpdateFromQPS_FreeSet"
static PetscErrorCode PCUpdateFromQPS_FreeSet(PC pc, QPS qps)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;
  QP          qp;
  QPC         qpc;
  IS          qpcis;
  PetscInt    ilo, ihi;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps, &qp));
  PetscCall(QPGetQPC(qp, &qpc));
  /* TODO only if freeset changed */
  if (ctx->type == PC_FREESET_BASIC) {
    if (qpc->setchanged) {
      PetscCall(QPCGetFreeSet(qpc, PETSC_TRUE, ctx->xlayout, &ctx->is));
      PetscCall(PCReset_FreeSet(pc));
      PetscCall(PCSetUpInnerPC_FreeSet(pc));
    }
  } else if (ctx->type == PC_FREESET_CHEAP) {
    PetscCall(QPCGetActiveSet(qpc, PETSC_TRUE, &ctx->is));
  } else if (ctx->type == PC_FREESET_FIXED) {
    if (!ctx->is) {
      PetscCall(QPCGetIS(qpc, &qpcis));
      PetscCheck(qpcis, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Fixed type PCFREESET is used but no free set is provided and it cannot be obtained from QPC");
      PetscCall(VecGetOwnershipRange(ctx->xlayout, &ilo, &ihi));
      PetscCall(ISComplement(qpcis, ilo, ihi, &ctx->is)); // TODO check if the IS is valid when qpcis is the full space
      PetscCall(PetscInfo(pc, "computed free set from QPC\n"));
      PetscCall(PCSetUpInnerPC_FreeSet(pc));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_FreeSet"
static PetscErrorCode PCApply_FreeSet(PC pc, Vec x, Vec y)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  //PetscCall(PCFreeSetSetUpInnerPC(pc));
  PetscCall(PetscLogEventBegin(PC_FreeSet_Apply, pc, x, y, 0));
  PetscCall(VecSet(y, 0.));
  PetscCall(VecGetSubVector(x, ctx->is, &ctx->xwork));
  PetscCall(VecGetSubVector(y, ctx->is, &ctx->ywork));
  PetscCall(PCApply(ctx->pc, ctx->xwork, ctx->ywork));
  PetscCall(VecRestoreSubVector(x, ctx->is, &ctx->xwork));
  PetscCall(VecRestoreSubVector(y, ctx->is, &ctx->ywork));
  PetscCall(PetscLogEventEnd(PC_FreeSet_Apply, pc, x, y, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_FreeSet_Cheap"
static PetscErrorCode PCApply_FreeSet_Cheap(PC pc, Vec x, Vec y)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  //PetscCall(PCFreeSetSetUpInnerPC(pc));
  PetscCall(PetscLogEventBegin(PC_FreeSet_Apply, pc, x, y, 0));
  PetscCall(PCApply(ctx->pc, x, y));
  PetscCall(VecISSet(y, ctx->is, 0.));
  PetscCall(PetscLogEventEnd(PC_FreeSet_Apply, pc, x, y, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_FreeSet_Fixed"
static PetscErrorCode PCApply_FreeSet_Fixed(PC pc, Vec x, Vec y)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_FreeSet_Apply, pc, x, y, 0));
  PetscCall(VecCopy(x, y)); // could apply Jacobi precond on the rest of the freeset
  PetscCall(VecGetSubVector(x, ctx->is, &ctx->xwork));
  PetscCall(VecGetSubVector(y, ctx->is, &ctx->ywork));
  PetscCall(PCApply(ctx->pc, ctx->xwork, ctx->ywork));
  PetscCall(VecRestoreSubVector(x, ctx->is, &ctx->xwork));
  PetscCall(VecRestoreSubVector(y, ctx->is, &ctx->ywork));
  PetscCall(PetscLogEventEnd(PC_FreeSet_Apply, pc, x, y, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_FreeSet"
static PetscErrorCode PCView_FreeSet(PC pc, PetscViewer viewer)
{
  PC_FreeSet   *ctx = (PC_FreeSet *)pc->data;
  PetscBool     iascii;
  PCFreeSetType type;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (!iascii) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCFreeSetGetType(pc, &type));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  type %s\n", PCFreeSetTypes[type]));
  PetscCall(PCView(ctx->pc, viewer));
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
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetSetIS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetGetIS_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_FreeSet"
PetscErrorCode PCSetFromOptions_FreeSet(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_FreeSet *ctx = (PC_FreeSet *)pc->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PCFREESET options");
  PetscCall(PetscOptionsEnum("-pc_freeset_type", "PCFREESET type", "PCFreeSetSetType", PCFreeSetTypes, (PetscEnum)ctx->type, (PetscEnum *)&ctx->type, NULL));
  if (!ctx->pc) {
    PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &ctx->pc));
    PetscCall(PCGetOptionsPrefix(pc, &prefix));
    PetscCall(PCSetOptionsPrefix(ctx->pc, prefix));
    PetscCall(PCAppendOptionsPrefix(ctx->pc, "pc_inner_"));
  }
  PetscCall(PCSetFromOptions(ctx->pc));
  PetscOptionsHeadEnd();
  ctx->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PCFREESET - preconditioner on the free set

  Options Database Keys:
+ -pc_freeset_type <basic, cheap, fixed> - approach for forming the preconditioner
- -pc_freeset_inner_pc - options for the inner preconditioner

  Level: intermediate

  Notes:
  The basic variant corresponds to the preconditioning in face (given
  by the free set). It computes inner preconditioner on pmat restricted
  to the free set. The inner preconditioner has to be recomputed whenever
  the free set changes.

  The cheap variant approximates the basic variant by applying the inner
  preconditioner (computed once with the whole pmat) and zeroing out
  the active set components.

  The fixed variant is meant to apply preconditoner only on a set that is
  known to be always free. This is set should be supplied by user through
  `PCFreeSetSetIS()` or is computed automatically as a complement
  of the `QPC` index set.

  The inner preconditioner can be controlled with
  `PCFreeSetGetInnerPC()` or -pc_freeset_inner_.

  Developer Notes:
  This is typically applied to the free gradient.


.seealso:  `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
           `PCFreeSetSetType()`, `PCFreeSetGetType`,
           `PCFreeSetSetIS()`, `PCFreeSetGetIS()`
M*/
#undef __FUNCT__
#define __FUNCT__ "PCCreate_FreeSet"
PERMON_EXTERN PetscErrorCode PCCreate_FreeSet(PC pc)
{
  PC_FreeSet      *ctx;
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  /* Create the private data structure for this preconditioner and
     attach it to the PC object.  */
  PetscCall(PetscNew(&ctx));
  pc->data = (void *)ctx;

  ctx->setfromoptionscalled = PETSC_FALSE;

  /* set general PC functions already implemented for this PC type */
  pc->ops->apply          = PCApply_FreeSet;
  pc->ops->destroy        = PCDestroy_FreeSet;
  pc->ops->reset          = PCReset_FreeSet;
  pc->ops->setfromoptions = PCSetFromOptions_FreeSet;
  pc->ops->setup          = PCSetUp_FreeSet;
  pc->ops->view           = PCView_FreeSet;

  /* set type-specific functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetSetType_C", PCFreeSetSetType_FreeSet));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetGetType_C", PCFreeSetGetType_FreeSet));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetSetIS_C", PCFreeSetSetIS_FreeSet));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCFreeSetGetIS_C", PCFreeSetGetIS_FreeSet));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCUpdateFromQPS_C", PCUpdateFromQPS_FreeSet));

  /* prepare log events*/
  if (!registered) {
    PetscCall(PetscLogEventRegister("PCFreeSet:Apply", PC_CLASSID, &PC_FreeSet_Apply));
    registered = PETSC_TRUE;
  }

  /* initialize inner data */
  PetscFunctionReturn(PETSC_SUCCESS);
}
