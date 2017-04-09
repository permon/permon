
#include <permon/private/flloppcimpl.h>
#include <petscmat.h>

const char *PCDualTypes[]={"none","lumped","PCDualType","PC_DUAL_",0};

PetscLogEvent PC_Dual_Apply, PC_Dual_MatMultSchur;

/* Private context (data structure) for the Dual preconditioner. */
typedef struct {
  PetscBool setfromoptionscalled;
  PCDualType pcdualtype;
  Mat C_bb, At;
  Vec xwork,ywork;
} PC_Dual;

#undef __FUNCT__
#define __FUNCT__ "PCDualSetType_Dual"
static PetscErrorCode PCDualSetType_Dual(PC pc,PCDualType type)
{
  PC_Dual *data = (PC_Dual*) pc->data;

  PetscFunctionBegin;
  data->pcdualtype = type;
  TRY( PCReset(pc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDualSetType"
PetscErrorCode PCDualSetType(PC pc,PCDualType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  TRY( PetscTryMethod(pc,"PCDualSetType_Dual_C",(PC,PCDualType),(pc,type)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDualGetType_Dual"
static PetscErrorCode PCDualGetType_Dual(PC pc,PCDualType *type)
{
  PC_Dual *data = (PC_Dual*) pc->data;

  PetscFunctionBegin;
  *type = data->pcdualtype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDualGetType"
PetscErrorCode PCDualGetType(PC pc,PCDualType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(type,2);
  TRY( PetscTryMethod(pc,"PCDualGetType_Dual_C",(PC,PCDualType*),(pc,type)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_Dual"
static PetscErrorCode PCApply_Dual(PC pc,Vec x,Vec y)
{
  PC_Dual         *ctx = (PC_Dual*)pc->data;

  PetscFunctionBegin;
  TRY( PetscLogEventBegin(PC_Dual_Apply,pc,x,y,0) );
  TRY( MatMult(ctx->At,x,ctx->xwork) );

  TRY( PetscLogEventBegin(PC_Dual_MatMultSchur,pc,x,y,0) );
  TRY( MatMult(ctx->C_bb,ctx->xwork,ctx->ywork) );
  TRY( PetscLogEventEnd(PC_Dual_MatMultSchur,pc,x,y,0) );

  TRY( MatMultTranspose(ctx->At,ctx->ywork,y) );
  TRY( PetscLogEventEnd(PC_Dual_Apply,pc,x,y,0) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_Dual_None"
static PetscErrorCode PCApply_Dual_None(PC pc,Vec x,Vec y)
{
  PetscFunctionBegin;
  TRY( VecCopy(x,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Dual"
static PetscErrorCode PCSetUp_Dual(PC pc)
{
  PC_Dual *ctx = (PC_Dual*)pc->data;
  Mat F = pc->mat;
  Mat Bt, K;

  PetscFunctionBegin;
  TRY( PetscInfo1(pc,"using PCDualType %s\n",PCDualTypes[ctx->pcdualtype]) );

  if (ctx->pcdualtype == PC_DUAL_NONE) {
    pc->ops->apply = PCApply_Dual_None;
    PetscFunctionReturn(0);
  }

  pc->ops->apply = PCApply_Dual;

  TRY( PetscObjectQuery((PetscObject)F,"Bt",(PetscObject*)&Bt) );
  TRY( PetscObjectQuery((PetscObject)F,"K",(PetscObject*)&K) );

  if (ctx->pcdualtype == PC_DUAL_LUMPED) {
    ctx->At = Bt;
    TRY( PetscObjectReference((PetscObject)Bt) );
    ctx->C_bb = K;
    TRY( PetscObjectReference((PetscObject)K) );
    TRY( MatCreateVecs(ctx->C_bb,&ctx->xwork,&ctx->ywork) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_Dual"
static PetscErrorCode PCReset_Dual(PC pc)
{
  PC_Dual         *ctx = (PC_Dual*)pc->data;

  PetscFunctionBegin;
  TRY( MatDestroy(&ctx->At) );
  TRY( MatDestroy(&ctx->C_bb) );
  TRY( VecDestroy(&ctx->xwork) );
  TRY( VecDestroy(&ctx->ywork) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_Dual"
static PetscErrorCode PCView_Dual(PC pc,PetscViewer viewer)
{
  PC_Dual         *ctx = (PC_Dual*)pc->data;
  PetscBool       iascii;

  PetscFunctionBegin;
  TRY( PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii) );
  if (!iascii) PetscFunctionReturn(0);
  TRY( PetscViewerASCIIPrintf(viewer,"  PCDualType: %d (%s)\n",ctx->pcdualtype,PCDualTypes[ctx->pcdualtype]) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_Dual"
static PetscErrorCode PCDestroy_Dual(PC pc)
{
  //PC_Dual         *ctx = (PC_Dual*)pc->data;

  PetscFunctionBegin;
  TRY( PCReset_Dual(pc) );
  TRY( PetscFree(pc->data) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_Dual"
PetscErrorCode PCSetFromOptions_Dual(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Dual         *ctx = (PC_Dual*)pc->data;

  PetscFunctionBegin;
  TRY( PetscOptionsHead(PetscOptionsObject,"PCDUAL options") );
  TRY( PetscOptionsEnum("-pc_dual_type", "PCDUAL type", "PCDualSetType", PCDualTypes, (PetscEnum)ctx->pcdualtype, (PetscEnum*)&ctx->pcdualtype, NULL) );
  TRY( PetscOptionsTail() );
  ctx->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCreate_Dual"
FLLOP_EXTERN PetscErrorCode PCCreate_Dual(PC pc)
{
  PC_Dual      *ctx;
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  /* Create the private data structure for this preconditioner and
     attach it to the PC object.  */
  TRY( PetscNewLog(pc,&ctx) );
  pc->data = (void*)ctx;
  
  ctx->setfromoptionscalled = PETSC_FALSE;
  ctx->pcdualtype = PC_DUAL_NONE;
  
  /* set general PC functions already implemented for this PC type */
  pc->ops->apply               = PCApply_Dual;
  pc->ops->destroy             = PCDestroy_Dual;
  pc->ops->reset               = PCReset_Dual;
  pc->ops->setfromoptions      = PCSetFromOptions_Dual;
  pc->ops->setup               = PCSetUp_Dual;
  pc->ops->view                = PCView_Dual;

  /* set type-specific functions */
  TRY( PetscObjectComposeFunction((PetscObject)pc,"PCDualSetType_Dual_C",PCDualSetType_Dual) );
  TRY( PetscObjectComposeFunction((PetscObject)pc,"PCDualGetType_Dual_C",PCDualGetType_Dual) );

  /* prepare log events*/
  if (!registered) {
    TRY( PetscLogEventRegister("PCdual:Apply", PC_CLASSID, &PC_Dual_Apply) );
    TRY( PetscLogEventRegister("PCdual:ApplySchur", PC_CLASSID, &PC_Dual_MatMultSchur) );
    registered = PETSC_TRUE;
  }

  /* initialize inner data */
  PetscFunctionReturn(0);
}

