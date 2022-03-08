
#include <permon/private/permonpcimpl.h>
#include <petscmat.h>

const char *PCDualTypes[]={"none","lumped","dirichlet","dirichlet_diag","lumped_full","PCDualType","PC_DUAL_",0};

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
  Mat Bt, Kplus, K, K_loc;
  Mat C_bb_loc;
  IS iis, bis;   /* local indices of internal dofs and boundary dofs, respectively */
  MPI_Comm comm;
  PetscBool flg=PETSC_FALSE;

  PetscFunctionBegin;
  TRY( PetscInfo1(pc,"using PCDualType %s\n",PCDualTypes[ctx->pcdualtype]) );

  if (ctx->pcdualtype == PC_DUAL_NONE) {
    pc->ops->apply = PCApply_Dual_None;
    PetscFunctionReturn(0);
  }

  pc->ops->apply = PCApply_Dual;

  TRY( PetscObjectQuery((PetscObject)F,"Bt",(PetscObject*)&Bt) );
  TRY( PetscObjectQuery((PetscObject)F,"K",(PetscObject*)&K) );
  TRY( PetscObjectQuery((PetscObject)F,"Kplus",(PetscObject*)&Kplus) );
  TRY( MatGetDiagonalBlock(K,&K_loc) );

  if (ctx->pcdualtype == PC_DUAL_LUMPED_FULL) {
    ctx->At = Bt;
    TRY( PetscObjectReference((PetscObject)Bt) );
    ctx->C_bb = K;
    TRY( PetscObjectReference((PetscObject)K) );
    TRY( MatCreateVecs(ctx->C_bb,&ctx->xwork,&ctx->ywork) );
    PetscFunctionReturn(0);
  }
  TRY( MatExtensionCreateCondensedRows(Bt, &ctx->At) );
  TRY( MatExtensionGetRowISLocal(Bt,&bis) );
  
  switch (ctx->pcdualtype) {
  
    case PC_DUAL_LUMPED:
    {
      /* lumped preconditioner with restriction */
      TRY( MatCreateSubMatrix(K_loc,bis,bis,MAT_INITIAL_MATRIX,&C_bb_loc) );
    }
    break;

    case PC_DUAL_DIRICHLET:
    {
      /* Dirichlet preconditioner */
      Mat K_loc_aij;
      KSP ksp;
      PC  pc_inner;
      PetscInt m;

      TRY( MatGetSize(K_loc,&m,NULL) );
      TRY( ISComplement(bis,0,m,&iis) );
      TRY( MatConvert(K_loc,MATAIJ,MAT_INITIAL_MATRIX,&K_loc_aij) );
      TRY( MatGetSchurComplement(K_loc_aij,iis,iis,bis,bis,MAT_INITIAL_MATRIX,&C_bb_loc,MAT_SCHUR_COMPLEMENT_AINV_DIAG,MAT_IGNORE_MATRIX,NULL) );

      if (FllopObjectInfoEnabled) {
        Mat A;
        TRY( MatSchurComplementGetSubMatrices(C_bb_loc,&A,NULL,NULL,NULL,NULL) );
        TRY( PetscObjectSetName((PetscObject)A,"pcdual_schur_A00") );
        TRY( MatPrintInfo(A) );
      }

      /* set Schur inner KSP properties */
      TRY( MatSchurComplementGetKSP(C_bb_loc,&ksp) );
      TRY( KSPSetType(ksp,KSPPREONLY) );
      TRY( KSPGetPC(ksp,&pc_inner) );
      TRY( PCSetType(pc_inner,PCCHOLESKY) );
      TRY( FllopPetscObjectInheritPrefix((PetscObject)ksp,(PetscObject)pc,"pcdual_schur_") );
      TRY( FllopPetscObjectInheritPrefix((PetscObject)pc_inner,(PetscObject)pc,"pcdual_schur_") );
      if (pc->setfromoptionscalled) TRY( KSPSetFromOptions(ksp) );
      TRY( KSPSetUp(ksp) );

      TRY( MatDestroy(&K_loc_aij) );
      TRY( ISDestroy(&iis) );
    }
    break;

    case PC_DUAL_DIRICHLET_DIAG:
    {
      /* Dirichlet preconditioner */
      Mat K_loc_aij;
      Mat S;
      PetscInt m;

      TRY( MatGetSize(K_loc,&m,NULL) );
      TRY( ISComplement(bis,0,m,&iis) );
      TRY( MatConvert(K_loc,MATAIJ,MAT_INITIAL_MATRIX,&K_loc_aij) );

      //TODO fix MatGetSchurComplement_Basic in PETSc so that creating S is not needed
      TRY( MatGetSchurComplement(K_loc_aij,iis,iis,bis,bis,MAT_INITIAL_MATRIX,&S,MAT_SCHUR_COMPLEMENT_AINV_DIAG,MAT_INITIAL_MATRIX,&C_bb_loc) );
      TRY( MatDestroy(&S) );
      TRY( MatDestroy(&K_loc_aij) );
      TRY( ISDestroy(&iis) );
    }
    break;
      
    default:
      FLLOP_SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"unknown PCDualType: %d",ctx->pcdualtype);
  }
  
  TRY( PetscOptionsGetBool(NULL,NULL,"-dual_pc_explicit_schur",&flg,NULL) );
  if (flg) {
    Mat mat;
    TRY( MatComputeOperator(C_bb_loc,NULL,&mat) );
    TRY( MatDestroy(&C_bb_loc) );
    C_bb_loc = mat;
  }

  /* C_bb = blkdiag(C_bb_loc) */
  TRY( MatCreateBlockDiag(comm,C_bb_loc,&ctx->C_bb) );
  
  TRY( MatCreateVecs(ctx->C_bb,&ctx->xwork,&ctx->ywork) );

  TRY( MatDestroy(&C_bb_loc) );
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

