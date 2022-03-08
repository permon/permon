#include <permon/private/permonpcimpl.h>
#include <petscmat.h>

const char *PCDualTypes[]={"none","lumped","dirichlet","dirichlet_diag","lumped_full","PCDualType","PC_DUAL_",0};

PetscLogEvent PC_Dual_Apply, PC_Dual_MatMultSchur;

/* Private context (data structure) for the Dual preconditioner. */
typedef struct {
  PetscBool  setfromoptionscalled;
  PCDualType pcdualtype;
  Mat        C_bb, At;
  Vec        xwork, ywork;
} PC_Dual;

#undef __FUNCT__
#define __FUNCT__ "PCDualSetType_Dual"
static PetscErrorCode PCDualSetType_Dual(PC pc, PCDualType type)
{
  PC_Dual *data = (PC_Dual *)pc->data;

  PetscFunctionBegin;
  data->pcdualtype = type;
  PetscCall(PCReset(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCDualSetType"
PetscErrorCode PCDualSetType(PC pc, PCDualType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(pc, type, 2);
  PetscTryMethod(pc, "PCDualSetType_Dual_C", (PC, PCDualType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCDualGetType_Dual"
static PetscErrorCode PCDualGetType_Dual(PC pc, PCDualType *type)
{
  PC_Dual *data = (PC_Dual *)pc->data;

  PetscFunctionBegin;
  *type = data->pcdualtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCDualGetType"
PetscErrorCode PCDualGetType(PC pc, PCDualType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscTryMethod(pc, "PCDualGetType_Dual_C", (PC, PCDualType *), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_Dual"
static PetscErrorCode PCApply_Dual(PC pc, Vec x, Vec y)
{
  PC_Dual *ctx = (PC_Dual *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_Dual_Apply, pc, x, y, 0));
  PetscCall(MatMult(ctx->At, x, ctx->xwork));

  PetscCall(PetscLogEventBegin(PC_Dual_MatMultSchur, pc, x, y, 0));
  PetscCall(MatMult(ctx->C_bb, ctx->xwork, ctx->ywork));
  PetscCall(PetscLogEventEnd(PC_Dual_MatMultSchur, pc, x, y, 0));

  PetscCall(MatMultTranspose(ctx->At, ctx->ywork, y));
  PetscCall(PetscLogEventEnd(PC_Dual_Apply, pc, x, y, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_Dual_None"
static PetscErrorCode PCApply_Dual_None(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Dual"
static PetscErrorCode PCSetUp_Dual(PC pc)
{
<<<<<<< HEAD
  PC_Dual *ctx = (PC_Dual *)pc->data;
  Mat      F   = pc->mat;
  Mat      Bt, K;
=======
  PC_Dual *ctx = (PC_Dual*)pc->data;
  Mat F = pc->mat;
  Mat Bt, Kplus, K, K_loc;
  Mat C_bb_loc;
  IS iis, bis;   /* local indices of internal dofs and boundary dofs, respectively */
  MPI_Comm comm;
  PetscBool flg=PETSC_FALSE;
>>>>>>> fb4c00f (dirichlet pc)

  PetscFunctionBegin;
  PetscCall(PetscInfo(pc, "using PCDualType %s\n", PCDualTypes[ctx->pcdualtype]));

  if (ctx->pcdualtype == PC_DUAL_NONE) {
    pc->ops->apply = PCApply_Dual_None;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  pc->ops->apply = PCApply_Dual;

<<<<<<< HEAD
  PetscCall(PetscObjectQuery((PetscObject)F, "Bt", (PetscObject *)&Bt));
  PetscCall(PetscObjectQuery((PetscObject)F, "K", (PetscObject *)&K));
=======
  PetscCall(PetscObjectQuery((PetscObject)F,"Bt",(PetscObject*)&Bt));
  PetscCall(PetscObjectQuery((PetscObject)F,"K",(PetscObject*)&K));
  PetscCall(PetscObjectQuery((PetscObject)F,"Kplus",(PetscObject*)&Kplus));
  PetscCall(MatGetDiagonalBlock(K,&K_loc));
>>>>>>> fb4c00f (dirichlet pc)

  if (ctx->pcdualtype == PC_DUAL_LUMPED_FULL) {
    ctx->At = Bt;
    PetscCall(PetscObjectReference((PetscObject)Bt));
    ctx->C_bb = K;
    PetscCall(PetscObjectReference((PetscObject)K));
<<<<<<< HEAD
    PetscCall(MatCreateVecs(ctx->C_bb, &ctx->xwork, &ctx->ywork));
=======
    PetscCall(MatCreateVecs(ctx->C_bb,&ctx->xwork,&ctx->ywork));
    PetscFunctionReturn(PETSC_SUCCESS);
>>>>>>> fb4c00f (dirichlet pc)
  }
  PetscCall(MatExtensionCreateCondensedRows(Bt, &ctx->At) );
  PetscCall(MatExtensionGetRowISLocal(Bt,&bis) );

  switch (ctx->pcdualtype) {

    case PC_DUAL_LUMPED:
    {
      /* lumped preconditioner with restriction */
      PetscCall(MatCreateSubMatrix(K_loc,bis,bis,MAT_INITIAL_MATRIX,&C_bb_loc) );
    }
    break;

    case PC_DUAL_DIRICHLET:
    {
      /* Dirichlet preconditioner */
      Mat K_loc_aij;
      KSP ksp;
      PC  pc_inner;
      PetscInt m;

      PetscCall(MatGetSize(K_loc,&m,NULL) );
      PetscCall(ISComplement(bis,0,m,&iis) );
      PetscCall(MatConvert(K_loc,MATAIJ,MAT_INITIAL_MATRIX,&K_loc_aij) );
      PetscCall(MatGetSchurComplement(K_loc_aij,iis,iis,bis,bis,MAT_INITIAL_MATRIX,&C_bb_loc,MAT_SCHUR_COMPLEMENT_AINV_DIAG,MAT_IGNORE_MATRIX,NULL) );

      if (FllopObjectInfoEnabled) {
        Mat A;
        PetscCall(MatSchurComplementGetSubMatrices(C_bb_loc,&A,NULL,NULL,NULL,NULL) );
        PetscCall(PetscObjectSetName((PetscObject)A,"pcdual_schur_A00") );
        PetscCall(MatPrintInfo(A) );
      }

      /* set Schur inner KSP properties */
      PetscCall(MatSchurComplementGetKSP(C_bb_loc,&ksp) );
      PetscCall(KSPSetType(ksp,KSPPREONLY) );
      PetscCall(KSPGetPC(ksp,&pc_inner) );
      PetscCall(PCSetType(pc_inner,PCCHOLESKY) );
      PetscCall(FllopPetscObjectInheritPrefix((PetscObject)ksp,(PetscObject)pc,"pcdual_schur_") );
      PetscCall(FllopPetscObjectInheritPrefix((PetscObject)pc_inner,(PetscObject)pc,"pcdual_schur_") );
      if (pc->setfromoptionscalled) PetscCall(KSPSetFromOptions(ksp) );
      PetscCall(KSPSetUp(ksp) );

      PetscCall(MatDestroy(&K_loc_aij) );
      PetscCall(ISDestroy(&iis) );
    }
    break;

    case PC_DUAL_DIRICHLET_DIAG:
    {
      /* Dirichlet preconditioner */
      Mat K_loc_aij;
      Mat S;
      PetscInt m;

      PetscCall(MatGetSize(K_loc,&m,NULL) );
      PetscCall(ISComplement(bis,0,m,&iis) );
      PetscCall(MatConvert(K_loc,MATAIJ,MAT_INITIAL_MATRIX,&K_loc_aij) );

      //TODO fix MatGetSchurComplement_Basic in PETSc so that creating S is not needed
      PetscCall(MatGetSchurComplement(K_loc_aij,iis,iis,bis,bis,MAT_INITIAL_MATRIX,&S,MAT_SCHUR_COMPLEMENT_AINV_DIAG,MAT_INITIAL_MATRIX,&C_bb_loc) );
      PetscCall(MatDestroy(&S) );
      PetscCall(MatDestroy(&K_loc_aij) );
      PetscCall(ISDestroy(&iis) );
    }
    break;

    default:
      FLLOP_SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"unknown PCDualType: %d",ctx->pcdualtype);
  }

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-dual_pc_explicit_schur",&flg,NULL) );
  if (flg) {
    Mat mat;
    PetscCall(MatComputeOperator(C_bb_loc,NULL,&mat) );
    PetscCall(MatDestroy(&C_bb_loc) );
    C_bb_loc = mat;
  }

  /* C_bb = blkdiag(C_bb_loc) */
  PetscCall(MatCreateBlockDiag(comm,C_bb_loc,&ctx->C_bb) );

  PetscCall(MatCreateVecs(ctx->C_bb,&ctx->xwork,&ctx->ywork) );

  PetscCall(MatDestroy(&C_bb_loc) );
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_Dual"
static PetscErrorCode PCReset_Dual(PC pc)
{
  PC_Dual *ctx = (PC_Dual *)pc->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ctx->At));
  PetscCall(MatDestroy(&ctx->C_bb));
  PetscCall(VecDestroy(&ctx->xwork));
  PetscCall(VecDestroy(&ctx->ywork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_Dual"
static PetscErrorCode PCView_Dual(PC pc, PetscViewer viewer)
{
  PC_Dual  *ctx = (PC_Dual *)pc->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (!iascii) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscViewerASCIIPrintf(viewer, "  PCDualType: %d (%s)\n", ctx->pcdualtype, PCDualTypes[ctx->pcdualtype]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_Dual"
static PetscErrorCode PCDestroy_Dual(PC pc)
{
  //PC_Dual         *ctx = (PC_Dual*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_Dual(pc));
  PetscCall(PetscFree(pc->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCDualSetType_Dual_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCDualGetType_Dual_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_Dual"
PetscErrorCode PCSetFromOptions_Dual(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_Dual *ctx = (PC_Dual *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PCDUAL options");
  PetscCall(PetscOptionsEnum("-pc_dual_type", "PCDUAL type", "PCDualSetType", PCDualTypes, (PetscEnum)ctx->pcdualtype, (PetscEnum *)&ctx->pcdualtype, NULL));
  PetscOptionsHeadEnd();
  ctx->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PCCreate_Dual"
PERMON_EXTERN PetscErrorCode PCCreate_Dual(PC pc)
{
  PC_Dual         *ctx;
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  /* Create the private data structure for this preconditioner and
     attach it to the PC object.  */
  PetscCall(PetscNew(&ctx));
  pc->data = (void *)ctx;

  ctx->setfromoptionscalled = PETSC_FALSE;
  ctx->pcdualtype           = PC_DUAL_NONE;

  /* set general PC functions already implemented for this PC type */
  pc->ops->apply          = PCApply_Dual;
  pc->ops->destroy        = PCDestroy_Dual;
  pc->ops->reset          = PCReset_Dual;
  pc->ops->setfromoptions = PCSetFromOptions_Dual;
  pc->ops->setup          = PCSetUp_Dual;
  pc->ops->view           = PCView_Dual;

  /* set type-specific functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCDualSetType_Dual_C", PCDualSetType_Dual));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCDualGetType_Dual_C", PCDualGetType_Dual));

  /* prepare log events*/
  if (!registered) {
    PetscCall(PetscLogEventRegister("PCdual:Apply", PC_CLASSID, &PC_Dual_Apply));
    PetscCall(PetscLogEventRegister("PCdual:ApplySchur", PC_CLASSID, &PC_Dual_MatMultSchur));
    registered = PETSC_TRUE;
  }

  /* initialize inner data */
  PetscFunctionReturn(PETSC_SUCCESS);
}
