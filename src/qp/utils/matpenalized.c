
#include <permonqp.h>
#include <permon/private/permonmatimpl.h>

typedef struct {
  Mat  A,BtB;
  PetscReal rho;
  Vec xwork;
} Mat_Penalized;

#undef __FUNCT__
#define __FUNCT__ "MatMult_Penalized"
PetscErrorCode MatMult_Penalized(Mat Arho,Vec x,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  CHKERRQ(MatMult(ctx->BtB,x,y));
  CHKERRQ(VecScale(y,ctx->rho));
  CHKERRQ(MatMultAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Penalized"
PetscErrorCode MatMultTranspose_Penalized(Mat Arho,Vec x,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  CHKERRQ(MatMult(ctx->BtB,x,y));
  CHKERRQ(VecScale(y,ctx->rho));
  CHKERRQ(MatMultTransposeAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Penalized"
PetscErrorCode MatMultAdd_Penalized(Mat Arho,Vec x,Vec x2,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  if (x2 != y) {
    CHKERRQ(MatMult(ctx->BtB,x,y));
    CHKERRQ(VecAYPX(y,ctx->rho,x2));
  } else {
    if (!ctx->xwork) CHKERRQ(VecDuplicate(y,&ctx->xwork));
    CHKERRQ(MatMult(ctx->BtB,x,ctx->xwork));
    CHKERRQ(VecScale(ctx->xwork,ctx->rho));
    CHKERRQ(VecAXPY(y,1.0,ctx->xwork));
  }
  CHKERRQ(MatMultAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Penalized"
PetscErrorCode MatMultTransposeAdd_Penalized(Mat Arho,Vec x,Vec x2,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  if (x2 != y) {
    CHKERRQ(MatMult(ctx->BtB,x,y));
    CHKERRQ(VecAYPX(y,ctx->rho,x2));
  } else {
    if (!ctx->xwork) CHKERRQ(VecDuplicate(y,&ctx->xwork));
    CHKERRQ(MatMult(ctx->BtB,x,ctx->xwork));
    CHKERRQ(VecScale(ctx->xwork,ctx->rho));
    CHKERRQ(VecAXPY(y,1.0,ctx->xwork));
  }
  CHKERRQ(MatMultTransposeAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Penalized"
PetscErrorCode MatGetDiagonal_Penalized(Mat Arho,Vec d)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  CHKERRQ(MatGetDiagonal(ctx->A,d));
  if (!ctx->xwork) CHKERRQ(VecDuplicate(d,&ctx->xwork));
  CHKERRQ(MatGetDiagonal(ctx->BtB,ctx->xwork));
  CHKERRQ(VecAXPY(d,1.0,ctx->xwork));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Penalized"
PetscErrorCode MatDestroy_Penalized(Mat Arho) {
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  CHKERRQ(MatDestroy(&ctx->A));
  CHKERRQ(MatDestroy(&ctx->BtB));
  CHKERRQ(VecDestroy(&ctx->xwork));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(MatShellSetContext(Arho, NULL));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedSetPenalty_Penalty"
static PetscErrorCode MatPenalizedSetPenalty_Penalty(Mat Arho,PetscReal rho)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  ctx->rho = rho;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedUpdatePenalty_Penalty"
static PetscErrorCode MatPenalizedUpdatePenalty_Penalty(Mat Arho,PetscReal rho_update)
{
  Mat_Penalized *ctx;
  PetscReal rho_new;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  rho_new = ctx->rho * rho_update;
  CHKERRQ(PetscInfo(fllop,"updating rho := %.4e*%.4e = %.4e\n",ctx->rho,rho_update,rho_new));
  ctx->rho = rho_new;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalty_Penalty"
static PetscErrorCode MatPenalizedGetPenalty_Penalty(Mat Arho,PetscReal *rho)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  *rho = ctx->rho;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalizedTerm_Penalty"
static PetscErrorCode MatPenalizedGetPenalizedTerm_Penalty(Mat Arho,Mat *BtB)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Arho,(void*)&ctx));
  *BtB = ctx->BtB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedUpdatePenalty"
PetscErrorCode MatPenalizedUpdatePenalty(Mat Arho,PetscReal rho_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(Arho,rho_update,2);
  PetscTryMethod(Arho,"MatPenalizedUpdatePenalty_Penalty_C",(Mat,PetscReal),(Arho,rho_update));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedSetPenalty"
PetscErrorCode MatPenalizedSetPenalty(Mat Arho,PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(Arho,rho,2);
  PetscTryMethod(Arho,"MatPenalizedSetPenalty_Penalty_C",(Mat,PetscReal),(Arho,rho));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalty"
PetscErrorCode MatPenalizedGetPenalty(Mat Arho,PetscReal *rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidRealPointer(rho,2);
  PetscUseMethod(Arho,"MatPenalizedGetPenalty_Penalty_C",(Mat,PetscReal*),(Arho,rho));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalizedTerm"
PetscErrorCode MatPenalizedGetPenalizedTerm(Mat Arho,Mat *BtB)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidPointer(BtB,2);
  PetscUseMethod(Arho,"MatPenalizedGetPenalizedTerm_Penalty_C",(Mat,Mat*),(Arho,BtB));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreatePenalized"
PetscErrorCode MatCreatePenalized(QP qp,PetscReal rho,Mat *Arho_new)
{
  Mat_Penalized *ctx;
  Mat Arho;
  Mat A;
  QPPF pf;

  PetscFunctionBegin;
  CHKERRQ(QPGetOperator(qp,&A));
  CHKERRQ(QPGetQPPF(qp,&pf));
  PERMON_ASSERT(A,"A specified");

  CHKERRQ(PetscMalloc(sizeof(Mat_Penalized),&ctx));
  ctx->A = A; CHKERRQ(PetscObjectReference((PetscObject)A));
  CHKERRQ(QPPFCreateGtG(pf,&ctx->BtB));
  ctx->rho = rho;
  ctx->xwork = NULL;
  CHKERRQ(MatCreateShellPermon(PetscObjectComm((PetscObject)qp), A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N, ctx,&Arho));
  CHKERRQ(MatShellSetOperation(Arho,MATOP_DESTROY,(void(*)(void))MatDestroy_Penalized));
  CHKERRQ(MatShellSetOperation(Arho,MATOP_MULT,(void(*)(void))MatMult_Penalized));
  CHKERRQ(MatShellSetOperation(Arho,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_Penalized));
  CHKERRQ(MatShellSetOperation(Arho,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Penalized));
  CHKERRQ(MatShellSetOperation(Arho,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_Penalized));
  CHKERRQ(MatShellSetOperation(Arho,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Penalized));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedGetPenalty_Penalty_C",MatPenalizedGetPenalty_Penalty));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedSetPenalty_Penalty_C",MatPenalizedSetPenalty_Penalty));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedUpdatePenalty_Penalty_C",MatPenalizedUpdatePenalty_Penalty));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedGetPenalizedTerm_Penalty_C",MatPenalizedGetPenalizedTerm_Penalty));
  *Arho_new = Arho;
  PetscFunctionReturn(0);
}
