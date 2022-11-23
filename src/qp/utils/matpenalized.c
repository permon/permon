
#include <permonqp.h>
#include <permon/private/permonmatimpl.h>

typedef struct {
  Mat  A,BtB;
  PetscReal rho;
  Vec xwork;
} Mat_Penalized;

PetscErrorCode MatMult_Penalized(Mat Arho,Vec x,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  PetscCall(MatMult(ctx->BtB,x,y));
  PetscCall(VecScale(y,ctx->rho));
  PetscCall(MatMultAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Penalized(Mat Arho,Vec x,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  PetscCall(MatMult(ctx->BtB,x,y));
  PetscCall(VecScale(y,ctx->rho));
  PetscCall(MatMultTransposeAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Penalized(Mat Arho,Vec x,Vec x2,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  if (x2 != y) {
    PetscCall(MatMult(ctx->BtB,x,y));
    PetscCall(VecAYPX(y,ctx->rho,x2));
  } else {
    if (!ctx->xwork) PetscCall(VecDuplicate(y,&ctx->xwork));
    PetscCall(MatMult(ctx->BtB,x,ctx->xwork));
    PetscCall(VecScale(ctx->xwork,ctx->rho));
    PetscCall(VecAXPY(y,1.0,ctx->xwork));
  }
  PetscCall(MatMultAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Penalized(Mat Arho,Vec x,Vec x2,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  if (x2 != y) {
    PetscCall(MatMult(ctx->BtB,x,y));
    PetscCall(VecAYPX(y,ctx->rho,x2));
  } else {
    if (!ctx->xwork) PetscCall(VecDuplicate(y,&ctx->xwork));
    PetscCall(MatMult(ctx->BtB,x,ctx->xwork));
    PetscCall(VecScale(ctx->xwork,ctx->rho));
    PetscCall(VecAXPY(y,1.0,ctx->xwork));
  }
  PetscCall(MatMultTransposeAdd(ctx->A,x,y,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Penalized(Mat Arho,Vec d)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  PetscCall(MatGetDiagonal(ctx->A,d));
  if (!ctx->xwork) PetscCall(VecDuplicate(d,&ctx->xwork));
  PetscCall(MatGetDiagonal(ctx->BtB,ctx->xwork));
  PetscCall(VecAXPY(d,1.0,ctx->xwork));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Penalized(Mat Arho) {
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(MatDestroy(&ctx->BtB));
  PetscCall(VecDestroy(&ctx->xwork));
  PetscCall(PetscFree(ctx));
  PetscCall(MatShellSetContext(Arho, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPenalizedSetPenalty_Penalty(Mat Arho,PetscReal rho)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  ctx->rho = rho;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPenalizedUpdatePenalty_Penalty(Mat Arho,PetscReal rho_update)
{
  Mat_Penalized *ctx;
  PetscReal rho_new;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  rho_new = ctx->rho * rho_update;
  PetscCall(PetscInfo(fllop,"updating rho := %.4e*%.4e = %.4e\n",ctx->rho,rho_update,rho_new));
  ctx->rho = rho_new;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPenalizedGetPenalty_Penalty(Mat Arho,PetscReal *rho)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  *rho = ctx->rho;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatPenalizedGetPenalizedTerm_Penalty(Mat Arho,Mat *BtB)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Arho,(void*)&ctx));
  *BtB = ctx->BtB;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPenalizedUpdatePenalty(Mat Arho,PetscReal rho_update)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(Arho,rho_update,2);
  PetscTryMethod(Arho,"MatPenalizedUpdatePenalty_Penalty_C",(Mat,PetscReal),(Arho,rho_update));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPenalizedSetPenalty(Mat Arho,PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(Arho,rho,2);
  PetscTryMethod(Arho,"MatPenalizedSetPenalty_Penalty_C",(Mat,PetscReal),(Arho,rho));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPenalizedGetPenalty(Mat Arho,PetscReal *rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidRealPointer(rho,2);
  PetscUseMethod(Arho,"MatPenalizedGetPenalty_Penalty_C",(Mat,PetscReal*),(Arho,rho));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPenalizedGetPenalizedTerm(Mat Arho,Mat *BtB)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidPointer(BtB,2);
  PetscUseMethod(Arho,"MatPenalizedGetPenalizedTerm_Penalty_C",(Mat,Mat*),(Arho,BtB));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreatePenalized(QP qp,PetscReal rho,Mat *Arho_new)
{
  Mat_Penalized *ctx;
  Mat Arho;
  Mat A;
  QPPF pf;

  PetscFunctionBegin;
  PetscCall(QPGetOperator(qp,&A));
  PetscCall(QPGetQPPF(qp,&pf));
  PERMON_ASSERT(A,"A specified");

  PetscCall(PetscMalloc(sizeof(Mat_Penalized),&ctx));
  ctx->A = A; PetscCall(PetscObjectReference((PetscObject)A));
  PetscCall(QPPFCreateGtG(pf,&ctx->BtB));
  ctx->rho = rho;
  ctx->xwork = NULL;
  PetscCall(MatCreateShellPermon(PetscObjectComm((PetscObject)qp), A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N, ctx,&Arho));
  PetscCall(MatShellSetOperation(Arho,MATOP_DESTROY,(void(*)(void))MatDestroy_Penalized));
  PetscCall(MatShellSetOperation(Arho,MATOP_MULT,(void(*)(void))MatMult_Penalized));
  PetscCall(MatShellSetOperation(Arho,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_Penalized));
  PetscCall(MatShellSetOperation(Arho,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Penalized));
  PetscCall(MatShellSetOperation(Arho,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_Penalized));
  PetscCall(MatShellSetOperation(Arho,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Penalized));
  PetscCall(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedGetPenalty_Penalty_C",MatPenalizedGetPenalty_Penalty));
  PetscCall(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedSetPenalty_Penalty_C",MatPenalizedSetPenalty_Penalty));
  PetscCall(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedUpdatePenalty_Penalty_C",MatPenalizedUpdatePenalty_Penalty));
  PetscCall(PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedGetPenalizedTerm_Penalty_C",MatPenalizedGetPenalizedTerm_Penalty));
  *Arho_new = Arho;
  PetscFunctionReturn(0);
}
