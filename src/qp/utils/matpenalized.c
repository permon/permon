
#include <fllopqp.h>
#include <private/fllopmatimpl.h>

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
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  TRY( MatMult(ctx->BtB,x,y) );
  TRY( VecScale(y,ctx->rho) );
  TRY( MatMultAdd(ctx->A,x,y,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Penalized"
PetscErrorCode MatMultTranspose_Penalized(Mat Arho,Vec x,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  TRY( MatMult(ctx->BtB,x,y) );
  TRY( VecScale(y,ctx->rho) );
  TRY( MatMultTransposeAdd(ctx->A,x,y,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Penalized"
PetscErrorCode MatMultAdd_Penalized(Mat Arho,Vec x,Vec x2,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  if (x2 != y) {
    TRY( MatMult(ctx->BtB,x,y) );
    TRY( VecAYPX(y,ctx->rho,x2) );
  } else {
    if (!ctx->xwork) TRY( VecDuplicate(y,&ctx->xwork) );
    TRY( MatMult(ctx->BtB,x,ctx->xwork) );
    TRY( VecScale(ctx->xwork,ctx->rho) );
    TRY( VecAXPY(y,1.0,ctx->xwork) );
  }
  TRY( MatMultAdd(ctx->A,x,y,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Penalized"
PetscErrorCode MatMultTransposeAdd_Penalized(Mat Arho,Vec x,Vec x2,Vec y)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  if (x2 != y) {
    TRY( MatMult(ctx->BtB,x,y) );
    TRY( VecAYPX(y,ctx->rho,x2) );
  } else {
    if (!ctx->xwork) TRY( VecDuplicate(y,&ctx->xwork) );
    TRY( MatMult(ctx->BtB,x,ctx->xwork) );
    TRY( VecScale(ctx->xwork,ctx->rho) );
    TRY( VecAXPY(y,1.0,ctx->xwork) );
  }
  TRY( MatMultTransposeAdd(ctx->A,x,y,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Penalized"
PetscErrorCode MatDestroy_Penalized(Mat Arho) {
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  TRY( MatDestroy(&ctx->A) );
  TRY( MatDestroy(&ctx->BtB) );
  TRY( VecDestroy(&ctx->xwork) );
  TRY( PetscFree(ctx) );
  TRY( MatShellSetContext(Arho, NULL) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedSetPenalty_Penalty"
static PetscErrorCode MatPenalizedSetPenalty_Penalty(Mat Arho,PetscReal rho)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
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
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  rho_new = ctx->rho * rho_update;
  TRY( PetscInfo3(fllop,"updating rho := %.4e*%.4e = %.4e\n",ctx->rho,rho_update,rho_new) );
  ctx->rho = rho_new;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalty_Penalty"
static PetscErrorCode MatPenalizedGetPenalty_Penalty(Mat Arho,PetscReal *rho)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
  *rho = ctx->rho;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalizedTerm_Penalty"
static PetscErrorCode MatPenalizedGetPenalizedTerm_Penalty(Mat Arho,Mat *BtB)
{
  Mat_Penalized *ctx;
  PetscFunctionBegin;
  TRY( MatShellGetContext(Arho,(void*)&ctx) );
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
  TRY( PetscTryMethod(Arho,"MatPenalizedUpdatePenalty_Penalty_C",(Mat,PetscReal),(Arho,rho_update)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedSetPenalty"
PetscErrorCode MatPenalizedSetPenalty(Mat Arho,PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(Arho,rho,2);
  TRY( PetscTryMethod(Arho,"MatPenalizedSetPenalty_Penalty_C",(Mat,PetscReal),(Arho,rho)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalty"
PetscErrorCode MatPenalizedGetPenalty(Mat Arho,PetscReal *rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidRealPointer(rho,2);
  TRY( PetscUseMethod(Arho,"MatPenalizedGetPenalty_Penalty_C",(Mat,PetscReal*),(Arho,rho)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPenalizedGetPenalizedTerm"
PetscErrorCode MatPenalizedGetPenalizedTerm(Mat Arho,Mat *BtB)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(Arho,MAT_CLASSID,1);
  PetscValidPointer(BtB,2);
  TRY( PetscUseMethod(Arho,"MatPenalizedGetPenalizedTerm_Penalty_C",(Mat,Mat*),(Arho,BtB)) );
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
  TRY( QPGetOperator(qp,&A) );
  TRY( QPGetQPPF(qp,&pf) );
  FLLOP_ASSERT(A,"A specified");

  TRY( PetscMalloc(sizeof(Mat_Penalized),&ctx) );
  ctx->A = A; TRY( PetscObjectReference((PetscObject)A) );
  TRY( QPPFCreateGtG(pf,&ctx->BtB) );
  ctx->rho = rho;
  ctx->xwork = NULL;
  TRY( MatCreateShellPermon(PetscObjectComm((PetscObject)qp), A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N, ctx,&Arho) );
  TRY( MatShellSetOperation(Arho,MATOP_DESTROY,(void(*)(void))MatDestroy_Penalized) );
  TRY( MatShellSetOperation(Arho,MATOP_MULT,(void(*)(void))MatMult_Penalized) );
  TRY( MatShellSetOperation(Arho,MATOP_MULT_ADD,(void(*)(void))MatMultAdd_Penalized) );
  TRY( MatShellSetOperation(Arho,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Penalized) );
  TRY( MatShellSetOperation(Arho,MATOP_MULT_TRANSPOSE_ADD,(void(*)(void))MatMultTransposeAdd_Penalized) );
  TRY( PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedGetPenalty_Penalty_C",MatPenalizedGetPenalty_Penalty) );
  TRY( PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedSetPenalty_Penalty_C",MatPenalizedSetPenalty_Penalty) );
  TRY( PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedUpdatePenalty_Penalty_C",MatPenalizedUpdatePenalty_Penalty) );
  TRY( PetscObjectComposeFunction((PetscObject)Arho,"MatPenalizedGetPenalizedTerm_Penalty_C",MatPenalizedGetPenalizedTerm_Penalty) );
  *Arho_new = Arho;
  PetscFunctionReturn(0);
}
