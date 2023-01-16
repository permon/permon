
#include <permon/private/permonmatimpl.h>

PetscErrorCode MatMult_Complete(Mat A, Vec x, Vec y)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  PetscCall(PetscContainerGetPointer(container, (void**)&ctx));
  PetscCall(VecPointwiseMult(y,x,ctx->d));
  PetscCall(VecScale(y, -1.0));
  PetscCall((ctx->multadd)(A,x,y,y));
  PetscCall((ctx->multtransposeadd)(A,x,y,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Complete(Mat A, Vec x, Vec x1, Vec y)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  PetscCall(PetscContainerGetPointer(container, (void**)&ctx));
  PetscCall(VecPointwiseMult(y,x,ctx->d));
  PetscCall(VecScale(y, -1.0));
  PetscCall((ctx->multadd)(A,x,y,y));
  PetscCall((ctx->multtransposeadd)(A,x,y,y));
  PetscCall(VecAXPY(y, 1.0, x1));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Complete(Mat A,MatDuplicateOption op,Mat *M)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  Mat _M;
  
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  PetscCall(PetscContainerGetPointer(container, (void**)&ctx));
  PetscCall((ctx->duplicate)(A,op,&_M));
  _M->ops->mult              = ctx->mult;
  _M->ops->multtranspose     = ctx->multtranspose;
  _M->ops->multadd           = ctx->multadd;
  _M->ops->multtransposeadd  = ctx->multtransposeadd;
  _M->ops->duplicate         = ctx->duplicate;
  *M = _M;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCompleteCtxCreate(Mat A, MatCompleteCtx *ctxout)
{
  MatCompleteCtx ctx;
  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->mult             = A->ops->mult;
  ctx->multtranspose    = A->ops->multtranspose;
  ctx->multadd          = A->ops->multadd;
  ctx->multtransposeadd = A->ops->multtransposeadd;
  ctx->duplicate        = A->ops->duplicate;
  PetscCall(MatCreateVecs(A, NULL, &ctx->d));
  PetscCall(MatGetDiagonal(A, ctx->d));
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCompleteCtxDestroy(MatCompleteCtx ctx)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&ctx->d));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCompleteFromUpperTriangular(Mat A)
{
  MPI_Comm comm;
  MatCompleteCtx ctx;
  PetscContainer container;
  PetscBool flg;
  
  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  if (container) PetscFunctionReturn(0);

  PetscCall(MatIsSymmetricByType(A, &flg));
  if (flg) {
    PetscCall(PetscInfo(fllop, "A is symmetric by type\n"));
    PetscFunctionReturn(0);
  } else {
    PetscCall(PetscInfo(fllop, "A is NOT symmetric by type\n"));
  }

  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatCompleteCtxCreate(A, &ctx));
  PetscCall(PetscContainerCreate(comm, &container));
  PetscCall(PetscContainerSetPointer(container, ctx));
  PetscCall(PetscContainerSetUserDestroy(container, (PetscErrorCode (*)(void*))MatCompleteCtxDestroy));
  PetscCall(PetscObjectCompose((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  A->ops->mult              = MatMult_Complete;
  A->ops->multtranspose     = MatMult_Complete;
  A->ops->multadd           = MatMultAdd_Complete;
  A->ops->multtransposeadd  = MatMultAdd_Complete;
  A->ops->duplicate         = MatDuplicate_Complete;
  PetscFunctionReturn(0);
}

