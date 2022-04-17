
#include <permon/private/permonmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_Complete"
PetscErrorCode MatMult_Complete(Mat A, Vec x, Vec y)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  
  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  CHKERRQ(PetscContainerGetPointer(container, (void**)&ctx));
  CHKERRQ(VecPointwiseMult(y,x,ctx->d));
  CHKERRQ(VecScale(y, -1.0));
  CHKERRQ((ctx->multadd)(A,x,y,y));
  CHKERRQ((ctx->multtransposeadd)(A,x,y,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Complete"
PetscErrorCode MatMultAdd_Complete(Mat A, Vec x, Vec x1, Vec y)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  
  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  CHKERRQ(PetscContainerGetPointer(container, (void**)&ctx));
  CHKERRQ(VecPointwiseMult(y,x,ctx->d));
  CHKERRQ(VecScale(y, -1.0));
  CHKERRQ((ctx->multadd)(A,x,y,y));
  CHKERRQ((ctx->multtransposeadd)(A,x,y,y));
  CHKERRQ(VecAXPY(y, 1.0, x1));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_Complete"
PetscErrorCode MatDuplicate_Complete(Mat A,MatDuplicateOption op,Mat *M)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  Mat _M;
  
  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  CHKERRQ(PetscContainerGetPointer(container, (void**)&ctx));
  CHKERRQ((ctx->duplicate)(A,op,&_M));
  _M->ops->mult              = ctx->mult;
  _M->ops->multtranspose     = ctx->multtranspose;
  _M->ops->multadd           = ctx->multadd;
  _M->ops->multtransposeadd  = ctx->multtransposeadd;
  _M->ops->duplicate         = ctx->duplicate;
  *M = _M;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCompleteCtxCreate"
PetscErrorCode MatCompleteCtxCreate(Mat A, MatCompleteCtx *ctxout)
{
  MatCompleteCtx ctx;
  PetscFunctionBegin;
  CHKERRQ(PetscNew(&ctx));
  ctx->mult             = A->ops->mult;
  ctx->multtranspose    = A->ops->multtranspose;
  ctx->multadd          = A->ops->multadd;
  ctx->multtransposeadd = A->ops->multtransposeadd;
  ctx->duplicate        = A->ops->duplicate;
  CHKERRQ(MatCreateVecs(A, NULL, &ctx->d));
  CHKERRQ(MatGetDiagonal(A, ctx->d));
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCompleteCtxDestroy"
PetscErrorCode MatCompleteCtxDestroy(MatCompleteCtx ctx)
{
  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&ctx->d));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCompleteFromUpperTriangular"
PetscErrorCode MatCompleteFromUpperTriangular(Mat A)
{
  MPI_Comm comm;
  MatCompleteCtx ctx;
  PetscContainer container;
  PetscBool flg;
  
  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container));
  if (container) PetscFunctionReturn(0);

  CHKERRQ(MatIsSymmetricByType(A, &flg));
  if (flg) {
    CHKERRQ(PetscInfo(fllop, "A is symmetric by type\n"));
    PetscFunctionReturn(0);
  } else {
    CHKERRQ(PetscInfo(fllop, "A is NOT symmetric by type\n"));
  }

  CHKERRQ(PetscObjectGetComm((PetscObject)A, &comm));
  CHKERRQ(MatCompleteCtxCreate(A, &ctx));
  CHKERRQ(PetscContainerCreate(comm, &container));
  CHKERRQ(PetscContainerSetPointer(container, ctx));
  CHKERRQ(PetscContainerSetUserDestroy(container, (PetscErrorCode (*)(void*))MatCompleteCtxDestroy));
  CHKERRQ(PetscObjectCompose((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject)container));
  CHKERRQ(PetscContainerDestroy(&container));
  A->ops->mult              = MatMult_Complete;
  A->ops->multtranspose     = MatMult_Complete;
  A->ops->multadd           = MatMultAdd_Complete;
  A->ops->multtransposeadd  = MatMultAdd_Complete;
  A->ops->duplicate         = MatDuplicate_Complete;
  PetscFunctionReturn(0);
}

