
#include <private/fllopmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_Complete"
PetscErrorCode MatMult_Complete(Mat A, Vec x, Vec y)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  
  PetscFunctionBegin;
  TRY( PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container) );
  TRY( PetscContainerGetPointer(container, (void**)&ctx) );
  TRY( VecPointwiseMult(y,x,ctx->d) );
  TRY( VecScale(y, -1.0) );
  TRY( (ctx->multadd)(A,x,y,y) );
  TRY( (ctx->multtransposeadd)(A,x,y,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Complete"
PetscErrorCode MatMultAdd_Complete(Mat A, Vec x, Vec x1, Vec y)
{
  MatCompleteCtx ctx;
  PetscContainer container;
  
  PetscFunctionBegin;
  TRY( PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container) );
  TRY( PetscContainerGetPointer(container, (void**)&ctx) );
  TRY( VecPointwiseMult(y,x,ctx->d) );
  TRY( VecScale(y, -1.0) );
  TRY( (ctx->multadd)(A,x,y,y) );
  TRY( (ctx->multtransposeadd)(A,x,y,y) );
  TRY( VecAXPY(y, 1.0, x1) );
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
  TRY( PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container) );
  TRY( PetscContainerGetPointer(container, (void**)&ctx) );
  TRY( (ctx->duplicate)(A,op,&_M) );
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
  TRY( PetscNew(&ctx) );
  ctx->mult             = A->ops->mult;
  ctx->multtranspose    = A->ops->multtranspose;
  ctx->multadd          = A->ops->multadd;
  ctx->multtransposeadd = A->ops->multtransposeadd;
  ctx->duplicate        = A->ops->duplicate;
  TRY( MatCreateVecs(A, NULL, &ctx->d) );
  TRY( MatGetDiagonal(A, ctx->d) );
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCompleteCtxDestroy"
PetscErrorCode MatCompleteCtxDestroy(MatCompleteCtx ctx)
{
  PetscFunctionBegin;
  TRY( VecDestroy(&ctx->d) );
  TRY( PetscFree(ctx) );
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
  TRY( PetscObjectQuery((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject*)&container) );
  if (container) PetscFunctionReturn(0);

  TRY( MatIsSymmetricByType(A, &flg) );
  if (flg) {
    TRY( PetscInfo(fllop, "A is symmetric by type\n") );
    PetscFunctionReturn(0);
  } else {
    TRY( PetscInfo(fllop, "A is NOT symmetric by type\n") );
  }

  TRY( PetscObjectGetComm((PetscObject)A, &comm) );
  TRY( MatCompleteCtxCreate(A, &ctx) );
  TRY( PetscContainerCreate(comm, &container) );
  TRY( PetscContainerSetPointer(container, ctx) );
  TRY( PetscContainerSetUserDestroy(container, (PetscErrorCode (*)(void*))MatCompleteCtxDestroy) );
  TRY( PetscObjectCompose((PetscObject)A, "fllop_mat_complete_ctx", (PetscObject)container) );
  TRY( PetscContainerDestroy(&container) );
  A->ops->mult              = MatMult_Complete;
  A->ops->multtranspose     = MatMult_Complete;
  A->ops->multadd           = MatMultAdd_Complete;
  A->ops->multtransposeadd  = MatMultAdd_Complete;
  A->ops->duplicate         = MatDuplicate_Complete;
  PetscFunctionReturn(0);
}

