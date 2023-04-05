
#include <permon/private/permonmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_Timer"
PetscErrorCode MatMult_Timer(Mat W, Vec x, Vec y) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    PetscCall(MatShellGetContext(W, (void*) &ctx));
    PetscCall(PetscLogEventBegin(ctx->events[MATOP_MULT],ctx->A,x,y,0));
    PetscCall(MatMult(ctx->A,x,y));
    PetscCall(PetscLogEventEnd(  ctx->events[MATOP_MULT],ctx->A,x,y,0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Timer"
PetscErrorCode MatMultAdd_Timer(Mat W, Vec x, Vec y, Vec z) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    PetscCall(MatShellGetContext(W, (void*) &ctx));
    PetscCall(PetscLogEventBegin(ctx->events[MATOP_MULT_ADD],ctx->A,x,y,0));
    PetscCall(MatMultAdd(ctx->A,x,y,z));
    PetscCall(PetscLogEventEnd(  ctx->events[MATOP_MULT_ADD],ctx->A,x,y,0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Timer"
PetscErrorCode MatMultTranspose_Timer(Mat W, Vec x, Vec y) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    PetscCall(MatShellGetContext(W, (void*) &ctx));
    PetscCall(PetscLogEventBegin(ctx->events[MATOP_MULT_TRANSPOSE],ctx->A,x,y,0));
    PetscCall(MatMultTranspose(ctx->A,x,y));
    PetscCall(PetscLogEventEnd(  ctx->events[MATOP_MULT_TRANSPOSE],ctx->A,x,y,0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Timer"
PetscErrorCode MatMultTransposeAdd_Timer(Mat W, Vec x, Vec y, Vec z) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    PetscCall(MatShellGetContext(W, (void*) &ctx));
    PetscCall(PetscLogEventBegin(ctx->events[MATOP_MULT_TRANSPOSE_ADD],ctx->A,x,y,0));
    PetscCall(MatMultTransposeAdd(ctx->A,x,y,z));
    PetscCall(PetscLogEventEnd(  ctx->events[MATOP_MULT_TRANSPOSE_ADD],ctx->A,x,y,0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Timer"
PetscErrorCode MatDestroy_Timer(Mat W) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    PetscCall(MatShellGetContext(W, (void*) &ctx));
    PetscCall(MatDestroy(&ctx->A));
    PetscCall(PetscFree(ctx));
    PetscCall(MatShellSetContext(W, NULL));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateTimer"
/*@
   MatCreateTimer - Creates a matrix that behaves like original but logs all MatMult operations

   Collective

   Input Parameters:
.  A - original matrix

   Output Parameters:
.  B - matrix A that logs MatMult operations 

   Level: developer

.seealso MatTimerSetOperation(), MatTimerGetMat()
@*/
PetscErrorCode MatCreateTimer(Mat A, Mat *B) {
    Mat_Timer *ctx;
    Mat W;
    
    PetscFunctionBegin;
    PetscCall(PetscMalloc(sizeof(Mat_Timer),&ctx));
    ctx->A = A;
    PetscCall(PetscObjectReference((PetscObject)A));
    
    PetscCall(MatCreateShellPermon(PetscObjectComm((PetscObject)A), A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N, ctx,&W));
    PetscCall(FllopPetscObjectInheritName((PetscObject)W,(PetscObject)A,NULL));

    PetscCall(MatShellSetOperation(W,MATOP_DESTROY,(void(*)(void))MatDestroy_Timer));
    PetscCall(MatTimerSetOperation(W,MATOP_MULT,"MatMult",(void(*)(void))MatMult_Timer));
    PetscCall(MatTimerSetOperation(W,MATOP_MULT_ADD,"MatMultAdd",(void(*)(void))MatMultAdd_Timer));
    PetscCall(MatTimerSetOperation(W,MATOP_MULT_TRANSPOSE,"MatMultTr",(void(*)(void))MatMultTranspose_Timer));
    PetscCall(MatTimerSetOperation(W,MATOP_MULT_TRANSPOSE_ADD,"MatMultTrAdd",(void(*)(void))MatMultTransposeAdd_Timer));
    
    *B = W;
    PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatTimerSetOperation"
PetscErrorCode MatTimerSetOperation(Mat mat, MatOperation op, const char *opname, void(*opf)(void))
{
  Mat_Timer *ctx;
  const char *name;
  char *eventName = FLLOP_ObjNameBuffer_Global;
  PetscLogEvent event;
  PetscBool exists;
  
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,(void*)&ctx));
  PetscCall(PetscObjectGetName((PetscObject)ctx->A,&name));

  PetscCall(PetscStrcpy(eventName, opname));
  PetscCall(PetscStrcat(eventName, "_"));
  PetscCall(PetscStrcat(eventName, name));

  PetscCall(FllopPetscLogEventGetId(eventName,&event,&exists));
  if (!exists) PetscCall(PetscLogEventRegister(eventName, MAT_CLASSID, &event));
  ctx->events[op] = event;

  PetscCall(MatShellSetOperation(mat,op,opf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatTimerGetMat"
PetscErrorCode MatTimerGetMat(Mat W, Mat *A) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    PetscCall(MatShellGetContext(W,(void*)&ctx));
    *A = ctx->A;
    PetscFunctionReturn(PETSC_SUCCESS);
}

