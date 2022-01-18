
#include <permon/private/permonmatimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatMult_Timer"
PetscErrorCode MatMult_Timer(Mat W, Vec x, Vec y) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    TRY( MatShellGetContext(W, (void*) &ctx) );
    TRY( PetscLogEventBegin(ctx->events[MATOP_MULT],ctx->A,x,y,0) );
    TRY( MatMult(ctx->A,x,y) );
    TRY( PetscLogEventEnd(  ctx->events[MATOP_MULT],ctx->A,x,y,0) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Timer"
PetscErrorCode MatMultAdd_Timer(Mat W, Vec x, Vec y, Vec z) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    TRY( MatShellGetContext(W, (void*) &ctx) );
    TRY( PetscLogEventBegin(ctx->events[MATOP_MULT_ADD],ctx->A,x,y,0) );
    TRY( MatMultAdd(ctx->A,x,y,z) );
    TRY( PetscLogEventEnd(  ctx->events[MATOP_MULT_ADD],ctx->A,x,y,0) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Timer"
PetscErrorCode MatMultTranspose_Timer(Mat W, Vec x, Vec y) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    TRY( MatShellGetContext(W, (void*) &ctx) );
    TRY( PetscLogEventBegin(ctx->events[MATOP_MULT_TRANSPOSE],ctx->A,x,y,0) );
    TRY( MatMultTranspose(ctx->A,x,y) );
    TRY( PetscLogEventEnd(  ctx->events[MATOP_MULT_TRANSPOSE],ctx->A,x,y,0) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Timer"
PetscErrorCode MatMultTransposeAdd_Timer(Mat W, Vec x, Vec y, Vec z) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    TRY( MatShellGetContext(W, (void*) &ctx) );
    TRY( PetscLogEventBegin(ctx->events[MATOP_MULT_TRANSPOSE_ADD],ctx->A,x,y,0) );
    TRY( MatMultTransposeAdd(ctx->A,x,y,z) );
    TRY( PetscLogEventEnd(  ctx->events[MATOP_MULT_TRANSPOSE_ADD],ctx->A,x,y,0) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Timer"
PetscErrorCode MatDestroy_Timer(Mat W) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    TRY( MatShellGetContext(W, (void*) &ctx) );
    TRY( MatDestroy(&ctx->A) );
    TRY( PetscFree(ctx) );
    TRY( MatShellSetContext(W, NULL) );
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateTimer"
PetscErrorCode MatCreateTimer(Mat A, Mat *W_inout) {
    Mat_Timer *ctx;
    Mat W;
    
    PetscFunctionBegin;
    TRY( PetscMalloc(sizeof(Mat_Timer),&ctx) );
    ctx->A = A;
    TRY( PetscObjectReference((PetscObject)A) );
    
    TRY( MatCreateShellPermon(PetscObjectComm((PetscObject)A), A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N, ctx,&W) );
    TRY( FllopPetscObjectInheritName((PetscObject)W,(PetscObject)A,NULL) );

    TRY( MatShellSetOperation(W,MATOP_DESTROY,(void(*)(void))MatDestroy_Timer) );
    TRY( MatTimerSetOperation(W,MATOP_MULT,"MatMult",(void(*)(void))MatMult_Timer) );
    TRY( MatTimerSetOperation(W,MATOP_MULT_ADD,"MatMultAdd",(void(*)(void))MatMultAdd_Timer) );
    TRY( MatTimerSetOperation(W,MATOP_MULT_TRANSPOSE,"MatMultTr",(void(*)(void))MatMultTranspose_Timer) );
    TRY( MatTimerSetOperation(W,MATOP_MULT_TRANSPOSE_ADD,"MatMultTrAdd",(void(*)(void))MatMultTransposeAdd_Timer) );
    
    *W_inout = W;
    PetscFunctionReturn(0);
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
  TRY( MatShellGetContext(mat,(void*)&ctx) );
  TRY( PetscObjectGetName((PetscObject)ctx->A,&name) );

  TRY( PetscStrcpy(eventName, opname) );
  TRY( PetscStrcat(eventName, "_") );
  TRY( PetscStrcat(eventName, name) );

  TRY( FllopPetscLogEventGetId(eventName,&event,&exists) );
  if (!exists) TRY( PetscLogEventRegister(eventName, MAT_CLASSID, &event) );
  ctx->events[op] = event;

  TRY( MatShellSetOperation(mat,op,opf) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTimerGetMat"
PetscErrorCode MatTimerGetMat(Mat W, Mat *A) {
    Mat_Timer *ctx;
    PetscFunctionBegin;
    TRY( MatShellGetContext(W,(void*)&ctx) );
    *A = ctx->A;
    PetscFunctionReturn(0);
}

