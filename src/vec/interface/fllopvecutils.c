#include <fllopvec.h>
#include <private/fllopimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/isimpl.h>

#undef __FUNCT__
#define __FUNCT__ "ISAdd"
PetscErrorCode ISAdd(IS is,PetscInt value,IS *isnew)
{
  PetscInt i,n;
  const PetscInt *idx_read;
  PetscInt *idx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(isnew,3);
  TRY( ISGetLocalSize(is,&n) );
  TRY( ISGetIndices(is,&idx_read) );
  TRY( PetscMalloc1(n,&idx) );
  for (i=0; i<n; i++) {
    idx[i] = idx_read[i] + value;
  }
  TRY( ISRestoreIndices(is,&idx_read) );
  TRY( ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idx,PETSC_OWN_POINTER,isnew) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMergeAndDestroy"
PetscErrorCode VecMergeAndDestroy(MPI_Comm comm, Vec *local_in, Vec *global_out)
{
  Vec local, global;
  PetscInt n;
  PetscBool any_nonnull;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidPointer(local_in,2);
  PetscValidPointer(global_out,3);
  local = *local_in;
  TRY( MPI_Comm_size(comm,&size) );
  TRY( PetscBoolGlobalOr(comm, local ? PETSC_TRUE : PETSC_FALSE, &any_nonnull) );
  if (!any_nonnull) { /* all local vecs are null => global vec is null */
    *global_out = NULL;
    PetscFunctionReturn(0);
  }
  if (size == 1) { /* seq. case => return the orig. matrix */
    *global_out = *local_in;
    *local_in = NULL;
    PetscFunctionReturn(0);
  }
  if (!local) { /* my local vec is null */
    n = 0;
  } else { /* my local vec is non-null */
    PetscValidHeaderSpecific(local, VEC_CLASSID, 2);
    TRY( VecGetLocalSize(local, &n) );
  }
  TRY( VecCreateMPI(comm, n, PETSC_DECIDE, &global) );
  if (n) {
    PetscScalar *global_arr, *local_arr;
    TRY( VecGetArray(    local,   &local_arr) );
    TRY( VecGetArray(    global,  &global_arr) );
    TRY( PetscMemcpy(global_arr, local_arr, n*sizeof(PetscScalar)) );
    TRY( VecRestoreArray(local,   &local_arr) );
    TRY( VecRestoreArray(global,  &global_arr) );
  }
  TRY( VecDestroy(local_in) );
  *global_out = global;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPrintInfo"
PetscErrorCode VecPrintInfo(Vec vec)
{
  PetscInt m, M, i, tablevel;
  const char *name, *type;
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  if (!FllopObjectInfoEnabled) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  TRY( PetscObjectGetTabLevel((PetscObject)vec, &tablevel) );
  for (i=0; i<tablevel; i++) {
    TRY( PetscPrintf(comm, "  ") );
  }

  if (vec == NULL) {
    TRY( PetscPrintf(comm, "Vec NULL\n") );
    PetscFunctionReturn(0);
  }
  TRY( VecGetSize(vec, &M) );
  TRY( VecGetLocalSize(vec, &m) );
  TRY( VecGetType(vec, &type) );
  TRY( PetscObjectGetName((PetscObject) vec, &name) );
  TRY( PetscPrintf(comm,
      "Vec %8x %-16s %-10s size(m,  M  )=[%6d %10d]\n",
          vec,name, type,                m,  M) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISCreateFromVec"
PetscErrorCode ISCreateFromVec(Vec vec, IS *is)
{
  PetscInt    n, i;
  PetscScalar *a;
  PetscInt    *ia;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidPointer(is,2);
  TRY( VecGetLocalSize(vec, &n) );
  TRY( PetscMalloc(n*sizeof(PetscInt), &ia) );
    
  TRY( VecGetArray(    vec, &a) );
  for (i=0; i<n; i++) ia[i] = (PetscInt) a[i];
  TRY( VecRestoreArray(vec, &a) );
    
  TRY( ISCreateGeneral(PetscObjectComm((PetscObject)vec), n, ia, PETSC_OWN_POINTER, is) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCreateFromIS"
PetscErrorCode VecCreateFromIS(IS is, Vec *vecout)
{
  PetscInt    n,N,i;
  PetscScalar *a;
  const PetscInt *ia;
  Vec         vec;
    
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(vecout,2);
  TRY( ISGetLocalSize(is, &n) );
  TRY( ISGetSize(is, &N) );
  TRY( VecCreate(PetscObjectComm((PetscObject)is),&vec) );
  TRY( VecSetSizes(vec,n,N) );
  TRY( VecSetType(vec,VECSTANDARD) );

  TRY( ISGetIndices(    is, &ia) );
  TRY( VecGetArray(    vec, &a) );
  for (i=0; i<n; i++) a[i] = (PetscScalar) ia[i];
  TRY( ISRestoreIndices(is, &ia) );
  TRY( VecRestoreArray(vec, &a) );

  *vecout = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGetVec"
/*
ISGetVec - set the layout of vector subject to the layout defined by index set

Parameters:
+ is - index set
- vec - given vector 
*/
PetscErrorCode ISGetVec(IS is, Vec *vec)
{
  PetscInt m,M,bs;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(vec,2);
  TRY( VecCreate(PetscObjectComm((PetscObject)is),vec) );
  TRY( ISGetLocalSize(is,&m) );
  TRY( ISGetSize(is,&M) );
  TRY( ISGetBlockSize(is,&bs) );
  TRY( VecSetSizes(*vec,m,M) );
  TRY( VecSetBlockSize(*vec,bs) );
  TRY( VecSetType(*vec,VECSTANDARD) );
  TRY( PetscLayoutReference(is->map,&(*vec)->map) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGetVecBlock"
/*
ISGetVecBlock - set the layout of vector subject to the layout defined by index set; each block of IS has one component in vector

Parameters:
+ is - index set with
. vec - given vector 
- bs - the size of block
*/
PetscErrorCode ISGetVecBlock(IS is, Vec *vec, PetscInt bs)
{
  PetscInt M; /* global size of is = (global size of vec * block size) */ 
  PetscInt m; /* local size of is */
  
  PetscInt M_vec; /* global size of vec */
  PetscInt m_vec; /* local size of vec */
  
  PetscFunctionBegin;

  /* control the validity of given objects*/
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(vec,2);

  /* create the vector */
  TRY( VecCreate(PetscObjectComm((PetscObject)is),vec) );

  /* get the properties of index set */
  TRY( ISGetLocalSize(is,&m) );
  TRY( ISGetSize(is,&M) );

  /* set size of vector */
  M_vec = M/bs; /* the dimension of each block is reduced to 1 */
  m_vec = m/bs; 
  
  TRY( VecSetSizes(*vec,m_vec,M_vec) );
  TRY( VecSetBlockSize(*vec,1) ); /* the block size is 1 */
  TRY( VecSetType(*vec,VECSTANDARD) );

  TRY( PetscLayoutReference(is->map,&(*vec)->map) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCheckSameLayoutIS"
PetscErrorCode VecCheckSameLayoutIS(Vec vec, IS is)
{
  PetscInt n,N,bs;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscCheckSameComm(vec,1,is,2);
  TRY( ISGetLocalSize(is,&n) );
  TRY( ISGetSize(is,&N) );
  TRY( ISGetBlockSize(is,&bs) );
  if (vec->map->n   != n)  FLLOP_SETERRQ2(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"Vec local size %d != IS local size %d",  vec->map->n,  n);
  if (vec->map->N   != N)  FLLOP_SETERRQ2(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"Vec global size %d != IS global size %d",vec->map->N,  N);
  if (vec->map->bs  != bs) FLLOP_SETERRQ2(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"Vec block size %d != IS block size %d",  vec->map->bs, bs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCheckSameLayoutVec"
PetscErrorCode VecCheckSameLayoutVec(Vec v1, Vec v2)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,1);
  PetscCheckSameComm(v1,1,v2,2);
  if (v1->map->n    != v2->map->n)  FLLOP_SETERRQ2(PetscObjectComm((PetscObject)v1),PETSC_ERR_ARG_INCOMP,"Vec #1 local size %d != Vec #2 local size %d",  v1->map->n, v2->map->n);
  if (v1->map->N    != v2->map->N)  FLLOP_SETERRQ2(PetscObjectComm((PetscObject)v1),PETSC_ERR_ARG_INCOMP,"Vec #1 global size %d != Vec #2 global size %d",v1->map->N, v2->map->N);
  if (v1->map->bs   != v2->map->bs) FLLOP_SETERRQ2(PetscObjectComm((PetscObject)v1),PETSC_ERR_ARG_INCOMP,"Vec #1 block size %d != Vec #2 block size %d",  v1->map->bs,v2->map->bs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecInvalidate"
PetscErrorCode VecInvalidate(Vec vec)
{
  PetscContainer container;
  StateContainer sc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  TRY( VecSetInf(vec) );

  TRY( PetscContainerCreate(PetscObjectComm((PetscObject)vec),&container) );
  TRY( PetscNew(&sc) );
  TRY( PetscObjectStateGet((PetscObject)vec,&sc->state) );
  TRY( PetscContainerSetPointer(container,(void*)sc) );
  TRY( PetscObjectCompose((PetscObject)vec,"VecInvalidState",(PetscObject)container) );
  TRY( PetscContainerDestroy(&container) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecIsInvalidated"
PetscErrorCode VecIsInvalidated(Vec vec,PetscBool *flg)
{
  PetscContainer container;
  StateContainer sc;
  PetscObjectState state;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidPointer(flg,2);
  TRY( PetscObjectQuery((PetscObject)vec,"VecInvalidState",(PetscObject*)&container) );
  if (!container) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  TRY( PetscContainerGetPointer(container,(void**)&sc) );
  TRY( PetscObjectStateGet((PetscObject)vec,&state) );
  if (state > sc->state) {
    *flg = PETSC_FALSE;
    TRY( PetscObjectCompose((PetscObject)vec,"VecInvalidState",NULL) );
  } else {
    *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecIsValid"
PetscErrorCode VecIsValid(Vec vec,PetscBool *flg)
{
  PetscBool flg_;
  PetscFunctionBegin;
  TRY( VecIsInvalidated(vec,&flg_) );
  *flg = !flg_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecHasValidValues"
PetscErrorCode VecHasValidValues(Vec vec,PetscBool *flg)
{
  PetscInt          n,i;
  const PetscScalar *x;
  PetscBool tflg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscValidPointer(flg,2);
  tflg = PETSC_TRUE;
  TRY( VecGetLocalSize(vec,&n) );
  TRY( VecGetArrayRead(vec,&x) );
  for (i=0; i<n; i++) {
    if (PetscIsInfOrNanScalar(x[i])) {
      tflg = PETSC_FALSE;
    }
  }
  TRY( VecRestoreArrayRead(vec,&x) );
  TRY( PetscBoolGlobalAnd(PetscObjectComm((PetscObject)vec),tflg,flg) );
  PetscFunctionReturn(0);
}


struct _n_VecNestGetMPICtx {
  VecScatter sc;
  Vec *origvecs;
};
typedef struct _n_VecNestGetMPICtx *VecNestGetMPICtx;

#undef __FUNCT__
#define __FUNCT__ "VecGetMPIVector"
PetscErrorCode   VecGetMPIVector(MPI_Comm comm, PetscInt N,Vec vecs[], Vec *VecOut)
{  
  IS isl, isg;
  VecScatter sc;
  Vec locNest, *origVec, glVec;
  VecNestGetMPICtx ctx;
  PetscContainer container;
  PetscInt i;
  
  PetscFunctionBegin;
  FLLOP_ASSERT(N>0,"N>0");
  
  *VecOut = NULL;
      
  for (i = 0; i < N; i++) if(!vecs[i]) PetscFunctionReturn(0);
  
  TRY( VecCreateNest(PETSC_COMM_SELF, N, NULL, vecs, &locNest) );  
  
  TRY( VecCreateMPI(comm,locNest->map->N,PETSC_DECIDE,&glVec) );
  TRY( ISCreateStride(comm,locNest->map->N,0,1,&isl) );
  TRY( ISCreateStride(comm,locNest->map->N,glVec->map->rstart,1,&isg) );
  TRY( VecScatterCreate(locNest,isl,glVec,isg,&sc) );
  TRY( ISDestroy(&isl) ); TRY( ISDestroy(&isg) );
  
  TRY( VecScatterBegin(sc,locNest,glVec,INSERT_VALUES,SCATTER_FORWARD_LOCAL) );
  TRY( VecScatterEnd(  sc,locNest,glVec,INSERT_VALUES,SCATTER_FORWARD_LOCAL) );
  
  TRY( PetscNew(&ctx) );
  TRY( PetscMalloc(sizeof(Vec), &origVec) ); 
  ctx->sc = sc;
  origVec[0]=locNest;
  ctx->origvecs = origVec;
  
  TRY( PetscContainerCreate(comm, &container) );
  TRY( PetscContainerSetPointer(container, ctx) );
  TRY( PetscObjectCompose((PetscObject)glVec,"VecGetMPIVector_context",(PetscObject)container) );
  TRY( PetscContainerDestroy(&container) );
    
  *VecOut = glVec;
   
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreMPIVector"
PetscErrorCode VecRestoreMPIVector(MPI_Comm comm, PetscInt N,Vec vecs[], Vec *VecIn)
{
  Vec locNest;
  VecScatter sc;
  PetscContainer container;
  VecNestGetMPICtx ctx;
  
  PetscFunctionBegin;
  vecs = NULL;
  TRY( PetscObjectQuery((PetscObject)(*VecIn),"VecGetMPIVector_context",(PetscObject*)&container) );
  if (!container) PetscFunctionReturn(0);

  TRY( PetscContainerGetPointer(container,(void**)&ctx) );
  sc = ctx->sc;
  locNest = ctx->origvecs[0];
  TRY( VecScatterBegin(sc,*VecIn,locNest,INSERT_VALUES,SCATTER_REVERSE_LOCAL) );
  TRY( VecScatterEnd(  sc,*VecIn,locNest,INSERT_VALUES,SCATTER_REVERSE_LOCAL) );
  
  TRY( PetscObjectCompose((PetscObject)(*VecIn),"VecGetMPIVector_context",NULL) ); 
  TRY( VecScatterDestroy(&sc) );
  TRY( VecDestroy(&locNest) );
  TRY( VecDestroy(VecIn) );
  TRY( PetscFree(ctx->origvecs) );
  TRY( PetscFree(ctx) ); 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecNestGetMPI"
PetscErrorCode VecNestGetMPI(PetscInt N,Vec *vecs[])
{
  Vec x,y;
  Vec *nestv=*vecs, *mpiv;
  VecScatter sc;
  IS ix;
  PetscInt j;
  VecNestGetMPICtx ctx;
  PetscContainer container;
  MPI_Comm comm;
  PetscBool flg;

  FllopTracedFunctionBegin;
  if (!N) PetscFunctionReturn(0);

  x=nestv[0];
  TRY( PetscObjectTypeCompare((PetscObject)x,VECMPI,&flg) );
  if (flg) PetscFunctionReturn(0);

  FllopTraceBegin;
  TRY( PetscObjectGetComm((PetscObject)x,&comm) );
  TRY( PetscNew(&ctx) );

  TRY( VecCreateMPI(comm,x->map->n,x->map->N,&y) );
  TRY( VecDuplicateVecs(y,N,&mpiv) );
  TRY( VecDestroy(&y) );
  y=mpiv[0];
  
  TRY( ISCreateStride(comm,x->map->n,x->map->rstart,1,&ix) );
  TRY( VecScatterCreate(x,ix,y,ix,&sc) );
  TRY( ISDestroy(&ix) );

  for (j=0; j<N; j++) {
    TRY( VecScatterBegin(sc,nestv[j],mpiv[j],INSERT_VALUES,SCATTER_FORWARD_LOCAL) );
    TRY( VecScatterEnd(  sc,nestv[j],mpiv[j],INSERT_VALUES,SCATTER_FORWARD_LOCAL) );
  }

  ctx->sc = sc;
  ctx->origvecs = nestv;
  TRY( PetscContainerCreate(comm, &container) );
  TRY( PetscContainerSetPointer(container, ctx) );
  TRY( PetscObjectCompose((PetscObject)y,"VecNestGetMPI_context",(PetscObject)container) );
  TRY( PetscContainerDestroy(&container) );
  *vecs = mpiv;
  PetscFunctionReturnI(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecNestRestoreMPI"
PetscErrorCode VecNestRestoreMPI(PetscInt N,Vec *vecs[])
{
  Vec y;
  Vec *mpiv = *vecs;
  Vec *nestv;
  VecScatter sc;
  PetscInt j;
  PetscContainer container;
  VecNestGetMPICtx ctx;

  FllopTracedFunctionBegin;
  if (!N) PetscFunctionReturn(0);

  y=mpiv[0];
  TRY( PetscObjectQuery((PetscObject)y,"VecNestGetMPI_context",(PetscObject*)&container) );
  if (!container) PetscFunctionReturn(0);

  FllopTraceBegin;
  TRY( PetscContainerGetPointer(container,(void**)&ctx) );
  sc = ctx->sc;
  nestv = ctx->origvecs;

  for (j=0; j<N; j++) {
    TRY( VecScatterBegin(sc,mpiv[j],nestv[j],INSERT_VALUES,SCATTER_REVERSE_LOCAL) );
    TRY( VecScatterEnd(  sc,mpiv[j],nestv[j],INSERT_VALUES,SCATTER_REVERSE_LOCAL) );
  }
  TRY( PetscObjectCompose((PetscObject)y,"VecNestGetMPI_context",NULL) );
  TRY( VecDestroyVecs(N,vecs) );
  TRY( VecScatterDestroy(&sc) );
  TRY( PetscFree(ctx) );
  *vecs = nestv;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecScaleSkipInf"
PetscErrorCode VecScaleSkipInf(Vec x,PetscScalar alpha)
{
  PetscInt i,n;
  PetscScalar *varr,v;

  PetscFunctionBeginI;
  if (alpha == 1.0) PetscFunctionReturnI(0);
  TRY( VecGetLocalSize(x,&n) );
  TRY( VecGetArray(x,&varr) );
  for (i=0; i<n; i++) {
    v = varr[i];
    if( PetscAbsScalar(v) < PETSC_INFINITY) {
      varr[i] = alpha*v;
    }
  }
  TRY( VecRestoreArray(x,&varr) );
  PetscFunctionReturnI(0);
}
