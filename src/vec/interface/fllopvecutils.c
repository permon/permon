#include <permonvec.h>
#include <permon/private/permonimpl.h>
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
  PetscAssertPointer(isnew,3);
  PetscCall(ISGetLocalSize(is,&n));
  PetscCall(ISGetIndices(is,&idx_read));
  PetscCall(PetscMalloc1(n,&idx));
  for (i=0; i<n; i++) {
    idx[i] = idx_read[i] + value;
  }
  PetscCall(ISRestoreIndices(is,&idx_read));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idx,PETSC_OWN_POINTER,isnew));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(local_in,2);
  PetscAssertPointer(global_out,3);
  local = *local_in;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscBoolGlobalOr(comm, local ? PETSC_TRUE : PETSC_FALSE, &any_nonnull));
  if (!any_nonnull) { /* all local vecs are null => global vec is null */
    *global_out = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (size == 1) { /* seq. case => return the orig. matrix */
    *global_out = *local_in;
    *local_in = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!local) { /* my local vec is null */
    n = 0;
  } else { /* my local vec is non-null */
    PetscValidHeaderSpecific(local, VEC_CLASSID, 2);
    PetscCall(VecGetLocalSize(local, &n));
  }
  PetscCall(VecCreateMPI(comm, n, PETSC_DECIDE, &global));
  if (n) {
    PetscScalar *global_arr, *local_arr;
    PetscCall(VecGetArray(    local,   &local_arr));
    PetscCall(VecGetArray(    global,  &global_arr));
    PetscCall(PetscMemcpy(global_arr, local_arr, n*sizeof(PetscScalar)));
    PetscCall(VecRestoreArray(local,   &local_arr));
    PetscCall(VecRestoreArray(global,  &global_arr));
  }
  PetscCall(VecDestroy(local_in));
  *global_out = global;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "VecPrintInfo"
PetscErrorCode VecPrintInfo(Vec vec)
{
  PetscInt m, M, i, tablevel;
  const char *name, *type;
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  if (!FllopObjectInfoEnabled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCall(PetscObjectGetTabLevel((PetscObject)vec, &tablevel));
  for (i=0; i<tablevel; i++) {
    PetscCall(PetscPrintf(comm, "  "));
  }

  if (vec == NULL) {
    PetscCall(PetscPrintf(comm, "Vec NULL\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetSize(vec, &M));
  PetscCall(VecGetLocalSize(vec, &m));
  PetscCall(VecGetType(vec, &type));
  PetscCall(PetscObjectGetName((PetscObject) vec, &name));
  PetscCall(PetscPrintf(comm,
      "Vec %p %-16s %-10s size(m,  M  )=[%6d %10d]\n",
          (void*)vec,name, type,                m,  M));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(is,2);
  PetscCall(VecGetLocalSize(vec, &n));
  PetscCall(PetscMalloc(n*sizeof(PetscInt), &ia));

  PetscCall(VecGetArray(    vec, &a));
  for (i=0; i<n; i++) ia[i] = (PetscInt) a[i];
  PetscCall(VecRestoreArray(vec, &a));

  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)vec), n, ia, PETSC_OWN_POINTER, is));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(vecout,2);
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetSize(is, &N));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)is),&vec));
  PetscCall(VecSetSizes(vec,n,N));
  PetscCall(VecSetType(vec,VECSTANDARD));

  PetscCall(ISGetIndices(    is, &ia));
  PetscCall(VecGetArray(    vec, &a));
  for (i=0; i<n; i++) a[i] = (PetscScalar) ia[i];
  PetscCall(ISRestoreIndices(is, &ia));
  PetscCall(VecRestoreArray(vec, &a));

  *vecout = vec;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(vec,2);
  PetscCall(VecCreate(PetscObjectComm((PetscObject)is),vec));
  PetscCall(ISGetLocalSize(is,&m));
  PetscCall(ISGetSize(is,&M));
  PetscCall(ISGetBlockSize(is,&bs));
  PetscCall(VecSetSizes(*vec,m,M));
  PetscCall(VecSetBlockSize(*vec,bs));
  PetscCall(VecSetType(*vec,VECSTANDARD));
  PetscCall(PetscLayoutReference(is->map,&(*vec)->map));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(vec,2);

  /* create the vector */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)is),vec));

  /* get the properties of index set */
  PetscCall(ISGetLocalSize(is,&m));
  PetscCall(ISGetSize(is,&M));

  /* set size of vector */
  M_vec = M/bs; /* the dimension of each block is reduced to 1 */
  m_vec = m/bs;

  PetscCall(VecSetSizes(*vec,m_vec,M_vec));
  PetscCall(VecSetBlockSize(*vec,1)); /* the block size is 1 */
  PetscCall(VecSetType(*vec,VECSTANDARD));

  PetscCall(PetscLayoutReference(is->map,&(*vec)->map));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(ISGetLocalSize(is,&n));
  PetscCall(ISGetSize(is,&N));
  PetscCall(ISGetBlockSize(is,&bs));
  if (vec->map->n   != n)  SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"Vec local size %d != IS local size %d",  vec->map->n,  n);
  if (vec->map->N   != N)  SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"Vec global size %d != IS global size %d",vec->map->N,  N);
  if (vec->map->bs  != bs) SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_INCOMP,"Vec block size %d != IS block size %d",  vec->map->bs, bs);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "VecCheckSameLayoutVec"
PetscErrorCode VecCheckSameLayoutVec(Vec v1, Vec v2)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,1);
  PetscCheckSameComm(v1,1,v2,2);
  if (v1->map->n    != v2->map->n)  SETERRQ(PetscObjectComm((PetscObject)v1),PETSC_ERR_ARG_INCOMP,"Vec #1 local size %d != Vec #2 local size %d",  v1->map->n, v2->map->n);
  if (v1->map->N    != v2->map->N)  SETERRQ(PetscObjectComm((PetscObject)v1),PETSC_ERR_ARG_INCOMP,"Vec #1 global size %d != Vec #2 global size %d",v1->map->N, v2->map->N);
  if (v1->map->bs   != v2->map->bs) SETERRQ(PetscObjectComm((PetscObject)v1),PETSC_ERR_ARG_INCOMP,"Vec #1 block size %d != Vec #2 block size %d",  v1->map->bs,v2->map->bs);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "VecInvalidate"
/*@
   VecInvalidate - Mark vector invalid and set entries to Inf

   Logically Collective on Vec

   Input Parameters:
.  vec - vector to mark

  Level: Advanced

  Notes:
   Vector becomes valid whenever it is changed (PetscObjectState increased).

.seealso VecIsInvalidated()
@*/
PetscErrorCode VecInvalidate(Vec vec)
{
  PetscContainer container;
  PetscObjectState *state,vecstate;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCall(VecFlag(vec,PETSC_TRUE));

  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)vec),&container));
  PetscCall(PetscObjectStateGet((PetscObject)vec,&vecstate));
  PetscCall(PetscNew(&state));
  *state = vecstate;
  PetscCall(PetscContainerSetPointer(container,(void*)state));
  PetscCall(PetscContainerSetUserDestroy(container,PetscContainerUserDestroyDefault));
  PetscCall(PetscObjectCompose((PetscObject)vec,"VecInvalidState",(PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "VecIsInvalidated"
/*@
   VecIsInvalidated - Check if vector is invalid

   Not Collective

   Input Parameters:
.  vec - vector to mark

   Output Parameters:
.  flg - false if vec is valid

  Level: Advanced

.seealso VecInvalidate()
@*/
PetscErrorCode VecIsInvalidated(Vec vec,PetscBool *flg)
{
  PetscContainer container;
  PetscObjectState *state,vecstate;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscAssertPointer(flg,2);
  PetscCall(PetscObjectQuery((PetscObject)vec,"VecInvalidState",(PetscObject*)&container));
  if (!container) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscContainerGetPointer(container,(void**)&state));
  PetscCall(PetscObjectStateGet((PetscObject)vec,&vecstate));
  if (vecstate > *state) {
    *flg = PETSC_FALSE;
    PetscCall(PetscObjectCompose((PetscObject)vec,"VecInvalidState",NULL));
  } else {
    *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscAssertPointer(flg,2);
  tflg = PETSC_TRUE;
  PetscCall(VecGetLocalSize(vec,&n));
  PetscCall(VecGetArrayRead(vec,&x));
  for (i=0; i<n; i++) {
    if (PetscIsInfOrNanScalar(x[i])) {
      tflg = PETSC_FALSE;
    }
  }
  PetscCall(VecRestoreArrayRead(vec,&x));
  PetscCall(PetscBoolGlobalAnd(PetscObjectComm((PetscObject)vec),tflg,flg));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PERMON_ASSERT(N>0,"N>0");

  *VecOut = NULL;

  for (i = 0; i < N; i++) if(!vecs[i]) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecCreateNest(PETSC_COMM_SELF, N, NULL, vecs, &locNest));

  PetscCall(VecCreateMPI(comm,locNest->map->N,PETSC_DECIDE,&glVec));
  PetscCall(ISCreateStride(comm,locNest->map->N,0,1,&isl));
  PetscCall(ISCreateStride(comm,locNest->map->N,glVec->map->rstart,1,&isg));
  PetscCall(VecScatterCreate(locNest,isl,glVec,isg,&sc));
  PetscCall(ISDestroy(&isl)); PetscCall(ISDestroy(&isg));

  PetscCall(VecScatterBegin(sc,locNest,glVec,INSERT_VALUES,SCATTER_FORWARD_LOCAL));
  PetscCall(VecScatterEnd(  sc,locNest,glVec,INSERT_VALUES,SCATTER_FORWARD_LOCAL));

  PetscCall(PetscNew(&ctx));
  PetscCall(PetscMalloc(sizeof(Vec), &origVec));
  ctx->sc = sc;
  origVec[0]=locNest;
  ctx->origvecs = origVec;

  PetscCall(PetscContainerCreate(comm, &container));
  PetscCall(PetscContainerSetPointer(container, ctx));
  PetscCall(PetscObjectCompose((PetscObject)glVec,"VecGetMPIVector_context",(PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));

  *VecOut = glVec;
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(PetscObjectQuery((PetscObject)(*VecIn),"VecGetMPIVector_context",(PetscObject*)&container));
  if (!container) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscContainerGetPointer(container,(void**)&ctx));
  sc = ctx->sc;
  locNest = ctx->origvecs[0];
  PetscCall(VecScatterBegin(sc,*VecIn,locNest,INSERT_VALUES,SCATTER_REVERSE_LOCAL));
  PetscCall(VecScatterEnd(  sc,*VecIn,locNest,INSERT_VALUES,SCATTER_REVERSE_LOCAL));

  PetscCall(PetscObjectCompose((PetscObject)(*VecIn),"VecGetMPIVector_context",NULL));
  PetscCall(VecScatterDestroy(&sc));
  PetscCall(VecDestroy(&locNest));
  PetscCall(VecDestroy(VecIn));
  PetscCall(PetscFree(ctx->origvecs));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);

  x=nestv[0];
  PetscCall(PetscObjectTypeCompare((PetscObject)x,VECMPI,&flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  FllopTraceBegin;
  PetscCall(PetscObjectGetComm((PetscObject)x,&comm));
  PetscCall(PetscNew(&ctx));

  PetscCall(VecCreateMPI(comm,x->map->n,x->map->N,&y));
  PetscCall(VecDuplicateVecs(y,N,&mpiv));
  PetscCall(VecDestroy(&y));
  y=mpiv[0];

  PetscCall(ISCreateStride(comm,x->map->n,x->map->rstart,1,&ix));
  PetscCall(VecScatterCreate(x,ix,y,ix,&sc));
  PetscCall(ISDestroy(&ix));

  for (j=0; j<N; j++) {
    PetscCall(VecScatterBegin(sc,nestv[j],mpiv[j],INSERT_VALUES,SCATTER_FORWARD_LOCAL));
    PetscCall(VecScatterEnd(  sc,nestv[j],mpiv[j],INSERT_VALUES,SCATTER_FORWARD_LOCAL));
  }

  ctx->sc = sc;
  ctx->origvecs = nestv;
  PetscCall(PetscContainerCreate(comm, &container));
  PetscCall(PetscContainerSetPointer(container, ctx));
  PetscCall(PetscObjectCompose((PetscObject)y,"VecNestGetMPI_context",(PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  *vecs = mpiv;
  PetscFunctionReturnI(PETSC_SUCCESS);
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
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);

  y=mpiv[0];
  PetscCall(PetscObjectQuery((PetscObject)y,"VecNestGetMPI_context",(PetscObject*)&container));
  if (!container) PetscFunctionReturn(PETSC_SUCCESS);

  FllopTraceBegin;
  PetscCall(PetscContainerGetPointer(container,(void**)&ctx));
  sc = ctx->sc;
  nestv = ctx->origvecs;

  for (j=0; j<N; j++) {
    PetscCall(VecScatterBegin(sc,mpiv[j],nestv[j],INSERT_VALUES,SCATTER_REVERSE_LOCAL));
    PetscCall(VecScatterEnd(  sc,mpiv[j],nestv[j],INSERT_VALUES,SCATTER_REVERSE_LOCAL));
  }
  PetscCall(PetscObjectCompose((PetscObject)y,"VecNestGetMPI_context",NULL));
  PetscCall(VecDestroyVecs(N,vecs));
  PetscCall(VecScatterDestroy(&sc));
  PetscCall(PetscFree(ctx));
  *vecs = nestv;
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "VecScaleSkipInf"
PetscErrorCode VecScaleSkipInf(Vec x,PetscScalar alpha)
{
  PetscInt i,n;
  PetscScalar *varr,v;

  PetscFunctionBeginI;
  if (alpha == 1.0) PetscFunctionReturnI(PETSC_SUCCESS);
  PetscCall(VecGetLocalSize(x,&n));
  PetscCall(VecGetArray(x,&varr));
  for (i=0; i<n; i++) {
    v = varr[i];
    if( PetscAbsScalar(v) < PETSC_INFINITY) {
      varr[i] = alpha*v;
    }
  }
  PetscCall(VecRestoreArray(x,&varr));
  PetscFunctionReturnI(PETSC_SUCCESS);
}
