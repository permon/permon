
#include <../src/qp/impls/feti/qpfetiimpl.h>
#include <petscsf.h>

PetscLogEvent QP_Feti_SetUp, QP_Feti_AssembleDirichlet, QP_Feti_GetBgtSF, QP_Feti_GetOrthoBgtSF, QP_Feti_GetNotOrthoBgtSF;
PetscLogEvent QP_Feti_AssemGluing, QP_Feti_GetI2Lmapping, QP_AddEq;

const char *const FetiGluingTypes[] = {
  "nonred",
  "full",
  "orth",
  "FetiGluingType",
  "FETI_GLUING_",
  0
};

#undef __FUNCT__
#define __FUNCT__ "QPFetiCtxCreate"
PetscErrorCode QPFetiCtxCreate(QPFetiCtx *ctxout)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&ctx));
  ctx->dbc = NULL;
  ctx->i2g = NULL;
  ctx->l2g = NULL;
  ctx->i2g_map = NULL;
  ctx->l2g_map = NULL;
  ctx->setupcalled = PETSC_FALSE;
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiCtxDestroy"
PetscErrorCode QPFetiCtxDestroy(QPFetiCtx ctx)
{
  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&ctx->i2g));
  CHKERRQ(ISDestroy(&ctx->l2g));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&ctx->i2g_map));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&ctx->l2g_map));
  CHKERRQ(QPFetiDirichletDestroy(&ctx->dbc));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiGetCtx"
PetscErrorCode QPFetiGetCtx(QP qp,QPFetiCtx *ctxout)
{
  PetscContainer ctr;
  QPFetiCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)qp,__FUNCT__,(PetscObject*)&ctr));
  if (!ctr) {
    CHKERRQ(QPFetiCtxCreate(&ctx));
    CHKERRQ(PetscContainerCreate(PetscObjectComm((PetscObject)qp),&ctr));
    CHKERRQ(PetscContainerSetPointer(ctr,ctx));
    CHKERRQ(PetscContainerSetUserDestroy(ctr,(PetscErrorCode(*)(void*))QPFetiCtxDestroy));
    CHKERRQ(PetscObjectCompose((PetscObject)qp,__FUNCT__,(PetscObject)ctr));
    CHKERRQ(PetscObjectDereference((PetscObject)ctr));
  }
  CHKERRQ(PetscContainerGetPointer(ctr,(void**)&ctx));
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiSetLocalToGlobalMapping"
PetscErrorCode QPFetiSetLocalToGlobalMapping(QP qp, IS l2g)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(l2g,IS_CLASSID,2);
  CHKERRQ(QPFetiGetCtx(qp,&ctx));
  ctx->l2g = l2g;
  CHKERRQ(PetscObjectReference((PetscObject)l2g));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(l2g,&ctx->l2g_map));
  ctx->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiSetInterfaceToGlobalMapping"
PetscErrorCode QPFetiSetInterfaceToGlobalMapping(QP qp, IS i2g)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(i2g,IS_CLASSID,2);
  CHKERRQ(QPFetiGetCtx(qp,&ctx));
  ctx->i2g = i2g;
  CHKERRQ(PetscObjectReference((PetscObject)i2g));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(i2g,&ctx->i2g_map));
  ctx->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiSetDirichlet"
PetscErrorCode QPFetiSetDirichlet(QP qp, IS dbcis, QPFetiNumberingType numtype, PetscBool enforce_by_B)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(dbcis,IS_CLASSID,2);
  PetscValidLogicalCollectiveEnum(qp,numtype,3);
  PetscValidLogicalCollectiveBool(qp,enforce_by_B,4);
  CHKERRQ(QPFetiGetCtx(qp,&ctx));
  CHKERRQ(QPFetiDirichletDestroy(&ctx->dbc));
  CHKERRQ(QPFetiDirichletCreate(dbcis,numtype,enforce_by_B,&ctx->dbc));
  ctx->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiAssembleDirichlet_ModifyR_Private"
static PetscErrorCode QPFetiAssembleDirichlet_ModifyR_Private(QP qp, IS dbcis)
{
  Mat R, R_loc;
  PetscInt m, n_dbc_local;
  PetscMPIInt rank;

  PetscFunctionBegin;
  CHKERRQ(QPGetOperatorNullSpace(qp, &R));
  if (!R) PetscFunctionReturn(0);

  CHKERRQ(MatGetDiagonalBlock(R, &R_loc));
  CHKERRQ(ISGetLocalSize(dbcis, &n_dbc_local));

  CHKERRQ(MPI_Comm_rank(PetscObjectComm((PetscObject)qp),&rank));
  CHKERRQ(PetscInfo(qp,"n_dbc_local=%d\n",n_dbc_local));
  if (n_dbc_local) {
    CHKERRQ(MatGetSize(R_loc, &m, NULL));
    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF, m, 0, 0, NULL, &R_loc));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)R_loc));
  }
  CHKERRQ(MatCreateBlockDiag(PetscObjectComm((PetscObject)R), R_loc, &R));
  CHKERRQ(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(QPSetOperatorNullSpace(qp, R));
  CHKERRQ(MatDestroy(&R_loc));
  CHKERRQ(MatDestroy(&R));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiAssembleDirichlet"
PetscErrorCode QPFetiAssembleDirichlet(QP qp)
{
  static PetscBool registered = PETSC_FALSE;
  PetscBool flg;
  QPFetiCtx ctx;
  ISLocalToGlobalMapping l2dg=NULL;
  IS dbc_l=NULL, dbc_dg=NULL;
  IS dbcis;
  QPFetiNumberingType numtype;
  PetscBool enforce_by_B;

  PetscFunctionBeginI;
  if (!registered) {
    CHKERRQ(PetscLogEventRegister("QPFetiAssembleDir",QP_CLASSID,&QP_Feti_AssembleDirichlet));
    CHKERRQ(PetscLogEventRegister("QPAddEq",QP_CLASSID,&QP_AddEq));
    registered = PETSC_TRUE;
  }
  CHKERRQ(PetscLogEventBegin(QP_Feti_AssembleDirichlet,qp,0,0,0));
  CHKERRQ(QPFetiGetCtx(qp,&ctx));

  if (!ctx->dbc) PetscFunctionReturnI(0);
  dbcis = ctx->dbc->is;
  numtype = ctx->dbc->numtype;
  enforce_by_B = ctx->dbc->enforce_by_B;

  if (numtype==FETI_GLOBAL_UNDECOMPOSED || numtype==FETI_LOCAL) {
    if (numtype==FETI_GLOBAL_UNDECOMPOSED) {
      const PetscInt *dbcarr;
      PetscInt *dbc_l_arr;
      PetscInt ndbc,nout;

      /* convert global undecomposed indices to local */
      CHKERRQ(ISGetLocalSize(dbcis,&ndbc));
      CHKERRQ(ISGetIndices(dbcis,&dbcarr));
      CHKERRQ(ISGlobalToLocalMappingApply(ctx->l2g_map,IS_GTOLM_DROP,ndbc,dbcarr,&nout,NULL));
      PERMON_ASSERT(ndbc==nout,"n==nout");
      CHKERRQ(PetscMalloc1(ndbc,&dbc_l_arr));
      CHKERRQ(ISGlobalToLocalMappingApply(ctx->l2g_map,IS_GTOLM_DROP,ndbc,dbcarr,NULL,dbc_l_arr));
      CHKERRQ(ISRestoreIndices(dbcis,&dbcarr));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)qp),ndbc,dbc_l_arr,PETSC_OWN_POINTER,&dbc_l));
    } else {
      dbc_l = dbcis;
      CHKERRQ(PetscObjectReference((PetscObject)dbcis));
    }

    /* convert local indices to global decomposed */
    CHKERRQ(MatGetLocalToGlobalMapping(qp->A,&l2dg,NULL));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(l2dg,dbc_l,&dbc_dg));
  } else {
    dbc_dg = dbcis;
    CHKERRQ(PetscObjectReference((PetscObject)dbcis));
  }

  if (enforce_by_B) {
    Mat B,Bt;
    Vec c;
    PetscInt  i, m, ndbc, start, stop;
    flg=PETSC_TRUE;

    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-EXTENSION_ON",&flg,NULL));

    if (flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MATEXTENSION type used for Dirichlet condition\n"));
      Mat At;
      IS cis;
      PetscLayout layout;
      PetscInt  *cis_arr;

      CHKERRQ(ISGetLocalSize(dbc_dg, &ndbc));
      CHKERRQ(MatGetLocalSize(qp->A, &m, NULL));

      CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF, ndbc, ndbc, 1, NULL, &At));
      CHKERRQ(MatSetFromOptions(At));

      for (i=0; i<ndbc; i++)  CHKERRQ(MatSetValue(At, i, i, 1, INSERT_VALUES));
      CHKERRQ(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(  At, MAT_FINAL_ASSEMBLY));
      CHKERRQ(PetscObjectSetName((PetscObject)At,"Bdt_cond"));

      CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject)qp), &layout));
      CHKERRQ(PetscLayoutSetBlockSize(layout, 1));
      CHKERRQ(PetscLayoutSetLocalSize(layout, ndbc));
      CHKERRQ(PetscLayoutSetUp(layout));
      CHKERRQ(PetscLayoutGetRange(layout, &start, &stop));
      CHKERRQ(PetscLayoutDestroy(&layout));

      CHKERRQ(PetscMalloc1(ndbc, &cis_arr));
      for (i=start; i<stop; i++)  cis_arr[i-start]=i;
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)qp), ndbc, cis_arr, PETSC_OWN_POINTER, &cis));

      CHKERRQ(MatCreateExtension(PetscObjectComm((PetscObject)qp), m, ndbc, PETSC_DECIDE, PETSC_DECIDE, At, dbc_dg, PETSC_TRUE, cis, &Bt));

      CHKERRQ(MatDestroy(&At));
      CHKERRQ(ISDestroy(&cis));
    } else {
      const PetscInt  *dbc_dg_arr;
      CHKERRQ(ISGetLocalSize(dbc_dg, &ndbc));
      CHKERRQ(MatGetLocalSize(qp->A, &m, NULL));

      CHKERRQ(MatCreate(PetscObjectComm((PetscObject)qp), &Bt));
      CHKERRQ(MatSetSizes(Bt, m, ndbc, PETSC_DETERMINE, PETSC_DETERMINE));
      CHKERRQ(MatSetType(Bt, MATMPIAIJ));
      CHKERRQ(MatMPIAIJSetPreallocation(Bt, 1, NULL, 0, NULL));

      CHKERRQ(MatGetOwnershipRangeColumn(Bt, &start, &stop));
      CHKERRQ(ISGetIndices(dbc_dg, &dbc_dg_arr));

      for (i = start; i<stop; i++) {
        CHKERRQ(MatSetValue(Bt,dbc_dg_arr[i-start],i,1.0,INSERT_VALUES));
      }

      CHKERRQ(ISRestoreIndices(dbc_dg, &dbc_dg_arr));
      CHKERRQ(MatAssemblyBegin(Bt, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(Bt,MAT_FINAL_ASSEMBLY));
    }

    CHKERRQ(PetscObjectSetName((PetscObject)Bt,"Bdt"));
    CHKERRQ(PermonMatTranspose(Bt,MAT_TRANSPOSE_IMPLICIT,&B));
    CHKERRQ(PetscObjectSetName((PetscObject)B,"Bd"));
    CHKERRQ(MatDestroy(&Bt));

    CHKERRQ(PetscLogEventBegin(QP_AddEq,qp,0,0,0));
    CHKERRQ(MatCreateVecs(B,NULL,&c));
    CHKERRQ(MatMult(B,qp->x,c));
    CHKERRQ(QPAddEq(qp, B, c));
    CHKERRQ(PetscLogEventEnd(QP_AddEq,qp,0,0,0));

    CHKERRQ(VecDestroy(&c));
    CHKERRQ(MatDestroy(&B));

    /* parent is matis -> add dirichlet IS */
    if (qp->parent) {
      CHKERRQ(PetscStrcmp(qp->transform_name,"QPTMatISToBlockDiag",&flg));
      if (flg) {
        if (!dbc_l) {
          CHKERRQ(MatGetLocalToGlobalMapping(qp->A,&l2dg,NULL));
          CHKERRQ(ISGlobalToLocalMappingApplyIS(l2dg,IS_GTOLM_DROP,dbc_dg,&dbc_l));
        }
        ((QPTMatISToBlockDiag_Ctx*)qp->postSolveCtx)->isDir = dbc_dg;
        CHKERRQ(PetscObjectReference((PetscObject)dbc_dg));
      }
    }
  } else {
    Vec d;
    PetscScalar alpha;

    /* alpha=max(abs(diag(A)) */
    CHKERRQ(MatCreateVecs(qp->A,NULL,&d));
    CHKERRQ(MatGetDiagonal(qp->A,d));
    CHKERRQ(VecAbs(d));
    CHKERRQ(VecMax(d,NULL,&alpha));
    CHKERRQ(VecDestroy(&d));

    CHKERRQ(MatZeroRowsColumnsIS(qp->A, dbc_dg, alpha, qp->x, qp->b));
    CHKERRQ(QPFetiAssembleDirichlet_ModifyR_Private(qp, dbc_dg));
  }

  CHKERRQ(ISDestroy(&dbc_dg));
  CHKERRQ(ISDestroy(&dbc_l));
  CHKERRQ(PetscLogEventEnd(QP_Feti_AssembleDirichlet,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiSetUp"
PetscErrorCode QPFetiSetUp(QP qp)
{
  MPI_Comm comm;
  static PetscBool registered = PETSC_FALSE;
  QPFetiCtx ctx;
  Mat Bg;
  PetscInt nlocaldofs;
  FetiGluingType type = FETI_GLUING_FULL;
  PetscBool exclude_dir = PETSC_FALSE;

  FllopTracedFunctionBegin;
  CHKERRQ(QPFetiGetCtx(qp,&ctx));
  if (ctx->setupcalled) PetscFunctionReturn(0);

  if (!registered) {
    CHKERRQ(PetscLogEventRegister("QPFetiSetUp",QP_CLASSID,&QP_Feti_SetUp));
    registered = PETSC_TRUE;
  }

  FllopTraceBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  CHKERRQ(PetscLogEventBegin(QP_Feti_SetUp,qp,0,0,0));

  PERMON_ASSERT(qp->A,"Operator must be specified");
  CHKERRQ(MatGetLocalSize(qp->A, &nlocaldofs, NULL));

  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-feti_gluing_type",FetiGluingTypes,(PetscEnum*)&type, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-feti_gluing_exclude_dirichlet",&exclude_dir,NULL));
  CHKERRQ(PetscPrintf(comm, "============\n FETI gluing type: %s\n excluding Dirichlet DOFs? %d\n",FetiGluingTypes[type],exclude_dir));
  CHKERRQ(QPFetiAssembleDirichlet(qp));
  
  if (!ctx->l2g) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_WRONGSTATE,"L2G mapping must be set first - call QPFetiSetLocalToGlobalMapping before QPFetiSetUp");
  if (!ctx->i2g) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_WRONGSTATE,"I2G mapping must be set first - call QPFetiSetInterfaceToGlobalMapping before QPFetiSetUp");
  CHKERRQ(QPFetiAssembleGluing(qp, type, exclude_dir, &Bg));
  CHKERRQ(PetscPrintf(comm, "============\n"));

  CHKERRQ(PetscObjectSetName((PetscObject)Bg,"Bg"));
  CHKERRQ(QPAddEq(qp,Bg,NULL));
  CHKERRQ(MatDestroy(&Bg));

  if (!qp->BE) printf("child (BE) is needed for dualization\n");
  ctx->setupcalled = PETSC_TRUE;
  CHKERRQ(PetscLogEventEnd  (QP_Feti_SetUp,qp,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiGetI2Lmapping"
PetscErrorCode QPFetiGetI2Lmapping(MPI_Comm comm, IS l2g,  IS i2g,  IS *i2l_new)
{
  static PetscBool registered = PETSC_FALSE;
  const PetscInt  *i2g_arr, *l2g_arr;
  PetscInt ni, nl, i, j, *i2l_arr, *idx, *l2g_arr_copy;

  PetscFunctionBeginI;

  if (!registered) {
    CHKERRQ(PetscLogEventRegister("QPFetiGetI2Lmapping",QP_CLASSID,&QP_Feti_GetI2Lmapping));
    registered = PETSC_TRUE;
  }
  CHKERRQ(PetscLogEventBegin(QP_Feti_GetI2Lmapping,0,0,0,0));

  CHKERRQ(ISGetIndices( i2g, &i2g_arr));
  CHKERRQ(ISGetIndices( l2g, &l2g_arr));
  CHKERRQ(ISGetLocalSize( i2g, &ni));
  CHKERRQ(ISGetLocalSize( l2g, &nl));

  CHKERRQ(PetscMalloc3(ni, &i2l_arr, nl, &idx, nl, &l2g_arr_copy));

  for (i=0; i<nl; i++) {
    idx[i]=i;
    l2g_arr_copy[i]=l2g_arr[i];
  }
  CHKERRQ(PetscSortIntWithArray(nl, l2g_arr_copy, idx));

  for (i=0; i< ni; i++) {
    CHKERRQ(PetscFindInt(i2g_arr[i], nl, l2g_arr_copy, &j));
    i2l_arr[i]=idx[j];
  }

  CHKERRQ(ISCreateGeneral(comm, ni, i2l_arr, PETSC_COPY_VALUES, i2l_new));

  CHKERRQ(ISRestoreIndices( i2g, &i2g_arr));
  CHKERRQ(ISRestoreIndices( l2g, &l2g_arr));
  CHKERRQ(PetscFree3(i2l_arr, idx, l2g_arr_copy));

  CHKERRQ(PetscLogEventEnd(QP_Feti_GetI2Lmapping,0,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiAssembleGluing"
PetscErrorCode QPFetiAssembleGluing(QP qp, FetiGluingType type, PetscBool exclude_dir, Mat *Bg_new)
{
  static PetscBool registered = PETSC_FALSE;
  QPFetiCtx   ctx;
  Mat         Bgt;
  PetscInt    nl, Nu, Nug;
  IS          i2l, i2g_less, all_dir;
  PetscMPIInt commsize;
  IS          l2g, i2g, global_dir;
  MPI_Comm    comm;

  PetscFunctionBeginI;
  if (!registered) {
    CHKERRQ(PetscLogEventRegister("QPFetiAssemGluing",QP_CLASSID,&QP_Feti_AssemGluing));
    registered = PETSC_TRUE;
  }
  CHKERRQ(PetscLogEventBegin(QP_Feti_AssemGluing,0,0,0,0));

  CHKERRQ(PetscObjectGetComm((PetscObject)qp,&comm));
  CHKERRQ(QPFetiGetCtx(qp,&ctx));
  l2g = ctx->l2g;
  i2g = ctx->i2g;
  CHKERRQ(MPI_Comm_size(comm, &commsize));  
  CHKERRQ(ISGetLocalSize(l2g, &nl)); 
  
  if (exclude_dir) {
    CHKERRQ(QPFetiGetGlobalDir( qp, ctx->dbc->is, ctx->dbc->numtype, &global_dir));
    CHKERRQ(ISAllGather(global_dir, &all_dir));
    CHKERRQ(ISSortRemoveDups(all_dir));
    CHKERRQ(ISDifference(i2g,all_dir,&i2g_less));
    CHKERRQ(ISDestroy(&global_dir));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)i2g));
    i2g_less=i2g;
  }     
  
  /* get i2l from l2g and i2g */
  CHKERRQ(QPFetiGetI2Lmapping(comm, ctx->l2g, i2g_less, &i2l));

  /* create Adt using SF */
  CHKERRQ(ISGetMinMax(i2g_less,NULL,&Nu));
  CHKERRQ(MPI_Allreduce(&Nu, &Nug, 1, MPIU_INT, MPIU_MAX, comm));
  Nu = Nug+1;
   
  CHKERRQ(QPFetiGetBgtSF(comm, i2g_less, Nu, i2l, nl, type, &Bgt));

  CHKERRQ(PetscObjectSetName((PetscObject)Bgt,"Bgt"));

  /* create Bg from Bgt */
  CHKERRQ(MatCreateTranspose(Bgt,Bg_new));
  CHKERRQ(PetscObjectSetName((PetscObject)*Bg_new,"Bg"));

  CHKERRQ(ISDestroy(&i2l));
  CHKERRQ(ISDestroy(&i2g_less));
  CHKERRQ(MatDestroy(&Bgt));

  CHKERRQ(PetscLogEventEnd(QP_Feti_AssemGluing,0,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiGetBgtSF"
PetscErrorCode QPFetiGetBgtSF(MPI_Comm comm, IS i2g, PetscInt Nu, IS i2l, PetscInt nl, FetiGluingType type, Mat *Bgt_out)
{
  static PetscBool registered = PETSC_FALSE;
  Mat         At, Bgt;
  PetscMPIInt commsize, rank;
  PetscInt    i, j, k, idx, idx2, data_scat_size, rstart, rend;
  PetscInt    nleaves_SF1, nroots_SF1, n_link=0, nleaves_SF2=0, nleaves_SF2_unique=0;
  const  PetscInt *i2g_array, *i2l_array, *root_degree_onroots_SF1, *root_degree_onroots_SF2;
  PetscInt    *root_degree_onleaves_SF1, *future_leavesSF2_onleaves, *future_leavesSF2_onroots;
  PetscInt    *leaves_SF2, *local_idx, *condensed_local_idx, *root_degree_onleaves_fromSF1onSF2, *ris_array;
  PetscInt    *link_onroot, *link_onleaves, *division, *max_rank_onroot_linkSF, *max_rank_onleaves_linkSF;
  PetscInt    *prealloc_seqEX; // for Extension type
  PetscInt    *prealloc_diag, *prealloc_ofdiag, *prealloc_seq; // only for notExtension type
  PetscReal   *values;
  PetscLayout layout_SF1, links;
  PetscSF SF1, SF2, link_SF;
  IS myneighbors;

  PetscFunctionBeginI;
  if (!registered) {
    CHKERRQ(PetscLogEventRegister("QPFetiGetBgtSF",QP_CLASSID,&QP_Feti_GetBgtSF));
    registered = PETSC_TRUE;
  }
  CHKERRQ(PetscLogEventBegin(QP_Feti_GetBgtSF,0,0,0,0)); 

  CHKERRQ(MPI_Comm_size(comm, &commsize));
  CHKERRQ(MPI_Comm_rank(comm, &rank));

  // first SF (all nodes from i2g) /STEP 1/
  CHKERRQ(ISGetLocalSize(i2g, &nleaves_SF1));
  CHKERRQ(ISGetIndices(i2g, &i2g_array));
  CHKERRQ(ISGetIndices(i2l, &i2l_array));

  CHKERRQ(PetscLayoutCreate(comm, &layout_SF1));
  CHKERRQ(PetscLayoutSetBlockSize(layout_SF1, 1));
  CHKERRQ(PetscLayoutSetSize(layout_SF1, Nu));
  CHKERRQ(PetscLayoutSetUp(layout_SF1));
  CHKERRQ(PetscSFCreate(comm, &SF1));
  CHKERRQ(PetscLayoutGetLocalSize(layout_SF1, &nroots_SF1));
  CHKERRQ(PetscSFSetGraphLayout(SF1, layout_SF1, nleaves_SF1, NULL, PETSC_COPY_VALUES, i2g_array));
  CHKERRQ(PetscSFSetRankOrder(SF1, PETSC_TRUE));

  // find out root's degree and send to leaves /STEP 2/
  CHKERRQ(PetscSFComputeDegreeBegin(SF1, &root_degree_onroots_SF1));
  // work between communication   
  CHKERRQ(PetscMalloc2( nleaves_SF1, &root_degree_onleaves_SF1, nleaves_SF1, &future_leavesSF2_onleaves));
  for (i=0; i<nleaves_SF1; i++) {
    root_degree_onleaves_SF1[i] = -1;
    future_leavesSF2_onleaves[i] = -1;
  }
  CHKERRQ(PetscSFComputeDegreeEnd(SF1, &root_degree_onroots_SF1));

  CHKERRQ(PetscSFBcastBegin(SF1, MPIU_INT, root_degree_onroots_SF1, root_degree_onleaves_SF1, MPI_REPLACE));
  // work between communication  
  // scatter - how much i will sent? 
  for (i=0, data_scat_size=0; i<nroots_SF1; i++) {
    data_scat_size += root_degree_onroots_SF1[i];
  }
  CHKERRQ(PetscMalloc1( data_scat_size, &future_leavesSF2_onroots));

  // what I will scatter? different multiplicity of leaf for full/non-redundant or orthonormal   /STEP 2 continues/
  idx=0;
  for (i=0; i<nroots_SF1; i++) {
    if (root_degree_onroots_SF1[i] == 1) { //leaves with degree==1 will be deleted (degree==0 dont exist)
      future_leavesSF2_onroots[idx] = 0;
      idx++;
    } else if (root_degree_onroots_SF1[i]==2) { //leaves with degree==2, are not multiplied for all type 
      future_leavesSF2_onroots[idx] = 1;
      future_leavesSF2_onroots[idx+1] = 1;
      idx=idx+2;
    } else { //leaves with degree>2, are multiplied according type  /if (root_degree_onroots[i]>2) 
      for (j=0; j<root_degree_onroots_SF1[i]; j++) {
        switch ( type ) {
          case FETI_GLUING_NONRED:
            if (j<1) {
              future_leavesSF2_onroots[idx]= root_degree_onroots_SF1[i] - 1;
              idx++;
            } else {
              future_leavesSF2_onroots[idx]= 1;
              idx++;
            }
            break;
          case FETI_GLUING_FULL:
            future_leavesSF2_onroots[idx]= root_degree_onroots_SF1[i] - 1;
            idx++;
            break;
          case FETI_GLUING_ORTH:
            if (j<2) {
              future_leavesSF2_onroots[idx]= root_degree_onroots_SF1[i] - 1;
              idx++;
            } else {
              future_leavesSF2_onroots[idx]= root_degree_onroots_SF1[i] -1 - (j-1);
              idx++;
            }
            break;
          default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown FETI gluing type");
        }
      }
    }
  }
  CHKERRQ(PetscSFBcastEnd(SF1, MPIU_INT, root_degree_onroots_SF1, root_degree_onleaves_SF1, MPI_REPLACE));
  // scatter itself /STEP 3/
  CHKERRQ(PetscSFScatterBegin(SF1, MPIU_INT, future_leavesSF2_onroots, future_leavesSF2_onleaves));
  // work between communication
  // find out actual number of links on processor /STEP 4/
  for (i=0; i<nroots_SF1; i++) {
    if (root_degree_onroots_SF1[i]>1) {
      switch ( type ) {
        case FETI_GLUING_NONRED:
          n_link = n_link + root_degree_onroots_SF1[i]-1;
          break;
        case FETI_GLUING_FULL:
          n_link = n_link + root_degree_onroots_SF1[i]*(root_degree_onroots_SF1[i]-1)/2;
          break;
        case FETI_GLUING_ORTH:
          n_link = n_link + root_degree_onroots_SF1[i]-1;
          break;
        default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown FETI gluing type");
      }
    }
  }
  // link's laylout
  CHKERRQ(PetscLayoutCreate(comm, &links));
  CHKERRQ(PetscLayoutSetBlockSize(links, 1));
  CHKERRQ(PetscLayoutSetLocalSize(links, n_link));
  CHKERRQ(PetscLayoutSetUp(links));
  CHKERRQ(PetscLayoutGetRange(links, &rstart, NULL));
  CHKERRQ(PetscSFScatterEnd(SF1, MPIU_INT, future_leavesSF2_onroots, future_leavesSF2_onleaves));

  // "make" leaves for SF2, create local indexes  /STEP 5/
  for (i=0; i<nleaves_SF1; i++) {
    nleaves_SF2 = nleaves_SF2 + future_leavesSF2_onleaves[i];
    if (future_leavesSF2_onleaves[i]>0) nleaves_SF2_unique++;
  }

  CHKERRQ(PetscMalloc4(nleaves_SF2, &leaves_SF2, nleaves_SF2, &local_idx, nleaves_SF2, &condensed_local_idx, nleaves_SF2, &root_degree_onleaves_fromSF1onSF2));
  CHKERRQ(PetscMalloc1(nleaves_SF2_unique, &ris_array));
  CHKERRQ(PetscMalloc1(nleaves_SF2_unique, &prealloc_seqEX));

  for (i=0, idx=0, idx2=0; i<nleaves_SF1; i++) {
    for (j=0; j<future_leavesSF2_onleaves[i]; j++) {
      if (j==0) {
        ris_array[idx2]=i2l_array[i];
      }
      leaves_SF2[idx]=i2g_array[i];
      local_idx[idx]=i;
      condensed_local_idx[idx]=idx2;
      root_degree_onleaves_fromSF1onSF2[idx]=root_degree_onleaves_SF1[i];
      idx++;
    }
    if (future_leavesSF2_onleaves[i]!=0) {
      prealloc_seqEX[idx2]=future_leavesSF2_onleaves[i];
      idx2++;
    }
  }

  // second SF with same layout
  CHKERRQ(PetscSFCreate(comm, &SF2));
  CHKERRQ(PetscSFSetGraphLayout(SF2, layout_SF1, nleaves_SF2, NULL, PETSC_COPY_VALUES, leaves_SF2));
  CHKERRQ(PetscSFSetRankOrder(SF2, PETSC_TRUE));

  // scatter links /STEP 6/
  // how much i will sent? 
  CHKERRQ(PetscSFComputeDegreeBegin(SF2, &root_degree_onroots_SF2));
  CHKERRQ(PetscSFComputeDegreeEnd(SF2, &root_degree_onroots_SF2));

  for (i=0, data_scat_size=0; i<nroots_SF1; i++) {
    data_scat_size += root_degree_onroots_SF2[i];
  }
  CHKERRQ(PetscMalloc1(data_scat_size, &link_onroot));
  CHKERRQ(PetscMalloc1(nleaves_SF2, &link_onleaves));
  for (i=0; i<nleaves_SF2; i++) link_onleaves[i] = -1;

  // what I will sent?  
  idx2=rstart;
  idx=0;
   
  for (j=0; j<nroots_SF1; j++) { 

    int end_it;
    switch ( type ) {
      case FETI_GLUING_NONRED:
        for (i=0; i<root_degree_onroots_SF1[j]-1; i++) {
          link_onroot[i+idx]=idx2;
          link_onroot[i+idx +root_degree_onroots_SF1[j]-1 ]=idx2;
          idx2++;
        }
        break;
      case FETI_GLUING_FULL:
        for (i=0; i<root_degree_onroots_SF1[j]-1; i++) {
          for (k=0; k < root_degree_onroots_SF1[j]-1 -i; k++) {
            link_onroot[i*root_degree_onroots_SF1[j] +k +idx]=idx2;
            link_onroot[(i+1)*root_degree_onroots_SF1[j] - 1 + k*(root_degree_onroots_SF1[j]-1)+idx]=idx2;
            idx2++;
          }
        }
        break;
      case FETI_GLUING_ORTH:
        for (i=0; i<=root_degree_onroots_SF1[j]-1; i++) { 
          end_it= root_degree_onroots_SF1[j]-1;
          if (i>1) end_it=end_it-i+1;
          for (k=0; k < end_it; k++) {
            link_onroot[idx]=idx2+k; 
            idx++;
          }
        }
        break;
      default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown FETI gluing type");
    }
    if (root_degree_onroots_SF1[j]>1) {
      switch ( type ) {
        case FETI_GLUING_NONRED:
          idx=idx+ 2*(root_degree_onroots_SF1[j]-1);
          break;
        case FETI_GLUING_FULL:
          idx=idx+ root_degree_onroots_SF1[j]*(root_degree_onroots_SF1[j]-1);
          break;
        case FETI_GLUING_ORTH:
          idx2=idx2+root_degree_onroots_SF1[j]-1;
          break;
        default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown FETI gluing type");
      }
    }
  } 

  CHKERRQ(PetscSFScatterBegin(SF2, MPIU_INT, link_onroot, link_onleaves));
  // work between communication  
  // new layout for links 
  CHKERRQ(PetscLayoutGetSize(links, &n_link));
  CHKERRQ(PetscLayoutDestroy(&links));
  CHKERRQ(PetscLayoutCreate(comm, &links));
  CHKERRQ(PetscLayoutSetBlockSize(links, 1));
  CHKERRQ(PetscLayoutSetSize(links, n_link));
  CHKERRQ(PetscLayoutSetUp(links));
  // compute divison - only orthonormal /STEP 6.5/  
  CHKERRQ(PetscMalloc1(nleaves_SF2, &division));
  if (type == FETI_GLUING_ORTH) {
    j=0;
    for (i=0; i<nleaves_SF2; i++) {
      if (j==0) { 
        division[i] = root_degree_onleaves_fromSF1onSF2[i]-1;
        if (root_degree_onleaves_fromSF1onSF2[i]>2) {
          j=root_degree_onleaves_fromSF1onSF2[i]-2;
        }
      } else if (j>=1) {

        if (leaves_SF2[i]==leaves_SF2[i-1]) {
          division[i] = j;
          j--;
        } else {
          division[i] = root_degree_onleaves_fromSF1onSF2[i]-1;
          if (root_degree_onleaves_fromSF1onSF2[i]>2) {
            j=root_degree_onleaves_fromSF1onSF2[i]-2;
          }
        }
      }
    }
  }
  CHKERRQ(PetscSFScatterEnd(SF2, MPIU_INT, link_onroot, link_onleaves));
  CHKERRQ(PetscSortInt( nleaves_SF2, link_onleaves));
  // link SF /STEP 7/
  CHKERRQ(PetscSFCreate(comm, &link_SF));
  CHKERRQ(PetscSFSetGraphLayout(link_SF, links, nleaves_SF2, NULL, PETSC_COPY_VALUES, link_onleaves));
  CHKERRQ(PetscSFSetRankOrder(link_SF, PETSC_TRUE));

  //get max ranks of leaves (find out if + or -) /STEP 8/
  CHKERRQ(PetscMalloc2(n_link, &max_rank_onroot_linkSF, nleaves_SF2, &max_rank_onleaves_linkSF));
  for (i=0; i<n_link; i++) max_rank_onroot_linkSF[i] = -1;
  for (i=0; i<nleaves_SF2; i++) max_rank_onleaves_linkSF[i] = rank;

  CHKERRQ(PetscSFReduceBegin(link_SF, MPIU_INT, max_rank_onleaves_linkSF, max_rank_onroot_linkSF, MPIU_MAX));
  // work between communication  
  CHKERRQ(PetscLayoutGetRange(links, &rstart, &rend));
  CHKERRQ(PetscLayoutGetLocalSize(links, &n_link));
  CHKERRQ(PetscMalloc3(nleaves_SF1, &prealloc_diag, nleaves_SF1, &prealloc_ofdiag, nleaves_SF1, &prealloc_seq));
  for (i=0; i<nleaves_SF1; i++) {
    prealloc_diag[i] = 0;
    prealloc_ofdiag[i] = 0;
    prealloc_seq[i] = 0;
  }
  CHKERRQ(PetscSFReduceEnd(link_SF, MPIU_INT, max_rank_onleaves_linkSF, max_rank_onroot_linkSF, MPIU_MAX));

  CHKERRQ(PetscSFBcastBegin(link_SF, MPIU_INT, max_rank_onroot_linkSF, max_rank_onleaves_linkSF, MPI_REPLACE));
  // work between communication  
  // get preallocation pattern 
  for (i=0; i<nleaves_SF2; i++) {
    if (link_onleaves[i]>=rstart && link_onleaves[i]<rend) {
      prealloc_diag[local_idx[i]]++;
      prealloc_seq[local_idx[i]]++;
    } else {
      prealloc_ofdiag[local_idx[i]]++;
      prealloc_seq[local_idx[i]]++;
    }
  }
  CHKERRQ(PetscMalloc1(nleaves_SF2, &values));

  CHKERRQ(PetscSFBcastEnd(link_SF, MPIU_INT, max_rank_onroot_linkSF, max_rank_onleaves_linkSF, MPI_REPLACE));

  PetscBool flg_SCALE_ON=PETSC_TRUE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-SCALE_ON",&flg_SCALE_ON,NULL));
  if (flg_SCALE_ON && (type==FETI_GLUING_NONRED || type==FETI_GLUING_FULL)) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," SCALING\n"));

/* compose array of neighbors to Bg */
  {
    PetscInt n;
    PetscInt *rank_onroot_linkSF, *rank_onleaves_linkSF;

    CHKERRQ(PetscMalloc1(n_link, &rank_onroot_linkSF));
    for (i=0; i<n_link; i++) rank_onroot_linkSF[i] = PETSC_MAX_INT;
    CHKERRQ(PetscMalloc1(2*nleaves_SF2, &rank_onleaves_linkSF));
    for (i=0; i<nleaves_SF2; i++) rank_onleaves_linkSF[i] = rank;
    for (i=nleaves_SF2; i<2*nleaves_SF2; i++) rank_onleaves_linkSF[i] = commsize;

    CHKERRQ(PetscSFReduceBegin(link_SF, MPIU_INT, rank_onleaves_linkSF, rank_onroot_linkSF, MPIU_MIN));
    CHKERRQ(PetscSFReduceEnd(link_SF, MPIU_INT, rank_onleaves_linkSF, rank_onroot_linkSF, MPIU_MIN));
    CHKERRQ(PetscSFBcastBegin(link_SF, MPIU_INT, rank_onroot_linkSF, rank_onleaves_linkSF, MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(link_SF, MPIU_INT, rank_onroot_linkSF, rank_onleaves_linkSF, MPI_REPLACE));
    CHKERRQ(PetscMemcpy(&rank_onleaves_linkSF[nleaves_SF2],max_rank_onleaves_linkSF,nleaves_SF2*sizeof(PetscInt)));
    n = 2*nleaves_SF2;
    CHKERRQ(PetscSortRemoveDupsInt(&n,rank_onleaves_linkSF));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,rank_onleaves_linkSF,PETSC_COPY_VALUES,&myneighbors));

    CHKERRQ(PetscFree(rank_onleaves_linkSF));
    CHKERRQ(PetscFree(rank_onroot_linkSF));
  }

  //get values
  PetscReal x;
  switch ( type ) {
    case FETI_GLUING_NONRED:
      for (i=0; i<nleaves_SF2; i++) {
        if (max_rank_onleaves_linkSF[i]==rank) {
          values[i] = -1.0;
        } else {
          values[i] = 1.0;
        }
        if (flg_SCALE_ON) values[i] *= (1.0 / PetscSqrtReal((PetscReal)root_degree_onleaves_fromSF1onSF2[i]));
      }
      break;
    case FETI_GLUING_FULL:
      for (i=0; i<nleaves_SF2; i++) {
        if (max_rank_onleaves_linkSF[i]==rank) {
          values[i] = -1.0;
        } else {
          values[i] = 1.0;
        }
        if (flg_SCALE_ON) values[i] *= (1.0 / PetscSqrtReal((PetscReal)root_degree_onleaves_fromSF1onSF2[i]));
      }
      break;
    case FETI_GLUING_ORTH:
      for (i=0; i<nleaves_SF2; i++) {
        if (max_rank_onleaves_linkSF[i]==rank) {
          values[i] = -1.0;
        } else {
          values[i] = 1.0/(PetscReal)division[i];
        }
        x = 1.0/(PetscReal)division[i] + 1.0;
        values[i]/=sqrt(x);
      }
      break;
    default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown FETI gluing type");
  }

  CHKERRQ(ISRestoreIndices(i2l, &i2l_array));
  CHKERRQ(ISRestoreIndices(i2g, &i2g_array));
  PetscBool flg_EXTENSION_ON=PETSC_TRUE;
  PetscBool flg_MATGLUING_ON=PETSC_FALSE;

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-EXTENSION_ON",&flg_EXTENSION_ON,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-MATGLUING_ON",&flg_MATGLUING_ON,NULL));

  if (flg_EXTENSION_ON) {
    IS cis, ris;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MATEXTENSION type used for gluing matrix\n"));

    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF, nleaves_SF2_unique, nleaves_SF2, -1, prealloc_seqEX, &At));
    CHKERRQ(MatSetFromOptions(At));
    for (i=0; i<nleaves_SF2; i++)  CHKERRQ(MatSetValue(At, condensed_local_idx[i], i, values[i], INSERT_VALUES));

    CHKERRQ(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(At, MAT_FINAL_ASSEMBLY));

    CHKERRQ(PetscObjectSetName((PetscObject)At, "Bgt_cond"));

    CHKERRQ(ISCreateGeneral(comm, nleaves_SF2, link_onleaves, PETSC_OWN_POINTER, &cis));
    CHKERRQ(ISCreateGeneral(comm, nleaves_SF2_unique, ris_array, PETSC_OWN_POINTER, &ris));

    CHKERRQ(MatCreateExtension(comm, nl, n_link, PETSC_DECIDE, PETSC_DECIDE, At, ris, PETSC_FALSE, cis, &Bgt));
    CHKERRQ(MatDestroy(&At));
    CHKERRQ(ISDestroy(&cis));
    CHKERRQ(ISDestroy(&ris));

  } else {
    //TODO allocate this array only if needed
    CHKERRQ(PetscFree(ris_array));

    if (!flg_MATGLUING_ON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," just PetscSF (no special type) used for gluing matrix\n"));

      CHKERRQ(MatCreate(comm, &At ));
      CHKERRQ(MatSetSizes(At, nleaves_SF1, n_link,  PETSC_DETERMINE, PETSC_DETERMINE ));
      CHKERRQ(MatSetFromOptions(At));
      CHKERRQ(MatMPIAIJSetPreallocation(At , 0, prealloc_diag, 0, prealloc_ofdiag));
      CHKERRQ(MatSeqAIJSetPreallocation(At, 0, prealloc_seq));

      CHKERRQ(MatGetOwnershipRange(At,  &idx, &idx2));
      for (i=0; i<nleaves_SF2; i++)  CHKERRQ(MatSetValue(At, local_idx[i]+idx, link_onleaves[i], values[i], INSERT_VALUES));

      CHKERRQ(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(At, MAT_FINAL_ASSEMBLY));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MATGLUING type used for gluing matrix\n"));
      CHKERRQ(MatCreateGluing(comm, nleaves_SF1,  nleaves_SF2_unique,  n_link,  local_idx, values, link_SF, &At));
    }

    Mat T_loc, T;
    VecScatter scatter;
    Vec n_vec, N_vec;
    PetscInt ni;

    CHKERRQ(ISGetLocalSize(i2l,&ni));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, ni, &n_vec));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, nl, &N_vec));
    CHKERRQ(VecScatterCreate(n_vec, NULL, N_vec, i2l, &scatter));
    CHKERRQ(MatCreateScatter(PETSC_COMM_SELF, scatter, &T_loc));
    CHKERRQ(MatCreateBlockDiag(comm, T_loc, &T));

    Mat Bt_arr[] ={At, T};
    CHKERRQ(MatCreateProd(comm, 2, Bt_arr, &Bgt));

    /* hotfix for MatMatBlockDiagMultByColumns_Private */
    CHKERRQ(PetscObjectCompose((PetscObject)Bgt, "T_loc", (PetscObject)T_loc));
    CHKERRQ(PetscObjectCompose((PetscObject)Bgt, "Adt", (PetscObject)At));

    CHKERRQ(PetscFree(link_onleaves));
    CHKERRQ(MatDestroy(&At));
    CHKERRQ(MatDestroy(&T_loc));
    CHKERRQ(MatDestroy(&T));
    CHKERRQ(VecScatterDestroy(&scatter));
    CHKERRQ(VecDestroy(&n_vec));
    CHKERRQ(VecDestroy(&N_vec));
  }

  CHKERRQ(PetscObjectCompose((PetscObject)Bgt,"myneighbors",(PetscObject)myneighbors));
  CHKERRQ(ISDestroy(&myneighbors));
  *Bgt_out=Bgt;

  CHKERRQ(PetscFree2( root_degree_onleaves_SF1, future_leavesSF2_onleaves));
  CHKERRQ(PetscFree(future_leavesSF2_onroots));
  CHKERRQ(PetscFree(prealloc_seqEX));
  CHKERRQ(PetscFree(link_onroot));
  CHKERRQ(PetscFree(division));
  CHKERRQ(PetscFree4(leaves_SF2, local_idx, condensed_local_idx, root_degree_onleaves_fromSF1onSF2));
  CHKERRQ(PetscFree3(prealloc_diag, prealloc_ofdiag, prealloc_seq)); 
  CHKERRQ(PetscFree(values));
  CHKERRQ(PetscFree2(max_rank_onroot_linkSF, max_rank_onleaves_linkSF));
  CHKERRQ(PetscSFDestroy(&SF1));
  CHKERRQ(PetscSFDestroy(&SF2));
  CHKERRQ(PetscSFDestroy(&link_SF));
  CHKERRQ(PetscLayoutDestroy(&layout_SF1));
  CHKERRQ(PetscLayoutDestroy(&links));

  CHKERRQ(PetscLogEventEnd(QP_Feti_GetBgtSF,0,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiGetGlobalDir"
PetscErrorCode QPFetiGetGlobalDir(QP qp, IS dbc, QPFetiNumberingType numtype, IS *dbc_g)
{ 
  PetscInt ndbc;
  const PetscInt *dbc_arr;
  const PetscInt *dbc_l_arr;
  PetscInt *global_arr;

  PetscFunctionBeginI;
  if (numtype==FETI_GLOBAL_DECOMPOSED || numtype==FETI_LOCAL) {
    PetscInt *dbc_l_arr_nonconst = NULL;
    QPFetiCtx ctx;

    CHKERRQ(ISGetLocalSize(dbc,&ndbc));
    CHKERRQ(ISGetIndices(dbc,&dbc_arr));

    if (numtype==FETI_GLOBAL_DECOMPOSED) {
      PetscInt nout;
      ISLocalToGlobalMapping l2dg;

      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)qp), " FETI_GLOBAL_DECOMPOSED numbering of Dirichlet DOFs\n"));
      /* convert global decomposed indices to local */
      CHKERRQ(MatGetLocalToGlobalMapping(qp->A,&l2dg,NULL));
      CHKERRQ(ISGlobalToLocalMappingApply(l2dg,IS_GTOLM_DROP,ndbc,dbc_arr,&nout,NULL));
      PERMON_ASSERT(ndbc==nout,"n==nout");
      CHKERRQ(PetscMalloc1(ndbc,&dbc_l_arr_nonconst));
      CHKERRQ(ISGlobalToLocalMappingApply(l2dg,IS_GTOLM_DROP,ndbc,dbc_arr,NULL,dbc_l_arr_nonconst));
      dbc_l_arr = dbc_l_arr_nonconst;
    } else { //FETI_LOCAL
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)qp), " FETI_LOCAL numbering of Dirichlet DOFs\n"));
      dbc_l_arr = dbc_arr;
    }
    /* convert local indices to global undecomposed */
    CHKERRQ(QPFetiGetCtx(qp,&ctx));
    CHKERRQ(PetscMalloc1(ndbc,&global_arr));
    CHKERRQ(ISLocalToGlobalMappingApply(ctx->l2g_map, ndbc, dbc_l_arr, global_arr));

    CHKERRQ(ISRestoreIndices(dbc,&dbc_arr));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)qp),ndbc,global_arr,PETSC_OWN_POINTER, dbc_g));
    CHKERRQ(PetscFree(dbc_l_arr_nonconst));

  } else { //FETI_GLOBAL_UNDECOMPOSED
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)qp), " FETI_GLOBAL_UNDECOMPOSED numbering of Dirichlet DOFs\n"));
    CHKERRQ(PetscObjectReference((PetscObject)dbc));
    *dbc_g =dbc;
  } 
  PetscFunctionReturnI(0);
}
