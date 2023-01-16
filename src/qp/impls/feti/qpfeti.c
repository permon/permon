
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

PetscErrorCode QPFetiCtxCreate(QPFetiCtx *ctxout)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->dbc = NULL;
  ctx->i2g = NULL;
  ctx->l2g = NULL;
  ctx->i2g_map = NULL;
  ctx->l2g_map = NULL;
  ctx->setupcalled = PETSC_FALSE;
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode QPFetiCtxDestroy(QPFetiCtx ctx)
{
  PetscFunctionBegin;
  PetscCall(ISDestroy(&ctx->i2g));
  PetscCall(ISDestroy(&ctx->l2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&ctx->i2g_map));
  PetscCall(ISLocalToGlobalMappingDestroy(&ctx->l2g_map));
  PetscCall(QPFetiDirichletDestroy(&ctx->dbc));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode QPFetiGetCtx(QP qp,QPFetiCtx *ctxout)
{
  PetscContainer ctr;
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)qp,__FUNCT__,(PetscObject*)&ctr));
  if (!ctr) {
    PetscCall(QPFetiCtxCreate(&ctx));
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)qp),&ctr));
    PetscCall(PetscContainerSetPointer(ctr,ctx));
    PetscCall(PetscContainerSetUserDestroy(ctr,(PetscErrorCode(*)(void*))QPFetiCtxDestroy));
    PetscCall(PetscObjectCompose((PetscObject)qp,__FUNCT__,(PetscObject)ctr));
    PetscCall(PetscObjectDereference((PetscObject)ctr));
  }
  PetscCall(PetscContainerGetPointer(ctr,(void**)&ctx));
  *ctxout = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode QPFetiSetLocalToGlobalMapping(QP qp, IS l2g)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(l2g,IS_CLASSID,2);
  PetscCall(QPFetiGetCtx(qp,&ctx));
  ctx->l2g = l2g;
  PetscCall(PetscObjectReference((PetscObject)l2g));
  PetscCall(ISLocalToGlobalMappingCreateIS(l2g,&ctx->l2g_map));
  ctx->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode QPFetiSetInterfaceToGlobalMapping(QP qp, IS i2g)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(i2g,IS_CLASSID,2);
  PetscCall(QPFetiGetCtx(qp,&ctx));
  ctx->i2g = i2g;
  PetscCall(PetscObjectReference((PetscObject)i2g));
  PetscCall(ISLocalToGlobalMappingCreateIS(i2g,&ctx->i2g_map));
  ctx->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode QPFetiSetDirichlet(QP qp, IS dbcis, QPFetiNumberingType numtype, PetscBool enforce_by_B)
{
  QPFetiCtx ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qp,QP_CLASSID,1);
  PetscValidHeaderSpecific(dbcis,IS_CLASSID,2);
  PetscValidLogicalCollectiveEnum(qp,numtype,3);
  PetscValidLogicalCollectiveBool(qp,enforce_by_B,4);
  PetscCall(QPFetiGetCtx(qp,&ctx));
  PetscCall(QPFetiDirichletDestroy(&ctx->dbc));
  PetscCall(QPFetiDirichletCreate(dbcis,numtype,enforce_by_B,&ctx->dbc));
  ctx->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode QPFetiAssembleDirichlet_ModifyR_Private(QP qp, IS dbcis)
{
  Mat R, R_loc;
  PetscInt m, n_dbc_local;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(QPGetOperatorNullSpace(qp, &R));
  if (!R) PetscFunctionReturn(0);

  PetscCall(MatGetDiagonalBlock(R, &R_loc));
  PetscCall(ISGetLocalSize(dbcis, &n_dbc_local));

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)qp),&rank));
  PetscCall(PetscInfo(qp,"n_dbc_local=%d\n",n_dbc_local));
  if (n_dbc_local) {
    PetscCall(MatGetSize(R_loc, &m, NULL));
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, m, 0, 0, NULL, &R_loc));
  } else {
    PetscCall(PetscObjectReference((PetscObject)R_loc));
  }
  PetscCall(MatCreateBlockDiag(PetscObjectComm((PetscObject)R), R_loc, &R));
  PetscCall(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
  PetscCall(QPSetOperatorNullSpace(qp, R));
  PetscCall(MatDestroy(&R_loc));
  PetscCall(MatDestroy(&R));
  PetscFunctionReturn(0);
}

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
    PetscCall(PetscLogEventRegister("QPFetiAssembleDir",QP_CLASSID,&QP_Feti_AssembleDirichlet));
    PetscCall(PetscLogEventRegister("QPAddEq",QP_CLASSID,&QP_AddEq));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscLogEventBegin(QP_Feti_AssembleDirichlet,qp,0,0,0));
  PetscCall(QPFetiGetCtx(qp,&ctx));

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
      PetscCall(ISGetLocalSize(dbcis,&ndbc));
      PetscCall(ISGetIndices(dbcis,&dbcarr));
      PetscCall(ISGlobalToLocalMappingApply(ctx->l2g_map,IS_GTOLM_DROP,ndbc,dbcarr,&nout,NULL));
      PERMON_ASSERT(ndbc==nout,"n==nout");
      PetscCall(PetscMalloc1(ndbc,&dbc_l_arr));
      PetscCall(ISGlobalToLocalMappingApply(ctx->l2g_map,IS_GTOLM_DROP,ndbc,dbcarr,NULL,dbc_l_arr));
      PetscCall(ISRestoreIndices(dbcis,&dbcarr));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)qp),ndbc,dbc_l_arr,PETSC_OWN_POINTER,&dbc_l));
    } else {
      dbc_l = dbcis;
      PetscCall(PetscObjectReference((PetscObject)dbcis));
    }

    /* convert local indices to global decomposed */
    PetscCall(MatGetLocalToGlobalMapping(qp->A,&l2dg,NULL));
    PetscCall(ISLocalToGlobalMappingApplyIS(l2dg,dbc_l,&dbc_dg));
  } else {
    dbc_dg = dbcis;
    PetscCall(PetscObjectReference((PetscObject)dbcis));
  }

  if (enforce_by_B) {
    Mat B,Bt;
    Vec c;
    PetscInt  i, m, ndbc, start, stop;
    flg=PETSC_TRUE;

    PetscCall(PetscOptionsGetBool(NULL,NULL,"-EXTENSION_ON",&flg,NULL));

    if (flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD," MATEXTENSION type used for Dirichlet condition\n"));
      Mat At;
      IS cis;
      PetscLayout layout;
      PetscInt  *cis_arr;

      PetscCall(ISGetLocalSize(dbc_dg, &ndbc));
      PetscCall(MatGetLocalSize(qp->A, &m, NULL));

      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, ndbc, ndbc, 1, NULL, &At));
      PetscCall(MatSetFromOptions(At));

      for (i=0; i<ndbc; i++)  PetscCall(MatSetValue(At, i, i, 1, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(  At, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscObjectSetName((PetscObject)At,"Bdt_cond"));

      PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)qp), &layout));
      PetscCall(PetscLayoutSetBlockSize(layout, 1));
      PetscCall(PetscLayoutSetLocalSize(layout, ndbc));
      PetscCall(PetscLayoutSetUp(layout));
      PetscCall(PetscLayoutGetRange(layout, &start, &stop));
      PetscCall(PetscLayoutDestroy(&layout));

      PetscCall(PetscMalloc1(ndbc, &cis_arr));
      for (i=start; i<stop; i++)  cis_arr[i-start]=i;
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)qp), ndbc, cis_arr, PETSC_OWN_POINTER, &cis));

      PetscCall(MatCreateExtension(PetscObjectComm((PetscObject)qp), m, ndbc, PETSC_DECIDE, PETSC_DECIDE, At, dbc_dg, PETSC_TRUE, cis, &Bt));

      PetscCall(MatDestroy(&At));
      PetscCall(ISDestroy(&cis));
    } else {
      const PetscInt  *dbc_dg_arr;
      PetscCall(ISGetLocalSize(dbc_dg, &ndbc));
      PetscCall(MatGetLocalSize(qp->A, &m, NULL));

      PetscCall(MatCreate(PetscObjectComm((PetscObject)qp), &Bt));
      PetscCall(MatSetSizes(Bt, m, ndbc, PETSC_DETERMINE, PETSC_DETERMINE));
      PetscCall(MatSetType(Bt, MATMPIAIJ));
      PetscCall(MatMPIAIJSetPreallocation(Bt, 1, NULL, 0, NULL));

      PetscCall(MatGetOwnershipRangeColumn(Bt, &start, &stop));
      PetscCall(ISGetIndices(dbc_dg, &dbc_dg_arr));

      for (i = start; i<stop; i++) {
        PetscCall(MatSetValue(Bt,dbc_dg_arr[i-start],i,1.0,INSERT_VALUES));
      }

      PetscCall(ISRestoreIndices(dbc_dg, &dbc_dg_arr));
      PetscCall(MatAssemblyBegin(Bt, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Bt,MAT_FINAL_ASSEMBLY));
    }

    PetscCall(PetscObjectSetName((PetscObject)Bt,"Bdt"));
    PetscCall(PermonMatTranspose(Bt,MAT_TRANSPOSE_IMPLICIT,&B));
    PetscCall(PetscObjectSetName((PetscObject)B,"Bd"));
    PetscCall(MatDestroy(&Bt));

    PetscCall(PetscLogEventBegin(QP_AddEq,qp,0,0,0));
    PetscCall(MatCreateVecs(B,NULL,&c));
    PetscCall(MatMult(B,qp->x,c));
    PetscCall(QPAddEq(qp, B, c));
    PetscCall(PetscLogEventEnd(QP_AddEq,qp,0,0,0));

    PetscCall(VecDestroy(&c));
    PetscCall(MatDestroy(&B));

    /* parent is matis -> add dirichlet IS */
    if (qp->parent) {
      PetscCall(PetscStrcmp(qp->transform_name,"QPTMatISToBlockDiag",&flg));
      if (flg) {
        if (!dbc_l) {
          PetscCall(MatGetLocalToGlobalMapping(qp->A,&l2dg,NULL));
          PetscCall(ISGlobalToLocalMappingApplyIS(l2dg,IS_GTOLM_DROP,dbc_dg,&dbc_l));
        }
        ((QPTMatISToBlockDiag_Ctx*)qp->postSolveCtx)->isDir = dbc_dg;
        PetscCall(PetscObjectReference((PetscObject)dbc_dg));
      }
    }
  } else {
    Vec d;
    PetscScalar alpha;

    /* alpha=max(abs(diag(A)) */
    PetscCall(MatCreateVecs(qp->A,NULL,&d));
    PetscCall(MatGetDiagonal(qp->A,d));
    PetscCall(VecAbs(d));
    PetscCall(VecMax(d,NULL,&alpha));
    PetscCall(VecDestroy(&d));

    PetscCall(MatZeroRowsColumnsIS(qp->A, dbc_dg, alpha, qp->x, qp->b));
    PetscCall(QPFetiAssembleDirichlet_ModifyR_Private(qp, dbc_dg));
  }

  PetscCall(ISDestroy(&dbc_dg));
  PetscCall(ISDestroy(&dbc_l));
  PetscCall(PetscLogEventEnd(QP_Feti_AssembleDirichlet,qp,0,0,0));
  PetscFunctionReturnI(0);
}

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
  PetscCall(QPFetiGetCtx(qp,&ctx));
  if (ctx->setupcalled) PetscFunctionReturn(0);

  if (!registered) {
    PetscCall(PetscLogEventRegister("QPFetiSetUp",QP_CLASSID,&QP_Feti_SetUp));
    registered = PETSC_TRUE;
  }

  FllopTraceBegin;
  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  PetscCall(PetscLogEventBegin(QP_Feti_SetUp,qp,0,0,0));

  PERMON_ASSERT(qp->A,"Operator must be specified");
  PetscCall(MatGetLocalSize(qp->A, &nlocaldofs, NULL));

  PetscCall(PetscOptionsGetEnum(NULL,NULL,"-feti_gluing_type",FetiGluingTypes,(PetscEnum*)&type, NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-feti_gluing_exclude_dirichlet",&exclude_dir,NULL));
  PetscCall(PetscPrintf(comm, "============\n FETI gluing type: %s\n excluding Dirichlet DOFs? %d\n",FetiGluingTypes[type],exclude_dir));
  PetscCall(QPFetiAssembleDirichlet(qp));
  
  if (!ctx->l2g) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_WRONGSTATE,"L2G mapping must be set first - call QPFetiSetLocalToGlobalMapping before QPFetiSetUp");
  if (!ctx->i2g) SETERRQ(PetscObjectComm((PetscObject)qp),PETSC_ERR_ARG_WRONGSTATE,"I2G mapping must be set first - call QPFetiSetInterfaceToGlobalMapping before QPFetiSetUp");
  PetscCall(QPFetiAssembleGluing(qp, type, exclude_dir, &Bg));
  PetscCall(PetscPrintf(comm, "============\n"));

  PetscCall(PetscObjectSetName((PetscObject)Bg,"Bg"));
  PetscCall(QPAddEq(qp,Bg,NULL));
  PetscCall(MatDestroy(&Bg));

  if (!qp->BE) printf("child (BE) is needed for dualization\n");
  ctx->setupcalled = PETSC_TRUE;
  PetscCall(PetscLogEventEnd  (QP_Feti_SetUp,qp,0,0,0));
  PetscFunctionReturnI(0);
}

PetscErrorCode QPFetiGetI2Lmapping(MPI_Comm comm, IS l2g,  IS i2g,  IS *i2l_new)
{
  static PetscBool registered = PETSC_FALSE;
  const PetscInt  *i2g_arr, *l2g_arr;
  PetscInt ni, nl, i, j, *i2l_arr, *idx, *l2g_arr_copy;

  PetscFunctionBeginI;

  if (!registered) {
    PetscCall(PetscLogEventRegister("QPFetiGetI2Lmapping",QP_CLASSID,&QP_Feti_GetI2Lmapping));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscLogEventBegin(QP_Feti_GetI2Lmapping,0,0,0,0));

  PetscCall(ISGetIndices( i2g, &i2g_arr));
  PetscCall(ISGetIndices( l2g, &l2g_arr));
  PetscCall(ISGetLocalSize( i2g, &ni));
  PetscCall(ISGetLocalSize( l2g, &nl));

  PetscCall(PetscMalloc3(ni, &i2l_arr, nl, &idx, nl, &l2g_arr_copy));

  for (i=0; i<nl; i++) {
    idx[i]=i;
    l2g_arr_copy[i]=l2g_arr[i];
  }
  PetscCall(PetscSortIntWithArray(nl, l2g_arr_copy, idx));

  for (i=0; i< ni; i++) {
    PetscCall(PetscFindInt(i2g_arr[i], nl, l2g_arr_copy, &j));
    i2l_arr[i]=idx[j];
  }

  PetscCall(ISCreateGeneral(comm, ni, i2l_arr, PETSC_COPY_VALUES, i2l_new));

  PetscCall(ISRestoreIndices( i2g, &i2g_arr));
  PetscCall(ISRestoreIndices( l2g, &l2g_arr));
  PetscCall(PetscFree3(i2l_arr, idx, l2g_arr_copy));

  PetscCall(PetscLogEventEnd(QP_Feti_GetI2Lmapping,0,0,0,0));
  PetscFunctionReturnI(0);
}

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
    PetscCall(PetscLogEventRegister("QPFetiAssemGluing",QP_CLASSID,&QP_Feti_AssemGluing));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscLogEventBegin(QP_Feti_AssemGluing,0,0,0,0));

  PetscCall(PetscObjectGetComm((PetscObject)qp,&comm));
  PetscCall(QPFetiGetCtx(qp,&ctx));
  l2g = ctx->l2g;
  i2g = ctx->i2g;
  PetscCallMPI(MPI_Comm_size(comm, &commsize));  
  PetscCall(ISGetLocalSize(l2g, &nl)); 
  
  if (exclude_dir) {
    PetscCall(QPFetiGetGlobalDir( qp, ctx->dbc->is, ctx->dbc->numtype, &global_dir));
    PetscCall(ISAllGather(global_dir, &all_dir));
    PetscCall(ISSortRemoveDups(all_dir));
    PetscCall(ISDifference(i2g,all_dir,&i2g_less));
    PetscCall(ISDestroy(&global_dir));
  } else {
    PetscCall(PetscObjectReference((PetscObject)i2g));
    i2g_less=i2g;
  }     
  
  /* get i2l from l2g and i2g */
  PetscCall(QPFetiGetI2Lmapping(comm, ctx->l2g, i2g_less, &i2l));

  /* create Adt using SF */
  PetscCall(ISGetMinMax(i2g_less,NULL,&Nu));
  PetscCallMPI(MPI_Allreduce(&Nu, &Nug, 1, MPIU_INT, MPIU_MAX, comm));
  Nu = Nug+1;
   
  PetscCall(QPFetiGetBgtSF(comm, i2g_less, Nu, i2l, nl, type, &Bgt));

  PetscCall(PetscObjectSetName((PetscObject)Bgt,"Bgt"));

  /* create Bg from Bgt */
  PetscCall(MatCreateTranspose(Bgt,Bg_new));
  PetscCall(PetscObjectSetName((PetscObject)*Bg_new,"Bg"));

  PetscCall(ISDestroy(&i2l));
  PetscCall(ISDestroy(&i2g_less));
  PetscCall(MatDestroy(&Bgt));

  PetscCall(PetscLogEventEnd(QP_Feti_AssemGluing,0,0,0,0));
  PetscFunctionReturnI(0);
}

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
    PetscCall(PetscLogEventRegister("QPFetiGetBgtSF",QP_CLASSID,&QP_Feti_GetBgtSF));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscLogEventBegin(QP_Feti_GetBgtSF,0,0,0,0)); 

  PetscCallMPI(MPI_Comm_size(comm, &commsize));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  // first SF (all nodes from i2g) /STEP 1/
  PetscCall(ISGetLocalSize(i2g, &nleaves_SF1));
  PetscCall(ISGetIndices(i2g, &i2g_array));
  PetscCall(ISGetIndices(i2l, &i2l_array));

  PetscCall(PetscLayoutCreate(comm, &layout_SF1));
  PetscCall(PetscLayoutSetBlockSize(layout_SF1, 1));
  PetscCall(PetscLayoutSetSize(layout_SF1, Nu));
  PetscCall(PetscLayoutSetUp(layout_SF1));
  PetscCall(PetscSFCreate(comm, &SF1));
  PetscCall(PetscLayoutGetLocalSize(layout_SF1, &nroots_SF1));
  PetscCall(PetscSFSetGraphLayout(SF1, layout_SF1, nleaves_SF1, NULL, PETSC_COPY_VALUES, i2g_array));
  PetscCall(PetscSFSetRankOrder(SF1, PETSC_TRUE));

  // find out root's degree and send to leaves /STEP 2/
  PetscCall(PetscSFComputeDegreeBegin(SF1, &root_degree_onroots_SF1));
  // work between communication   
  PetscCall(PetscMalloc2( nleaves_SF1, &root_degree_onleaves_SF1, nleaves_SF1, &future_leavesSF2_onleaves));
  for (i=0; i<nleaves_SF1; i++) {
    root_degree_onleaves_SF1[i] = -1;
    future_leavesSF2_onleaves[i] = -1;
  }
  PetscCall(PetscSFComputeDegreeEnd(SF1, &root_degree_onroots_SF1));

  PetscCall(PetscSFBcastBegin(SF1, MPIU_INT, root_degree_onroots_SF1, root_degree_onleaves_SF1, MPI_REPLACE));
  // work between communication  
  // scatter - how much i will sent? 
  for (i=0, data_scat_size=0; i<nroots_SF1; i++) {
    data_scat_size += root_degree_onroots_SF1[i];
  }
  PetscCall(PetscMalloc1( data_scat_size, &future_leavesSF2_onroots));

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
  PetscCall(PetscSFBcastEnd(SF1, MPIU_INT, root_degree_onroots_SF1, root_degree_onleaves_SF1, MPI_REPLACE));
  // scatter itself /STEP 3/
  PetscCall(PetscSFScatterBegin(SF1, MPIU_INT, future_leavesSF2_onroots, future_leavesSF2_onleaves));
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
  PetscCall(PetscLayoutCreate(comm, &links));
  PetscCall(PetscLayoutSetBlockSize(links, 1));
  PetscCall(PetscLayoutSetLocalSize(links, n_link));
  PetscCall(PetscLayoutSetUp(links));
  PetscCall(PetscLayoutGetRange(links, &rstart, NULL));
  PetscCall(PetscSFScatterEnd(SF1, MPIU_INT, future_leavesSF2_onroots, future_leavesSF2_onleaves));

  // "make" leaves for SF2, create local indexes  /STEP 5/
  for (i=0; i<nleaves_SF1; i++) {
    nleaves_SF2 = nleaves_SF2 + future_leavesSF2_onleaves[i];
    if (future_leavesSF2_onleaves[i]>0) nleaves_SF2_unique++;
  }

  PetscCall(PetscMalloc4(nleaves_SF2, &leaves_SF2, nleaves_SF2, &local_idx, nleaves_SF2, &condensed_local_idx, nleaves_SF2, &root_degree_onleaves_fromSF1onSF2));
  PetscCall(PetscMalloc1(nleaves_SF2_unique, &ris_array));
  PetscCall(PetscMalloc1(nleaves_SF2_unique, &prealloc_seqEX));

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
  PetscCall(PetscSFCreate(comm, &SF2));
  PetscCall(PetscSFSetGraphLayout(SF2, layout_SF1, nleaves_SF2, NULL, PETSC_COPY_VALUES, leaves_SF2));
  PetscCall(PetscSFSetRankOrder(SF2, PETSC_TRUE));

  // scatter links /STEP 6/
  // how much i will sent? 
  PetscCall(PetscSFComputeDegreeBegin(SF2, &root_degree_onroots_SF2));
  PetscCall(PetscSFComputeDegreeEnd(SF2, &root_degree_onroots_SF2));

  for (i=0, data_scat_size=0; i<nroots_SF1; i++) {
    data_scat_size += root_degree_onroots_SF2[i];
  }
  PetscCall(PetscMalloc1(data_scat_size, &link_onroot));
  PetscCall(PetscMalloc1(nleaves_SF2, &link_onleaves));
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

  PetscCall(PetscSFScatterBegin(SF2, MPIU_INT, link_onroot, link_onleaves));
  // work between communication  
  // new layout for links 
  PetscCall(PetscLayoutGetSize(links, &n_link));
  PetscCall(PetscLayoutDestroy(&links));
  PetscCall(PetscLayoutCreate(comm, &links));
  PetscCall(PetscLayoutSetBlockSize(links, 1));
  PetscCall(PetscLayoutSetSize(links, n_link));
  PetscCall(PetscLayoutSetUp(links));
  // compute divison - only orthonormal /STEP 6.5/  
  PetscCall(PetscMalloc1(nleaves_SF2, &division));
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
  PetscCall(PetscSFScatterEnd(SF2, MPIU_INT, link_onroot, link_onleaves));
  PetscCall(PetscSortInt( nleaves_SF2, link_onleaves));
  // link SF /STEP 7/
  PetscCall(PetscSFCreate(comm, &link_SF));
  PetscCall(PetscSFSetGraphLayout(link_SF, links, nleaves_SF2, NULL, PETSC_COPY_VALUES, link_onleaves));
  PetscCall(PetscSFSetRankOrder(link_SF, PETSC_TRUE));

  //get max ranks of leaves (find out if + or -) /STEP 8/
  PetscCall(PetscMalloc2(n_link, &max_rank_onroot_linkSF, nleaves_SF2, &max_rank_onleaves_linkSF));
  for (i=0; i<n_link; i++) max_rank_onroot_linkSF[i] = -1;
  for (i=0; i<nleaves_SF2; i++) max_rank_onleaves_linkSF[i] = rank;

  PetscCall(PetscSFReduceBegin(link_SF, MPIU_INT, max_rank_onleaves_linkSF, max_rank_onroot_linkSF, MPIU_MAX));
  // work between communication  
  PetscCall(PetscLayoutGetRange(links, &rstart, &rend));
  PetscCall(PetscLayoutGetLocalSize(links, &n_link));
  PetscCall(PetscMalloc3(nleaves_SF1, &prealloc_diag, nleaves_SF1, &prealloc_ofdiag, nleaves_SF1, &prealloc_seq));
  for (i=0; i<nleaves_SF1; i++) {
    prealloc_diag[i] = 0;
    prealloc_ofdiag[i] = 0;
    prealloc_seq[i] = 0;
  }
  PetscCall(PetscSFReduceEnd(link_SF, MPIU_INT, max_rank_onleaves_linkSF, max_rank_onroot_linkSF, MPIU_MAX));

  PetscCall(PetscSFBcastBegin(link_SF, MPIU_INT, max_rank_onroot_linkSF, max_rank_onleaves_linkSF, MPI_REPLACE));
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
  PetscCall(PetscMalloc1(nleaves_SF2, &values));

  PetscCall(PetscSFBcastEnd(link_SF, MPIU_INT, max_rank_onroot_linkSF, max_rank_onleaves_linkSF, MPI_REPLACE));

  PetscBool flg_SCALE_ON=PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-SCALE_ON",&flg_SCALE_ON,NULL));
  if (flg_SCALE_ON && (type==FETI_GLUING_NONRED || type==FETI_GLUING_FULL)) PetscCall(PetscPrintf(PETSC_COMM_WORLD," SCALING\n"));

/* compose array of neighbors to Bg */
  {
    PetscInt n;
    PetscInt *rank_onroot_linkSF, *rank_onleaves_linkSF;

    PetscCall(PetscMalloc1(n_link, &rank_onroot_linkSF));
    for (i=0; i<n_link; i++) rank_onroot_linkSF[i] = PETSC_MAX_INT;
    PetscCall(PetscMalloc1(2*nleaves_SF2, &rank_onleaves_linkSF));
    for (i=0; i<nleaves_SF2; i++) rank_onleaves_linkSF[i] = rank;
    for (i=nleaves_SF2; i<2*nleaves_SF2; i++) rank_onleaves_linkSF[i] = commsize;

    PetscCall(PetscSFReduceBegin(link_SF, MPIU_INT, rank_onleaves_linkSF, rank_onroot_linkSF, MPIU_MIN));
    PetscCall(PetscSFReduceEnd(link_SF, MPIU_INT, rank_onleaves_linkSF, rank_onroot_linkSF, MPIU_MIN));
    PetscCall(PetscSFBcastBegin(link_SF, MPIU_INT, rank_onroot_linkSF, rank_onleaves_linkSF, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(link_SF, MPIU_INT, rank_onroot_linkSF, rank_onleaves_linkSF, MPI_REPLACE));
    PetscCall(PetscMemcpy(&rank_onleaves_linkSF[nleaves_SF2],max_rank_onleaves_linkSF,nleaves_SF2*sizeof(PetscInt)));
    n = 2*nleaves_SF2;
    PetscCall(PetscSortRemoveDupsInt(&n,rank_onleaves_linkSF));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,rank_onleaves_linkSF,PETSC_COPY_VALUES,&myneighbors));

    PetscCall(PetscFree(rank_onleaves_linkSF));
    PetscCall(PetscFree(rank_onroot_linkSF));
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

  PetscCall(ISRestoreIndices(i2l, &i2l_array));
  PetscCall(ISRestoreIndices(i2g, &i2g_array));
  PetscBool flg_EXTENSION_ON=PETSC_TRUE;
  PetscBool flg_MATGLUING_ON=PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-EXTENSION_ON",&flg_EXTENSION_ON,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-MATGLUING_ON",&flg_MATGLUING_ON,NULL));

  if (flg_EXTENSION_ON) {
    IS cis, ris;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," MATEXTENSION type used for gluing matrix\n"));

    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nleaves_SF2_unique, nleaves_SF2, -1, prealloc_seqEX, &At));
    PetscCall(MatSetFromOptions(At));
    for (i=0; i<nleaves_SF2; i++)  PetscCall(MatSetValue(At, condensed_local_idx[i], i, values[i], INSERT_VALUES));

    PetscCall(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(At, MAT_FINAL_ASSEMBLY));

    PetscCall(PetscObjectSetName((PetscObject)At, "Bgt_cond"));

    PetscCall(ISCreateGeneral(comm, nleaves_SF2, link_onleaves, PETSC_OWN_POINTER, &cis));
    PetscCall(ISCreateGeneral(comm, nleaves_SF2_unique, ris_array, PETSC_OWN_POINTER, &ris));

    PetscCall(MatCreateExtension(comm, nl, n_link, PETSC_DECIDE, PETSC_DECIDE, At, ris, PETSC_FALSE, cis, &Bgt));
    PetscCall(MatDestroy(&At));
    PetscCall(ISDestroy(&cis));
    PetscCall(ISDestroy(&ris));

  } else {
    //TODO allocate this array only if needed
    PetscCall(PetscFree(ris_array));

    if (!flg_MATGLUING_ON) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD," just PetscSF (no special type) used for gluing matrix\n"));

      PetscCall(MatCreate(comm, &At ));
      PetscCall(MatSetSizes(At, nleaves_SF1, n_link,  PETSC_DETERMINE, PETSC_DETERMINE ));
      PetscCall(MatSetFromOptions(At));
      PetscCall(MatMPIAIJSetPreallocation(At , 0, prealloc_diag, 0, prealloc_ofdiag));
      PetscCall(MatSeqAIJSetPreallocation(At, 0, prealloc_seq));

      PetscCall(MatGetOwnershipRange(At,  &idx, &idx2));
      for (i=0; i<nleaves_SF2; i++)  PetscCall(MatSetValue(At, local_idx[i]+idx, link_onleaves[i], values[i], INSERT_VALUES));

      PetscCall(MatAssemblyBegin(At, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(At, MAT_FINAL_ASSEMBLY));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD," MATGLUING type used for gluing matrix\n"));
      PetscCall(MatCreateGluing(comm, nleaves_SF1,  nleaves_SF2_unique,  n_link,  local_idx, values, link_SF, &At));
    }

    Mat T_loc, T;
    VecScatter scatter;
    Vec n_vec, N_vec;
    PetscInt ni;

    PetscCall(ISGetLocalSize(i2l,&ni));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, ni, &n_vec));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nl, &N_vec));
    PetscCall(VecScatterCreate(n_vec, NULL, N_vec, i2l, &scatter));
    PetscCall(MatCreateScatter(PETSC_COMM_SELF, scatter, &T_loc));
    PetscCall(MatCreateBlockDiag(comm, T_loc, &T));

    Mat Bt_arr[] ={At, T};
    PetscCall(MatCreateProd(comm, 2, Bt_arr, &Bgt));

    /* hotfix for MatMatBlockDiagMultByColumns_Private */
    PetscCall(PetscObjectCompose((PetscObject)Bgt, "T_loc", (PetscObject)T_loc));
    PetscCall(PetscObjectCompose((PetscObject)Bgt, "Adt", (PetscObject)At));

    PetscCall(PetscFree(link_onleaves));
    PetscCall(MatDestroy(&At));
    PetscCall(MatDestroy(&T_loc));
    PetscCall(MatDestroy(&T));
    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&n_vec));
    PetscCall(VecDestroy(&N_vec));
  }

  PetscCall(PetscObjectCompose((PetscObject)Bgt,"myneighbors",(PetscObject)myneighbors));
  PetscCall(ISDestroy(&myneighbors));
  *Bgt_out=Bgt;

  PetscCall(PetscFree2( root_degree_onleaves_SF1, future_leavesSF2_onleaves));
  PetscCall(PetscFree(future_leavesSF2_onroots));
  PetscCall(PetscFree(prealloc_seqEX));
  PetscCall(PetscFree(link_onroot));
  PetscCall(PetscFree(division));
  PetscCall(PetscFree4(leaves_SF2, local_idx, condensed_local_idx, root_degree_onleaves_fromSF1onSF2));
  PetscCall(PetscFree3(prealloc_diag, prealloc_ofdiag, prealloc_seq)); 
  PetscCall(PetscFree(values));
  PetscCall(PetscFree2(max_rank_onroot_linkSF, max_rank_onleaves_linkSF));
  PetscCall(PetscSFDestroy(&SF1));
  PetscCall(PetscSFDestroy(&SF2));
  PetscCall(PetscSFDestroy(&link_SF));
  PetscCall(PetscLayoutDestroy(&layout_SF1));
  PetscCall(PetscLayoutDestroy(&links));

  PetscCall(PetscLogEventEnd(QP_Feti_GetBgtSF,0,0,0,0));
  PetscFunctionReturnI(0);
}

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

    PetscCall(ISGetLocalSize(dbc,&ndbc));
    PetscCall(ISGetIndices(dbc,&dbc_arr));

    if (numtype==FETI_GLOBAL_DECOMPOSED) {
      PetscInt nout;
      ISLocalToGlobalMapping l2dg;

      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)qp), " FETI_GLOBAL_DECOMPOSED numbering of Dirichlet DOFs\n"));
      /* convert global decomposed indices to local */
      PetscCall(MatGetLocalToGlobalMapping(qp->A,&l2dg,NULL));
      PetscCall(ISGlobalToLocalMappingApply(l2dg,IS_GTOLM_DROP,ndbc,dbc_arr,&nout,NULL));
      PERMON_ASSERT(ndbc==nout,"n==nout");
      PetscCall(PetscMalloc1(ndbc,&dbc_l_arr_nonconst));
      PetscCall(ISGlobalToLocalMappingApply(l2dg,IS_GTOLM_DROP,ndbc,dbc_arr,NULL,dbc_l_arr_nonconst));
      dbc_l_arr = dbc_l_arr_nonconst;
    } else { //FETI_LOCAL
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)qp), " FETI_LOCAL numbering of Dirichlet DOFs\n"));
      dbc_l_arr = dbc_arr;
    }
    /* convert local indices to global undecomposed */
    PetscCall(QPFetiGetCtx(qp,&ctx));
    PetscCall(PetscMalloc1(ndbc,&global_arr));
    PetscCall(ISLocalToGlobalMappingApply(ctx->l2g_map, ndbc, dbc_l_arr, global_arr));

    PetscCall(ISRestoreIndices(dbc,&dbc_arr));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)qp),ndbc,global_arr,PETSC_OWN_POINTER, dbc_g));
    PetscCall(PetscFree(dbc_l_arr_nonconst));

  } else { //FETI_GLOBAL_UNDECOMPOSED
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)qp), " FETI_GLOBAL_UNDECOMPOSED numbering of Dirichlet DOFs\n"));
    PetscCall(PetscObjectReference((PetscObject)dbc));
    *dbc_g =dbc;
  } 
  PetscFunctionReturnI(0);
}
