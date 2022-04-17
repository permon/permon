
#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>
#include <permonksp.h>
#include <petsc/private/pcimpl.h>
#if defined(PETSC_HAVE_MUMPS)
#include <permon/private/petsc/mat/mumpsimpl.h>
#endif

PetscLogEvent Mat_Inv_Explicitly, Mat_Inv_SetUp;

static PetscErrorCode MatInvCreateInnerObjects_Inv(Mat imat);

#undef __FUNCT__
#define __FUNCT__ "MatInvKSPSetOptionsPrefix_Inv"
static PetscErrorCode MatInvKSPSetOptionsPrefix_Inv(Mat imat)
{
  Mat_Inv        *inv = (Mat_Inv*)imat->data;
  const char     *prefix;

  PetscFunctionBegin;
  CHKERRQ(MatGetOptionsPrefix(imat,&prefix));
  CHKERRQ(KSPSetOptionsPrefix(inv->innerksp,prefix));
  CHKERRQ(KSPAppendOptionsPrefix(inv->innerksp,"mat_inv_"));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvSetRegularizationType_Inv"
static PetscErrorCode MatInvSetRegularizationType_Inv(Mat imat,MatRegularizationType type)
{
  Mat_Inv        *inv = (Mat_Inv*)imat->data;

  PetscFunctionBegin;
  if (type != inv->regtype) {
    inv->regtype = type;
    inv->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvGetRegularizationType_Inv"
static PetscErrorCode MatInvGetRegularizationType_Inv(Mat imat,MatRegularizationType *type)
{
  Mat_Inv        *inv = (Mat_Inv*)imat->data;
  
  PetscFunctionBegin;
  *type = inv->regtype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvComputeNullSpace_Inv"
#if defined(PETSC_HAVE_MUMPS)
static PetscErrorCode MatInvComputeNullSpace_Inv(Mat imat)
{
  Mat_Inv *inv = (Mat_Inv*)imat->data;
  Mat Kl=NULL,R=NULL,Rl=NULL,F=NULL;
  KSP ksp;
  PC pc;
  PetscInt m,M,mm = 0,defect;
  PetscBool flg,blockdiag = PETSC_FALSE;
  MatSolverType type;
  Mat_MUMPS *mumps = NULL;
  PetscReal null_pivot_threshold = -1e-8;
  MPI_Comm blockComm;

  PetscFunctionBeginI;
  CHKERRQ(MatInvGetKSP(imat,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCGetOperators(pc,&Kl,NULL));
  CHKERRQ(MatGetLocalSize(Kl,&m,NULL));
  CHKERRQ(MatGetSize(Kl,&M,NULL));
  CHKERRQ(MatPrintInfo(Kl));
  CHKERRQ(PetscObjectGetComm((PetscObject)Kl,&blockComm));

  {
    Mat K;
    PetscMPIInt commsize;

    CHKERRQ(MatInvGetMat(imat,&K));
    CHKERRQ(MPI_Comm_size(blockComm,&commsize));
    if (K != Kl) {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)K,MATBLOCKDIAG,&blockdiag));
      PERMON_ASSERT(blockdiag, "K should be blockdiag in this case");
      PERMON_ASSERT(commsize==1, "Kl should be serial");
    }
  }
  
  if (Kl->spd_set && Kl->spd) {
    defect = 0;
    mm = m;
  } else {
    /* MUMPS matrix type (sym) is set to 2 automatically (see MatGetFactor_aij_mumps). */
    CHKERRQ(PCFactorGetMatSolverType(pc,&type));
    CHKERRQ(PetscStrcmp(type,MATSOLVERMUMPS,&flg)); 
    if (flg) CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCCHOLESKY,&flg));
    /* TODO We need to call PCSetUP() (which does the factorization) before being able to call PCFactorGetMatrix().
       Maybe we could do something on the PETSc side to overcome this. */
    if (flg) {
      /* If MUMPS Cholesky is used, avoid doubled factorization. */
      char opts[128];
      if (inv->type == MAT_INV_BLOCKDIAG) {
        CHKERRQ(PetscSNPrintf(opts,sizeof(opts),"-%ssub_mat_mumps_icntl_24 1 -%ssub_mat_mumps_cntl_3 %e",((PetscObject)pc)->prefix,((PetscObject)pc)->prefix,null_pivot_threshold));
      } else {
        CHKERRQ(PetscSNPrintf(opts,sizeof(opts),"-%smat_mumps_icntl_24 1 -%smat_mumps_cntl_3 %e",((PetscObject)pc)->prefix,((PetscObject)pc)->prefix,null_pivot_threshold));
      }
      CHKERRQ(PetscOptionsInsertString(NULL,opts));
      CHKERRQ(PCSetFromOptions(pc));
      CHKERRQ(PCSetUp(pc));
      CHKERRQ(PCFactorGetMatrix(pc,&F));
      mumps =(Mat_MUMPS*)F->data;
    } else {
      CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)imat), "WARNING: Performing extra factorization with MUMPS Cholesky just for nullspace detection. Avoid this by setting MUMPS Cholesky as MATINV solver.\n"));
      CHKERRQ(MatGetFactor(Kl,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F));
      CHKERRQ(MatMumpsSetIcntl(F,24,1)); /* null pivot detection */
      CHKERRQ(MatMumpsSetCntl(F,3,null_pivot_threshold)); /* null pivot threshold */
      mumps =(Mat_MUMPS*)F->data;
      CHKERRQ(MatCholeskyFactorSymbolic(F,Kl,NULL,NULL));
      CHKERRQ(MatCholeskyFactorNumeric(F,Kl,NULL));
    }
    CHKERRQ(MatMumpsGetInfog(F,28,&defect)); /* get numerical defect, i.e. number of null pivots encountered during factorization */
    /* mumps->petsc_size > 1 implies mumps->id.ICNTL(21) = 1 (distributed solution ) */
    mm = (defect && mumps->petsc_size > 1) ? mumps->id.lsol_loc : m;  /* = length of sol_loc = INFO(23) */
  }

  CHKERRQ(MatCreateDensePermon(blockComm,mm,PETSC_DECIDE,M,defect,NULL,&Rl));

  if (defect) {
    /* stash sol_loc allocated in MatFactorNumeric_MUMPS() */
    MumpsScalar *sol_loc_orig = mumps->id.sol_loc;
    MumpsScalar *array;

    /* inject matrix array as sol_loc */
    CHKERRQ(MatDenseGetArray(Rl,(MumpsScalar**)&array));
    if (mumps->petsc_size > 1) {
      mumps->id.sol_loc = array;
      if (!mumps->myid) {
        /* Define dummy rhs on the host otherwise MUMPS fails with INFOG(1)=-22,INFOG(2)=7 */
        CHKERRQ(PetscMalloc1(M,&mumps->id.rhs));
      }
    } else mumps->id.rhs = array;
    /* mumps->id.nrhs is reset by MatMatSolve_MUMPS()/MatSolve_MUMPS() */
    mumps->id.nrhs = defect;
    mumps->id.lrhs = (mumps->petsc_size > 1) ? M : mm;
    mumps->id.lrhs_loc = mm;
    CHKERRQ(MatMumpsSetIcntl(F,25,-1)); /* compute complete null space */
    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(&mumps->id);
    if (mumps->id.INFOG(1) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d,INFOG(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFOG(2));

    if (mumps->petsc_size > 1 && !mumps->myid) {
      CHKERRQ(PetscFree(mumps->id.rhs));
    }
    CHKERRQ(MatMumpsSetIcntl(F,25,0)); /* perform a normal solution step next time */

    /* restore matrix array */
    CHKERRQ(MatDenseRestoreArray(Rl,(MumpsScalar**)&array));
    /* restore stashed sol_loc */
    mumps->id.sol_loc = sol_loc_orig;
  }

  //TODO return just NULL if defect=0 ?
  if (blockdiag) {
    CHKERRQ(MatCreateBlockDiag(PETSC_COMM_WORLD,Rl,&R));
    CHKERRQ(FllopPetscObjectInheritName((PetscObject)Rl,(PetscObject)R,"_loc"));
    CHKERRQ(MatDestroy(&Rl));
  } else if (defect && mumps->petsc_size > 1) {
    IS isol_is;
    /* redistribute to get conforming local size */
    CHKERRQ(MatCreateDensePermon(blockComm,m,PETSC_DECIDE,M,defect,NULL,&R));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,mm,mumps->id.isol_loc,PETSC_USE_POINTER,&isol_is));
    CHKERRQ(MatRedistributeRows(Rl,isol_is,1,R)); /* MUMPS uses 1-based numbering */
    CHKERRQ(MatDestroy(&Rl));
    CHKERRQ(ISDestroy(&isol_is));
  } else {
    R = Rl;
  }
  CHKERRQ(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscObjectSetName((PetscObject)R,"R"));
  CHKERRQ(MatPrintInfo(R));
  CHKERRQ(MatInvSetNullSpace(imat,R));
  CHKERRQ(MatDestroy(&R));
  PetscFunctionReturnI(0);
}
#else
static PetscErrorCode MatInvComputeNullSpace_Inv(Mat imat)
{
  PetscFunctionBeginI;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"MUMPS library is currently needed for nullspace computation");
  PetscFunctionReturnI(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "MatInvSetNullSpace_Inv"
static PetscErrorCode MatInvSetNullSpace_Inv(Mat imat,Mat R)
{
  Mat_Inv        *inv = (Mat_Inv*)imat->data;
  
  PetscFunctionBegin;
  if (R != inv->R) {
    if (R) {
#if defined(PETSC_USE_DEBUG)
      CHKERRQ(MatCheckNullSpace(inv->A, R, PETSC_SMALL));
#endif
      CHKERRQ(PetscObjectReference((PetscObject)R));
    }
    CHKERRQ(MatDestroy(&inv->R));
    inv->R = R;
    inv->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvGetNullSpace_Inv"
static PetscErrorCode MatInvGetNullSpace_Inv(Mat imat,Mat *R)
{
  Mat_Inv        *inv = (Mat_Inv*)imat->data;
  
  PetscFunctionBegin;
  *R = inv->R;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetTolerances_Inv"
static PetscErrorCode MatInvSetTolerances_Inv(Mat imat, PetscReal rtol, PetscReal abstol,
    PetscReal dtol, PetscInt maxits)
{
  KSP ksp;

  PetscFunctionBegin;
  CHKERRQ(MatInvGetKSP(imat,&ksp));
  CHKERRQ(KSPSetTolerances(ksp, rtol, abstol, dtol, maxits));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetKSP_Inv"
static PetscErrorCode MatInvGetKSP_Inv(Mat imat, KSP *ksp)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;

  PetscFunctionBegin;
  if (!inv->ksp) {
    CHKERRQ(MatInvCreateInnerObjects_Inv(imat));
  }
  *ksp = inv->innerksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetRegularizedMat_Inv"
static PetscErrorCode MatInvGetRegularizedMat_Inv(Mat imat, Mat *A)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  KSP ksp;

  PetscFunctionBegin;
  if (!inv->setupcalled) SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_ARG_WRONGSTATE,"This function can be called only after MatInvSetUp");
  CHKERRQ(MatInvGetKSP(imat,&ksp));
  CHKERRQ(KSPGetOperators(ksp, A, NULL));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetMat_Inv"
static PetscErrorCode MatInvGetMat_Inv(Mat imat, Mat *A)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;

  PetscFunctionBegin;
  *A = inv->A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPC_Inv"
static PetscErrorCode MatInvGetPC_Inv(Mat imat, PC *pc)
{
  KSP ksp;

  PetscFunctionBegin;
  CHKERRQ(MatInvGetKSP(imat, &ksp));
  CHKERRQ(KSPGetPC(ksp, pc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetMat_Inv"
static PetscErrorCode MatInvSetMat_Inv(Mat imat, Mat A)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  PetscInt m, n, M, N;

  PetscFunctionBegin;
  if (A == inv->A) PetscFunctionReturn(0);

  CHKERRQ(MatInvReset(imat));
  CHKERRQ(MatGetSize(A, &M, &N));
  CHKERRQ(MatGetLocalSize(A, &m, &n));
  CHKERRQ(MatSetSizes(imat, m, n, M, N));
  inv->A = A;
  CHKERRQ(PetscObjectReference((PetscObject)A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetRedundancy_Inv"
static PetscErrorCode MatInvGetRedundancy_Inv(Mat imat, PetscInt *red)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  
  PetscFunctionBegin;
  *red = inv->redundancy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetRedundancy_Inv"
static PetscErrorCode MatInvSetRedundancy_Inv(Mat imat, PetscInt red)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  
  PetscFunctionBegin;
  if (inv->redundancy == red) PetscFunctionReturn(0);
  inv->redundancy = red;
  inv->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPsubcommType_Inv"
static PetscErrorCode MatInvGetPsubcommType_Inv(Mat imat, PetscSubcommType *type)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  
  PetscFunctionBegin;
  *type = inv->psubcommType;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetPsubcommType_Inv"
static PetscErrorCode MatInvSetPsubcommType_Inv(Mat imat, PetscSubcommType type)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  
  PetscFunctionBegin;
  if (inv->psubcommType == type) PetscFunctionReturn(0);
  inv->psubcommType = type;
  inv->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPsubcommColor_Inv"
static PetscErrorCode MatInvGetPsubcommColor_Inv(Mat imat, PetscMPIInt *color)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  PC pc;
  PC_Redundant *red;

  PetscFunctionBegin;
  CHKERRQ(KSPGetPC(inv->ksp, &pc));
  red = (PC_Redundant*)pc->data;
  *color = red->psubcomm->color;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetType_Inv"
static PetscErrorCode MatInvGetType_Inv(Mat imat, MatInvType *type)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;

  PetscFunctionBegin;
  *type = inv->type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetType_Inv"
static PetscErrorCode MatInvSetType_Inv(Mat imat, MatInvType type)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  
  PetscFunctionBegin;
  if (inv->type == type) PetscFunctionReturn(0);
  inv->type = type;
  inv->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvReset_Inv"
static PetscErrorCode MatInvReset_Inv(Mat imat)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;

  PetscFunctionBeginI;
  CHKERRQ(KSPReset(inv->ksp));
  inv->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetUp_Inv"
static PetscErrorCode MatInvSetUp_Inv(Mat imat)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;

  FllopTracedFunctionBegin;
  if (inv->setupcalled) PetscFunctionReturn(0);
  if (inv->type == MAT_INV_BLOCKDIAG && inv->redundancy > 0) SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_SUP, "Cannot use MAT_INV_BLOCKDIAG and redundancy at the same time");
  if (inv->regtype && !inv->R) SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_ARG_WRONGSTATE,"regularization is requested but nullspace is not set");

  FllopTraceBegin;
  CHKERRQ(PetscLogEventBegin(Mat_Inv_SetUp,imat,0,0,0));
  {
    CHKERRQ(MatInvCreateInnerObjects_Inv(imat));
    CHKERRQ(KSPSetUp(inv->ksp));
    CHKERRQ(KSPSetUpOnBlocks(inv->ksp));
  }

  inv->setupcalled = PETSC_TRUE;
  CHKERRQ(MatInheritSymmetry(inv->A,imat));
  CHKERRQ(PetscLogEventEnd(Mat_Inv_SetUp,imat,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvCreateInnerObjects_Inv"
static PetscErrorCode MatInvCreateInnerObjects_Inv(Mat imat)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  Mat Areg,A_inner;
  PC pc;
  PetscBool factorizable,parallel,flg,own;
  KSPType default_ksptype;
  PCType  default_pctype;
  MatSolverType default_pkg;
  PetscMPIInt size;

  FllopTracedFunctionBegin;
  if (inv->inner_objects_created) PetscFunctionReturn(0);
  if (inv->type == MAT_INV_BLOCKDIAG && inv->redundancy > 0) SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_SUP, "Cannot use MAT_INV_BLOCKDIAG and redundancy at the same time");
  if (inv->regtype && !inv->R) SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_ARG_WRONGSTATE,"regularization is requested but nullspace is not set");
  CHKERRQ(MPI_Comm_size(PetscObjectComm((PetscObject)imat),&size));

  FllopTraceBegin;
  if (!inv->ksp) {
    CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)imat),&inv->ksp));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)inv->ksp,(PetscObject)imat,1));
    CHKERRQ(PetscLogObjectParent((PetscObject)imat,(PetscObject)inv->ksp));
  }

  own = ((PetscObject)inv->A)->refct == 1 ? PETSC_TRUE : PETSC_FALSE;
  CHKERRQ(MatRegularize(inv->A,inv->R,inv->regtype,&Areg));
  CHKERRQ(PetscOptionsHasName(NULL,((PetscObject)imat)->prefix,"-mat_inv_mat_type",&flg));
  if (inv->setfromoptionscalled && flg && inv->A == Areg && !own) {
    CHKERRQ(PetscInfo(fllop,"duplicating inner matrix to allow to apply options only internally\n"));
    CHKERRQ(PetscObjectDereference((PetscObject)Areg));
    CHKERRQ(MatDuplicate(Areg, MAT_COPY_VALUES, &Areg));
  }
  CHKERRQ(KSPSetOperators(inv->ksp, Areg, Areg));

  if (inv->type == MAT_INV_BLOCKDIAG) {
    CHKERRQ(MatGetDiagonalBlock(Areg,&A_inner));
  } else {
    A_inner = Areg;
  }

  CHKERRQ(FllopPetscObjectInheritPrefixIfNotSet((PetscObject)Areg,(PetscObject)imat,"mat_inv_"));
  CHKERRQ(FllopPetscObjectInheritPrefixIfNotSet((PetscObject)A_inner,(PetscObject)imat,"mat_inv_"));
  if (inv->setfromoptionscalled) {
    CHKERRQ(PetscInfo(fllop,"setting inner matrix with prefix %s from options\n",((PetscObject)A_inner)->prefix));
    CHKERRQ(PermonMatSetFromOptions(A_inner));
  }

  factorizable = A_inner->ops->getvalues ? PETSC_TRUE : PETSC_FALSE;    /* if elements of matrix are accessible it is likely an explicitly assembled matrix */
  default_pkg           = MATSOLVERPETSC;
  if (factorizable) {
#if defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_PASTIX) || defined(PETSC_HAVE_SUPERLU_DIST)
    PetscBool flg;
#endif
    default_ksptype     = KSPPREONLY;
    default_pctype      = PCCHOLESKY;
    parallel            = PETSC_FALSE;
#if defined(PETSC_HAVE_MUMPS)
    CHKERRQ(MatGetFactorAvailable(A_inner,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&flg));
    if (flg) {
      default_pkg       = MATSOLVERMUMPS;
      default_pctype    = PCCHOLESKY;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_PASTIX)
    CHKERRQ(MatGetFactorAvailable(A_inner,MATSOLVERPASTIX,MAT_FACTOR_CHOLESKY,&flg));
    if (flg) {
      default_pkg       = MATSOLVERPASTIX;
      default_pctype    = PCCHOLESKY;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
    CHKERRQ(MatGetFactorAvailable(A_inner,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&flg));
    if (flg) {
      default_pkg       = MATSOLVERSUPERLU_DIST;
      default_pctype    = PCLU;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_MUMPS)
    CHKERRQ(MatGetFactorAvailable(A_inner,MATSOLVERMUMPS,MAT_FACTOR_LU,&flg));
    if (flg) {
      default_pkg       = MATSOLVERMUMPS;
      default_pctype    = PCLU;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_PASTIX)
    CHKERRQ(MatGetFactorAvailable(A_inner,MATSOLVERPASTIX,MAT_FACTOR_LU,&flg));
    if (flg) {
      default_pkg       = MATSOLVERPASTIX;
      default_pctype    = PCLU;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
  } else {
    default_ksptype     = KSPCG;
    default_pctype      = PCNONE;
    parallel            = PETSC_TRUE;
    goto chosen;
  }
  chosen:

  if (inv->redundancy < 0) {
    if (parallel || size==1) {
      inv->redundancy = 0;
    } else {
      inv->redundancy = size;
    }
  }

  if (inv->type == MAT_INV_BLOCKDIAG || inv->redundancy) {
    const char *prefix;
    CHKERRQ(KSPSetType(inv->ksp, KSPPREONLY));
    CHKERRQ(KSPGetPC(inv->ksp, &pc));
    CHKERRQ(MatGetOptionsPrefix(imat,&prefix));
    CHKERRQ(PCSetOptionsPrefix(pc,prefix));
    CHKERRQ(PCAppendOptionsPrefix(pc,"mat_inv_"));
    if (inv->type == MAT_INV_BLOCKDIAG) {
      KSP *kspp;
      CHKERRQ(PCSetType(pc, PCBJACOBI));
      CHKERRQ(PCSetUp(pc));
      CHKERRQ(PCBJacobiGetSubKSP(pc, PETSC_IGNORE, PETSC_IGNORE, &kspp));
      inv->innerksp = *kspp;
    } else {
      char stri[1024];
      CHKERRQ(PetscSNPrintf(stri, sizeof(stri), "-%smat_inv_psubcomm_type %s",prefix,PetscSubcommTypes[inv->psubcommType]));
      CHKERRQ(PetscOptionsInsertString(NULL,stri));
      CHKERRQ(PCSetType(pc, PCREDUNDANT));
      CHKERRQ(PCRedundantSetNumber(pc, inv->redundancy));
      CHKERRQ(PCSetFromOptions(pc));
      CHKERRQ(PCRedundantGetKSP(pc, &inv->innerksp));
    }
  } else {
    inv->innerksp = inv->ksp;
  }
  CHKERRQ(KSPGetPC(inv->innerksp,&pc));

  CHKERRQ(KSPSetType(inv->innerksp,default_ksptype));
  CHKERRQ(PCSetType(pc,default_pctype));
  CHKERRQ(PCFactorSetMatSolverType(pc,default_pkg));

  CHKERRQ(KSPSetTolerances(inv->ksp, PETSC_SMALL, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  CHKERRQ(MatInvKSPSetOptionsPrefix_Inv(imat));
  if (inv->setfromoptionscalled) {
    CHKERRQ(KSPSetFromOptions(inv->innerksp));
  }

  CHKERRQ(MatDestroy(&Areg));
  inv->inner_objects_created = PETSC_TRUE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvExplicitly_Private"
static PetscErrorCode MatInvExplicitly_Private(KSP ksp, Mat imat_explicit)
{
  PetscInt m, M, i, ilo, ihi;
  PetscInt *rows;
  PetscScalar *v;
  Mat A;
  Vec col_I, col_imat;

  PetscFunctionBeginI;
  CHKERRQ(KSPGetOperators(ksp, &A, NULL));
  CHKERRQ(MatGetSize(     A, &M, NULL));
  CHKERRQ(MatGetLocalSize(A, &m, NULL));
  CHKERRQ(MatGetOwnershipRange(A, &ilo, &ihi));

  CHKERRQ(PetscMalloc(m*sizeof(PetscInt), &rows));
  for (i=0; i<m; i++) rows[i] = ilo + i;

  /* col_I is a j-th column of eye(M) */
  CHKERRQ(MatCreateVecs(A, &col_imat, &col_I));
  CHKERRQ(VecZeroEntries(col_I));
  for (i = 0; i < M; i++) {
    if (i >= ilo && i < ihi) CHKERRQ(VecSetValue(col_I, i, 1, INSERT_VALUES));
    if (i > ilo && i <= ihi) CHKERRQ(VecSetValue(col_I, i - 1, 0, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(col_I));
    CHKERRQ(VecAssemblyEnd(  col_I));

    CHKERRQ(KSPSolve(ksp, col_I, col_imat));
    CHKERRQ(VecGetArray(col_imat, &v));
    CHKERRQ(MatSetValues(imat_explicit, m, rows, 1, &i, v, INSERT_VALUES));
    CHKERRQ(VecRestoreArray(col_imat, &v));
  }
  CHKERRQ(VecDestroy(&col_imat));
  CHKERRQ(VecDestroy(&col_I));
  PetscFree(rows);
  PetscFunctionReturnI(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvExplicitlyTranspose_Private"
static PetscErrorCode MatInvExplicitlyTranspose_Private(PetscInt ilo, PetscInt ihi, KSP ksp, Mat imat_explicit)
{
  PetscInt i, Ailo, Aihi, localSize;
  PetscInt *idxn;
  PetscScalar *v;
  Mat A;
  Vec col_I, row_imat;

  PetscFunctionBeginI;
  CHKERRQ(KSPGetOperators(ksp, &A, NULL));
  CHKERRQ(MatGetOwnershipRange(A, &Ailo, &Aihi));
  localSize = Aihi-Ailo;
  
  CHKERRQ(PetscMalloc(localSize * sizeof(PetscInt), &idxn));
  for (i = 0; i < localSize; i++) idxn[i] = i+Ailo;

  /* col_I is a j-th column of eye(M) */
  CHKERRQ(MatCreateVecs(A, &col_I, &row_imat));
  CHKERRQ(VecZeroEntries(col_I));
  for (i = ilo; i < ihi; i++) {
    CHKERRQ(VecSetValue(col_I, i, 1, INSERT_VALUES));
    if (i > ilo) CHKERRQ(VecSetValue(col_I, i - 1, 0, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(col_I));
    CHKERRQ(VecAssemblyEnd(  col_I));
    CHKERRQ(KSPSolve(ksp, col_I, row_imat));
    CHKERRQ(VecGetArray(row_imat, &v));
    CHKERRQ(MatSetValues(imat_explicit, 1, &i, localSize, idxn, v, INSERT_VALUES));
    CHKERRQ(VecRestoreArray(row_imat, &v));
  }
  CHKERRQ(VecDestroy(&row_imat));
  CHKERRQ(VecDestroy(&col_I));
  PetscFree(idxn);
  PetscFunctionReturnI(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvExplicitly_Inv"
static PetscErrorCode MatInvExplicitly_Inv(Mat imat, PetscBool transpose, MatReuse scall, Mat *imat_explicit)
{
  PetscInt M;
  PetscInt m, n;
  MPI_Comm comm;
  PetscMPIInt rank, subrank, scColor=0;
  KSP ksp;
  PetscInt iloihi[2];
  PetscInt redundancy;
  Mat B;

  PetscFunctionBeginI;
  CHKERRQ(MatInvSetUp_Inv(imat));
  
  CHKERRQ(PetscLogEventBegin(Mat_Inv_Explicitly,imat,0,0,0));
  CHKERRQ(PetscObjectGetComm((PetscObject)imat, &comm));
  CHKERRQ(MatGetSize(     imat, &M, PETSC_IGNORE));
  CHKERRQ(MatGetLocalSize(imat, &m, &n));
  if (m != n) SETERRQ(comm, PETSC_ERR_ARG_SIZ, "only for locally square matrices, m != n, %d != %d", m, n);
  CHKERRQ(MatInvGetKSP(imat, &ksp));//innerksp
  CHKERRQ(MatInvGetRedundancy(imat, &redundancy));

  if (imat==*imat_explicit) SETERRQ(comm, PETSC_ERR_ARG_IDN, "Arguments #1 and #3 cannot be the same matrix.");

  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatCreateDensePermon(comm, m, m, M, M, NULL, &B));
    *imat_explicit = B;
  } else {
    B = *imat_explicit;
  }

  if (transpose) {
    if (redundancy){
      CHKERRQ(MPI_Comm_rank(comm, &rank));
      CHKERRQ(PetscObjectGetComm((PetscObject)ksp, &comm));
      CHKERRQ(MPI_Comm_rank(comm, &subrank));
      CHKERRQ(MatInvGetPsubcommColor_Inv(imat, &scColor));
      if (!subrank){
        iloihi[0] = scColor*floor(M/redundancy);
        if (rank == redundancy-1){
          iloihi[1] = M;
        }else{
          iloihi[1] = iloihi[0] + floor(M/redundancy);
        }
      }
      CHKERRQ(MPI_Bcast((PetscMPIInt*) &iloihi,2,MPIU_INT,0,comm));
    }else{
      iloihi[0] = 0;
      iloihi[1] = M;
    }
    CHKERRQ(MatInvExplicitlyTranspose_Private(iloihi[0], iloihi[1], ksp, B));
  } else {
    CHKERRQ(MatInvExplicitly_Private(         ksp, B));
  }
  CHKERRQ(PetscInfo(fllop,"calling MatAssemblyBegin\n"));
  CHKERRQ(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscInfo(fllop,"calling MatAssemblyEnd\n"));
  CHKERRQ(MatAssemblyEnd(  B, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscInfo(fllop,"MatAssemblyEnd done\n"));
  CHKERRQ(PetscLogEventEnd(Mat_Inv_Explicitly,imat,0,0,0));
  PetscFunctionReturnI(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_Inv"
PetscErrorCode MatMult_Inv(Mat imat, Vec right, Vec left)
{
  Mat_Inv *inv;

  PetscFunctionBegin;
  inv = (Mat_Inv*) imat->data;
  CHKERRQ(MatInvSetUp_Inv(imat));
  CHKERRQ(KSPSolve(inv->ksp, right, left));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_Inv"
PetscErrorCode MatGetInfo_Inv(Mat imat, MatInfoType type, MatInfo *info)
{
  Mat mat;
  PC pc;
  
  PetscFunctionBegin;
  info->assemblies            = -1.0;
  info->block_size            = -1.0;
  info->factor_mallocs        = -1.0;
  info->fill_ratio_given      = -1.0;
  info->fill_ratio_needed     = -1.0;
  info->mallocs               = -1.0;
  info->memory                = -1.0;
  info->nz_allocated          = -1.0;
  info->nz_unneeded           = -1.0;
  info->nz_used               = -1.0;

  CHKERRQ(MatInvGetRegularizedMat(imat, &mat));  
  CHKERRQ(MatInvGetPC(imat, &pc));
  
  if (pc && mat->factortype && pc->ops->getfactoredmatrix) {
    CHKERRQ(PCFactorGetMatrix(pc, &mat));
    if (mat && mat->ops->getinfo) {
      CHKERRQ(MatGetInfo(mat, type, info));
    }
  }

  if (info->nz_used == -1) {
    if (mat->ops->getinfo) CHKERRQ(MatGetInfo(mat, type, info));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Inv"
PetscErrorCode MatDestroy_Inv(Mat imat)
{
  Mat_Inv *inv;

  PetscFunctionBegin;
  inv = (Mat_Inv*) imat->data;
  CHKERRQ(MatDestroy(&inv->A));
  CHKERRQ(MatDestroy(&inv->R));
  CHKERRQ(KSPDestroy(&inv->ksp));
  CHKERRQ(PetscFree(inv));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetFromOptions_Inv"
PetscErrorCode MatSetFromOptions_Inv(PetscOptionItems *PetscOptionsObject,Mat imat)
{
  PetscBool set;
  PetscSubcommType psubcommType;
  Mat_Inv *inv;
  
  PetscFunctionBegin;
  inv = (Mat_Inv*) imat->data;
  PetscOptionsHead(PetscOptionsObject,"Mat Inv options");
  
  CHKERRQ(PetscOptionsEnum("-mat_inv_psubcomm_type", "subcommunicator type", "", PetscSubcommTypes, (PetscEnum) inv->psubcommType, (PetscEnum*)&psubcommType, &set));
  if (set) MatInvSetPsubcommType(imat, psubcommType);

  inv->setfromoptionscalled = PETSC_TRUE;

  PetscOptionsTail();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_Inv"
PetscErrorCode MatView_Inv(Mat imat, PetscViewer viewer)
{
  MPI_Comm                comm;
  PetscMPIInt             rank,size;
  Mat_Inv                 *inv;
  PetscViewerFormat       format;
  PetscBool               iascii;

  PetscFunctionBegin;
  inv = (Mat_Inv*) imat->data;
  CHKERRQ(PetscObjectGetComm((PetscObject)imat, &comm));
  CHKERRQ(MPI_Comm_rank(comm, &rank));
  CHKERRQ(MPI_Comm_size(comm, &size));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for matrix type %s",((PetscObject)viewer)->type,((PetscObject)imat)->type_name);
  CHKERRQ(PetscViewerGetFormat(viewer,&format));

  if (format == PETSC_VIEWER_DEFAULT) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)imat,viewer));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
  }

  if (format == PETSC_VIEWER_ASCII_INFO) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"redundancy:  %d\n", inv->redundancy));
    if (inv->redundancy) { /* inv->ksp is PCREDUNDANT */
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"subcomm type:  %s\n",PetscSubcommTypes[inv->psubcommType]));
    }
    CHKERRQ(KSPViewBriefInfo(inv->ksp, viewer));
    if (inv->innerksp != inv->ksp) {
      PetscBool show;
      MPI_Comm subcomm;
      PetscViewer sv;
      //
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      if (inv->redundancy) { /* inv->ksp is PCREDUNDANT */
        PC pc;
        PC_Redundant *red;
        //
        CHKERRQ(KSPGetPC(inv->ksp, &pc));
        red = (PC_Redundant*)pc->data;
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"Redundant preconditioner: First (color=0) of %D nested KSPs follows\n",red->nsubcomm));
        show = PetscNot(red->psubcomm->color);
      } else {                /* inv->ksp is PCBJACOBI */
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"Block Jacobi preconditioner: First (rank=0) of %D nested diagonal block KSPs follows\n",size));
        show = PetscNot(rank);
      }
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ(PetscObjectGetComm((PetscObject)inv->innerksp, &subcomm));
      CHKERRQ(PetscViewerGetSubViewer(viewer, subcomm, &sv));
      if (show) {
        CHKERRQ(KSPViewBriefInfo(inv->innerksp, sv));
      }
      CHKERRQ(PetscViewerRestoreSubViewer(viewer, subcomm, &sv));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
  } else {
    CHKERRQ(KSPView(inv->ksp, viewer));
  }
  
  if (format == PETSC_VIEWER_DEFAULT) {
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  CHKERRQ(MPI_Barrier(comm));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_Inv"
PetscErrorCode MatSetOption_Inv(Mat imat, MatOption op, PetscBool flg)
{
  Mat A;
  
  PetscFunctionBegin;
  CHKERRQ(MatInvGetRegularizedMat(imat, &A));
  CHKERRQ(MatSetOption(A, op, flg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_Inv"
PetscErrorCode MatAssemblyBegin_Inv(Mat imat, MatAssemblyType type)
{
  Mat A;
  
  PetscFunctionBegin;
  CHKERRQ(MatInvGetMat_Inv(imat, &A));
  CHKERRQ(MatAssemblyBegin(A, type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Inv"
PetscErrorCode MatAssemblyEnd_Inv(Mat imat, MatAssemblyType type)
{
  Mat A;
  
  PetscFunctionBegin;
  CHKERRQ(MatInvGetMat_Inv(imat, &A));
  CHKERRQ(MatAssemblyEnd(A, type));
  CHKERRQ(MatInvSetUp_Inv(imat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreate_Inv"
FLLOP_EXTERN PetscErrorCode MatCreate_Inv(Mat imat)
{
  Mat_Inv *inv;
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBegin;
  if (!registered) {
    CHKERRQ(PetscLogEventRegister("MatInvExplicitly", MAT_CLASSID, &Mat_Inv_Explicitly));
    CHKERRQ(PetscLogEventRegister("MatInvSetUp",      MAT_CLASSID, &Mat_Inv_SetUp));
    registered = PETSC_TRUE;
  }

  CHKERRQ(PetscNew(&inv));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)imat, MATINV));
  imat->data                          = (void*) inv;
  imat->assembled                     = PETSC_TRUE;
  imat->preallocated                  = PETSC_TRUE;

  /* Set standard operations of matrix. */
  imat->ops->destroy                  = MatDestroy_Inv;
  imat->ops->mult                     = MatMult_Inv;
  imat->ops->getinfo                  = MatGetInfo_Inv;
  imat->ops->setfromoptions           = MatSetFromOptions_Inv;
  imat->ops->view                     = MatView_Inv;
  imat->ops->setoption                = MatSetOption_Inv;
  imat->ops->assemblybegin            = MatAssemblyBegin_Inv;
  imat->ops->assemblyend              = MatAssemblyEnd_Inv;
  
  /* Set type-specific operations of matrix. */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvExplicitly_Inv_C",MatInvExplicitly_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvReset_Inv_C",MatInvReset_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetUp_Inv_C",MatInvSetUp_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetRegularizationType_Inv_C",MatInvGetRegularizationType_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetRegularizationType_Inv_C",MatInvSetRegularizationType_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetNullSpace_Inv_C",MatInvGetNullSpace_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetNullSpace_Inv_C",MatInvSetNullSpace_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvComputeNullSpace_Inv_C",MatInvComputeNullSpace_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetTolerances_Inv_C",MatInvSetTolerances_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetKSP_Inv_C",MatInvGetKSP_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetRegularizedMat_Inv_C",MatInvGetRegularizedMat_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetMat_Inv_C",MatInvGetMat_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetMat_Inv_C",MatInvSetMat_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetPC_Inv_C",MatInvGetPC_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetRedundancy_Inv_C",MatInvGetRedundancy_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetRedundancy_Inv_C",MatInvSetRedundancy_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetPsubcommType_Inv_C",MatInvGetPsubcommType_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetPsubcommType_Inv_C",MatInvSetPsubcommType_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvGetType_Inv_C",MatInvGetType_Inv));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)imat,"MatInvSetType_Inv_C",MatInvSetType_Inv));

  /* set default values of inner inv */
  inv->A                            = NULL;
  inv->R                            = NULL;
  inv->setfromoptionscalled         = PETSC_FALSE;
  inv->setupcalled                  = PETSC_FALSE;
  inv->inner_objects_created        = PETSC_FALSE;
  inv->redundancy                   = PETSC_DECIDE;
  inv->psubcommType                 = PETSC_SUBCOMM_CONTIGUOUS;
  inv->regtype                      = MAT_REG_NONE;
  inv->type                         = MAT_INV_MONOLITHIC;
  inv->innerksp                     = NULL;
  inv->ksp                          = NULL;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCreateInv"
PetscErrorCode MatCreateInv(Mat A, MatInvType invType, Mat *newimat)
{
  Mat imat;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(A,invType,2);
  PetscValidPointer(newimat,3);

  CHKERRQ(PetscObjectGetComm((PetscObject)A, &comm));
  CHKERRQ(MatCreate(comm, &imat));
  CHKERRQ(MatSetType(imat, MATINV));
  CHKERRQ(MatInvSetMat(imat, A));
  CHKERRQ(MatInvSetType(imat, invType));
  *newimat = imat;
  PetscFunctionReturn(0);
}

/* PetscBool transpose ... imat_explicit is tranposed - allows (imat_implicit is seq && imat_explicit is mpi) */
#undef __FUNCT__  
#define __FUNCT__ "MatInvExplicitly"
PetscErrorCode MatInvExplicitly(Mat imat, PetscBool transpose, MatReuse scall, Mat *imat_explicit)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(imat,transpose,2);
  PetscValidLogicalCollectiveEnum(imat,scall,3);
  PetscValidPointer(imat_explicit,4);
  PetscUseMethod(imat,"MatInvExplicitly_Inv_C",(Mat,PetscBool,MatReuse,Mat*),(imat,transpose,scall,imat_explicit));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvReset"
PetscErrorCode MatInvReset(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscTryMethod(imat,"MatInvReset_Inv_C",(Mat),(imat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetUp"
PetscErrorCode MatInvSetUp(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscTryMethod(imat,"MatInvSetUp_Inv_C",(Mat),(imat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvCreateInnerObjects"
PetscErrorCode MatInvCreateInnerObjects(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscTryMethod(imat,"MatInvCreateInnerObjects_Inv_C",(Mat),(imat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvSetRegularizationType"
PetscErrorCode MatInvSetRegularizationType(Mat imat,MatRegularizationType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(imat,type,2);
  PetscTryMethod(imat,"MatInvSetRegularizationType_Inv_C",(Mat,MatRegularizationType),(imat,type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvGetRegularizationType"
PetscErrorCode MatInvGetRegularizationType(Mat imat,MatRegularizationType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(type,2);
  PetscUseMethod(imat,"MatInvGetRegularizationType_Inv_C",(Mat,MatRegularizationType*),(imat,type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvComputeNullSpace"
PetscErrorCode MatInvComputeNullSpace(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscTryMethod(imat,"MatInvComputeNullSpace_Inv_C",(Mat),(imat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvSetNullSpace"
PetscErrorCode MatInvSetNullSpace(Mat imat,Mat R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  if (R) PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  PetscTryMethod(imat,"MatInvSetNullSpace_Inv_C",(Mat,Mat),(imat,R));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvGetNullSpace"
PetscErrorCode MatInvGetNullSpace(Mat imat,Mat *R)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(R,2);
  PetscUseMethod(imat,"MatInvGetNullSpace_Inv_C",(Mat,Mat*),(imat,R));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetTolerances"
PetscErrorCode MatInvSetTolerances(Mat imat, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(imat,rtol,2);
  PetscValidLogicalCollectiveReal(imat,abstol,3);
  PetscValidLogicalCollectiveReal(imat,dtol,4);
  PetscValidLogicalCollectiveInt(imat,maxits,5);
  PetscTryMethod(imat,"MatInvSetTolerances_Inv_C",(Mat,PetscReal,PetscReal,PetscReal,PetscInt),(imat,rtol,abstol,dtol,maxits));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetKSP"
PetscErrorCode MatInvGetKSP(Mat imat, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(ksp,2);
  PetscUseMethod(imat,"MatInvGetKSP_Inv_C",(Mat,KSP*),(imat,ksp));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetRegularizedMat"
PetscErrorCode MatInvGetRegularizedMat(Mat imat, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  PetscUseMethod(imat,"MatInvGetRegularizedMat_Inv_C",(Mat,Mat*),(imat,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetMat"
PetscErrorCode MatInvGetMat(Mat imat, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  PetscUseMethod(imat,"MatInvGetMat_Inv_C",(Mat,Mat*),(imat,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPC"
PetscErrorCode MatInvGetPC(Mat imat, PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(pc,2);
  PetscUseMethod(imat,"MatInvGetPC_Inv_C",(Mat,PC*),(imat,pc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetMat"
PetscErrorCode MatInvSetMat(Mat imat, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(imat,1,A,2);
  PetscTryMethod(imat,"MatInvSetMat_Inv_C",(Mat,Mat),(imat,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetRedundancy"
PetscErrorCode MatInvGetRedundancy(Mat imat, PetscInt *red)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidIntPointer(red,2);
  PetscUseMethod(imat,"MatInvGetRedundancy_Inv_C",(Mat,PetscInt*),(imat,red));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetRedundancy"
PetscErrorCode MatInvSetRedundancy(Mat imat, PetscInt red)
{
  MPI_Comm comm;
  PetscMPIInt comm_size;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(imat,red,2);
  CHKERRQ(PetscObjectGetComm((PetscObject)imat, &comm));
  CHKERRQ(MPI_Comm_size(comm, &comm_size));
  if ( (red < 0 && red != PETSC_DECIDE && red != PETSC_DEFAULT) || (red > comm_size) ) {
    SETERRQ(PetscObjectComm((PetscObject)imat), PETSC_ERR_ARG_WRONG, "invalid redundancy parameter: must be an intteger in closed interval [0, comm size]");
  }
  PetscTryMethod(imat,"MatInvSetRedundancy_Inv_C",(Mat,PetscInt),(imat,red));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPsubcommType"
PetscErrorCode MatInvGetPsubcommType(Mat imat, PetscSubcommType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidIntPointer(type,2);
  PetscUseMethod(imat,"MatInvGetPsubcommType_Inv_C",(Mat,PetscSubcommType*),(imat,type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetPsubcommType"
PetscErrorCode MatInvSetPsubcommType(Mat imat, PetscSubcommType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(imat,type,2);
  PetscTryMethod(imat,"MatInvSetPsubcommType_Inv_C",(Mat,PetscSubcommType),(imat,type));
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatInvGetType"
PetscErrorCode MatInvGetType(Mat imat, MatInvType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(type,2);
  PetscUseMethod(imat,"MatInvGetType_Inv_C",(Mat,MatInvType*),(imat,type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetType"
PetscErrorCode MatInvSetType(Mat imat, MatInvType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(imat,type,2);
  PetscTryMethod(imat,"MatInvSetType_Inv_C",(Mat,MatInvType),(imat,type));
  PetscFunctionReturn(0);
}
