
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
  TRY( MatGetOptionsPrefix(imat,&prefix) );
  TRY( KSPSetOptionsPrefix(inv->innerksp,prefix) );
  TRY( KSPAppendOptionsPrefix(inv->innerksp,"mat_inv_") );
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

//TODO MOVE THIS TO permonmatnullspace.c
//TODO Passing the PC is awkward. We will want to avoid that but it needs deeper refactoring.
#undef __FUNCT__
#define __FUNCT__ "MatComputeNullSpaceMat"
#if defined(PETSC_HAVE_MUMPS)
PetscErrorCode MatComputeNullSpaceMat(Mat K, PC pc, MatOrthType orthType, MatOrthForm orthForm, Mat *R_new)
{
  Mat Kl=NULL,R=NULL,Rl=NULL,F=NULL;
  PetscInt m,M,mm = 0,defect;
  PetscBool flg,blockdiag = PETSC_FALSE;
  MatSolverType type;
  Mat_MUMPS *mumps = NULL;
  PetscReal null_pivot_threshold = -1e-8;
  MPI_Comm blockComm;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(K,MAT_CLASSID,1);
  if (pc) PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  TRY( PetscObjectTypeCompare((PetscObject)K,MATBLOCKDIAG,&blockdiag) );
  if (blockdiag) {
    TRY( MatGetDiagonalBlock(K,&Kl) );
  } else {
    Kl = K;
  }
  TRY( PetscObjectGetComm((PetscObject)Kl,&blockComm) );
  TRY( MatGetLocalSize(Kl,&m,NULL) );
  TRY( MatGetSize(Kl,&M,NULL) );
  if (pc) {
    Mat Kl_pc;
    TRY( PCGetOperators(pc,&Kl_pc,NULL) );
    if (Kl != Kl_pc) SETERRQ(blockComm, PETSC_ERR_ARG_WRONG, "the passed preconditioner must contain the local block");
  }

  if (Kl->spd_set && Kl->spd) {
    defect = 0;
    mm = m;
  } else {
    /* MUMPS matrix type (sym) is set to 2 automatically (see MatGetFactor_aij_mumps). */
    flg = PETSC_FALSE;
    if (pc) {
      TRY( PCFactorGetMatSolverType(pc,&type) );
      TRY( PetscStrcmp(type,MATSOLVERMUMPS,&flg) );
      if (flg) TRY( PetscObjectTypeCompare((PetscObject)pc,PCCHOLESKY,&flg) );
    }
    /* TODO We need to call PCSetUP() (which does the factorization) before being able to call PCFactorGetMatrix().
       Maybe we could do something on the PETSc side to overcome this. */
    if (flg) {
      /* If MUMPS Cholesky is used, avoid doubled factorization. */
      char opts[128];
      TRY( PetscSNPrintf(opts,sizeof(opts),"-%smat_mumps_icntl_24 1 -%smat_mumps_cntl_3 %e",((PetscObject)pc)->prefix,((PetscObject)pc)->prefix,null_pivot_threshold) );
      TRY( PetscOptionsInsertString(NULL,opts) );
      TRY( PCSetFromOptions(pc) );
      TRY( PCSetUp(pc) );
      TRY( PCFactorGetMatrix(pc,&F) );
      mumps =(Mat_MUMPS*)F->data;
      TRY( PetscObjectReference((PetscObject)F) );
    } else {
      if (pc) TRY( PetscPrintf(PetscObjectComm((PetscObject)K), "WARNING: Performing extra factorization with MUMPS Cholesky just for nullspace detection. Avoid this by setting MUMPS Cholesky as MATINV solver.\n") );
      TRY( MatGetFactor(Kl,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F) );
      TRY( MatMumpsSetIcntl(F,24,1) ); /* null pivot detection */
      TRY( MatMumpsSetCntl(F,3,null_pivot_threshold) ); /* null pivot threshold */
      mumps =(Mat_MUMPS*)F->data;
      TRY( MatCholeskyFactorSymbolic(F,Kl,NULL,NULL) );
      TRY( MatCholeskyFactorNumeric(F,Kl,NULL) );
    }
    TRY( MatMumpsGetInfog(F,28,&defect) ); /* get numerical defect, i.e. number of null pivots encountered during factorization */
    /* mumps->petsc_size > 1 implies mumps->id.ICNTL(21) = 1 (distributed solution ) */
    mm = (defect && mumps->petsc_size > 1) ? mumps->id.lsol_loc : m;  /* = length of sol_loc = INFO(23) */
  }

  TRY( MatCreateDensePermon(blockComm,mm,PETSC_DECIDE,M,defect,NULL,&Rl) );

  if (defect) {
    /* stash sol_loc allocated in MatFactorNumeric_MUMPS() */
    MumpsScalar *sol_loc_orig = mumps->id.sol_loc;
    MumpsScalar *array;

    /* inject matrix array as sol_loc */
    TRY( MatDenseGetArray(Rl,(MumpsScalar**)&array) );
    if (mumps->petsc_size > 1) {
      mumps->id.sol_loc = array;
      if (!mumps->myid) {
        /* Define dummy rhs on the host otherwise MUMPS fails with INFOG(1)=-22,INFOG(2)=7 */
        TRY( PetscMalloc1(M,&mumps->id.rhs) );
      }
    } else mumps->id.rhs = array;
    /* mumps->id.nrhs is reset by MatMatSolve_MUMPS()/MatSolve_MUMPS() */
    mumps->id.nrhs = defect;
    TRY( MatMumpsSetIcntl(F,25,-1) ); /* compute complete null space */
    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(&mumps->id);
    if (mumps->id.INFOG(1) < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d,INFOG(2)=%d\n",mumps->id.INFOG(1),mumps->id.INFOG(2));

    if (mumps->petsc_size > 1 && !mumps->myid) {
      TRY( PetscFree(mumps->id.rhs) );
    }
    TRY( MatMumpsSetIcntl(F,25,0) ); /* perform a normal solution step next time */

    /* restore matrix array */
    TRY( MatDenseRestoreArray(Rl,(MumpsScalar**)&array) );
    /* restore stashed sol_loc */
    mumps->id.sol_loc = sol_loc_orig;
  }

  //TODO return just NULL if defect=0 ?
  if (blockdiag) {
    TRY( MatCreateBlockDiag(PETSC_COMM_WORLD,Rl,&R) );
    TRY( FllopPetscObjectInheritName((PetscObject)Rl,(PetscObject)R,"_loc") );
    TRY( MatSetNullSpaceMat(Kl,Rl) );
    TRY( MatDestroy(&Rl) );
  } else if (defect && mumps->petsc_size > 1) {
    IS isol_is;
    /* redistribute to get conforming local size */
    TRY( MatCreateDensePermon(blockComm,m,PETSC_DECIDE,M,defect,NULL,&R) );
    TRY( ISCreateGeneral(PETSC_COMM_SELF,mm,mumps->id.isol_loc,PETSC_USE_POINTER,&isol_is) );
    TRY( MatRedistributeRows(Rl,isol_is,1,R) ); /* MUMPS uses 1-based numbering */
    TRY( MatDestroy(&Rl) );
    TRY( ISDestroy(&isol_is) );
  } else {
    R = Rl;
  }
  TRY( MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY) );
  TRY( MatOrthColumns(R, orthType, orthForm, R_new, NULL) );
  TRY( PetscObjectSetName((PetscObject)*R_new,"R") );
  TRY( MatDestroy(&R) );
  TRY( MatDestroy(&F) );
  PetscFunctionReturnI(0);
}
#else
PetscErrorCode MatComputeNullSpaceMat(Mat K, PC pc, MatOrthType orthType, MatOrthForm orthForm, Mat *R_new)
{
  PetscFunctionBeginI;
  FLLOP_SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"MUMPS library is currently needed for nullspace computation");
  PetscFunctionReturnI(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetTolerances_Inv"
static PetscErrorCode MatInvSetTolerances_Inv(Mat imat, PetscReal rtol, PetscReal abstol,
    PetscReal dtol, PetscInt maxits)
{
  KSP ksp;

  PetscFunctionBegin;
  TRY( MatInvGetKSP(imat,&ksp) );
  TRY( KSPSetTolerances(ksp, rtol, abstol, dtol, maxits) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetKSP_Inv"
static PetscErrorCode MatInvGetKSP_Inv(Mat imat, KSP *ksp)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;

  PetscFunctionBegin;
  if (!inv->ksp) {
    TRY( MatInvCreateInnerObjects_Inv(imat) );
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
  if (!inv->setupcalled) FLLOP_SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_ARG_WRONGSTATE,"This function can be called only after MatInvSetUp");
  TRY( MatInvGetKSP(imat,&ksp) );
  TRY( KSPGetOperators(ksp, A, NULL) );
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
  TRY( MatInvGetKSP(imat, &ksp) );
  TRY( KSPGetPC(ksp, pc) );
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

  TRY( MatInvReset(imat) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );
  TRY( MatSetSizes(imat, m, n, M, N) );
  inv->A = A;
  TRY( PetscObjectReference((PetscObject)A) );
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
  TRY( KSPGetPC(inv->ksp, &pc) );
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
  TRY( KSPReset(inv->ksp) );
  inv->setupcalled = PETSC_FALSE;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetUp_Inv"
static PetscErrorCode MatInvSetUp_Inv(Mat imat)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  Mat     R;

  FllopTracedFunctionBegin;
  TRY( MatGetNullSpaceMat(inv->A, &R) );
  if (inv->setupcalled) PetscFunctionReturn(0);
  if (inv->type == MAT_INV_BLOCKDIAG && inv->redundancy > 0) FLLOP_SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_SUP, "Cannot use MAT_INV_BLOCKDIAG and redundancy at the same time");
  if (inv->regtype && !R) FLLOP_SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_ARG_WRONGSTATE,"regularization is requested but nullspace is not set");

  FllopTraceBegin;
  TRY( PetscLogEventBegin(Mat_Inv_SetUp,imat,0,0,0) );
  {
    TRY( MatInvCreateInnerObjects_Inv(imat) );
    TRY( KSPSetUp(inv->ksp) );
    TRY( KSPSetUpOnBlocks(inv->ksp) );
  }

  inv->setupcalled = PETSC_TRUE;
  TRY( MatInheritSymmetry(inv->A,imat) );
  TRY( PetscLogEventEnd(Mat_Inv_SetUp,imat,0,0,0) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvCreateInnerObjects_Inv"
static PetscErrorCode MatInvCreateInnerObjects_Inv(Mat imat)
{
  Mat_Inv *inv = (Mat_Inv*) imat->data;
  Mat R;
  Mat Areg,A_inner;
  PC pc;
  PetscBool factorizable,parallel,flg,own;
  KSPType default_ksptype;
  PCType  default_pctype;
  MatSolverType default_pkg;
  PetscMPIInt size;

  FllopTracedFunctionBegin;
  TRY( MatGetNullSpaceMat(inv->A, &R) );
  if (inv->inner_objects_created) PetscFunctionReturn(0);
  if (inv->type == MAT_INV_BLOCKDIAG && inv->redundancy > 0) FLLOP_SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_SUP, "Cannot use MAT_INV_BLOCKDIAG and redundancy at the same time");
  if (inv->regtype && !R) FLLOP_SETERRQ(PetscObjectComm((PetscObject)imat),PETSC_ERR_ARG_WRONGSTATE,"regularization is requested but nullspace is not set");
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)imat),&size) );

  FllopTraceBegin;
  if (!inv->ksp) {
    TRY( KSPCreate(PetscObjectComm((PetscObject)imat),&inv->ksp) );
    TRY( PetscObjectIncrementTabLevel((PetscObject)inv->ksp,(PetscObject)imat,1) );
    TRY( PetscLogObjectParent((PetscObject)imat,(PetscObject)inv->ksp) );
  }

  own = ((PetscObject)inv->A)->refct == 1 ? PETSC_TRUE : PETSC_FALSE;
  //TODO MatRegularize can take R directly from A
  TRY( MatRegularize(inv->A,R,inv->regtype,&Areg) );
  TRY( PetscOptionsHasName(NULL,((PetscObject)imat)->prefix,"-mat_inv_mat_type",&flg) );
  if (inv->setfromoptionscalled && flg && inv->A == Areg && !own) {
    TRY( PetscInfo(fllop,"duplicating inner matrix to allow to apply options only internally\n") );
    TRY( PetscObjectDereference((PetscObject)Areg) );
    TRY( MatDuplicate(Areg, MAT_COPY_VALUES, &Areg) );
  }
  TRY( KSPSetOperators(inv->ksp, Areg, Areg) );

  if (inv->type == MAT_INV_BLOCKDIAG) {
    TRY( MatGetDiagonalBlock(Areg,&A_inner) );
  } else {
    A_inner = Areg;
  }

  TRY( FllopPetscObjectInheritPrefixIfNotSet((PetscObject)Areg,(PetscObject)imat,"mat_inv_") );
  TRY( FllopPetscObjectInheritPrefixIfNotSet((PetscObject)A_inner,(PetscObject)imat,"mat_inv_") );
  if (inv->setfromoptionscalled) {
    TRY( PetscInfo1(fllop,"setting inner matrix with prefix %s from options\n",((PetscObject)A_inner)->prefix) );
    TRY( PermonMatSetFromOptions(A_inner) );
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
    TRY( MatGetFactorAvailable(A_inner,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&flg) );
    if (flg) {
      default_pkg       = MATSOLVERMUMPS;
      default_pctype    = PCCHOLESKY;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_PASTIX)
    TRY( MatGetFactorAvailable(A_inner,MATSOLVERPASTIX,MAT_FACTOR_CHOLESKY,&flg) );
    if (flg) {
      default_pkg       = MATSOLVERPASTIX;
      default_pctype    = PCCHOLESKY;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
    TRY( MatGetFactorAvailable(A_inner,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&flg) );
    if (flg) {
      default_pkg       = MATSOLVERSUPERLU_DIST;
      default_pctype    = PCLU;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_MUMPS)
    TRY( MatGetFactorAvailable(A_inner,MATSOLVERMUMPS,MAT_FACTOR_LU,&flg) );
    if (flg) {
      default_pkg       = MATSOLVERMUMPS;
      default_pctype    = PCLU;
      parallel          = PETSC_TRUE;
      goto chosen;
    }
#endif
#if defined(PETSC_HAVE_PASTIX)
    TRY( MatGetFactorAvailable(A_inner,MATSOLVERPASTIX,MAT_FACTOR_LU,&flg) );
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

  if (inv->type == MAT_INV_BLOCKDIAG) {
    KSP *kspp;
    TRY( KSPSetType(inv->ksp, KSPPREONLY) );
    TRY( KSPGetPC(inv->ksp, &pc) );
    TRY( PCSetType(pc, PCBJACOBI) );
    TRY( PCSetUp(pc) );
    TRY( PCBJacobiGetSubKSP(pc, PETSC_IGNORE, PETSC_IGNORE, &kspp) );
    inv->innerksp = *kspp;
  } else if (inv->redundancy) {
    const char *prefix;
    char stri[1024];
    TRY( KSPSetType(inv->ksp, KSPPREONLY) );
    TRY( KSPGetPC(inv->ksp, &pc) );
    TRY( MatGetOptionsPrefix(imat,&prefix) );
    TRY( PCSetOptionsPrefix(pc,prefix) );
    TRY( PCAppendOptionsPrefix(pc,"mat_inv_") );
    TRY( PetscSNPrintf(stri, sizeof(stri), "-%smat_inv_redundant_pc_type none",prefix) );
    TRY( PetscOptionsInsertString(NULL,stri) );
    //petsc bug start https://bitbucket.org/petsc/petsc/branch/hzhang/fix-pcredundant#diff
    TRY( PetscSNPrintf(stri, sizeof(stri), "-psubcomm_type %s", PetscSubcommTypes[inv->psubcommType]) );
    TRY( PetscOptionsInsertString(NULL,stri) );
    TRY( PCSetFromOptions(pc) );
    //bug end
    TRY( PCSetType(pc, PCREDUNDANT) );
    TRY( PCRedundantSetNumber(pc, inv->redundancy) );
    TRY( PCSetUp(pc) );//not necessary after fix, see above
    TRY( PCRedundantGetKSP(pc, &inv->innerksp) );
  } else {
    inv->innerksp = inv->ksp;
  }
  TRY( KSPGetPC(inv->innerksp,&pc) );

  TRY( KSPSetType(inv->innerksp,default_ksptype) );
  TRY( PCSetType(pc,default_pctype) );
  TRY( PCFactorSetMatSolverType(pc,default_pkg) );

  TRY( KSPSetTolerances(inv->ksp, PETSC_SMALL, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT) );

  TRY( MatInvKSPSetOptionsPrefix_Inv(imat) );
  if (inv->setfromoptionscalled) {
    TRY( KSPSetFromOptions(inv->innerksp) );
  }

  TRY( MatDestroy(&Areg) );
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
  TRY( KSPGetOperators(ksp, &A, NULL) );
  TRY( MatGetSize(     A, &M, NULL) );
  TRY( MatGetLocalSize(A, &m, NULL) );
  TRY( MatGetOwnershipRange(A, &ilo, &ihi) );

  TRY( PetscMalloc(m*sizeof(PetscInt), &rows) );
  for (i=0; i<m; i++) rows[i] = ilo + i;

  /* col_I is a j-th column of eye(M) */
  TRY( MatCreateVecs(A, &col_imat, &col_I) );
  TRY( VecZeroEntries(col_I) );
  for (i = 0; i < M; i++) {
    if (i >= ilo && i < ihi) TRY( VecSetValue(col_I, i, 1, INSERT_VALUES) );
    if (i > ilo && i <= ihi) TRY( VecSetValue(col_I, i - 1, 0, INSERT_VALUES) );
    TRY( VecAssemblyBegin(col_I) );
    TRY( VecAssemblyEnd(  col_I) );

    TRY( KSPSolve(ksp, col_I, col_imat) );
    TRY( VecGetArray(col_imat, &v));
    TRY( MatSetValues(imat_explicit, m, rows, 1, &i, v, INSERT_VALUES) );
    TRY( VecRestoreArray(col_imat, &v));
  }
  TRY( VecDestroy(&col_imat) );
  TRY( VecDestroy(&col_I) );
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
  TRY( KSPGetOperators(ksp, &A, NULL) );
  TRY( MatGetOwnershipRange(A, &Ailo, &Aihi) );
  localSize = Aihi-Ailo;
  
  TRY( PetscMalloc(localSize * sizeof(PetscInt), &idxn) );
  for (i = 0; i < localSize; i++) idxn[i] = i+Ailo;

  /* col_I is a j-th column of eye(M) */
  TRY( MatCreateVecs(A, &col_I, &row_imat) );
  TRY( VecZeroEntries(col_I) );
  for (i = ilo; i < ihi; i++) {
    TRY( VecSetValue(col_I, i, 1, INSERT_VALUES) );
    if (i > ilo) TRY( VecSetValue(col_I, i - 1, 0, INSERT_VALUES) );
    TRY( VecAssemblyBegin(col_I) );
    TRY( VecAssemblyEnd(  col_I) );
    TRY( KSPSolve(ksp, col_I, row_imat) );
    TRY( VecGetArray(row_imat, &v));
    TRY( MatSetValues(imat_explicit, 1, &i, localSize, idxn, v, INSERT_VALUES) );
    TRY( VecRestoreArray(row_imat, &v));
  }
  TRY( VecDestroy(&row_imat) );
  TRY( VecDestroy(&col_I) );
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
  TRY( MatInvSetUp_Inv(imat) );
  
  TRY( PetscLogEventBegin(Mat_Inv_Explicitly,imat,0,0,0) );
  TRY( PetscObjectGetComm((PetscObject)imat, &comm) );
  TRY( MatGetSize(     imat, &M, PETSC_IGNORE) );
  TRY( MatGetLocalSize(imat, &m, &n) );
  if (m != n) FLLOP_SETERRQ2(comm, PETSC_ERR_ARG_SIZ, "only for locally square matrices, m != n, %d != %d", m, n);
  TRY( MatInvGetKSP(imat, &ksp) );//innerksp
  TRY( MatInvGetRedundancy(imat, &redundancy) );

  if (imat==*imat_explicit) FLLOP_SETERRQ(comm, PETSC_ERR_ARG_IDN, "Arguments #1 and #3 cannot be the same matrix.");

  if (scall == MAT_INITIAL_MATRIX) {
    TRY( MatCreateDensePermon(comm, m, m, M, M, NULL, &B) );
    *imat_explicit = B;
  } else {
    B = *imat_explicit;
  }

  if (transpose) {
    if (redundancy){
      TRY( MPI_Comm_rank(comm, &rank) );
      TRY( PetscObjectGetComm((PetscObject)ksp, &comm) );
      TRY( MPI_Comm_rank(comm, &subrank) );
      TRY( MatInvGetPsubcommColor_Inv(imat, &scColor) );
      if (!subrank){
        iloihi[0] = scColor*floor(M/redundancy);
        if (rank == redundancy-1){
          iloihi[1] = M;
        }else{
          iloihi[1] = iloihi[0] + floor(M/redundancy);
        }
      }
      TRY( MPI_Bcast((PetscMPIInt*) &iloihi,2,MPIU_INT,0,comm) );
    }else{
      iloihi[0] = 0;
      iloihi[1] = M;
    }
    TRY( MatInvExplicitlyTranspose_Private(iloihi[0], iloihi[1], ksp, B) );
  } else {
    TRY( MatInvExplicitly_Private(         ksp, B) );
  }
  TRY( PetscInfo(fllop,"calling MatAssemblyBegin\n") );
  TRY( MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY) );
  TRY( PetscInfo(fllop,"calling MatAssemblyEnd\n") );
  TRY( MatAssemblyEnd(  B, MAT_FINAL_ASSEMBLY) );
  TRY( PetscInfo(fllop,"MatAssemblyEnd done\n") );
  TRY( PetscLogEventEnd(Mat_Inv_Explicitly,imat,0,0,0) );
  PetscFunctionReturnI(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_Inv"
PetscErrorCode MatMult_Inv(Mat imat, Vec right, Vec left)
{
  Mat_Inv *inv;

  PetscFunctionBegin;
  inv = (Mat_Inv*) imat->data;
  TRY( MatInvSetUp_Inv(imat) );
  TRY( KSPSolve(inv->ksp, right, left) );
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

  TRY( MatInvGetRegularizedMat(imat, &mat) );  
  TRY( MatInvGetPC(imat, &pc) );
  
  if (pc && mat->factortype && pc->ops->getfactoredmatrix) {
    TRY( PCFactorGetMatrix(pc, &mat) );
    if (mat && mat->ops->getinfo) {
      TRY( MatGetInfo(mat, type, info) );
    }
  }

  if (info->nz_used == -1) {
    if (mat->ops->getinfo) TRY( MatGetInfo(mat, type, info) );
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
  TRY( MatDestroy(&inv->A) );
  TRY( KSPDestroy(&inv->ksp) );
  TRY( PetscFree(inv) );
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
  TRY( PetscOptionsHead(PetscOptionsObject,"Mat Inv options") );
  
  TRY( PetscOptionsEnum("-mat_inv_psubcomm_type", "subcommunicator type", "", PetscSubcommTypes, (PetscEnum) inv->psubcommType, (PetscEnum*)&psubcommType, &set) );
  if (set) MatInvSetPsubcommType(imat, psubcommType);

  inv->setfromoptionscalled = PETSC_TRUE;

  TRY( PetscOptionsTail() );
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
  TRY( PetscObjectGetComm((PetscObject)imat, &comm) );
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( MPI_Comm_size(comm, &size) );
  TRY( PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii) );
  if (!iascii) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for matrix type "MATINV, ((PetscObject)viewer)->type);
  TRY( PetscViewerGetFormat(viewer,&format) );

  if (format == PETSC_VIEWER_DEFAULT) {
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject)imat,viewer) );
    TRY( PetscViewerASCIIPushTab(viewer) );
  }

  if (format == PETSC_VIEWER_ASCII_INFO) {
    TRY( PetscViewerASCIIPrintf(viewer,"redundancy:  %d\n", inv->redundancy) );
    if (inv->redundancy) { /* inv->ksp is PCREDUNDANT */
      TRY( PetscViewerASCIIPrintf(viewer,"subcomm type:  %s\n",PetscSubcommTypes[inv->psubcommType]) );
    }
    TRY( KSPViewBriefInfo(inv->ksp, viewer) );
    if (inv->innerksp != inv->ksp) {
      PetscBool show;
      MPI_Comm subcomm;
      PetscViewer sv;
      //
      TRY( PetscViewerASCIIPushTab(viewer) );
      if (inv->redundancy) { /* inv->ksp is PCREDUNDANT */
        PC pc;
        PC_Redundant *red;
        //
        TRY( KSPGetPC(inv->ksp, &pc) );
        red = (PC_Redundant*)pc->data;
        TRY( PetscViewerASCIIPrintf(viewer,"Redundant preconditioner: First (color=0) of %D nested KSPs follows\n",red->nsubcomm) );
        show = !red->psubcomm->color;
      } else {                /* inv->ksp is PCBJACOBI */
        TRY( PetscViewerASCIIPrintf(viewer,"Block Jacobi preconditioner: First (rank=0) of %D nested diagonal block KSPs follows\n",size) );
        show = !rank;
      }
      TRY( PetscViewerASCIIPushTab(viewer) );
      TRY( PetscObjectGetComm((PetscObject)inv->innerksp, &subcomm) );
      TRY( PetscViewerGetSubViewer(viewer, subcomm, &sv) );
      if (show) {
        TRY( KSPViewBriefInfo(inv->innerksp, sv) );
      }
      TRY( PetscViewerRestoreSubViewer(viewer, subcomm, &sv) );
      TRY( PetscViewerASCIIPopTab(viewer) );
      TRY( PetscViewerASCIIPopTab(viewer) );
    }
  } else {
    TRY( KSPView(inv->ksp, viewer) );
  }
  
  if (format == PETSC_VIEWER_DEFAULT) {
    TRY( PetscViewerASCIIPopTab(viewer) );
  }
  TRY( MPI_Barrier(comm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_Inv"
PetscErrorCode MatSetOption_Inv(Mat imat, MatOption op, PetscBool flg)
{
  Mat A;
  
  PetscFunctionBegin;
  TRY( MatInvGetRegularizedMat(imat, &A) );
  TRY( MatSetOption(A, op, flg) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_Inv"
PetscErrorCode MatAssemblyBegin_Inv(Mat imat, MatAssemblyType type)
{
  Mat A;
  
  PetscFunctionBegin;
  TRY( MatInvGetMat_Inv(imat, &A) );
  TRY( MatAssemblyBegin(A, type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Inv"
PetscErrorCode MatAssemblyEnd_Inv(Mat imat, MatAssemblyType type)
{
  Mat A;
  
  PetscFunctionBegin;
  TRY( MatInvGetMat_Inv(imat, &A) );
  TRY( MatAssemblyEnd(A, type) );
  TRY( MatInvSetUp_Inv(imat) );
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
    TRY( PetscLogEventRegister("MatInvExplicitly", MAT_CLASSID, &Mat_Inv_Explicitly) );
    TRY( PetscLogEventRegister("MatInvSetUp",      MAT_CLASSID, &Mat_Inv_SetUp) );
    registered = PETSC_TRUE;
  }

  TRY( PetscNew(&inv) );
  TRY( PetscObjectChangeTypeName((PetscObject)imat, MATINV) );
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
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvExplicitly_Inv_C",MatInvExplicitly_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvReset_Inv_C",MatInvReset_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetUp_Inv_C",MatInvSetUp_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetRegularizationType_Inv_C",MatInvGetRegularizationType_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetRegularizationType_Inv_C",MatInvSetRegularizationType_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetTolerances_Inv_C",MatInvSetTolerances_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetKSP_Inv_C",MatInvGetKSP_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetRegularizedMat_Inv_C",MatInvGetRegularizedMat_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetMat_Inv_C",MatInvGetMat_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetMat_Inv_C",MatInvSetMat_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetPC_Inv_C",MatInvGetPC_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetRedundancy_Inv_C",MatInvGetRedundancy_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetRedundancy_Inv_C",MatInvSetRedundancy_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetPsubcommType_Inv_C",MatInvGetPsubcommType_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetPsubcommType_Inv_C",MatInvSetPsubcommType_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvGetType_Inv_C",MatInvGetType_Inv) );
  TRY( PetscObjectComposeFunction((PetscObject)imat,"MatInvSetType_Inv_C",MatInvSetType_Inv) );

  /* set default values of inner inv */
  inv->A                            = NULL;
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

  TRY( PetscObjectGetComm((PetscObject)A, &comm) );
  TRY( MatCreate(comm, &imat) );
  TRY( MatSetType(imat, MATINV) );
  TRY( MatInvSetMat(imat, A) );
  TRY( MatInvSetType(imat, invType) );
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
  TRY( PetscUseMethod(imat,"MatInvExplicitly_Inv_C",(Mat,PetscBool,MatReuse,Mat*),(imat,transpose,scall,imat_explicit)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvReset"
PetscErrorCode MatInvReset(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  TRY( PetscTryMethod(imat,"MatInvReset_Inv_C",(Mat),(imat)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetUp"
PetscErrorCode MatInvSetUp(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  TRY( PetscTryMethod(imat,"MatInvSetUp_Inv_C",(Mat),(imat)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvCreateInnerObjects"
PetscErrorCode MatInvCreateInnerObjects(Mat imat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  TRY( PetscTryMethod(imat,"MatInvCreateInnerObjects_Inv_C",(Mat),(imat)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvSetRegularizationType"
PetscErrorCode MatInvSetRegularizationType(Mat imat,MatRegularizationType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(imat,type,2);
  TRY( PetscTryMethod(imat,"MatInvSetRegularizationType_Inv_C",(Mat,MatRegularizationType),(imat,type)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvGetRegularizationType"
PetscErrorCode MatInvGetRegularizationType(Mat imat,MatRegularizationType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(type,2);
  TRY( PetscUseMethod(imat,"MatInvGetRegularizationType_Inv_C",(Mat,MatRegularizationType*),(imat,type)) );
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
  TRY( PetscTryMethod(imat,"MatInvSetTolerances_Inv_C",(Mat,PetscReal,PetscReal,PetscReal,PetscInt),(imat,rtol,abstol,dtol,maxits)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetKSP"
PetscErrorCode MatInvGetKSP(Mat imat, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(ksp,2);
  TRY( PetscUseMethod(imat,"MatInvGetKSP_Inv_C",(Mat,KSP*),(imat,ksp)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetRegularizedMat"
PetscErrorCode MatInvGetRegularizedMat(Mat imat, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  TRY( PetscUseMethod(imat,"MatInvGetRegularizedMat_Inv_C",(Mat,Mat*),(imat,A)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetMat"
PetscErrorCode MatInvGetMat(Mat imat, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  TRY( PetscUseMethod(imat,"MatInvGetMat_Inv_C",(Mat,Mat*),(imat,A)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPC"
PetscErrorCode MatInvGetPC(Mat imat, PC *pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(pc,2);
  TRY( PetscUseMethod(imat,"MatInvGetPC_Inv_C",(Mat,PC*),(imat,pc)) );
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
  TRY( PetscTryMethod(imat,"MatInvSetMat_Inv_C",(Mat,Mat),(imat,A)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetRedundancy"
PetscErrorCode MatInvGetRedundancy(Mat imat, PetscInt *red)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidIntPointer(red,2);
  TRY( PetscUseMethod(imat,"MatInvGetRedundancy_Inv_C",(Mat,PetscInt*),(imat,red)) );
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
  TRY( PetscObjectGetComm((PetscObject)imat, &comm) );
  TRY( MPI_Comm_size(comm, &comm_size) );
  if ( (red < 0 && red != PETSC_DECIDE && red != PETSC_DEFAULT) || (red > comm_size) ) {
    FLLOP_SETERRQ(PetscObjectComm((PetscObject)imat), PETSC_ERR_ARG_WRONG, "invalid redundancy parameter: must be an intteger in closed interval [0, comm size]");
  }
  TRY( PetscTryMethod(imat,"MatInvSetRedundancy_Inv_C",(Mat,PetscInt),(imat,red)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvGetPsubcommType"
PetscErrorCode MatInvGetPsubcommType(Mat imat, PetscSubcommType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidIntPointer(type,2);
  TRY( PetscUseMethod(imat,"MatInvGetPsubcommType_Inv_C",(Mat,PetscSubcommType*),(imat,type)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetPsubcommType"
PetscErrorCode MatInvSetPsubcommType(Mat imat, PetscSubcommType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(imat,type,2);
  TRY( PetscTryMethod(imat,"MatInvSetPsubcommType_Inv_C",(Mat,PetscSubcommType),(imat,type)) );
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatInvGetType"
PetscErrorCode MatInvGetType(Mat imat, MatInvType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidPointer(type,2);
  TRY( PetscUseMethod(imat,"MatInvGetType_Inv_C",(Mat,MatInvType*),(imat,type)) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatInvSetType"
PetscErrorCode MatInvSetType(Mat imat, MatInvType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(imat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(imat,type,2);
  TRY( PetscTryMethod(imat,"MatInvSetType_Inv_C",(Mat,MatInvType),(imat,type)) );
  PetscFunctionReturn(0);
}
