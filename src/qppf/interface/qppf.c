
#include <permon/private/qppfimpl.h>
#include <permonksp.h>
PetscClassId QPPF_CLASSID;
PetscLogEvent QPPF_SetUp, QPPF_SetUp_Gt, QPPF_SetUp_GGt, QPPF_SetUp_GGtinv;
PetscLogEvent QPPF_ApplyCP, QPPF_ApplyCP_gt, QPPF_ApplyCP_sc;
PetscLogEvent QPPF_ApplyP, QPPF_ApplyQ, QPPF_ApplyHalfQ, QPPF_ApplyG, QPPF_ApplyGt;

const char *QPPFVariants[] = {"zero","all","dist", "QPPFVariant","QPPF_",0};

#define RANK0 0
#define G_RELATIVE_FILL 1.0


static PetscErrorCode QPPFMatMult_P(Mat matP, Vec v, Vec Pv)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(matP, (QPPF*)&cp));
  PetscCall(QPPFApplyP(cp, v, Pv));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPPFMatMult_Q(Mat matQ, Vec v, Vec Qv)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(matQ, (QPPF*)&cp));
  PetscCall(QPPFApplyQ(cp, v, Qv));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPPFMatMult_HalfQ(Mat matHalfQ, Vec x, Vec y)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(matHalfQ, (QPPF*)&cp));
  PetscCall(QPPFApplyHalfQ(cp,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPPFMatMultTranspose_HalfQ(Mat matHalfQ, Vec x, Vec y)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(matHalfQ, (QPPF*)&cp));
  PetscCall(QPPFApplyHalfQTranspose(cp,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode QPPFMatMult_GtG(Mat matGtG, Vec v, Vec GtGv)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(matGtG, (QPPF*)&cp));
  PetscCall(QPPFApplyGtG(cp, v, GtGv));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFCreate(MPI_Comm comm, QPPF* qppf_new)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscValidPointer(qppf_new, 2);
  *qppf_new = 0;
  PetscCall(QPPFInitializePackage());

  PetscCall(PetscHeaderCreate(cp,QPPF_CLASSID,"QPPF", "Projector Factory", "QPPF", comm, QPPFDestroy, QPPFView));

  cp->G                   = NULL;
  cp->Gt                  = NULL;
  cp->GGtinv              = NULL;
  cp->G_left              = NULL;
  cp->Gt_right            = NULL;
  cp->QPPFApplyQ_last_v   = NULL;
  cp->QPPFApplyQ_last_Qv  = NULL;
  
  cp->GGt_relative_fill   = 1.0;

  cp->setupcalled         = PETSC_FALSE;
  cp->setfromoptionscalled= 0;

  cp->dataChange          = PETSC_FALSE;
  cp->variantChange       = PETSC_FALSE;
  cp->explicitInvChange   = PETSC_FALSE;
  cp->GChange             = PETSC_FALSE;

  cp->G_has_orthonormal_rows_explicitly = PETSC_FALSE;
  cp->G_has_orthonormal_rows_implicitly = PETSC_FALSE;
  cp->explicitInv         = PETSC_FALSE;
  cp->redundancy          = PETSC_DEFAULT;

  *qppf_new = cp;
  PetscCallMPI(MPI_Barrier(comm));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFSetG(QPPF cp, Mat G)
{
  Mat G_orig;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 2);
  PetscCall(PetscObjectQuery((PetscObject)G, "MatOrthColumns_Implicit_A", (PetscObject*)&G_orig));
  if (G_orig) {
    G = G_orig;
    cp->G_has_orthonormal_rows_implicitly = PETSC_TRUE;
  }

  if (cp->G  == G) PetscFunctionReturn(0);
  PetscCall(MatDestroy(&cp->G));
  cp->G  = G;
  PetscCall(MatGetSize(G, &cp->GM, &cp->GN));
  PetscCall(MatGetLocalSize(G, &cp->Gm, &cp->Gn));
  PetscCall(PetscObjectReference((PetscObject) G));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject) G, (PetscObject) cp, 1));
  cp->dataChange = PETSC_TRUE;
  cp->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFSetExplicitInv(QPPF cp, PetscBool explicitInv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  PetscValidLogicalCollectiveBool(cp, explicitInv, 2);
  if (cp->explicitInv != explicitInv)
  {
    cp->explicitInv = explicitInv;
    cp->explicitInvChange = PETSC_TRUE;
    cp->setupcalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFSetRedundancy(QPPF cp, PetscInt nred)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  PetscValidLogicalCollectiveInt(cp, nred, 2);
  if (cp->redundancy == nred) PetscFunctionReturn(0);
  cp->redundancy = nred;
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFSetFromOptions(QPPF cp)
{
  PetscBool set, flg;
  PetscInt nred;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject) cp);

  PetscCall(PetscOptionsBool("-qppf_explicit", "", "QPPFSetExplicitInv", cp->explicitInv, &flg, &set));
  if (set) PetscCall(QPPFSetExplicitInv(cp, flg));

  PetscCall(PetscOptionsInt("-qppf_redundancy", "number of parallel redundant solves of CP, each with (size of CP's comm)/qppf_redundancy processes", "QPPFSetRedundancy", cp->redundancy, &nred, &set));
  if (set) PetscCall(QPPFSetRedundancy(cp, nred));
  
  cp->setfromoptionscalled++;
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode QPPFSetUpGt_Private(QPPF cp, Mat *newGt)
{
  Mat Gt;
  MatTransposeType ttype;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBeginI;
  ttype = cp->G_has_orthonormal_rows_explicitly ? MAT_TRANSPOSE_CHEAPEST : MAT_TRANSPOSE_EXPLICIT;
  PetscObjectOptionsBegin((PetscObject)cp);
  PetscCall(PetscOptionsBool("-MatTrMatMult_2extension","MatTransposeMatMult_BlockDiag_Extension_2extension","Mat type of resulting matrix will be extension",flg,&flg,NULL));
  PetscOptionsEnd();
  if (flg) {
    ttype = MAT_TRANSPOSE_CHEAPEST;
  }
  PetscCall(PermonMatTranspose(cp->G,ttype,&Gt));
  PetscCall(PetscObjectSetName((PetscObject)Gt,"Gt"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject) Gt,(PetscObject) cp, 1));
  *newGt = Gt;
  PetscFunctionReturnI(0);
}

static PetscErrorCode QPPFSetUpGGt_Private(QPPF cp, Mat *newGGt)
{
  MPI_Comm comm;
  Mat GGt=NULL;
  PetscBool GGt_explicit = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBeginI;
  PetscCall(PetscObjectGetComm((PetscObject) cp, &comm));

  PetscCall(QPPFSetUpGt_Private(cp,&cp->Gt));
  
  if (cp->G_has_orthonormal_rows_explicitly) {
    PetscCall(PetscInfo(cp, "G has orthonormal rows, returning GGt = NULL\n"));
    *newGGt = NULL;
    PetscFunctionReturnI(0);
  }

  PetscCall(PetscLogEventBegin(QPPF_SetUp_GGt,cp,0,0,0));
  
  PetscObjectOptionsBegin((PetscObject)cp);
  //TODO DIRTY
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_explicit_G",&GGt_explicit,NULL));
  if (GGt_explicit) {//implicit G -> implicit GGt
    PetscCall(PetscOptionsGetBool(((PetscObject)cp)->options,NULL,"-qppf_explicit_GGt",&GGt_explicit,NULL));
  }
  PetscOptionsEnd();

  if (GGt_explicit) {
    //TODO if GGt fill > 0.3, use PETSC_FALSE (dense result)
    PetscCall(PermonMatMatMult(cp->G,cp->Gt,MAT_INITIAL_MATRIX,cp->GGt_relative_fill,&GGt));
  } else {
    Mat GGt_arr[3];

    PetscCall(MatCreateTimer(cp->G,&GGt_arr[1]));
    PetscCall(MatCreateTimer(cp->Gt,&GGt_arr[0]));

    PetscCall(MatCreateProd(comm, 2, GGt_arr, &GGt));
    PetscCall(PetscObjectSetName((PetscObject)GGt,"GGt"));
    PetscCall(MatCreateTimer(GGt,&GGt_arr[2]));
    PetscCall(MatDestroy(&GGt));
    GGt = GGt_arr[2];

    PetscCall(PetscObjectCompose((PetscObject)GGt,"Gt",(PetscObject)cp->Gt));
    PetscCall(PetscObjectCompose((PetscObject)GGt,"G",(PetscObject)cp->G));
  }

  PetscCall(PetscLogEventEnd(  QPPF_SetUp_GGt,cp,0,0,0));

  {
    PetscCall(PetscObjectSetName((PetscObject)GGt,"GGt"));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)GGt,(PetscObject)cp,1));

    PetscCall(MatSetOption(GGt, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(GGt, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
    /* ignore PETSC_ERR_SUP - setting option missing in the mat format*/
    /* TODO add this as a macro TRYIGNOREERR*/
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,NULL));
    ierr =  MatSetOption(GGt, MAT_SPD, PETSC_TRUE);
    PetscCall(PetscPopErrorHandler());
    if (ierr != PETSC_ERR_SUP) {
      PetscCall(MatSetOption(GGt, MAT_SPD, PETSC_TRUE));
    }
    PetscCall(FllopDebug("assert GGt always PSD (MAT_SYMMETRIC=1, MAT_SYMMETRIC_ETERNAL=1, MAT_SPD=1)\n"));
  }
  *newGGt = GGt;
  PetscFunctionReturnI(0);
}

static PetscErrorCode QPPFSetUpGGtinv_Private(QPPF cp, Mat *GGtinv_new)
{
  Mat GGt=NULL, GGtinv=NULL;
  MPI_Comm comm;

  PetscFunctionBeginI;
  PetscCall(PetscObjectGetComm((PetscObject) cp, &comm));
  
  /* init GGt, can be NULL e.g. in case of orthonormalization */
  PetscCall(QPPFSetUpGGt_Private(cp, &GGt));
  if (!GGt) {
    *GGtinv_new = NULL;
    PetscFunctionReturnI(0)
  }

  PetscCall(PetscLogEventBegin(QPPF_SetUp_GGtinv,cp,0,0,0));
  PetscCall(MatCreateInv(GGt, MAT_INV_MONOLITHIC, &GGtinv));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject) GGtinv, (PetscObject) cp, 1));
  PetscCall(PetscObjectSetName((PetscObject) GGtinv, "GGtinv"));
  PetscCall(MatSetOptionsPrefix(GGtinv, ((PetscObject)cp)->prefix));
  PetscCall(MatAppendOptionsPrefix(GGtinv, "qppf_"));
  PetscCall(MatDestroy(&GGt));
  
  PetscCall(MatInvSetRedundancy(GGtinv, cp->redundancy));

  if (cp->setfromoptionscalled) {
    PetscCall(PermonMatSetFromOptions(GGtinv));
  }
 
  PetscCall(MatAssemblyBegin(GGtinv, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(  GGtinv, MAT_FINAL_ASSEMBLY));

  PetscCall(MatInvGetRedundancy(GGtinv, &cp->redundancy));
  
  if (cp->explicitInv) {
    Mat GGtinv_explicit;
    KSP ksp;

    PetscCall(PetscInfo(cp, "computing explicit inverse...\n"));
    PetscCall(MatInvExplicitly(GGtinv, PETSC_TRUE, MAT_INITIAL_MATRIX, &GGtinv_explicit));
    PetscCall(PetscInfo(cp, "explicit inverse computed\n"));

    /* retain ksp only to hold the solver options */    
    PetscCall(MatInvGetKSP(GGtinv, &ksp));
    PetscCall(KSPReset(ksp));    
    PetscCall(PetscObjectCompose((PetscObject)GGtinv_explicit,"ksp",(PetscObject)ksp));
    PetscCall(MatDestroy(&GGtinv));
    GGtinv = GGtinv_explicit;
  }

  *GGtinv_new = GGtinv;
  PetscCall(PetscLogEventEnd(  QPPF_SetUp_GGtinv,cp,0,0,0));
  PetscFunctionReturnI(0);
}

static PetscErrorCode QPPFSetUpView_Private(QPPF cp)
{
  PetscBool flg;
  char filename[PETSC_MAX_PATH_LEN];  

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetString(((PetscObject)cp)->options,((PetscObject)cp)->prefix,"-qppf_view",filename,PETSC_MAX_PATH_LEN,&flg));
  if (flg && !PetscPreLoadingOn) {
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIOpen(((PetscObject)cp)->comm,filename,&viewer));
    PetscCall(QPPFView(cp,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }  
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFReset(QPPF cp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  cp->setupcalled = PETSC_FALSE;
  PetscCall(MatDestroy(&cp->GGtinv));
  PetscCall(VecDestroy(&cp->Gt_right));
  PetscCall(VecDestroy(&cp->G_left));
  PetscCall(VecDestroy(&cp->alpha_tilde));
  PetscCall(VecDestroy(&cp->QPPFApplyQ_last_Qv));
  PetscCall(MatDestroy(&cp->Gt));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFSetUp(QPPF cp)
{
  MPI_Comm comm;
  PetscMPIInt rank, size;
  const char *name;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  
  //if (cp->setupcalled && !cp->explicitInv) PetscFunctionReturn(0); //TODO why the &&?
  if (cp->setupcalled) PetscFunctionReturn(0);

  FllopTraceBegin;
  PetscCall(PetscLogEventBegin(QPPF_SetUp, cp, cp->GGtinv, cp->G, cp->Gt));

  PetscCall(PetscObjectGetComm((PetscObject) cp, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscObjectGetName((PetscObject)cp->G,&name));
  PetscCall(PetscInfo(cp, "cp: %p  Mat %s: %p  change flags: %d %d %d %d\n",  (void*)cp, name, (void*)cp->G, cp->dataChange,  cp->variantChange,  cp->explicitInvChange,  cp->GChange));

  /* detect orthonormal rows quickly */
  if (!cp->G_has_orthonormal_rows_implicitly) {
    PetscCall(MatHasOrthonormalRows(cp->G,PETSC_SMALL,3,&cp->G_has_orthonormal_rows_explicitly ));
    PetscCall(PetscInfo(cp, "Mat %s has %sorthonormal rows\n",name,cp->G_has_orthonormal_rows_explicitly?"":"NOT "));
  }

  /* re-init GGt inverse */
  if (!cp->GGtinv) {
    PetscCall(MatDestroy(&cp->GGtinv));
    PetscCall(QPPFSetUpGGtinv_Private(cp, &cp->GGtinv));
  } else {
    //PERMON_ASSERT(((Mat_Inv*)cp->GGtinv->data)->ksp->pc->setupcalled,"setupcalled");
    if (!cp->Gt) {
      PetscCall(QPPFSetUpGt_Private(cp,&cp->Gt));
    }
  }

  PetscCall(MatCreateVecs(cp->G, PETSC_IGNORE, &(cp->G_left)));
  PetscCall(VecDuplicate(cp->G_left, &(cp->Gt_right)));
  PetscCall(VecDuplicate(cp->G_left, &(cp->alpha_tilde)));
  PetscCall(VecZeroEntries(cp->alpha_tilde));

  if (cp->GGtinv && !cp->explicitInv) PetscCall(MatInvSetUp(cp->GGtinv));

  cp->it_GGtinvv       = 0;
  cp->conv_GGtinvv     = (KSPConvergedReason) 0;  
  cp->dataChange          = PETSC_FALSE;
  cp->variantChange       = PETSC_FALSE;
  cp->explicitInvChange   = PETSC_FALSE;
  cp->GChange             = PETSC_FALSE;
  cp->setupcalled         = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(  QPPF_SetUp, cp, cp->GGtinv, cp->G, cp->Gt));

#if defined(PETSC_USE_DEBUG)
  /* check coarse problem, P*G' should be a zero matrix */
  /*TODO fix{
    Mat P;
    PetscBool flg;
    PetscCall(QPPFCreateP(cp,&P));
    PetscCall(MatMatIsZero(P,cp->Gt,PETSC_SMALL,2,&flg));
    PetscCall(MatDestroy(&P));
    if (!flg) SETERRQ(comm,PETSC_ERR_PLIB,"P*G' must give a zero matrix"); 
  }*/
#endif

  PetscCall(QPPFSetUpView_Private(cp));
  PetscFunctionReturnI(0);
}

PetscErrorCode QPPFGetAlphaTilde(QPPF cp, Vec *alpha_tilde)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(alpha_tilde,2);
  
  *alpha_tilde = cp->alpha_tilde;
  PetscFunctionReturn(0);
}

/* Applies the orthogonal projector Q = G'*inv(G*G')*G to vector v */
PetscErrorCode QPPFApplyQ(QPPF cp, Vec v, Vec Qv)
{
  Vec Gt_right;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Qv,VEC_CLASSID,3);

  /* if v is the same as the last time, reuse the last computed product Q*v */
  if (v == cp->QPPFApplyQ_last_v && ((PetscObject)v)->state == cp->QPPFApplyQ_last_v_state && cp->QPPFApplyQ_last_Qv) {
    PetscCall(VecCopy(cp->QPPFApplyQ_last_Qv, Qv));
    PetscFunctionReturn(0);
  }
  
  PetscCall(QPPFSetUp(cp));

  PetscCall(PetscLogEventBegin(QPPF_ApplyQ,cp,v,Qv,0));

  /* G_left = G*v */
  PetscCall(PetscLogEventBegin(QPPF_ApplyG,cp,v,0,0));
  PetscCall(MatMult(cp->G, v, cp->G_left));
  PetscCall(PetscLogEventEnd(QPPF_ApplyG,cp,v,0,0));

  if (!cp->G_has_orthonormal_rows_explicitly) {
    /* Gt_right = (GG^T)^{-1} * G_left */
    PetscCall(QPPFApplyCP(cp, cp->G_left, cp->Gt_right));
    Gt_right = cp->Gt_right;
  } else {
    Gt_right = cp->G_left;
  }

  /* alpha_tilde = (GG^T)^{-1} * G_left */
  PetscCall(VecCopy(cp->Gt_right, cp->alpha_tilde));
  
  /* Qv = Gt*Gt_right */
  PetscCall(PetscLogEventBegin(QPPF_ApplyGt,cp,Qv,0,0));
  PetscCall(MatMult(cp->Gt, Gt_right, Qv));
  PetscCall(PetscLogEventEnd(QPPF_ApplyGt,cp,Qv,0,0));

  /* remember current v and Qv */
  cp->QPPFApplyQ_last_v  = v;
  if (!cp->QPPFApplyQ_last_Qv) PetscCall(VecDuplicate(Qv,&cp->QPPFApplyQ_last_Qv));
  PetscCall(VecCopy(Qv, cp->QPPFApplyQ_last_Qv));
  PetscCall(PetscObjectStateGet((PetscObject)v,&cp->QPPFApplyQ_last_v_state));

  PetscCall(PetscLogEventEnd(QPPF_ApplyQ,cp,v,Qv,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)Qv));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFApplyHalfQ(QPPF cp, Vec x, Vec y)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCall(QPPFSetUp(cp));

  PetscCall(PetscLogEventBegin(QPPF_ApplyHalfQ,cp,x,y,0));

  /* G_left = G*v */
  PetscCall(PetscLogEventBegin(QPPF_ApplyG,cp,x,0,0));
  PetscCall(MatMult(cp->G, x, cp->G_left));
  PetscCall(PetscLogEventEnd(QPPF_ApplyG,cp,x,0,0));

  /* y = (GG^T)^{-1} * G_left */
  PetscCall(QPPFApplyCP(cp, cp->G_left, y));

  PetscCall(PetscLogEventEnd(QPPF_ApplyHalfQ,cp,x,y,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturnI(0);
}

PetscErrorCode QPPFApplyHalfQTranspose(QPPF cp, Vec x, Vec y)
{
  Vec Gt_right;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscCall(QPPFSetUp(cp));

  PetscCall(PetscLogEventBegin(QPPF_ApplyHalfQ,cp,x,y,0));
  if (!cp->G_has_orthonormal_rows_explicitly) {
    /* Gt_right = (GG^T)^{-1} * G_left */
    PetscCall(QPPFApplyCP(cp, x, cp->Gt_right));
    Gt_right = cp->Gt_right;
  } else {
    Gt_right = x;
  }

  /* y = Gt*Gt_right */
  PetscCall(PetscLogEventBegin(QPPF_ApplyGt,cp,y,0,0));
  PetscCall(MatMult(cp->Gt, Gt_right, y));
  PetscCall(PetscLogEventEnd(QPPF_ApplyGt,cp,y,0,0));

  PetscCall(PetscLogEventEnd(QPPF_ApplyHalfQ,cp,x,y,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturnI(0);
}

PetscErrorCode QPPFApplyP(QPPF cp, Vec v, Vec Pv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Pv,VEC_CLASSID,3);
  PetscCall(PetscLogEventBegin(QPPF_ApplyP,cp,v,Pv,0));
  PetscCall(QPPFApplyQ(cp, v, Pv));
  PetscCall(VecAYPX(Pv, -1.0, v));  //Pv = v - Pv
  PetscCall(PetscLogEventEnd(QPPF_ApplyP,cp,v,Pv,0));
  PetscCall(PetscObjectStateIncrease((PetscObject)Pv));
  PetscFunctionReturn(0);
}

/* Applies GtG = G'*G to vector v */
PetscErrorCode QPPFApplyGtG(QPPF cp, Vec v, Vec GtGv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidHeaderSpecific(GtGv,VEC_CLASSID,3);
  if (cp->G_has_orthonormal_rows_explicitly || cp->G_has_orthonormal_rows_implicitly) {
    PetscCall(QPPFApplyQ(cp,v,GtGv));
    PetscFunctionReturn(0);
  }

  PetscCall(QPPFSetUp(cp));
  
  /* G_left = G*v */
  PetscCall(PetscLogEventBegin(QPPF_ApplyG,cp,v,GtGv,0));
  PetscCall(MatMult(cp->G, v, cp->G_left));
  PetscCall(PetscLogEventEnd(QPPF_ApplyG,cp,v,GtGv,0));

  /* GtGv = Gt*G_left */
  PetscCall(PetscLogEventBegin(QPPF_ApplyGt,cp,v,GtGv,0));
  PetscCall(MatMult(cp->Gt, cp->G_left, GtGv));
  PetscCall(PetscLogEventEnd(QPPF_ApplyGt,cp,v,GtGv,0));

  PetscCall(PetscObjectStateIncrease((PetscObject)GtGv));
  PetscFunctionReturn(0);
}

/* Applies inv(G*G') to vector x; in other words y solves the coarse problem G*G'*y = x */
PetscErrorCode QPPFApplyCP(QPPF cp, Vec x, Vec y)
{
  MPI_Comm comm = ((PetscObject) cp)->comm;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(QPPFSetUp(cp));

  PetscCall(PetscLogEventBegin(QPPF_ApplyCP, cp, cp->GGtinv, x, y));
  
  /* Gt_right = (GG^T)^{-1} * x */
  if (cp->GGtinv) {
    PetscInt iter;
    KSP GGtinv_ksp;

    PetscCall(MatMult(cp->GGtinv, x, y));

    if (!cp->explicitInv) {
      PetscCall(MatInvGetKSP(cp->GGtinv, &GGtinv_ksp));
      PetscCall(KSPGetIterationNumber(GGtinv_ksp, &iter));
      cp->it_GGtinvv += iter;
      PetscCall(KSPGetConvergedReason(GGtinv_ksp, &cp->conv_GGtinvv));
    }
  } else {
    PetscCall(VecCopy(x, y));
  }
    
  PetscCall(PetscLogEventEnd(  QPPF_ApplyCP, cp, cp->GGtinv, x, y));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(0);
}

/* Get operator Q = G'*inv(G*G')*G in implicit form */
PetscErrorCode QPPFCreateQ(QPPF cp, Mat *newQ)
{
  Mat Q;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newQ,2);
  PetscCall(MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gn, cp->Gn, cp->GN, cp->GN, cp, &Q));
  PetscCall(MatShellSetOperation(Q, MATOP_MULT, (void(*)()) QPPFMatMult_Q));
  PetscCall(PetscObjectCompose((PetscObject)Q,"qppf",(PetscObject)cp));
  PetscCall(PetscObjectSetName((PetscObject)Q, "Q"));
  *newQ = Q;
  PetscFunctionReturn(0);
}

/* Get operator Q = inv(G*G')*G in implicit form */
PetscErrorCode QPPFCreateHalfQ(QPPF cp, Mat *newHalfQ)
{
  Mat HalfQ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newHalfQ,2);
  PetscCall(MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gm, cp->Gn, cp->GM, cp->GN, cp, &HalfQ));
  PetscCall(MatShellSetOperation(HalfQ, MATOP_MULT, (void(*)()) QPPFMatMult_HalfQ));
  PetscCall(MatShellSetOperation(HalfQ, MATOP_MULT_TRANSPOSE, (void(*)()) QPPFMatMultTranspose_HalfQ));
  PetscCall(PetscObjectCompose((PetscObject)HalfQ,"qppf",(PetscObject)cp));
  PetscCall(PetscObjectSetName((PetscObject)HalfQ, "HalfQ"));
  *newHalfQ = HalfQ;
  PetscFunctionReturn(0);
}

/* Get operator P = I - G'*inv(G*G')*G in implicit form */
PetscErrorCode QPPFCreateP(QPPF cp, Mat *newP)
{
  Mat P;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newP,2);
  PetscCall(MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gn, cp->Gn, cp->GN, cp->GN, cp, &P));
  PetscCall(MatShellSetOperation(P, MATOP_MULT, (void(*)()) QPPFMatMult_P));
  PetscCall(PetscObjectCompose((PetscObject)P,"qppf",(PetscObject)cp));
  PetscCall(PetscObjectSetName((PetscObject)P, "P"));
  *newP = P;
  PetscFunctionReturn(0);
}

/* Get operator GtG = G'*G in implicit form */
PetscErrorCode QPPFCreateGtG(QPPF cp, Mat *newGtG)
{
  Mat GtG;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newGtG,2);
  PetscCall(MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gn, cp->Gn, cp->GN, cp->GN, cp, &GtG));
  PetscCall(MatShellSetOperation(GtG, MATOP_MULT, (void(*)()) QPPFMatMult_GtG));
  PetscCall(PetscObjectCompose((PetscObject)GtG,"qppf",(PetscObject)cp));
  PetscCall(PetscObjectSetName((PetscObject)GtG, "GtG"));
  *newGtG = GtG;
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFGetG(QPPF cp, Mat *G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(G,2);
  *G = cp->G;
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFGetGHasOrthonormalRows(QPPF cp, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = (PetscBool) (cp->G_has_orthonormal_rows_explicitly || cp->G_has_orthonormal_rows_implicitly);
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFGetGGt(QPPF cp, Mat *GGt)
{
  Mat GGtinv;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(GGt,2);
  PetscCall(QPPFGetGGtinv(cp,&GGtinv));
  *GGt = NULL;
  if ((GGtinv != NULL) & !cp->explicitInv) PetscCall(MatInvGetMat(GGtinv,GGt));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFGetGGtinv(QPPF cp, Mat *GGtinv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(GGtinv,2);
  *GGtinv = cp->GGtinv;
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFGetKSP(QPPF cp, KSP *ksp)
{
  Mat GGtinv;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(ksp,2);
  PetscCall(QPPFGetGGtinv(cp,&GGtinv));
  *ksp = NULL;
  if (GGtinv) PetscCall(MatInvGetKSP(GGtinv,ksp));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFView(QPPF cp, PetscViewer viewer)
{
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscCall(PetscObjectGetComm((PetscObject)cp, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_(comm);
  } else {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(cp,1,viewer,2);
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (!iascii) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by QPPF",((PetscObject)viewer)->type_name);

  PetscCall(PetscObjectName((PetscObject)cp));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)cp, viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "setup called:       %c\n", cp->setupcalled ? 'y' : 'n'));
  PetscCall(PetscViewerASCIIPrintf(viewer, "G has orth. rows e.:%c\n", cp->G_has_orthonormal_rows_explicitly ? 'y' : 'n'));
  PetscCall(PetscViewerASCIIPrintf(viewer, "G has orth. rows i.:%c\n", cp->G_has_orthonormal_rows_implicitly ? 'y' : 'n'));
  PetscCall(PetscViewerASCIIPrintf(viewer, "explicit:           %c\n", cp->explicitInv ? 'y' : 'n'));
  PetscCall(PetscViewerASCIIPrintf(viewer, "redundancy:         %d\n", cp->redundancy));
  PetscCall(PetscViewerASCIIPrintf(viewer, "last conv. reason:  %d\n", cp->conv_GGtinvv));
  PetscCall(PetscViewerASCIIPrintf(viewer, "cumulative #iter.:  %d\n", cp->it_GGtinvv));

  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
  if (cp->explicitInv) {
    KSP ksp;
    PetscViewer scv;
    PetscCall(PetscObjectQuery((PetscObject)cp->GGtinv,"ksp",(PetscObject*)&ksp));
    PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
    PetscCall(PetscViewerGetSubViewer(viewer, comm, &scv));
    PetscCall(KSPViewBriefInfo(ksp, scv));
    PetscCall(PetscViewerRestoreSubViewer(viewer, comm, &scv));
  } else {
    if (cp->GGtinv) PetscCall(MatView(cp->GGtinv, viewer));
  }
  if (cp->G)    PetscCall(MatPrintInfo(cp->G));
  if (cp->Gt)   PetscCall(MatPrintInfo(cp->Gt));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode QPPFDestroy(QPPF *cp)
{
  PetscFunctionBegin;
  if (!*cp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*cp, QPPF_CLASSID, 1);
  if (--((PetscObject) (*cp))->refct > 0)
  {
    *cp = 0;
    PetscFunctionReturn(0);
  }
  PetscCall(QPPFReset(*cp));
  PetscCall(MatDestroy(&(*cp)->G));
  PetscCall(PetscHeaderDestroy(cp));
  PetscFunctionReturn(0);
}
