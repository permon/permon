
#include <permon/private/qppfimpl.h>
#include <permonksp.h>
PetscClassId QPPF_CLASSID;
PetscLogEvent QPPF_SetUp, QPPF_SetUp_Gt, QPPF_SetUp_GGt, QPPF_SetUp_GGtinv;
PetscLogEvent QPPF_ApplyCP, QPPF_ApplyCP_gt, QPPF_ApplyCP_sc;
PetscLogEvent QPPF_ApplyP, QPPF_ApplyQ, QPPF_ApplyHalfQ, QPPF_ApplyG, QPPF_ApplyGt;

const char *QPPFVariants[] = {"zero","all","dist", "QPPFVariant","QPPF_",0};

#define RANK0 0
#define G_RELATIVE_FILL 1.0


#undef __FUNCT__
#define __FUNCT__ "QPPFMatMult_P"
static PetscErrorCode QPPFMatMult_P(Mat matP, Vec v, Vec Pv)
{
  QPPF cp;

  PetscFunctionBegin;
  TRY( MatShellGetContext(matP, (QPPF*)&cp) );
  TRY( QPPFApplyP(cp, v, Pv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFMatMult_Q"
static PetscErrorCode QPPFMatMult_Q(Mat matQ, Vec v, Vec Qv)
{
  QPPF cp;

  PetscFunctionBegin;
  TRY( MatShellGetContext(matQ, (QPPF*)&cp) );
  TRY( QPPFApplyQ(cp, v, Qv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFMatMult_HalfQ"
static PetscErrorCode QPPFMatMult_HalfQ(Mat matHalfQ, Vec x, Vec y)
{
  QPPF cp;

  PetscFunctionBegin;
  TRY( MatShellGetContext(matHalfQ, (QPPF*)&cp) );
  TRY( QPPFApplyHalfQ(cp,x,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFMatMultTranspose_HalfQ"
static PetscErrorCode QPPFMatMultTranspose_HalfQ(Mat matHalfQ, Vec x, Vec y)
{
  QPPF cp;

  PetscFunctionBegin;
  TRY( MatShellGetContext(matHalfQ, (QPPF*)&cp) );
  TRY( QPPFApplyHalfQTranspose(cp,x,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFMatMult_GtG"
static PetscErrorCode QPPFMatMult_GtG(Mat matGtG, Vec v, Vec GtGv)
{
  QPPF cp;

  PetscFunctionBegin;
  TRY( MatShellGetContext(matGtG, (QPPF*)&cp) );
  TRY( QPPFApplyGtG(cp, v, GtGv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFCreate"
PetscErrorCode QPPFCreate(MPI_Comm comm, QPPF* qppf_new)
{
  QPPF cp;

  PetscFunctionBegin;
  PetscValidPointer(qppf_new, 2);
  *qppf_new = 0;
  TRY( QPPFInitializePackage() );

  TRY( PetscHeaderCreate(cp,QPPF_CLASSID,"QPPF", "Projector Factory", "QPPF", comm, QPPFDestroy, QPPFView) );

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
  TRY( MPI_Barrier(comm) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetG"
PetscErrorCode QPPFSetG(QPPF cp, Mat G)
{
  Mat G_orig;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 2);
  TRY( PetscObjectQuery((PetscObject)G, "MatOrthColumns_Implicit_A", (PetscObject*)&G_orig) );
  if (G_orig) {
    G = G_orig;
    cp->G_has_orthonormal_rows_implicitly = PETSC_TRUE;
  }

  if (cp->G  == G) PetscFunctionReturn(0);
  TRY( MatDestroy(&cp->G) );
  cp->G  = G;
  TRY( MatGetSize(G, &cp->GM, &cp->GN) );
  TRY( MatGetLocalSize(G, &cp->Gm, &cp->Gn) );
  TRY( PetscObjectReference((PetscObject) G) );
  TRY( PetscObjectIncrementTabLevel((PetscObject) G, (PetscObject) cp, 1) );
  cp->dataChange = PETSC_TRUE;
  cp->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetExplicitInv"
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

#undef __FUNCT__
#define __FUNCT__ "QPPFSetRedundancy"
PetscErrorCode QPPFSetRedundancy(QPPF cp, PetscInt nred)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  PetscValidLogicalCollectiveInt(cp, nred, 2);
  if (cp->redundancy == nred) PetscFunctionReturn(0);
  cp->redundancy = nred;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetFromOptions"
PetscErrorCode QPPFSetFromOptions(QPPF cp)
{
  PetscBool set, flg;
  PetscInt nred;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  _fllop_ierr = PetscObjectOptionsBegin((PetscObject) cp);CHKERRQ(_fllop_ierr);

  TRY( PetscOptionsBool("-qppf_explicit", "", "QPPFSetExplicitInv", cp->explicitInv, &flg, &set) );
  if (set) TRY( QPPFSetExplicitInv(cp, flg) );

  TRY( PetscOptionsInt("-qppf_redundancy", "number of parallel redundant solves of CP, each with (size of CP's comm)/qppf_redundancy processes", "QPPFSetRedundancy", cp->redundancy, &nred, &set) );
  if (set) TRY( QPPFSetRedundancy(cp, nred) );
  
  cp->setfromoptionscalled++;
  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetUpGt_Private"
static PetscErrorCode QPPFSetUpGt_Private(QPPF cp, Mat *newGt)
{
  Mat Gt;
  MatTransposeType ttype;

  PetscFunctionBeginI;
  ttype = cp->G_has_orthonormal_rows_explicitly ? MAT_TRANSPOSE_CHEAPEST : MAT_TRANSPOSE_EXPLICIT;
  TRY( FllopMatTranspose(cp->G,ttype,&Gt) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)Gt) );
  TRY( PetscObjectSetName((PetscObject)Gt,"Gt") );
  TRY( PetscObjectIncrementTabLevel((PetscObject) Gt,(PetscObject) cp, 1) );
  *newGt = Gt;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetUpGGt_Private"
static PetscErrorCode QPPFSetUpGGt_Private(QPPF cp, Mat *newGGt)
{
  MPI_Comm comm;
  Mat GGt=NULL;
  PetscBool GGt_explicit = PETSC_TRUE;

  PetscFunctionBeginI;
  TRY( PetscObjectGetComm((PetscObject) cp, &comm) );

  TRY( QPPFSetUpGt_Private(cp,&cp->Gt) );
  
  if (cp->G_has_orthonormal_rows_explicitly) {
    TRY( PetscInfo(cp, "G has orthonormal rows, returning GGt = NULL\n") );
    *newGGt = NULL;
    PetscFunctionReturnI(0);
  }

  TRY( PetscLogEventBegin(QPPF_SetUp_GGt,cp,0,0,0) );
  
  _fllop_ierr = PetscObjectOptionsBegin((PetscObject)cp);CHKERRQ(_fllop_ierr);
  //TODO DIRTY
  TRY( PetscOptionsGetBool(NULL,NULL,"-qpt_dualize_explicit_G",&GGt_explicit,NULL) );
  if (GGt_explicit) {//implicit G -> implicit GGt
    TRY( PetscOptionsGetBool(((PetscObject)cp)->options,NULL,"-qppf_explicit_GGt",&GGt_explicit,NULL) );
  }
  _fllop_ierr = PetscOptionsEnd();CHKERRQ(_fllop_ierr);

  if (GGt_explicit) {
    //TODO if GGt fill > 0.3, use PETSC_FALSE (dense result)
    TRY( FllopMatMatMult(cp->G,cp->Gt,MAT_INITIAL_MATRIX,cp->GGt_relative_fill,&GGt) );
  } else {
    Mat GGt_arr[2];

    TRY( MatCreateTimer(cp->G,&GGt_arr[1]) );
    TRY( MatCreateTimer(cp->Gt,&GGt_arr[0]) );

    TRY( MatCreateProd(comm, 2, GGt_arr, &GGt) );
    TRY( PetscObjectSetName((PetscObject)GGt,"GGt") );
    TRY( MatCreateTimer(GGt,&GGt) );

    TRY( PetscObjectCompose((PetscObject)GGt,"Gt",(PetscObject)cp->Gt) );
    TRY( PetscObjectCompose((PetscObject)GGt,"G",(PetscObject)cp->G) );
  }

  TRY( PetscLogEventEnd(  QPPF_SetUp_GGt,cp,0,0,0) );

  {
    TRY( PetscObjectSetName((PetscObject)GGt,"GGt") );
    TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)GGt) );
    TRY( PetscObjectIncrementTabLevel((PetscObject)GGt,(PetscObject)cp,1) );

    TRY( MatSetOption(GGt, MAT_SYMMETRIC, PETSC_TRUE) );
    TRY( MatSetOption(GGt, MAT_SYMMETRY_ETERNAL, PETSC_TRUE) );
    /* ignore PETSC_ERR_SUP - setting option missing in the mat format*/
    /* TODO add this as a macro TRYIGNOREERR*/
    TRY( PetscPushErrorHandler(PetscReturnErrorHandler,NULL) );
    _fllop_ierr =  MatSetOption(GGt, MAT_SPD, PETSC_TRUE);
    TRY( PetscPopErrorHandler() );
    if (_fllop_ierr != PETSC_ERR_SUP) {
      TRY( MatSetOption(GGt, MAT_SPD, PETSC_TRUE) );
    }
    TRY( FllopDebug("assert GGt always PSD (MAT_SYMMETRIC=1, MAT_SYMMETRIC_ETERNAL=1, MAT_SPD=1)\n") );
  }
  *newGGt = GGt;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetUpGGtinv_Private"
static PetscErrorCode QPPFSetUpGGtinv_Private(QPPF cp, Mat *GGtinv_new)
{
  Mat GGt=NULL, GGtinv=NULL;
  MPI_Comm comm;

  PetscFunctionBeginI;
  TRY( PetscObjectGetComm((PetscObject) cp, &comm) );
  
  /* init GGt, can be NULL e.g. in case of orthonormalization */
  TRY( QPPFSetUpGGt_Private(cp, &GGt) );
  if (!GGt) {
    *GGtinv_new = NULL;
    PetscFunctionReturnI(0)
  }

  TRY( PetscLogEventBegin(QPPF_SetUp_GGtinv,cp,0,0,0) );
  TRY( MatCreateInv(GGt, MAT_INV_MONOLITHIC, &GGtinv) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)GGtinv) );
  TRY( PetscObjectIncrementTabLevel((PetscObject) GGtinv, (PetscObject) cp, 1) );
  TRY( PetscObjectSetName((PetscObject) GGtinv, "GGtinv") );
  TRY( MatSetOptionsPrefix(GGtinv, ((PetscObject)cp)->prefix) );
  TRY( MatAppendOptionsPrefix(GGtinv, "qppf_") );
  TRY( MatDestroy(&GGt) );
  
  TRY( MatInvSetRedundancy(GGtinv, cp->redundancy) );

  if (cp->setfromoptionscalled) {
    TRY( PermonMatSetFromOptions(GGtinv) );
  }
 
  TRY( MatAssemblyBegin(GGtinv, MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(  GGtinv, MAT_FINAL_ASSEMBLY) );

  TRY( MatInvGetRedundancy(GGtinv, &cp->redundancy) );
  
  if (cp->explicitInv) {
    Mat GGtinv_explicit;
    KSP ksp;

    TRY( PetscInfo(cp, "computing explicit inverse...\n") );
    TRY( MatInvExplicitly(GGtinv, PETSC_TRUE, MAT_INITIAL_MATRIX, &GGtinv_explicit) );
    TRY( PetscInfo(cp, "explicit inverse computed\n") );

    /* retain ksp only to hold the solver options */    
    TRY( MatInvGetKSP(GGtinv, &ksp) );
    TRY( KSPReset(ksp) );    
    TRY( PetscObjectCompose((PetscObject)GGtinv_explicit,"ksp",(PetscObject)ksp) );
    TRY( MatDestroy(&GGtinv) );
    GGtinv = GGtinv_explicit;
  }

  *GGtinv_new = GGtinv;
  TRY( PetscLogEventEnd(  QPPF_SetUp_GGtinv,cp,0,0,0) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetUpView_Private"
static PetscErrorCode QPPFSetUpView_Private(QPPF cp)
{
  PetscBool flg;
  char filename[PETSC_MAX_PATH_LEN];  

  PetscFunctionBegin;
  TRY( PetscOptionsGetString(((PetscObject)cp)->options,((PetscObject)cp)->prefix,"-qppf_view",filename,PETSC_MAX_PATH_LEN,&flg) );
  if (flg && !PetscPreLoadingOn) {
    PetscViewer viewer;
    TRY( PetscViewerASCIIOpen(((PetscObject)cp)->comm,filename,&viewer) );
    TRY( QPPFView(cp,viewer) );
    TRY( PetscViewerDestroy(&viewer) );
  }  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFReset"
PetscErrorCode QPPFReset(QPPF cp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp, QPPF_CLASSID, 1);
  cp->setupcalled = PETSC_FALSE;
  TRY( MatDestroy(&cp->GGtinv) );
  TRY( VecDestroy(&cp->Gt_right) );
  TRY( VecDestroy(&cp->G_left) );
  TRY( VecDestroy(&cp->alpha_tilde) );
  TRY( VecDestroy(&cp->QPPFApplyQ_last_Qv) );
  TRY( MatDestroy(&cp->Gt) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFSetUp"
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
  TRY( PetscLogEventBegin(QPPF_SetUp, cp, cp->GGtinv, cp->G, cp->Gt) );

  TRY( PetscObjectGetComm((PetscObject) cp, &comm) );
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( MPI_Comm_size(comm, &size) );
  TRY( PetscObjectGetName((PetscObject)cp->G,&name) );
  TRY( PetscInfo7(cp, "cp: %x  Mat %s: %x  change flags: %d %d %d %d\n",  cp, name, cp->G, cp->dataChange,  cp->variantChange,  cp->explicitInvChange,  cp->GChange) );

  /* detect orthonormal rows quickly */
  if (!cp->G_has_orthonormal_rows_implicitly) {
    TRY( MatHasOrthonormalRows(cp->G,PETSC_SMALL,3,&cp->G_has_orthonormal_rows_explicitly ) );
    TRY( PetscInfo2(cp, "Mat %s has %sorthonormal rows\n",name,cp->G_has_orthonormal_rows_explicitly?"":"NOT ") );
  }

  /* re-init GGt inverse */
  if (!cp->GGtinv) {
    TRY( MatDestroy(&cp->GGtinv) );
    TRY( QPPFSetUpGGtinv_Private(cp, &cp->GGtinv) );
    TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)cp->GGtinv) );
  } else {
    //FLLOP_ASSERT(((Mat_Inv*)cp->GGtinv->data)->ksp->pc->setupcalled,"setupcalled");
    if (!cp->Gt) {
      TRY( QPPFSetUpGt_Private(cp,&cp->Gt) );
    }
  }

  TRY( MatCreateVecs(cp->G, PETSC_IGNORE, &(cp->G_left)) );
  TRY( VecDuplicate(cp->G_left, &(cp->Gt_right)) );
  TRY( VecDuplicate(cp->G_left, &(cp->alpha_tilde)) );
  TRY( VecZeroEntries(cp->alpha_tilde) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)cp->G_left) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)cp->Gt_right) );

  if (cp->GGtinv && !cp->explicitInv) TRY( MatInvSetUp(cp->GGtinv) );

  cp->it_GGtinvv       = 0;
  cp->conv_GGtinvv     = (KSPConvergedReason) 0;  
  cp->dataChange          = PETSC_FALSE;
  cp->variantChange       = PETSC_FALSE;
  cp->explicitInvChange   = PETSC_FALSE;
  cp->GChange             = PETSC_FALSE;
  cp->setupcalled         = PETSC_TRUE;
  TRY( PetscLogEventEnd(  QPPF_SetUp, cp, cp->GGtinv, cp->G, cp->Gt) );

#if defined(PETSC_USE_DEBUG)
  /* check coarse problem, P*G' should be a zero matrix */
  /*TODO fix{
    Mat P;
    PetscBool flg;
    TRY( QPPFCreateP(cp,&P) );
    TRY( MatMatIsZero(P,cp->Gt,PETSC_SMALL,2,&flg) );
    TRY( MatDestroy(&P) );
    if (!flg) FLLOP_SETERRQ(comm,PETSC_ERR_PLIB,"P*G' must give a zero matrix"); 
  }*/
#endif

  TRY( QPPFSetUpView_Private(cp) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFGetAlphaTilde"
PetscErrorCode QPPFGetAlphaTilde(QPPF cp, Vec *alpha_tilde)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(alpha_tilde,2);
  
  *alpha_tilde = cp->alpha_tilde;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFApplyQ"
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
    TRY( VecCopy(cp->QPPFApplyQ_last_Qv, Qv) );
    PetscFunctionReturn(0);
  }
  
  TRY( QPPFSetUp(cp) );

  TRY( PetscLogEventBegin(QPPF_ApplyQ,cp,v,Qv,0) );

  /* G_left = G*v */
  TRY( PetscLogEventBegin(QPPF_ApplyG,cp,v,0,0) );
  TRY( MatMult(cp->G, v, cp->G_left) );
  TRY( PetscLogEventEnd(QPPF_ApplyG,cp,v,0,0) );

  if (!cp->G_has_orthonormal_rows_explicitly) {
    /* Gt_right = (GG^T)^{-1} * G_left */
    TRY( QPPFApplyCP(cp, cp->G_left, cp->Gt_right) );
    Gt_right = cp->Gt_right;
  } else {
    Gt_right = cp->G_left;
  }

  /* alpha_tilde = (GG^T)^{-1} * G_left */
  TRY( VecCopy(cp->Gt_right, cp->alpha_tilde) );
  
  /* Qv = Gt*Gt_right */
  TRY( PetscLogEventBegin(QPPF_ApplyGt,cp,Qv,0,0) );
  TRY( MatMult(cp->Gt, Gt_right, Qv) );
  TRY( PetscLogEventEnd(QPPF_ApplyGt,cp,Qv,0,0) );

  /* remember current v and Qv */
  cp->QPPFApplyQ_last_v  = v;
  if (!cp->QPPFApplyQ_last_Qv) TRY( VecDuplicate(Qv,&cp->QPPFApplyQ_last_Qv) );
  TRY( VecCopy(Qv, cp->QPPFApplyQ_last_Qv) );
  TRY( PetscObjectStateGet((PetscObject)v,&cp->QPPFApplyQ_last_v_state) );

  TRY( PetscLogEventEnd(QPPF_ApplyQ,cp,v,Qv,0) );
  TRY( PetscObjectStateIncrease((PetscObject)Qv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFApplyHalfQ"
PetscErrorCode QPPFApplyHalfQ(QPPF cp, Vec x, Vec y)
{

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  TRY( QPPFSetUp(cp) );

  TRY( PetscLogEventBegin(QPPF_ApplyHalfQ,cp,x,y,0) );

  /* G_left = G*v */
  TRY( PetscLogEventBegin(QPPF_ApplyG,cp,x,0,0) );
  TRY( MatMult(cp->G, x, cp->G_left) );
  TRY( PetscLogEventEnd(QPPF_ApplyG,cp,x,0,0) );

  /* y = (GG^T)^{-1} * G_left */
  TRY( QPPFApplyCP(cp, cp->G_left, y) );

  TRY( PetscLogEventEnd(QPPF_ApplyHalfQ,cp,x,y,0) );
  TRY( PetscObjectStateIncrease((PetscObject)y) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFApplyHalfQTranspose"
PetscErrorCode QPPFApplyHalfQTranspose(QPPF cp, Vec x, Vec y)
{
  Vec Gt_right;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  TRY( QPPFSetUp(cp) );

  TRY( PetscLogEventBegin(QPPF_ApplyHalfQ,cp,x,y,0) );
  if (!cp->G_has_orthonormal_rows_explicitly) {
    /* Gt_right = (GG^T)^{-1} * G_left */
    TRY( QPPFApplyCP(cp, x, cp->Gt_right) );
    Gt_right = cp->Gt_right;
  } else {
    Gt_right = x;
  }

  /* y = Gt*Gt_right */
  TRY( PetscLogEventBegin(QPPF_ApplyGt,cp,y,0,0) );
  TRY( MatMult(cp->Gt, Gt_right, y) );
  TRY( PetscLogEventEnd(QPPF_ApplyGt,cp,y,0,0) );

  TRY( PetscLogEventEnd(QPPF_ApplyHalfQ,cp,x,y,0) );
  TRY( PetscObjectStateIncrease((PetscObject)y) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFApplyP"
PetscErrorCode QPPFApplyP(QPPF cp, Vec v, Vec Pv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Pv,VEC_CLASSID,3);
  TRY( PetscLogEventBegin(QPPF_ApplyP,cp,v,Pv,0) );
  TRY( QPPFApplyQ(cp, v, Pv) );
  TRY( VecAYPX(Pv, -1.0, v) );  //Pv = v - Pv
  TRY( PetscLogEventEnd(QPPF_ApplyP,cp,v,Pv,0) );
  TRY( PetscObjectStateIncrease((PetscObject)Pv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFApplyGtG"
/* Applies GtG = G'*G to vector v */
PetscErrorCode QPPFApplyGtG(QPPF cp, Vec v, Vec GtGv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidHeaderSpecific(GtGv,VEC_CLASSID,3);
  if (cp->G_has_orthonormal_rows_explicitly || cp->G_has_orthonormal_rows_implicitly) {
    TRY( QPPFApplyQ(cp,v,GtGv) );
    PetscFunctionReturn(0);
  }

  TRY( QPPFSetUp(cp) );
  
  /* G_left = G*v */
  TRY( PetscLogEventBegin(QPPF_ApplyG,cp,v,GtGv,0) );
  TRY( MatMult(cp->G, v, cp->G_left) );
  TRY( PetscLogEventEnd(QPPF_ApplyG,cp,v,GtGv,0) );

  /* GtGv = Gt*G_left */
  TRY( PetscLogEventBegin(QPPF_ApplyGt,cp,v,GtGv,0) );
  TRY( MatMult(cp->Gt, cp->G_left, GtGv) );
  TRY( PetscLogEventEnd(QPPF_ApplyGt,cp,v,GtGv,0) );

  TRY( PetscObjectStateIncrease((PetscObject)GtGv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFApplyCP"
/* Applies inv(G*G') to vector x; in other words y solves the coarse problem G*G'*y = x */
PetscErrorCode QPPFApplyCP(QPPF cp, Vec x, Vec y)
{
  MPI_Comm comm = ((PetscObject) cp)->comm;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( QPPFSetUp(cp) );

  TRY( PetscLogEventBegin(QPPF_ApplyCP, cp, cp->GGtinv, x, y) );
  
  /* Gt_right = (GG^T)^{-1} * x */
  if (cp->GGtinv) {
    PetscInt iter;
    KSP GGtinv_ksp;

    TRY( MatMult(cp->GGtinv, x, y) );

    if (!cp->explicitInv) {
      TRY( MatInvGetKSP(cp->GGtinv, &GGtinv_ksp) );
      TRY( KSPGetIterationNumber(GGtinv_ksp, &iter) );
      cp->it_GGtinvv += iter;
      TRY( KSPGetConvergedReason(GGtinv_ksp, &cp->conv_GGtinvv) );
    }
  } else {
    TRY( VecCopy(x, y) );
  }
    
  TRY( PetscLogEventEnd(  QPPF_ApplyCP, cp, cp->GGtinv, x, y) );
  TRY( PetscObjectStateIncrease((PetscObject)y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFCreateQ"
/* Get operator Q = G'*inv(G*G')*G in implicit form */
PetscErrorCode QPPFCreateQ(QPPF cp, Mat *newQ)
{
  Mat Q;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newQ,2);
  TRY( MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gn, cp->Gn, cp->GN, cp->GN, cp, &Q) );
  TRY( MatShellSetOperation(Q, MATOP_MULT, (void(*)()) QPPFMatMult_Q) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)Q) );
  TRY( PetscObjectCompose((PetscObject)Q,"qppf",(PetscObject)cp) );
  TRY( PetscObjectSetName((PetscObject)Q, "Q") );
  *newQ = Q;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFCreateHalfQ"
/* Get operator Q = inv(G*G')*G in implicit form */
PetscErrorCode QPPFCreateHalfQ(QPPF cp, Mat *newHalfQ)
{
  Mat HalfQ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newHalfQ,2);
  TRY( MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gm, cp->Gn, cp->GM, cp->GN, cp, &HalfQ) );
  TRY( MatShellSetOperation(HalfQ, MATOP_MULT, (void(*)()) QPPFMatMult_HalfQ) );
  TRY( MatShellSetOperation(HalfQ, MATOP_MULT_TRANSPOSE, (void(*)()) QPPFMatMultTranspose_HalfQ) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)HalfQ) );
  TRY( PetscObjectCompose((PetscObject)HalfQ,"qppf",(PetscObject)cp) );
  TRY( PetscObjectSetName((PetscObject)HalfQ, "HalfQ") );
  *newHalfQ = HalfQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFCreateP"
/* Get operator P = I - G'*inv(G*G')*G in implicit form */
PetscErrorCode QPPFCreateP(QPPF cp, Mat *newP)
{
  Mat P;
      
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newP,2);
  TRY( MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gn, cp->Gn, cp->GN, cp->GN, cp, &P) );
  TRY( MatShellSetOperation(P, MATOP_MULT, (void(*)()) QPPFMatMult_P) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)P) );
  TRY( PetscObjectCompose((PetscObject)P,"qppf",(PetscObject)cp) );
  TRY( PetscObjectSetName((PetscObject)P, "P") );
  *newP = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFCreateGtG"
/* Get operator GtG = G'*G in implicit form */
PetscErrorCode QPPFCreateGtG(QPPF cp, Mat *newGtG)
{
  Mat GtG;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(newGtG,2);
  TRY( MatCreateShellPermon(((PetscObject) cp)->comm, cp->Gn, cp->Gn, cp->GN, cp->GN, cp, &GtG) );
  TRY( MatShellSetOperation(GtG, MATOP_MULT, (void(*)()) QPPFMatMult_GtG) );
  TRY( PetscLogObjectParent((PetscObject)cp,(PetscObject)GtG) );
  TRY( PetscObjectCompose((PetscObject)GtG,"qppf",(PetscObject)cp) );
  TRY( PetscObjectSetName((PetscObject)GtG, "GtG") );
  *newGtG = GtG;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFGetG"
PetscErrorCode QPPFGetG(QPPF cp, Mat *G)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(G,2);
  *G = cp->G;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFGetGHasOrthonormalRows"
PetscErrorCode QPPFGetGHasOrthonormalRows(QPPF cp, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = (PetscBool) (cp->G_has_orthonormal_rows_explicitly || cp->G_has_orthonormal_rows_implicitly);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFGetGGt"
PetscErrorCode QPPFGetGGt(QPPF cp, Mat *GGt)
{
  Mat GGtinv;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(GGt,2);
  TRY( QPPFGetGGtinv(cp,&GGtinv) );
  *GGt = NULL;
  if ((GGtinv != NULL) & !cp->explicitInv) TRY( MatInvGetMat(GGtinv,GGt) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFGetGGtinv"
PetscErrorCode QPPFGetGGtinv(QPPF cp, Mat *GGtinv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(GGtinv,2);
  *GGtinv = cp->GGtinv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFGetKSP"
PetscErrorCode QPPFGetKSP(QPPF cp, KSP *ksp)
{
  Mat GGtinv;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  PetscValidPointer(ksp,2);
  TRY( QPPFGetGGtinv(cp,&GGtinv) );
  *ksp = NULL;
  if (GGtinv) TRY( MatInvGetKSP(GGtinv,ksp) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFView"
PetscErrorCode QPPFView(QPPF cp, PetscViewer viewer)
{
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(cp,QPPF_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)cp, &comm) );
  TRY( MPI_Comm_rank(comm, &rank) );
  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_(comm);
  } else {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(cp,1,viewer,2);
  }

  TRY( PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii) );
  if (!iascii) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by QPPF",((PetscObject)viewer)->type_name);

  TRY( PetscObjectName((PetscObject)cp) );
  TRY( PetscObjectPrintClassNamePrefixType((PetscObject)cp, viewer) );
  TRY( PetscViewerASCIIPushTab(viewer) );
  TRY( PetscViewerASCIIPrintf(viewer, "setup called:       %c\n", cp->setupcalled ? 'y' : 'n') );
  TRY( PetscViewerASCIIPrintf(viewer, "G has orth. rows e.:%c\n", cp->G_has_orthonormal_rows_explicitly ? 'y' : 'n') );
  TRY( PetscViewerASCIIPrintf(viewer, "G has orth. rows i.:%c\n", cp->G_has_orthonormal_rows_implicitly ? 'y' : 'n') );
  TRY( PetscViewerASCIIPrintf(viewer, "explicit:           %c\n", cp->explicitInv ? 'y' : 'n') );
  TRY( PetscViewerASCIIPrintf(viewer, "redundancy:         %d\n", cp->redundancy) );
  TRY( PetscViewerASCIIPrintf(viewer, "last conv. reason:  %d\n", cp->conv_GGtinvv) );
  TRY( PetscViewerASCIIPrintf(viewer, "cumulative #iter.:  %d\n", cp->it_GGtinvv) );

  TRY( PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO) );
  if (cp->explicitInv) {
    KSP ksp;
    PetscViewer scv;
    TRY( PetscObjectQuery((PetscObject)cp->GGtinv,"ksp",(PetscObject*)&ksp) );
    TRY( PetscObjectGetComm((PetscObject)ksp, &comm) );
    TRY( PetscViewerGetSubViewer(viewer, comm, &scv) );
    TRY( KSPViewBriefInfo(ksp, scv) );
    TRY( PetscViewerRestoreSubViewer(viewer, comm, &scv) );
  } else {
    if (cp->GGtinv) TRY( MatView(cp->GGtinv, viewer) );
  }
  if (cp->G)    TRY( MatPrintInfo(cp->G) );
  if (cp->Gt)   TRY( MatPrintInfo(cp->Gt) );
  TRY( PetscViewerPopFormat(viewer) );
  TRY( PetscViewerASCIIPopTab(viewer) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPPFDestroy"
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
  TRY( QPPFReset(*cp) );
  TRY( MatDestroy(&(*cp)->G) );
  TRY( PetscHeaderDestroy(cp) );
  PetscFunctionReturn(0);
}
