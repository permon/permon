
#include <permonaif.h>
#include <permon/private/permonimpl.h>

PetscBool FllopAIFInitializeCalled=PETSC_FALSE;

static QP aif_qp = NULL;
static QPS aif_qps = NULL;
static MPI_Comm aif_comm;
static PetscInt aif_base=0;
static PetscBool aif_feti=PETSC_FALSE, aif_setup_called=PETSC_FALSE;
static PetscLogStage aif_stage, aif_setup_stage, aif_solve_stage;

#undef __FUNCT__
#define __FUNCT__ "FllopAIFApplyBase_Private"
static PetscErrorCode FllopAIFApplyBase_Private(PetscBool coo, PetscInt nI,PetscInt Iarr[],PetscInt nJ,PetscInt Jarr[])
{
  PetscInt i;
  PetscFunctionBegin;
  if(!coo) nI -= 1;
  
  if (aif_base) {
      for (i=0;i<nI;i++) Iarr[i]-=aif_base;
      for (i=0;i<nJ;i++) Jarr[i]-=aif_base;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFMatCompleteFromUpperTriangular"
static PetscErrorCode FllopAIFMatCompleteFromUpperTriangular(Mat A, AIFMatSymmetry flg)
{
  PetscFunctionBegin;
  if (flg==AIF_MAT_SYM_SYMMETRIC) {
    TRY( MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE) );
    TRY( MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE) );
  }
  if (flg==AIF_MAT_SYM_UPPER_TRIANGULAR) {
    TRY( MatCompleteFromUpperTriangular(A) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFInitialize"
PetscErrorCode FllopAIFInitialize(int *argc, char ***args, const char rcfile[])
{
  PetscFunctionBegin;
  TRY( PermonInitialize(argc,args,rcfile,(char*)0) );
  TRY( FllopAIFInitializeInComm(PETSC_COMM_WORLD,argc,args,rcfile) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFInitializeInComm"
PetscErrorCode FllopAIFInitializeInComm(MPI_Comm comm, int *argc, char ***args, const char rcfile[])
{
  PetscFunctionBegin;
  if (FllopAIFInitializeCalled) { PetscFunctionReturn(0); }
  aif_comm = comm;
  TRY( PermonInitialize(argc,args,rcfile,(char*)0) );
  TRY( FllopAIFReset() );
  TRY( PetscLogStageRegister("FllopAIF  Main", &aif_stage) );
  TRY( PetscLogStageRegister("FllopAIF Setup", &aif_setup_stage) );
  TRY( PetscLogStageRegister("FllopAIF Solve", &aif_solve_stage) );
  TRY( PetscLogStagePush(aif_stage) );
  FllopAIFInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFReset"
PetscErrorCode FllopAIFReset()
{
  PetscFunctionBegin;
  TRY( QPDestroy(&aif_qp) );
  TRY( QPSDestroy(&aif_qps) );
  TRY( QPCreate(aif_comm,&aif_qp) );
  TRY( QPSCreate(aif_comm,&aif_qps) );
  TRY( QPSSetQP(aif_qps,aif_qp) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFFinalize"
PetscErrorCode FllopAIFFinalize()
{
  PetscFunctionBegin;
  if (!FllopAIFInitializeCalled) PetscFunctionReturn(0);
  TRY( QPDestroy(&aif_qp) );
  TRY( QPSDestroy(&aif_qps) );
  TRY( PetscLogStagePop() );
  TRY( PermonFinalize() );
  aif_comm=0; aif_base=0; aif_feti=0; aif_stage=0; aif_setup_stage=0; aif_solve_stage=0; 
  FllopAIFInitializeCalled = PETSC_FALSE;
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFGetQP"
PetscErrorCode FllopAIFGetQP(QP *qp)
{
  PetscFunctionBegin;
  *qp = aif_qp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFGetQPS"
PetscErrorCode FllopAIFGetQPS(QPS *qps)
{
  PetscFunctionBegin;
  *qps = aif_qps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetSolutionVector"
PetscErrorCode FllopAIFSetSolutionVector(PetscInt n,PetscReal *x,const char *name)
{
  Vec x_g;
  
  PetscFunctionBegin;
  TRY( VecCreateMPIWithArray(aif_comm,1,n,PETSC_DECIDE,x,&x_g) );
  TRY( PetscObjectSetName((PetscObject)x_g,name) );
  TRY( QPSetInitialVector(aif_qp,x_g) );
  TRY( VecDestroy(&x_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetFETIOperator"
PetscErrorCode FllopAIFSetFETIOperator(PetscInt n,PetscInt *i,PetscInt *j,PetscScalar *A,AIFMatSymmetry symflg,const char *name)
{
  Mat A_l,A_g;
  PetscInt ni=n+1, nj=i[n];
  
  PetscFunctionBegin;
  aif_feti=PETSC_TRUE;
  TRY( FllopAIFApplyBase_Private(PETSC_FALSE,ni,i,nj,j) );
  
  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    TRY( MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,1,n,n,i,j,A,&A_l) );
  } else {
    TRY( MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,i,j,A,&A_l) );
  }
  TRY( FllopAIFMatCompleteFromUpperTriangular(A_l,symflg) );
  TRY( MatCreateBlockDiag(aif_comm,A_l,&A_g) );
  TRY( PetscObjectSetName((PetscObject)A_g,name) );
  TRY( FllopPetscObjectInheritName((PetscObject)A_l,(PetscObject)A_g,"_loc") );
  TRY( MatDestroy(&A_l) );
  
  TRY( QPSetOperator(aif_qp,A_g) );
  TRY( MatDestroy(&A_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetFETIOperatorMATIS"
PetscErrorCode FllopAIFSetFETIOperatorMATIS(PetscInt n,PetscInt N,PetscInt *i,PetscInt *j,PetscScalar *A,AIFMatSymmetry symflg,IS l2g,const char *name)
{
  Mat A_l,A_g;
  Vec x,b,x_new,b_new;
  PetscInt ni=n+1,nj=i[n];
  PetscScalar zero=0.0;
  ISLocalToGlobalMapping l2gmap;
  
  PetscFunctionBegin;
  TRY( QPGetRhs(aif_qp,&b) );
  TRY( QPGetSolutionVector(aif_qp,&x) );
  if (!x || !b) FLLOP_SETERRQ(aif_comm,PETSC_ERR_SUP,"x and b has to be set before operator");

  aif_feti=PETSC_TRUE;
  TRY( FllopAIFApplyBase_Private(PETSC_FALSE,ni,i,nj,j) );
  
  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    TRY( MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,1,n,n,i,j,A,&A_l) );
  } else {
    TRY( MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,i,j,A,&A_l) );
  }
  TRY( FllopAIFMatCompleteFromUpperTriangular(A_l,symflg) );
  TRY( ISLocalToGlobalMappingCreateIS(l2g,&l2gmap) );
  TRY( MatCreateIS(aif_comm,1,PETSC_DECIDE,PETSC_DECIDE,N,N,l2gmap,l2gmap,&A_g) );
  TRY( ISLocalToGlobalMappingDestroy(&l2gmap) );
  TRY( MatISSetLocalMat(A_g,A_l) );
  TRY( MatAssemblyBegin(A_g,MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(A_g,MAT_FINAL_ASSEMBLY) );
  TRY( PetscObjectSetName((PetscObject)A_g,name) );
  TRY( FllopPetscObjectInheritName((PetscObject)A_l,(PetscObject)A_g,"_loc") );
  TRY( MatDestroy(&A_l) );

  Mat_IS *matis  = (Mat_IS*)A_g->data;
  TRY( MatCreateVecs(A_g,&x_new,&b_new) );
  TRY( VecGetLocalVector(x,matis->x) );
  //TRY( VecSet(x_new,zero) );
  TRY( VecScatterBegin(matis->rctx,matis->x,x_new,INSERT_VALUES,SCATTER_REVERSE) );
  TRY( VecScatterEnd(matis->rctx,matis->x,x_new,INSERT_VALUES,SCATTER_REVERSE) );
  TRY( VecRestoreLocalVector(x,matis->x) );
  TRY( VecGetLocalVector(b,matis->y) );
  TRY( VecSet(b_new,zero) );
  TRY( VecScatterBegin(matis->rctx,matis->y,b_new,ADD_VALUES,SCATTER_REVERSE) );
  TRY( VecScatterEnd(matis->rctx,matis->y,b_new,ADD_VALUES,SCATTER_REVERSE) );
  TRY( VecRestoreLocalVector(b,matis->y) );

  TRY( PetscObjectCompose((PetscObject)b_new,"b_decomp",(PetscObject)b) );
  TRY( PetscObjectCompose((PetscObject)x_new,"x_decomp",(PetscObject)x) );
  TRY( QPSetInitialVector(aif_qp,x_new) );
  TRY( QPSetRhs(aif_qp,b_new) );
  TRY( QPSetOperator(aif_qp,A_g) );
  TRY( QPTMatISToBlockDiag(aif_qp) );
  TRY( QPGetChild(aif_qp,&aif_qp) );

  TRY( VecDestroy(&x_new) );
  TRY( VecDestroy(&b_new) );
  TRY( MatDestroy(&A_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetFETIOperatorNullspace"
PetscErrorCode FllopAIFSetFETIOperatorNullspace(PetscInt n,PetscInt d,PetscScalar *R,const char *name)
{
  Mat R_l,R_g;
  
  PetscFunctionBegin;
  aif_feti=PETSC_TRUE;
  TRY( MatCreateSeqDense(PETSC_COMM_SELF,n,d,R,&R_l) );
    
  TRY( MatCreateBlockDiag(aif_comm,R_l,&R_g) );
  TRY( PetscObjectSetName((PetscObject)R_g,name) );
  TRY( FllopPetscObjectInheritName((PetscObject)R_l,(PetscObject)R_g,"_loc") );
  TRY( MatDestroy(&R_l) );
  
  TRY( QPSetOperatorNullSpace(aif_qp,R_g) );
  TRY( MatDestroy(&R_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetOperatorByStripes"
PetscErrorCode FllopAIFSetOperatorByStripes(PetscInt m,PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,const char *name)
{
  Mat A_g;
  PetscInt ni=n+1, nj=i[n];
  
  PetscFunctionBegin;
  TRY( FllopAIFApplyBase_Private(PETSC_FALSE, ni,i, nj, j) );
  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    TRY( MatCreateMPISBAIJWithArrays(aif_comm,1,m,n,PETSC_DECIDE,N,i,j,A,&A_g) );
  } else {
    TRY( MatCreateMPIAIJWithArrays(aif_comm,m,n,PETSC_DECIDE,N,i,j,A,&A_g) );
  }
  TRY( FllopAIFMatCompleteFromUpperTriangular(A_g,symflg) );
  TRY( PetscObjectSetName((PetscObject)A_g,name) );
  TRY( QPSetOperator(aif_qp,A_g) );
  TRY( MatDestroy(&A_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetRhs"
PetscErrorCode FllopAIFSetRhs(PetscInt n,PetscScalar *b,const char *name)
{
  Vec b_g;
  
  PetscFunctionBegin;
  TRY( VecCreateMPIWithArray(aif_comm,1,n,PETSC_DECIDE,b,&b_g) );
  TRY( PetscObjectSetName((PetscObject)b_g,name) );
  TRY( QPSetRhs(aif_qp,b_g) );
  TRY( VecDestroy(&b_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFCreateLinearConstraints_Private"
static PetscErrorCode FllopAIFCreateLinearConstraints_Private(PetscBool coo,PetscInt m,PetscInt n,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname,Mat *B_new,Vec *c_new)
{
  Vec c_g;
  Mat B_l, B_g, tmat;
  PetscInt ni, nj;
  Vec column_layout;
  
  PetscFunctionBegin;
  c_g=NULL;
  
  if (coo) {
    ni=Bnnz+1;
    nj=Bnnz;
  } else {
    ni=m+1;
    nj=Bi[m];
  }
  TRY( FllopAIFApplyBase_Private(coo, ni,Bi,nj,Bj) );
  
  if (cv) {
    PetscValidPointer(cv,8);
    TRY( VecCreateMPIWithArray(aif_comm,1,m,PETSC_DECIDE,cv,&c_g) );
    TRY( PetscObjectSetName((PetscObject)c_g,cname) );
  }
  
  if (coo && B_dist_horizontal) {
    /* no need for transpose, just swap the meaning of rows and columns */
    PetscInt t,*tp;
    t=m; m=n; n=t;
    tp=Bi; Bi=Bj; Bj=tp;
  }

  if (coo) {
    TRY( MatCreateSeqAIJFromTriple(PETSC_COMM_SELF,m,n,Bi,Bj,Bv,&B_l,Bnnz,0) );
  } else {
    TRY( MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,m,n,Bi,Bj,Bv,&B_l) );
  }

  if (!coo && B_dist_horizontal) {
    tmat = B_l;
    TRY( PermonMatTranspose(tmat, MAT_TRANSPOSE_EXPLICIT, &B_l) );
    TRY( MatDestroy(&tmat) );
  }

  if (B_dist_horizontal != B_trans) { /* this means B is stored in transposed form, distributed across longer side */
    column_layout = NULL;
  } else {
    TRY( QPGetRhs(aif_qp, &column_layout) );
    FLLOP_ASSERT(column_layout,"RHS specified");
  }
  
  TRY( MatMergeAndDestroy(aif_comm,&B_l,column_layout,&B_g) );
  
  if (B_dist_horizontal != B_trans) {
    tmat = B_g;
    TRY( PermonMatTranspose(tmat, MAT_TRANSPOSE_CHEAPEST, &B_g) );
    TRY( PetscObjectSetName((PetscObject)B_g, Bname) );
    TRY( FllopPetscObjectInheritName((PetscObject)tmat,(PetscObject)B_g,"_T") );
    TRY( MatDestroy(&tmat) );
  } else {
    TRY( PetscObjectSetName((PetscObject)B_g,Bname) );
  } 
  
  *B_new = B_g;
  *c_new = c_g;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetIneq"
PetscErrorCode FllopAIFSetIneq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp,N,2);
  PetscValidLogicalCollectiveBool(aif_qp,B_trans,3);
  PetscValidLogicalCollectiveBool(aif_qp,B_dist_horizontal,4);
  PetscValidIntPointer(Bi,5);
  PetscValidIntPointer(Bj,6);
  PetscValidIntPointer(Bv,7);

  TRY( FllopAIFCreateLinearConstraints_Private(PETSC_FALSE,m,N,B_trans,B_dist_horizontal,Bi,Bj,Bv,-1,Bname,cv,cname,&B,&c) );
  TRY( QPSetIneq(aif_qp,B,c) );
  TRY( VecDestroy(&c) );
  TRY( MatDestroy(&B) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetEq"
PetscErrorCode FllopAIFSetEq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp,N,2);
  PetscValidLogicalCollectiveBool(aif_qp,B_trans,3);
  PetscValidLogicalCollectiveBool(aif_qp,B_dist_horizontal,4);
  PetscValidIntPointer(Bi,5);
  PetscValidIntPointer(Bj,6);
  PetscValidIntPointer(Bv,7);

  TRY( FllopAIFCreateLinearConstraints_Private(PETSC_FALSE,m,N,B_trans,B_dist_horizontal,Bi,Bj,Bv,-1,Bname,cv,cname,&B,&c) );
  TRY( QPSetEq(aif_qp,B,c) );
  TRY( VecDestroy(&c) );
  TRY( MatDestroy(&B) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFAddEq"
PetscErrorCode FllopAIFAddEq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp,N,2);
  PetscValidLogicalCollectiveBool(aif_qp,B_trans,3);
  PetscValidLogicalCollectiveBool(aif_qp,B_dist_horizontal,4);
  PetscValidIntPointer(Bi,5);
  PetscValidIntPointer(Bj,6);
  PetscValidIntPointer(Bv,7);

  TRY( FllopAIFCreateLinearConstraints_Private(PETSC_FALSE,m,N,B_trans,B_dist_horizontal,Bi,Bj,Bv,-1,Bname,cv,cname,&B,&c) );
  TRY( QPAddEq(aif_qp,B,c) );
  TRY( VecDestroy(&c) );
  TRY( MatDestroy(&B) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetIneqCOO"
PetscErrorCode FllopAIFSetIneqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp,N,2);
  PetscValidLogicalCollectiveBool(aif_qp,B_trans,3);
  PetscValidLogicalCollectiveBool(aif_qp,B_dist_horizontal,4);
  PetscValidIntPointer(Bi,5);
  PetscValidIntPointer(Bj,6);
  PetscValidIntPointer(Bv,7);

  TRY( FllopAIFCreateLinearConstraints_Private(PETSC_TRUE,m,N,B_trans,B_dist_horizontal,Bi,Bj,Bv,Bnnz,Bname,cv,cname,&B,&c) );
  TRY( QPSetIneq(aif_qp,B,c) );
  TRY( VecDestroy(&c) );
  TRY( MatDestroy(&B) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetEqCOO"
PetscErrorCode FllopAIFSetEqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp,N,2);
  PetscValidLogicalCollectiveBool(aif_qp,B_trans,3);
  PetscValidLogicalCollectiveBool(aif_qp,B_dist_horizontal,4);
  PetscValidIntPointer(Bi,5);
  PetscValidIntPointer(Bj,6);
  PetscValidIntPointer(Bv,7);

  TRY( FllopAIFCreateLinearConstraints_Private(PETSC_TRUE,m,N,B_trans,B_dist_horizontal,Bi,Bj,Bv,Bnnz,Bname,cv,cname,&B,&c) );
  TRY( QPSetEq(aif_qp,B,c) );
  TRY( VecDestroy(&c) );
  TRY( MatDestroy(&B) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFAddEqCOO"
PetscErrorCode FllopAIFAddEqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp,N,2);
  PetscValidLogicalCollectiveBool(aif_qp,B_trans,3);
  PetscValidLogicalCollectiveBool(aif_qp,B_dist_horizontal,4);
  PetscValidIntPointer(Bi,5);
  PetscValidIntPointer(Bj,6);
  PetscValidIntPointer(Bv,7);

  TRY( FllopAIFCreateLinearConstraints_Private(PETSC_TRUE,m,N,B_trans,B_dist_horizontal,Bi,Bj,Bv,Bnnz,Bname,cv,cname,&B,&c) );
  TRY( QPAddEq(aif_qp,B,c) );
  TRY( VecDestroy(&c) );
  TRY( MatDestroy(&B) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetType"
PetscErrorCode FllopAIFSetType(const char type[])
{
  PetscFunctionBegin;  
  TRY( QPSSetType(aif_qps,type) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetDefaultType"
PetscErrorCode FllopAIFSetDefaultType()
{
  PetscFunctionBegin;
  TRY( QPSSetDefaultType(aif_qps) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);   
}


#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetBox"
PetscErrorCode FllopAIFSetBox(PetscInt n,PetscScalar *lb,const char *lbname,PetscScalar *ub,const char *ubname)
{
  Vec lb_g, ub_g;
  
  PetscFunctionBegin;
  lb_g=NULL; ub_g=NULL;
  
  if (lb) {
    TRY( VecCreateMPIWithArray(aif_comm,1,n,PETSC_DECIDE,lb,&lb_g) );
    TRY( PetscObjectSetName((PetscObject)lb_g,lbname) );
  }
  
  if (ub) {
    TRY( VecCreateMPIWithArray(aif_comm,1,n,PETSC_DECIDE,ub,&ub_g) );
    TRY( PetscObjectSetName((PetscObject)ub_g,ubname) );
  }
  
  TRY( QPSetBox(aif_qp,NULL,lb_g,ub_g) );
  TRY( VecDestroy(&lb_g) );
  TRY( VecDestroy(&ub_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetArrayBase"
PetscErrorCode FllopAIFSetArrayBase(PetscInt base)
{
  PetscFunctionBegin;
  aif_base = base;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFEnforceEqByProjector"
PetscErrorCode FllopAIFEnforceEqByProjector()
{
  PetscFunctionBegin;
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPTEnforceEqByProjector(aif_qp) );
  TRY( PetscLogStagePop() );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFEnforceEqByPenalty"
PetscErrorCode FllopAIFEnforceEqByPenalty(PetscReal rho)
{
  PetscFunctionBegin;
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPTEnforceEqByPenalty(aif_qp,rho,PETSC_FALSE) );
  TRY( PetscLogStagePop() );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFHomogenizeEq"
PetscErrorCode FllopAIFHomogenizeEq()
{
  PetscFunctionBegin;
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPTHomogenizeEq(aif_qp) );
  TRY( PetscLogStagePop() );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFDualize"
PetscErrorCode FllopAIFDualize(MatRegularizationType regtype)
{
  PetscFunctionBegin;
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPTDualize(aif_qp,(MatInvType) aif_feti,regtype) );
  TRY( PetscLogStagePop() );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFFromOptions"
PetscErrorCode FllopAIFFromOptions()
{
  PetscFunctionBegin;
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPTFromOptions(aif_qp) );
  TRY( PetscLogStagePop() );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFOperatorShift"
PetscErrorCode FllopAIFOperatorShift(PetscScalar a)
{
  Mat A,Aloc;
  PetscFunctionBegin;
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPGetOperator(aif_qp,&A) );
  TRY( MatGetDiagonalBlock(A,&Aloc) );
  TRY( MatShift(Aloc,a) );
  TRY( PetscLogStagePop() );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetUp"
PetscErrorCode FllopAIFSetUp()
{
  PetscFunctionBegin;
  if (aif_setup_called) PetscFunctionReturn(0);
  TRY( PetscLogStagePush(aif_setup_stage) );
  TRY( QPSSetFromOptions(aif_qps) );
  TRY( QPSSetUp(aif_qps) );
  aif_setup_called = PETSC_TRUE;
  TRY( PetscLogStagePop() );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFKSPSolveMATIS"
PetscErrorCode FllopAIFKSPSolveMATIS(IS isDir,PetscInt n,PetscInt N,PetscInt *i,PetscInt *j,PetscScalar *A,AIFMatSymmetry symflg,IS l2g,const char *name)
{
  Mat A_l,A_g;
  Vec x,b,x_new,b_new;
  PetscInt ni=n+1,nj=i[n];
  PetscScalar zero=0.0;
  ISLocalToGlobalMapping l2gmap;
  
  PetscFunctionBegin;
  TRY( QPGetRhs(aif_qp,&b) );
  TRY( QPGetSolutionVector(aif_qp,&x) );
  if (!x || !b) FLLOP_SETERRQ(aif_comm,PETSC_ERR_SUP,"x and b has to be set before operator");

  aif_feti=PETSC_TRUE;
  TRY( FllopAIFApplyBase_Private(PETSC_FALSE,ni,i,nj,j) );
  
  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    TRY( MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF,1,n,n,i,j,A,&A_l) );
  } else {
    TRY( MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,i,j,A,&A_l) );
  }
  TRY( FllopAIFMatCompleteFromUpperTriangular(A_l,symflg) );
  TRY( ISLocalToGlobalMappingCreateIS(l2g,&l2gmap) );
  TRY( MatCreateIS(aif_comm,1,PETSC_DECIDE,PETSC_DECIDE,N,N,l2gmap,l2gmap,&A_g) );
  TRY( ISLocalToGlobalMappingDestroy(&l2gmap) );
  TRY( MatISSetLocalMat(A_g,A_l) );
  TRY( MatAssemblyBegin(A_g,MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(A_g,MAT_FINAL_ASSEMBLY) );
  TRY( PetscObjectSetName((PetscObject)A_g,name) );
  TRY( FllopPetscObjectInheritName((PetscObject)A_l,(PetscObject)A_g,"_loc") );
  TRY( MatDestroy(&A_l) );

  Mat_IS *matis  = (Mat_IS*)A_g->data;
  TRY( MatCreateVecs(A_g,&x_new,&b_new) );
  TRY( VecGetLocalVector(x,matis->x) );
  //TRY( VecSet(x_new,zero) );
  TRY( VecScatterBegin(matis->rctx,matis->x,x_new,INSERT_VALUES,SCATTER_REVERSE) );
  TRY( VecScatterEnd(matis->rctx,matis->x,x_new,INSERT_VALUES,SCATTER_REVERSE) );
  TRY( VecRestoreLocalVector(x,matis->x) );
  TRY( VecGetLocalVector(b,matis->y) );
  TRY( VecSet(b_new,zero) );
  TRY( VecScatterBegin(matis->rctx,matis->y,b_new,ADD_VALUES,SCATTER_REVERSE) );
  TRY( VecScatterEnd(matis->rctx,matis->y,b_new,ADD_VALUES,SCATTER_REVERSE) );
  TRY( VecRestoreLocalVector(b,matis->y) );

  TRY( PetscObjectCompose((PetscObject)b_new,"b_decomp",(PetscObject)b) );
  TRY( PetscObjectCompose((PetscObject)x_new,"x_decomp",(PetscObject)x) );
  
  KSP ksp;
  TRY( KSPCreate(aif_comm,&ksp) );
  TRY( KSPSetOperators(ksp,A_g,A_g) );
  TRY( KSPSetFromOptions(ksp) );
  TRY( KSPFETISetDirichlet(ksp,isDir,FETI_LOCAL,PETSC_TRUE) );
  TRY( KSPSolve(ksp,b_new,x_new) );

  TRY( VecDestroy(&x_new) );
  TRY( VecDestroy(&b_new) );
  TRY( MatDestroy(&A_g) );
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSolve"
PetscErrorCode FllopAIFSolve()
{
  PetscInt test=0;

  PetscFunctionBegin;
  TRY( PetscOptionsGetInt(NULL,NULL,"-aif_test",&test,NULL) );
  TRY( PetscLogStagePush(aif_solve_stage) );
  switch (test) {
    case 1: {
      TRY( PetscPrintf(PETSC_COMM_WORLD, "Hello world!\n") );
      break;
    }
    case 0:
    default:
      TRY( FllopAIFSetUp() );
      TRY( QPSSolve(aif_qps) );
  }
  TRY( PetscLogStagePop() );
  PetscFunctionReturn(0);
}
