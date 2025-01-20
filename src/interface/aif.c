
#include <permonaif.h>
#include <permon/private/permonimpl.h>

PetscBool FllopAIFInitializeCalled = PETSC_FALSE;

static QP            aif_qp  = NULL;
static QPS           aif_qps = NULL;
static MPI_Comm      aif_comm;
static PetscInt      aif_base = 0;
static PetscBool     aif_feti = PETSC_FALSE, aif_setup_called = PETSC_FALSE;
static PetscLogStage aif_stage, aif_setup_stage, aif_solve_stage;

#undef __FUNCT__
#define __FUNCT__ "FllopAIFApplyBase_Private"
static PetscErrorCode FllopAIFApplyBase_Private(PetscBool coo, PetscInt nI, PetscInt Iarr[], PetscInt nJ, PetscInt Jarr[])
{
  PetscInt i;
  PetscFunctionBegin;
  if (!coo) nI -= 1;

  if (aif_base) {
    for (i = 0; i < nI; i++) Iarr[i] -= aif_base;
    for (i = 0; i < nJ; i++) Jarr[i] -= aif_base;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFMatCompleteFromUpperTriangular"
static PetscErrorCode FllopAIFMatCompleteFromUpperTriangular(Mat A, AIFMatSymmetry flg)
{
  PetscFunctionBegin;
  if (flg == AIF_MAT_SYM_SYMMETRIC) {
    PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  }
  if (flg == AIF_MAT_SYM_UPPER_TRIANGULAR) { PetscCall(MatCompleteFromUpperTriangular(A)); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFInitialize"
PetscErrorCode FllopAIFInitialize(int *argc, char ***args, const char rcfile[])
{
  PetscFunctionBegin;
  PetscCall(PermonInitialize(argc, args, rcfile, (char *)0));
  PetscCall(FllopAIFInitializeInComm(PETSC_COMM_WORLD, argc, args, rcfile));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFInitializeInComm"
PetscErrorCode FllopAIFInitializeInComm(MPI_Comm comm, int *argc, char ***args, const char rcfile[])
{
  PetscFunctionBegin;
  if (FllopAIFInitializeCalled) { PetscFunctionReturn(0); }
  aif_comm = comm;
  PetscCall(PermonInitialize(argc, args, rcfile, (char *)0));
  PetscCall(FllopAIFReset());
  PetscCall(PetscLogStageRegister("FllopAIF  Main", &aif_stage));
  PetscCall(PetscLogStageRegister("FllopAIF Setup", &aif_setup_stage));
  PetscCall(PetscLogStageRegister("FllopAIF Solve", &aif_solve_stage));
  PetscCall(PetscLogStagePush(aif_stage));
  FllopAIFInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFReset"
PetscErrorCode FllopAIFReset()
{
  PetscFunctionBegin;
  PetscCall(QPDestroy(&aif_qp));
  PetscCall(QPSDestroy(&aif_qps));
  PetscCall(QPCreate(aif_comm, &aif_qp));
  PetscCall(QPSCreate(aif_comm, &aif_qps));
  PetscCall(QPSSetQP(aif_qps, aif_qp));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFFinalize"
PetscErrorCode FllopAIFFinalize()
{
  PetscFunctionBegin;
  if (!FllopAIFInitializeCalled) PetscFunctionReturn(0);
  PetscCall(QPDestroy(&aif_qp));
  PetscCall(QPSDestroy(&aif_qps));
  PetscCall(PetscLogStagePop());
  PetscCall(PermonFinalize());
  aif_comm                 = 0;
  aif_base                 = 0;
  aif_feti                 = 0;
  aif_stage                = 0;
  aif_setup_stage          = 0;
  aif_solve_stage          = 0;
  FllopAIFInitializeCalled = PETSC_FALSE;
  aif_setup_called         = PETSC_FALSE;
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
PetscErrorCode FllopAIFSetSolutionVector(PetscInt n, PetscReal *x, const char *name)
{
  Vec x_g;

  PetscFunctionBegin;
  PetscCall(VecCreateMPIWithArray(aif_comm, 1, n, PETSC_DECIDE, x, &x_g));
  PetscCall(PetscObjectSetName((PetscObject)x_g, name));
  PetscCall(QPSetInitialVector(aif_qp, x_g));
  PetscCall(VecDestroy(&x_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetFETIOperator"
PetscErrorCode FllopAIFSetFETIOperator(PetscInt n, PetscInt *i, PetscInt *j, PetscScalar *A, AIFMatSymmetry symflg, const char *name)
{
  Mat      A_l, A_g;
  PetscInt ni = n + 1, nj = i[n];

  PetscFunctionBegin;
  aif_feti = PETSC_TRUE;
  PetscCall(FllopAIFApplyBase_Private(PETSC_FALSE, ni, i, nj, j));

  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    PetscCall(MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF, 1, n, n, i, j, A, &A_l));
  } else {
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, i, j, A, &A_l));
  }
  PetscCall(FllopAIFMatCompleteFromUpperTriangular(A_l, symflg));
  PetscCall(MatCreateBlockDiag(aif_comm, A_l, &A_g));
  PetscCall(PetscObjectSetName((PetscObject)A_g, name));
  PetscCall(PermonPetscObjectInheritName((PetscObject)A_l, (PetscObject)A_g, "_loc"));
  PetscCall(MatDestroy(&A_l));

  PetscCall(QPSetOperator(aif_qp, A_g));
  PetscCall(MatDestroy(&A_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetFETIOperatorMATIS"
PetscErrorCode FllopAIFSetFETIOperatorMATIS(PetscInt n, PetscInt N, PetscInt *i, PetscInt *j, PetscScalar *A, AIFMatSymmetry symflg, IS l2g, const char *name)
{
  Mat                    A_l, A_g;
  Vec                    x, b, x_new, b_new;
  PetscInt               ni = n + 1, nj = i[n];
  PetscScalar            zero = 0.0;
  ISLocalToGlobalMapping l2gmap;

  PetscFunctionBegin;
  PetscCall(QPGetRhs(aif_qp, &b));
  PetscCall(QPGetSolutionVector(aif_qp, &x));
  if (!x || !b) SETERRQ(aif_comm, PETSC_ERR_SUP, "x and b has to be set before operator");

  aif_feti = PETSC_TRUE;
  PetscCall(FllopAIFApplyBase_Private(PETSC_FALSE, ni, i, nj, j));

  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    PetscCall(MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF, 1, n, n, i, j, A, &A_l));
  } else {
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, i, j, A, &A_l));
  }
  PetscCall(FllopAIFMatCompleteFromUpperTriangular(A_l, symflg));
  PetscCall(ISLocalToGlobalMappingCreateIS(l2g, &l2gmap));
  PetscCall(MatCreateIS(aif_comm, 1, PETSC_DECIDE, PETSC_DECIDE, N, N, l2gmap, l2gmap, &A_g));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2gmap));
  PetscCall(MatISSetLocalMat(A_g, A_l));
  PetscCall(MatAssemblyBegin(A_g, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_g, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)A_g, name));
  PetscCall(PermonPetscObjectInheritName((PetscObject)A_l, (PetscObject)A_g, "_loc"));
  PetscCall(MatDestroy(&A_l));

  Mat_IS *matis = (Mat_IS *)A_g->data;
  PetscCall(MatCreateVecs(A_g, &x_new, &b_new));
  PetscCall(VecGetLocalVector(x, matis->x));
  //PetscCall(VecSet(x_new,zero));
  PetscCall(VecScatterBegin(matis->rctx, matis->x, x_new, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(matis->rctx, matis->x, x_new, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecRestoreLocalVector(x, matis->x));
  PetscCall(VecGetLocalVector(b, matis->y));
  PetscCall(VecSet(b_new, zero));
  PetscCall(VecScatterBegin(matis->rctx, matis->y, b_new, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(matis->rctx, matis->y, b_new, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecRestoreLocalVector(b, matis->y));

  PetscCall(PetscObjectCompose((PetscObject)b_new, "b_decomp", (PetscObject)b));
  PetscCall(PetscObjectCompose((PetscObject)x_new, "x_decomp", (PetscObject)x));
  PetscCall(QPSetInitialVector(aif_qp, x_new));
  PetscCall(QPSetRhs(aif_qp, b_new));
  PetscCall(QPSetOperator(aif_qp, A_g));
  PetscCall(QPTMatISToBlockDiag(aif_qp));
  PetscCall(QPGetChild(aif_qp, &aif_qp));

  PetscCall(VecDestroy(&x_new));
  PetscCall(VecDestroy(&b_new));
  PetscCall(MatDestroy(&A_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetFETIOperatorNullspace"
PetscErrorCode FllopAIFSetFETIOperatorNullspace(PetscInt n, PetscInt d, PetscScalar *R, const char *name)
{
  Mat R_l, R_g;

  PetscFunctionBegin;
  aif_feti = PETSC_TRUE;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, d, R, &R_l));

  PetscCall(MatCreateBlockDiag(aif_comm, R_l, &R_g));
  PetscCall(PetscObjectSetName((PetscObject)R_g, name));
  PetscCall(PermonPetscObjectInheritName((PetscObject)R_l, (PetscObject)R_g, "_loc"));
  PetscCall(MatDestroy(&R_l));

  PetscCall(QPSetOperatorNullSpace(aif_qp, R_g));
  PetscCall(MatDestroy(&R_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetOperatorByStripes"
PetscErrorCode FllopAIFSetOperatorByStripes(PetscInt m, PetscInt n, PetscInt N, PetscInt *i, PetscInt *j, PetscScalar *A, AIFMatSymmetry symflg, const char *name)
{
  Mat      A_g;
  PetscInt ni = n + 1, nj = i[n];

  PetscFunctionBegin;
  PetscCall(FllopAIFApplyBase_Private(PETSC_FALSE, ni, i, nj, j));
  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    PetscCall(MatCreateMPISBAIJWithArrays(aif_comm, 1, m, n, PETSC_DECIDE, N, i, j, A, &A_g));
  } else {
    PetscCall(MatCreateMPIAIJWithArrays(aif_comm, m, n, PETSC_DECIDE, N, i, j, A, &A_g));
  }
  PetscCall(FllopAIFMatCompleteFromUpperTriangular(A_g, symflg));
  PetscCall(PetscObjectSetName((PetscObject)A_g, name));
  PetscCall(QPSetOperator(aif_qp, A_g));
  PetscCall(MatDestroy(&A_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetRhs"
PetscErrorCode FllopAIFSetRhs(PetscInt n, PetscScalar *b, const char *name)
{
  Vec b_g;

  PetscFunctionBegin;
  PetscCall(VecCreateMPIWithArray(aif_comm, 1, n, PETSC_DECIDE, b, &b_g));
  PetscCall(PetscObjectSetName((PetscObject)b_g, name));
  PetscCall(QPSetRhs(aif_qp, b_g));
  PetscCall(VecDestroy(&b_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFCreateLinearConstraints_Private"
static PetscErrorCode FllopAIFCreateLinearConstraints_Private(PetscBool coo, PetscInt m, PetscInt n, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, PetscInt Bnnz, const char *Bname, PetscScalar *cv, const char *cname, Mat *B_new, Vec *c_new)
{
  Vec      c_g;
  Mat      B_l, B_g, tmat;
  PetscInt ni, nj;
  Vec      column_layout;

  PetscFunctionBegin;
  c_g = NULL;

  if (coo) {
    ni = Bnnz + 1;
    nj = Bnnz;
  } else {
    ni = m + 1;
    nj = Bi[m];
  }
  PetscCall(FllopAIFApplyBase_Private(coo, ni, Bi, nj, Bj));

  if (cv) {
    PetscAssertPointer(cv, 8);
    PetscCall(VecCreateMPIWithArray(aif_comm, 1, m, PETSC_DECIDE, cv, &c_g));
    PetscCall(PetscObjectSetName((PetscObject)c_g, cname));
  }

  if (coo && B_dist_horizontal) {
    /* no need for transpose, just swap the meaning of rows and columns */
    PetscInt t, *tp;
    t  = m;
    m  = n;
    n  = t;
    tp = Bi;
    Bi = Bj;
    Bj = tp;
  }

  if (coo) {
    PetscCall(MatCreateSeqAIJFromTriple(PETSC_COMM_SELF, m, n, Bi, Bj, Bv, &B_l, Bnnz, 0));
  } else {
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, Bi, Bj, Bv, &B_l));
  }

  if (!coo && B_dist_horizontal) {
    tmat = B_l;
    PetscCall(PermonMatTranspose(tmat, MAT_TRANSPOSE_EXPLICIT, &B_l));
    PetscCall(MatDestroy(&tmat));
  }

  if (B_dist_horizontal != B_trans) { /* this means B is stored in transposed form, distributed across longer side */
    column_layout = NULL;
  } else {
    PetscCall(QPGetRhs(aif_qp, &column_layout));
    PERMON_ASSERT(column_layout, "RHS specified");
  }

  PetscCall(MatMergeAndDestroy(aif_comm, &B_l, column_layout, &B_g));

  if (B_dist_horizontal != B_trans) {
    tmat = B_g;
    PetscCall(PermonMatTranspose(tmat, MAT_TRANSPOSE_CHEAPEST, &B_g));
    PetscCall(PetscObjectSetName((PetscObject)B_g, Bname));
    PetscCall(PermonPetscObjectInheritName((PetscObject)tmat, (PetscObject)B_g, "_T"));
    PetscCall(MatDestroy(&tmat));
  } else {
    PetscCall(PetscObjectSetName((PetscObject)B_g, Bname));
  }

  *B_new = B_g;
  *c_new = c_g;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetIneq"
PetscErrorCode FllopAIFSetIneq(PetscInt m, PetscInt N, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, const char *Bname, PetscScalar *cv, const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp, N, 2);
  PetscValidLogicalCollectiveBool(aif_qp, B_trans, 3);
  PetscValidLogicalCollectiveBool(aif_qp, B_dist_horizontal, 4);
  PetscAssertPointer(Bi, 5);
  PetscAssertPointer(Bj, 6);
  PetscAssertPointer(Bv, 7);

  PetscCall(FllopAIFCreateLinearConstraints_Private(PETSC_FALSE, m, N, B_trans, B_dist_horizontal, Bi, Bj, Bv, -1, Bname, cv, cname, &B, &c));
  PetscCall(QPSetIneq(aif_qp, B, c));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&B));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetEq"
PetscErrorCode FllopAIFSetEq(PetscInt m, PetscInt N, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, const char *Bname, PetscScalar *cv, const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp, N, 2);
  PetscValidLogicalCollectiveBool(aif_qp, B_trans, 3);
  PetscValidLogicalCollectiveBool(aif_qp, B_dist_horizontal, 4);
  PetscAssertPointer(Bi, 5);
  PetscAssertPointer(Bj, 6);
  PetscAssertPointer(Bv, 7);

  PetscCall(FllopAIFCreateLinearConstraints_Private(PETSC_FALSE, m, N, B_trans, B_dist_horizontal, Bi, Bj, Bv, -1, Bname, cv, cname, &B, &c));
  PetscCall(QPSetEq(aif_qp, B, c));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&B));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFAddEq"
PetscErrorCode FllopAIFAddEq(PetscInt m, PetscInt N, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, const char *Bname, PetscScalar *cv, const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp, N, 2);
  PetscValidLogicalCollectiveBool(aif_qp, B_trans, 3);
  PetscValidLogicalCollectiveBool(aif_qp, B_dist_horizontal, 4);
  PetscAssertPointer(Bi, 5);
  PetscAssertPointer(Bj, 6);
  PetscAssertPointer(Bv, 7);

  PetscCall(FllopAIFCreateLinearConstraints_Private(PETSC_FALSE, m, N, B_trans, B_dist_horizontal, Bi, Bj, Bv, -1, Bname, cv, cname, &B, &c));
  PetscCall(QPAddEq(aif_qp, B, c));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&B));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetIneqCOO"
PetscErrorCode FllopAIFSetIneqCOO(PetscInt m, PetscInt N, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, PetscInt Bnnz, const char *Bname, PetscScalar *cv, const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp, N, 2);
  PetscValidLogicalCollectiveBool(aif_qp, B_trans, 3);
  PetscValidLogicalCollectiveBool(aif_qp, B_dist_horizontal, 4);
  PetscAssertPointer(Bi, 5);
  PetscAssertPointer(Bj, 6);
  PetscAssertPointer(Bv, 7);

  PetscCall(FllopAIFCreateLinearConstraints_Private(PETSC_TRUE, m, N, B_trans, B_dist_horizontal, Bi, Bj, Bv, Bnnz, Bname, cv, cname, &B, &c));
  PetscCall(QPSetIneq(aif_qp, B, c));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&B));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetEqCOO"
PetscErrorCode FllopAIFSetEqCOO(PetscInt m, PetscInt N, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, PetscInt Bnnz, const char *Bname, PetscScalar *cv, const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp, N, 2);
  PetscValidLogicalCollectiveBool(aif_qp, B_trans, 3);
  PetscValidLogicalCollectiveBool(aif_qp, B_dist_horizontal, 4);
  PetscAssertPointer(Bi, 5);
  PetscAssertPointer(Bj, 6);
  PetscAssertPointer(Bv, 7);

  PetscCall(FllopAIFCreateLinearConstraints_Private(PETSC_TRUE, m, N, B_trans, B_dist_horizontal, Bi, Bj, Bv, Bnnz, Bname, cv, cname, &B, &c));
  PetscCall(QPSetEq(aif_qp, B, c));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&B));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFAddEqCOO"
PetscErrorCode FllopAIFAddEqCOO(PetscInt m, PetscInt N, PetscBool B_trans, PetscBool B_dist_horizontal, PetscInt *Bi, PetscInt *Bj, PetscScalar *Bv, PetscInt Bnnz, const char *Bname, PetscScalar *cv, const char *cname)
{
  Mat B;
  Vec c;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(aif_qp, N, 2);
  PetscValidLogicalCollectiveBool(aif_qp, B_trans, 3);
  PetscValidLogicalCollectiveBool(aif_qp, B_dist_horizontal, 4);
  PetscAssertPointer(Bi, 5);
  PetscAssertPointer(Bj, 6);
  PetscAssertPointer(Bv, 7);

  PetscCall(FllopAIFCreateLinearConstraints_Private(PETSC_TRUE, m, N, B_trans, B_dist_horizontal, Bi, Bj, Bv, Bnnz, Bname, cv, cname, &B, &c));
  PetscCall(QPAddEq(aif_qp, B, c));
  PetscCall(VecDestroy(&c));
  PetscCall(MatDestroy(&B));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetType"
PetscErrorCode FllopAIFSetType(const char type[])
{
  PetscFunctionBegin;
  PetscCall(QPSSetType(aif_qps, type));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetDefaultType"
PetscErrorCode FllopAIFSetDefaultType()
{
  PetscFunctionBegin;
  PetscCall(QPSSetDefaultType(aif_qps));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetBox"
PetscErrorCode FllopAIFSetBox(PetscInt n, PetscScalar *lb, const char *lbname, PetscScalar *ub, const char *ubname)
{
  Vec lb_g, ub_g;

  PetscFunctionBegin;
  lb_g = NULL;
  ub_g = NULL;

  if (lb) {
    PetscCall(VecCreateMPIWithArray(aif_comm, 1, n, PETSC_DECIDE, lb, &lb_g));
    PetscCall(PetscObjectSetName((PetscObject)lb_g, lbname));
  }

  if (ub) {
    PetscCall(VecCreateMPIWithArray(aif_comm, 1, n, PETSC_DECIDE, ub, &ub_g));
    PetscCall(PetscObjectSetName((PetscObject)ub_g, ubname));
  }

  PetscCall(QPSetBox(aif_qp, NULL, lb_g, ub_g));
  PetscCall(VecDestroy(&lb_g));
  PetscCall(VecDestroy(&ub_g));
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
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPTEnforceEqByProjector(aif_qp));
  PetscCall(PetscLogStagePop());
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFEnforceEqByPenalty"
PetscErrorCode FllopAIFEnforceEqByPenalty(PetscReal rho)
{
  PetscFunctionBegin;
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPTEnforceEqByPenalty(aif_qp, rho, PETSC_FALSE));
  PetscCall(PetscLogStagePop());
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFHomogenizeEq"
PetscErrorCode FllopAIFHomogenizeEq()
{
  PetscFunctionBegin;
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPTHomogenizeEq(aif_qp));
  PetscCall(PetscLogStagePop());
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFDualize"
PetscErrorCode FllopAIFDualize(MatRegularizationType regtype)
{
  PetscFunctionBegin;
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPTDualize(aif_qp, (MatInvType)aif_feti, regtype));
  PetscCall(PetscLogStagePop());
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFFromOptions"
PetscErrorCode FllopAIFFromOptions()
{
  PetscFunctionBegin;
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPTFromOptions(aif_qp));
  PetscCall(PetscLogStagePop());
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFOperatorShift"
PetscErrorCode FllopAIFOperatorShift(PetscScalar a)
{
  Mat A, Aloc;
  PetscFunctionBegin;
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPGetOperator(aif_qp, &A));
  PetscCall(MatGetDiagonalBlock(A, &Aloc));
  PetscCall(MatShift(Aloc, a));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSetUp"
PetscErrorCode FllopAIFSetUp()
{
  PetscFunctionBegin;
  if (aif_setup_called) PetscFunctionReturn(0);
  PetscCall(PetscLogStagePush(aif_setup_stage));
  PetscCall(QPSSetFromOptions(aif_qps));
  PetscCall(QPSSetUp(aif_qps));
  aif_setup_called = PETSC_TRUE;
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFKSPSolveMATIS"
PetscErrorCode FllopAIFKSPSolveMATIS(IS isDir, PetscInt n, PetscInt N, PetscInt *i, PetscInt *j, PetscScalar *A, AIFMatSymmetry symflg, IS l2g, const char *name)
{
  Mat                    A_l, A_g;
  Vec                    x, b, x_new, b_new;
  PetscInt               ni = n + 1, nj = i[n];
  PetscScalar            zero = 0.0;
  ISLocalToGlobalMapping l2gmap;

  PetscFunctionBegin;
  PetscCall(QPGetRhs(aif_qp, &b));
  PetscCall(QPGetSolutionVector(aif_qp, &x));
  if (!x || !b) SETERRQ(aif_comm, PETSC_ERR_SUP, "x and b has to be set before operator");

  aif_feti = PETSC_TRUE;
  PetscCall(FllopAIFApplyBase_Private(PETSC_FALSE, ni, i, nj, j));

  if (symflg == AIF_MAT_SYM_UPPER_TRIANGULAR) {
    PetscCall(MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF, 1, n, n, i, j, A, &A_l));
  } else {
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, i, j, A, &A_l));
  }
  PetscCall(FllopAIFMatCompleteFromUpperTriangular(A_l, symflg));
  PetscCall(ISLocalToGlobalMappingCreateIS(l2g, &l2gmap));
  PetscCall(MatCreateIS(aif_comm, 1, PETSC_DECIDE, PETSC_DECIDE, N, N, l2gmap, l2gmap, &A_g));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2gmap));
  PetscCall(MatISSetLocalMat(A_g, A_l));
  PetscCall(MatAssemblyBegin(A_g, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_g, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)A_g, name));
  PetscCall(PermonPetscObjectInheritName((PetscObject)A_l, (PetscObject)A_g, "_loc"));
  PetscCall(MatDestroy(&A_l));

  Mat_IS *matis = (Mat_IS *)A_g->data;
  PetscCall(MatCreateVecs(A_g, &x_new, &b_new));
  PetscCall(VecGetLocalVector(x, matis->x));
  //PetscCall(VecSet(x_new,zero));
  PetscCall(VecScatterBegin(matis->rctx, matis->x, x_new, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(matis->rctx, matis->x, x_new, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecRestoreLocalVector(x, matis->x));
  PetscCall(VecGetLocalVector(b, matis->y));
  PetscCall(VecSet(b_new, zero));
  PetscCall(VecScatterBegin(matis->rctx, matis->y, b_new, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(matis->rctx, matis->y, b_new, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecRestoreLocalVector(b, matis->y));

  PetscCall(PetscObjectCompose((PetscObject)b_new, "b_decomp", (PetscObject)b));
  PetscCall(PetscObjectCompose((PetscObject)x_new, "x_decomp", (PetscObject)x));

  KSP ksp;
  PetscCall(KSPCreate(aif_comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A_g, A_g));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPFETISetDirichlet(ksp, isDir, FETI_LOCAL, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, b_new, x_new));

  PetscCall(VecDestroy(&x_new));
  PetscCall(VecDestroy(&b_new));
  PetscCall(MatDestroy(&A_g));
  aif_setup_called = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopAIFSolve"
PetscErrorCode FllopAIFSolve()
{
  PetscInt test = 0;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-aif_test", &test, NULL));
  PetscCall(PetscLogStagePush(aif_solve_stage));
  switch (test) {
  case 1: {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Hello world!\n"));
    break;
  }
  case 0:
  default:
    PetscCall(FllopAIFSetUp());
    PetscCall(QPSSolve(aif_qps));
  }
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(0);
}
