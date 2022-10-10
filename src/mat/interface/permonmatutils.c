
#include <permon/private/permonmatimpl.h>

PetscLogEvent Mat_GetMaxEigenvalue,Mat_FilterZeros,Mat_MergeAndDestroy,PermonMat_GetLocalMat;
PetscInt MatGetMaxEigenvalue_composed_id;

#undef __FUNCT__
#define __FUNCT__ "MatFactored"
PetscErrorCode MatFactored(Mat mat, PetscBool *flg)
{
  PetscFunctionBegin;
  *flg = (PetscBool) (mat->factortype != MAT_FACTOR_NONE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPrintInfo"
PetscErrorCode MatPrintInfo(Mat mat)
{
  PetscInt m, n, M, N, i, tablevel;
  PetscScalar fill;
  const char *name, *type, *prefix;
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscBool flg;
  Mat inmat;

  PetscFunctionBegin;
  if (!FllopObjectInfoEnabled) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscCall(PetscObjectGetTabLevel((PetscObject)mat, &tablevel));
  for (i=0; i<tablevel; i++) {
    PetscCall(PetscPrintf(comm, "  "));
  }

  if (mat == NULL) {
    PetscCall(PetscPrintf(comm, "Mat NULL\n"));
    PetscFunctionReturn(0);
  }

  PetscCall(MatGetSize(mat, &M, &N));
  PetscCall(MatGetLocalSize(mat, &m, &n));
  PetscCall(MatGetType(mat, &type));
  PetscCall(PetscObjectGetName((PetscObject) mat, &name));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) mat, &prefix));

  PetscCall(PetscPrintf(comm,
      "Mat %p %-16s %-10s %-10s sizes %d %d %d %d",
          (void*)mat, name, prefix, type,                m, n, M, N));

  if (mat->ops->getinfo) {
    MatInfo info;
    PetscInt sumnnz,sumalloc,maxnnz;

    PetscCall(MatGetInfo(mat, MAT_GLOBAL_SUM, &info));
    sumnnz  =  (PetscInt) info.nz_used;
    sumalloc = (PetscInt) info.nz_allocated;
    fill = ((PetscReal) sumnnz) / M / N;
    PetscCall(MatGetInfo(mat, MAT_GLOBAL_MAX, &info));
    maxnnz  =  (PetscInt) info.nz_used;
    PetscCall(PetscPrintf(comm,"  [sum(nnz) sum(alloc) max(nnz) fill]=[%d %d %d %.2f]",sumnnz,sumalloc,maxnnz,fill));
  }

  PetscCall(PetscPrintf(comm, "\n"));

  //TODO implement as type-specific methods
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATTRANSPOSEVIRTUAL, &flg));
  if (flg) {
    PetscCall(MatTransposeGetMat(mat,&inmat));
    PetscCall(PetscObjectGetTabLevel((PetscObject)inmat,&tablevel));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)inmat,(PetscObject)mat,1));
    PetscCall(MatPrintInfo(inmat));
    PetscCall(PetscObjectSetTabLevel((PetscObject)inmat,tablevel));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATBLOCKDIAG, &flg));
  if (flg) {
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    if (!rank) {
      PetscCall(MatGetDiagonalBlock(mat,&inmat));
      PetscCall(PetscObjectGetTabLevel((PetscObject)inmat,&tablevel));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)inmat,(PetscObject)mat,1));
      PetscCall(MatPrintInfo(inmat));
      PetscCall(PetscObjectSetTabLevel((PetscObject)inmat,tablevel));
    }
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATNESTPERMON, &flg));
  if (flg) {
    PetscInt Mn,Nn,i,j;
    PetscCall(MatNestGetSize(mat,&Mn,&Nn));
    for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
      PetscCall(MatNestGetSubMat(mat,i,j,&inmat));
      PetscCall(PetscObjectGetTabLevel((PetscObject)inmat,&tablevel));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)inmat,(PetscObject)mat,1));
      PetscCall(PetscPrintf(comm, "  nested Mat (%d,%d):\n",i,j));
      PetscCall(MatPrintInfo(inmat));
      PetscCall(PetscObjectSetTabLevel((PetscObject)inmat,tablevel));
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsIdentity"
PetscErrorCode MatIsIdentity(Mat A, PetscReal tol, PetscInt ntrials, PetscBool *flg)
{
  Mat E;
  PetscInt M, N, m, n;
  MPI_Comm comm;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) A, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetSize(A, &M, &N));
  PERMON_ASSERT(M==N, "M==N");

  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatCreateIdentity(comm, m, n, N, &E));
  PetscCall(MatMultEqualTol(A, E, ntrials, tol, flg));
  PetscCall(MatDestroy(&E));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsZero"
PetscErrorCode MatIsZero(Mat A, PetscReal tol, PetscInt ntrials, PetscBool *flg)
{
  Mat O;
  PetscInt M, N, m, n;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) A, &comm));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatCreateZero(comm, m, n, M, N, &O));
  PetscCall(MatMultEqualTol(A, O, ntrials, tol, flg));
  PetscCall(MatDestroy(&O));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsSymmetricByType"
PetscErrorCode MatIsSymmetricByType(Mat A, PetscBool *flg)
{
  PetscBool _flg = PETSC_FALSE;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATBLOCKDIAG, &_flg));
  if (_flg) PetscCall(MatGetDiagonalBlock(A, &A));
  {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &_flg, MATSBAIJ, MATSEQSBAIJ, MATMPISBAIJ, ""));
  }
  PetscCall(PetscBoolGlobalAnd(comm, _flg, flg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Identity"
PetscErrorCode MatMult_Identity(Mat E, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Identity"
PetscErrorCode MatMultAdd_Identity(Mat E, Vec x, Vec y, Vec z)
{
  PetscFunctionBegin;
  PetscCall(VecWAXPY(z,1.0,x,y));
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "MatCreateIdentity"
PetscErrorCode MatCreateIdentity(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt N, Mat *E)
{
  PetscFunctionBegin;
  PetscCall(MatCreateShellPermon(comm, m, n, N, N, NULL, E));
  PetscCall(MatShellSetOperation(*E, MATOP_MULT, (void(*)()) MatMult_Identity));
  PetscCall(MatShellSetOperation(*E, MATOP_MULT_TRANSPOSE, (void(*)()) MatMult_Identity));
  PetscCall(MatShellSetOperation(*E, MATOP_MULT_ADD, (void(*)()) MatMultAdd_Identity));
  PetscCall(MatShellSetOperation(*E, MATOP_MULT_TRANSPOSE_ADD, (void(*)()) MatMultAdd_Identity));
  PetscCall(MatSetOption(*E, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*E, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Zero"
PetscErrorCode MatMult_Zero(Mat O, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Zero"
PetscErrorCode MatMultAdd_Zero(Mat O, Vec x, Vec y, Vec z)
{
  PetscFunctionBegin;
  PetscCall(VecZeroEntries(z));
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "MatCreateZero"
PetscErrorCode MatCreateZero(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat *O)
{
  PetscFunctionBegin;
  PetscCall(MatCreateShellPermon(comm, m, n, M, N, NULL, O));
  PetscCall(MatShellSetOperation(*O, MATOP_MULT, (void(*)()) MatMult_Zero));
  PetscCall(MatShellSetOperation(*O, MATOP_MULT_TRANSPOSE, (void(*)()) MatMult_Zero));
  PetscCall(MatShellSetOperation(*O, MATOP_MULT_ADD, (void(*)()) MatMultAdd_Zero));
  PetscCall(MatShellSetOperation(*O, MATOP_MULT_TRANSPOSE_ADD, (void(*)()) MatMultAdd_Zero));
  PetscCall(MatSetOption(*O, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*O, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Diag"
PetscErrorCode MatMult_Diag(Mat D, Vec x, Vec y)
{
  Vec d;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(D, (Vec*)&d));
  PetscCall(VecPointwiseMult(y, d, x));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Diag"
PetscErrorCode MatMultAdd_Diag(Mat D, Vec x, Vec y, Vec z)
{
  Vec d;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(D, (Vec*)&d));
  PetscCall(VecPointwiseMult(z, d, x));
  PetscCall(VecAXPY(z, 1.0, y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Diag"
PetscErrorCode MatDestroy_Diag(Mat D, Vec x, Vec y, Vec z)
{
  Vec d;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(D, (Vec*)&d));
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Diag"
PetscErrorCode MatGetDiagonal_Diag(Mat D,Vec out_d)
{
  Vec d;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(D, (Vec*)&d));
  PetscCall(VecCopy(d,out_d));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateDiag"
PetscErrorCode MatCreateDiag(Vec d, Mat *D)
{
  PetscInt m,M;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(d, &m));
  PetscCall(VecGetSize(d, &M));
  PetscCall(MatCreateShellPermon(PetscObjectComm((PetscObject)d), m, m, M, M, d, D));
  PetscCall(PetscObjectReference((PetscObject)d));
  PetscCall(MatShellSetOperation(*D, MATOP_MULT, (void(*)()) MatMult_Diag));
  PetscCall(MatShellSetOperation(*D, MATOP_MULT_TRANSPOSE, (void(*)()) MatMult_Diag));
  PetscCall(MatShellSetOperation(*D, MATOP_MULT_ADD, (void(*)()) MatMultAdd_Diag));
  PetscCall(MatShellSetOperation(*D, MATOP_MULT_TRANSPOSE_ADD, (void(*)()) MatMultAdd_Diag));
  PetscCall(MatShellSetOperation(*D, MATOP_DESTROY, (void(*)()) MatDestroy_Diag));
  PetscCall(MatShellSetOperation(*D, MATOP_GET_DIAGONAL, (void(*)()) MatGetDiagonal_Diag));
  PetscCall(MatSetOption(*D, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*D, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateOperatorFromUpperTriangular"
PetscErrorCode MatCreateOperatorFromUpperTriangular(Mat U, Mat *A)
{
  Mat A_arr[3],L,D;
  Vec d;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(U, NULL, &d));
  PetscCall(MatGetDiagonal(U, d));
  PetscCall(MatCreateDiag(d, &D));
  PetscCall(MatScale(D, -1.0));
  PetscCall(PermonMatTranspose(U, MAT_TRANSPOSE_CHEAPEST, &L));
  A_arr[0] = U;
  A_arr[1] = L;
  A_arr[2] = D;
  PetscCall(MatCreateSum(PetscObjectComm((PetscObject)U), 3, A_arr, A));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(MatDestroy(&L));
  PetscCall(MatDestroy(&D));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultEqualTol_Private"
static PetscErrorCode MatMultEqualTol_Private(Mat A,PetscBool transpose,Mat B,PetscInt n,PetscReal tol,PetscBool  *flg)
{
  Vec            x,s1,s2;
  PetscRandom    rctx;
  PetscReal      r1,r2;
  PetscInt       am,an,bm,bn,k;
  PetscScalar    none = -1.0;
  PetscErrorCode (*f)(Mat,Vec,Vec);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  f = transpose ? MatMultTranspose : MatMult;
  PetscCall(MatGetLocalSize(A,&am,&an));
  PetscCall(MatGetLocalSize(B,&bm,&bn));
  if (am != bm || an != bn) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "",am,bm,an,bn);
  PetscCheckSameComm(A,1,B,2);

  if (n==PETSC_DECIDE || n==PETSC_DEFAULT) {
    n = 3;
  } else {
    PetscValidLogicalCollectiveInt(A,n,4);
  }
  if (tol==PETSC_DECIDE || tol==PETSC_DEFAULT) {
    tol = PETSC_SMALL;
  } else {
    PetscValidLogicalCollectiveReal(A,tol,5);
  }

  PetscCall(PetscRandomCreate(((PetscObject)A)->comm,&rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  if (transpose) {
    PetscCall(MatCreateVecs(B,&s1,&x));
  } else {
    PetscCall(MatCreateVecs(B,&x,&s1));
  }
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetFromOptions(s1));
  PetscCall(VecDuplicate(s1,&s2));

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    PetscCall(VecSetRandom(x,rctx));
    PetscCall((*f)(A,x,s1));
    PetscCall((*f)(B,x,s2));
    PetscCall(VecNorm(s2,NORM_INFINITY,&r2));
    if (r2 < tol){
      PetscCall(VecNorm(s1,NORM_INFINITY,&r1));
    } else {
      PetscCall(VecAXPY(s2,none,s1));
      PetscCall(VecNorm(s2,NORM_INFINITY,&r1));
      r1 /= r2;
    }
    PetscCall(PetscInfo(fllop,"relative error of %" PetscInt_FMT "-th MatMult() %g\n",k,r1));
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      break;
    }
  }
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultEqualTol"
PetscErrorCode MatMultEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqualTol_Private(A,PETSC_FALSE,B,n,tol,flg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeEqualTol"
PetscErrorCode MatMultTransposeEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqualTol_Private(A,PETSC_TRUE,B,n,tol,flg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatIsZero"
PetscErrorCode MatMatIsZero(Mat A, Mat B, PetscReal tol, PetscInt ntrials, PetscBool *flg)
{
  Mat KR, KR_arr[2];
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  KR_arr[1]=A;
  KR_arr[0]=B;
  PetscCall(MatCreateProd(comm, 2, KR_arr, &KR));
  PetscCall(MatIsZero(KR, tol, ntrials, flg));
  PetscCall(MatDestroy(&KR));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetMaxEigenvalue"
/*@
   MatGetMaxEigenvalue - Computes approximate maximum eigenvalue lambda
   and its associated eigenvector v (i.e. A*v = lambda*v) with basic power method.

   Collective on Mat

   Input Parameters:
+  A - the matrix
.  v - vector of initial guess (optional)
.  tol - convergence tolerance
-  maxits - maximum number of iterations before divergence error is thrown

   Output Parameters:
+  lambda - approximate maximum eigenvalue of A (optional)
-  v - corresponding eigenvector (optional)

   Level: intermediate
@*/
PetscErrorCode MatGetMaxEigenvalue(Mat A, Vec v, PetscScalar *lambda_out, PetscReal tol, PetscInt maxits)
{
  static PetscBool registered = PETSC_FALSE;
  Vec Av;
  PetscInt  i;
  PetscScalar  lambda, lambda0;
  PetscReal  err, relerr;
  Vec y_v[2];
  PetscReal vAv_vv[2];
  PetscBool destroy_v=PETSC_FALSE;
  PetscBool flg;
  PetscRandom rand=NULL;

  PetscFunctionBeginI;
  lambda = lambda0 = err = relerr = 0.0;
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatGetMaxEig",MAT_CLASSID,&Mat_GetMaxEigenvalue));
    PetscCall(PetscObjectComposedDataRegister(&MatGetMaxEigenvalue_composed_id));
    registered = PETSC_TRUE;
  }
  if (lambda_out && !v) {
    PetscCall(PetscObjectComposedDataGetScalar((PetscObject)A,MatGetMaxEigenvalue_composed_id,lambda,flg));
    if (flg) {
      PetscCall(PetscInfo(fllop,"returning stashed estimate ||A|| = %.12e\n",lambda));
      *lambda_out = lambda;
      PetscFunctionReturnI(0);
    }
  }

  PetscCall(PetscLogEventBegin(Mat_GetMaxEigenvalue,A,0,0,0));

  if (tol==PETSC_DECIDE || tol == PETSC_DEFAULT)    tol = 1e-4;
  if (maxits==PETSC_DECIDE || maxits == PETSC_DEFAULT) maxits = 50;
  if (!v) {
    PetscCall(MatCreateVecs(A,&v,NULL));
    PetscCall(VecSet(v, 1.0));
    destroy_v = PETSC_TRUE;
  }
  PetscCall(VecDuplicate(v,&Av));
  y_v[0] = Av; y_v[1] = v;

  lambda = 0.0;
  for(i=1; i <= maxits; i++) {
    lambda0 = lambda;
    PetscCall(MatMult(A, v, Av));         /* y = A*v */
    /* lambda = (v,A*v)/(v,v) */
    //PetscCall(VecDot(v, y, &vAv));
    //PetscCall(VecDot(v, v, &vv));
    PetscCall(VecMDot(v,2,y_v,vAv_vv));
    lambda = vAv_vv[0]/vAv_vv[1];
    if (lambda < PETSC_MACHINE_EPSILON) {
      PetscCall(PetscInfo(fllop,"hit nullspace of A and setting A*v to random vector in iteration %d\n",i));

      if (!rand) {
        PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)A),&rand));
        PetscCall(PetscRandomSetType(rand,PETSCRAND48));
      }
      PetscCall(VecSetRandom(Av,rand));
      PetscCall(VecDot(v, Av, &vAv_vv[0]));
    }

    err = PetscAbsScalar(lambda-lambda0);
    relerr = err / PetscAbsScalar(lambda);
    if (relerr < tol) break;

    /* v = A*v/||A*v||  replaced by v = A*v/||v|| */
    PetscCall(VecCopy(Av, v));
    PetscCall(VecScale(v, 1.0/PetscSqrtReal(vAv_vv[1])));
  }

  PetscCall(PetscInfo(fllop,"%s  lambda = %.12e  [err relerr reltol] = [%.12e %.12e %.12e]  actual/max iterations = %d/%d\n", (i<=maxits)?"CONVERGED":"NOT CONVERGED", lambda,err,relerr,tol,i,maxits));

  if (lambda_out) *lambda_out = lambda;

  if (destroy_v) PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Av));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscLogEventEnd(Mat_GetMaxEigenvalue,A,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFilterZeros_Default"
static PetscErrorCode MatFilterZeros_Default(Mat A, PetscReal tol, Mat *newAf)
{
  PetscInt m, n, M, N, i, ilo, ihi, j, jlo, jhi, jf, ncols, ncolsf;
  PetscInt *colsf, *d_nnz, *o_nnz;
  const PetscInt *cols;
  PetscScalar *valsf;
  const PetscScalar *vals;
  MPI_Comm comm;
  Mat Af;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetOwnershipRange(A, &ilo, &ihi));
  PetscCall(MatGetOwnershipRangeColumn(A, &jlo, &jhi));
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));

  PetscCall(PetscMalloc(m*sizeof(PetscInt), &d_nnz));
  PetscCall(PetscMalloc(m*sizeof(PetscInt), &o_nnz));
  for (i=0; i<m; i++) {
    PetscCall(MatGetRow(    A, i+ilo, &ncols, &cols, &vals));
    d_nnz[i] = 0;
    o_nnz[i] = 0;
    for (j=0; j<ncols; j++) {
      if (PetscAbs(vals[j]) > tol) {
        if (cols[j] >= jlo && cols[j] < jhi) d_nnz[i]++;
        else                                 o_nnz[i]++;
      }
    }
    PetscCall(MatRestoreRow(A, i+ilo, &ncols, &cols, &vals));
  }

  PetscCall(MatCreate(comm, &Af));
  PetscCall(MatSetSizes(Af,m,n,M,N));
  PetscCall(MatSetType(Af, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(Af, 0, d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(Af, 0, d_nnz, 0, o_nnz));

  PetscCall(PetscFree(d_nnz));
  PetscCall(PetscFree(o_nnz));

  PetscCall(PetscMalloc(N*sizeof(PetscScalar), &valsf));
  PetscCall(PetscMalloc(N*sizeof(PetscInt),    &colsf));
  for (i=ilo; i<ihi; i++) {
    PetscCall(MatGetRow(    A, i, &ncols, &cols, &vals));
    jf = 0;
    for (j=0; j<ncols; j++) {
      if (PetscAbs(vals[j]) > tol) {
        valsf[jf] = vals[j];
        colsf[jf] = cols[j];
        jf++;
      }
    }
    ncolsf = jf;
    PetscCall(MatSetValues(Af,1,&i,ncolsf,colsf,valsf,INSERT_VALUES));
    PetscCall(MatRestoreRow(A, i, &ncols, &cols, &vals));
  }
  PetscCall(MatAssemblyBegin(Af, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(  Af, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(valsf));
  PetscCall(PetscFree(colsf));
  *newAf = Af;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFilterZeros"
PetscErrorCode MatFilterZeros(Mat A, PetscReal tol, Mat *Af_new)
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(Mat,PetscReal,Mat*);

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(A,tol,2);
  PetscValidPointer(Af_new,3);
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatFilterZeros",MAT_CLASSID,&Mat_FilterZeros));
    registered = PETSC_TRUE;
  }
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatFilterZeros_C",&f));
  if (!f) f = MatFilterZeros_Default;

  PetscCall(PetscLogEventBegin(Mat_FilterZeros,A,0,0,0));
  PetscCall((*f)(A,tol,Af_new));
  PetscCall(PetscLogEventEnd(Mat_FilterZeros,A,0,0,0));
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMergeAndDestroy_Default"
static PetscErrorCode MatMergeAndDestroy_Default(MPI_Comm comm, Mat *local_in, Vec x, Mat *global_out)
{
  PetscFunctionBegin;
  PetscInt n = PETSC_DECIDE;
  if (x) PetscCall(VecGetLocalSize(x,&n));
  PetscCall(MatCreateMPIMatConcatenateSeqMat(comm, *local_in, n, MAT_INITIAL_MATRIX, global_out));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMergeAndDestroy_SeqDense"
static PetscErrorCode MatMergeAndDestroy_SeqDense(MPI_Comm comm, Mat *local_in, Vec x, Mat *global_out)
{
  PetscScalar *arr_in,*arr_out;
  PetscInt n = PETSC_DECIDE;
  Mat global;
  Mat A = *local_in;

  PetscFunctionBegin;
  if (x) PetscCall(VecGetLocalSize(x,&n));
  PetscCall(MatCreateDensePermon(comm,A->rmap->n,n,PETSC_DECIDE,A->cmap->N,NULL,&global));
  PetscCall(MatDenseGetArray(A,&arr_in));
  PetscCall(MatDenseGetArray(global,&arr_out));
  //TODO pointer copy
  PetscCall(PetscMemcpy(arr_out,arr_in, A->rmap->n * A->cmap->n * sizeof(PetscScalar)));
  PetscCall(MatDenseRestoreArray(global,&arr_out));
  PetscCall(MatDenseRestoreArray(A,&arr_in));
  PetscCall(MatAssemblyBegin(global,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(global,MAT_FINAL_ASSEMBLY));
  *global_out = global;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMergeAndDestroy"
PetscErrorCode MatMergeAndDestroy(MPI_Comm comm, Mat *local_in, Vec column_layout, Mat *global_out)
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(MPI_Comm,Mat*,Vec,Mat*);
  PetscBool any_nonnull,all_nonnull,flg;
  PetscMPIInt size,rank;
  Mat A = *local_in;

  PetscFunctionBeginI;
  PetscValidPointer(local_in,2);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidPointer(global_out,4);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size > 1) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"currently input matrices must be sequential");
  if (!registered) {
    PetscCall(PetscLogEventRegister("MatMergeAndDestr",MAT_CLASSID,&Mat_MergeAndDestroy));
    registered = PETSC_TRUE;
  }

  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscBoolGlobalOr( comm, local_in ? PETSC_TRUE : PETSC_FALSE, &any_nonnull));
  if (!any_nonnull) {
    *global_out = NULL;
    PetscFunctionReturnI(0);
  }

  PetscCall(PetscBoolGlobalAnd(comm, local_in ? PETSC_TRUE : PETSC_FALSE, &all_nonnull));
  if (!all_nonnull) {
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    SETERRQ(comm,PETSC_ERR_ARG_NULL,"null local matrix on rank %d; currently, local matrices must be either all non-null or all null", rank);
  }

  if (size > 1) {

    /* try to find a type-specific implementation */
    PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatMergeAndDestroy_C",&f));

    /* work-around for MATSEQDENSE to avoid need of a new constructor */
    if (!f) {
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&flg));
      if (flg) f = MatMergeAndDestroy_SeqDense;
    }

    /* if no type-specific implementation is found, use the default one */
    if (!f) f = MatMergeAndDestroy_Default;

    /* call the implementation */
    PetscCall(PetscLogEventBegin(Mat_MergeAndDestroy,A,0,0,0));
    PetscCall((*f)(comm,local_in,column_layout,global_out));
    PetscCall(PetscLogEventEnd(  Mat_MergeAndDestroy,A,0,0,0));

    PetscCall(MatDestroy(local_in));
  } else {
    *global_out = *local_in;
    *local_in = NULL;
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInheritSymmetry"
PetscErrorCode MatInheritSymmetry(Mat A, Mat B)
{
  PetscBool symset, symflg;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscCall(MatIsSymmetricKnown(A, &symset, &symflg));
  if (symset) PetscCall(MatSetOption(B, MAT_SYMMETRIC, symflg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRowNormalization2"
/* not working because N->cmap->rend is 0 in MatGetDiagonal_Normal */
PetscErrorCode MatGetRowNormalization2(Mat A, Vec *d_new)
{
  Mat At,AAt;
  Vec d;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(A,NULL,&d));
  PetscCall(PermonMatTranspose(A,MAT_TRANSPOSE_EXPLICIT, &At));
  PetscCall(MatCreateNormal(At,&AAt));
  PetscCall(MatGetDiagonal(AAt,d));
  PetscCall(VecSqrtAbs(d));
  PetscCall(VecReciprocal(d));
  PetscCall(MatDestroy(&At));
  PetscCall(MatDestroy(&AAt));
  *d_new = d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRowNormalization"
/*@
   MatGetRowNormalization - Get a vector d.

   Not Collective

   Input Parameter:
.  A - the matrix to be row-normalized

   Output Parameter:
.  d - the vector holding normalization, i.e. MatDiagonalScale(mat,NULL,d)
       causes mat to have rows with 2-norm equal to 1

   Level: intermediate

.seealso: MatDiagonalScale()
@*/
PetscErrorCode MatGetRowNormalization(Mat A, Vec *d_new)
{
  Vec d;
  PetscInt i, ilo, ihi;
  PetscInt M, N, ncols, maxncols=0;
  const PetscScalar *vals;
  PetscScalar *rvv;
  Vec rv;
  PetscScalar s;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(A,NULL,&d));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetOwnershipRange(A, &ilo, &ihi));

  /* create vector rv of length equal to the maximum number of nonzeros per row */
  for (i=ilo; i<ihi; i++) {
    PetscCall(MatGetRow(    A, i, &ncols, NULL, NULL));
    if (ncols > maxncols) maxncols = ncols;
    PetscCall(MatRestoreRow(A, i, &ncols, NULL, NULL));
  }
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, maxncols, &rv));

  for (i=ilo; i<ihi; i++) {
    /* copy values from the i-th row to the vector rv */
    PetscCall(MatGetRow(    A, i, &ncols, NULL, &vals));
    PetscCall(VecZeroEntries(rv));
    PetscCall(VecGetArray(rv,&rvv));
    PetscCall(PetscMemcpy(rvv,vals,ncols*sizeof(PetscScalar)));
    PetscCall(VecRestoreArray(rv,&rvv));
    PetscCall(MatRestoreRow(A, i, &ncols, NULL, &vals));

    PetscCall(VecPointwiseMult(rv, rv, rv));                /* rv = rv.^2           */
    PetscCall(VecSum(rv, &s));                              /* s = sum(rv)          */
    PetscCall(VecSetValue(d, i, s, INSERT_VALUES));         /* d(i) = s             */
  }
  PetscCall(VecAssemblyBegin(d));
  PetscCall(VecAssemblyEnd(d));

  PetscCall(VecSqrtAbs(d));                                 /* d = sqrt(abs(d))     */
  PetscCall(VecReciprocal(d));                              /* d = 1./d              */
  PetscCall(VecDestroy(&rv));
  *d_new = d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatGetLocalMat_Default"
static PetscErrorCode PermonMatGetLocalMat_Default(Mat A,Mat *Aloc)
{
  IS ris,cis;
  Mat *Aloc_ptr;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipIS(A,&ris,&cis));
  PetscCall(MatCreateSubMatrices(A,1,&ris,&cis,MAT_INITIAL_MATRIX,&Aloc_ptr));
  *Aloc = *Aloc_ptr;
  PetscCall(PetscFree(Aloc_ptr));
  PetscCall(ISDestroy(&ris));
  PetscCall(ISDestroy(&cis));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatGetLocalMat_MPIAIJ"
static PetscErrorCode PermonMatGetLocalMat_MPIAIJ(Mat A,Mat *Aloc)
{
  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetLocalMat(A, MAT_INITIAL_MATRIX, Aloc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatGetLocalMat_MPIDense"
static PetscErrorCode PermonMatGetLocalMat_MPIDense(Mat A,Mat *Aloc)
{
  PetscFunctionBegin;
  PetscCall(MatDenseGetLocalMatrix(A, Aloc));
  PetscCall(PetscObjectReference((PetscObject)*Aloc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatGetLocalMat"
PetscErrorCode PermonMatGetLocalMat(Mat A,Mat *Aloc)
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(Mat,Mat*);
  PetscMPIInt size;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(Aloc,2);
  if (!registered) {
    PetscCall(PetscLogEventRegister("FlMatGetLocalMat",MAT_CLASSID,&PermonMat_GetLocalMat));
    registered = PETSC_TRUE;
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size > 1) {
    PetscCall(PetscObjectQueryFunction((PetscObject)A,"PermonMatGetLocalMat_C",&f));

    /* work-around for MATMPIAIJ to avoid need of a new constructor */
    if (!f) {
      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg));
      if (flg) f = PermonMatGetLocalMat_MPIAIJ;
    }

    /* work-around for MATDENSEAIJ to avoid need of a new constructor */
    if (!f) {
      PetscCall(PetscObjectTypeCompareAny((PetscObject)A,&flg,MATMPIDENSEPERMON,MATMPIDENSE,""));
      if (flg) f = PermonMatGetLocalMat_MPIDense;
    }

    if (!f) f = PermonMatGetLocalMat_Default;

    {
      Mat T_loc, Adt, Adt_loc;
      Mat Bt = A;
      Mat Bt_arr[2];

      PetscCall(PetscObjectQuery((PetscObject)Bt,"T_loc",(PetscObject*)&T_loc));
      if (T_loc) {
        /* hotfix for B=T*Adt */
        PetscCall(PetscObjectQuery((PetscObject)Bt,"Adt",(PetscObject*)&Adt));
        PERMON_ASSERT(Adt,"Adt");
        PetscCall(PermonMatGetLocalMat(Adt, &Adt_loc));
        Bt_arr[1]=T_loc;
        Bt_arr[0]=Adt_loc;
        PetscCall(MatCreateProd(PETSC_COMM_SELF, 2, Bt_arr, Aloc));
        PetscCall(MatPrintInfo(Bt));
        PetscCall(MatPrintInfo(*Aloc));
        PetscFunctionReturn(0);
      }
    }

    PetscCall(PetscLogEventBegin(PermonMat_GetLocalMat,A,0,0,0));
    PetscCall((*f)(A,Aloc));
    PetscCall(PetscLogEventEnd(  PermonMat_GetLocalMat,A,0,0,0));
  } else {
    *Aloc = A;
    PetscCall(PetscObjectReference((PetscObject)A));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatCreateDenseProductMatrix_Default"
static PetscErrorCode PermonMatCreateDenseProductMatrix_Default(Mat A, PetscBool A_transpose, Mat B, Mat *C)
{
  PetscFunctionBegin;
  PetscCall(MatCreateDensePermon(PetscObjectComm((PetscObject)A), (A_transpose) ? A->cmap->n : A->rmap->n, B->cmap->n, (A_transpose) ? A->cmap->N : A->rmap->N, B->cmap->N, NULL, C));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatCreateDenseProductMatrix"
PetscErrorCode PermonMatCreateDenseProductMatrix(Mat A, PetscBool A_transpose, Mat B, Mat *C_new)
{
  PetscErrorCode (*f)(Mat,PetscBool,Mat,Mat*);

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,A_transpose,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscValidPointer(C_new,4);
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"PermonMatCreateDenseProductMatrix_C",&f));
  if (!f) f = PermonMatCreateDenseProductMatrix_Default;
  PetscCall((*f)(A,A_transpose,B,C_new));
  PetscFunctionReturnI(0);
}

// TODO can be removed with MatProd?
#undef __FUNCT__
#define __FUNCT__ "PermonMatMatMult"
PetscErrorCode PermonMatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscBool flg_A,flg_B;
  Mat T;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidLogicalCollectiveEnum(A,scall,3);
  PetscValidLogicalCollectiveReal(A,fill,4);
  PetscValidPointer(C,5);

  PetscCall(MatIsImplicitTranspose(A, &flg_A));
  PetscCall(MatIsImplicitTranspose(B, &flg_B));
  if (flg_A && flg_B) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"both matrices #1,#2 cannot be implicit transposes (MATTRANSPOSEVIRTUAL)");
  if (flg_A) {
    PetscCall(MatTransposeGetMat(A,&T));
    PetscCall(MatTransposeMatMult(T,B,scall,fill,C));
  } else if (flg_B) {
    PetscCall(MatTransposeGetMat(B,&T));
    PetscCall(MatMatTransposeMult(A,T,scall,fill,C));
  } else {
    PetscCall(MatMatMult(A,B,scall,fill,C));
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatConvertBlocks"
PetscErrorCode PermonMatConvertBlocks(Mat A, MatType newtype,MatReuse reuse,Mat *B)
{
  PetscErrorCode (*f)(Mat,MatType,MatReuse,Mat*);
  char *name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(B,4);

  PetscCall(PetscStrallocpy(((PetscObject)A)->name,&name));

  PetscCall(PetscObjectQueryFunction((PetscObject)A,"PermonMatConvertBlocks_C",&f));
  PetscCall(PetscInfo(A,"%sfound PermonMatConvertBlocks implementation for type %s\n", f?"":"NOT ", ((PetscObject)A)->type_name));
  if (!f) f = MatConvert;
  PetscCall((*f)(A,newtype,reuse,B));

  PetscCall(PetscFree(((PetscObject)*B)->name));
  ((PetscObject)*B)->name = name;
  PetscFunctionReturn(0);
}

//TODO remove or fix with MatProd
#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultWorks"
PetscErrorCode  MatTransposeMatMultWorks(Mat A,Mat B,PetscBool *flg)
{
  PetscErrorCode (*fA)(Mat,Mat,Mat);
  PetscErrorCode (*fB)(Mat,Mat,Mat);
  PetscErrorCode (*transposematmult)(Mat,Mat,Mat) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidType(B,2);
  MatCheckPreallocated(B,2);
  if (!B->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (B->rmap->N!=A->rmap->N) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %" PetscInt_FMT " != %" PetscInt_FMT "",B->rmap->N,A->rmap->N);
  MatCheckPreallocated(A,1);

  *flg = PETSC_TRUE;
  fA = A->ops->transposematmultnumeric;
  fB = B->ops->transposematmultnumeric;
  if (fB==fA) {
    if (!fA) *flg = PETSC_FALSE;
  } else {
    /* dispatch based on the type of A and B from their PetscObject's PetscFunctionLists. */
    char multname[256];
    PetscCall(PetscStrcpy(multname,"MatTransposeMatMult_"));
    PetscCall(PetscStrcat(multname,((PetscObject)A)->type_name));
    PetscCall(PetscStrcat(multname,"_"));
    PetscCall(PetscStrcat(multname,((PetscObject)B)->type_name));
    PetscCall(PetscStrcat(multname,"_C")); /* e.g., multname = "MatMatMult_seqdense_seqaij_C" */
    PetscCall(PetscObjectQueryFunction((PetscObject)B,multname,&transposematmult));
    if (!transposematmult) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatSetFromOptions"
/*@
   PermonMatSetFromOptions - The same as MatSetFromOptions but converts
   the matrix if the type has been already set.

   Collective on Mat

   Input Parameter:
.  A - the matrix

   Options Database Keys:
+    -mat_type seqaij   - AIJ type, uses MatCreateSeqAIJ()
.    -mat_type mpiaij   - AIJ type, uses MatCreateAIJ()
.    -mat_type seqdense - dense type, uses MatCreateSeqDense()
.    -mat_type mpidense - dense type, uses MatCreateDense()
.    -mat_type seqbaij  - block AIJ type, uses MatCreateSeqBAIJ()
-    -mat_type mpibaij  - block AIJ type, uses MatCreateBAIJ()

   Even More Options Database Keys:
   See the manpages for particular formats (e.g., MatCreateSeqAIJ())
   for additional format-specific options.

   Level: beginner

.keywords: matrix, create

.seealso: MatCreateSeqAIJ((), MatCreateAIJ(),
          MatCreateSeqDense(), MatCreateDense(),
          MatCreateSeqBAIJ(), MatCreateBAIJ(),
          MatCreateSeqSBAIJ(), MatCreateSBAIJ(),
          MatConvert()
@*/
PetscErrorCode  PermonMatSetFromOptions(Mat B)
{
  const char     *deft = MATAIJ;
  char           type[256];
  PetscBool      flg,set;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);

  PetscObjectOptionsBegin((PetscObject)B);

  if (B->rmap->bs < 0) {
    PetscInt newbs = -1;
    PetscCall(PetscOptionsInt("-mat_block_size","Set the blocksize used to store the matrix","MatSetBlockSize",newbs,&newbs,&flg));
    if (flg) {
      PetscCall(PetscLayoutSetBlockSize(B->rmap,newbs));
      PetscCall(PetscLayoutSetBlockSize(B->cmap,newbs));
    }
  }

  PetscCall(PetscOptionsFList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg));
  if (!((PetscObject)B)->type_name) {
    if (flg) {
      PetscCall(MatSetType(B,type));
    } else if (!((PetscObject)B)->type_name) {
      PetscCall(MatSetType(B,deft));
    }
  } else if (flg) {
    PetscCall(PermonMatConvertInplace(B,type));
  }

  PetscCall(PetscOptionsName("-mat_is_symmetric","Checks if mat is symmetric on MatAssemblyEnd()","MatIsSymmetric",&B->checksymmetryonassembly));
  PetscCall(PetscOptionsReal("-mat_is_symmetric","Checks if mat is symmetric on MatAssemblyEnd()","MatIsSymmetric",B->checksymmetrytol,&B->checksymmetrytol,NULL));
  PetscCall(PetscOptionsBool("-mat_null_space_test","Checks if provided null space is correct in MatAssemblyEnd()","MatSetNullSpaceTest",B->checknullspaceonassembly,&B->checknullspaceonassembly,NULL));

  if (B->ops->setfromoptions) {
    PetscCall((*B->ops->setfromoptions)(B,PetscOptionsObject));
  }

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_new_nonzero_location_err","Generate an error if new nonzeros are created in the matrix structure (useful to test preallocation)","MatSetOption",flg,&flg,&set));
  if (set) PetscCall(MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,flg));
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_new_nonzero_allocation_err","Generate an error if new nonzeros are allocated in the matrix structure (useful to test preallocation)","MatSetOption",flg,&flg,&set));
  if (set) PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,flg));

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)B,PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatCopyProperties"
PetscErrorCode PermonMatCopyProperties(Mat A,Mat B)
{
  PetscFunctionBegin;
  B->nooffprocentries            = A->nooffprocentries;
  B->assembly_subset             = A->assembly_subset;
  B->nooffproczerorows           = A->nooffproczerorows;
  B->spd_set                     = A->spd_set;
  B->spd                         = A->spd;
  B->symmetric                   = A->symmetric;
  B->symmetric_set               = A->symmetric_set;
  B->symmetric_eternal           = A->symmetric_eternal;
  B->structurally_symmetric      = A->structurally_symmetric;
  B->structurally_symmetric_set  = A->structurally_symmetric_set;
  B->hermitian                   = A->hermitian;
  B->hermitian_set               = A->hermitian_set;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatConvertInplace"
PetscErrorCode PermonMatConvertInplace(Mat A, MatType type)
{
  PetscErrorCode   ierr;
  PetscBool        sametype,issame;
  PetscInt         refct;
  PetscObjectState state;
  char             *name,*prefix;
  Mat              tmp;

  PetscFunctionBeginI;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,type,&sametype));
  PetscCall(PetscStrcmp(type,MATSAME,&issame));
  if (issame || sametype) PetscFunctionReturn(0);

  refct = ((PetscObject)A)->refct;
  state = ((PetscObject)A)->state;
  name = ((PetscObject)A)->name;
  prefix = ((PetscObject)A)->prefix;

  ((PetscObject)A)->name = 0;
  ((PetscObject)A)->prefix = 0;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&tmp));
  PetscCall(PermonMatCopyProperties(A,tmp));

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler,NULL));
  ierr = MatConvert(A,type,MAT_INPLACE_MATRIX,&A);
  PetscCall(PetscPopErrorHandler());
  if (ierr == PETSC_ERR_SUP) {
    PetscCall(PetscInfo(fllop,"matrix type not supported, trying to convert to MATAIJ first\n"));
    PetscCall(MatConvert(A,MATAIJ,MAT_INPLACE_MATRIX,&A));
    PetscCall(MatConvert(A,type,MAT_INPLACE_MATRIX,&A));
  } else if (ierr) {
    /* re-throw error in other error cases */
    PetscCall(MatConvert(A,type,MAT_INPLACE_MATRIX,&A));
  }

  PetscCall(PermonMatCopyProperties(tmp,A));
  PetscCall(MatDestroy(&tmp));

  ((PetscObject)A)->refct = refct;
  ((PetscObject)A)->state = state + 1;
  ((PetscObject)A)->name = name;
  ((PetscObject)A)->prefix = prefix;
  PetscFunctionReturnI(0);
}

PetscErrorCode MatCheckNullSpace(Mat K,Mat R,PetscReal tol)
{
  Vec d,x,y;
  PetscReal normd,normy;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(K,MAT_CLASSID,1);
  PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  PetscValidLogicalCollectiveReal(K,tol,3);
  if (K->cmap->N != R->rmap->N) SETERRQ(PetscObjectComm((PetscObject)K),PETSC_ERR_ARG_SIZ,"non-conforming global size of K and R: %" PetscInt_FMT " != %" PetscInt_FMT "",K->cmap->N,R->rmap->N);
  if (K->cmap->n != R->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"non-conforming local size of K and R: %" PetscInt_FMT " != %" PetscInt_FMT "",K->cmap->n,R->rmap->n);

  PetscCall(MatCreateVecs(K,&d,&y));
  PetscCall(MatCreateVecs(R,&x,NULL));
  PetscCall(MatGetDiagonal(K,d));
  PetscCall(VecNorm(d,NORM_2,&normd));
  PetscCall(VecSetRandom(x,NULL));
  PetscCall(MatMult(R,x,d));
  PetscCall(MatMult(K,d,y));
  PetscCall(VecNorm(y,NORM_2,&normy));
  PetscCall(PetscInfo(fllop,"||K*R*x|| = %.3e   ||diag(K)|| = %.3e    ||K*R*x|| / ||diag(K)|| = %.3e\n",normy,normd,normy/normd));
  PERMON_ASSERT(normy / normd < tol, "||K*R*x|| / ||diag(K)|| < %.1e", tol);
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscFunctionReturnI(0);
}

/* TODO: this is factored out from PETSc MatMatSolve and could be employed also therein */
PetscErrorCode MatRedistributeRows(Mat mat_from,IS rowperm,PetscInt base,Mat mat_to)
{
  PetscMPIInt     commsize;
  PetscInt        i,m_from,m_to,M_to,N;
  PetscInt        *idxx;
  const PetscInt  *rstart,*indices;
  PetscScalar     *arr_from,*arr_to;
  Vec             v_from,v_to;
  IS              is_to;
  VecScatter      sc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat_from,MAT_CLASSID,1);
  PetscValidHeaderSpecific(mat_to,MAT_CLASSID,4);
  PetscCheckSameComm(mat_from,1,mat_to,4);
  if (mat_from->rmap->N != mat_to->rmap->N) SETERRQ(PetscObjectComm((PetscObject)mat_from),PETSC_ERR_ARG_SIZ,"Input matrices must have equal global number of rows, %" PetscInt_FMT " != %" PetscInt_FMT "",mat_from->rmap->N,mat_to->rmap->N);
  if (mat_from->cmap->N != mat_to->cmap->N) SETERRQ(PetscObjectComm((PetscObject)mat_from),PETSC_ERR_ARG_SIZ,"Input matrices must have equal global number of columns, %" PetscInt_FMT " != %" PetscInt_FMT "",mat_from->cmap->N,mat_to->cmap->N);
  m_from = mat_from->rmap->n;
  m_to = mat_to->rmap->n;
  M_to = mat_to->rmap->N;
  N = mat_to->cmap->N;
  PetscCall(MatGetOwnershipRanges(mat_to,&rstart));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat_to),&commsize));
  PetscCall(ISGetIndices(rowperm,&indices));

  PetscCall(MatDenseGetArray(mat_from,&arr_from));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m_from*N,arr_from,&v_from));
  PetscCall(MatDenseGetArray(mat_to,&arr_to));
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat_to),1,m_to*N,M_to*N,arr_to,&v_to));

  PetscCall(PetscMalloc1(m_from*N,&idxx));
  for (i=0; i<m_from; i++) {
    PetscInt proc,idx,iidx,j,m;

    idx = indices[i] - base;
    for (proc=0; proc<commsize; proc++){
      if (idx >= rstart[proc] && idx < rstart[proc+1]) {
        iidx     = idx - rstart[proc] + rstart[proc]*N;
        m        = rstart[proc+1] - rstart[proc];
        break;
      }
    }

    for (j=0; j<N; j++) idxx[i+j*m_from] = iidx + j*m;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,m_from*N,idxx,PETSC_COPY_VALUES,&is_to));
  PetscCall(VecScatterCreate(v_from,NULL,v_to,is_to,&sc));
  PetscCall(VecScatterBegin(sc,v_from,v_to,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sc,v_from,v_to,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(ISDestroy(&is_to));
  PetscCall(VecScatterDestroy(&sc));

  PetscCall(MatAssemblyBegin(mat_to,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_to,MAT_FINAL_ASSEMBLY));

  PetscCall(ISRestoreIndices(rowperm,&indices));
  PetscCall(MatDenseRestoreArray(mat_to,&arr_to));
  PetscCall(VecDestroy(&v_to));
  PetscCall(PetscFree(idxx));
  PetscCall(VecDestroy(&v_from));
  PetscCall(MatDenseRestoreArray(mat_from,&arr_from));
  PetscFunctionReturn(0);
}
