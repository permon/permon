
#include <private/fllopmatimpl.h>

PetscLogEvent Mat_GetMaxEigenvalue,Mat_FilterZeros,Mat_MergeAndDestroy,FllopMat_GetLocalMat;
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
  const char *name, *type;
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscBool flg;
  Mat inmat;

  PetscFunctionBegin;
  if (!FllopObjectInfoEnabled) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  TRY( PetscObjectGetTabLevel((PetscObject)mat, &tablevel) );
  for (i=0; i<tablevel; i++) {
    TRY( PetscPrintf(comm, "  ") );
  }

  if (mat == NULL) {
    TRY( PetscPrintf(comm, "Mat NULL\n") );
    PetscFunctionReturn(0);
  }

  TRY( MatGetSize(mat, &M, &N) );
  TRY( MatGetLocalSize(mat, &m, &n) );
  TRY( MatGetType(mat, &type) );
  TRY( PetscObjectGetName((PetscObject) mat, &name) );

  TRY( PetscPrintf(comm,
      "Mat %8x %-16s %-10s size(m,n,M,N)=[%6d %6d %10d %10d]",
          mat,name, type,                m, n, M, N) );

  if (mat->ops->getinfo) {
    MatInfo info;
    PetscInt sumnnz,sumalloc,maxnnz;

    TRY( MatGetInfo(mat, MAT_GLOBAL_SUM, &info) );
    sumnnz  =  (PetscInt) info.nz_used;
    sumalloc = (PetscInt) info.nz_allocated;
    fill = ((PetscReal) sumnnz) / M / N;
    TRY( MatGetInfo(mat, MAT_GLOBAL_MAX, &info) );
    maxnnz  =  (PetscInt) info.nz_used;
    TRY( PetscPrintf(comm,"  [sum(nnz) sum(alloc) max(nnz) fill]=[%7d %7d %5d %2.2f]",sumnnz,sumalloc,maxnnz,fill) );
  }

  TRY( PetscPrintf(comm, "\n") );

  //TODO implement as type-specific methods
  TRY( PetscObjectTypeCompare((PetscObject)mat, MATTRANSPOSEMAT, &flg) );
  if (flg) {
    TRY( MatTransposeGetMat(mat,&inmat) );
    TRY( PetscObjectGetTabLevel((PetscObject)inmat,&tablevel) );
    TRY( PetscObjectIncrementTabLevel((PetscObject)inmat,(PetscObject)mat,1) );
    TRY( MatPrintInfo(inmat) );
    TRY( PetscObjectSetTabLevel((PetscObject)inmat,tablevel) );
  }

  TRY( PetscObjectTypeCompare((PetscObject)mat, MATBLOCKDIAG, &flg) );
  if (flg) {
    PetscMPIInt rank;

    TRY( MPI_Comm_rank(comm,&rank) );
    if (!rank) {
      TRY( MatGetDiagonalBlock(mat,&inmat) );
      TRY( PetscObjectGetTabLevel((PetscObject)inmat,&tablevel) );
      TRY( PetscObjectIncrementTabLevel((PetscObject)inmat,(PetscObject)mat,1) );
      TRY( MatPrintInfo(inmat) );
      TRY( PetscObjectSetTabLevel((PetscObject)inmat,tablevel) );
    }
  }

  TRY( PetscObjectTypeCompare((PetscObject)mat, MATNESTPERMON, &flg) );
  if (flg) {
    PetscInt Mn,Nn,i,j;
    TRY( MatNestGetSize(mat,&Mn,&Nn) );
    for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
      TRY( MatNestGetSubMat(mat,i,j,&inmat) );
      TRY( PetscObjectGetTabLevel((PetscObject)inmat,&tablevel) );
      TRY( PetscObjectIncrementTabLevel((PetscObject)inmat,(PetscObject)mat,1) );
      TRY( PetscPrintf(comm, "  nested Mat (%d,%d):\n",i,j) );
      TRY( MatPrintInfo(inmat) );
      TRY( PetscObjectSetTabLevel((PetscObject)inmat,tablevel) );
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
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MPI_Comm_rank(comm, &rank) );
  TRY( MatGetSize(A, &M, &N) );
  FLLOP_ASSERT(M==N, "M==N");

  TRY( MatGetLocalSize(A, &m, &n) );
  TRY( MatCreateIdentity(comm, m, n, N, &E) );
  TRY( MatMultEqualTol(A, E, ntrials, tol, flg) );
  TRY( MatDestroy(&E) );
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
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );
  TRY( MatCreateZero(comm, m, n, M, N, &O) );
  TRY( MatMultEqualTol(A, O, ntrials, tol, flg) );
  TRY( MatDestroy(&O) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsSymmetricByType"
PetscErrorCode MatIsSymmetricByType(Mat A, PetscBool *flg)
{
  PetscBool _flg = PETSC_FALSE;
  MPI_Comm comm;
  
  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)A, &comm) );
  TRY( PetscObjectTypeCompare((PetscObject)A, MATBLOCKDIAG, &_flg) );
  if (_flg) TRY( MatGetDiagonalBlock(A, &A) );
  {
    TRY( PetscObjectTypeCompareAny((PetscObject)A, &_flg, MATSBAIJ, MATSEQSBAIJ, MATMPISBAIJ, "") );
  }
  TRY( PetscBoolGlobalAnd(comm, _flg, flg) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Identity"
PetscErrorCode MatMult_Identity(Mat E, Vec x, Vec y)
{
  PetscFunctionBegin;
  TRY( VecCopy(x, y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Identity"
PetscErrorCode MatMultAdd_Identity(Mat E, Vec x, Vec y, Vec z)
{
  PetscFunctionBegin;
  TRY( VecWAXPY(z,1.0,x,y) );
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "MatCreateIdentity"
PetscErrorCode MatCreateIdentity(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt N, Mat *E)
{
  PetscFunctionBegin;
  TRY( MatCreateShellPermon(comm, m, n, N, N, NULL, E) );
  TRY( MatShellSetOperation(*E, MATOP_MULT, (void(*)()) MatMult_Identity) );
  TRY( MatShellSetOperation(*E, MATOP_MULT_TRANSPOSE, (void(*)()) MatMult_Identity) );
  TRY( MatShellSetOperation(*E, MATOP_MULT_ADD, (void(*)()) MatMultAdd_Identity) );
  TRY( MatShellSetOperation(*E, MATOP_MULT_TRANSPOSE_ADD, (void(*)()) MatMultAdd_Identity) );
  TRY( MatSetOption(*E, MAT_SYMMETRIC, PETSC_TRUE) );
  TRY( MatSetOption(*E, MAT_SYMMETRY_ETERNAL, PETSC_TRUE) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Zero"
PetscErrorCode MatMult_Zero(Mat O, Vec x, Vec y)
{
  PetscFunctionBegin;
  TRY( VecZeroEntries(y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Zero"
PetscErrorCode MatMultAdd_Zero(Mat O, Vec x, Vec y, Vec z)
{
  PetscFunctionBegin;
  TRY( VecZeroEntries(z) );
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "MatCreateZero"
PetscErrorCode MatCreateZero(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat *O)
{
  PetscFunctionBegin;
  TRY( MatCreateShellPermon(comm, m, n, M, N, NULL, O) );
  TRY( MatShellSetOperation(*O, MATOP_MULT, (void(*)()) MatMult_Zero) );
  TRY( MatShellSetOperation(*O, MATOP_MULT_TRANSPOSE, (void(*)()) MatMult_Zero) );
  TRY( MatShellSetOperation(*O, MATOP_MULT_ADD, (void(*)()) MatMultAdd_Zero) );
  TRY( MatShellSetOperation(*O, MATOP_MULT_TRANSPOSE_ADD, (void(*)()) MatMultAdd_Zero) );
  TRY( MatSetOption(*O, MAT_SYMMETRIC, PETSC_TRUE) );
  TRY( MatSetOption(*O, MAT_SYMMETRY_ETERNAL, PETSC_TRUE) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Diag"
PetscErrorCode MatMult_Diag(Mat D, Vec x, Vec y)
{
  Vec d;

  PetscFunctionBegin;
  TRY( MatShellGetContext(D, (Vec*)&d) );
  TRY( VecPointwiseMult(y, d, x) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Diag"
PetscErrorCode MatMultAdd_Diag(Mat D, Vec x, Vec y, Vec z)
{
  Vec d;

  PetscFunctionBegin;
  TRY( MatShellGetContext(D, (Vec*)&d) );
  TRY( VecPointwiseMult(z, d, x) );
  TRY( VecAXPY(z, 1.0, y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Diag"
PetscErrorCode MatDestroy_Diag(Mat D, Vec x, Vec y, Vec z)
{
  Vec d;

  PetscFunctionBegin;
  TRY( MatShellGetContext(D, (Vec*)&d) );
  TRY( VecDestroy(&d) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateDiag"
PetscErrorCode MatCreateDiag(Vec d, Mat *D)
{
  PetscInt m,M;

  PetscFunctionBegin;
  TRY( VecGetLocalSize(d, &m) );
  TRY( VecGetSize(d, &M) );
  TRY( MatCreateShellPermon(PetscObjectComm((PetscObject)d), m, m, M, M, d, D) );
  TRY( PetscObjectReference((PetscObject)d) );
  TRY( MatShellSetOperation(*D, MATOP_MULT, (void(*)()) MatMult_Diag) );
  TRY( MatShellSetOperation(*D, MATOP_MULT_TRANSPOSE, (void(*)()) MatMult_Diag) );
  TRY( MatShellSetOperation(*D, MATOP_MULT_ADD, (void(*)()) MatMultAdd_Diag) );
  TRY( MatShellSetOperation(*D, MATOP_MULT_TRANSPOSE_ADD, (void(*)()) MatMultAdd_Diag) );
  TRY( MatShellSetOperation(*D, MATOP_DESTROY, (void(*)()) MatDestroy_Diag) );
  TRY( MatSetOption(*D, MAT_SYMMETRIC, PETSC_TRUE) );
  TRY( MatSetOption(*D, MAT_SYMMETRY_ETERNAL, PETSC_TRUE) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateOperatorFromUpperTriangular"
PetscErrorCode MatCreateOperatorFromUpperTriangular(Mat U, Mat *A)
{
  Mat A_arr[3],L,D;
  Vec d;

  PetscFunctionBegin;
  TRY( MatCreateVecs(U, NULL, &d) );
  TRY( MatGetDiagonal(U, d) );
  TRY( MatCreateDiag(d, &D) );
  TRY( MatScale(D, -1.0) );
  TRY( FllopMatTranspose(U, MAT_TRANSPOSE_CHEAPEST, &L) );
  A_arr[0] = U;
  A_arr[1] = L;
  A_arr[2] = D;
  TRY( MatCreateSum(PetscObjectComm((PetscObject)U), 3, A_arr, A) );
  TRY( MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE) );
  TRY( MatSetOption(*A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE) );
  TRY( MatDestroy(&L) );
  TRY( MatDestroy(&D) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultEqualTol_Private"
static PetscErrorCode MatMultEqualTol_Private(Mat A,PetscBool transpose,Mat B,PetscInt n,PetscReal tol,PetscBool  *flg)
{
  PetscErrorCode ierr;
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
  ierr = MatGetLocalSize(A,&am,&an);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&bm,&bn);CHKERRQ(ierr);
  if (am != bm || an != bn) FLLOP_SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat A,Mat B: local dim %D %D %D %D",am,bm,an,bn);
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

  ierr = PetscRandomCreate(((PetscObject)A)->comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  if (transpose) {
    ierr = MatCreateVecs(B,&s1,&x);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(B,&x,&s1);CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s1);CHKERRQ(ierr);
  ierr = VecDuplicate(s1,&s2);CHKERRQ(ierr);

  *flg = PETSC_TRUE;
  for (k=0; k<n; k++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = (*f)(A,x,s1);CHKERRQ(ierr);
    ierr = (*f)(B,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_INFINITY,&r2);CHKERRQ(ierr);
    if (r2 < tol){
      ierr = VecNorm(s1,NORM_INFINITY,&r1);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(s2,none,s1);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_INFINITY,&r1);CHKERRQ(ierr);
      r1 /= r2;
    }
    ierr = PetscInfo2(fllop,"relative error of %D-th MatMult() %g\n",k,r1);CHKERRQ(ierr);
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      break;
    }
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultEqualTol"
PetscErrorCode MatMultEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  TRY( MatMultEqualTol_Private(A,PETSC_FALSE,B,n,tol,flg) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeEqualTol"
PetscErrorCode MatMultTransposeEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool  *flg)
{
  PetscFunctionBegin;
  TRY( MatMultEqualTol_Private(A,PETSC_TRUE,B,n,tol,flg) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatIsZero"
PetscErrorCode MatMatIsZero(Mat A, Mat B, PetscReal tol, PetscInt ntrials, PetscBool *flg)
{
  Mat KR, KR_arr[2];
  MPI_Comm comm;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)A, &comm) );
  KR_arr[1]=A;
  KR_arr[0]=B;
  TRY( MatCreateProd(comm, 2, KR_arr, &KR) );
  TRY( MatIsZero(KR, tol, ntrials, flg) );
  TRY( MatDestroy(&KR) );
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
    TRY( PetscLogEventRegister("MatGetMaxEig",MAT_CLASSID,&Mat_GetMaxEigenvalue) );
    TRY( PetscObjectComposedDataRegister(&MatGetMaxEigenvalue_composed_id) );
    registered = PETSC_TRUE;
  }
  if (lambda_out && !v) {
    TRY( PetscObjectComposedDataGetScalar((PetscObject)A,MatGetMaxEigenvalue_composed_id,lambda,flg) );
    if (flg) {
      TRY( PetscInfo1(fllop,"returning stashed estimate ||A|| = %.12e\n",lambda) );
      *lambda_out = lambda;
      PetscFunctionReturnI(0);
    }
  }

  TRY( PetscLogEventBegin(Mat_GetMaxEigenvalue,A,0,0,0) );

  if (tol==PETSC_DECIDE || tol == PETSC_DEFAULT)    tol = 1e-4;
  if (maxits==PETSC_DECIDE || maxits == PETSC_DEFAULT) maxits = 50;
  if (!v) {
    TRY( MatCreateVecs(A,&v,NULL) );
    TRY( VecSet(v, 1.0) );
    destroy_v = PETSC_TRUE;
  }
  TRY( VecDuplicate(v,&Av) );
  y_v[0] = Av; y_v[1] = v;

  lambda = 0.0;
  for(i=1; i <= maxits; i++) {
    lambda0 = lambda;
    TRY( MatMult(A, v, Av) );         /* y = A*v */
    /* lambda = (v,A*v)/(v,v) */
    //TRY( VecDot(v, y, &vAv) );
    //TRY( VecDot(v, v, &vv) );
    TRY( VecMDot(v,2,y_v,vAv_vv) );
    lambda = vAv_vv[0]/vAv_vv[1];
    if (lambda < PETSC_MACHINE_EPSILON) {
      TRY( PetscInfo1(fllop,"hit nullspace of A and setting A*v to random vector in iteration %d\n",i) );

      if (!rand) {
        TRY( PetscRandomCreate(PetscObjectComm((PetscObject)A),&rand) );
        TRY( PetscRandomSetType(rand,PETSCRAND48) );
      }
      TRY( VecSetRandom(Av,rand) );
      TRY( VecDot(v, Av, &vAv_vv[0]) );
    }

    err = PetscAbsScalar(lambda-lambda0);
    relerr = err / PetscAbsScalar(lambda);
    if (relerr < tol) break;

    /* v = A*v/||A*v||  replaced by v = A*v/||v|| */
    TRY( VecCopy(Av, v) );
    TRY( VecScale(v, 1.0/PetscSqrtReal(vAv_vv[1])) );
  }

  TRY( PetscInfo7(fllop,"%s  lambda = %.12e  [err relerr reltol] = [%.12e %.12e %.12e]  actual/max iterations = %d/%d\n", (i<=maxits)?"CONVERGED":"NOT CONVERGED", lambda,err,relerr,tol,i,maxits) );

  if (lambda_out) *lambda_out = lambda;

  if (destroy_v) TRY( VecDestroy(&v) );
  TRY( VecDestroy(&Av) );
  TRY( PetscRandomDestroy(&rand) );
  TRY( PetscLogEventEnd(Mat_GetMaxEigenvalue,A,0,0,0) );
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
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );
  TRY( MatGetOwnershipRange(A, &ilo, &ihi) );
  TRY( MatGetOwnershipRangeColumn(A, &jlo, &jhi) );
  TRY( PetscObjectGetComm((PetscObject)A, &comm) );

  TRY( PetscMalloc(m*sizeof(PetscInt), &d_nnz) );
  TRY( PetscMalloc(m*sizeof(PetscInt), &o_nnz) );
  for (i=0; i<m; i++) {
    TRY( MatGetRow(    A, i+ilo, &ncols, &cols, &vals) );
    d_nnz[i] = 0;
    o_nnz[i] = 0;
    for (j=0; j<ncols; j++) {
      if (PetscAbs(vals[j]) > tol) {
        if (cols[j] >= jlo && cols[j] < jhi) d_nnz[i]++;
        else                                 o_nnz[i]++;
      }
    }
    TRY( MatRestoreRow(A, i+ilo, &ncols, &cols, &vals) );
  }

  TRY( MatCreate(comm, &Af) );
  TRY( MatSetSizes(Af,m,n,M,N) );
  TRY( MatSetType(Af, MATAIJ) );
  TRY( MatSeqAIJSetPreallocation(Af, 0, d_nnz) );
  TRY( MatMPIAIJSetPreallocation(Af, 0, d_nnz, 0, o_nnz) );

  TRY( PetscFree(d_nnz) );
  TRY( PetscFree(o_nnz) );

  TRY( PetscMalloc(N*sizeof(PetscScalar), &valsf) );
  TRY( PetscMalloc(N*sizeof(PetscInt),    &colsf) );
  for (i=ilo; i<ihi; i++) {
    TRY( MatGetRow(    A, i, &ncols, &cols, &vals) );
    jf = 0;
    for (j=0; j<ncols; j++) {
      if (PetscAbs(vals[j]) > tol) {
        valsf[jf] = vals[j];
        colsf[jf] = cols[j];
        jf++;
      }
    }
    ncolsf = jf;
    TRY( MatSetValues(Af,1,&i,ncolsf,colsf,valsf,INSERT_VALUES) );
    TRY( MatRestoreRow(A, i, &ncols, &cols, &vals) );
  }
  TRY( MatAssemblyBegin(Af, MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(  Af, MAT_FINAL_ASSEMBLY) );
  TRY( PetscFree(valsf) );
  TRY( PetscFree(colsf) );
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
    TRY( PetscLogEventRegister("MatFilterZeros",MAT_CLASSID,&Mat_FilterZeros) );
    registered = PETSC_TRUE;
  }
  TRY( PetscObjectQueryFunction((PetscObject)A,"MatFilterZeros_C",&f) );
  if (!f) f = MatFilterZeros_Default;

  TRY( PetscLogEventBegin(Mat_FilterZeros,A,0,0,0) );
  TRY( (*f)(A,tol,Af_new) );
  TRY( PetscLogEventEnd(Mat_FilterZeros,A,0,0,0) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMergeAndDestroy_Default"
static PetscErrorCode MatMergeAndDestroy_Default(MPI_Comm comm, Mat *local_in, Vec x, Mat *global_out)
{
  PetscFunctionBegin;
  PetscInt n = PETSC_DECIDE;
  if (x) TRY( VecGetLocalSize(x,&n) );
  TRY( MatCreateMPIMatConcatenateSeqMat(comm, *local_in, n, MAT_INITIAL_MATRIX, global_out) );
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
  if (x) TRY( VecGetLocalSize(x,&n) );
  TRY( MatCreateDensePermon(comm,A->rmap->n,n,PETSC_DECIDE,A->cmap->N,NULL,&global) );
  TRY( MatDenseGetArray(A,&arr_in) );
  TRY( MatDenseGetArray(global,&arr_out) );
  //TODO pointer copy
  TRY( PetscMemcpy(arr_out,arr_in, A->rmap->n * A->cmap->n * sizeof(PetscScalar)) );
  TRY( MatDenseRestoreArray(global,&arr_out) );
  TRY( MatDenseRestoreArray(A,&arr_in) );
  TRY( MatAssemblyBegin(global,MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(global,MAT_FINAL_ASSEMBLY) );
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
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)A),&size) );
  if (size > 1) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_WRONG,"currently input matrices must be sequential");
  if (!registered) {
    TRY( PetscLogEventRegister("MatMergeAndDestr",MAT_CLASSID,&Mat_MergeAndDestroy) );
    registered = PETSC_TRUE;
  }

  TRY( MPI_Comm_size(comm,&size) );
  TRY( PetscBoolGlobalOr( comm, local_in ? PETSC_TRUE : PETSC_FALSE, &any_nonnull) );
  if (!any_nonnull) {
    *global_out = NULL;
    PetscFunctionReturnI(0);
  }

  TRY( PetscBoolGlobalAnd(comm, local_in ? PETSC_TRUE : PETSC_FALSE, &all_nonnull) );
  if (!all_nonnull) {
    TRY( MPI_Comm_rank(comm,&rank) );
    FLLOP_SETERRQ1(comm,PETSC_ERR_ARG_NULL,"null local matrix on rank %d; currently, local matrices must be either all non-null or all null", rank);
  }

  if (size > 1) {

    /* try to find a type-specific implementation */
    TRY( PetscObjectQueryFunction((PetscObject)A,"MatMergeAndDestroy_C",&f) );

    /* work-around for MATSEQDENSE to avoid need of a new constructor */
    if (!f) {
      TRY( PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&flg) );
      if (flg) f = MatMergeAndDestroy_SeqDense;
    }
    
    /* if no type-specific implementation is found, use the default one */
    if (!f) f = MatMergeAndDestroy_Default;
    
    /* call the implementation */
    TRY( PetscLogEventBegin(Mat_MergeAndDestroy,A,0,0,0) );
    TRY( (*f)(comm,local_in,column_layout,global_out) );
    TRY( PetscLogEventEnd(  Mat_MergeAndDestroy,A,0,0,0) );

    TRY( MatDestroy(local_in) );
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
  TRY( MatIsSymmetricKnown(A, &symset, &symflg) );
  if (symset) TRY( MatSetOption(B, MAT_SYMMETRIC, symflg) );
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
  TRY( MatCreateVecs(A,NULL,&d) );
  TRY( FllopMatTranspose(A,MAT_TRANSPOSE_EXPLICIT, &At) );
  TRY( MatCreateNormal(At,&AAt) );
  TRY( MatGetDiagonal(AAt,d) );
  TRY( VecSqrtAbs(d) );
  TRY( VecReciprocal(d) );
  TRY( MatDestroy(&At) );
  TRY( MatDestroy(&AAt) );
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
  TRY( MatCreateVecs(A,NULL,&d) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetOwnershipRange(A, &ilo, &ihi) );

  /* create vector rv of length equal to the maximum number of nonzeros per row */
  for (i=ilo; i<ihi; i++) {
    TRY( MatGetRow(    A, i, &ncols, NULL, NULL) );
    if (ncols > maxncols) maxncols = ncols;
    TRY( MatRestoreRow(A, i, &ncols, NULL, NULL) );
  }
  TRY( VecCreateSeq(PETSC_COMM_SELF, maxncols, &rv) );

  for (i=ilo; i<ihi; i++) {
    /* copy values from the i-th row to the vector rv */
    TRY( MatGetRow(    A, i, &ncols, NULL, &vals) );
    TRY( VecZeroEntries(rv) );
    TRY( VecGetArray(rv,&rvv) );
    TRY( PetscMemcpy(rvv,vals,ncols*sizeof(PetscScalar)) );
    TRY( VecRestoreArray(rv,&rvv) );
    TRY( MatRestoreRow(A, i, &ncols, NULL, &vals) );

    TRY( VecPointwiseMult(rv, rv, rv) );                /* rv = rv.^2           */
    TRY( VecSum(rv, &s) );                              /* s = sum(rv)          */
    TRY( VecSetValue(d, i, s, INSERT_VALUES) );         /* d(i) = s             */
  }
  TRY( VecAssemblyBegin(d) );
  TRY( VecAssemblyEnd(d) );

  TRY( VecSqrtAbs(d) );                                 /* d = sqrt(abs(d))     */
  TRY( VecReciprocal(d) );                              /* d = 1./d              */
  TRY( VecDestroy(&rv) );
  *d_new = d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatGetLocalMat_Default"
static PetscErrorCode FllopMatGetLocalMat_Default(Mat A,Mat *Aloc)
{
  IS ris,cis;
  Mat *Aloc_ptr;

  PetscFunctionBegin;
  TRY( MatGetOwnershipIS(A,&ris,&cis) );
  TRY( MatGetSubMatrices(A,1,&ris,&cis,MAT_INITIAL_MATRIX,&Aloc_ptr) );
  *Aloc = *Aloc_ptr;
  TRY( PetscFree(Aloc_ptr) );
  TRY( ISDestroy(&ris) );
  TRY( ISDestroy(&cis) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatGetLocalMat_MPIAIJ"
static PetscErrorCode FllopMatGetLocalMat_MPIAIJ(Mat A,Mat *Aloc)
{
  PetscFunctionBegin;
  TRY( MatMPIAIJGetLocalMat(A, MAT_INITIAL_MATRIX, Aloc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatGetLocalMat_MPIDense"
static PetscErrorCode FllopMatGetLocalMat_MPIDense(Mat A,Mat *Aloc)
{
  PetscFunctionBegin;
  TRY( MatDenseGetLocalMatrix(A, Aloc) );
  TRY( PetscObjectReference((PetscObject)*Aloc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatGetLocalMat"
PetscErrorCode FllopMatGetLocalMat(Mat A,Mat *Aloc)
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(Mat,Mat*);
  PetscMPIInt size;
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(Aloc,2);
  if (!registered) {
    TRY( PetscLogEventRegister("FlMatGetLocalMat",MAT_CLASSID,&FllopMat_GetLocalMat) );
    registered = PETSC_TRUE;
  }
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)A),&size) );
  if (size > 1) {
    TRY( PetscObjectQueryFunction((PetscObject)A,"FllopMatGetLocalMat_C",&f) );

    /* work-around for MATMPIAIJ to avoid need of a new constructor */
    if (!f) {
      TRY( PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg) );
      if (flg) f = FllopMatGetLocalMat_MPIAIJ;
    }

    /* work-around for MATDENSEAIJ to avoid need of a new constructor */
    if (!f) {
      TRY( PetscObjectTypeCompareAny((PetscObject)A,&flg,MATMPIDENSEPERMON,MATMPIDENSE,"") );
      if (flg) f = FllopMatGetLocalMat_MPIDense;
    }

    if (!f) f = FllopMatGetLocalMat_Default;

    {
      Mat T_loc, Adt, Adt_loc;
      Mat Bt = A;
      Mat Bt_arr[2];

      TRY( PetscObjectQuery((PetscObject)Bt,"T_loc",(PetscObject*)&T_loc) );
      if (T_loc) {
        /* hotfix for B=T*Adt */
        TRY( PetscObjectQuery((PetscObject)Bt,"Adt",(PetscObject*)&Adt) );
        FLLOP_ASSERT(Adt,"Adt");
        TRY( FllopMatGetLocalMat(Adt, &Adt_loc) );
        Bt_arr[1]=T_loc;
        Bt_arr[0]=Adt_loc;
        TRY( MatCreateProd(PETSC_COMM_SELF, 2, Bt_arr, Aloc) );
        TRY( MatPrintInfo(Bt) );
        TRY( MatPrintInfo(*Aloc) );
        PetscFunctionReturn(0);
      }
    }

    TRY( PetscLogEventBegin(FllopMat_GetLocalMat,A,0,0,0) );
    TRY( (*f)(A,Aloc) );
    TRY( PetscLogEventEnd(  FllopMat_GetLocalMat,A,0,0,0) );
  } else {
    *Aloc = A;
    TRY( PetscObjectReference((PetscObject)A) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatCreateDenseProductMatrix_Default"
static PetscErrorCode FllopMatCreateDenseProductMatrix_Default(Mat A, PetscBool A_transpose, Mat B, Mat *C)
{
  PetscFunctionBegin;
  TRY( MatCreateDensePermon(PetscObjectComm((PetscObject)A), (A_transpose) ? A->cmap->n : A->rmap->n, B->cmap->n, (A_transpose) ? A->cmap->N : A->rmap->N, B->cmap->N, NULL, C) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatCreateDenseProductMatrix"
PetscErrorCode FllopMatCreateDenseProductMatrix(Mat A, PetscBool A_transpose, Mat B, Mat *C_new)
{
  PetscErrorCode (*f)(Mat,PetscBool,Mat,Mat*);

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,A_transpose,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscValidPointer(C_new,4);
  TRY( PetscObjectQueryFunction((PetscObject)A,"FllopMatCreateDenseProductMatrix_C",&f) );
  if (!f) f = FllopMatCreateDenseProductMatrix_Default;
  TRY( (*f)(A,A_transpose,B,C_new) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatMatMult"
PetscErrorCode FllopMatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscBool flg_A,flg_B;
  Mat T;

  PetscFunctionBeginI;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidLogicalCollectiveEnum(A,scall,3);
  PetscValidLogicalCollectiveReal(A,fill,4);
  PetscValidPointer(C,5);

  TRY( MatIsImplicitTranspose(A, &flg_A) );
  TRY( MatIsImplicitTranspose(B, &flg_B) );
  if (flg_A && flg_B) FLLOP_SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"both matrices #1,#2 cannot be implicit transposes (MATTRANSPOSEMAT)");
  if (flg_A) {
    TRY( MatTransposeGetMat(A,&T) );
    TRY( MatTransposeMatMult(T,B,scall,fill,C) );
  } else if (flg_B) {
    TRY( MatTransposeGetMat(B,&T) );
    TRY( MatMatTransposeMult(A,T,scall,fill,C) );
  } else {
    TRY( MatMatMult(A,B,scall,fill,C) );
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatConvertBlocks"
PetscErrorCode FllopMatConvertBlocks(Mat A, MatType newtype,MatReuse reuse,Mat *B)
{
  PetscErrorCode (*f)(Mat,MatType,MatReuse,Mat*);
  char *name;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(B,4);

  TRY( PetscStrallocpy(((PetscObject)A)->name,&name) );

  TRY( PetscObjectQueryFunction((PetscObject)A,"FllopMatConvertBlocks_C",&f) );
  TRY( PetscInfo2(A,"%sfound FllopMatConvertBlocks implementation for type %s\n", f?"":"NOT ", ((PetscObject)A)->type_name) );
  if (!f) f = MatConvert;
  TRY( (*f)(A,newtype,reuse,B) );

  TRY( PetscFree(((PetscObject)*B)->name) );
  ((PetscObject)*B)->name = name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultWorks"
PetscErrorCode  MatTransposeMatMultWorks(Mat A,Mat B,PetscBool *flg)
{
  PetscErrorCode ierr;
  PetscErrorCode (*fA)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*fB)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*transposematmult)(Mat,Mat,MatReuse,PetscReal,Mat*) = NULL;

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
  if (B->rmap->N!=A->rmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",B->rmap->N,A->rmap->N);
  MatCheckPreallocated(A,1);

  *flg = PETSC_TRUE;
  fA = A->ops->transposematmult;
  fB = B->ops->transposematmult;
  if (fB==fA) {
    if (!fA) *flg = PETSC_FALSE;
    transposematmult = fA;
  } else {
    /* dispatch based on the type of A and B from their PetscObject's PetscFunctionLists. */
    char multname[256];
    ierr = PetscStrcpy(multname,"MatTransposeMatMult_");CHKERRQ(ierr);
    ierr = PetscStrcat(multname,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(multname,"_");CHKERRQ(ierr);
    ierr = PetscStrcat(multname,((PetscObject)B)->type_name);CHKERRQ(ierr);
    ierr = PetscStrcat(multname,"_C");CHKERRQ(ierr); /* e.g., multname = "MatMatMult_seqdense_seqaij_C" */
    ierr = PetscObjectQueryFunction((PetscObject)B,multname,&transposematmult);CHKERRQ(ierr);
    if (!transposematmult) *flg = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
