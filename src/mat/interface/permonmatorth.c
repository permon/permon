
#include <permon/private/permonmatimpl.h>
#include <petscblaslapack.h>

PetscLogEvent Mat_OrthColumns;

const char *MatOrthTypes[]={"none","gs","gslingen","cholesky","implicit","inexact","MatOrthType","MAT_ORTH_",0};
const char *MatOrthForms[]={"implicit","explicit","MatOrthForm","MAT_ORTH_",0};

#undef __FUNCT__
#define __FUNCT__ "MatMult_ForwardSolve"
static PetscErrorCode MatMult_ForwardSolve(Mat T, Vec x, Vec y)
{
  Mat t=NULL;

  PetscFunctionBegin;
  TRY( MatShellGetContext(T, (Mat*)&t) );
  TRY( MatForwardSolve(t, x, y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_ForwardSolve"
static PetscErrorCode MatMultTranspose_ForwardSolve(Mat T, Vec x, Vec y)
{
  Mat t=NULL;

  PetscFunctionBegin;
  TRY( MatShellGetContext(T, (Mat*)&t) );
  TRY( MatBackwardSolve(t, x, y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_Cholesky_Default"
static PetscErrorCode MatOrthColumns_Cholesky_Default(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  MPI_Comm        comm;
  Mat             Q=NULL, S=NULL;
  PetscInt        M, N, m, n, i, Alo, Ahi, Slo, Shi;
  Vec             *s, v;
  Mat             AtA;
  Vec             Arowsol, Arow;
  PetscInt        ncols, *ic;
  const PetscInt  *cols;
  PetscScalar     *vals;
  const PetscScalar *mvals;
  IS              rperm, cperm;
  MatFactorInfo   info;
  Mat             L;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)A,&comm) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );

  TRY( MatGetOwnershipRange(A, &Alo, &Ahi) );

  if (form == MAT_ORTH_FORM_EXPLICIT && S_new) {
    TRY( MatCreateVecs(A, &v, NULL) );
    TRY( VecDuplicateVecs(v, N, &s) );
    TRY( VecGetOwnershipRange(v, &Slo, &Shi) );
    for (i = Slo; i < Shi; i++) {
      TRY( VecSetValue(s[i], i, 1.0, INSERT_VALUES) );
      TRY( VecAssemblyBegin(s[i]) );
      TRY( VecAssemblyEnd(s[i]) );
    }
    TRY( MatCreateDensePermon(comm, n, n, N, N, NULL, &S) );
  }

  TRY( PetscMalloc(N*sizeof(PetscInt), &ic) );
  for (i=0; i<N; i++) ic[i]=i;

  //TODO expected fill
  TRY( MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DECIDE, &AtA) );
  TRY( MatSetOption(AtA, MAT_SYMMETRIC, PETSC_TRUE) );
  TRY( MatCreateVecs(AtA, &Arowsol, &Arow) );

  TRY( MatGetOrdering(AtA, MATORDERINGNATURAL, &rperm, &cperm) );
  TRY( MatFactorInfoInitialize(&info) );
  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;
  TRY( MatGetFactor(AtA, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &L) );
  TRY( MatCholeskyFactorSymbolic(L, AtA, rperm, &info) );
  TRY( MatCholeskyFactorNumeric(L, AtA, &info) );
  
  TRY( MatDestroy(&AtA) );

  if (form == MAT_ORTH_FORM_EXPLICIT) {
    TRY( PermonMatConvertBlocks(A, MATDENSEPERMON, MAT_INITIAL_MATRIX, &Q) );
    for (i = Alo; i < Ahi; i++) {
      TRY( VecZeroEntries(Arow) );
      TRY( MatGetRow(A, i, &ncols, &cols, &mvals) );
      TRY( VecSetValues(Arow, ncols, cols, mvals, INSERT_VALUES) );
      TRY( VecAssemblyBegin(Arow) );
      TRY( VecAssemblyEnd(Arow) );
      TRY( MatRestoreRow(A, i, &ncols, &cols, &mvals) );

      TRY( MatForwardSolve(L, Arow, Arowsol) );

      TRY( VecGetArray(Arowsol, &vals) );
      TRY( MatSetValues(Q, 1, &i, N, ic, vals, INSERT_VALUES) );
      TRY( VecRestoreArray(Arowsol, &vals) );
    }
    TRY( MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY) );
    TRY( MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY) );
    TRY( VecDestroy(&Arow) );
    if (S_new) {
      for (i = Slo; i < Shi; i++) {
        TRY( MatForwardSolve(L, s[i], Arowsol) );
        TRY( VecGetArray(Arowsol, &vals) );
        TRY( MatSetValues(S, 1, &i, N, ic, vals, INSERT_VALUES) );
        TRY( VecRestoreArray(Arowsol, &vals) );
      }
      TRY( MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY) );
      TRY( MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY) );
    }
    TRY( VecDestroy(&Arowsol) );    
  } else {
    Mat mats[2];
    TRY( MatCreateShellPermon(comm, n, n, N, N, L, &S) );
    TRY( MatShellSetOperation(S, MATOP_MULT, (void(*)(void))MatMult_ForwardSolve) );
    TRY( MatShellSetOperation(S, MATOP_MULT_TRANSPOSE, (void(*)(void))MatMultTranspose_ForwardSolve) );
    mats[1] = A;
    mats[0] = S;
    TRY( MatCreateProd(comm, 2, mats, &Q) );
  }

  if (Q_new) {
    *Q_new = Q;
  } else {
    TRY( MatDestroy(&Q) );
  }
  
  if (S_new) {
    *S_new = S;
  } else {
    TRY( MatDestroy(&S) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_Implicit_Default"
static PetscErrorCode MatOrthColumns_Implicit_Default(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  MPI_Comm        comm;
  Mat             Q=NULL, S=NULL;
  PetscInt        M, N, m, n;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)A,&comm) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );

  TRY( MatCreateDummy(comm, n, n, N, N, NULL, &S) );
  TRY( MatCreateDummy(comm, m, n, M, N, NULL, &Q) );
  TRY( PetscObjectCompose((PetscObject)Q, "MatOrthColumns_Implicit_A", (PetscObject)A) );
  TRY( PetscObjectCompose((PetscObject)Q, "MatOrthColumns_Implicit_S", (PetscObject)S) );

  if (Q_new) {
    *Q_new = Q;
  } else {
    TRY( MatDestroy(&Q) );
  }
  if (S_new) {
    *S_new = S;
  } else {
    TRY( MatDestroy(&S) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthRows_Implicit_Default"
static PetscErrorCode MatOrthRows_Implicit_Default(Mat A, MatOrthType type, MatOrthForm form, Mat *Qt_new, Mat *T_new)
{
  MPI_Comm        comm;
  Mat             Qt=NULL, T=NULL;
  PetscInt        M, N, m, n;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject)A,&comm) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );

  TRY( MatCreateDummy(comm, m, m, M, M, NULL, &T) );
  TRY( MatCreateDummy(comm, m, n, M, N, NULL, &Qt) );
  TRY( PetscObjectCompose((PetscObject)Qt, "MatOrthColumns_Implicit_A", (PetscObject)A) );
  TRY( PetscObjectCompose((PetscObject)Qt, "MatOrthColumns_Implicit_T", (PetscObject)T) );

  if (Qt_new) {
    *Qt_new = Qt;
  } else {
    TRY( MatDestroy(&Qt) );
  }
  if (T_new) {
    *T_new = T;
  } else {
    TRY( MatDestroy(&T) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_GS_Default"
static PetscErrorCode MatOrthColumns_GS_Default(MPI_Comm comm, PetscInt N, Vec q[], Vec s[], PetscScalar dots[], PetscInt *o_max, PetscInt *o_acc)
{
  PetscReal      norm, norm_last, alpha=0.5;
  PetscInt       i, j, o=0, o_acc_=0, o_max_=0;
  Vec qcur;

  PetscFunctionBegin;
  for (i = 0; i < N; i++) {
    qcur = q[i];
    TRY( VecNorm(qcur, NORM_2, &norm) );
    o=0;
    do {
      norm_last = norm;
      TRY( VecMDot(qcur, i, q, dots) );
      for (j = 0; j < i; j++) dots[j] = -dots[j];
      TRY( VecMAXPY(qcur, i, dots, q) );            if (s) { TRY( VecMAXPY(s[i], i, dots, s) ); }
      TRY( VecNorm(qcur, NORM_2, &norm) );
      o++;o_acc_++;
      if (norm < 1e2*PETSC_MACHINE_EPSILON) FLLOP_SETERRQ2(comm,PETSC_ERR_NOT_CONVERGED,"MatOrthColumns has not converged due to zero norm of the current column %d (i.e. columns 0 - %d are linearly dependent)",i,i);
    } while (norm <= alpha*norm_last);
    TRY( VecScale(qcur, 1.0/norm) );               if (s) { TRY( VecScale(s[i], 1.0/norm) ); }
    if (o_max_ < o) o_max_ = o;
  }
  *o_acc = o_acc_;
  *o_max = o_max_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscScalarNormSquared"
PETSC_STATIC_INLINE PetscErrorCode PetscScalarNormSquared(PetscInt n,const PetscScalar xx[],PetscReal *z)
{
  PetscBLASInt      one = 1, bn;
  PetscFunctionBegin;
  TRY( PetscBLASIntCast(n,&bn) );
  *z   = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_GS_Lingen"
static PetscErrorCode MatOrthColumns_GS_Lingen(MPI_Comm comm, PetscInt N, Vec q[], Vec s[], PetscScalar p[], PetscInt *o_max, PetscInt *o_acc)
{
  PetscReal      delta, delta_last, alpha=0.5, beta, gamma;
  PetscInt       k, j, o=0, o_acc_=0, o_max_=0;
  Vec qk;

  PetscFunctionBeginI;
  for (k = 0; k < N; k++) {
    qk = q[k];
    o=0;
    TRY( VecMDot(qk, k+1, q, p) );
    delta_last = PetscSqrtScalar(p[k]);
    while (1) {
      for (j = 0; j < k; j++) p[j] = -p[j];
      TRY( VecMAXPY(qk, k, p, q) );
      if (s) { TRY( VecMAXPY(s[k], k, p, s) ); }
      o++;o_acc_++;

      TRY( PetscScalarNormSquared(k,p,&beta) );
      beta = 1.0 - beta/PetscSqr(delta_last);
      gamma = PetscSqrtReal(PetscAbsReal(beta));
      delta = delta_last * gamma;
      if (delta < 1e2*PETSC_MACHINE_EPSILON) FLLOP_SETERRQ2(comm,PETSC_ERR_NOT_CONVERGED,"MatOrthColumns has not converged due to zero norm of the current column %d (i.e. columns 0 - %d are linearly dependent)",k,k);
      if (delta > alpha*delta_last) break;

      TRY( VecMDot(qk, k, q, p) );
      delta_last = delta;
    }
    TRY( VecScale(qk, 1.0/delta) );
    if (s) { TRY( VecScale(s[k], 1.0/delta) ); }
    if (o_max_ < o) o_max_ = o;
  }
  *o_acc = o_acc_;
  *o_max = o_max_;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_GS"
static PetscErrorCode MatOrthColumns_GS(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  MPI_Comm       comm;
  Mat            Q=NULL, S=NULL;
  PetscInt       M, N, m, n, i, Alo, Ahi, Slo, Shi;
  Vec            *q=NULL, *s=NULL;
  PetscScalar    *dots;
  PetscInt       o_acc=0, o_max=0;
  PetscErrorCode (*f)(MPI_Comm,PetscInt,Vec[],Vec[],PetscScalar[],PetscInt*,PetscInt*);
  PetscBool      computeS = (PetscBool)(form==MAT_ORTH_FORM_IMPLICIT || S_new);

  PetscFunctionBeginI;
  TRY( PetscObjectGetComm((PetscObject)A,&comm) );
  TRY( MatGetSize(A, &M, &N) );
  TRY( MatGetLocalSize(A, &m, &n) );
  TRY( PetscObjectQueryFunction((PetscObject)A,"MatOrthColumns_GS_C",&f) );
  if (!f) {
    switch (type) {
      case MAT_ORTH_GS:
        f = MatOrthColumns_GS_Default; break;
      case MAT_ORTH_GS_LINGEN:
        f = MatOrthColumns_GS_Lingen; break;
      default:
        FLLOP_ASSERT(0,"this should never happen");
    }
  }

  TRY( MatGetOwnershipRange(A, &Alo, &Ahi) );
  TRY( PermonMatConvertBlocks(A, MATDENSEPERMON, MAT_INITIAL_MATRIX, &Q) );

  if (computeS) {
    TRY( MatCreateDensePermon(comm, n, n, N, N, NULL, &S) );
    TRY( MatGetColumnVectors(S, NULL, &s) );
    TRY( VecNestGetMPI(N,&s) );
    TRY( MatGetOwnershipRange(S, &Slo, &Shi) );
    for (i = Slo; i < Shi; i++) {
      TRY( VecSetValue(s[i], i, 1.0, INSERT_VALUES) );
    }
    for (i = 0; i < N; i++) {
      TRY( VecAssemblyBegin(s[i]) );
    }
    for (i = 0; i < N; i++) {
      TRY( VecAssemblyEnd(s[i]) );
    }
  }

  TRY( MatGetColumnVectors(Q, NULL, &q) );
  TRY( VecNestGetMPI(N,&q) );
  TRY( PetscMalloc(N*sizeof(PetscScalar), &dots) );

  TRY( (*f)(comm,N, q, s, dots, &o_max, &o_acc) );
  TRY( PetscInfo4(fllop,"number of columns %d,  orthogonalizations max, avg, total %d, %g, %d\n", N, o_max, ((PetscReal)o_acc)/N, o_acc) );

  /* copy column vectors back to the matrix Q */
  TRY( VecNestRestoreMPI(N,&q) );
  TRY( MatRestoreColumnVectors(Q, NULL, &q) );
  TRY( PetscFree(dots) );    

  if (Q_new) {
    if (form == MAT_ORTH_FORM_IMPLICIT) {
      Mat mats[2] = {S, A};
      TRY( MatDestroy(&Q) );
      TRY( MatCreateProd(PetscObjectComm((PetscObject)A),2,mats,&Q) );
    }
    *Q_new = Q;
  } else {
    TRY( MatDestroy(&Q) );
  }
  
  if (computeS) {
    TRY( VecNestRestoreMPI(N,&s) );
    TRY( MatRestoreColumnVectors(S, NULL, &s) );
    if (S_new) {
      *S_new = S;
    } else {
      TRY( MatDestroy(&S) );
    }
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns"
/*@
   MatOrthColumns - Perform a QR factorization.

   Collective on Mat

   Input Parameter:
+  A - the matrix whose columns will be orthonormalized
.  type - the algorithm used for orthonormalization
     (one of MAT_ORTH_NONE, MAT_ORTH_GS, MAT_ORTH_CHOLESKY, MAT_ORTH_CHOLESKY_EXPLICIT)
-  form - specify whether Q is computed explicitly or as an implicit product of A and S
     (one of MAT_ORTH_FORM_IMPLICIT, MAT_ORTH_FORM_EXPLICIT)
 
   Output Parameter:
+  Q_new - (optional) the Q factor of the same size as A 
-  S_new - (optional) the inverse of the R factor
 
   Notes:
   This routine computes Q and S so that A = Q*inv(S), Q = A*S.

   Level: intermediate

.seealso: MatOrthRows(), MatOrthType
@*/
PetscErrorCode MatOrthColumns(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  static PetscBool registered = PETSC_FALSE;
  PetscErrorCode (*f)(Mat,MatOrthType,MatOrthForm,Mat*,Mat*);
  PetscBool flg;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(A,type,2);
  if (!registered) {
    TRY( PetscLogEventRegister(__FUNCT__, MAT_CLASSID, &Mat_OrthColumns) );
    registered = PETSC_TRUE;
  }

  if (type == MAT_ORTH_NONE) {
    if (Q_new) {
      *Q_new = A;
      TRY( PetscObjectReference((PetscObject)A) );
    }
    if (S_new) TRY( MatCreateIdentity(PetscObjectComm((PetscObject)A),A->cmap->n,A->cmap->n,A->cmap->N,S_new) );
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  TRY( PetscObjectQueryFunction((PetscObject)A,"MatOrthColumns_C",&f) );

  if (!f) {
    switch (type) {
      case MAT_ORTH_GS:
      case MAT_ORTH_GS_LINGEN:
        f = MatOrthColumns_GS; break;
      case MAT_ORTH_CHOLESKY:
        f = MatOrthColumns_Cholesky_Default; break;
      case MAT_ORTH_IMPLICIT:
      case MAT_ORTH_INEXACT:
        f = MatOrthColumns_Implicit_Default; break;
      case MAT_ORTH_NONE:
        FLLOP_ASSERT(0,"this should never happen");
    }
  }
  
  TRY( PetscLogEventBegin(Mat_OrthColumns,A,0,0,0) );
  TRY( (*f)(A,type,form,Q_new,S_new) );
  TRY( PetscLogEventEnd(Mat_OrthColumns,A,0,0,0) );

  if (Q_new) {
    /* test whether Q has orthonormal columns */
    TRY( MatHasOrthonormalColumns(*Q_new,PETSC_SMALL,PETSC_DECIDE,&flg) );
    if (!flg) FLLOP_SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Columns of the resulting matrix are not orthonormal");

    if (S_new) {
      Mat AS, AS_arr[2] = {*S_new,A};
      /* test whether Q = A*S */
      TRY( MatCreateProd(PetscObjectComm((PetscObject)A),2,AS_arr,&AS) );
      TRY( MatMultEqualTol(AS,*Q_new,3,PETSC_SMALL,&flg) );
      if (!flg) FLLOP_SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Resulting factors are wrong, A*S != Q (tolerance %e)",PETSC_SMALL);
      TRY( MatDestroy(&AS) );
    }
  } else if (S_new) {
    Mat AS, AS_arr[2] = {*S_new,A};
    /* test whether A*S has orthonormal rows */
    TRY( MatCreateProd(PetscObjectComm((PetscObject)A),2,AS_arr,&AS) );
    TRY( MatHasOrthonormalColumns(AS,PETSC_SMALL,PETSC_DECIDE,&flg) );
    if (!flg) FLLOP_SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Columns of A*S are not orthonormal (tolerance %e)",PETSC_SMALL);
    TRY( MatDestroy(&AS) );
  }

  if (Q_new) {
    TRY( PetscObjectSetName((PetscObject)*Q_new,"Q") );
  }  
  if (S_new) {
    TRY( PetscObjectSetName((PetscObject)*S_new,"S") );
  }
  PetscFunctionReturnI(0);
}

/*@
  MatOrthRows - Perform an LQ factorization (transposed QR).

  Collective on Mat

  Input Parameter:
+ A - the matrix whose rows will be orthonormalized
- type - the algorithm used for orthonormalization
  (one of MAT_ORTH_NONE, MAT_ORTH_GS, MAT_ORTH_CHOLESKY, MAT_ORTH_CHOLESKY_EXPLICIT)
 
  Output Parameter:
+ Qt_new - (optional) the Qt factor of the same size as A 
- T_new - (optional) the inverse of the L factor
 
  Notes:
  This routine computes Qt and T so that A = inv(T)*Qt, Qt = T*A.

  Level: intermediate

.seealso: MatOrthColumns(), MatOrthType
@*/
#undef __FUNCT__
#define __FUNCT__ "MatOrthRows"
PetscErrorCode MatOrthRows(Mat A, MatOrthType type, MatOrthForm form, Mat *Qt_new, Mat *T_new)
{
  Mat At,Qt,Tt;

  PetscFunctionBegin;
  if (type == MAT_ORTH_IMPLICIT || type == MAT_ORTH_INEXACT) {
    TRY( MatOrthRows_Implicit_Default(A,type,form,Qt_new,T_new) );
    TRY( PetscObjectSetName((PetscObject)*Qt_new,"Qt") );
    TRY( PetscObjectSetName((PetscObject)*T_new,"T") );
    PetscFunctionReturn(0);
  }

  TRY( PermonMatTranspose(A,MAT_TRANSPOSE_EXPLICIT,&At) );
  TRY( MatOrthColumns(At, type, form, Qt_new?&Qt:NULL, T_new?&Tt:NULL) );
  if (Qt_new) {
    TRY( PermonMatTranspose(Qt,MAT_TRANSPOSE_CHEAPEST,Qt_new) );
    TRY( PetscObjectSetName((PetscObject)*Qt_new,"Qt") );
    TRY( MatDestroy(&Qt) );
  }
  if (T_new) {
    TRY( PermonMatTranspose(Tt,MAT_TRANSPOSE_CHEAPEST,T_new) );
    TRY( PetscObjectSetName((PetscObject)*T_new,"T") );
    TRY( MatDestroy(&Tt) );
  }
  TRY( MatDestroy(&At) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHasOrthonormalRowsImplicitly"
PetscErrorCode MatHasOrthonormalRowsImplicitly(Mat A,PetscBool *flg)
{
  Mat T;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  TRY( PetscObjectQuery((PetscObject)A, "MatOrthColumns_Implicit_T", (PetscObject*)&T) );
  if (T) {
    *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHasOrthonormalColumnsImplicitly"
PetscErrorCode MatHasOrthonormalColumnsImplicitly(Mat A,PetscBool *flg)
{
  Mat S;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  TRY( PetscObjectQuery((PetscObject)A, "MatOrthColumns_Implicit_S", (PetscObject*)&S) );
  if (S) {
    *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHasOrthonormalRows"
PetscErrorCode MatHasOrthonormalRows(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg)
{
  Mat At,AAt;

  PetscFunctionBegin;
  TRY( MatHasOrthonormalRowsImplicitly(A,flg) );
  if (*flg) {
    PetscFunctionReturn(0);
  }
  TRY( PermonMatTranspose(A,MAT_TRANSPOSE_CHEAPEST,&At) );
  TRY( MatCreateNormal(At,&AAt) );
  TRY( MatIsIdentity(AAt,tol,ntrials,flg) );
  TRY( MatDestroy(&At) );
  TRY( MatDestroy(&AAt) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHasOrthonormalColumns"
PetscErrorCode MatHasOrthonormalColumns(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg)
{
  Mat AtA;

  PetscFunctionBegin;
  TRY( MatHasOrthonormalColumnsImplicitly(A,flg) );
  if (*flg) {
    PetscFunctionReturn(0);
  }
  TRY( MatCreateNormal(A,&AtA) );
  TRY( MatIsIdentity(AtA,tol,ntrials,flg) );
  TRY( MatDestroy(&AtA) );
  PetscFunctionReturn(0);
}
