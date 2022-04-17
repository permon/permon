
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
  CHKERRQ(MatShellGetContext(T, (Mat*)&t));
  CHKERRQ(MatForwardSolve(t, x, y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_ForwardSolve"
static PetscErrorCode MatMultTranspose_ForwardSolve(Mat T, Vec x, Vec y)
{
  Mat t=NULL;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(T, (Mat*)&t));
  CHKERRQ(MatBackwardSolve(t, x, y));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatGetSize(A, &M, &N));
  CHKERRQ(MatGetLocalSize(A, &m, &n));

  CHKERRQ(MatGetOwnershipRange(A, &Alo, &Ahi));

  if (form == MAT_ORTH_FORM_EXPLICIT && S_new) {
    CHKERRQ(MatCreateVecs(A, &v, NULL));
    CHKERRQ(VecDuplicateVecs(v, N, &s));
    CHKERRQ(VecGetOwnershipRange(v, &Slo, &Shi));
    for (i = Slo; i < Shi; i++) {
      CHKERRQ(VecSetValue(s[i], i, 1.0, INSERT_VALUES));
      CHKERRQ(VecAssemblyBegin(s[i]));
      CHKERRQ(VecAssemblyEnd(s[i]));
    }
    CHKERRQ(MatCreateDensePermon(comm, n, n, N, N, NULL, &S));
  }

  CHKERRQ(PetscMalloc(N*sizeof(PetscInt), &ic));
  for (i=0; i<N; i++) ic[i]=i;

  //TODO expected fill
  CHKERRQ(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DECIDE, &AtA));
  CHKERRQ(MatSetOption(AtA, MAT_SYMMETRIC, PETSC_TRUE));
  CHKERRQ(MatCreateVecs(AtA, &Arowsol, &Arow));

  CHKERRQ(MatGetOrdering(AtA, MATORDERINGNATURAL, &rperm, &cperm));
  CHKERRQ(MatFactorInfoInitialize(&info));
  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;
  CHKERRQ(MatGetFactor(AtA, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &L));
  CHKERRQ(MatCholeskyFactorSymbolic(L, AtA, rperm, &info));
  CHKERRQ(MatCholeskyFactorNumeric(L, AtA, &info));
  
  CHKERRQ(MatDestroy(&AtA));

  if (form == MAT_ORTH_FORM_EXPLICIT) {
    CHKERRQ(PermonMatConvertBlocks(A, MATDENSEPERMON, MAT_INITIAL_MATRIX, &Q));
    for (i = Alo; i < Ahi; i++) {
      CHKERRQ(VecZeroEntries(Arow));
      CHKERRQ(MatGetRow(A, i, &ncols, &cols, &mvals));
      CHKERRQ(VecSetValues(Arow, ncols, cols, mvals, INSERT_VALUES));
      CHKERRQ(VecAssemblyBegin(Arow));
      CHKERRQ(VecAssemblyEnd(Arow));
      CHKERRQ(MatRestoreRow(A, i, &ncols, &cols, &mvals));

      CHKERRQ(MatForwardSolve(L, Arow, Arowsol));

      CHKERRQ(VecGetArray(Arowsol, &vals));
      CHKERRQ(MatSetValues(Q, 1, &i, N, ic, vals, INSERT_VALUES));
      CHKERRQ(VecRestoreArray(Arowsol, &vals));
    }
    CHKERRQ(MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecDestroy(&Arow));
    if (S_new) {
      for (i = Slo; i < Shi; i++) {
        CHKERRQ(MatForwardSolve(L, s[i], Arowsol));
        CHKERRQ(VecGetArray(Arowsol, &vals));
        CHKERRQ(MatSetValues(S, 1, &i, N, ic, vals, INSERT_VALUES));
        CHKERRQ(VecRestoreArray(Arowsol, &vals));
      }
      CHKERRQ(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
    }
    CHKERRQ(VecDestroy(&Arowsol));    
  } else {
    Mat mats[2];
    CHKERRQ(MatCreateShellPermon(comm, n, n, N, N, L, &S));
    CHKERRQ(MatShellSetOperation(S, MATOP_MULT, (void(*)(void))MatMult_ForwardSolve));
    CHKERRQ(MatShellSetOperation(S, MATOP_MULT_TRANSPOSE, (void(*)(void))MatMultTranspose_ForwardSolve));
    mats[1] = A;
    mats[0] = S;
    CHKERRQ(MatCreateProd(comm, 2, mats, &Q));
  }

  if (Q_new) {
    *Q_new = Q;
  } else {
    CHKERRQ(MatDestroy(&Q));
  }
  
  if (S_new) {
    *S_new = S;
  } else {
    CHKERRQ(MatDestroy(&S));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatGetSize(A, &M, &N));
  CHKERRQ(MatGetLocalSize(A, &m, &n));

  CHKERRQ(MatCreateDummy(comm, n, n, N, N, NULL, &S));
  CHKERRQ(MatCreateDummy(comm, m, n, M, N, NULL, &Q));
  CHKERRQ(PetscObjectCompose((PetscObject)Q, "MatOrthColumns_Implicit_A", (PetscObject)A));
  CHKERRQ(PetscObjectCompose((PetscObject)Q, "MatOrthColumns_Implicit_S", (PetscObject)S));

  if (Q_new) {
    *Q_new = Q;
  } else {
    CHKERRQ(MatDestroy(&Q));
  }
  if (S_new) {
    *S_new = S;
  } else {
    CHKERRQ(MatDestroy(&S));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatGetSize(A, &M, &N));
  CHKERRQ(MatGetLocalSize(A, &m, &n));

  CHKERRQ(MatCreateDummy(comm, m, m, M, M, NULL, &T));
  CHKERRQ(MatCreateDummy(comm, m, n, M, N, NULL, &Qt));
  CHKERRQ(PetscObjectCompose((PetscObject)Qt, "MatOrthColumns_Implicit_A", (PetscObject)A));
  CHKERRQ(PetscObjectCompose((PetscObject)Qt, "MatOrthColumns_Implicit_T", (PetscObject)T));

  if (Qt_new) {
    *Qt_new = Qt;
  } else {
    CHKERRQ(MatDestroy(&Qt));
  }
  if (T_new) {
    *T_new = T;
  } else {
    CHKERRQ(MatDestroy(&T));
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
    CHKERRQ(VecNorm(qcur, NORM_2, &norm));
    o=0;
    do {
      norm_last = norm;
      CHKERRQ(VecMDot(qcur, i, q, dots));
      for (j = 0; j < i; j++) dots[j] = -dots[j];
      CHKERRQ(VecMAXPY(qcur, i, dots, q));            if (s) { CHKERRQ(VecMAXPY(s[i], i, dots, s)); }
      CHKERRQ(VecNorm(qcur, NORM_2, &norm));
      o++;o_acc_++;
      if (norm < 1e2*PETSC_MACHINE_EPSILON) SETERRQ(comm,PETSC_ERR_NOT_CONVERGED,"MatOrthColumns has not converged due to zero norm of the current column %d (i.e. columns 0 - %d are linearly dependent)",i,i);
    } while (norm <= alpha*norm_last);
    CHKERRQ(VecScale(qcur, 1.0/norm));               if (s) { CHKERRQ(VecScale(s[i], 1.0/norm)); }
    if (o_max_ < o) o_max_ = o;
  }
  *o_acc = o_acc_;
  *o_max = o_max_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscScalarNormSquared"
static inline PetscErrorCode PetscScalarNormSquared(PetscInt n,const PetscScalar xx[],PetscReal *z)
{
  PetscBLASInt      one = 1, bn;
  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(n,&bn));
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
    CHKERRQ(VecMDot(qk, k+1, q, p));
    delta_last = PetscSqrtScalar(p[k]);
    while (1) {
      for (j = 0; j < k; j++) p[j] = -p[j];
      CHKERRQ(VecMAXPY(qk, k, p, q));
      if (s) { CHKERRQ(VecMAXPY(s[k], k, p, s)); }
      o++;o_acc_++;

      CHKERRQ(PetscScalarNormSquared(k,p,&beta));
      beta = 1.0 - beta/PetscSqr(delta_last);
      gamma = PetscSqrtReal(PetscAbsReal(beta));
      delta = delta_last * gamma;
      if (delta < 1e2*PETSC_MACHINE_EPSILON) SETERRQ(comm,PETSC_ERR_NOT_CONVERGED,"MatOrthColumns has not converged due to zero norm of the current column %d (i.e. columns 0 - %d are linearly dependent)",k,k);
      if (delta > alpha*delta_last) break;

      CHKERRQ(VecMDot(qk, k, q, p));
      delta_last = delta;
    }
    CHKERRQ(VecScale(qk, 1.0/delta));
    if (s) { CHKERRQ(VecScale(s[k], 1.0/delta)); }
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
  Mat            Q=NULL, S=NULL,Q1;
  PetscInt       M, N, m, n, i, Alo, Ahi, Slo, Shi;
  Vec            *q=NULL, *s=NULL;
  PetscScalar    *dots;
  PetscInt       o_acc=0, o_max=0;
  PetscErrorCode (*f)(MPI_Comm,PetscInt,Vec[],Vec[],PetscScalar[],PetscInt*,PetscInt*);
  PetscBool      computeS = (PetscBool)(form==MAT_ORTH_FORM_IMPLICIT || S_new);

  PetscFunctionBeginI;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatGetSize(A, &M, &N));
  CHKERRQ(MatGetLocalSize(A, &m, &n));
  CHKERRQ(PetscObjectQueryFunction((PetscObject)A,"MatOrthColumns_GS_C",&f));
  if (!f) {
    switch (type) {
      case MAT_ORTH_GS:
        f = MatOrthColumns_GS_Default; break;
      case MAT_ORTH_GS_LINGEN:
        f = MatOrthColumns_GS_Lingen; break;
      default:
        PERMON_ASSERT(0,"this should never happen");
    }
  }

  CHKERRQ(MatGetOwnershipRange(A, &Alo, &Ahi));
  /*TODO: fix this workaround for prealloc check in MatDenseGetLDA */
  CHKERRQ(PermonMatConvertBlocks(A, MATDENSE, MAT_INITIAL_MATRIX, &Q1));
  CHKERRQ(PermonMatConvertBlocks(Q1, MATDENSEPERMON, MAT_INITIAL_MATRIX, &Q));
  CHKERRQ(MatDestroy(&Q1));

  if (computeS) {
    CHKERRQ(MatCreateDensePermon(comm, n, n, N, N, NULL, &S));
    CHKERRQ(MatGetColumnVectors(S, NULL, &s));
    CHKERRQ(VecNestGetMPI(N,&s));
    CHKERRQ(MatGetOwnershipRange(S, &Slo, &Shi));
    for (i = Slo; i < Shi; i++) {
      CHKERRQ(VecSetValue(s[i], i, 1.0, INSERT_VALUES));
    }
    for (i = 0; i < N; i++) {
      CHKERRQ(VecAssemblyBegin(s[i]));
    }
    for (i = 0; i < N; i++) {
      CHKERRQ(VecAssemblyEnd(s[i]));
    }
  }

  CHKERRQ(MatGetColumnVectors(Q, NULL, &q));
  CHKERRQ(VecNestGetMPI(N,&q));
  CHKERRQ(PetscMalloc(N*sizeof(PetscScalar), &dots));

  CHKERRQ((*f)(comm,N, q, s, dots, &o_max, &o_acc));
  CHKERRQ(PetscInfo(fllop,"number of columns %d,  orthogonalizations max, avg, total %d, %g, %d\n", N, o_max, ((PetscReal)o_acc)/N, o_acc));

  /* copy column vectors back to the matrix Q */
  CHKERRQ(VecNestRestoreMPI(N,&q));
  CHKERRQ(MatRestoreColumnVectors(Q, NULL, &q));
  CHKERRQ(PetscFree(dots));    

  if (Q_new) {
    if (form == MAT_ORTH_FORM_IMPLICIT) {
      Mat mats[2] = {S, A};
      CHKERRQ(MatDestroy(&Q));
      CHKERRQ(MatCreateProd(PetscObjectComm((PetscObject)A),2,mats,&Q));
    }
    *Q_new = Q;
  } else {
    CHKERRQ(MatDestroy(&Q));
  }
  
  if (computeS) {
    CHKERRQ(VecNestRestoreMPI(N,&s));
    CHKERRQ(MatRestoreColumnVectors(S, NULL, &s));
    if (S_new) {
      *S_new = S;
    } else {
      CHKERRQ(MatDestroy(&S));
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
    CHKERRQ(PetscLogEventRegister(__FUNCT__, MAT_CLASSID, &Mat_OrthColumns));
    registered = PETSC_TRUE;
  }

  if (type == MAT_ORTH_NONE) {
    if (Q_new) {
      *Q_new = A;
      CHKERRQ(PetscObjectReference((PetscObject)A));
    }
    if (S_new) CHKERRQ(MatCreateIdentity(PetscObjectComm((PetscObject)A),A->cmap->n,A->cmap->n,A->cmap->N,S_new));
    PetscFunctionReturn(0);
  }

  FllopTraceBegin;
  CHKERRQ(PetscObjectQueryFunction((PetscObject)A,"MatOrthColumns_C",&f));

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
        PERMON_ASSERT(0,"this should never happen");
    }
  }
  
  CHKERRQ(PetscLogEventBegin(Mat_OrthColumns,A,0,0,0));
  CHKERRQ((*f)(A,type,form,Q_new,S_new));
  CHKERRQ(PetscLogEventEnd(Mat_OrthColumns,A,0,0,0));

  if (Q_new) {
    /* test whether Q has orthonormal columns */
    CHKERRQ(MatHasOrthonormalColumns(*Q_new,PETSC_SMALL,PETSC_DECIDE,&flg));
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Columns of the resulting matrix are not orthonormal");

    if (S_new) {
      Mat AS, AS_arr[2] = {*S_new,A};
      /* test whether Q = A*S */
      CHKERRQ(MatCreateProd(PetscObjectComm((PetscObject)A),2,AS_arr,&AS));
      CHKERRQ(MatMultEqualTol(AS,*Q_new,3,PETSC_SMALL,&flg));
      if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Resulting factors are wrong, A*S != Q (tolerance %e)",PETSC_SMALL);
      CHKERRQ(MatDestroy(&AS));
    }
  } else if (S_new) {
    Mat AS, AS_arr[2] = {*S_new,A};
    /* test whether A*S has orthonormal rows */
    CHKERRQ(MatCreateProd(PetscObjectComm((PetscObject)A),2,AS_arr,&AS));
    CHKERRQ(MatHasOrthonormalColumns(AS,PETSC_SMALL,PETSC_DECIDE,&flg));
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Columns of A*S are not orthonormal (tolerance %e)",PETSC_SMALL);
    CHKERRQ(MatDestroy(&AS));
  }

  if (Q_new) {
    CHKERRQ(PetscObjectSetName((PetscObject)*Q_new,"Q"));
  }  
  if (S_new) {
    CHKERRQ(PetscObjectSetName((PetscObject)*S_new,"S"));
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
    CHKERRQ(MatOrthRows_Implicit_Default(A,type,form,Qt_new,T_new));
    CHKERRQ(PetscObjectSetName((PetscObject)*Qt_new,"Qt"));
    CHKERRQ(PetscObjectSetName((PetscObject)*T_new,"T"));
    PetscFunctionReturn(0);
  }

  CHKERRQ(PermonMatTranspose(A,MAT_TRANSPOSE_EXPLICIT,&At));
  CHKERRQ(MatOrthColumns(At, type, form, Qt_new?&Qt:NULL, T_new?&Tt:NULL));
  if (Qt_new) {
    CHKERRQ(PermonMatTranspose(Qt,MAT_TRANSPOSE_CHEAPEST,Qt_new));
    CHKERRQ(PetscObjectSetName((PetscObject)*Qt_new,"Qt"));
    CHKERRQ(MatDestroy(&Qt));
  }
  if (T_new) {
    CHKERRQ(PermonMatTranspose(Tt,MAT_TRANSPOSE_CHEAPEST,T_new));
    CHKERRQ(PetscObjectSetName((PetscObject)*T_new,"T"));
    CHKERRQ(MatDestroy(&Tt));
  }
  CHKERRQ(MatDestroy(&At));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHasOrthonormalRowsImplicitly"
PetscErrorCode MatHasOrthonormalRowsImplicitly(Mat A,PetscBool *flg)
{
  Mat T;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  CHKERRQ(PetscObjectQuery((PetscObject)A, "MatOrthColumns_Implicit_T", (PetscObject*)&T));
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
  CHKERRQ(PetscObjectQuery((PetscObject)A, "MatOrthColumns_Implicit_S", (PetscObject*)&S));
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
  CHKERRQ(MatHasOrthonormalRowsImplicitly(A,flg));
  if (*flg) {
    PetscFunctionReturn(0);
  }
  CHKERRQ(PermonMatTranspose(A,MAT_TRANSPOSE_CHEAPEST,&At));
  CHKERRQ(MatCreateNormal(At,&AAt));
  CHKERRQ(MatIsIdentity(AAt,tol,ntrials,flg));
  CHKERRQ(MatDestroy(&At));
  CHKERRQ(MatDestroy(&AAt));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHasOrthonormalColumns"
PetscErrorCode MatHasOrthonormalColumns(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg)
{
  Mat AtA;

  PetscFunctionBegin;
  CHKERRQ(MatHasOrthonormalColumnsImplicitly(A,flg));
  if (*flg) {
    PetscFunctionReturn(0);
  }
  CHKERRQ(MatCreateNormal(A,&AtA));
  CHKERRQ(MatIsIdentity(AtA,tol,ntrials,flg));
  CHKERRQ(MatDestroy(&AtA));
  PetscFunctionReturn(0);
}
