#pragma once

#include <petscmat.h>
#include "permonvec.h"

/* Mat types provided by PERMON */
#define MATDUMMY        "dummy"
#define MATINV          "inv"
#define MATBLOCKDIAG    "blockdiag"
#define MATGLUING       "gluing"
#define MATDENSEPERMON    "densepermon"
#define MATSEQDENSEPERMON "seqdensepermon"
#define MATMPIDENSEPERMON "mpidensepermon"
#define MATNESTPERMON   "nestpermon"
#define MATEXTENSION    "extension"

PERMON_EXTERN PetscErrorCode PermonMatRegisterAll();
PERMON_EXTERN PetscBool PermonMatRegisterAllCalled;

typedef enum {MAT_INV_MONOLITHIC=0, MAT_INV_BLOCKDIAG=1} MatInvType;

/* Mat constructors */
PERMON_EXTERN PetscErrorCode MatCreateBlockDiag(MPI_Comm comm, Mat localBlock, Mat *BlockDiag);
PERMON_EXTERN PetscErrorCode MatCreateGluing(MPI_Comm comm, PetscInt n_localRow, PetscInt r,  PetscInt c, const PetscInt *leaves_lrow, const PetscReal *leaves_sign, PetscSF SF, Mat *B_out);
PERMON_EXTERN PetscErrorCode MatCreateProd(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat);
PERMON_EXTERN PetscErrorCode MatCreateInv(Mat A, MatInvType invType, Mat *imat);
PERMON_EXTERN PetscErrorCode MatCreateTimer(Mat A, Mat *W);
PERMON_EXTERN PetscErrorCode MatCreateIdentity(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt N, Mat *E);
PERMON_EXTERN PetscErrorCode MatCreateZero(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat *O);
PERMON_EXTERN PetscErrorCode MatCreateDiag(Vec d, Mat *D);
PERMON_EXTERN PetscErrorCode MatCreateOperatorFromUpperTriangular(Mat U, Mat *A);
PERMON_EXTERN PetscErrorCode MatCreateExtension(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat A, IS ris, PetscBool rows_use_global_numbering, IS cis, Mat *TA_new);
PERMON_EXTERN PetscErrorCode MatCreateOneRow(Vec a, Mat *A_new);

/* PETSc fixes */
PERMON_EXTERN PetscErrorCode MatCreateShellPermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A);
PERMON_EXTERN PetscErrorCode MatCreateDummy(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A);

PERMON_EXTERN PetscErrorCode MatCreateSeqSBAIJWithArrays_permonfix(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt *i,PetscInt *j,PetscScalar *a,Mat *mat);
#define MatCreateSeqSBAIJWithArrays(a,b,c,d,e,f,g,h) MatCreateSeqSBAIJWithArrays_permonfix(a,b,c,d,e,f,g,h)

PERMON_EXTERN PetscErrorCode MatCreateDensePermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A_new);
PERMON_EXTERN PetscErrorCode MatCreateNestPermon(MPI_Comm comm,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[],Mat *B);
PERMON_EXTERN PetscErrorCode MatCreateNestPermonVerticalMerge(MPI_Comm comm,PetscInt nmats,Mat mats[],Mat *merged);
PERMON_EXTERN PetscErrorCode MatCreateTransposePermon(Mat A,Mat *At);

/*   REGULARIZATION   */
typedef enum {MAT_REG_NONE=0, MAT_REG_EXPLICIT=1, MAT_REG_IMPLICIT=2} MatRegularizationType;
PERMON_EXTERN PetscErrorCode MatRegularize(Mat K, Mat R, MatRegularizationType type, Mat *newKreg);

/* MATEXTENSION specific methods */
PERMON_EXTERN PetscErrorCode MatExtensionCreateCondensedRows(Mat TA,Mat *A,IS *ris_local);
PERMON_EXTERN PetscErrorCode MatExtensionCreateLocalMat(Mat TA,Mat *local);
PERMON_EXTERN PetscErrorCode MatExtensionGetColumnIS(Mat TA,IS *cis);
PERMON_EXTERN PetscErrorCode MatExtensionGetRowIS(Mat TA,IS *ris);
PERMON_EXTERN PetscErrorCode MatExtensionGetRowISLocal(Mat TA,IS *ris);
PERMON_EXTERN PetscErrorCode MatExtensionGetCondensed(Mat TA,Mat *A);
PERMON_EXTERN PetscErrorCode MatExtensionSetColumnIS(Mat TA,IS  cis);
PERMON_EXTERN PetscErrorCode MatExtensionSetRowIS(Mat TA,IS  ris,PetscBool rows_use_global_numbering);
PERMON_EXTERN PetscErrorCode MatExtensionSetCondensed(Mat TA,Mat  A);
PERMON_EXTERN PetscErrorCode MatExtensionSetUp(Mat TA);

/* MATINV specific methods */
PERMON_EXTERN PetscErrorCode MatInvGetMat(Mat imat, Mat *A);
PERMON_EXTERN PetscErrorCode MatInvGetRegularizedMat(Mat imat, Mat *A);
PERMON_EXTERN PetscErrorCode MatInvGetKSP(Mat imat, KSP *ksp);
PERMON_EXTERN PetscErrorCode MatInvGetPC(Mat imat, PC *pc);
PERMON_EXTERN PetscErrorCode MatInvGetType(Mat imat, MatInvType *type);
PERMON_EXTERN PetscErrorCode MatInvGetRedundancy(Mat imat, PetscInt *red);
PERMON_EXTERN PetscErrorCode MatInvGetPsubcommType(Mat imat, PetscSubcommType *type);
PERMON_EXTERN PetscErrorCode MatInvGetRegularizationType(Mat imat,MatRegularizationType *type);
PERMON_EXTERN PetscErrorCode MatInvGetNullSpace(Mat imat,Mat *R);

PERMON_EXTERN PetscErrorCode MatInvSetMat(Mat imat, Mat A);
PERMON_EXTERN PetscErrorCode MatInvSetType(Mat imat, MatInvType type);
PERMON_EXTERN PetscErrorCode MatInvSetTolerances(Mat imat, PetscReal rtol, PetscReal abstol, PetscReal dtol,PetscInt maxits);
PERMON_EXTERN PetscErrorCode MatInvSetRedundancy(Mat imat, PetscInt red);
PERMON_EXTERN PetscErrorCode MatInvSetPsubcommType(Mat imat, PetscSubcommType type);
PERMON_EXTERN PetscErrorCode MatInvSetRegularizationType(Mat imat,MatRegularizationType type);
PERMON_EXTERN PetscErrorCode MatInvComputeNullSpace(Mat imat);
PERMON_EXTERN PetscErrorCode MatInvSetNullSpace(Mat imat,Mat R);

PERMON_EXTERN PetscErrorCode MatInvExplicitly(Mat imat, PetscBool transpose, MatReuse scall, Mat *imat_explicit);
PERMON_EXTERN PetscErrorCode MatInvReset(Mat imat);
PERMON_EXTERN PetscErrorCode MatInvSetUp(Mat imat);
PERMON_EXTERN PetscErrorCode MatInvCreateInnerObjects(Mat imat);

/* MATTIMER specific methods */
PERMON_EXTERN PetscErrorCode MatTimerGetMat(Mat W, Mat *A);
PERMON_EXTERN PetscErrorCode MatTimerSetOperation(Mat mat,MatOperation op,const char *opname,void(*opf)(void));

/* MATGLUING specific methods */
PERMON_EXTERN PetscErrorCode MatGluingSetLocalBlock(Mat B,Mat Block,PetscInt nghosts);
PERMON_EXTERN PetscErrorCode MatGluingLayoutSetUp(Mat B);

/* MATTRANSPOSE specific methods */
typedef enum {MAT_TRANSPOSE_EXPLICIT, MAT_TRANSPOSE_IMPLICIT, MAT_TRANSPOSE_CHEAPEST} MatTransposeType;
PERMON_EXTERN PetscErrorCode MatIsImplicitTranspose(Mat A,PetscBool *flg);

/* MATNEST specific methods */
PERMON_EXTERN PetscErrorCode MatNestPermonGetVecs(Mat A,Vec *x,Vec *y);
PERMON_EXTERN PetscErrorCode MatNestPermonGetColumnISs(Mat A,IS **is_new);

/* MATPROD specific methods */
PERMON_EXTERN PetscErrorCode MatProdGetMat(Mat A,PetscInt i,Mat *Ai);

/*   GENERAL Mat   */
PERMON_EXTERN PetscInt MatGetMaxEigenvalue_composed_id;
PERMON_EXTERN PetscErrorCode MatFactored(Mat A,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatPrintInfo(Mat mat);
PERMON_EXTERN PetscErrorCode MatMultEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatMultTransposeEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatIsIdentity(Mat A, PetscReal tol, PetscInt ntrials, PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatHasOrthonormalColumns(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatHasOrthonormalRows(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatHasOrthonormalColumnsImplicitly(Mat A,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatHasOrthonormalRowsImplicitly(Mat A,PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatIsZero(Mat A, PetscReal tol, PetscInt ntrials, PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatMatIsZero(Mat A, Mat B, PetscReal tol, PetscInt ntrials, PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatIsSymmetricByType(Mat A, PetscBool *flg);
PERMON_EXTERN PetscErrorCode MatGetMaxEigenvalue(Mat A, Vec v, PetscScalar *lambda_out, PetscReal tol, PetscInt maxits);
PERMON_EXTERN PetscErrorCode MatGetColumnVectors(Mat A, PetscInt *ncols, Vec *cols_new[]);
PERMON_EXTERN PetscErrorCode MatRestoreColumnVectors(Mat A, PetscInt *ncols, Vec *cols_new[]);
PERMON_EXTERN PetscErrorCode MatFilterZeros(Mat A, PetscReal tol, Mat *Af);
PERMON_EXTERN PetscErrorCode MatMergeAndDestroy(MPI_Comm comm,Mat *inmat,Vec column_layout,Mat *outmat);
PERMON_EXTERN PetscErrorCode MatInheritSymmetry(Mat A, Mat B);
PERMON_EXTERN PetscErrorCode MatCompleteFromUpperTriangular(Mat A);
PERMON_EXTERN PetscErrorCode MatGetRowNormalization(Mat A, Vec *d);
PERMON_EXTERN PetscErrorCode MatGetRowNormalization2(Mat A, Vec *d);
PERMON_EXTERN PetscErrorCode MatMatMultByColumns(Mat A, Mat B, PetscBool filter, Mat *C_new);
PERMON_EXTERN PetscErrorCode MatTransposeMatMultByColumns(Mat A, Mat B, PetscBool filter, Mat *C_new);
PERMON_EXTERN PetscErrorCode MatTransposeMatMultWorks(Mat A,Mat B,PetscBool *flg);
PERMON_EXTERN PetscErrorCode PermonMatTranspose(Mat A,MatTransposeType type,Mat *At_out);
PERMON_EXTERN PetscErrorCode PermonMatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C);
PERMON_EXTERN PetscErrorCode PermonMatGetLocalMat(Mat A,Mat *Aloc);
PERMON_EXTERN PetscErrorCode PermonMatCreateDenseProductMatrix(Mat A, PetscBool A_transpose, Mat B, Mat *C_new);
PERMON_EXTERN PetscErrorCode PermonMatConvertBlocks(Mat A, MatType newtype,MatReuse reuse,Mat *B);
PERMON_EXTERN PetscErrorCode PermonMatCopyProperties(Mat A,Mat B);
PERMON_EXTERN PetscErrorCode PermonMatSetFromOptions(Mat B);
PERMON_EXTERN PetscErrorCode PermonMatConvertInplace(Mat B, MatType type);
PERMON_EXTERN PetscErrorCode MatCheckNullSpace(Mat K,Mat R,PetscReal tol);
PERMON_EXTERN PetscErrorCode MatRedistributeRows(Mat mat_from,IS rowperm,PetscInt base,Mat mat_to);

/* FETI UTILITIES */
PERMON_EXTERN PetscErrorCode MatRemoveGluingOfDirichletDofs(Mat Bgt, Vec cg, Mat Bdt, Mat *Bgt_new, Vec *cg_new, IS *is_new);
PERMON_EXTERN PetscErrorCode MatRemoveGluingOfDirichletDofs_old(Mat Bgt, Vec cg, Mat Bdt, Mat *Bgt_new, Vec *cg_new, IS *is_new);

/*   ORTHONORMALIZATION   */
typedef enum {MAT_ORTH_NONE=0, MAT_ORTH_GS, MAT_ORTH_GS_LINGEN, MAT_ORTH_CHOLESKY, MAT_ORTH_IMPLICIT, MAT_ORTH_INEXACT} MatOrthType;
typedef enum {MAT_ORTH_FORM_IMPLICIT=0, MAT_ORTH_FORM_EXPLICIT=1} MatOrthForm;
PERMON_EXTERN const char *MatOrthTypes[], *MatOrthForms[];
PERMON_EXTERN PetscErrorCode MatOrthColumns(Mat mat, MatOrthType type, MatOrthForm form, Mat *matOrth, Mat *T);
PERMON_EXTERN PetscErrorCode MatOrthRows(Mat mat, MatOrthType type, MatOrthForm form, Mat *matOrth, Mat *T);
