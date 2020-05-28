#if !defined(__FLLOPMAT_H)
#define __FLLOPMAT_H
#include <petscmat.h>
#include "permonvec.h"

/* Mat types provided by FLLOP */
#define MATDUMMY        "dummy"
#define MATINV          "inv"
#define MATBLOCKDIAG    "blockdiag"
#define MATGLUING       "gluing"
#define MATSUM          "sum"
#define MATPROD         "prod"
#define MATDENSEPERMON    "densepermon"
#define MATSEQDENSEPERMON "seqdensepermon"
#define MATMPIDENSEPERMON "mpidensepermon"
#define MATNESTPERMON   "nestpermon"
#define MATEXTENSION    "extension"

FLLOP_EXTERN PetscErrorCode PermonMatRegisterAll();
FLLOP_EXTERN PetscBool PermonMatRegisterAllCalled;

typedef enum {MAT_INV_MONOLITHIC=0, MAT_INV_BLOCKDIAG=1} MatInvType;

/* Mat constructors */
FLLOP_EXTERN PetscErrorCode MatCreateBlockDiag(MPI_Comm comm, Mat localBlock, Mat *BlockDiag);
FLLOP_EXTERN PetscErrorCode MatCreateGluing(MPI_Comm comm, PetscInt n_localRow, PetscInt r,  PetscInt c, const PetscInt *leaves_lrow,	const PetscReal *leaves_sign, PetscSF SF, Mat *B_out); //Alik
FLLOP_EXTERN PetscErrorCode MatCreateSum( MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat);
FLLOP_EXTERN PetscErrorCode MatCreateProd(MPI_Comm comm,PetscInt nmat,const Mat *mats,Mat *mat);
FLLOP_EXTERN PetscErrorCode MatCreateInv(Mat A, MatInvType invType, Mat *imat);
FLLOP_EXTERN PetscErrorCode MatCreateTimer(Mat A, Mat *W);
FLLOP_EXTERN PetscErrorCode MatCreateIdentity(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt N, Mat *E);
FLLOP_EXTERN PetscErrorCode MatCreateZero(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat *O);
FLLOP_EXTERN PetscErrorCode MatCreateDiag(Vec d, Mat *D);
FLLOP_EXTERN PetscErrorCode MatCreateOperatorFromUpperTriangular(Mat U, Mat *A);
FLLOP_EXTERN PetscErrorCode MatCreateExtension(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat A, IS ris, PetscBool rows_use_global_numbering, IS cis, Mat *TA_new);
FLLOP_EXTERN PetscErrorCode MatCreateOneRow(Vec a, Mat *A_new);


/* PETSc fixes */
FLLOP_EXTERN PetscErrorCode MatCreateNormal_permonfix(Mat A,Mat *N);
#define MatCreateNormal(A,N) MatCreateNormal_permonfix(A,N)

FLLOP_EXTERN PetscErrorCode MatCreateShellPermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A);
FLLOP_EXTERN PetscErrorCode MatCreateDummy(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A);

FLLOP_EXTERN PetscErrorCode MatCreateSeqSBAIJWithArrays_permonfix(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt *i,PetscInt *j,PetscScalar *a,Mat *mat);
#define MatCreateSeqSBAIJWithArrays(a,b,c,d,e,f,g,h) MatCreateSeqSBAIJWithArrays_permonfix(a,b,c,d,e,f,g,h)

FLLOP_EXTERN PetscErrorCode MatCreateDensePermon(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscScalar *data,Mat *A_new);
FLLOP_EXTERN PetscErrorCode MatCreateNestPermon(MPI_Comm comm,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[],Mat *B);
FLLOP_EXTERN PetscErrorCode MatCreateNestPermonVerticalMerge(MPI_Comm comm,PetscInt nmats,Mat mats[],Mat *merged);
FLLOP_EXTERN PetscErrorCode MatCreateTransposePermon(Mat A,Mat *At);

PetscErrorCode MatProductSymbolic_NOP(Mat C);

/*   REGULARIZATION   */
typedef enum {MAT_REG_NONE=0, MAT_REG_EXPLICIT=1, MAT_REG_IMPLICIT=2} MatRegularizationType;
FLLOP_EXTERN PetscErrorCode MatRegularize(Mat K, Mat R, MatRegularizationType type, Mat *newKreg);

/* MATEXTENSION specific methods */
FLLOP_EXTERN PetscErrorCode MatExtensionCreateCondensedRows(Mat TA,Mat *A,IS *ris_local);
FLLOP_EXTERN PetscErrorCode MatExtensionCreateLocalMat(Mat TA,Mat *local);
FLLOP_EXTERN PetscErrorCode MatExtensionGetColumnIS(Mat TA,IS *cis);
FLLOP_EXTERN PetscErrorCode MatExtensionGetRowIS(Mat TA,IS *ris);
FLLOP_EXTERN PetscErrorCode MatExtensionGetRowISLocal(Mat TA,IS *ris);
FLLOP_EXTERN PetscErrorCode MatExtensionGetCondensed(Mat TA,Mat *A);
FLLOP_EXTERN PetscErrorCode MatExtensionSetColumnIS(Mat TA,IS  cis);
FLLOP_EXTERN PetscErrorCode MatExtensionSetRowIS(Mat TA,IS  ris,PetscBool rows_use_global_numbering);
FLLOP_EXTERN PetscErrorCode MatExtensionSetCondensed(Mat TA,Mat  A);
FLLOP_EXTERN PetscErrorCode MatExtensionSetUp(Mat TA);

/* MATINV specific methods */
FLLOP_EXTERN PetscErrorCode MatInvGetMat(Mat imat, Mat *A);
FLLOP_EXTERN PetscErrorCode MatInvGetRegularizedMat(Mat imat, Mat *A);
FLLOP_EXTERN PetscErrorCode MatInvGetKSP(Mat imat, KSP *ksp);
FLLOP_EXTERN PetscErrorCode MatInvGetPC(Mat imat, PC *pc);
FLLOP_EXTERN PetscErrorCode MatInvGetType(Mat imat, MatInvType *type);
FLLOP_EXTERN PetscErrorCode MatInvGetRedundancy(Mat imat, PetscInt *red);
FLLOP_EXTERN PetscErrorCode MatInvGetPsubcommType(Mat imat, PetscSubcommType *type);
FLLOP_EXTERN PetscErrorCode MatInvGetRegularizationType(Mat imat,MatRegularizationType *type);
FLLOP_EXTERN PetscErrorCode MatInvGetNullSpace(Mat imat,Mat *R);

FLLOP_EXTERN PetscErrorCode MatInvSetMat(Mat imat, Mat A);
FLLOP_EXTERN PetscErrorCode MatInvSetType(Mat imat, MatInvType type);
FLLOP_EXTERN PetscErrorCode MatInvSetTolerances(Mat imat, PetscReal rtol, PetscReal abstol, PetscReal dtol,PetscInt maxits);
FLLOP_EXTERN PetscErrorCode MatInvSetRedundancy(Mat imat, PetscInt red);
FLLOP_EXTERN PetscErrorCode MatInvSetPsubcommType(Mat imat, PetscSubcommType type);
FLLOP_EXTERN PetscErrorCode MatInvSetRegularizationType(Mat imat,MatRegularizationType type);
FLLOP_EXTERN PetscErrorCode MatInvComputeNullSpace(Mat imat);
FLLOP_EXTERN PetscErrorCode MatInvSetNullSpace(Mat imat,Mat R);

FLLOP_EXTERN PetscErrorCode MatInvExplicitly(Mat imat, PetscBool transpose, MatReuse scall, Mat *imat_explicit);
FLLOP_EXTERN PetscErrorCode MatInvReset(Mat imat);
FLLOP_EXTERN PetscErrorCode MatInvSetUp(Mat imat);
FLLOP_EXTERN PetscErrorCode MatInvCreateInnerObjects(Mat imat);

/* MATTIMER specific methods */
FLLOP_EXTERN PetscErrorCode MatTimerGetMat(Mat W, Mat *A);
FLLOP_EXTERN PetscErrorCode MatTimerSetOperation(Mat mat,MatOperation op,const char *opname,void(*opf)(void));

/* MATGLUING specific methods */
FLLOP_EXTERN PetscErrorCode MatGluingSetLocalBlock(Mat B,Mat Block,PetscInt nghosts);
FLLOP_EXTERN PetscErrorCode MatGluingLayoutSetUp(Mat B);

/* MATTRANSPOSE specific methods */
typedef enum {MAT_TRANSPOSE_EXPLICIT, MAT_TRANSPOSE_IMPLICIT, MAT_TRANSPOSE_CHEAPEST} MatTransposeType;
FLLOP_EXTERN PetscErrorCode MatIsImplicitTranspose(Mat A,PetscBool *flg);

/* MATNEST specific methods */
FLLOP_EXTERN PetscErrorCode MatNestPermonGetVecs(Mat A,Vec *x,Vec *y);
FLLOP_EXTERN PetscErrorCode MatNestPermonGetColumnISs(Mat A,IS **is_new);

/* MATPROD,MATSUM specific methods */
FLLOP_EXTERN PetscErrorCode MatProdGetMat(Mat A,PetscInt i,Mat *Ai);
FLLOP_EXTERN PetscErrorCode MatSumGetMat(Mat A,PetscInt i,Mat *Ai);

/*   GENERAL Mat   */
FLLOP_EXTERN PetscInt MatGetMaxEigenvalue_composed_id;
FLLOP_EXTERN PetscErrorCode MatFactored(Mat A,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatPrintInfo(Mat mat);
FLLOP_EXTERN PetscErrorCode MatMultEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatMultTransposeEqualTol(Mat A,Mat B,PetscInt n,PetscReal tol,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatIsIdentity(Mat A, PetscReal tol, PetscInt ntrials, PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatHasOrthonormalColumns(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatHasOrthonormalRows(Mat A,PetscReal tol,PetscInt ntrials,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatHasOrthonormalColumnsImplicitly(Mat A,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatHasOrthonormalRowsImplicitly(Mat A,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatIsZero(Mat A, PetscReal tol, PetscInt ntrials, PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatMatIsZero(Mat A, Mat B, PetscReal tol, PetscInt ntrials, PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatIsSymmetricByType(Mat A, PetscBool *flg);
FLLOP_EXTERN PetscErrorCode MatGetMaxEigenvalue(Mat A, Vec v, PetscScalar *lambda_out, PetscReal tol, PetscInt maxits);
FLLOP_EXTERN PetscErrorCode MatGetColumnVectors(Mat A, PetscInt *ncols, Vec *cols_new[]);
FLLOP_EXTERN PetscErrorCode MatRestoreColumnVectors(Mat A, PetscInt *ncols, Vec *cols_new[]);
FLLOP_EXTERN PetscErrorCode MatFilterZeros(Mat A, PetscReal tol, Mat *Af);
FLLOP_EXTERN PetscErrorCode MatMergeAndDestroy(MPI_Comm comm,Mat *inmat,Vec column_layout,Mat *outmat);
FLLOP_EXTERN PetscErrorCode MatInheritSymmetry(Mat A, Mat B);
FLLOP_EXTERN PetscErrorCode MatCompleteFromUpperTriangular(Mat A);
FLLOP_EXTERN PetscErrorCode MatGetRowNormalization(Mat A, Vec *d);
FLLOP_EXTERN PetscErrorCode MatGetRowNormalization2(Mat A, Vec *d);
FLLOP_EXTERN PetscErrorCode MatMatMultByColumns(Mat A, Mat B, PetscBool filter, Mat *C_new);
FLLOP_EXTERN PetscErrorCode MatTransposeMatMultByColumns(Mat A, Mat B, PetscBool filter, Mat *C_new);
FLLOP_EXTERN PetscErrorCode MatTransposeMatMultWorks(Mat A,Mat B,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode PermonMatTranspose(Mat A,MatTransposeType type,Mat *At_out);
FLLOP_EXTERN PetscErrorCode PermonMatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C);
FLLOP_EXTERN PetscErrorCode PermonMatGetLocalMat(Mat A,Mat *Aloc);
FLLOP_EXTERN PetscErrorCode PermonMatCreateDenseProductMatrix(Mat A, PetscBool A_transpose, Mat B, Mat *C_new);
FLLOP_EXTERN PetscErrorCode PermonMatConvertBlocks(Mat A, MatType newtype,MatReuse reuse,Mat *B);
FLLOP_EXTERN PetscErrorCode PermonMatCopyProperties(Mat A,Mat B);
FLLOP_EXTERN PetscErrorCode PermonMatSetFromOptions(Mat B);
FLLOP_EXTERN PetscErrorCode PermonMatConvertInplace(Mat B, MatType type);
FLLOP_EXTERN PetscErrorCode MatRedistributeRows(Mat mat_from,IS rowperm,PetscInt base,Mat mat_to);

/* FETI UTILITIES */
FLLOP_EXTERN PetscErrorCode MatRemoveGluingOfDirichletDofs(Mat Bgt, Vec cg, Mat Bdt, Mat *Bgt_new, Vec *cg_new, IS *is_new);

/*   ORTHONORMALIZATION   */
typedef enum {MAT_ORTH_NONE=0, MAT_ORTH_GS, MAT_ORTH_GS_LINGEN, MAT_ORTH_CHOLESKY, MAT_ORTH_IMPLICIT, MAT_ORTH_INEXACT} MatOrthType;
typedef enum {MAT_ORTH_FORM_IMPLICIT=0, MAT_ORTH_FORM_EXPLICIT=1} MatOrthForm;
FLLOP_EXTERN const char *MatOrthTypes[], *MatOrthForms[];
FLLOP_EXTERN PetscErrorCode MatOrthColumns(Mat mat, MatOrthType type, MatOrthForm form, Mat *matOrth, Mat *T);
FLLOP_EXTERN PetscErrorCode MatOrthRows(Mat mat, MatOrthType type, MatOrthForm form, Mat *matOrth, Mat *T);

/* Null Space */
FLLOP_EXTERN PetscErrorCode MatCheckNullSpaceMat(Mat K,Mat R,PetscReal tol);

#endif
