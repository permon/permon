#if !defined(__FLLOPAIF_H)
#define	__FLLOPAIF_H
#include "permonqps.h"
#include "permonksp.h"
#include "permonqpfeti.h"

#define FllopAIFLogEventRegister(ename,eid) PetscLogEventRegister(ename,0,eid);
#define FllopAIFLogEventBegin(eid)          PetscLogEventBegin(eid,0,0,0,0)
#define FllopAIFLogEventEnd(eid)            PetscLogEventEnd(eid,0,0,0,0)

typedef enum {AIF_MAT_SYM_UNDEF,AIF_MAT_SYM_SYMMETRIC,AIF_MAT_SYM_UPPER_TRIANGULAR} AIFMatSymmetry;

FLLOP_EXTERN PetscErrorCode FllopAIFInitialize(int *argc, char ***args, const char rcfile[]);
FLLOP_EXTERN PetscErrorCode FllopAIFInitializeInComm(MPI_Comm comm, int *argc, char ***args, const char rcfile[]);
//FLLOP_EXTERN PetscErrorCode FllopAIFInitializeStr(MPI_Comm comm,char *argstr);
//FLLOP_EXTERN PetscErrorCode FllopAIFInitializeF(MPI_Fint comm,char *argstr);
FLLOP_EXTERN PetscErrorCode FllopAIFReset();
FLLOP_EXTERN PetscErrorCode FllopAIFFinalize();

//TODO pointery na pole mohou byt PETSC_NULL!
FLLOP_EXTERN PetscErrorCode FllopAIFGetQP(QP *qp);
FLLOP_EXTERN PetscErrorCode FllopAIFGetQPS(QPS *qps);

FLLOP_EXTERN PetscErrorCode FllopAIFSetSolutionVector(PetscInt n,PetscReal *x,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSetFETIOperator(PetscInt n,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSetFETIOperatorMATIS(PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,IS l2g,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSetFETIOperatorNullspace(PetscInt n,PetscInt d,PetscScalar *R,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSetOperatorByStripes(PetscInt m,PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSetRhs(PetscInt n,PetscScalar *b,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSetEq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname);
FLLOP_EXTERN PetscErrorCode FllopAIFAddEq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname);
FLLOP_EXTERN PetscErrorCode FllopAIFSetIneq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname);
FLLOP_EXTERN PetscErrorCode FllopAIFSetEqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname);
FLLOP_EXTERN PetscErrorCode FllopAIFAddEqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname);
FLLOP_EXTERN PetscErrorCode FllopAIFSetIneqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname);
FLLOP_EXTERN PetscErrorCode FllopAIFSetBox(PetscInt n,PetscScalar *lb,const char *lbname,PetscScalar *ub,const char *ubname);

FLLOP_EXTERN PetscErrorCode FllopAIFSetArrayBase(PetscInt base);
FLLOP_EXTERN PetscErrorCode FllopAIFSetType(const char type[]);
FLLOP_EXTERN PetscErrorCode FllopAIFSetDefaultType();

FLLOP_EXTERN PetscErrorCode FllopAIFEnforceEqByProjector();
FLLOP_EXTERN PetscErrorCode FllopAIFEnforceEqByPenalty(PetscReal rho);
FLLOP_EXTERN PetscErrorCode FllopAIFHomogenizeEq();
FLLOP_EXTERN PetscErrorCode FllopAIFDualize();
FLLOP_EXTERN PetscErrorCode FllopAIFFromOptions();
FLLOP_EXTERN PetscErrorCode FllopAIFOperatorShift(PetscScalar a);

FLLOP_EXTERN PetscErrorCode FllopAIFSetUp();

FLLOP_EXTERN PetscErrorCode FllopAIFKSPSolveMATIS(IS isDir,PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,IS l2g,const char *name);
FLLOP_EXTERN PetscErrorCode FllopAIFSolve();
#endif
