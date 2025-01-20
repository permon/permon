#pragma once

#include "permonqps.h"
#include "permonksp.h"
#include "permonqpfeti.h"

#define FllopAIFLogEventRegister(ename,eid) PetscLogEventRegister(ename,0,eid);
#define FllopAIFLogEventBegin(eid)          PetscLogEventBegin(eid,0,0,0,0)
#define FllopAIFLogEventEnd(eid)            PetscLogEventEnd(eid,0,0,0,0)

typedef enum {AIF_MAT_SYM_UNDEF,AIF_MAT_SYM_SYMMETRIC,AIF_MAT_SYM_UPPER_TRIANGULAR} AIFMatSymmetry;

PERMON_EXTERN PetscErrorCode FllopAIFInitialize(int *argc, char ***args, const char rcfile[]);
PERMON_EXTERN PetscErrorCode FllopAIFInitializeInComm(MPI_Comm comm, int *argc, char ***args, const char rcfile[]);
//PERMON_EXTERN PetscErrorCode FllopAIFInitializeStr(MPI_Comm comm,char *argstr);
//PERMON_EXTERN PetscErrorCode FllopAIFInitializeF(MPI_Fint comm,char *argstr);
PERMON_EXTERN PetscErrorCode FllopAIFReset();
PERMON_EXTERN PetscErrorCode FllopAIFFinalize();

//TODO pointery na pole mohou byt PETSC_NULL!
PERMON_EXTERN PetscErrorCode FllopAIFGetQP(QP *qp);
PERMON_EXTERN PetscErrorCode FllopAIFGetQPS(QPS *qps);

PERMON_EXTERN PetscErrorCode FllopAIFSetSolutionVector(PetscInt n,PetscReal *x,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSetFETIOperator(PetscInt n,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSetFETIOperatorMATIS(PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,IS l2g,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSetFETIOperatorNullspace(PetscInt n,PetscInt d,PetscScalar *R,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSetOperatorByStripes(PetscInt m,PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSetRhs(PetscInt n,PetscScalar *b,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSetEq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname);
PERMON_EXTERN PetscErrorCode FllopAIFAddEq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname);
PERMON_EXTERN PetscErrorCode FllopAIFSetIneq(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,const char *Bname,PetscScalar *cv,const char *cname);
PERMON_EXTERN PetscErrorCode FllopAIFSetEqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname);
PERMON_EXTERN PetscErrorCode FllopAIFAddEqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname);
PERMON_EXTERN PetscErrorCode FllopAIFSetIneqCOO(PetscInt m,PetscInt N,PetscBool B_trans,PetscBool B_dist_horizontal,PetscInt *Bi,PetscInt *Bj,PetscScalar *Bv,PetscInt Bnnz,const char *Bname,PetscScalar *cv,const char *cname);
PERMON_EXTERN PetscErrorCode FllopAIFSetBox(PetscInt n,PetscScalar *lb,const char *lbname,PetscScalar *ub,const char *ubname);

PERMON_EXTERN PetscErrorCode FllopAIFSetArrayBase(PetscInt base);
PERMON_EXTERN PetscErrorCode FllopAIFSetType(const char type[]);
PERMON_EXTERN PetscErrorCode FllopAIFSetDefaultType();

PERMON_EXTERN PetscErrorCode FllopAIFEnforceEqByProjector();
PERMON_EXTERN PetscErrorCode FllopAIFEnforceEqByPenalty(PetscReal rho);
PERMON_EXTERN PetscErrorCode FllopAIFHomogenizeEq();
PERMON_EXTERN PetscErrorCode FllopAIFDualize(MatRegularizationType regtype);
PERMON_EXTERN PetscErrorCode FllopAIFFromOptions();
PERMON_EXTERN PetscErrorCode FllopAIFOperatorShift(PetscScalar a);

PERMON_EXTERN PetscErrorCode FllopAIFSetUp();

PERMON_EXTERN PetscErrorCode FllopAIFKSPSolveMATIS(IS isDir,PetscInt n,PetscInt N,PetscInt* i,PetscInt*j,PetscScalar *A,AIFMatSymmetry symflg,IS l2g,const char *name);
PERMON_EXTERN PetscErrorCode FllopAIFSolve();
