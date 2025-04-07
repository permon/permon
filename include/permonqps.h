#pragma once

#include <petsctao.h>
#include "permonqp.h"

typedef struct _p_QPS *QPS;

PERMON_EXTERN PetscClassId QPS_CLASSID;
#define QPS_CLASS_NAME "qps"

#define QPSType    char *
#define QPSKSP     "ksp"
#define QPSMPGP    "mpgp"
#define QPSMPGPQPC "mpgpqpc"
#define QPSPCPG    "pcpg"
#define QPSSMALXE  "smalxe"
#define QPSTAO     "tao"

typedef enum {
  QPS_ARG_MULTIPLE = 0,
  QPS_ARG_DIRECT   = 1
} QPSScalarArgType;

PERMON_EXTERN PetscErrorCode QPSInitializePackage(void);
PERMON_EXTERN PetscErrorCode QPSFinalizePackage(void);

PERMON_EXTERN PetscFunctionList QPSList;
PERMON_EXTERN PetscBool         QPSRegisterAllCalled;
PERMON_EXTERN PetscErrorCode    QPSRegisterAll(void);
PERMON_EXTERN PetscErrorCode    QPSRegister(const char[], PetscErrorCode (*)(QPS));

PERMON_EXTERN PetscErrorCode QPSCreate(MPI_Comm comm, QPS *qps_new);
PERMON_EXTERN PetscErrorCode QPSView(QPS qps, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPSViewConvergence(QPS qps, PetscViewer viewer);
PERMON_EXTERN PetscErrorCode QPSDestroy(QPS *qps);
PERMON_EXTERN PetscErrorCode QPSSetFromOptions(QPS qps);
PERMON_EXTERN PetscErrorCode QPSSetUp(QPS qps);
PERMON_EXTERN PetscErrorCode QPSReset(QPS qps);
PERMON_EXTERN PetscErrorCode QPSResetStatistics(QPS qps);
PERMON_EXTERN PetscErrorCode QPSSolve(QPS qps);
PERMON_EXTERN PetscErrorCode QPSPostSolve(QPS qps);
PERMON_EXTERN PetscErrorCode QPSIsQPCompatible(QPS qps, QP qp, PetscBool *flg);

PERMON_EXTERN PetscErrorCode QPSSetDefaultType(QPS qps);
PERMON_EXTERN PetscErrorCode QPSSetDefaultTypeIfNotSpecified(QPS qps);
PERMON_EXTERN PetscErrorCode QPSSetType(QPS qps, const QPSType type);
PERMON_EXTERN PetscErrorCode QPSSetQP(QPS qps, QP qp);
PERMON_EXTERN PetscErrorCode QPSSetTolerances(QPS qps, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt maxits);
PERMON_EXTERN PetscErrorCode QPSSetOptionsPrefix(QPS qps, const char prefix[]);
PERMON_EXTERN PetscErrorCode QPSAppendOptionsPrefix(QPS qps, const char prefix[]);
PERMON_EXTERN PetscErrorCode QPSSetConvergenceTest(QPS qps, PetscErrorCode (*converge)(QPS, KSPConvergedReason *), void *cctx, PetscErrorCode (*destroy)(void *));
PERMON_EXTERN PetscErrorCode QPSSetAutoPostSolve(QPS qps, PetscBool flg);

PERMON_EXTERN PetscErrorCode QPSGetType(QPS qps, const QPSType *type);
PERMON_EXTERN PetscErrorCode QPSGetQP(QPS qps, QP *qp);
PERMON_EXTERN PetscErrorCode QPSGetSolvedQP(QPS qps, QP *qp);
PERMON_EXTERN PetscErrorCode QPSGetTolerances(QPS qps, PetscReal *rtol, PetscReal *abstol, PetscReal *dtol, PetscInt *maxits);
PERMON_EXTERN PetscErrorCode QPSGetOptionsPrefix(QPS qps, const char *prefix[]);
PERMON_EXTERN PetscErrorCode QPSGetConvergenceContext(QPS, void **);
PERMON_EXTERN PetscErrorCode QPSGetConvergedReason(QPS, KSPConvergedReason *);
PERMON_EXTERN PetscErrorCode QPSGetResidualNorm(QPS qps, PetscReal *rnorm);
PERMON_EXTERN PetscErrorCode QPSGetIterationNumber(QPS qps, PetscInt *its);
PERMON_EXTERN PetscErrorCode QPSGetAccumulatedIterationNumber(QPS qps, PetscInt *its);
PERMON_EXTERN PetscErrorCode QPSGetAutoPostSolve(QPS qps, PetscBool *flg);
PERMON_EXTERN PetscErrorCode QPSGetVecs(QPS qps, PetscInt rightn, Vec **right, PetscInt leftn, Vec **left);

PERMON_EXTERN PetscErrorCode QPSConvergedSkip(QPS qps, KSPConvergedReason *reason);
PERMON_EXTERN PetscErrorCode QPSConvergedDefault(QPS qps, KSPConvergedReason *reason);
PERMON_EXTERN PetscErrorCode QPSConvergedDefaultSetUp(QPS qps);
PERMON_EXTERN PetscErrorCode QPSConvergedDefaultSetRhsForDivergence(void *cctx, Vec b);
PERMON_EXTERN PetscErrorCode QPSConvergedDefaultDestroy(void *);
PERMON_EXTERN PetscErrorCode QPSConvergedDefaultCreate(void **);

PERMON_EXTERN PetscErrorCode QPSSetWorkVecs(QPS, PetscInt);
PERMON_EXTERN PetscErrorCode QPSDestroyDefault(QPS);

/* QPSMonitor */
PERMON_EXTERN PetscErrorCode QPSMonitor(QPS, PetscInt, PetscReal);
PERMON_EXTERN PetscErrorCode QPSMonitorSet(QPS, PetscErrorCode (*)(QPS, PetscInt, PetscReal, void *), void *, PetscCtxDestroyFn *);
PERMON_EXTERN PetscErrorCode QPSMonitorCancel(QPS);
PERMON_EXTERN PetscErrorCode QPSGetMonitorContext(QPS, void **);
PERMON_EXTERN PetscErrorCode QPSGetResidualHistory(QPS, PetscReal *[], PetscInt *);
PERMON_EXTERN PetscErrorCode QPSSetResidualHistory(QPS, PetscReal[], PetscInt, PetscBool);
PERMON_EXTERN PetscErrorCode QPSMonitorDefault(QPS qps, PetscInt n, PetscReal rnorm, void *dummy);
PERMON_EXTERN PetscErrorCode QPSMonitorCostFunction(QPS qps, PetscInt n, PetscReal rnorm, void *dummy);

/* *** type-specific stuff *** */
/* KSP */
PERMON_EXTERN PetscErrorCode QPSKSPSetKSP(QPS qps, KSP ksp);
PERMON_EXTERN PetscErrorCode QPSKSPGetKSP(QPS qps, KSP *ksp);
PERMON_EXTERN PetscErrorCode QPSKSPSetType(QPS qps, KSPType type);
PERMON_EXTERN PetscErrorCode QPSKSPGetType(QPS qps, KSPType *type);

/*TAO */
PERMON_EXTERN PetscErrorCode QPSTaoSetType(QPS qps, TaoType type);
PERMON_EXTERN PetscErrorCode QPSTaoGetType(QPS qps, TaoType *type);
PERMON_EXTERN PetscErrorCode QPSTaoGetTao(QPS qps, Tao *tao);

/* MPGP */
typedef enum {
  QPS_MPGP_EXPANSION_STD,
  QPS_MPGP_EXPANSION_PROJCG,
  QPS_MPGP_EXPANSION_GF,
  QPS_MPGP_EXPANSION_G,
  QPS_MPGP_EXPANSION_GFGR,
  QPS_MPGP_EXPANSION_GGR
} QPSMPGPExpansionType;
PERMON_EXTERN const char *const QPSMPGPExpansionTypes[];
typedef enum {
  QPS_MPGP_EXPANSION_LENGTH_FIXED,
  QPS_MPGP_EXPANSION_LENGTH_OPT,
  QPS_MPGP_EXPANSION_LENGTH_OPTAPPROX,
  QPS_MPGP_EXPANSION_LENGTH_BB
} QPSMPGPExpansionLengthType;
PERMON_EXTERN const char *const QPSMPGPExpansionLengthTypes[];

PERMON_EXTERN PetscErrorCode QPSMPGPGetCurrentStepType(QPS qps, char *stepType);
PERMON_EXTERN PetscErrorCode QPSMPGPSetAlpha(QPS qps, PetscReal alpha, QPSScalarArgType argtype);
PERMON_EXTERN PetscErrorCode QPSMPGPGetAlpha(QPS qps, PetscReal *alpha, QPSScalarArgType *argtype);
PERMON_EXTERN PetscErrorCode QPSMPGPSetGamma(QPS qps, PetscReal gamma);
PERMON_EXTERN PetscErrorCode QPSMPGPGetGamma(QPS qps, PetscReal *gamma);
PERMON_EXTERN PetscErrorCode QPSMPGPGetOperatorMaxEigenvalue(QPS qps, PetscReal *maxeig);
PERMON_EXTERN PetscErrorCode QPSMPGPSetOperatorMaxEigenvalue(QPS qps, PetscReal maxeig);
PERMON_EXTERN PetscErrorCode QPSMPGPUpdateMaxEigenvalue(QPS qps, PetscReal maxeig_update);
PERMON_EXTERN PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueTolerance(QPS qps, PetscReal tol);
PERMON_EXTERN PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueTolerance(QPS qps, PetscReal *tol);
PERMON_EXTERN PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueIterations(QPS qps, PetscInt *numit);
PERMON_EXTERN PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueIterations(QPS qps, PetscInt numit);

/* MPGPQPC */
PERMON_EXTERN PetscErrorCode QPSMPGPQPCSetAlpha(QPS qps, PetscReal alpha, QPSScalarArgType argtype);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCGetAlpha(QPS qps, PetscReal *alpha, QPSScalarArgType *argtype);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCSetGamma(QPS qps, PetscReal gamma);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCGetGamma(QPS qps, PetscReal *gamma);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCGetOperatorMaxEigenvalue(QPS qps, PetscReal *maxeig);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCSetOperatorMaxEigenvalue(QPS qps, PetscReal maxeig);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCSetOperatorMaxEigenvalueTolerance(QPS qps, PetscReal tol);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCGetOperatorMaxEigenvalueTolerance(QPS qps, PetscReal *tol);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCGetOperatorMaxEigenvalueIterations(QPS qps, PetscInt *numit);
PERMON_EXTERN PetscErrorCode QPSMPGPQPCSetOperatorMaxEigenvalueIterations(QPS qps, PetscInt numit);

/* SMALXE */
PERMON_EXTERN PetscErrorCode QPSSMALXEGetInnerQPS(QPS qps, QPS *inner);
PERMON_EXTERN PetscErrorCode QPSSMALXESetOperatorMaxEigenvalue(QPS qps, PetscReal maxeig);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalue(QPS qps, PetscReal *maxeig);
PERMON_EXTERN PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueTolerance(QPS qps, PetscReal tol);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueTolerance(QPS qps, PetscReal *tol);
PERMON_EXTERN PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueIterations(QPS qps, PetscInt numit);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueIterations(QPS qps, PetscInt *numit);
PERMON_EXTERN PetscErrorCode QPSSMALXESetInjectOperatorMaxEigenvalue(QPS qps, PetscBool flg);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetInjectOperatorMaxEigenvalue(QPS qps, PetscBool *flg);
PERMON_EXTERN PetscErrorCode QPSSMALXESetEta(QPS qps, PetscReal eta, QPSScalarArgType argtype);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetEta(QPS qps, PetscReal *eta, QPSScalarArgType *argtype);
PERMON_EXTERN PetscErrorCode QPSSMALXESetM1Initial(QPS qps, PetscReal M1_initial, QPSScalarArgType argtype);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetM1Initial(QPS qps, PetscReal *M1_initial, QPSScalarArgType *argtype);
PERMON_EXTERN PetscErrorCode QPSSMALXESetM1Update(QPS qps, PetscReal M1_update);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetM1Update(QPS qps, PetscReal *M1_update);
PERMON_EXTERN PetscErrorCode QPSSMALXESetRhoInitial(QPS qps, PetscReal rho_initial, QPSScalarArgType argtype);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetRhoInitial(QPS qps, PetscReal *rho_initial, QPSScalarArgType *argtype);
PERMON_EXTERN PetscErrorCode QPSSMALXESetRhoUpdate(QPS qps, PetscReal rho_update);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetRhoUpdate(QPS qps, PetscReal *rho_update);
PERMON_EXTERN PetscErrorCode QPSSMALXESetRhoUpdateLate(QPS qps, PetscReal rho_update_late);
PERMON_EXTERN PetscErrorCode QPSSMALXEGetRhoUpdateLate(QPS qps, PetscReal *rho_update_late);
//TODO temporary solution, monitors should be implemented more generally
PERMON_EXTERN PetscErrorCode QPSSMALXESetMonitor(QPS qps, PetscBool flg);
