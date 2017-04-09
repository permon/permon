#if !defined(__FLLOPQPS_H)
#define	__FLLOPQPS_H
#include "permonqp.h"
#include <petsctao.h>
        
typedef struct _p_QPS* QPS;

FLLOP_EXTERN PetscClassId QPS_CLASSID;
#define QPS_CLASS_NAME  "qps"

#define QPSType char*
#define QPSKSP          "ksp"
#define QPSMPGP         "mpgp"
#define QPSMPGPQPC      "mpgpqpc"
#define QPSPCPG         "pcpg"
#define QPSSMALXE       "smalxe"
#define QPSTAO          "tao"

typedef enum {QPS_ARG_MULTIPLE=0, QPS_ARG_DIRECT=1} QPSScalarArgType;

FLLOP_EXTERN PetscErrorCode QPSInitializePackage(void);
FLLOP_EXTERN PetscErrorCode QPSFinalizePackage(void);

FLLOP_EXTERN PetscFunctionList QPSList;
FLLOP_EXTERN PetscBool  QPSRegisterAllCalled;
FLLOP_EXTERN PetscErrorCode QPSRegisterAll(void);
FLLOP_EXTERN PetscErrorCode QPSRegister(const char[],PetscErrorCode (*)(QPS));

FLLOP_EXTERN PetscErrorCode QPSCreate(MPI_Comm comm,QPS *qps_new);
FLLOP_EXTERN PetscErrorCode QPSView(QPS qps,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPSViewConvergence(QPS qps, PetscViewer viewer);
FLLOP_EXTERN PetscErrorCode QPSDestroy(QPS *qps);
FLLOP_EXTERN PetscErrorCode QPSSetFromOptions(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSSetUp(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSReset(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSSolve(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSPostSolve(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSIsQPCompatible(QPS qps,QP qp,PetscBool *flg);

FLLOP_EXTERN PetscErrorCode QPSSetDefaultType(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSSetDefaultTypeIfNotSpecified(QPS qps);
FLLOP_EXTERN PetscErrorCode QPSSetType(QPS qps,const QPSType type);
FLLOP_EXTERN PetscErrorCode QPSSetQP(QPS qps,QP qp);
FLLOP_EXTERN PetscErrorCode QPSSetTolerances(QPS qps,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits);
FLLOP_EXTERN PetscErrorCode QPSSetOptionsPrefix(QPS qps,const char prefix[]);
FLLOP_EXTERN PetscErrorCode QPSAppendOptionsPrefix(QPS qps,const char prefix[]);
FLLOP_EXTERN PetscErrorCode QPSSetConvergenceTest(QPS qps,PetscErrorCode (*converge)(QPS,QP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *cctx,PetscErrorCode (*destroy)(void*));
FLLOP_EXTERN PetscErrorCode QPSSetAutoPostSolve(QPS qps,PetscBool flg);

FLLOP_EXTERN PetscErrorCode QPSGetType(QPS qps,const QPSType *type);
FLLOP_EXTERN PetscErrorCode QPSGetQP(QPS qps,QP *qp);
FLLOP_EXTERN PetscErrorCode QPSGetSolvedQP(QPS qps,QP *qp);
FLLOP_EXTERN PetscErrorCode QPSGetTolerances(QPS qps,PetscReal *rtol,PetscReal *abstol,PetscReal *dtol,PetscInt *maxits);
FLLOP_EXTERN PetscErrorCode QPSGetOptionsPrefix(QPS qps,const char *prefix[]);
FLLOP_EXTERN PetscErrorCode QPSGetConvergenceContext(QPS,void **);
FLLOP_EXTERN PetscErrorCode QPSGetConvergedReason(QPS,KSPConvergedReason *);
FLLOP_EXTERN PetscErrorCode QPSGetResidualNorm(QPS qps,PetscReal *rnorm);
FLLOP_EXTERN PetscErrorCode QPSGetIterationNumber(QPS qps,PetscInt *its);
FLLOP_EXTERN PetscErrorCode QPSGetAutoPostSolve(QPS qps,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode QPSGetVecs(QPS qps,PetscInt rightn, Vec **right,PetscInt leftn,Vec **left);

FLLOP_EXTERN PetscErrorCode QPSConvergedSkip(QPS qps,QP qp,PetscInt i,PetscReal rnorm,KSPConvergedReason *reason,void *ctx);
FLLOP_EXTERN PetscErrorCode QPSConvergedDefault(QPS qps,QP qp,PetscInt i,PetscReal rnorm,KSPConvergedReason *reason,void*);
FLLOP_EXTERN PetscErrorCode QPSConvergedDefaultSetUp(void *ctx, QPS qps);
FLLOP_EXTERN PetscErrorCode QPSConvergedDefaultSetRhsForDivergence(void *cctx, Vec b);
FLLOP_EXTERN PetscErrorCode QPSConvergedDefaultDestroy(void *);
FLLOP_EXTERN PetscErrorCode QPSConvergedDefaultCreate(void **);

FLLOP_EXTERN PetscErrorCode QPSSetWorkVecs(QPS,PetscInt);
FLLOP_EXTERN PetscErrorCode QPSDestroyDefault(QPS);

/* QPSMonitor */
FLLOP_EXTERN PetscErrorCode QPSMonitor(QPS,PetscInt,PetscReal);
FLLOP_EXTERN PetscErrorCode QPSMonitorSet(QPS,PetscErrorCode (*)(QPS,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void**));
FLLOP_EXTERN PetscErrorCode QPSMonitorCancel(QPS);
FLLOP_EXTERN PetscErrorCode QPSGetMonitorContext(QPS,void **);
FLLOP_EXTERN PetscErrorCode QPSGetResidualHistory(QPS,PetscReal*[],PetscInt *);
FLLOP_EXTERN PetscErrorCode QPSSetResidualHistory(QPS,PetscReal[],PetscInt,PetscBool );
FLLOP_EXTERN PetscErrorCode QPSMonitorDefault(QPS qps,PetscInt n,PetscReal rnorm,void *dummy);

/* *** type-specific stuff *** */
/* KSP */
FLLOP_EXTERN PetscErrorCode QPSKSPSetKSP(QPS qps,KSP ksp);
FLLOP_EXTERN PetscErrorCode QPSKSPGetKSP(QPS qps,KSP *ksp);
FLLOP_EXTERN PetscErrorCode QPSKSPSetType(QPS qps,KSPType type);
FLLOP_EXTERN PetscErrorCode QPSKSPGetType(QPS qps,KSPType *type);

FLLOP_EXTERN PetscErrorCode QPSTaoGetTao(QPS qps,Tao *tao);

/* MPGP */
FLLOP_EXTERN PetscErrorCode QPSMPGPGetCurrentStepType(QPS qps,char *stepType);
FLLOP_EXTERN PetscErrorCode QPSMPGPSetAlpha(QPS qps,PetscReal alpha,QPSScalarArgType argtype);
FLLOP_EXTERN PetscErrorCode QPSMPGPGetAlpha(QPS qps,PetscReal *alpha,QPSScalarArgType *argtype);
FLLOP_EXTERN PetscErrorCode QPSMPGPSetGamma(QPS qps,PetscReal gamma);
FLLOP_EXTERN PetscErrorCode QPSMPGPGetGamma(QPS qps,PetscReal *gamma);
FLLOP_EXTERN PetscErrorCode QPSMPGPGetOperatorMaxEigenvalue(QPS qps,PetscReal *maxeig);
FLLOP_EXTERN PetscErrorCode QPSMPGPSetOperatorMaxEigenvalue(QPS qps,PetscReal maxeig);
FLLOP_EXTERN PetscErrorCode QPSMPGPUpdateMaxEigenvalue(QPS qps, PetscReal maxeig_update);
FLLOP_EXTERN PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal tol);
FLLOP_EXTERN PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal *tol);
FLLOP_EXTERN PetscErrorCode QPSMPGPGetOperatorMaxEigenvalueIterations(QPS qps,PetscInt *numit);
FLLOP_EXTERN PetscErrorCode QPSMPGPSetOperatorMaxEigenvalueIterations(QPS qps,PetscInt numit);

/* MPGPQPC */
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCSetAlpha(QPS qps,PetscReal alpha,QPSScalarArgType argtype);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCGetAlpha(QPS qps,PetscReal *alpha,QPSScalarArgType *argtype);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCSetGamma(QPS qps,PetscReal gamma);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCGetGamma(QPS qps,PetscReal *gamma);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCGetOperatorMaxEigenvalue(QPS qps,PetscReal *maxeig);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCSetOperatorMaxEigenvalue(QPS qps,PetscReal maxeig);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCSetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal tol);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCGetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal *tol);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCGetOperatorMaxEigenvalueIterations(QPS qps,PetscInt *numit);
FLLOP_EXTERN PetscErrorCode QPSMPGPQPCSetOperatorMaxEigenvalueIterations(QPS qps,PetscInt numit);

//TODO move to Constraints
FLLOP_EXTERN PetscErrorCode QPGetScaledProjectedGradient(QP qp, PetscReal alpha, Vec galpha);

/* SMALXE */
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetInnerQPS(QPS qps,QPS *inner);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetOperatorMaxEigenvalue(QPS qps,PetscReal maxeig);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalue(QPS qps,PetscReal *maxeig);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal tol);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueTolerance(QPS qps,PetscReal *tol);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetOperatorMaxEigenvalueIterations(QPS qps,PetscInt numit);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetOperatorMaxEigenvalueIterations(QPS qps,PetscInt *numit);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetInjectOperatorMaxEigenvalue(QPS qps,PetscBool flg);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetInjectOperatorMaxEigenvalue(QPS qps,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetEta(QPS qps,PetscReal eta,QPSScalarArgType argtype);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetEta(QPS qps,PetscReal *eta,QPSScalarArgType *argtype);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetM1Initial(QPS qps,PetscReal M1_initial,QPSScalarArgType argtype);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetM1Initial(QPS qps,PetscReal *M1_initial,QPSScalarArgType *argtype);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetM1Update(QPS qps,PetscReal M1_update);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetM1Update(QPS qps,PetscReal *M1_update);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetRhoInitial(QPS qps,PetscReal rho_initial,QPSScalarArgType argtype);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetRhoInitial(QPS qps,PetscReal *rho_initial,QPSScalarArgType *argtype);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetRhoUpdate(QPS qps,PetscReal rho_update);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetRhoUpdate(QPS qps,PetscReal *rho_update);
FLLOP_EXTERN PetscErrorCode QPSSMALXESetRhoUpdateLate(QPS qps,PetscReal rho_update_late);
FLLOP_EXTERN PetscErrorCode QPSSMALXEGetRhoUpdateLate(QPS qps,PetscReal *rho_update_late);
//TODO temporary solution, monitors should be implemented more generally
FLLOP_EXTERN PetscErrorCode QPSSMALXESetMonitor(QPS qps,PetscBool flg);

#endif

