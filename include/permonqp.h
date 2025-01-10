#pragma once

#include "permonqpc.h"

typedef struct _p_QP *QP;

PERMON_EXTERN PetscClassId QP_CLASSID;
#define QP_CLASS_NAME "qp"

typedef enum {
  QP_SCALE_NONE,
  QP_SCALE_ROWS_NORM_2,
  QP_SCALE_DDM_MULTIPLICITY
} QPScaleType;
PERMON_EXTERN const char *QPScaleTypes[];
typedef enum {
  QP_DUPLICATE_DO_NOT_COPY,
  QP_DUPLICATE_COPY_POINTERS
} QPDuplicateOption;

PERMON_EXTERN PetscErrorCode QPChainAdd(QP qp, QPDuplicateOption opt, QP *newchild);
PERMON_EXTERN PetscErrorCode QPChainPop(QP qp);
PERMON_EXTERN PetscErrorCode QPChainFind(QP qp, PetscErrorCode (*transform)(QP), QP *child);
PERMON_EXTERN PetscErrorCode QPChainGetLast(QP qp, QP *child);
PERMON_EXTERN PetscErrorCode QPChainPostSolve(QP qp);
PERMON_EXTERN PetscErrorCode QPChainSetFromOptions(QP qp);
PERMON_EXTERN PetscErrorCode QPChainSetUp(QP qp);
PERMON_EXTERN PetscErrorCode QPChainView(QP qp, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPChainViewKKT(QP qp, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPChainViewQPPF(QP qp, PetscViewer v);

PERMON_EXTERN PetscErrorCode QPAddChild(QP qp, QPDuplicateOption opt, QP *newchild);
PERMON_EXTERN PetscErrorCode QPRemoveChild(QP qp);
PERMON_EXTERN PetscErrorCode QPGetChild(QP qp, QP *child);
PERMON_EXTERN PetscErrorCode QPGetParent(QP qp, QP *parent);
PERMON_EXTERN PetscErrorCode QPGetPostSolve(QP qp, PetscErrorCode (**f)(QP, QP));
PERMON_EXTERN PetscErrorCode QPGetTransform(QP qp, PetscErrorCode (**f)(QP));

PERMON_EXTERN PetscErrorCode QPInitializePackage();
PERMON_EXTERN PetscErrorCode QPCreate(MPI_Comm comm, QP *qp);
PERMON_EXTERN PetscErrorCode QPDuplicate(QP qp1, QPDuplicateOption opt, QP *qp2);
PERMON_EXTERN PetscErrorCode QPView(QP qp, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPViewKKT(QP qp, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPReset(QP qp);
PERMON_EXTERN PetscErrorCode QPSetUpInnerObjects(QP qp);
PERMON_EXTERN PetscErrorCode QPSetUp(QP qp);
PERMON_EXTERN PetscErrorCode QPDestroy(QP *qp);
PERMON_EXTERN PetscErrorCode QPDump(QP qp);

PERMON_EXTERN PetscErrorCode QPCompareEqMultiplierWithLeastSquare(QP qp, PetscReal *norm);
PERMON_EXTERN PetscErrorCode QPComputeMissingEqMultiplier(QP qp);
PERMON_EXTERN PetscErrorCode QPComputeMissingBoxMultipliers(QP qp);
PERMON_EXTERN PetscErrorCode QPComputeEqMultiplier(QP qp, Vec lambda_E_LS, Vec BEt_lambda_E_LS);
PERMON_EXTERN PetscErrorCode QPComputeLagrangianGradient(QP qp, Vec x, Vec r, char *kkt_name[]);
PERMON_EXTERN PetscErrorCode QPComputeObjective(QP qp, Vec x, PetscReal *f);
PERMON_EXTERN PetscErrorCode QPComputeObjectiveGradient(QP qp, Vec x, Vec g);
PERMON_EXTERN PetscErrorCode QPComputeObjectiveFromGradient(QP qp, Vec x, Vec g, PetscReal *f);
PERMON_EXTERN PetscErrorCode QPComputeObjectiveAndGradient(QP qp, Vec x, Vec g, PetscReal *f);

PERMON_EXTERN PetscErrorCode QPSetInitialVector(QP qp, Vec x);
PERMON_EXTERN PetscErrorCode QPSetOperator(QP qp, Mat A);
PERMON_EXTERN PetscErrorCode QPSetPC(QP qp, PC pc);
PERMON_EXTERN PetscErrorCode QPSetOperatorNullSpace(QP qp, Mat R);
PERMON_EXTERN PetscErrorCode QPSetRhs(QP qp, Vec b);
PERMON_EXTERN PetscErrorCode QPSetRhsPlus(QP qp, Vec b);
PERMON_EXTERN PetscErrorCode QPAddEq(QP qp, Mat Beq, Vec ceq);
PERMON_EXTERN PetscErrorCode QPSetEq(QP qp, Mat Beq, Vec ceq);
PERMON_EXTERN PetscErrorCode QPGetEqMultiplicityScaling(QP qp, Vec *dE, Vec *dI);
PERMON_EXTERN PetscErrorCode QPSetIneq(QP qp, Mat Bineq, Vec cineq);
PERMON_EXTERN PetscErrorCode QPSetBox(QP qp, IS is, Vec lb, Vec ub);
PERMON_EXTERN PetscErrorCode QPSetQPPF(QP qp, QPPF pf);
PERMON_EXTERN PetscErrorCode QPSetChangeListener(QP qp, PetscErrorCode (*f)(QP));
PERMON_EXTERN PetscErrorCode QPSetChangeListenerContext(QP qp, void *ctx);
PERMON_EXTERN PetscErrorCode QPSetOptionsPrefix(QP qp, const char prefix[]);
PERMON_EXTERN PetscErrorCode QPAppendOptionsPrefix(QP qp, const char prefix[]);
PERMON_EXTERN PetscErrorCode QPSetFromOptions(QP qp);

PERMON_EXTERN PetscErrorCode QPGetVecs(QP qp, Vec *right, Vec *left);
PERMON_EXTERN PetscErrorCode QPGetSolutionVector(QP qp, Vec *x);
PERMON_EXTERN PetscErrorCode QPGetOperator(QP qp, Mat *A);
PERMON_EXTERN PetscErrorCode QPGetPC(QP qp, PC *pc);
PERMON_EXTERN PetscErrorCode QPGetOperatorNullSpace(QP qp, Mat *R);
PERMON_EXTERN PetscErrorCode QPGetRhs(QP qp, Vec *b);
PERMON_EXTERN PetscErrorCode QPGetIneq(QP qp, Mat *Bineq, Vec *cineq);
PERMON_EXTERN PetscErrorCode QPGetEq(QP qp, Mat *Beq, Vec *ceq);
PERMON_EXTERN PetscErrorCode QPGetBox(QP qp, IS *is, Vec *lb, Vec *ub);
PERMON_EXTERN PetscErrorCode QPGetQPPF(QP qp, QPPF *pf);
PERMON_EXTERN PetscErrorCode QPGetChangeListener(QP qp, PetscErrorCode (**f)(QP));
PERMON_EXTERN PetscErrorCode QPGetChangeListenerContext(QP qp, void *ctx);
PERMON_EXTERN PetscErrorCode QPIsSolved(QP qp, PetscBool *flg);
PERMON_EXTERN PetscErrorCode QPGetOptionsPrefix(QP qp, const char *prefix[]);

/* QPC stuff */
PERMON_EXTERN PetscErrorCode QPSetQPC(QP qp, QPC qpc);
PERMON_EXTERN PetscErrorCode QPGetQPC(QP qp, QPC *qpc);

/* QP transforms */
PERMON_EXTERN PetscErrorCode QPTEnforceEqByProjector(QP qp);
PERMON_EXTERN PetscErrorCode QPTEnforceEqByPenalty(QP qp, PetscReal rho_user, PetscBool rho_direct);
PERMON_EXTERN PetscErrorCode QPTHomogenizeEq(QP qp);
PERMON_EXTERN PetscErrorCode QPTOrthonormalizeEq(QP qp, MatOrthType type, MatOrthForm form);
PERMON_EXTERN PetscErrorCode QPTOrthonormalizeEqFromOptions(QP qp);
PERMON_EXTERN PetscErrorCode QPTDualize(QP qp, MatInvType invType, MatRegularizationType regType);
PERMON_EXTERN PetscErrorCode QPTRemoveGluingOfDirichletDofs(QP qp);
PERMON_EXTERN PetscErrorCode QPTFetiPrepare(QP qp, PetscBool regularize);
PERMON_EXTERN PetscErrorCode QPTFetiPrepareReuseCP(QP qp, PetscBool regularize);
PERMON_EXTERN PetscErrorCode QPTFetiPrepareReuseCPReset();
PERMON_EXTERN PetscErrorCode QPTFreezeIneq(QP qp);
PERMON_EXTERN PetscErrorCode QPTSplitBE(QP qp);
PERMON_EXTERN PetscErrorCode QPTAllInOne(QP qp, MatInvType invType, PetscBool dual, PetscBool project, PetscReal penalty, PetscBool penalty_direct, PetscBool regularize);
PERMON_EXTERN PetscErrorCode QPTFromOptions(QP qp);
PERMON_EXTERN PetscErrorCode QPTScale(QP qp);
PERMON_EXTERN PetscErrorCode QPTScaleObjectiveByScalar(QP qp, PetscScalar scale_A, PetscScalar scale_b);
PERMON_EXTERN PetscErrorCode QPTNormalizeHessian(QP qp);
PERMON_EXTERN PetscErrorCode QPTNormalizeObjective(QP qp);
PERMON_EXTERN PetscErrorCode QPTMatISToBlockDiag(QP qp);

/* MatPenalized */
PERMON_EXTERN PetscErrorCode MatCreatePenalized(QP qp, PetscReal rho, Mat *A_inner);
PERMON_EXTERN PetscErrorCode MatPenalizedSetPenalty(Mat Arho, PetscReal rho);
PERMON_EXTERN PetscErrorCode MatPenalizedUpdatePenalty(Mat Arho, PetscReal rho_update);
PERMON_EXTERN PetscErrorCode MatPenalizedGetPenalty(Mat Arho, PetscReal *rho);
PERMON_EXTERN PetscErrorCode MatPenalizedGetPenalizedTerm(Mat Arho, Mat *rhoBtB);
