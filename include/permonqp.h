#if !defined(__PERMONQP_H)
#define	__PERMONQP_H
#include "permonqpc.h"

typedef struct _p_QP* QP;

FLLOP_EXTERN PetscClassId QP_CLASSID;
#define QP_CLASS_NAME  "qp"

typedef enum {QP_SCALE_NONE,QP_SCALE_ROWS_NORM_2,QP_SCALE_DDM_MULTIPLICITY} QPScaleType;
FLLOP_EXTERN const char *QPScaleTypes[];
typedef enum {QP_DUPLICATE_DO_NOT_COPY,QP_DUPLICATE_COPY_POINTERS} QPDuplicateOption;

FLLOP_EXTERN PetscErrorCode QPChainAdd(QP qp,QPDuplicateOption opt,QP *newchild);
FLLOP_EXTERN PetscErrorCode QPChainPop(QP qp);
FLLOP_EXTERN PetscErrorCode QPChainFind(QP qp,PetscErrorCode(*transform)(QP),QP *child);
FLLOP_EXTERN PetscErrorCode QPChainGetLast(QP qp,QP *child);
FLLOP_EXTERN PetscErrorCode QPChainPostSolve(QP qp);
FLLOP_EXTERN PetscErrorCode QPChainSetFromOptions(QP qp);
FLLOP_EXTERN PetscErrorCode QPChainSetUp(QP qp);
FLLOP_EXTERN PetscErrorCode QPChainView(QP qp,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPChainViewKKT(QP qp,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPChainViewQPPF(QP qp,PetscViewer v);

FLLOP_EXTERN PetscErrorCode QPAddChild(QP qp,QPDuplicateOption opt,QP *newchild);
FLLOP_EXTERN PetscErrorCode QPRemoveChild(QP qp);
FLLOP_EXTERN PetscErrorCode QPGetChild(QP qp,QP *child);
FLLOP_EXTERN PetscErrorCode QPGetParent(QP qp,QP *parent);
FLLOP_EXTERN PetscErrorCode QPGetPostSolve(QP qp,PetscErrorCode (**f)(QP,QP));
FLLOP_EXTERN PetscErrorCode QPGetTransform(QP qp,PetscErrorCode (**f)(QP));

FLLOP_EXTERN PetscErrorCode QPInitializePackage();
FLLOP_EXTERN PetscErrorCode QPCreate(MPI_Comm comm,QP *qp);
FLLOP_EXTERN PetscErrorCode QPDuplicate(QP qp1,QPDuplicateOption opt,QP *qp2);
FLLOP_EXTERN PetscErrorCode QPView(QP qp,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPViewKKT(QP qp,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPReset(QP qp);
FLLOP_EXTERN PetscErrorCode QPSetUpInnerObjects(QP qp);
FLLOP_EXTERN PetscErrorCode QPSetUp(QP qp);
FLLOP_EXTERN PetscErrorCode QPDestroy(QP *qp);
FLLOP_EXTERN PetscErrorCode QPDump(QP qp);

FLLOP_EXTERN PetscErrorCode QPCheckNullSpace(QP qp,PetscReal tol);
FLLOP_EXTERN PetscErrorCode QPCompareEqMultiplierWithLeastSquare(QP qp,PetscReal *norm);
FLLOP_EXTERN PetscErrorCode QPComputeMissingEqMultiplier(QP qp);
FLLOP_EXTERN PetscErrorCode QPComputeMissingBoxMultipliers(QP qp);
FLLOP_EXTERN PetscErrorCode QPComputeEqMultiplier(QP qp,Vec lambda_E_LS,Vec BEt_lambda_E_LS);
FLLOP_EXTERN PetscErrorCode QPComputeLagrangianGradient(QP qp, Vec x, Vec r, char *kkt_name[]);
FLLOP_EXTERN PetscErrorCode QPComputeObjective(QP qp, Vec x, PetscReal *f);
FLLOP_EXTERN PetscErrorCode QPComputeObjectiveGradient(QP qp, Vec x, Vec g);
FLLOP_EXTERN PetscErrorCode QPComputeObjectiveFromGradient(QP qp, Vec x, Vec g, PetscReal *f);
FLLOP_EXTERN PetscErrorCode QPComputeObjectiveAndGradient(QP qp, Vec x, Vec g, PetscReal *f);
FLLOP_EXTERN PetscErrorCode QPRemoveInactiveBounds(QP qp);

FLLOP_EXTERN PetscErrorCode QPSetInitialVector(QP qp,Vec x);
FLLOP_EXTERN PetscErrorCode QPSetOperator(QP qp,Mat A);
FLLOP_EXTERN PetscErrorCode QPSetPC(QP qp,PC pc);
FLLOP_EXTERN PetscErrorCode QPSetOperatorNullSpace(QP qp,Mat R);
FLLOP_EXTERN PetscErrorCode QPSetRhs(QP qp,Vec b);
FLLOP_EXTERN PetscErrorCode QPSetRhsPlus(QP qp,Vec b);
FLLOP_EXTERN PetscErrorCode QPAddEq(QP qp,Mat Beq,Vec ceq);
FLLOP_EXTERN PetscErrorCode QPSetEq(QP qp,Mat Beq,Vec ceq);
FLLOP_EXTERN PetscErrorCode QPGetEqMultiplicityScaling(QP qp, Vec *dE, Vec *dI);
FLLOP_EXTERN PetscErrorCode QPSetIneq(QP qp,Mat Bineq,Vec cineq);
FLLOP_EXTERN PetscErrorCode QPSetBox(QP qp,Vec lb,Vec ub);
FLLOP_EXTERN PetscErrorCode QPSetQPPF(QP qp,QPPF pf);
FLLOP_EXTERN PetscErrorCode QPSetChangeListener(QP qp,PetscErrorCode (*f)(QP));
FLLOP_EXTERN PetscErrorCode QPSetChangeListenerContext(QP qp,void *ctx);
FLLOP_EXTERN PetscErrorCode QPSetOptionsPrefix(QP qp,const char prefix[]);
FLLOP_EXTERN PetscErrorCode QPAppendOptionsPrefix(QP qp,const char prefix[]);
FLLOP_EXTERN PetscErrorCode QPSetFromOptions(QP qp);

FLLOP_EXTERN PetscErrorCode QPGetVecs(QP qp,Vec *right,Vec *left);
FLLOP_EXTERN PetscErrorCode QPGetSolutionVector(QP qp,Vec *x);
FLLOP_EXTERN PetscErrorCode QPGetOperator(QP qp,Mat *A);
FLLOP_EXTERN PetscErrorCode QPGetPC(QP qp,PC *pc);
FLLOP_EXTERN PetscErrorCode QPGetOperatorNullSpace(QP qp,Mat *R);
FLLOP_EXTERN PetscErrorCode QPGetRhs(QP qp,Vec *b);
FLLOP_EXTERN PetscErrorCode QPGetIneq(QP qp,Mat *Bineq,Vec *cineq);
FLLOP_EXTERN PetscErrorCode QPGetEq(QP qp,Mat *Beq,Vec *ceq);
FLLOP_EXTERN PetscErrorCode QPGetBox(QP qp,Vec *lb,Vec *ub);
FLLOP_EXTERN PetscErrorCode QPGetBoxQPC(QP qp,Vec *lb,Vec *ub);
FLLOP_EXTERN PetscErrorCode QPGetQPPF(QP qp,QPPF *pf);
FLLOP_EXTERN PetscErrorCode QPGetChangeListener(QP qp,PetscErrorCode (**f)(QP));
FLLOP_EXTERN PetscErrorCode QPGetChangeListenerContext(QP qp,void *ctx);
FLLOP_EXTERN PetscErrorCode QPIsSolved(QP qp,PetscBool *flg);
FLLOP_EXTERN PetscErrorCode QPGetOptionsPrefix(QP qp,const char *prefix[]);

/* QPC stuff */
FLLOP_EXTERN PetscErrorCode QPSetQPC(QP qp,QPC qpc);
FLLOP_EXTERN PetscErrorCode QPGetQPC(QP qp, QPC *qpc);
FLLOP_EXTERN PetscErrorCode QPSetBoxQPC(QP qp,IS is, Vec lb,Vec ub);

/* QP transforms */
FLLOP_EXTERN PetscErrorCode QPTEnforceEqByProjector(QP qp);
FLLOP_EXTERN PetscErrorCode QPTEnforceEqByPenalty(QP qp,PetscReal rho_user,PetscBool rho_direct);
FLLOP_EXTERN PetscErrorCode QPTHomogenizeEq(QP qp);
FLLOP_EXTERN PetscErrorCode QPTOrthonormalizeEq(QP qp,MatOrthType type,MatOrthForm form);
FLLOP_EXTERN PetscErrorCode QPTOrthonormalizeEqFromOptions(QP qp);
FLLOP_EXTERN PetscErrorCode QPTDualize(QP qp,MatInvType invType,MatRegularizationType regType);
FLLOP_EXTERN PetscErrorCode QPTRemoveGluingOfDirichletDofs(QP qp);
FLLOP_EXTERN PetscErrorCode QPTFetiPrepare(QP qp,PetscBool regularize);
FLLOP_EXTERN PetscErrorCode QPTFetiPrepareReuseCP(QP qp,PetscBool regularize);
FLLOP_EXTERN PetscErrorCode QPTFetiPrepareReuseCPReset();
FLLOP_EXTERN PetscErrorCode QPTFreezeIneq(QP qp);
FLLOP_EXTERN PetscErrorCode QPTSplitBE(QP qp);
FLLOP_EXTERN PetscErrorCode QPTAllInOne(QP qp,MatInvType invType,PetscBool dual,PetscBool project,PetscReal penalty,PetscBool penalty_direct,PetscBool regularize);
FLLOP_EXTERN PetscErrorCode QPTFromOptions(QP qp);
FLLOP_EXTERN PetscErrorCode QPTScale(QP qp);
FLLOP_EXTERN PetscErrorCode QPTScaleObjectiveByScalar(QP qp,PetscScalar scale_A,PetscScalar scale_b);
FLLOP_EXTERN PetscErrorCode QPTNormalizeHessian(QP qp);
FLLOP_EXTERN PetscErrorCode QPTNormalizeObjective(QP qp);
FLLOP_EXTERN PetscErrorCode QPTMatISToBlockDiag(QP qp);

/* MatPenalized */
FLLOP_EXTERN PetscErrorCode MatCreatePenalized(QP qp,PetscReal rho,Mat *A_inner);
FLLOP_EXTERN PetscErrorCode MatPenalizedSetPenalty(Mat Arho,PetscReal rho);
FLLOP_EXTERN PetscErrorCode MatPenalizedUpdatePenalty(Mat Arho,PetscReal rho_update);
FLLOP_EXTERN PetscErrorCode MatPenalizedGetPenalty(Mat Arho,PetscReal *rho);
FLLOP_EXTERN PetscErrorCode MatPenalizedGetPenalizedTerm(Mat Arho,Mat *rhoBtB);

#endif
