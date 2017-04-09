#if !defined(__QPIMPL_H)
#define	__QPIMPL_H
#include <fllopqp.h>
#include <permon/private/fllopimpl.h>

struct _p_QP {
  PETSCHEADER(int);

  /* general QP properties */
  QP               parent,child;
  QPPF             pf;
  PetscInt         id;
  PetscBool        solved;
  PetscBool        setupcalled;
  PetscInt         setfromoptionscalled;
  void             *changeListenerCtx;
  PetscErrorCode   (*changeListener)(QP);

  /* Hessian matrix, Hessian symmetry flag, Hessian kernel, and right hand-side vector */
  Mat              A;
  Mat              R;
  Vec              b;

  /* preconditioner */
  PC               pc;

  /* solution vectors */
  Vec              x;
  Vec              xwork;

  /* general linear constraints */
  Mat              B;
  Vec              c;
  Vec              lambda, Bt_lambda;
  
  /* linear equality constraints */
  Mat              BE;
  PetscInt         BE_nest_count;
  Vec              cE;
  Vec              lambda_E;
  
  /* linear inequality constraints */
  Mat              BI;
  Vec              cI;
  Vec              lambda_I;

  /* box constraints */
  Vec              lb;
  Vec              lambda_lb;
  Vec              ub;
  Vec              lambda_ub;

  /* post-processing action after THIS QP's solve */
  void             *postSolveCtx;
  PetscErrorCode   (*postSolve)(QP,QP);
  PetscErrorCode   (*postSolveCtxDestroy)(void*);
  PetscErrorCode   (*transform)(QP);
  char             transform_name[FLLOP_MAX_NAME_LEN];
};

typedef struct {
  Vec dE, dI, dO;
} QPTScale_Ctx;

typedef struct {
  PetscReal norm_A, norm_b;
} QPTNormalizeObjective_Ctx;

typedef struct {
  PetscReal scale_A, scale_b;
} QPTScaleObjectiveByScalar_Ctx;

FLLOP_EXTERN PetscLogEvent QPT_HomogenizeEq, QPT_OrthonormalizeEq, QPT_EnforceEqByProjector, QPT_EnforceEqByPenalty, QPT_Dualize, QPT_Dualize_AssembleG, QPT_Dualize_FactorK, QPT_Dualize_PrepareBt, QPT_AllInOne;
FLLOP_EXTERN PetscLogEvent QPT_SplitBE;

FLLOP_INTERN PetscErrorCode QPCompute_BEt_lambda(QP qp,Vec *BEt_lambda);
FLLOP_INTERN PetscErrorCode QPDefaultPostSolve(QP child,QP parent);
FLLOP_INTERN PetscErrorCode QPSetEqMultiplier(QP qp, Vec lambda_E);
FLLOP_INTERN PetscErrorCode QPSetIneqMultiplier(QP qp, Vec lambda_I);
FLLOP_INTERN PetscErrorCode QPSetLowerBoundMultiplier(QP qp, Vec lambda_lb);
FLLOP_INTERN PetscErrorCode QPSetUpperBoundMultiplier(QP qp, Vec lambda_ub);
FLLOP_INTERN PetscErrorCode QPSetWorkVector(QP qp,Vec xwork);
#endif
