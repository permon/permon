#pragma once

#include <permonqps.h>
#include <permon/private/permonimpl.h>
#include <permon/private/qpimpl.h>

/* Maximum number of monitors you can run with a single QPS */
#define MAXQPSMONITORS 5

typedef struct _QPSOps *QPSOps;

struct _QPSOps {
  PetscErrorCode (*solve)(QPS);
  PetscErrorCode (*setup)(QPS);
  PetscErrorCode (*destroy)(QPS);
  PetscErrorCode (*view)(QPS, PetscViewer);
  PetscErrorCode (*viewconvergence)(QPS, PetscViewer);
  PetscErrorCode (*setfromoptions)(QPS, PetscOptionItems);
  PetscErrorCode (*reset)(QPS);
  PetscErrorCode (*resetstatistics)(QPS);
  PetscErrorCode (*isqpcompatible)(QPS, QP, PetscBool *);
  PetscErrorCode (*monitor)(QPS, PetscInt, PetscViewer);
  PetscErrorCode (*monitorcostfunction)(QPS, PetscInt, PetscViewer);
};

struct _p_QPS {
  PETSCHEADER(struct _QPSOps);

  /* properties */
  QP        topQP, solQP;
  PetscReal rtol;
  PetscReal atol;
  PetscReal divtol;
  PetscInt  max_it;
  PetscBool autoPostSolve;
  PetscBool user_type;

  /* holder for misc stuff associated with a particular iterative solver */
  void *data;

  /* work vectors */
  PetscInt          nwork;
  Vec              *work;
  PetscObjectState *work_state;
  PetscObjectState  xstate;

  /* convergence tests */
  PetscErrorCode (*convergencetest)(QPS, KSPConvergedReason *);
  PetscErrorCode (*convergencetestdestroy)(void *);
  void *cnvctx;

  /* current state */
  PetscReal          rnorm;
  PetscInt           iteration;
  PetscInt           iterations_accumulated;
  PetscInt           nsolves;
  PetscBool          setupcalled;
  PetscBool          postsolvecalled;
  KSPConvergedReason reason;

  /* monitor */
  PetscReal *res_hist;                                                         /* If !0 stores residual at iterations*/
  PetscReal *res_hist_alloc;                                                   /* If !0 means user did not provide buffer, needs deallocation */
  PetscInt   res_hist_len;                                                     /* current size of residual history array */
  PetscInt   res_hist_max;                                                     /* actual amount of data in residual_history */
  PetscBool  res_hist_reset;                                                   /* reset history to size zero for each new solve */
  PetscErrorCode (*monitor[MAXQPSMONITORS])(QPS, PetscInt, PetscReal, void *); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXQPSMONITORS])(void **);                   /* */
  void    *monitorcontext[MAXQPSMONITORS];                                     /* residual calculation, allows user */
  PetscInt numbermonitors;                                                     /* to, for instance, print residual norm, etc. */
};

typedef struct {
  PetscReal norm_rhs, norm_rhs_div, ttol;
  PetscBool setup_called;
} QPSConvergedDefaultCtx;

PERMON_EXTERN PetscLogEvent QPS_Solve, QPS_Solve_solve, QPS_PostSolve;

PERMON_INTERN PetscErrorCode QPSWorkVecStateUpdate(QPS qps, PetscInt idx);
PERMON_INTERN PetscErrorCode QPSWorkVecStateChanged(QPS qps, PetscInt idx, PetscBool *flg);
PERMON_INTERN PetscErrorCode QPSSolutionVecStateUpdate(QPS qps);
PERMON_INTERN PetscErrorCode QPSSolutionVecStateChanged(QPS qps, PetscBool *flg);
