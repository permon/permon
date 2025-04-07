#pragma once

#include <permon/private/qpsimpl.h>

typedef struct {
  PetscReal gtol;
  PetscReal norm_rhs_outer, ttol_outer;
  PetscReal MNormBu;
  QPS       qps_outer;
  QP        qp_outer;
} QPSConvergedCtx_Inner_SMALXE;

typedef struct {
  QPS                           inner;
  QP                            qp_penalized;
  QPSConvergedCtx_Inner_SMALXE *cctx_inner;
  Vec                           Bu;

  PetscReal        M1_user;
  PetscReal        M1_initial;
  QPSScalarArgType M1_type;
  PetscReal        M1_update; /* M1_update > 1  (also known as beta) */
  PetscReal        M1;
  PetscInt         M1_updates;
  PetscInt         M1_hits;

  PetscReal rtol_E;

  PetscReal        rho_user; /* rho > 0 */
  QPSScalarArgType rho_type;
  PetscReal        rho_update;      /* rho_update > 1 */
  PetscReal        rho_update_late; /* rho_update_late > 1 */
  PetscInt         rho_updates;

  PetscReal        eta_user, eta; /* eta > 0 */
  QPSScalarArgType eta_type;
  PetscInt         eta_hits;

  PetscReal update_threshold;

  PetscReal maxeig;
  PetscReal maxeig_tol;
  PetscInt  maxeig_iter;
  PetscBool inject_maxeig, inject_maxeig_set;

  PetscBool monitor, monitor_outer;
  PetscBool monitor_excel;
  PetscBool get_lambda, get_Bt_lambda;
  PetscInt  inner_iter_min;
  PetscInt  inner_no_gtol_stop;

  PetscInt  state;
  PetscInt  inner_iter_accu;
  PetscBool setfromoptionscalled;
  PetscReal normBu, normBu_old, normBu_prev;
  PetscInt  offset;
  PetscReal enorm;

  PetscBool lag_enabled;
  PetscBool lag_monitor;
  PetscBool lag_compare;
  PetscInt  norm_update_lag_offset, Jstart, Jstep, Jend;
  PetscReal lower, upper;

  PetscBool knoll;
  PetscErrorCode (*updateNormBu)(QPS qps, Vec u, PetscReal *normBu, PetscReal *enorm);
} QPS_SMALXE;

PERMON_EXTERN PetscErrorCode QPSCreate_SMALXE(QPS qps);
PERMON_INTERN PetscErrorCode QPSSetFromOptions_SMALXE(QPS qps, PetscOptionItems PetscOptionsObject);
PERMON_INTERN PetscErrorCode QPSSetUp_SMALXE(QPS qps);
PERMON_INTERN PetscErrorCode QPSConverged_Inner_SMALXE(QPS qps_inner, KSPConvergedReason *reason);
PERMON_INTERN PetscErrorCode QPSConvergedSetUp_Inner_SMALXE(QPS qps_inner);
PERMON_INTERN PetscErrorCode QPSConvergedCreate_Inner_SMALXE(QPS qps_outer, void **ctx);
PERMON_INTERN PetscErrorCode QPSConvergedDestroy_Inner_SMALXE(void *ctx);
PERMON_INTERN PetscErrorCode QPSSMALXEUpdate_SMALXE(QPS qps, PetscReal Lag_old, PetscReal Lag, PetscReal rho);
PERMON_INTERN PetscErrorCode QPSSMALXEUpdateNormBu_SMALXE(QPS qps, Vec u, PetscReal *normBu, PetscReal *enorm);
