#pragma once

#include "permonqp.h"

typedef enum {
  FETI_LOCAL,
  FETI_GLOBAL_DECOMPOSED,
  FETI_GLOBAL_UNDECOMPOSED
} QPFetiNumberingType;

typedef enum {
  FETI_GLUING_NONRED = 0,
  FETI_GLUING_FULL   = 1,
  FETI_GLUING_ORTH   = 2
} FetiGluingType;
PERMON_EXTERN const char *const FetiGluingTypes[];

/* if (!enforceByB) MatZeroRowsIS(qp->A, dbcis, diag, qp->x, qp->b) */
PERMON_EXTERN PetscErrorCode QPFetiSetDirichlet(QP qp, IS dbcis, QPFetiNumberingType numtype, PetscBool enforce_by_B);
/* l2g_dof_map is mapping from the local dof indexing of decomposed problem to global dof indexing of undecomposed problem */
PERMON_EXTERN PetscErrorCode QPFetiSetLocalToGlobalMapping(QP qp, IS l2g_dof_map);
PERMON_EXTERN PetscErrorCode QPFetiSetInterfaceToGlobalMapping(QP qp, IS i2g);
PERMON_EXTERN PetscErrorCode QPFetiSetUp(QP qp);
