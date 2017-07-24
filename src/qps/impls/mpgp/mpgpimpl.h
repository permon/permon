#if !defined(__MPGPIMPL_H)
#define __MPGPIMPL_H
#include <permon/private/qpsimpl.h>

typedef struct {
  PetscReal alpha;
  PetscReal alpha_user;
  QPSScalarArgType alpha_type;
  PetscReal gamma;
  PetscReal maxeig;
  PetscReal maxeig_tol;
  PetscInt  maxeig_iter;
  PetscReal btol;

  PetscReal phinorm;
  PetscReal betanorm;

  PetscInt  nmv;              /* ... matrix-vector mult. counter      */
  PetscInt  ncg;              /* ... cg step counter                  */ 
  PetscInt  nprop;            /* ... proportional step counter        */
  PetscInt  nexp;             /* ... expansion step counter           */
  char      currentStepType;
} QPS_MPGP;

#endif
