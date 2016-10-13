#if !defined(__MPGPIMPL_H)
#define __MPGPIMPL_H
#include <private/qpsimpl.h>

typedef struct {
  PetscReal alpha;
  PetscReal alpha_user;
  QPSScalarArgType alpha_type;
  PetscReal gamma;
  PetscReal maxeig;
  PetscReal maxeig_tol;
  PetscInt  maxeig_iter;

  /* compute norm of the gradient types */  
  PetscBool compute_phinorm;
  PetscBool compute_betanorm;
  PetscBool compute_gPalphanorm;
  
  PetscReal phinorm;
  PetscReal betanorm;
  PetscReal gPalphanorm;

  PetscInt  nmv;              /* ... matrix-vector mult. counter      */
  PetscInt  ncg;              /* ... cg step counter                  */ 
  PetscInt  nprop;            /* ... proportional step counter        */
  PetscInt  nexp;             /* ... expansion step counter           */
} QPS_MPGP;

#endif
