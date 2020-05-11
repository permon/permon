#if !defined(__GPIMPL_H)
#define __GPIMPL_H
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
  PetscReal bchop_tol;

  PetscReal gfnorm;
  PetscReal gcnorm;

  PetscInt  nmv;              /* ... matrix-vector mult. counter      */
  PetscInt  ncg;              /* ... cg step counter                  */
  PetscInt  nprop;            /* ... proportional step counter        */
  PetscInt  nexp;             /* ... expansion step counter           */
  PetscInt  nfinc;            /* ... functional increase counter      */
  PetscInt  nfall;            /* ... fallback step counter            */
  char      currentStepType;

  QPSMPGPExpansionType       exptype;
  QPSMPGPExpansionLengthType explengthtype;
  PetscErrorCode             (*expansion)(QPS,PetscReal,PetscReal);
  Vec                        expdirection;
  Vec                        explengthvec;
  Vec                        explengthvecold;
  Vec                        xold;
  PetscBool                  expproject;
  PetscBool                  resetalpha;
  PetscBool                  fallback;
  PetscBool                  fallback2;

  IS isactive,isactive_old;
  
  /* line search */
  PetscErrorCode (*linesearch)(QPS); /*                                */
  PetscReal      *ls_f;                /* cost function value array      */
  PetscReal      ls_alpha;             /* step length                    */
  PetscReal      ls_beta;              /* multiplier for alpha increment */
  PetscReal      ls_gamma;             /* Armijo rule parametr           */
  PetscInt       ls_M;                 /* size of ls_f                   */
  PetscInt       ls_maxit;             /* max line search iterations     */

  /* step length */
  PetscErrorCode (*steplength)(QPS); /*                                */
  PetscReal      sl_alphamin;          /* minimal step length            */
  PetscReal      sl_alphamax;          /* maximal step length            */
  PetscReal      sl_alpha;             /* step length                    */
  PetscReal      sl_tau;               /* variability control            */
  PetscReal      sl_fact;              /* sl_tau update factor           */
  PetscReal      *sl_bb2;              /* bb2 step lengths array         */
  PetscInt       sl_M;                 /* size of sl_bb2                 */

} QPS_GP;

#endif
