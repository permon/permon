#pragma once

#include <permonqpc.h>
#include <permon/private/permonimpl.h>

typedef struct _QPCOps *QPCOps;

struct _QPCOps {
  PetscErrorCode (*setup)(QPC);
  PetscErrorCode (*destroy)(QPC);
  PetscErrorCode (*view)(QPC, PetscViewer);
  PetscErrorCode (*viewkkt)(QPC, Vec, PetscReal, PetscViewer);
  PetscErrorCode (*setfromoptions)(QPC, PetscOptionItems);
  PetscErrorCode (*reset)(QPC);
  PetscErrorCode (*getactiveset)(QPC, IS *);
  PetscErrorCode (*getfreeset)(QPC, Vec, IS *);
  PetscErrorCode (*getblocksize)(QPC, PetscInt *);
  PetscErrorCode (*islinear)(QPC, PetscBool *);
  PetscErrorCode (*issubsymmetric)(QPC, PetscBool *);
  PetscErrorCode (*getnumberofconstraints)(QPC, PetscInt *);
  PetscErrorCode (*getconstraintfunction)(QPC, Vec, Vec *);
  PetscErrorCode (*restoreconstraintfunction)(QPC, Vec, Vec *);
  PetscErrorCode (*project)(QPC, Vec, Vec);
  PetscErrorCode (*feas)(QPC, Vec, Vec, PetscScalar *);
  PetscErrorCode (*grads)(QPC, Vec, Vec, Vec, Vec);
  PetscErrorCode (*gradreduced)(QPC, Vec, Vec, PetscReal, Vec);
};

struct _p_QPC {
  PETSCHEADER(struct _QPCOps);
  Vec       lambdawork;  /* working vector with same layout as blocks of IS  */
  IS        is;          /* index set with indices corresponding to each constraint */
  IS        activeset;   /* index set with indices corresponding to active constraints */
  IS        freeset;     /* index set with indices corresponding to free constraints */
  PetscBool setchanged;  /* indicate if active/free set has changed */
  PetscReal astol;       /* active set tolerance - used e.g. in grad splitting */
  void     *data;        /* holder for misc stuff associated with a particular constraints type */
  PetscBool setupcalled; /* current state */
  PetscInt *activeset_a; /* array holding active set indices */
  PetscInt *freeset_a;   /* array holding free set indices */
  Vec       setmask;     /* global vector holding active/free set mask (0. is free) */
  Vec       setmask_sub; /* subvector on qpc->is holding active/free set mask (0. is free) */
};
