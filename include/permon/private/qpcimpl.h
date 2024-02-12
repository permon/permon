#if !defined(__QPCIMPL_H)
#define	__QPCIMPL_H
#include <permonqpc.h>
#include <permon/private/permonimpl.h>

typedef struct _QPCOps *QPCOps;

struct _QPCOps {
  PetscErrorCode (*setup)(QPC);
  PetscErrorCode (*destroy)(QPC);
  PetscErrorCode (*view)(QPC,PetscViewer);
  PetscErrorCode (*viewkkt)(QPC,Vec,PetscReal,PetscViewer);
  PetscErrorCode (*setfromoptions)(QPC,PetscOptionItems*);
  PetscErrorCode (*reset)(QPC);
  PetscErrorCode (*getactiveset)(QPC,IS*);
  PetscErrorCode (*getfreeset)(QPC,Vec,IS*);
  PetscErrorCode (*getblocksize)(QPC,PetscInt*);
  PetscErrorCode (*islinear)(QPC,PetscBool*);
  PetscErrorCode (*issubsymmetric)(QPC,PetscBool*);
  PetscErrorCode (*getnumberofconstraints)(QPC,PetscInt*);
  PetscErrorCode (*getconstraintfunction)(QPC,Vec,Vec *);
  PetscErrorCode (*restoreconstraintfunction)(QPC,Vec,Vec *);
  PetscErrorCode (*project)(QPC,Vec,Vec);
  PetscErrorCode (*feas)(QPC,Vec,Vec,PetscScalar*);
  PetscErrorCode (*grads)(QPC,Vec,Vec,Vec,Vec);
  PetscErrorCode (*gradreduced)(QPC,Vec,Vec,PetscReal,Vec);
};

struct _p_QPC {
  PETSCHEADER(struct _QPCOps);
  Vec lambdawork; /* working vector with same layout as blocks of IS  */
  IS is; /* index set with indexes corresponding to each constraint */
  IS activeset; /* index set with indexes corresponding to active constraints */
  IS activesetglobal; /* global index set with indexes corresponding to active constraints */
  IS freeset; /* index set with indexes corresponding to free constraints */
  PetscBool setchanged; /* indicate if active/free set has changed */
  PetscReal astol; /* active set tolerance - used e.g. in grad splitting */
  void *data; /* holder for misc stuff associated with a particular constraints type */
  PetscBool setupcalled; /* current state */
};

#endif

