#if !defined(__FLLOPIMPL_H)
#define	__FLLOPIMPL_H
#include <petsc/private/petscimpl.h>
#include <permon/private/petscimpl.h>
#include <fllopsys.h>

struct _p_FLLOP {
  PETSCHEADER(int);
};

struct _n_StateContainer {
  PetscObjectState state;
};
#endif
