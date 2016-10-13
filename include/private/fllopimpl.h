#if !defined(__FLLOPIMPL_H)
#define	__FLLOPIMPL_H
#if PETSC_VERSION_MINOR<6
#include <petsc-private/petscimpl.h>
#else
#include <petsc/private/petscimpl.h>
#endif
#include <private/petscimpl.h>
#include <fllopsys.h>

struct _p_FLLOP {
  PETSCHEADER(int);
};

#endif
