#if !defined(__FLLOPKSPIMPL_H)
#define	__FLLOPKSPIMPL_H
#include <fllopksp.h>
#if PETSC_VERSION_MINOR<6
#include <petsc-private/kspimpl.h>
#else
#include <petsc/private/kspimpl.h>
#endif
#include <private/fllopimpl.h>

#endif
