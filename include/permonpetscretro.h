#if !defined(__FLLOPPETSCRETRO_H)
#define	__FLLOPPETSCRETRO_H
#include <petscsys.h>

#if PETSC_VERSION_MAJOR!=3 || (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR!=8)
#error "PERMON requires PETSc version 3.8"
#endif

#endif

