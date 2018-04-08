#if !defined(__FLLOPPETSCRETRO_H)
#define	__FLLOPPETSCRETRO_H
#include <petscsys.h>

#if PETSC_VERSION_MAJOR!=3 || (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR!=9)
#error "PERMON requires PETSc version 3.9"
#endif

#endif

