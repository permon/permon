#if !defined(__FLLOPPETSCRETRO_H)
#define	__FLLOPPETSCRETRO_H
#include <petscsys.h>

#if PETSC_VERSION_MAJOR!=3 || (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR!=9)
#error "PERMON requires PETSc version 3.9"
#endif

/* allow to use petsc master branch while still supporting maint */
#if PETSC_VERSION_RELEASE

/* TaoType changed from char* to const char* in master */
#if defined(TaoType)
#undef TaoType
#endif
#define TaoType const char*

#endif /* #if PETSC_VERSION_RELEASE */

#endif

