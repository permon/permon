#pragma once

#include <petscsys.h>

#if PETSC_VERSION_MAJOR != 3 || (PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 17)
  #error "PERMON requires PETSc version 3.17 or higher"
#endif

/* allow to use petsc master branch while still supporting maint */
#if PETSC_VERSION_RELEASE
#endif /* #if PETSC_VERSION_RELEASE */

/* allow old integer formatting strings until 3.18, see. https://gitlab.com/petsc/petsc/-/merge_requests/5085 */
#if (PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 18)
  #if !defined(PetscInt_FMT)
    #if defined(PETSC_USE_64BIT_INDICES)
      #if !defined(PetscInt64_FMT)
        #if defined(PETSC_HAVE_STDINT_H) && defined(PETSC_HAVE_INTTYPES_H) && defined(PETSC_HAVE_MPI_INT64_T)
          #include <inttypes.h>
          #define PetscInt64_FMT PRId64
        #elif (PETSC_SIZEOF_LONG_LONG == 8)
          #define PetscInt64_FMT "lld"
        #elif defined(PETSC_HAVE___INT64)
          #define PetscInt64_FMT "ld"
        #else
          #error "cannot determine PetscInt64 type"
        #endif
      #endif
      #define PetscInt_FMT PetscInt64_FMT
    #else
      #define PetscInt_FMT "d"
    #endif
  #endif
#endif
