#if !defined(__FLLOPPCIMPL_H)
#define	__FLLOPPCIMPL_H
#include <flloppc.h>
#if PETSC_VERSION_MINOR<6
#include <petsc-private/pcimpl.h>
#else
#include <petsc/private/pcimpl.h>
#endif
#include <private/fllopimpl.h>

FLLOP_EXTERN PetscLogEvent PC_Dual_Apply, PC_Dual_MatMultSchur;

#endif
