#if !defined(__FLLOPPC_H)
#define	__FLLOPPC_H
#include <petscpc.h>
#include "fllopmat.h" 

/* subclasses */
#define PCDUAL "dual"

FLLOP_EXTERN PetscErrorCode FllopPCRegisterAll();
FLLOP_EXTERN PetscBool FllopPCRegisterAllCalled;

/* PCDUAL type-specific functions */
typedef enum {PC_DUAL_NONE=0, PC_DUAL_LUMPED=1} PCDualType;
FLLOP_EXTERN const char *PCDualTypes[];
FLLOP_EXTERN PetscErrorCode PCDualSetType(PC pc,PCDualType type);
FLLOP_EXTERN PetscErrorCode PCDualGetType(PC pc,PCDualType *type);

#endif

