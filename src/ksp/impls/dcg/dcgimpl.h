
/*
    Private Krylov Context Structure (KSP) for Deflated Conjugate Gradient

    This one is very simple. It contains a flag indicating the symmetry
   structure of the matrix and work space for (optionally) computing
   eigenvalues.

*/

#if !defined(__DCGIMPL_H)
#define __DCGIMPL_H

/*
        Defines the basic KSP object
*/
#include <permon/private/permonkspimpl.h>

PETSC_INTERN PetscErrorCode KSPDestroy_DCG(KSP);
PETSC_INTERN PetscErrorCode KSPReset_DCG(KSP);
PETSC_INTERN PetscErrorCode KSPView_DCG(KSP,PetscViewer);
PETSC_INTERN PetscErrorCode KSPSetFromOptions_DCG(PetscOptionItems *PetscOptionsObject,KSP);
PETSC_INTERN PetscErrorCode KSPCGSetType_DCG(KSP,KSPCGType);



typedef struct {
  KSPCGType   type;                 /* type of system (symmetric or Hermitian) */
  PetscScalar emin,emax;           /* eigenvalues */
  PetscInt    ned;                 /* size of following arrays */
  PetscScalar *e,*d;
  PetscReal   *ee,*dd;             /* work space for Lanczos algorithm */

  PetscBool singlereduction;          /* use variant of CG that combines both inner products */
  PetscBool initcg;          /* do only init step - error correction of direction is omitted */
  PetscBool truenorm;          
  PetscInt redundancy;
  Mat W,WtAW; /* deflation space, coarse problem mats*/
  KSP WtAWinv; /* deflation coarse problem */
  Vec *work;
} KSP_DCG;

#endif
