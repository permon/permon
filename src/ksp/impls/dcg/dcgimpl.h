
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
#include <permonksp.h>
#include <permon/private/permonkspimpl.h>

PETSC_INTERN PetscErrorCode KSPDestroy_DCG(KSP);
PETSC_INTERN PetscErrorCode KSPReset_DCG(KSP);
PETSC_INTERN PetscErrorCode KSPView_DCG(KSP,PetscViewer);
PETSC_INTERN PetscErrorCode KSPSetFromOptions_DCG(PetscOptionItems *PetscOptionsObject,KSP);
PETSC_INTERN PetscErrorCode KSPCGSetType_DCG(KSP,KSPCGType);

PETSC_INTERN PetscErrorCode KSPDCGComputeDeflationSpace(KSP ksp);

typedef struct {
  KSPCGType   type;                 /* type of system (symmetric or Hermitian) */
  PetscScalar emin,emax;           /* eigenvalues */
  PetscInt    ned;                 /* size of following arrays */
  PetscScalar *e,*d;
  PetscReal   *ee,*dd;             /* work space for Lanczos algorithm */

  PetscBool singlereduction;          /* use variant of CG that combines both inner products */
  PetscBool initcg;          /* do only init step - error correction of direction is omitted */
  PetscBool correct;         /* add Qr correction to descent direction */
  PetscBool truenorm;          
  PetscInt redundancy;
  Mat W,Wt,AW,WtAW; /* deflation space, coarse problem mats*/
  KSP WtAWinv; /* deflation coarse problem */
  Vec *work;
  KSPDCGSpaceType spacetype;
  PetscInt spacesize;
  PetscInt nestedlvl,maxnestedlvl;
  PetscBool extendsp;
  } KSP_DCG;

#endif
