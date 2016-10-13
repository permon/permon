#include <petscsys.h>
#include <fllopsys.h>

#if PETSC_VERSION_MINOR<=5
#include <petsc-private/matimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatGetFactorAvailable_permonfix"
PetscErrorCode  MatGetFactorAvailable_permonfix(Mat mat, const MatSolverPackage type,MatFactorType ftype,PetscBool *available)
{
  PetscErrorCode ierr, (*conv)(Mat,MatFactorType,PetscBool*);
  char           convname[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);

  if (mat->factortype) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  MatCheckPreallocated(mat,1);

  TRY( PetscObjectTypeCompareAny((PetscObject)mat, &flg, MATSBAIJ, MATSEQSBAIJ, MATMPISBAIJ, "") );
  if (flg && ftype != MAT_FACTOR_CHOLESKY) {
    *available = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  TRY( PetscStrcmp(type,MATSOLVERPASTIX,&flg) );
  if (flg) {
    TRY( PetscObjectTypeCompareAny((PetscObject)mat, &flg, MATAIJ, MATSEQAIJ, MATMPIAIJ, "") );
    if (flg && ftype != MAT_FACTOR_LU) {
      *available = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }

  TRY( PetscStrcmp(type,MATSOLVERSUPERLU,&flg) );
  if (!flg) TRY( PetscStrcmp(type,MATSOLVERSUPERLU_DIST,&flg) );
  if (flg) {
    if (ftype != MAT_FACTOR_LU) {
      *available = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  
  ierr = PetscStrcpy(convname,"MatGetFactor_");CHKERRQ(ierr);
  ierr = PetscStrcat(convname,type);CHKERRQ(ierr);
  ierr = PetscStrcat(convname,"_C");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)mat,convname,&conv);CHKERRQ(ierr);
  *available = conv ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#endif
