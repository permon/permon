
#include <permonmat.h>

FLLOP_EXTERN PetscErrorCode MatCreate_Inv(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_BlockDiag(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Sum(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Prod(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_SeqDensePermon(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_MPIDensePermon(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Extension(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Gluing(Mat);
  
#undef __FUNCT__  
#define __FUNCT__ "PermonMatRegisterAll"
PetscErrorCode  PermonMatRegisterAll()
{
  PetscFunctionBegin;
  CHKERRQ(MatRegister(MATINV,           MatCreate_Inv));
  CHKERRQ(MatRegister(MATBLOCKDIAG,     MatCreate_BlockDiag));
  CHKERRQ(MatRegister(MATSUM,           MatCreate_Sum));
  CHKERRQ(MatRegister(MATPROD,          MatCreate_Prod));
  CHKERRQ(MatRegister(MATSEQDENSEPERMON,MatCreate_SeqDensePermon));
  CHKERRQ(MatRegister(MATMPIDENSEPERMON,MatCreate_MPIDensePermon));
  CHKERRQ(MatRegister(MATEXTENSION,     MatCreate_Extension));
  CHKERRQ(MatRegister(MATGLUING,        MatCreate_Gluing));
  CHKERRQ(MatRegisterRootName(MATDENSEPERMON,MATSEQDENSEPERMON,MATMPIDENSEPERMON));
  PetscFunctionReturn(0);
}
