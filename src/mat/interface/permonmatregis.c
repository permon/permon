
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
  TRY( MatRegister(MATINV,           MatCreate_Inv) );
  TRY( MatRegister(MATBLOCKDIAG,     MatCreate_BlockDiag) );
  TRY( MatRegister(MATSUM,           MatCreate_Sum) );
  TRY( MatRegister(MATPROD,          MatCreate_Prod) );
  TRY( MatRegister(MATSEQDENSEPERMON,MatCreate_SeqDensePermon) );
  TRY( MatRegister(MATMPIDENSEPERMON,MatCreate_MPIDensePermon) );
  TRY( MatRegister(MATEXTENSION,     MatCreate_Extension) );
  TRY( MatRegister(MATGLUING,        MatCreate_Gluing) );
  TRY( MatRegisterRootName(MATDENSEPERMON,MATSEQDENSEPERMON,MATMPIDENSEPERMON) );
  PetscFunctionReturn(0);
}
