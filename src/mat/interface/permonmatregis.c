
#include <permonmat.h>

FLLOP_EXTERN PetscErrorCode MatCreate_Inv(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_BlockDiag(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Sum(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Prod(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_SeqDensePermon(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_MPIDensePermon(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Extension(Mat);
FLLOP_EXTERN PetscErrorCode MatCreate_Gluing(Mat);
  
PetscErrorCode  PermonMatRegisterAll()
{
  PetscFunctionBegin;
  PetscCall(MatRegister(MATINV,           MatCreate_Inv));
  PetscCall(MatRegister(MATBLOCKDIAG,     MatCreate_BlockDiag));
  PetscCall(MatRegister(MATSUM,           MatCreate_Sum));
  PetscCall(MatRegister(MATPROD,          MatCreate_Prod));
  PetscCall(MatRegister(MATSEQDENSEPERMON,MatCreate_SeqDensePermon));
  PetscCall(MatRegister(MATMPIDENSEPERMON,MatCreate_MPIDensePermon));
  PetscCall(MatRegister(MATEXTENSION,     MatCreate_Extension));
  PetscCall(MatRegister(MATGLUING,        MatCreate_Gluing));
  PetscCall(MatRegisterRootName(MATDENSEPERMON,MATSEQDENSEPERMON,MATMPIDENSEPERMON));
  PetscFunctionReturn(0);
}
