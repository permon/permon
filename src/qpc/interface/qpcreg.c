
#include <permon/private/qpcimpl.h>

FLLOP_EXTERN PetscErrorCode QPCCreate_Box(QPC);

/*
   Contains the list of registered Create routines of all QPC types
*/
PetscFunctionList QPCList = 0;
PetscBool  QPCRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "QPCRegisterAll"
PetscErrorCode  QPCRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  QPCRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(QPCRegister(QPCBOX,      QPCCreate_Box));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPCRegister"
PetscErrorCode QPCRegister(const char sname[],PetscErrorCode (*function)(QPC))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListAdd(&QPCList,sname,function));
  PetscFunctionReturn(0);
}
