
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
  ierr = QPCRegister(QPCBOX,      QPCCreate_Box);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPCRegister"
PetscErrorCode QPCRegister(const char sname[],PetscErrorCode (*function)(QPC))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&QPCList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
