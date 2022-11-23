
#include <permon/private/qpcimpl.h>

FLLOP_EXTERN PetscErrorCode QPCCreate_Box(QPC);

/*
   Contains the list of registered Create routines of all QPC types
*/
PetscFunctionList QPCList = 0;
PetscBool  QPCRegisterAllCalled = PETSC_FALSE;

PetscErrorCode  QPCRegisterAll(void)
{
  PetscFunctionBegin;
  QPCRegisterAllCalled = PETSC_TRUE;
  PetscCall(QPCRegister(QPCBOX,      QPCCreate_Box));
  PetscFunctionReturn(0);
}

PetscErrorCode QPCRegister(const char sname[],PetscErrorCode (*function)(QPC))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&QPCList,sname,function));
  PetscFunctionReturn(0);
}
