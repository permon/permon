#include <permon/private/qpcimpl.h>

PERMON_EXTERN PetscErrorCode QPCCreate_Box(QPC);

/*
   Contains the list of registered Create routines of all QPC types
*/
PetscFunctionList QPCList              = 0;
PetscBool         QPCRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPCRegisterAll"
PetscErrorCode QPCRegisterAll(void)
{
  PetscFunctionBegin;
  QPCRegisterAllCalled = PETSC_TRUE;
  PetscCall(QPCRegister(QPCBOX, QPCCreate_Box));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPCRegister"
PetscErrorCode QPCRegister(const char sname[], PetscErrorCode (*function)(QPC))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&QPCList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
