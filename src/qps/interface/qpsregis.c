#include <permon/private/qpsimpl.h>

FLLOP_EXTERN PetscErrorCode QPSCreate_KSP(QPS);
FLLOP_EXTERN PetscErrorCode QPSCreate_MPGP(QPS);
FLLOP_EXTERN PetscErrorCode QPSCreate_SMALXE(QPS);
FLLOP_EXTERN PetscErrorCode QPSCreate_Tao(QPS);
FLLOP_EXTERN PetscErrorCode QPSCreate_PCPG(QPS);

/*
   Contains the list of registered Create routines of all QPS types
*/
PetscFunctionList QPSList = 0;
PetscBool  QPSRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QPSRegisterAll"
PetscErrorCode  QPSRegisterAll(void)
{
  PetscFunctionBegin;
  QPSRegisterAllCalled = PETSC_TRUE;
  PetscCall(QPSRegister(QPSKSP,      QPSCreate_KSP));
  PetscCall(QPSRegister(QPSMPGP,     QPSCreate_MPGP));
  PetscCall(QPSRegister(QPSSMALXE,   QPSCreate_SMALXE));
  PetscCall(QPSRegister(QPSTAO,      QPSCreate_Tao));
  PetscCall(QPSRegister(QPSPCPG,     QPSCreate_PCPG));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSRegister"
PetscErrorCode QPSRegister(const char sname[],PetscErrorCode (*function)(QPS))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&QPSList,sname,function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
