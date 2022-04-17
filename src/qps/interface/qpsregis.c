
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  QPSRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(QPSRegister(QPSKSP,      QPSCreate_KSP));
  CHKERRQ(QPSRegister(QPSMPGP,     QPSCreate_MPGP));
  CHKERRQ(QPSRegister(QPSSMALXE,   QPSCreate_SMALXE));
  CHKERRQ(QPSRegister(QPSTAO,      QPSCreate_Tao));
  CHKERRQ(QPSRegister(QPSPCPG,     QPSCreate_PCPG));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSRegister"
PetscErrorCode QPSRegister(const char sname[],PetscErrorCode (*function)(QPS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListAdd(&QPSList,sname,function));
  PetscFunctionReturn(0);
}
