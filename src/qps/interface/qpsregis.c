
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
  ierr = QPSRegister(QPSKSP,      QPSCreate_KSP);CHKERRQ(ierr);
  ierr = QPSRegister(QPSMPGP,     QPSCreate_MPGP);CHKERRQ(ierr);
  ierr = QPSRegister(QPSSMALXE,   QPSCreate_SMALXE);CHKERRQ(ierr);
  ierr = QPSRegister(QPSTAO,      QPSCreate_Tao);CHKERRQ(ierr);
  ierr = QPSRegister(QPSPCPG,     QPSCreate_PCPG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QPSRegister"
PetscErrorCode QPSRegister(const char sname[],PetscErrorCode (*function)(QPS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&QPSList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
