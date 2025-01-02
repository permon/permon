#include <../src/qp/impls/feti/qpfetiimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPFetiDirichletCreate"
PetscErrorCode QPFetiDirichletCreate(IS dbcis, QPFetiNumberingType numtype, PetscBool enforce_by_B, QPFetiDirichlet *dbc_new)
{
  QPFetiDirichlet dbc;

  PetscFunctionBegin;
  PetscCall(PetscNew(&dbc));
  dbc->is = dbcis;
  PetscCall(PetscObjectReference((PetscObject)dbcis));
  dbc->numtype      = numtype;
  dbc->enforce_by_B = enforce_by_B;
  *dbc_new          = dbc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiDirichletDestroy"
PetscErrorCode QPFetiDirichletDestroy(QPFetiDirichlet *dbc)
{
  PetscFunctionBegin;
  if (!*dbc) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISDestroy(&(*dbc)->is));
  PetscCall(PetscFree(*dbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
