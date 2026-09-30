#include <../src/qp/impls/feti/qpfetiimpl.h>

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

PetscErrorCode QPFetiDirichletDestroy(QPFetiDirichlet *dbc)
{
  PetscFunctionBegin;
  if (!*dbc) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISDestroy(&(*dbc)->is));
  PetscCall(PetscFree(*dbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
