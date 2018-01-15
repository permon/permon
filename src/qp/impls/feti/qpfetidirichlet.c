
#include <../src/qp/impls/feti/qpfetiimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPFetiDirichletCreate"
PetscErrorCode QPFetiDirichletCreate(IS dbcis, QPFetiNumberingType numtype, PetscBool enforce_by_B, QPFetiDirichlet *dbc_new)
{
  QPFetiDirichlet dbc;

  PetscFunctionBegin;
  TRY( PetscNew(&dbc) );
  dbc->is = dbcis;
  TRY( PetscObjectReference((PetscObject)dbcis) );
  dbc->numtype = numtype;
  dbc->enforce_by_B = enforce_by_B;
  *dbc_new = dbc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPFetiDirichletDestroy"
PetscErrorCode  QPFetiDirichletDestroy(QPFetiDirichlet *dbc)
{
  PetscFunctionBegin;
  if (!*dbc) PetscFunctionReturn(0);
  TRY( ISDestroy(&(*dbc)->is) );
  TRY( PetscFree(*dbc) );
  PetscFunctionReturn(0);
}


