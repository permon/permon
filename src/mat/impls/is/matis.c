#include <permonmat.h>
#include <petsc/private/matisimpl.h>

/* This is based on MatZeroRowsColumns_Private_IS */
#undef __FUNCT__
#define __FUNCT__ "MatISIndicesGlobalToLocal"
PetscErrorCode MatISIndicesGlobalToLocal(Mat A, PetscInt nglobal, const PetscInt global[], PetscInt *nlocal, PetscInt *local[])
{
  Mat_IS   *matis;
  PetscInt  nr, nl, len;
  PetscInt *localTemp;
  PetscBool ismatis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  if (nglobal) PetscAssertPointer(global, 3);
  PetscAssertPointer(nlocal, 4);
  PetscAssertPointer(local, 5);
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  PetscAssert(ismatis, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "A must be MATIS");
  matis = (Mat_IS *)A->data;

  PetscCall(MatGetSize(matis->A, &nl, NULL));
  /* get locally owned rows */
  PetscCall(PetscLayoutMapLocal(A->rmap, nglobal, global, &len, &localTemp, NULL));
  /* get rows associated to the local matrices */
  PetscCall(PetscArrayzero(matis->sf_leafdata, nl));
  PetscCall(PetscArrayzero(matis->sf_rootdata, A->rmap->n));
  for (PetscInt i = 0; i < len; i++) matis->sf_rootdata[localTemp[i]] = 1;
  PetscCall(PetscFree(localTemp));
  PetscCall(PetscSFBcastBegin(matis->sf, MPIU_INT, matis->sf_rootdata, matis->sf_leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(matis->sf, MPIU_INT, matis->sf_rootdata, matis->sf_leafdata, MPI_REPLACE));
  /* count the indices */
  nr = 0;
  for (PetscInt i = 0; i < nl; i++)
    if (matis->sf_leafdata[i]) nr++;
  PetscCall(PetscMalloc1(nr, local));
  for (PetscInt i = 0; i < nr; i++)
    if (matis->sf_leafdata[i]) (*local)[i] = i;
  *nlocal = nr;
  PetscFunctionReturn(PETSC_SUCCESS);
}
