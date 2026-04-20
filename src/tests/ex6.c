static char help[] = "Test QPFetiConvertNumberingIS\n\n";

#include <permonqps.h>
#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **args)
{
  Mat         A;
  DM          da;
  PetscInt    idx[2] = {2, 3};
  PetscInt   *idxl;
  PetscInt    nodes[2] = {3, 3};
  PetscInt    ng, nl, dof = 1;
  PetscMPIInt size, rank;

  PetscCall(PermonInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Test only works with 4 processes");

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, nodes[0], nodes[1], PETSC_DECIDE, PETSC_DECIDE, dof, 1, NULL, NULL, &da));
  PetscCall(DMSetMatType(da, MATIS));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMCreateMatrix(da, &A));

  ng = 2;
  PetscCall(MatISIndicesGlobalToLocal(A, ng, idx, &nl, &idxl));
  PetscCall(PetscIntView(nl, idxl, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscFree(idxl));

  idx[0] = 7;
  idx[1] = 7;
  if (rank) ng = 0;
  PetscCall(MatISIndicesGlobalToLocal(A, ng, idx, &nl, &idxl));
  PetscCall(PetscIntView(nl, idxl, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFree(idxl));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  test:
    nsize: 4
TEST*/
