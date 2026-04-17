static char help[] = "Test QPFetiConvertNumberingIS\n\n";

#include <permonqps.h>
#include <permonqpfeti.h>
#include <petscdm.h>
#include <petscdmda.h>

static PetscScalar poiss_2D_emat[] = {6.6666666666666674e-01,  -1.6666666666666666e-01, -1.6666666666666666e-01, -3.3333333333333337e-01, -1.6666666666666666e-01, 6.6666666666666674e-01,  -3.3333333333333337e-01, -1.6666666666666666e-01,
                                      -1.6666666666666666e-01, -3.3333333333333337e-01, 6.6666666666666674e-01,  -1.6666666666666666e-01, -3.3333333333333337e-01, -1.6666666666666666e-01, -1.6666666666666666e-01, 6.6666666666666674e-01};

int main(int argc, char **args)
{
  QP              qp, qpd;
  Mat             A;
  DM              da;
  Vec             x, b;
  IS              islocal, isglobal, isglobald, is;
  const PetscInt  idx[4] = {0, 1, 2, 3};
  const PetscInt *e_loc;
  PetscInt        nodes[2] = {3, 3};
  PetscInt        i, nel, nen, dof = 1;
  PetscBool       equal;
  PetscMPIInt     size;

  PetscCall(PermonInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 4, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Test only works with four processes");

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, nodes[0], nodes[1], PETSC_DECIDE, PETSC_DECIDE, dof, 1, NULL, NULL, &da));
  PetscCall(DMSetMatType(da, MATIS));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(DMCreateGlobalVector(da, &b));
  PetscCall(DMCreateGlobalVector(da, &x));
  /* have to provide matrix with valid entries for QPTMatISToBlockDiag to work */
  PetscCall(DMDAGetElements(da, &nel, &nen, &e_loc));
  for (i = 0; i < nel; ++i) {
    PetscInt ord[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    PetscInt j, idxs[8];
    for (j = 0; j < nen; j++) idxs[j] = e_loc[i * nen + ord[j]];
    PetscCall(MatSetValuesBlockedLocal(A, nen, idxs, nen, idxs, poiss_2D_emat, ADD_VALUES));
  }
  PetscCall(DMDARestoreElements(da, &nel, &nen, &e_loc));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Create QP */
  PetscCall(QPCreate(PETSC_COMM_WORLD, &qp));
  PetscCall(QPSetOperator(qp, A));
  PetscCall(QPSetRhs(qp, b));
  PetscCall(QPSetInitialVector(qp, x));
  /* Transform to disassembled numbering */
  PetscCall(QPTMatISToBlockDiag(qp)); /* this sets up the l2g and i2g numberings for QPFeti routines */
  PetscCall(QPGetChild(qp, &qpd));

  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 4, idx, PETSC_USE_POINTER, &islocal));
  PetscCall(QPFetiConvertNumberingIS(qpd, FETI_LOCAL, islocal, FETI_GLOBAL_UNDECOMPOSED, &isglobal));
  PetscCall(QPFetiConvertNumberingIS(qpd, FETI_LOCAL, islocal, FETI_GLOBAL_DECOMPOSED, &isglobald));
  PetscCall(ISView(isglobal, NULL));
  PetscCall(ISView(isglobald, NULL));

  /* check convert to local */
  PetscCall(QPFetiConvertNumberingIS(qpd, FETI_GLOBAL_UNDECOMPOSED, isglobal, FETI_LOCAL, &is));
  PetscCall(ISEqual(islocal, is, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Converted numbering is wrong");
  PetscCall(ISDestroy(&is));
  PetscCall(QPFetiConvertNumberingIS(qpd, FETI_GLOBAL_DECOMPOSED, isglobald, FETI_LOCAL, &is));
  PetscCall(ISEqual(islocal, is, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Converted numbering is wrong");
  PetscCall(ISDestroy(&is));

  /* check convert to decomposed */
  PetscCall(QPFetiConvertNumberingIS(qpd, FETI_GLOBAL_UNDECOMPOSED, isglobal, FETI_GLOBAL_DECOMPOSED, &is));
  PetscCall(ISEqual(isglobald, is, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Converted numbering is wrong");
  PetscCall(ISDestroy(&is));

  /* check convert to undecomposed */
  PetscCall(QPFetiConvertNumberingIS(qpd, FETI_GLOBAL_DECOMPOSED, isglobald, FETI_GLOBAL_UNDECOMPOSED, &is));
  PetscCall(ISEqual(isglobal, is, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Converted numbering is wrong");
  PetscCall(ISDestroy(&is));

  PetscCall(ISDestroy(&islocal));
  PetscCall(ISDestroy(&isglobal));
  PetscCall(ISDestroy(&isglobald));
  PetscCall(QPDestroy(&qp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  test:
    nsize: 4
TEST*/
