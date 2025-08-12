/* Test QPC active/free sets */
#include <permonqpc.h>

int main(int argc, char **args)
{
  Vec       lb, ub, x, g, gf, gc;
  IS        qpcis, globalis, is;
  QPC       qpc;
  PetscInt  i, ilo, ihi, localsize, n = 10, nl = 4;
  PetscInt  constraints[] = {2, 3, 4, n-1};
  PetscInt  qpcis_a[nl];
  PetscBool matching = PETSC_FALSE;

  PetscCall(PermonInitialize(&argc, &args, (char *)0, (char *)0));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecDuplicate(x, &gf));
  PetscCall(VecDuplicate(x, &gc));

  PetscCall(VecSet(x, .5));
  PetscCall(VecSetValue(x, constraints[0], 0., INSERT_VALUES));
  PetscCall(VecSetValue(x, constraints[1], 0., INSERT_VALUES));
  PetscCall(VecSetValue(x, constraints[nl-1], 1., INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecGetOwnershipRange(x, &ilo, &ihi));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, ihi-ilo, ilo, 1, &globalis));


  PetscCall(VecCreate(PETSC_COMM_WORLD, &lb));
  if (matching) {
    // create constraints with layout matching global vecs
    PetscCall(VecGetOwnershipRange(x, &ilo, &ihi));
    localsize = 0;
    for (i = 0; i < nl; i++) {
      if (ilo <= constraints[i]  && constraints[i] < ihi) {
        qpcis_a[localsize] = constraints[i];
        localsize++;
      }
    }
    // TODO add blocksize case
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, localsize, qpcis_a, PETSC_COPY_VALUES, &qpcis));
    PetscCall(VecSetSizes(lb, localsize, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(lb));
  } else {
    // distributed layout of constraints
    PetscCall(VecSetSizes(lb, PETSC_DECIDE, nl));
    PetscCall(VecSetFromOptions(lb));
    PetscCall(VecGetOwnershipRange(lb, &ilo, &ihi));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, ihi-ilo, &constraints[ilo], PETSC_COPY_VALUES, &qpcis));
  }
  PetscCall(VecDuplicate(lb, &ub));
  PetscCall(VecSet(lb, 0.));
  PetscCall(VecSet(ub, 1.));

  PetscCall(QPCCreateBox(PETSC_COMM_WORLD, qpcis, lb, ub, &qpc));
  PetscCall(QPCView(qpc, PETSC_VIEWER_STDOUT_WORLD));

  // TODO this is null !
  //PetscCall(QPCGetActiveSet(qpc, &is));
  //PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(QPCGrads(qpc, x, g, gf, gc));
  PetscCall(QPCGetActiveSet(qpc, &is));
  PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(QPCGetFreeSet(qpc, &is));
  PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&gf));
  PetscCall(VecDestroy(&gc));
  PetscCall(VecDestroy(&lb));
  PetscCall(ISDestroy(&globalis));
  PetscCall(ISDestroy(&qpcis));
  PetscCall(ISDestroy(&is));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  test:
TEST*/
