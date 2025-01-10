#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>
#include <petscsf.h>

//#define TAG_firstElemGlobIdx 198533

#undef __FUNCT__
#define __FUNCT__ "PermonMatGetLocalMat_Gluing"
static PetscErrorCode PermonMatGetLocalMat_Gluing(Mat A, Mat *Aloc)
{
  Mat_Gluing *data = (Mat_Gluing *)A->data;
  PetscSF     SF;
  PetscInt    i, N_col, n_row, n_col, start_col;
  PetscInt   *leafdata, *rootdata;
  PetscLayout links;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, NULL, &N_col));
  PetscCall(MatGetLocalSize(A, &n_row, &n_col));
  PetscCall(MatGetOwnershipRangeColumn(A, &start_col, NULL));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscInt), &leafdata));
  PetscCall(PetscMalloc(n_col * sizeof(PetscInt), &rootdata));

  for (i = 0; i < n_col; i++) { rootdata[i] = start_col + i; }
  PetscCall(PetscSFBcastBegin(data->SF, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(data->SF, MPIU_INT, rootdata, leafdata, MPI_REPLACE));

  PetscCall(PetscLayoutCreate(PETSC_COMM_SELF, &links));
  PetscCall(PetscLayoutSetBlockSize(links, 1));
  PetscCall(PetscLayoutSetSize(links, N_col));
  PetscCall(PetscLayoutSetUp(links));

  PetscCall(PetscSFCreate(PETSC_COMM_SELF, &SF));
  PetscCall(PetscSFSetGraphLayout(SF, links, data->n_leaves, NULL, PETSC_COPY_VALUES, leafdata));
  PetscCall(PetscSFSetRankOrder(SF, PETSC_TRUE));

  PetscCall(MatCreateGluing(PETSC_COMM_SELF, n_row, data->n_nonzeroRow, N_col, data->leaves_row, data->leaves_sign, SF, Aloc));

  PetscCall(PetscFree(leafdata));
  PetscCall(PetscFree(rootdata));
  PetscCall(PetscLayoutDestroy(&links));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Gluing"
PetscErrorCode MatMult_Gluing(Mat mat, Vec right, Vec left)
{
  Mat_Gluing  *data = (Mat_Gluing *)mat->data;
  PetscScalar *x_onleaves, *lambda_root, *lambda_onleaves;
  PetscInt     i, start, *idxX;

  PetscFunctionBegin;
  //right=lambda left=x

  PetscCall(VecGetArray(right, &lambda_root));
  PetscCall(VecGetOwnershipRange(left, &start, NULL));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscScalar), &lambda_onleaves));

  PetscCall(PetscSFBcastBegin(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscInt), &idxX));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscScalar), &x_onleaves));
  PetscCall(VecZeroEntries(left));
  PetscCall(PetscSFBcastEnd(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  PetscCall(VecRestoreArray(right, &lambda_root));

  for (i = 0; i < data->n_leaves; i++) {
    idxX[i]       = start + data->leaves_row[i];
    x_onleaves[i] = lambda_onleaves[i] * data->leaves_sign[i];
  }

  PetscCall(VecZeroEntries(left));
  PetscCall(VecSetValues(left, data->n_leaves, idxX, x_onleaves, ADD_VALUES););
  PetscCall(VecAssemblyBegin(left));
  PetscCall(VecAssemblyEnd(left));

  PetscCall(PetscFree(idxX));
  PetscCall(PetscFree(x_onleaves));
  PetscCall(PetscFree(lambda_onleaves));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Gluing"
PetscErrorCode MatMultAdd_Gluing(Mat mat, Vec right, Vec add, Vec left)
{
  Mat_Gluing  *data = (Mat_Gluing *)mat->data;
  PetscScalar *x_onleaves, *lambda_root, *lambda_onleaves;
  PetscInt     i, start, *idxX;

  PetscFunctionBegin;
  //right=lambda left=x

  PetscCall(VecGetArray(right, &lambda_root));
  PetscCall(VecGetOwnershipRange(left, &start, NULL));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscScalar), &lambda_onleaves));

  PetscCall(PetscSFBcastBegin(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscInt), &idxX));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscScalar), &x_onleaves));
  PetscCall(VecZeroEntries(left));
  PetscCall(PetscSFBcastEnd(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  PetscCall(VecRestoreArray(right, &lambda_root));

  for (i = 0; i < data->n_leaves; i++) {
    idxX[i]       = start + data->leaves_row[i];
    x_onleaves[i] = lambda_onleaves[i] * data->leaves_sign[i];
  }

  PetscCall(VecZeroEntries(left));
  PetscCall(VecSetValues(left, data->n_leaves, idxX, x_onleaves, ADD_VALUES););
  PetscCall(VecAssemblyBegin(left));
  PetscCall(VecAssemblyEnd(left));

  PetscCall(PetscFree(idxX));
  PetscCall(PetscFree(x_onleaves));
  PetscCall(PetscFree(lambda_onleaves));

  PetscCall(VecAXPY(left, 1, add));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Gluing"
PetscErrorCode MatMultTranspose_Gluing(Mat mat, Vec right, Vec left)
{
  Mat_Gluing  *data = (Mat_Gluing *)mat->data;
  PetscScalar *x, *lambda_onroot, *lambda_onleaves;
  PetscInt     i, start, n_col, *idxL;

  PetscFunctionBegin;
  //right=x left=lambda
  PetscCall(VecGetArray(right, &x));
  PetscCall(VecGetOwnershipRange(left, &start, NULL));
  PetscCall(MatGetLocalSize(mat, NULL, &n_col));

  PetscCall(PetscMalloc(n_col * sizeof(PetscScalar), &lambda_onroot));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscScalar), &lambda_onleaves));

  for (i = 0; i < n_col; i++) { lambda_onroot[i] = 0; }

  for (i = 0; i < data->n_leaves; i++) { lambda_onleaves[i] = x[data->leaves_row[i]] * data->leaves_sign[i]; }

  PetscCall(PetscSFReduceBegin(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));
  PetscCall(PetscMalloc(n_col * sizeof(PetscInt), &idxL));
  for (i = 0; i < n_col; i++) { idxL[i] = start + i; }
  PetscCall(PetscSFReduceEnd(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));

  PetscCall(VecZeroEntries(left));
  PetscCall(VecSetValues(left, n_col, idxL, lambda_onroot, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(left));
  PetscCall(VecAssemblyEnd(left));

  PetscCall(VecRestoreArray(right, &x));
  PetscCall(PetscFree(lambda_onroot));
  PetscCall(PetscFree(lambda_onleaves));
  PetscCall(PetscFree(idxL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Gluing"
PetscErrorCode MatMultTransposeAdd_Gluing(Mat mat, Vec right, Vec add, Vec left)
{
  Mat_Gluing  *data = (Mat_Gluing *)mat->data;
  PetscScalar *x, *lambda_onroot, *lambda_onleaves;
  PetscInt     i, start, n_col, *idxL;

  PetscFunctionBegin;
  //right=x left=lambda
  PetscCall(VecGetArray(right, &x));
  PetscCall(VecGetOwnershipRange(left, &start, NULL));
  PetscCall(MatGetLocalSize(mat, NULL, &n_col));

  PetscCall(PetscMalloc(n_col * sizeof(PetscScalar), &lambda_onroot));
  PetscCall(PetscMalloc(data->n_leaves * sizeof(PetscScalar), &lambda_onleaves));

  for (i = 0; i < n_col; i++) { lambda_onroot[i] = 0; }

  for (i = 0; i < data->n_leaves; i++) { lambda_onleaves[i] = x[data->leaves_row[i]] * data->leaves_sign[i]; }

  PetscCall(PetscSFReduceBegin(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));
  PetscCall(PetscMalloc(n_col * sizeof(PetscInt), &idxL));
  for (i = 0; i < n_col; i++) { idxL[i] = start + i; }
  PetscCall(PetscSFReduceEnd(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));
  PetscCall(VecZeroEntries(left));
  PetscCall(VecSetValues(left, n_col, idxL, lambda_onroot, ADD_VALUES));
  PetscCall(VecAssemblyBegin(left));
  PetscCall(VecAssemblyEnd(left));

  PetscCall(VecRestoreArray(right, &x));
  PetscCall(PetscFree(lambda_onroot));
  PetscCall(PetscFree(lambda_onleaves));
  PetscCall(PetscFree(idxL));

  PetscCall(VecAXPY(left, 1, add));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Gluing"
PetscErrorCode MatDestroy_Gluing(Mat mat)
{
  PetscFunctionBegin;
  Mat_Gluing *data = (Mat_Gluing *)mat->data;
  PetscCall(PetscSFDestroy(&data->SF));
  PetscCall(PetscFree(data->leaves_row));
  PetscCall(PetscFree(data->leaves_sign));
  PetscCall(PetscFree(data));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "PermonMatGetLocalMat_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateGluing"
PetscErrorCode MatCreateGluing(MPI_Comm comm, PetscInt n_x_localRow, PetscInt n_nonzeroRow, PetscInt n_l_localcol, const PetscInt *leaves_row, const PetscReal *leaves_sign, PetscSF SF, Mat *B_out)
{
  Mat_Gluing *data;
  PetscInt    rlo, rhi, clo, chi, n_l;
  PetscInt   *lr;
  PetscReal  *ls;
  Mat         B;

  PetscFunctionBegin;
  PetscAssertPointer(B_out, 7);

  /* Create matrix. */
  PetscCall(MatCreate(comm, &B));
  PetscCall(MatSetType(B, MATGLUING));
  data = (Mat_Gluing *)B->data;

  PetscCall(PetscSFGetLeafRange(SF, NULL, &n_l));

  PetscCall(PetscMalloc1(n_l + 1, &lr));
  PetscCall(PetscMemcpy(lr, leaves_row, (n_l + 1) * sizeof(PetscInt)));
  PetscCall(PetscMalloc1(n_l + 1, &ls));
  PetscCall(PetscMemcpy(ls, leaves_sign, (n_l + 1) * sizeof(PetscReal)));
  PetscCall(PetscObjectReference((PetscObject)SF));

  data->n_leaves     = n_l + 1;
  data->n_nonzeroRow = n_nonzeroRow;
  data->SF           = SF;
  data->leaves_row   = lr;
  data->leaves_sign  = ls;

  /* Set up row layout */
  PetscCall(PetscLayoutSetLocalSize(B->rmap, n_x_localRow));
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutGetRange(B->rmap, &rlo, &rhi));

  /* Set up column layout */
  PetscCall(PetscLayoutSetLocalSize(B->cmap, n_l_localcol));
  PetscCall(PetscLayoutSetUp(B->cmap));
  PetscCall(PetscLayoutGetRange(B->cmap, &clo, &chi));

  *B_out = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Gluing"
PERMON_EXTERN PetscErrorCode MatCreate_Gluing(Mat B)
{
  Mat_Gluing *data;

  PetscFunctionBegin;
  PetscCall(PetscNew(&data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATGLUING));
  B->data         = (void *)data;
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  data->SF           = NULL;
  data->leaves_row   = NULL;
  data->leaves_sign  = NULL;
  data->n_leaves     = 0;
  data->n_nonzeroRow = 0;

  /* Set operations of matrix. */
  B->ops->destroy          = MatDestroy_Gluing;
  B->ops->mult             = MatMult_Gluing;
  B->ops->multtranspose    = MatMultTranspose_Gluing;
  B->ops->multadd          = MatMultAdd_Gluing;
  B->ops->multtransposeadd = MatMultTransposeAdd_Gluing;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "PermonMatGetLocalMat_C", PermonMatGetLocalMat_Gluing));
  PetscFunctionReturn(PETSC_SUCCESS);
}
