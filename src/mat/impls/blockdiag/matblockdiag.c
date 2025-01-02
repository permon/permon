#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#define TAG_firstElemGlobIdx 198533

static PetscErrorCode MatGetDiagonalBlock_BlockDiag(Mat, Mat *);

#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsColumns_BlockDiag"
PetscErrorCode MatZeroRowsColumns_BlockDiag(Mat A, PetscInt n, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)A->data;
  Mat            Aloc = data->localBlock;
  Vec            xloc = data->xloc, bloc = data->yloc;
  PetscInt       nloc, *rows_loc;

  PetscFunctionBegin;
  PetscCall(ISGlobalToLocalMappingApply(A->rmap->mapping, IS_GTOLM_DROP, n, rows, &nloc, NULL));
  PERMON_ASSERT(n == nloc, "n==nloc");
  PetscCall(PetscMalloc(n * sizeof(PetscInt), &rows_loc));
  PetscCall(ISGlobalToLocalMappingApply(A->rmap->mapping, IS_GTOLM_DROP, n, rows, NULL, rows_loc));
  if (x) {
    PetscCall(VecGetLocalVectorRead(x, xloc));
  } else {
    xloc = NULL;
  }
  if (b) {
    PetscCall(VecGetLocalVector(b, bloc));
  } else {
    bloc = NULL;
  }
  PetscCall(MatZeroRowsColumns(Aloc, nloc, rows_loc, diag, xloc, bloc));
  if (x) PetscCall(VecRestoreLocalVectorRead(x, xloc));
  if (b) PetscCall(VecRestoreLocalVector(b, bloc));
  PetscCall(PetscFree(rows_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroRows_BlockDiag"
PetscErrorCode MatZeroRows_BlockDiag(Mat A, PetscInt n, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)A->data;
  Mat            Aloc = data->localBlock;
  Vec            xloc = data->xloc, bloc = data->yloc;
  PetscInt       nloc, *rows_loc;

  PetscFunctionBegin;
  PetscCall(ISGlobalToLocalMappingApply(A->rmap->mapping, IS_GTOLM_DROP, n, rows, &nloc, NULL));
  PERMON_ASSERT(n == nloc, "n==nloc");
  PetscCall(PetscMalloc(n * sizeof(PetscInt), &rows_loc));
  PetscCall(ISGlobalToLocalMappingApply(A->rmap->mapping, IS_GTOLM_DROP, n, rows, NULL, rows_loc));
  if (x) {
    PetscCall(VecGetLocalVectorRead(x, xloc));
  } else {
    xloc = NULL;
  }
  if (b) {
    PetscCall(VecGetLocalVector(b, bloc));
  } else {
    bloc = NULL;
  }
  PetscCall(MatZeroRows(Aloc, nloc, rows_loc, diag, xloc, bloc));
  if (x) PetscCall(VecRestoreLocalVectorRead(x, xloc));
  if (b) PetscCall(VecRestoreLocalVector(b, bloc));
  PetscCall(PetscFree(rows_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_BlockDiag"
PetscErrorCode MatZeroEntries_BlockDiag(Mat A)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(data->localBlock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_BlockDiag_SeqAIJ"
static PetscErrorCode MatConvert_BlockDiag_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newB)
{
  MPI_Comm    comm;
  PetscMPIInt size;
  Mat         A_loc, B;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_SUP, "conversion from MPI BlockDiag matrix to sequential matrix not currently implemented");

  PetscCall(MatGetDiagonalBlock(A, &A_loc));
  PetscCall(MatConvert(A_loc, newtype, MAT_INITIAL_MATRIX, &B));
  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    PetscCall(MatHeaderReplace(A, B));
#else
    PetscCall(MatHeaderReplace(A, &B));
#endif
  } else {
    *newB = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_BlockDiag_MPIAIJ"
static PetscErrorCode MatConvert_BlockDiag_MPIAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newB)
{
  PetscMPIInt size;
  Mat         A_loc, B;
  char       *loctype;
  Mat_MPIAIJ *mpidata;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCall(MatGetDiagonalBlock(A, &A_loc));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetType(B, newtype));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(MatMPIAIJSetPreallocation(B, 0, 0, 0, 0));

  mpidata = (Mat_MPIAIJ *)B->data;
  PetscCall(PetscStrallocpy(((PetscObject)mpidata->A)->type_name, &loctype));
  PetscCall(MatDestroy(&mpidata->A));
  PetscCall(MatConvert(A_loc, loctype, MAT_INITIAL_MATRIX, &mpidata->A));
  PetscCall(PetscFree(loctype));

  /* MatConvert should produce an already assembled matrix so we just set the 'assembled' flag to true */
  B->assembled = PETSC_TRUE;

  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    PetscCall(MatHeaderReplace(A, B));
#else
    PetscCall(MatHeaderReplace(A, &B));
#endif
  } else {
    *newB = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_BlockDiag_AIJ"
static PetscErrorCode MatConvert_BlockDiag_AIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newB)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    PetscCall(MatConvert_BlockDiag_SeqAIJ(A, newtype, reuse, newB));
  } else {
    PetscCall(MatConvert_BlockDiag_MPIAIJ(A, newtype, reuse, newB));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatConvertBlocks_BlockDiag"
static PetscErrorCode PermonMatConvertBlocks_BlockDiag(Mat A, MatType newtype, MatReuse reuse, Mat *B)
{
  Mat_BlockDiag *data   = (Mat_BlockDiag *)A->data;
  Mat            cblock = NULL;
  Mat            B_;

  PetscFunctionBegin;
  if (reuse == MAT_INPLACE_MATRIX) cblock = data->localBlock;
  PetscCall(MatConvert(data->localBlock, newtype, reuse, &cblock));
  PetscCall(MatCreateBlockDiag(PetscObjectComm((PetscObject)A), cblock, &B_));
  if (reuse != MAT_INPLACE_MATRIX) {
    PetscCall(MatDestroy(&cblock));
    *B = B_;
  } else {
#if PETSC_VERSION_MINOR < 7
    PetscCall(MatHeaderReplace(A, B_));
#else
    PetscCall(MatHeaderReplace(A, &B_));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_BlockDiag"
PetscErrorCode MatMult_BlockDiag(Mat mat, Vec right, Vec left)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorRead(right, data->xloc));
  PetscCall(VecGetLocalVector(left, data->yloc));
  PetscCall(MatMult(data->localBlock, data->xloc, data->yloc));
  PetscCall(VecRestoreLocalVectorRead(right, data->xloc));
  PetscCall(VecRestoreLocalVector(left, data->yloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_BlockDiag"
PetscErrorCode MatMultTranspose_BlockDiag(Mat mat, Vec right, Vec left)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorRead(right, data->yloc));
  PetscCall(VecGetLocalVector(left, data->xloc));
  PetscCall(MatMultTranspose(data->localBlock, data->yloc, data->xloc));
  PetscCall(VecRestoreLocalVectorRead(right, data->yloc));
  PetscCall(VecRestoreLocalVector(left, data->xloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_BlockDiag"
PetscErrorCode MatMultAdd_BlockDiag(Mat mat, Vec v1, Vec v2, Vec v3)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorRead(v1, data->xloc));
  PetscCall(VecGetLocalVector(v2, data->yloc1)); /* v2 can be same as v3 */
  PetscCall(VecGetLocalVector(v3, data->yloc));
  PetscCall(MatMultAdd(data->localBlock, data->xloc, data->yloc1, data->yloc));
  PetscCall(VecRestoreLocalVectorRead(v1, data->xloc));
  PetscCall(VecRestoreLocalVector(v2, data->yloc1));
  PetscCall(VecRestoreLocalVector(v3, data->yloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_BlockDiag"
PetscErrorCode MatMultTransposeAdd_BlockDiag(Mat mat, Vec v1, Vec v2, Vec v3)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorRead(v1, data->yloc));
  PetscCall(VecGetLocalVector(v2, data->xloc1)); /* v2 can be same as v3 */
  PetscCall(VecGetLocalVector(v3, data->xloc));
  PetscCall(MatMultTransposeAdd(data->localBlock, data->yloc, data->xloc1, data->xloc));
  PetscCall(VecRestoreLocalVectorRead(v1, data->yloc));
  PetscCall(VecRestoreLocalVector(v2, data->xloc1));
  PetscCall(VecRestoreLocalVector(v3, data->xloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_BlockDiag_BlockDiag"
PetscErrorCode MatMatMult_BlockDiag_BlockDiag(Mat A, Mat B, PetscReal fill, Mat *C)
{
  MPI_Comm comm;
  Mat      A_loc, B_loc, C_loc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  PetscCall(MatGetDiagonalBlock_BlockDiag(B, &B_loc));
  PetscCall(MatMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  PetscCall(MatCreateBlockDiag(comm, C_loc, C));
  PetscCall(MatDestroy(&C_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_BlockDiag_AIJ"
static PetscErrorCode MatMatMult_BlockDiag_AIJ(Mat A, Mat B, PetscReal fill, Mat *C)
{
  MPI_Comm    comm;
  Mat         A_loc, B_loc, C_loc;
  Vec         x;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  if (size > 1) {
    PetscCall(MatMPIAIJGetLocalMat(B, MAT_INITIAL_MATRIX, &B_loc));
  } else {
    B_loc = B;
    PetscCall(PetscObjectReference((PetscObject)B));
  }
  PetscCall(MatMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  PetscCall(MatDestroy(&B_loc));

  PetscCall(MatCreateVecs(B, &x, NULL));
  PetscCall(MatMergeAndDestroy(comm, &C_loc, x, C));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_BlockDiag"
PetscErrorCode MatTransposeMatMult_BlockDiag_BlockDiag(Mat A, Mat B, PetscReal fill, Mat *C)
{
  MPI_Comm comm;
  Mat      A_loc, B_loc, C_loc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  PetscCall(MatGetDiagonalBlock_BlockDiag(B, &B_loc));
  PetscCall(MatTransposeMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  PetscCall(MatCreateBlockDiag(comm, C_loc, C));
  PetscCall(MatDestroy(&C_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_AIJ"
static PetscErrorCode MatTransposeMatMult_BlockDiag_AIJ(Mat A, Mat B, PetscReal fill, Mat *C)
{
  MPI_Comm    comm;
  Mat         A_loc, B_loc, C_loc;
  Vec         x;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  if (size > 1) {
    PetscCall(MatMPIAIJGetLocalMat(B, MAT_INITIAL_MATRIX, &B_loc));
  } else {
    B_loc = B;
    PetscCall(PetscObjectReference((PetscObject)B));
  }
  PetscCall(MatTransposeMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  PetscCall(MatDestroy(&B_loc));

  PetscCall(MatCreateVecs(B, &x, NULL));
  PetscCall(MatMergeAndDestroy(comm, &C_loc, x, C));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_BlockDiag_AIJ"
static PetscErrorCode MatProductNumeric_BlockDiag_AIJ(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, B = product->B;
  Mat          newmat;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatMatMult_BlockDiag_AIJ(A, B, product->fill, &newmat));
    break;
  case MATPRODUCT_AtB:
    PetscCall(MatTransposeMatMult_BlockDiag_AIJ(A, B, product->fill, &newmat));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "MATPRODUCT type is not supported");
  }
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &newmat));
  C->product             = product;
  C->ops->productnumeric = MatProductNumeric_BlockDiag_AIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_BlockDiag_AIJ"
static PetscErrorCode MatProductSymbolic_BlockDiag_AIJ(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumeric_BlockDiag_AIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_BlockDiag_AIJ"
static PetscErrorCode MatProductSetFromOptions_BlockDiag_AIJ(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_BlockDiag_AIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_BlockDiag"
static PetscErrorCode MatProductNumeric_BlockDiag(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, B = product->B;
  Mat          newmat;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatMatMult_BlockDiag_BlockDiag(A, B, product->fill, &newmat));
    break;
  case MATPRODUCT_AtB:
    PetscCall(MatTransposeMatMult_BlockDiag_BlockDiag(A, B, product->fill, &newmat));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "MATPRODUCT type is not supported");
  }
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &newmat));
  C->product             = product;
  C->ops->productnumeric = MatProductNumeric_BlockDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_BlockDiag"
static PetscErrorCode MatProductSymbolic_BlockDiag(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric = MatProductNumeric_BlockDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_BlockDiag"
static PetscErrorCode MatProductSetFromOptions_BlockDiag(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_BlockDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_BlockDiag"
PetscErrorCode MatDestroy_BlockDiag(Mat mat)
{
  Mat_BlockDiag *data;

  PetscFunctionBegin;
  data = (Mat_BlockDiag *)mat->data;
  PetscCall(MatDestroy(&data->localBlock));
  PetscCall(VecDestroy(&data->xloc));
  PetscCall(VecDestroy(&data->yloc));
  PetscCall(VecDestroy(&data->xloc1));
  PetscCall(VecDestroy(&data->yloc1));
  PetscCall(PetscFree(data));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatGetColumnVectors_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatRestoreColumnVectors_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MaProductSetFromOptions_blockdiag_aij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MaProductSetFromOptions_blockdiag_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MaProductSetFromOptions_blockdiag_mpiaij", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_blockdiag_aij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatOrthColumns_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "PermonMatConvertBlocks_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_BlockDiag"
PetscErrorCode MatDuplicate_BlockDiag(Mat matin, MatDuplicateOption cpvalues, Mat *newmat)
{
  Mat            matout;
  Mat_BlockDiag *datain, *dataout;

  PetscFunctionBegin;
  datain = (Mat_BlockDiag *)matin->data;

  PetscCall(MatCreate(((PetscObject)matin)->comm, &matout));
  PetscCall(MatSetSizes(matout, matin->rmap->n, matin->cmap->n, matin->rmap->N, matin->cmap->N));
  PetscCall(MatSetType(matout, ((PetscObject)matin)->type_name));
  matout->ops[0] = matin->ops[0];
  dataout        = (Mat_BlockDiag *)matout->data;

  PetscCall(MatDuplicate(datain->localBlock, cpvalues, &dataout->localBlock));
  PetscCall(VecDuplicate(datain->yloc, &dataout->yloc));
  PetscCall(VecDuplicate(datain->yloc1, &dataout->yloc1));
  PetscCall(VecDuplicate(datain->xloc, &dataout->xloc));
  *newmat = matout;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_BlockDiag"
static PetscErrorCode MatGetDiagonalBlock_BlockDiag(Mat A, Mat *A_loc)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)A->data;

  PetscFunctionBegin;
  *A_loc = data->localBlock;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_BlockDiag"
PetscErrorCode MatGetInfo_BlockDiag(Mat matin, MatInfoType flag, MatInfo *info)
{
  Mat_BlockDiag *mat = (Mat_BlockDiag *)matin->data;
  Mat            A   = mat->localBlock;
  PetscReal      isend[5], irecv[5];

  PetscFunctionBegin;
  info->block_size = 1.0;
  PetscCall(MatGetInfo(A, MAT_LOCAL, info));
  isend[0] = info->nz_used;
  isend[1] = info->nz_allocated;
  isend[2] = info->nz_unneeded;
  isend[3] = info->memory;
  isend[4] = info->mallocs;

  switch (flag) {
  case MAT_LOCAL:
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
    break;
  case MAT_GLOBAL_MAX:
    PetscCallMPI(MPI_Allreduce(isend, irecv, 5, MPIU_REAL, MPIU_MAX, ((PetscObject)matin)->comm));
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
    break;
  case MAT_GLOBAL_SUM:
    PetscCallMPI(MPI_Allreduce(isend, irecv, 5, MPIU_REAL, MPIU_SUM, ((PetscObject)matin)->comm));
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_BlockDiag"
PetscErrorCode MatSetOption_BlockDiag(Mat mat, MatOption op, PetscBool flg)
{
  Mat_BlockDiag *bd = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatSetOption(bd->localBlock, op, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_BlockDiag"
PetscErrorCode MatGetDiagonal_BlockDiag(Mat mat, Vec d)
{
  Mat_BlockDiag *bd = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonal(bd->localBlock, bd->yloc));
  PetscCall(VecCopy(bd->yloc, d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_BlockDiag"
PetscErrorCode MatView_BlockDiag(Mat mat, PetscViewer viewer)
{
  Mat_BlockDiag    *bd = (Mat_BlockDiag *)mat->data;
  PetscViewer       sv;
  MPI_Comm          comm;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCheck(iascii, comm, PETSC_ERR_SUP, "Viewer type %s not supported for matrix type %s", ((PetscObject)viewer)->type_name, ((PetscObject)mat)->type_name);
  PetscCall(PetscViewerGetFormat(viewer, &format));

  if (format == PETSC_VIEWER_DEFAULT) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)mat, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "diagonal blocks:\n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "diagonal block on rank 0:\n"));
  }

  PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sv));
  if (format != PETSC_VIEWER_DEFAULT) {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (!rank) PetscCall(MatView(bd->localBlock, sv));
  } else {
    PetscCall(PetscSequentialPhaseBegin(comm, 1));
    PetscCall(MatView(bd->localBlock, sv));
    PetscCall(PetscSequentialPhaseEnd(comm, 1));
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sv));

  if (format == PETSC_VIEWER_DEFAULT) { PetscCall(PetscViewerASCIIPopTab(viewer)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_BlockDiag"
PetscErrorCode MatAssemblyBegin_BlockDiag(Mat mat, MatAssemblyType type)
{
  Mat_BlockDiag *bd = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyBegin(bd->localBlock, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_BlockDiag"
PetscErrorCode MatAssemblyEnd_BlockDiag(Mat mat, MatAssemblyType type)
{
  Mat_BlockDiag *bd = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd(bd->localBlock, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetLocalToGlobalMapping_BlockDiag"
PetscErrorCode MatSetLocalToGlobalMapping_BlockDiag(Mat x, ISLocalToGlobalMapping rmapping, ISLocalToGlobalMapping cmapping)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "custom LocalToGlobalMapping not allowed for matrix of type %s", ((PetscObject)x)->type_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesLocal_BlockDiag"
PetscErrorCode MatSetValuesLocal_BlockDiag(Mat mat, PetscInt nrow, const PetscInt irow[], PetscInt ncol, const PetscInt icol[], const PetscScalar y[], InsertMode addv)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatSetValues(data->localBlock, nrow, irow, ncol, icol, y, addv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_BlockDiag"
PetscErrorCode MatScale_BlockDiag(Mat mat, PetscScalar a)
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatScale(data->localBlock, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors_BlockDiag"
static PetscErrorCode MatGetColumnVectors_BlockDiag(Mat mat, Vec *cols_new[])
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;
  PetscInt       n, n1, N, j, jlo;
  Vec            d, *cols;
  PetscScalar   *arr;

  PetscFunctionBegin;
  n   = mat->cmap->n;
  N   = mat->cmap->N;
  jlo = mat->cmap->rstart;
  PetscCall(MatGetColumnVectors(data->localBlock, &n1, &data->cols_loc));
  PERMON_ASSERT(n == n1, "n==n1");

  PetscCall(MatCreateVecs(mat, PETSC_IGNORE, &d));
  PetscCall(VecDuplicateVecs(d, N, &cols));
  PetscCall(VecDestroy(&d));

  for (j = 0; j < N; j++) { PetscCall(VecSet(cols[j], 0.0)); }

  for (j = 0; j < n; j++) {
    PetscCall(VecGetArray(data->cols_loc[j], &arr));
    PetscCall(VecPlaceArray(cols[j + jlo], arr));
  }

  *cols_new = cols;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_BlockDiag"
static PetscErrorCode MatRestoreColumnVectors_BlockDiag(Mat mat, Vec *cols[])
{
  Mat_BlockDiag *data = (Mat_BlockDiag *)mat->data;
  PetscInt       n, N, j, jlo;

  PetscFunctionBegin;
  n   = mat->cmap->n;
  N   = mat->cmap->N;
  jlo = mat->cmap->rstart;

  for (j = 0; j < n; j++) {
    PetscCall(VecRestoreArray(data->cols_loc[j], NULL));
    PetscCall(VecResetArray((*cols)[j + jlo]));
  }

  PetscCall(MatRestoreColumnVectors(data->localBlock, NULL, &data->cols_loc));
  PetscCall(VecDestroyVecs(N, cols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_BlockDiag"
static PetscErrorCode MatOrthColumns_BlockDiag(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  Mat_BlockDiag *bd    = (Mat_BlockDiag *)A->data;
  Mat            Q_loc = NULL, S_loc = NULL;

  PetscFunctionBegin;
  PetscCall(MatOrthColumns(bd->localBlock, type, form, Q_new ? (&Q_loc) : NULL, S_new ? (&S_loc) : NULL));
  if (Q_new) PetscCall(MatCreateBlockDiag(PetscObjectComm((PetscObject)A), Q_loc, Q_new));
  if (S_new) PetscCall(MatCreateBlockDiag(PetscObjectComm((PetscObject)A), S_loc, S_new));
  PetscCall(MatDestroy(&Q_loc));
  PetscCall(MatDestroy(&S_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_BlockDiag"
PERMON_EXTERN PetscErrorCode MatCreate_BlockDiag(Mat B)
{
  Mat_BlockDiag *data;

  PetscFunctionBegin;
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATBLOCKDIAG));

  PetscCall(PetscNew(&data));
  B->data         = (void *)data;
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  data->cols_loc   = NULL;
  data->localBlock = NULL;
  data->xloc       = NULL;
  data->yloc       = NULL;
  data->xloc1      = NULL;
  data->yloc1      = NULL;

  /* Set operations of matrix. */
  B->ops->destroy                 = MatDestroy_BlockDiag;
  B->ops->mult                    = MatMult_BlockDiag;
  B->ops->multtranspose           = MatMultTranspose_BlockDiag;
  B->ops->multadd                 = MatMultAdd_BlockDiag;
  B->ops->multtransposeadd        = MatMultTransposeAdd_BlockDiag;
  B->ops->duplicate               = MatDuplicate_BlockDiag;
  B->ops->getinfo                 = MatGetInfo_BlockDiag;
  B->ops->setoption               = MatSetOption_BlockDiag;
  B->ops->getdiagonal             = MatGetDiagonal_BlockDiag;
  B->ops->getdiagonalblock        = MatGetDiagonalBlock_BlockDiag;
  B->ops->view                    = MatView_BlockDiag;
  B->ops->assemblybegin           = MatAssemblyBegin_BlockDiag;
  B->ops->assemblyend             = MatAssemblyEnd_BlockDiag;
  B->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_BlockDiag;
  B->ops->setvalueslocal          = MatSetValuesLocal_BlockDiag;
  B->ops->zeroentries             = MatZeroEntries_BlockDiag;
  B->ops->zerorows                = MatZeroRows_BlockDiag;
  B->ops->zerorowscolumns         = MatZeroRowsColumns_BlockDiag;
  B->ops->scale                   = MatScale_BlockDiag;
  B->ops->productsetfromoptions   = MatProductSetFromOptions_BlockDiag;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatGetColumnVectors_C", MatGetColumnVectors_BlockDiag));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatRestoreColumnVectors_C", MatRestoreColumnVectors_BlockDiag));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MaProductSetFromOptions_blockdiag_aij_C", MatProductSetFromOptions_BlockDiag_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MaProductSetFromOptions_blockdiag_seqaij_C", MatProductSetFromOptions_BlockDiag_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MaProductSetFromOptions_blockdiag_mpiaij", MatProductSetFromOptions_BlockDiag_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_blockdiag_aij_C", MatConvert_BlockDiag_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatOrthColumns_C", MatOrthColumns_BlockDiag));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "PermonMatConvertBlocks_C", PermonMatConvertBlocks_BlockDiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO comment, collective
//TODO MatBlockDiagSetDiagonalBlock
#undef __FUNCT__
#define __FUNCT__ "MatCreateBlockDiag"
PetscErrorCode MatCreateBlockDiag(MPI_Comm comm, Mat block, Mat *B_new)
{
  Mat_BlockDiag *data;
  PetscInt       rlo, rhi, clo, chi;
  PetscMPIInt    size;
  Mat            B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(block, MAT_CLASSID, 2);
  PetscAssertPointer(B_new, 3);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)block), &size));
  PetscCheck(size == 1, comm, PETSC_ERR_ARG_WRONG, "block (arg #2) must be sequential");

  /* Create matrix. */
  PetscCall(MatCreate(comm, &B));
  PetscCall(MatSetType(B, MATBLOCKDIAG));
  data = (Mat_BlockDiag *)B->data;

  /* Matrix data. */
  if (!block) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 0, 0, NULL, &block));
  } else {
    PetscCall(PetscObjectReference((PetscObject)block));
  }
  data->localBlock = block;

  /* Set up row layout */
  PetscCall(PetscLayoutSetBlockSize(B->rmap, block->rmap->bs));
  PetscCall(PetscLayoutSetLocalSize(B->rmap, block->rmap->n));
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutGetRange(B->rmap, &rlo, &rhi));

  /* Set up column layout */
  PetscCall(PetscLayoutSetBlockSize(B->cmap, block->cmap->bs));
  PetscCall(PetscLayoutSetLocalSize(B->cmap, block->cmap->n));
  PetscCall(PetscLayoutSetUp(B->cmap));
  PetscCall(PetscLayoutGetRange(B->cmap, &clo, &chi));

  /* Intermediate vectors for MatMult. */
  PetscCall(MatCreateVecs(block, &data->xloc, &data->yloc));
  PetscCall(VecDuplicate(data->yloc, &data->yloc1));
  PetscCall(VecDuplicate(data->xloc, &data->xloc1));

  /* TODO: test bs > 1 */
  {
    PetscInt              *l2grarr, *l2gcarr, i;
    IS                     l2gris, l2gcis;
    ISLocalToGlobalMapping l2gr, l2gc;

    if (B->rmap->bs > 1) {
      PetscCall(PetscMalloc1(B->rmap->n, &l2grarr));
      for (i = 0; i < B->rmap->n / B->rmap->bs; i++) l2grarr[i] = rlo / B->rmap->bs + i;
      PetscCall(ISCreateBlock(comm, B->rmap->bs, B->rmap->n, l2grarr, PETSC_OWN_POINTER, &l2gris));
    } else {
      PetscCall(ISCreateStride(comm, B->rmap->n, rlo, 1, &l2gris));
    }
    PetscCall(ISLocalToGlobalMappingCreateIS(l2gris, &l2gr));
    PetscCall(PetscLayoutSetISLocalToGlobalMapping(B->rmap, l2gr));
    PetscCall(ISDestroy(&l2gris));
    PetscCall(ISLocalToGlobalMappingDestroy(&l2gr));

    if (B->cmap->bs > 1) {
      PetscCall(PetscMalloc1(B->cmap->n, &l2gcarr));
      for (i = 0; i < B->cmap->n / B->cmap->bs; i++) l2gcarr[i] = clo / B->cmap->bs + i;
      PetscCall(ISCreateBlock(comm, B->cmap->bs, B->cmap->n, l2gcarr, PETSC_OWN_POINTER, &l2gcis));
    } else {
      PetscCall(ISCreateStride(comm, B->cmap->n, clo, 1, &l2gcis));
    }
    PetscCall(ISLocalToGlobalMappingCreateIS(l2gcis, &l2gc));
    PetscCall(PetscLayoutSetISLocalToGlobalMapping(B->cmap, l2gc));
    PetscCall(ISDestroy(&l2gcis));
    PetscCall(ISLocalToGlobalMappingDestroy(&l2gc));
  }

  PetscCall(MatInheritSymmetry(block, B));
  *B_new = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}
