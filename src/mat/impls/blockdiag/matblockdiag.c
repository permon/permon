
#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#define TAG_firstElemGlobIdx 198533

static PetscErrorCode MatGetDiagonalBlock_BlockDiag(Mat,Mat*);

#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsColumns_BlockDiag"
PetscErrorCode MatZeroRowsColumns_BlockDiag(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) A->data;
  Mat Aloc = data->localBlock;
  Vec xloc = data->xloc, bloc = data->yloc;
  PetscInt nloc, *rows_loc;

  PetscFunctionBegin;
  CHKERRQ(ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,&nloc,NULL));
  PERMON_ASSERT(n==nloc,"n==nloc");
  CHKERRQ(PetscMalloc(n*sizeof(PetscInt),&rows_loc));
  CHKERRQ(ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,NULL,rows_loc));
  if (x) {
    CHKERRQ(VecGetLocalVectorRead(x,xloc));
  } else {
    xloc = NULL;
  }
  if (b) {
    CHKERRQ(VecGetLocalVector(b,bloc));
  } else {
    bloc = NULL;
  }
  CHKERRQ(MatZeroRowsColumns(Aloc,nloc,rows_loc,diag,xloc,bloc));
  if (x) CHKERRQ(VecRestoreLocalVectorRead(x,xloc));
  if (b) CHKERRQ(VecRestoreLocalVector(b,bloc));
  CHKERRQ(PetscFree(rows_loc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroRows_BlockDiag"
PetscErrorCode MatZeroRows_BlockDiag(Mat A,PetscInt n,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) A->data;
  Mat Aloc = data->localBlock;
  Vec xloc = data->xloc, bloc = data->yloc;
  PetscInt nloc, *rows_loc;

  PetscFunctionBegin;
  CHKERRQ(ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,&nloc,NULL));
  PERMON_ASSERT(n==nloc,"n==nloc");
  CHKERRQ(PetscMalloc(n*sizeof(PetscInt),&rows_loc));
  CHKERRQ(ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,NULL,rows_loc));
  if (x) {
    CHKERRQ(VecGetLocalVectorRead(x,xloc));
  } else {
    xloc = NULL;
  }
  if (b) {
    CHKERRQ(VecGetLocalVector(b,bloc));
  } else {
    bloc = NULL;
  }
  CHKERRQ(MatZeroRows(Aloc,nloc,rows_loc,diag,xloc,bloc));
  if (x) CHKERRQ(VecRestoreLocalVectorRead(x,xloc));
  if (b) CHKERRQ(VecRestoreLocalVector(b,bloc));
  CHKERRQ(PetscFree(rows_loc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_BlockDiag"
PetscErrorCode MatZeroEntries_BlockDiag(Mat A)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) A->data;

  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(data->localBlock));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_BlockDiag_SeqAIJ"
static PetscErrorCode MatConvert_BlockDiag_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newB)
{
  MPI_Comm comm;
  PetscMPIInt size;
  Mat A_loc, B;
  
  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) A, &comm));
  CHKERRQ(MPI_Comm_size(comm, &size));
  if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"conversion from MPI BlockDiag matrix to sequential matrix not currently implemented");

  CHKERRQ(MatGetDiagonalBlock(A, &A_loc));
  CHKERRQ(MatConvert(A_loc,newtype,MAT_INITIAL_MATRIX,&B));
  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    CHKERRQ(MatHeaderReplace(A,B));
#else
    CHKERRQ(MatHeaderReplace(A,&B));
#endif
  } else {
    *newB = B;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_BlockDiag_MPIAIJ"
static PetscErrorCode MatConvert_BlockDiag_MPIAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newB)
{
  PetscMPIInt size;
  Mat A_loc, B;
  char *loctype;
  Mat_MPIAIJ *mpidata;
  
  PetscFunctionBegin;
  CHKERRQ(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  CHKERRQ(MatGetDiagonalBlock(A, &A_loc));

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A), &B));
  CHKERRQ(MatSetType(B, newtype));
  CHKERRQ(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  CHKERRQ(MatMPIAIJSetPreallocation(B, 0,0,0,0));

  mpidata = (Mat_MPIAIJ*) B->data;
  CHKERRQ(PetscStrallocpy( ((PetscObject)mpidata->A)->type_name,&loctype));
  CHKERRQ(MatDestroy(&mpidata->A));
  CHKERRQ(MatConvert(A_loc,loctype,MAT_INITIAL_MATRIX,&mpidata->A));
  CHKERRQ(PetscFree(loctype));

  /* MatConvert should produce an already assembled matrix so we just set the 'assembled' flag to true */
  B->assembled = PETSC_TRUE;

  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    CHKERRQ(MatHeaderReplace(A,B));
#else
    CHKERRQ(MatHeaderReplace(A,&B));
#endif
  } else {
    *newB = B;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_BlockDiag_AIJ"
static PetscErrorCode MatConvert_BlockDiag_AIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newB)
{
  PetscMPIInt size;
  
  PetscFunctionBegin;
  CHKERRQ(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) {
    CHKERRQ(MatConvert_BlockDiag_SeqAIJ(A,newtype,reuse,newB));    
  } else {
    CHKERRQ(MatConvert_BlockDiag_MPIAIJ(A,newtype,reuse,newB));    
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatConvertBlocks_BlockDiag"
static PetscErrorCode PermonMatConvertBlocks_BlockDiag(Mat A, MatType newtype, MatReuse reuse, Mat* B)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) A->data;
  Mat cblock = NULL;
  Mat B_;

  PetscFunctionBegin;
  if (reuse == MAT_INPLACE_MATRIX) cblock = data->localBlock;
  CHKERRQ(MatConvert(data->localBlock,newtype,reuse,&cblock));
  CHKERRQ(MatCreateBlockDiag(PetscObjectComm((PetscObject)A),cblock,&B_));
  if (reuse != MAT_INPLACE_MATRIX) {
    CHKERRQ(MatDestroy(&cblock));
    *B = B_;
  } else {
#if PETSC_VERSION_MINOR < 7
    CHKERRQ(MatHeaderReplace(A,B_));
#else
    CHKERRQ(MatHeaderReplace(A,&B_));
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_BlockDiag"
PetscErrorCode MatMult_BlockDiag(Mat mat, Vec right, Vec left) {
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorRead(right,data->xloc));
  CHKERRQ(VecGetLocalVector(left,data->yloc));
  CHKERRQ(MatMult(data->localBlock, data->xloc, data->yloc));
  CHKERRQ(VecRestoreLocalVectorRead(right,data->xloc));
  CHKERRQ(VecRestoreLocalVector(left,data->yloc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_BlockDiag"
PetscErrorCode MatMultTranspose_BlockDiag(Mat mat, Vec right, Vec left) {
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorRead(right,data->yloc));
  CHKERRQ(VecGetLocalVector(left,data->xloc));
  CHKERRQ(MatMultTranspose(data->localBlock, data->yloc, data->xloc));
  CHKERRQ(VecRestoreLocalVectorRead(right,data->yloc));
  CHKERRQ(VecRestoreLocalVector(left,data->xloc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_BlockDiag"
PetscErrorCode MatMultAdd_BlockDiag(Mat mat,Vec v1,Vec v2,Vec v3)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorRead(v1,data->xloc));
  CHKERRQ(VecGetLocalVector(v2,data->yloc1)); /* v2 can be same as v3 */
  CHKERRQ(VecGetLocalVector(v3,data->yloc));
  CHKERRQ(MatMultAdd(data->localBlock, data->xloc, data->yloc1, data->yloc));
  CHKERRQ(VecRestoreLocalVectorRead(v1,data->xloc));
  CHKERRQ(VecRestoreLocalVector(v2,data->yloc1));
  CHKERRQ(VecRestoreLocalVector(v3,data->yloc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_BlockDiag"
PetscErrorCode MatMultTransposeAdd_BlockDiag(Mat mat,Vec v1,Vec v2,Vec v3)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;
  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorRead(v1,data->yloc));
  CHKERRQ(VecGetLocalVector(v2,data->xloc1)); /* v2 can be same as v3 */
  CHKERRQ(VecGetLocalVector(v3,data->xloc));
  CHKERRQ(MatMultTransposeAdd(data->localBlock, data->yloc, data->xloc1, data->xloc));
  CHKERRQ(VecRestoreLocalVectorRead(v1,data->yloc));
  CHKERRQ(VecRestoreLocalVector(v2,data->xloc1));
  CHKERRQ(VecRestoreLocalVector(v3,data->xloc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_BlockDiag_BlockDiag"
PetscErrorCode MatMatMult_BlockDiag_BlockDiag(Mat A, Mat B, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) A, &comm));
  CHKERRQ(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  CHKERRQ(MatGetDiagonalBlock_BlockDiag(B, &B_loc));
  CHKERRQ(MatMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  CHKERRQ(MatCreateBlockDiag(comm,C_loc,C));
  CHKERRQ(MatDestroy(&C_loc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_BlockDiag_AIJ"
static PetscErrorCode MatMatMult_BlockDiag_AIJ(Mat A, Mat B, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;
  Vec x;
  PetscMPIInt size;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) A, &comm));
  CHKERRQ(MPI_Comm_size(comm, &size));
  CHKERRQ(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  if (size > 1) {
    CHKERRQ(MatMPIAIJGetLocalMat(B, MAT_INITIAL_MATRIX, &B_loc));
  } else {
    B_loc = B;
    CHKERRQ(PetscObjectReference((PetscObject)B));
  }
  CHKERRQ(MatMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  CHKERRQ(MatDestroy(&B_loc));

  CHKERRQ(MatCreateVecs(B, &x, NULL));
  CHKERRQ(MatMergeAndDestroy(comm, &C_loc, x, C));
  CHKERRQ(VecDestroy(&x));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_BlockDiag"
PetscErrorCode MatTransposeMatMult_BlockDiag_BlockDiag(Mat A, Mat B, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) A, &comm));
  CHKERRQ(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  CHKERRQ(MatGetDiagonalBlock_BlockDiag(B, &B_loc));
  CHKERRQ(MatTransposeMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  CHKERRQ(MatCreateBlockDiag(comm,C_loc,C));
  CHKERRQ(MatDestroy(&C_loc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_AIJ"
static PetscErrorCode MatTransposeMatMult_BlockDiag_AIJ(Mat A, Mat B, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;
  Vec x;
  PetscMPIInt size;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) A, &comm));
  CHKERRQ(MPI_Comm_size(comm, &size));
  CHKERRQ(MatGetDiagonalBlock_BlockDiag(A, &A_loc));
  if (size > 1) {
    CHKERRQ(MatMPIAIJGetLocalMat(B, MAT_INITIAL_MATRIX, &B_loc));
  } else {
    B_loc = B;
    CHKERRQ(PetscObjectReference((PetscObject)B));
  }
  CHKERRQ(MatTransposeMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc));
  CHKERRQ(MatDestroy(&B_loc));

  CHKERRQ(MatCreateVecs(B, &x, NULL));
  CHKERRQ(MatMergeAndDestroy(comm, &C_loc, x, C));
  CHKERRQ(VecDestroy(&x));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_BlockDiag_AIJ"
static PetscErrorCode MatProductNumeric_BlockDiag_AIJ(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  Mat            newmat;

  switch (product->type) {
  case MATPRODUCT_AB:
    CHKERRQ(MatMatMult_BlockDiag_AIJ(A,B,product->fill,&newmat));
    break;
  case MATPRODUCT_AtB:
    CHKERRQ(MatTransposeMatMult_BlockDiag_AIJ(A,B,product->fill,&newmat));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MATPRODUCT type is not supported");
  }
  C->product = NULL;
  CHKERRQ(MatHeaderReplace(C,&newmat));
  C->product = product;
  C->ops->productnumeric = MatProductNumeric_BlockDiag_AIJ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_BlockDiag_AIJ"
static PetscErrorCode MatProductSymbolic_BlockDiag_AIJ(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric  = MatProductNumeric_BlockDiag_AIJ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_BlockDiag_AIJ"
static PetscErrorCode MatProductSetFromOptions_BlockDiag_AIJ(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_BlockDiag_AIJ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_BlockDiag"
static PetscErrorCode MatProductNumeric_BlockDiag(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  Mat            newmat;

  switch (product->type) {
  case MATPRODUCT_AB:
    CHKERRQ(MatMatMult_BlockDiag_BlockDiag(A,B,product->fill,&newmat));
    break;
  case MATPRODUCT_AtB:
    CHKERRQ(MatTransposeMatMult_BlockDiag_BlockDiag(A,B,product->fill,&newmat));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MATPRODUCT type is not supported");
  }
  C->product = NULL;
  CHKERRQ(MatHeaderReplace(C,&newmat));
  C->product = product;
  C->ops->productnumeric = MatProductNumeric_BlockDiag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_BlockDiag"
static PetscErrorCode MatProductSymbolic_BlockDiag(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric  = MatProductNumeric_BlockDiag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_BlockDiag"
static PetscErrorCode MatProductSetFromOptions_BlockDiag(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_BlockDiag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_BlockDiag"
PetscErrorCode MatDestroy_BlockDiag(Mat mat) {
  Mat_BlockDiag *data;

  PetscFunctionBegin;
  data = (Mat_BlockDiag*) mat->data;
  CHKERRQ(MatDestroy(&data->localBlock));
  CHKERRQ(VecDestroy(&data->xloc));
  CHKERRQ(VecDestroy(&data->yloc));
  CHKERRQ(VecDestroy(&data->xloc1));
  CHKERRQ(VecDestroy(&data->yloc1));
  CHKERRQ(PetscFree(data));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_BlockDiag"
PetscErrorCode MatDuplicate_BlockDiag(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat matout;
  Mat_BlockDiag *datain,*dataout;

  PetscFunctionBegin;
  datain = (Mat_BlockDiag*) matin->data;
  
  CHKERRQ(MatCreate(((PetscObject)matin)->comm,&matout));
  CHKERRQ(MatSetSizes(matout,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N));
  CHKERRQ(MatSetType(matout,((PetscObject)matin)->type_name));
  CHKERRQ(PetscMemcpy(matout->ops,matin->ops,sizeof(struct _MatOps)));
  dataout = (Mat_BlockDiag*) matout->data;
  
  CHKERRQ(MatDuplicate(datain->localBlock,cpvalues,&dataout->localBlock));
  CHKERRQ(VecDuplicate(datain->yloc,&dataout->yloc));
  CHKERRQ(VecDuplicate(datain->yloc1,&dataout->yloc1));
  CHKERRQ(VecDuplicate(datain->xloc,&dataout->xloc));
  *newmat = matout;
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock_BlockDiag"
static PetscErrorCode MatGetDiagonalBlock_BlockDiag(Mat A, Mat *A_loc) {
  Mat_BlockDiag *data = (Mat_BlockDiag*) A->data;

  PetscFunctionBegin;
  *A_loc = data->localBlock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_BlockDiag"
PetscErrorCode MatGetInfo_BlockDiag(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_BlockDiag  *mat = (Mat_BlockDiag*) matin->data;
  Mat            A = mat->localBlock;
  PetscReal      isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size     = 1.0;
  CHKERRQ(MatGetInfo(A,MAT_LOCAL,info));
  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;

  switch (flag) {
    case MAT_LOCAL:
      info->nz_used      = isend[0];
      info->nz_allocated = isend[1];
      info->nz_unneeded  = isend[2];
      info->memory       = isend[3];
      info->mallocs      = isend[4];
      break;
    case MAT_GLOBAL_MAX:
      CHKERRQ(MPI_Allreduce(isend, irecv, 5, MPIU_REAL, MPIU_MAX, ((PetscObject) matin)->comm));
      info->nz_used      = irecv[0];
      info->nz_allocated = irecv[1];
      info->nz_unneeded  = irecv[2];
      info->memory       = irecv[3];
      info->mallocs      = irecv[4];
      break;
    case MAT_GLOBAL_SUM:
      CHKERRQ(MPI_Allreduce(isend, irecv, 5, MPIU_REAL, MPIU_SUM, ((PetscObject) matin)->comm));
      info->nz_used      = irecv[0];
      info->nz_allocated = irecv[1];
      info->nz_unneeded  = irecv[2];
      info->memory       = irecv[3];
      info->mallocs      = irecv[4];
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_BlockDiag"
PetscErrorCode MatSetOption_BlockDiag(Mat mat, MatOption op, PetscBool flg)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  CHKERRQ(MatSetOption(bd->localBlock, op, flg));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_BlockDiag"
PetscErrorCode MatGetDiagonal_BlockDiag(Mat mat,Vec d)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  CHKERRQ(MatGetDiagonal(bd->localBlock, bd->yloc));
  CHKERRQ(VecCopy(bd->yloc, d));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_BlockDiag"
PetscErrorCode MatView_BlockDiag(Mat mat,PetscViewer viewer)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  PetscViewer sv;
  MPI_Comm comm;
  PetscBool iascii;
  PetscViewerFormat format;
  
  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (!iascii) SETERRQ(comm,PETSC_ERR_SUP,"Viewer type %s not supported for matrix type %s",((PetscObject)viewer)->type,((PetscObject)mat)->type_name);
  CHKERRQ(PetscViewerGetFormat(viewer,&format));

  if (format == PETSC_VIEWER_DEFAULT) {
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)mat,viewer));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
  }

  CHKERRQ(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sv));
  if (format != PETSC_VIEWER_DEFAULT) {
    PetscMPIInt rank;
    CHKERRQ(MPI_Comm_rank(comm, &rank));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"diagonal block on rank 0:\n"));
    if (!rank) CHKERRQ(MatView(bd->localBlock,sv));
  } else {
    CHKERRQ(PetscSequentialPhaseBegin(comm,1));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"diagonal blocks:\n"));
    CHKERRQ(MatView(bd->localBlock,sv));
    CHKERRQ(PetscSequentialPhaseEnd(comm,1));
  }
  CHKERRQ(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sv));

  if (format == PETSC_VIEWER_DEFAULT) {
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_BlockDiag"
PetscErrorCode MatAssemblyBegin_BlockDiag(Mat mat, MatAssemblyType type)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  CHKERRQ(MatAssemblyBegin(bd->localBlock, type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_BlockDiag"
PetscErrorCode MatAssemblyEnd_BlockDiag(Mat mat, MatAssemblyType type)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  CHKERRQ(MatAssemblyEnd(bd->localBlock, type));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetLocalToGlobalMapping_BlockDiag"
PetscErrorCode MatSetLocalToGlobalMapping_BlockDiag(Mat x,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"custom LocalToGlobalMapping not allowed for matrix of type %s",((PetscObject)x)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesLocal_BlockDiag"
PetscErrorCode MatSetValuesLocal_BlockDiag(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  CHKERRQ(MatSetValues(data->localBlock,nrow,irow,ncol,icol,y,addv));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_BlockDiag"
PetscErrorCode MatScale_BlockDiag(Mat mat,PetscScalar a)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  CHKERRQ(MatScale(data->localBlock,a));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors_BlockDiag"
static PetscErrorCode MatGetColumnVectors_BlockDiag(Mat mat, Vec *cols_new[])
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;
  PetscInt n,n1,N,j,jlo;
  Vec d,*cols;
  PetscScalar *arr;

  PetscFunctionBegin;
  n = mat->cmap->n;
  N = mat->cmap->N;
  jlo = mat->cmap->rstart;
  CHKERRQ(MatGetColumnVectors(data->localBlock,&n1,&data->cols_loc));
  PERMON_ASSERT(n==n1,"n==n1");

  CHKERRQ(MatCreateVecs(mat, PETSC_IGNORE, &d));
  CHKERRQ(VecDuplicateVecs(d, N, &cols));
  CHKERRQ(VecDestroy(&d));

  for (j=0; j<N; j++) {
    CHKERRQ(VecSet(cols[j],0.0));
  }

  for (j=0; j<n; j++) {
    CHKERRQ(VecGetArray(data->cols_loc[j],&arr));
    CHKERRQ(VecPlaceArray(cols[j+jlo],arr));
  }

  *cols_new=cols;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_BlockDiag"
static PetscErrorCode MatRestoreColumnVectors_BlockDiag(Mat mat, Vec *cols[])
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;
  PetscInt n,N,j,jlo;
  
  PetscFunctionBegin;
  n = mat->cmap->n;
  N = mat->cmap->N;
  jlo = mat->cmap->rstart;

  for (j=0; j<n; j++) {
    CHKERRQ(VecRestoreArray(data->cols_loc[j],NULL));
    CHKERRQ(VecResetArray((*cols)[j+jlo]));
  }

  CHKERRQ(MatRestoreColumnVectors(data->localBlock,NULL,&data->cols_loc));
  CHKERRQ(VecDestroyVecs(N,cols));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_BlockDiag"
static PetscErrorCode MatOrthColumns_BlockDiag(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) A->data;
  Mat Q_loc=NULL,S_loc=NULL;

  PetscFunctionBegin;
  CHKERRQ(MatOrthColumns(bd->localBlock, type, form, Q_new?(&Q_loc):NULL, S_new?(&S_loc):NULL));
  if (Q_new) CHKERRQ(MatCreateBlockDiag(PetscObjectComm((PetscObject)A),Q_loc,Q_new));
  if (S_new) CHKERRQ(MatCreateBlockDiag(PetscObjectComm((PetscObject)A),S_loc,S_new));
  CHKERRQ(MatDestroy(&Q_loc));
  CHKERRQ(MatDestroy(&S_loc));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_BlockDiag"
FLLOP_EXTERN PetscErrorCode MatCreate_BlockDiag(Mat B) {
  Mat_BlockDiag *data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATBLOCKDIAG));

  CHKERRQ(PetscNewLog(B,&data));
  B->data                    = (void*) data;
  B->assembled               = PETSC_TRUE;
  B->preallocated            = PETSC_TRUE;

  data->cols_loc             = NULL;
  data->localBlock           = NULL;
  data->xloc                 = NULL;
  data->yloc                 = NULL;
  data->xloc1                = NULL;
  data->yloc1                = NULL;

  /* Set operations of matrix. */
  B->ops->destroy            = MatDestroy_BlockDiag;
  B->ops->mult               = MatMult_BlockDiag;
  B->ops->multtranspose      = MatMultTranspose_BlockDiag;
  B->ops->multadd            = MatMultAdd_BlockDiag;
  B->ops->multtransposeadd   = MatMultTransposeAdd_BlockDiag;
  B->ops->duplicate          = MatDuplicate_BlockDiag;
  B->ops->getinfo            = MatGetInfo_BlockDiag;
  B->ops->setoption          = MatSetOption_BlockDiag;
  B->ops->getdiagonal        = MatGetDiagonal_BlockDiag;
  B->ops->getdiagonalblock   = MatGetDiagonalBlock_BlockDiag;
  B->ops->view               = MatView_BlockDiag;
  B->ops->assemblybegin      = MatAssemblyBegin_BlockDiag;
  B->ops->assemblyend        = MatAssemblyEnd_BlockDiag;
  B->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_BlockDiag;
  B->ops->setvalueslocal     = MatSetValuesLocal_BlockDiag;
  B->ops->zeroentries        = MatZeroEntries_BlockDiag;
  B->ops->zerorows           = MatZeroRows_BlockDiag;
  B->ops->zerorowscolumns    = MatZeroRowsColumns_BlockDiag;
  B->ops->scale              = MatScale_BlockDiag;
  B->ops->productsetfromoptions = MatProductSetFromOptions_BlockDiag;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_BlockDiag));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_BlockDiag));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MaProductSetFromOptions_blockdiag_aij_C",MatProductSetFromOptions_BlockDiag_AIJ));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MaProductSetFromOptions_blockdiag_seqaij_C",MatProductSetFromOptions_BlockDiag_AIJ));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MaProductSetFromOptions_blockdiag_mpiaij",MatProductSetFromOptions_BlockDiag_AIJ));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_blockdiag_aij_C",MatConvert_BlockDiag_AIJ));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatOrthColumns_C",MatOrthColumns_BlockDiag));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"PermonMatConvertBlocks_C",PermonMatConvertBlocks_BlockDiag));
  PetscFunctionReturn(0);
}

//TODO comment, collective
//TODO MatBlockDiagSetDiagonalBlock
#undef __FUNCT__  
#define __FUNCT__ "MatCreateBlockDiag"
PetscErrorCode MatCreateBlockDiag(MPI_Comm comm, Mat block, Mat *B_new) {
  Mat_BlockDiag *data;
  PetscInt rlo,rhi,clo,chi;
  PetscMPIInt size;
  Mat B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(block,MAT_CLASSID,2);
  PetscValidPointer(B_new,3);
  CHKERRQ(MPI_Comm_size(PetscObjectComm((PetscObject)block),&size));
  if (size > 1) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"block (arg #2) must be sequential");
  
  /* Create matrix. */  
  CHKERRQ(MatCreate(comm, &B));
  CHKERRQ(MatSetType(B, MATBLOCKDIAG));
  data = (Mat_BlockDiag*) B->data;
  
  /* Matrix data. */
  if (!block) {
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,0,0,NULL,&block));
  } else {
      CHKERRQ(PetscObjectReference((PetscObject) block));
  }
  data->localBlock = block;

  /* Set up row layout */
  CHKERRQ(PetscLayoutSetBlockSize(B->rmap,block->rmap->bs));
  CHKERRQ(PetscLayoutSetLocalSize(B->rmap,block->rmap->n));
  CHKERRQ(PetscLayoutSetUp(B->rmap));
  CHKERRQ(PetscLayoutGetRange(B->rmap,&rlo,&rhi));
  
  /* Set up column layout */
  CHKERRQ(PetscLayoutSetBlockSize(B->cmap,block->cmap->bs));
  CHKERRQ(PetscLayoutSetLocalSize(B->cmap,block->cmap->n));
  CHKERRQ(PetscLayoutSetUp(B->cmap));
  CHKERRQ(PetscLayoutGetRange(B->cmap,&clo,&chi));
  
  /* Intermediate vectors for MatMult. */
  CHKERRQ(MatCreateVecs(block, &data->xloc, &data->yloc));
  CHKERRQ(VecDuplicate(data->yloc, &data->yloc1));
  CHKERRQ(VecDuplicate(data->xloc, &data->xloc1));

  /* TODO: test bs > 1 */
  {
    PetscInt *l2grarr,*l2gcarr,i;
    IS l2gris,l2gcis;
    ISLocalToGlobalMapping l2gr,l2gc;

    if (B->rmap->bs > 1) {
      CHKERRQ(PetscMalloc1(B->rmap->n,&l2grarr));
      for (i = 0; i < B->rmap->n/B->rmap->bs; i++) l2grarr[i] = rlo/B->rmap->bs + i;
      CHKERRQ(ISCreateBlock(comm,B->rmap->bs,B->rmap->n,l2grarr,PETSC_OWN_POINTER,&l2gris));
    } else {
      CHKERRQ(ISCreateStride(comm,B->rmap->n,rlo,1,&l2gris));
    }
    CHKERRQ(ISLocalToGlobalMappingCreateIS(l2gris,&l2gr));
    CHKERRQ(PetscLayoutSetISLocalToGlobalMapping(B->rmap,l2gr));
    CHKERRQ(ISDestroy(&l2gris));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&l2gr));

    if (B->cmap->bs > 1) {
      CHKERRQ(PetscMalloc1(B->cmap->n,&l2gcarr));
      for (i = 0; i < B->cmap->n/B->cmap->bs; i++) l2gcarr[i] = clo/B->cmap->bs + i;
      CHKERRQ(ISCreateBlock(comm,B->cmap->bs,B->cmap->n,l2gcarr,PETSC_OWN_POINTER,&l2gcis));
    } else {
      CHKERRQ(ISCreateStride(comm,B->cmap->n,clo,1,&l2gcis));
    }
    CHKERRQ(ISLocalToGlobalMappingCreateIS(l2gcis,&l2gc));
    CHKERRQ(PetscLayoutSetISLocalToGlobalMapping(B->cmap,l2gc));
    CHKERRQ(ISDestroy(&l2gcis));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&l2gc));
  }

  CHKERRQ(MatInheritSymmetry(block,B));
  *B_new = B;
  PetscFunctionReturn(0);
}
