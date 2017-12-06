
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
  TRY( ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,&nloc,NULL) );
  FLLOP_ASSERT(n==nloc,"n==nloc");
  TRY( PetscMalloc(n*sizeof(PetscInt),&rows_loc) );
  TRY( ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,NULL,rows_loc) );
  if (x) {
    TRY( VecGetLocalVectorRead(x,xloc) );
  } else {
    xloc = NULL;
  }
  if (b) {
    TRY( VecGetLocalVector(b,bloc) );
  } else {
    bloc = NULL;
  }
  TRY( MatZeroRowsColumns(Aloc,nloc,rows_loc,diag,xloc,bloc) );
  if (x) TRY( VecRestoreLocalVectorRead(x,xloc) );
  if (b) TRY( VecRestoreLocalVector(b,bloc) );
  TRY( PetscFree(rows_loc) );
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
  TRY( ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,&nloc,NULL) );
  FLLOP_ASSERT(n==nloc,"n==nloc");
  TRY( PetscMalloc(n*sizeof(PetscInt),&rows_loc) );
  TRY( ISGlobalToLocalMappingApply(A->rmap->mapping,IS_GTOLM_DROP,n,rows,NULL,rows_loc) );
  if (x) {
    TRY( VecGetLocalVectorRead(x,xloc) );
  } else {
    xloc = NULL;
  }
  if (b) {
    TRY( VecGetLocalVector(b,bloc) );
  } else {
    bloc = NULL;
  }
  TRY( MatZeroRows(Aloc,nloc,rows_loc,diag,xloc,bloc) );
  if (x) TRY( VecRestoreLocalVectorRead(x,xloc) );
  if (b) TRY( VecRestoreLocalVector(b,bloc) );
  TRY( PetscFree(rows_loc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_BlockDiag"
PetscErrorCode MatZeroEntries_BlockDiag(Mat A)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) A->data;

  PetscFunctionBegin;
  TRY( MatZeroEntries(data->localBlock) );
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
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MPI_Comm_size(comm, &size) );
  if (size > 1) FLLOP_SETERRQ(comm,PETSC_ERR_SUP,"conversion from MPI BlockDiag matrix to sequential matrix not currently implemented");

  TRY( MatGetDiagonalBlock(A, &A_loc) );
  TRY( MatConvert(A_loc,newtype,MAT_INITIAL_MATRIX,&B) );
  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    TRY( MatHeaderReplace(A,B) );
#else
    TRY( MatHeaderReplace(A,&B) );
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
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)A), &size) );
  TRY( MatGetDiagonalBlock(A, &A_loc) );

  TRY( MatCreate(PetscObjectComm((PetscObject)A), &B) );
  TRY( MatSetType(B, newtype) );
  TRY( MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N) );
  TRY( MatMPIAIJSetPreallocation(B, 0,0,0,0) );

  mpidata = (Mat_MPIAIJ*) B->data;
  TRY( PetscStrallocpy( ((PetscObject)mpidata->A)->type_name,&loctype) );
  TRY( MatDestroy(&mpidata->A) );
  TRY( MatConvert(A_loc,loctype,MAT_INITIAL_MATRIX,&mpidata->A) );
  TRY( PetscFree(loctype) );

  /* MatConvert should produce an already assembled matrix so we just set the 'assembled' flag to true */
  B->assembled = PETSC_TRUE;

  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    TRY( MatHeaderReplace(A,B) );
#else
    TRY( MatHeaderReplace(A,&B) );
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
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)A),&size) );
  if (size == 1) {
    TRY( MatConvert_BlockDiag_SeqAIJ(A,newtype,reuse,newB) );    
  } else {
    TRY( MatConvert_BlockDiag_MPIAIJ(A,newtype,reuse,newB) );    
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
  TRY( MatConvert(data->localBlock,newtype,reuse,&cblock) );
  TRY( MatCreateBlockDiag(PetscObjectComm((PetscObject)A),cblock,&B_) );
  if (reuse != MAT_INPLACE_MATRIX) {
    TRY( MatDestroy(&cblock) );
    *B = B_;
  } else {
#if PETSC_VERSION_MINOR < 7
    TRY( MatHeaderReplace(A,B_) );
#else
    TRY( MatHeaderReplace(A,&B_) );
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_BlockDiag"
PetscErrorCode MatMult_BlockDiag(Mat mat, Vec right, Vec left) {
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  TRY( VecGetLocalVectorRead(right,data->xloc) );
  TRY( VecGetLocalVector(left,data->yloc) );
  TRY( MatMult(data->localBlock, data->xloc, data->yloc) );
  TRY( VecRestoreLocalVectorRead(right,data->xloc) );
  TRY( VecRestoreLocalVector(left,data->yloc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_BlockDiag"
PetscErrorCode MatMultTranspose_BlockDiag(Mat mat, Vec right, Vec left) {
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  TRY( VecGetLocalVectorRead(right,data->yloc) );
  TRY( VecGetLocalVector(left,data->xloc) );
  TRY( MatMultTranspose(data->localBlock, data->yloc, data->xloc) );
  TRY( VecRestoreLocalVectorRead(right,data->yloc) );
  TRY( VecRestoreLocalVector(left,data->xloc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_BlockDiag"
PetscErrorCode MatMultAdd_BlockDiag(Mat mat,Vec v1,Vec v2,Vec v3)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  TRY( VecGetLocalVectorRead(v1,data->xloc) );
  TRY( VecGetLocalVector(v2,data->yloc1) ); /* v2 can be same as v3 */
  TRY( VecGetLocalVector(v3,data->yloc) );
  TRY( MatMultAdd(data->localBlock, data->xloc, data->yloc1, data->yloc) );
  TRY( VecRestoreLocalVectorRead(v1,data->xloc) );
  TRY( VecRestoreLocalVector(v2,data->yloc1) );
  TRY( VecRestoreLocalVector(v3,data->yloc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_BlockDiag"
PetscErrorCode MatMultTransposeAdd_BlockDiag(Mat mat,Vec v1,Vec v2,Vec v3)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;
  PetscFunctionBegin;
  TRY( VecGetLocalVectorRead(v1,data->yloc) );
  TRY( VecGetLocalVector(v2,data->xloc1) ); /* v2 can be same as v3 */
  TRY( VecGetLocalVector(v3,data->xloc) );
  TRY( MatMultTransposeAdd(data->localBlock, data->yloc, data->xloc1, data->xloc) );
  TRY( VecRestoreLocalVectorRead(v1,data->yloc) );
  TRY( VecRestoreLocalVector(v2,data->xloc1) );
  TRY( VecRestoreLocalVector(v3,data->xloc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_BlockDiag_BlockDiag"
PetscErrorCode MatMatMult_BlockDiag_BlockDiag(Mat A, Mat B, MatReuse scall, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MatGetDiagonalBlock_BlockDiag(A, &A_loc) );
  TRY( MatGetDiagonalBlock_BlockDiag(B, &B_loc) );
  TRY( MatMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc) );
  TRY( MatCreateBlockDiag(comm,C_loc,C) );
  TRY( MatDestroy(&C_loc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_BlockDiag_AIJ"
static PetscErrorCode MatMatMult_BlockDiag_AIJ(Mat A, Mat B, MatReuse scall, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;
  Vec x;
  PetscMPIInt size;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MPI_Comm_size(comm, &size) );
  TRY( MatGetDiagonalBlock_BlockDiag(A, &A_loc) );
  if (size > 1) {
    TRY( MatMPIAIJGetLocalMat(B, MAT_INITIAL_MATRIX, &B_loc) );
  } else {
    B_loc = B;
    TRY( PetscObjectReference((PetscObject)B) );
  }
  TRY( MatMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc) );
  TRY( MatDestroy(&B_loc) );

  TRY( MatCreateVecs(B, &x, NULL) );
  TRY( MatMergeAndDestroy(comm, &C_loc, x, C) );
  TRY( VecDestroy(&x) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_BlockDiag"
PetscErrorCode MatTransposeMatMult_BlockDiag_BlockDiag(Mat A, Mat B, MatReuse scall, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MatGetDiagonalBlock_BlockDiag(A, &A_loc) );
  TRY( MatGetDiagonalBlock_BlockDiag(B, &B_loc) );
  TRY( MatTransposeMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc) );
  TRY( MatCreateBlockDiag(comm,C_loc,C) );
  TRY( MatDestroy(&C_loc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_AIJ"
static PetscErrorCode MatTransposeMatMult_BlockDiag_AIJ(Mat A, Mat B, MatReuse scall, PetscReal fill, Mat *C) {
  MPI_Comm comm;
  Mat A_loc, B_loc, C_loc;
  Vec x;
  PetscMPIInt size;

  PetscFunctionBegin;
  TRY( PetscObjectGetComm((PetscObject) A, &comm) );
  TRY( MPI_Comm_size(comm, &size) );
  TRY( MatGetDiagonalBlock_BlockDiag(A, &A_loc) );
  if (size > 1) {
    TRY( MatMPIAIJGetLocalMat(B, MAT_INITIAL_MATRIX, &B_loc) );
  } else {
    B_loc = B;
    TRY( PetscObjectReference((PetscObject)B) );
  }
  TRY( MatTransposeMatMult(A_loc, B_loc, MAT_INITIAL_MATRIX, fill, &C_loc) );
  TRY( MatDestroy(&B_loc) );

  TRY( MatCreateVecs(B, &x, NULL) );
  TRY( MatMergeAndDestroy(comm, &C_loc, x, C) );
  TRY( VecDestroy(&x) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_BlockDiag"
PetscErrorCode MatDestroy_BlockDiag(Mat mat) {
  Mat_BlockDiag *data;

  PetscFunctionBegin;
  data = (Mat_BlockDiag*) mat->data;
  TRY( MatDestroy(&data->localBlock) );
  TRY( VecDestroy(&data->xloc) );
  TRY( VecDestroy(&data->yloc) );
  TRY( VecDestroy(&data->xloc1) );
  TRY( VecDestroy(&data->yloc1) );
  TRY( PetscFree(data) );
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
  
  TRY( MatCreate(((PetscObject)matin)->comm,&matout) );
  TRY( MatSetSizes(matout,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N) );
  TRY( MatSetType(matout,((PetscObject)matin)->type_name) );
  TRY( PetscMemcpy(matout->ops,matin->ops,sizeof(struct _MatOps)) );
  dataout = (Mat_BlockDiag*) matout->data;
  
  TRY( MatDuplicate(datain->localBlock,cpvalues,&dataout->localBlock) );
  TRY( VecDuplicate(datain->yloc,&dataout->yloc) );
  TRY( VecDuplicate(datain->yloc1,&dataout->yloc1) );
  TRY( VecDuplicate(datain->xloc,&dataout->xloc) );
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
  TRY( MatGetInfo(A,MAT_LOCAL,info) );
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
      TRY( MPI_Allreduce(isend, irecv, 5, MPIU_REAL, MPIU_MAX, ((PetscObject) matin)->comm) );
      info->nz_used      = irecv[0];
      info->nz_allocated = irecv[1];
      info->nz_unneeded  = irecv[2];
      info->memory       = irecv[3];
      info->mallocs      = irecv[4];
      break;
    case MAT_GLOBAL_SUM:
      TRY( MPI_Allreduce(isend, irecv, 5, MPIU_REAL, MPIU_SUM, ((PetscObject) matin)->comm) );
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
  TRY( MatSetOption(bd->localBlock, op, flg) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_BlockDiag"
PetscErrorCode MatGetDiagonal_BlockDiag(Mat mat,Vec d)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  TRY( MatGetDiagonal(bd->localBlock, bd->yloc) );
  TRY( VecCopy(bd->yloc, d) );
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
  TRY( PetscObjectGetComm((PetscObject)mat,&comm) );
  TRY( PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii) );
  if (!iascii) FLLOP_SETERRQ1(comm,PETSC_ERR_SUP,"Viewer type %s not supported for matrix type "MATBLOCKDIAG, ((PetscObject)viewer)->type);
  TRY( PetscViewerGetFormat(viewer,&format) );

  if (format == PETSC_VIEWER_DEFAULT) {
    TRY( PetscObjectPrintClassNamePrefixType((PetscObject)mat,viewer) );
    TRY( PetscViewerASCIIPushTab(viewer) );
  }

  TRY( PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sv) );
  if (format != PETSC_VIEWER_DEFAULT) {
    PetscMPIInt rank;
    TRY( MPI_Comm_rank(comm, &rank) );
    TRY( PetscViewerASCIIPrintf(viewer,"diagonal block on rank 0:\n") );
    if (!rank) TRY( MatView(bd->localBlock,sv) );
  } else {
    TRY( PetscSequentialPhaseBegin(comm,1) );
    TRY( PetscViewerASCIIPrintf(viewer,"diagonal blocks:\n") );
    TRY( MatView(bd->localBlock,sv) );
    TRY( PetscSequentialPhaseEnd(comm,1) );
  }
  TRY( PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sv) );

  if (format == PETSC_VIEWER_DEFAULT) {
    TRY( PetscViewerASCIIPopTab(viewer) );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_BlockDiag"
PetscErrorCode MatAssemblyBegin_BlockDiag(Mat mat, MatAssemblyType type)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  TRY( MatAssemblyBegin(bd->localBlock, type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_BlockDiag"
PetscErrorCode MatAssemblyEnd_BlockDiag(Mat mat, MatAssemblyType type)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  TRY( MatAssemblyEnd(bd->localBlock, type) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetLocalToGlobalMapping_BlockDiag"
PetscErrorCode MatSetLocalToGlobalMapping_BlockDiag(Mat x,ISLocalToGlobalMapping rmapping,ISLocalToGlobalMapping cmapping)
{
  PetscFunctionBegin;
  FLLOP_SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"custom LocalToGlobalMapping not allowed for matrix of type "MATBLOCKDIAG);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesLocal_BlockDiag"
PetscErrorCode MatSetValuesLocal_BlockDiag(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;
  
  PetscFunctionBegin;
  TRY( MatSetValues(data->localBlock,nrow,irow,ncol,icol,y,addv) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_BlockDiag"
PetscErrorCode MatScale_BlockDiag(Mat mat,PetscScalar a)
{
  Mat_BlockDiag *data = (Mat_BlockDiag*) mat->data;

  PetscFunctionBegin;
  TRY( MatScale(data->localBlock,a) );
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
  TRY( MatGetColumnVectors(data->localBlock,&n1,&data->cols_loc) );
  FLLOP_ASSERT(n==n1,"n==n1");

  TRY( MatCreateVecs(mat, PETSC_IGNORE, &d) );
  TRY( VecDuplicateVecs(d, N, &cols) );
  TRY( VecDestroy(&d) );

  for (j=0; j<N; j++) {
    TRY( VecSet(cols[j],0.0) );
  }

  for (j=0; j<n; j++) {
    TRY( VecGetArray(data->cols_loc[j],&arr) );
    TRY( VecPlaceArray(cols[j+jlo],arr) );
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
    TRY( VecRestoreArray(data->cols_loc[j],NULL) );
    TRY( VecResetArray((*cols)[j+jlo]) );
  }

  TRY( MatRestoreColumnVectors(data->localBlock,NULL,&data->cols_loc) );
  TRY( VecDestroyVecs(N,cols) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatOrthColumns_BlockDiag"
static PetscErrorCode MatOrthColumns_BlockDiag(Mat A, MatOrthType type, MatOrthForm form, Mat *Q_new, Mat *S_new)
{
  Mat_BlockDiag  *bd = (Mat_BlockDiag*) A->data;
  Mat Q_loc=NULL,S_loc=NULL;

  PetscFunctionBegin;
  TRY( MatOrthColumns(bd->localBlock, type, form, Q_new?(&Q_loc):NULL, S_new?(&S_loc):NULL) );
  if (Q_new) TRY( MatCreateBlockDiag(PetscObjectComm((PetscObject)A),Q_loc,Q_new) );
  if (S_new) TRY( MatCreateBlockDiag(PetscObjectComm((PetscObject)A),S_loc,S_new) );
  TRY( MatDestroy(&Q_loc) );
  TRY( MatDestroy(&S_loc) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_BlockDiag"
FLLOP_EXTERN PetscErrorCode MatCreate_BlockDiag(Mat B) {
  Mat_BlockDiag *data;

  PetscFunctionBegin;
  TRY( PetscObjectChangeTypeName((PetscObject)B,MATBLOCKDIAG) );

  TRY( PetscNewLog(B,&data) );
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
  B->ops->matmult            = MatMatMult_BlockDiag_BlockDiag;
  B->ops->duplicate          = MatDuplicate_BlockDiag;
  B->ops->getinfo            = MatGetInfo_BlockDiag;
  B->ops->setoption          = MatSetOption_BlockDiag;
  B->ops->getdiagonal        = MatGetDiagonal_BlockDiag;
  B->ops->view               = MatView_BlockDiag;
  B->ops->assemblybegin      = MatAssemblyBegin_BlockDiag;
  B->ops->assemblyend        = MatAssemblyEnd_BlockDiag;
  B->ops->setlocaltoglobalmapping = MatSetLocalToGlobalMapping_BlockDiag;
  B->ops->setvalueslocal     = MatSetValuesLocal_BlockDiag;
  B->ops->zeroentries        = MatZeroEntries_BlockDiag;
  B->ops->zerorows           = MatZeroRows_BlockDiag;
  B->ops->zerorowscolumns    = MatZeroRowsColumns_BlockDiag;
  B->ops->transposematmult   = MatTransposeMatMult_BlockDiag_BlockDiag;
  B->ops->scale              = MatScale_BlockDiag;
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatGetDiagonalBlock_C",MatGetDiagonalBlock_BlockDiag) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_BlockDiag) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_BlockDiag) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatMatMult_blockdiag_aij_C",MatMatMult_BlockDiag_AIJ) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMult_blockdiag_aij_C",MatTransposeMatMult_BlockDiag_AIJ) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatConvert_blockdiag_aij_C",MatConvert_BlockDiag_AIJ) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatOrthColumns_C",MatOrthColumns_BlockDiag) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"PermonMatConvertBlocks_C",PermonMatConvertBlocks_BlockDiag) );
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
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)block),&size) );
  if (size > 1) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_WRONG,"block (arg #2) must be sequential");
  
  /* Create matrix. */  
  TRY( MatCreate(comm, &B) );
  TRY( MatSetType(B, MATBLOCKDIAG) );
  data = (Mat_BlockDiag*) B->data;
  
  /* Matrix data. */
  if (!block) {
      TRY( MatCreateSeqDense(PETSC_COMM_SELF,0,0,NULL,&block) );
  } else {
      TRY( PetscObjectReference((PetscObject) block) );
  }
  data->localBlock = block;

  /* Set up row layout */
  TRY( PetscLayoutSetBlockSize(B->rmap,block->rmap->bs) );
  TRY( PetscLayoutSetLocalSize(B->rmap,block->rmap->n) );
  TRY( PetscLayoutSetUp(B->rmap) );
  TRY( PetscLayoutGetRange(B->rmap,&rlo,&rhi) );
  
  /* Set up column layout */
  TRY( PetscLayoutSetBlockSize(B->cmap,block->cmap->bs) );
  TRY( PetscLayoutSetLocalSize(B->cmap,block->cmap->n) );
  TRY( PetscLayoutSetUp(B->cmap) );
  TRY( PetscLayoutGetRange(B->cmap,&clo,&chi) );
  
  /* Intermediate vectors for MatMult. */
  TRY( MatCreateVecs(block, &data->xloc, &data->yloc) );
  TRY( VecDuplicate(data->yloc, &data->yloc1));
  TRY( VecDuplicate(data->xloc, &data->xloc1));

  {
    IS l2gris,l2gcis;
    ISLocalToGlobalMapping l2gr,l2gc;

    TRY( ISCreateStride(comm,B->rmap->n,rlo,1,&l2gris) );
    TRY( ISLocalToGlobalMappingCreateIS(l2gris,&l2gr) );
    TRY( PetscLayoutSetISLocalToGlobalMapping(B->rmap,l2gr) );
    TRY( ISDestroy(&l2gris) );
    TRY( ISLocalToGlobalMappingDestroy(&l2gr) );

    TRY( ISCreateStride(comm,B->cmap->n,clo,1,&l2gcis) );
    TRY( ISLocalToGlobalMappingCreateIS(l2gcis,&l2gc) );
    TRY( PetscLayoutSetISLocalToGlobalMapping(B->cmap,l2gc) );
    TRY( ISDestroy(&l2gcis) );
    TRY( ISLocalToGlobalMappingDestroy(&l2gc) );
  }
  
  TRY( MatInheritSymmetry(block,B) );
  *B_new = B;
  PetscFunctionReturn(0);
}
