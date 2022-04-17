
#include <permon/private/permonmatimpl.h>
#include <petscblaslapack.h>

typedef struct {
  Mat A;
  IS cis, ris, ris_local;
  Vec cwork, rwork;
  VecScatter cscatter, rscatter;
  PetscBool setupcalled, rows_use_global_numbering;
} Mat_Extension;

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetColumnIS_Extension"
static PetscErrorCode MatExtensionGetColumnIS_Extension(Mat TA,IS *cis)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  *cis = data->cis;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetColumnIS"
PetscErrorCode MatExtensionGetColumnIS(Mat TA,IS *cis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidPointer(cis,2);
  PetscUseMethod(TA,"MatExtensionGetColumnIS_Extension_C",(Mat,IS*),(TA,cis));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetColumnIS_Extension"
static PetscErrorCode MatExtensionSetColumnIS_Extension(Mat TA,IS cis)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  if (data->setupcalled) SETERRQ(PetscObjectComm((PetscObject)TA),PETSC_ERR_ARG_WRONGSTATE,"cannot alter inner data after first MatMult* call");
  data->cis = cis;
  CHKERRQ(PetscObjectReference((PetscObject)cis));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetColumnIS"
PetscErrorCode MatExtensionSetColumnIS(Mat TA,IS cis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  if (cis) PetscValidHeaderSpecific(cis,IS_CLASSID,2);
  PetscTryMethod(TA,"MatExtensionSetColumnIS_Extension_C",(Mat,IS),(TA,cis));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetRowIS_Extension"
static PetscErrorCode MatExtensionGetRowIS_Extension(Mat TA,IS *ris)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  *ris = data->ris;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetRowIS"
PetscErrorCode MatExtensionGetRowIS(Mat TA,IS *ris)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidPointer(ris,2);
  PetscUseMethod(TA,"MatExtensionGetRowIS_Extension_C",(Mat,IS*),(TA,ris));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetRowISLocal_Extension"
static PetscErrorCode MatExtensionGetRowISLocal_Extension(Mat TA,IS *ris)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  *ris = data->ris_local;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetRowISLocal"
PetscErrorCode MatExtensionGetRowISLocal(Mat TA,IS *ris)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidPointer(ris,2);
  PetscUseMethod(TA,"MatExtensionGetRowISLocal_Extension_C",(Mat,IS*),(TA,ris));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetRowIS_Extension"
static PetscErrorCode MatExtensionSetRowIS_Extension(Mat TA,IS ris,PetscBool rows_use_global_numbering)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  if (data->setupcalled) SETERRQ(PetscObjectComm((PetscObject)TA),PETSC_ERR_ARG_WRONGSTATE,"cannot alter inner data after first MatMult* call");
  if (rows_use_global_numbering) {
    data->ris = ris;
  } else {
    data->ris_local = ris;
  }
  CHKERRQ(PetscObjectReference((PetscObject)ris));
  data->rows_use_global_numbering = rows_use_global_numbering;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetRowIS"
PetscErrorCode MatExtensionSetRowIS(Mat TA,IS ris,PetscBool rows_use_global_numbering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  if (ris) PetscValidHeaderSpecific(ris,IS_CLASSID,2);
  PetscValidLogicalCollectiveBool(TA,rows_use_global_numbering,3);
  PetscTryMethod(TA,"MatExtensionSetRowIS_Extension_C",(Mat,IS,PetscBool),(TA,ris,rows_use_global_numbering));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetCondensed_Extension"
static PetscErrorCode MatExtensionGetCondensed_Extension(Mat TA,Mat *A)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  *A = data->A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionGetCondensed"
PetscErrorCode MatExtensionGetCondensed(Mat TA,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  PetscUseMethod(TA,"MatExtensionGetCondensed_Extension_C",(Mat,Mat*),(TA,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionCreateCondensedRows_Extension"
static PetscErrorCode MatExtensionCreateCondensedRows_Extension(Mat TA,Mat *A,IS *ris_local)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(MatCreateExtension(PetscObjectComm((PetscObject)TA),data->A->rmap->n,TA->cmap->n,PETSC_DECIDE,TA->cmap->N,data->A,NULL,PETSC_TRUE,data->cis,A));
  if (ris_local) CHKERRQ(ISDuplicate(data->ris_local,ris_local));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionCreateCondensedRows"
PetscErrorCode MatExtensionCreateCondensedRows(Mat TA,Mat *A,IS *ris_local)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidPointer(A,2);
  PetscUseMethod(TA,"MatExtensionCreateCondensedRows_Extension_C",(Mat,Mat*,IS*),(TA,A,ris_local));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionCreateLocalMat_Extension"
static PetscErrorCode MatExtensionCreateLocalMat_Extension(Mat TA,Mat *A)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;
  IS ris,cis;

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(ISOnComm(data->ris_local,PETSC_COMM_SELF,PETSC_COPY_VALUES,&ris));
  CHKERRQ(ISOnComm(data->cis,PETSC_COMM_SELF,PETSC_COPY_VALUES,&cis));
  CHKERRQ(MatCreateExtension(PETSC_COMM_SELF,TA->rmap->n,TA->cmap->N,TA->rmap->n,TA->cmap->N,data->A,ris,PETSC_TRUE,cis,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionCreateLocalMat"
PetscErrorCode MatExtensionCreateLocalMat(Mat TA,Mat *local)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidPointer(local,2);
  PetscUseMethod(TA,"MatExtensionCreateLocalMat_Extension_C",(Mat,Mat*),(TA,local));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetCondensed_Extension"
static PetscErrorCode MatExtensionSetCondensed_Extension(Mat TA,Mat A)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;
  PetscMPIInt commsize;

  PetscFunctionBegin;
  if (data->setupcalled) SETERRQ(PetscObjectComm((PetscObject)TA),PETSC_ERR_ARG_WRONGSTATE,"cannot alter inner data after first MatMult* call");
  CHKERRQ(MPI_Comm_size(PetscObjectComm((PetscObject)A),&commsize));
  if (commsize > 1) SETERRQ(PetscObjectComm((PetscObject)TA),PETSC_ERR_ARG_WRONG,"inner matrix must be sequential");
  data->A = A;
  CHKERRQ(PetscObjectReference((PetscObject)A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetCondensed"
PetscErrorCode MatExtensionSetCondensed(Mat TA,Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscTryMethod(TA,"MatExtensionSetCondensed_Extension_C",(Mat,Mat),(TA,A));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetUp_Extension"
static PetscErrorCode MatExtensionSetUp_Extension(Mat TA)
{
  Mat_Extension *data = (Mat_Extension*) TA->data;
  Vec c,r;
  PetscInt lo;

  PetscFunctionBegin;
  if (data->setupcalled) PetscFunctionReturn(0);
  CHKERRQ(PetscLayoutSetUp(TA->rmap));
  CHKERRQ(PetscLayoutSetUp(TA->cmap));
  lo = TA->rmap->rstart;

  if (!data->cis) {
    CHKERRQ(MatGetOwnershipIS(TA,NULL,&data->cis));
  }

  if (data->ris_local && !data->ris) {
    CHKERRQ(ISAdd(data->ris_local,lo,&data->ris));
  }

  if (!data->ris) {
    CHKERRQ(MatGetOwnershipIS(TA,&data->ris,NULL));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,TA->rmap->n,0,1,&data->ris_local));
  }

  if (!data->ris_local) {
    CHKERRQ(ISAdd(data->ris,-lo,&data->ris_local));
  }

  CHKERRQ(MatCreateVecs(data->A,&data->cwork,&data->rwork));
  CHKERRQ(MatCreateVecs(TA,&c,&r));
  CHKERRQ(VecScatterCreate(c,data->cis,data->cwork,NULL,&data->cscatter));
  CHKERRQ(VecScatterCreate(data->rwork,NULL,r,data->ris,&data->rscatter));
  CHKERRQ(VecDestroy(&c));
  CHKERRQ(VecDestroy(&r));
  data->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionSetUp"
PetscErrorCode MatExtensionSetUp(Mat TA)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(TA,MAT_CLASSID,1);
  PetscTryMethod(TA,"MatExtensionSetUp_Extension_C",(Mat),(TA));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_Extension"
PetscErrorCode MatConvertFrom_Extension(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  IS ris,cis,tempis;
  Mat Aloc;
  Mat As, Ast, Asts;
  Mat B;

  PetscFunctionBeginI;
  CHKERRQ(PermonMatGetLocalMat(A,&Aloc));

  CHKERRQ(MatFindNonzeroRows(Aloc,&ris));
  if (!ris) {
    CHKERRQ(MatGetOwnershipIS(Aloc,&ris,NULL));
  }
  CHKERRQ(MatCreateSubMatrix(Aloc,ris,NULL,MAT_INITIAL_MATRIX,&As));
  CHKERRQ(MatDestroy(&Aloc));

  CHKERRQ(ISAdd(ris,A->rmap->rstart,&tempis));
  CHKERRQ(ISDestroy(&ris));
  ris = tempis;

  CHKERRQ(MatTranspose(As,MAT_INITIAL_MATRIX,&Ast));
  CHKERRQ(MatDestroy(&As));

  CHKERRQ(MatFindNonzeroRows(Ast,&cis));
  if (!cis) {
    CHKERRQ(MatGetOwnershipIS(Ast,&cis,NULL));
  }
  CHKERRQ(MatCreateSubMatrix(Ast,cis,NULL,MAT_INITIAL_MATRIX,&Asts));
  CHKERRQ(MatDestroy(&Ast));

  CHKERRQ(MatTranspose(Asts,MAT_INITIAL_MATRIX,&As));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)As,(PetscObject)A,"_cond"));
  CHKERRQ(MatDestroy(&Asts));

  CHKERRQ(MatCreateExtension(PetscObjectComm((PetscObject)A),A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,As,ris,PETSC_TRUE,cis,&B));
  CHKERRQ(ISDestroy(&ris));
  CHKERRQ(ISDestroy(&cis));
  CHKERRQ(MatDestroy(&As));

  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    CHKERRQ(MatHeaderReplace(A,B));
#else
    CHKERRQ(MatHeaderReplace(A,&B));
#endif
  } else {
    *newmat = B;
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_NestPermon_Extension"
PETSC_EXTERN PetscErrorCode MatConvert_NestPermon_Extension(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat *mats_out,**mats_in;
  PetscInt i,j,Mn,Nn;
  Mat Aloc;
  Mat As, Ast, Asts;
  Mat B;
  IS risi,risij,cisij,tempis;
  IS *ris_arr, *cis_arr;
  IS ris,cis;
  IS *rows,*cols;
  ISLocalToGlobalMapping l2g;
  MPI_Comm comm;
  PetscMPIInt commsize;

  PetscFunctionBeginI;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MPI_Comm_size(comm,&commsize));

  CHKERRQ(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));

  CHKERRQ(PetscMalloc1(Mn,&ris_arr));
  CHKERRQ(PetscMalloc1(Nn,&cis_arr));
  CHKERRQ(PetscMalloc1(Mn*Nn,&mats_out));
  
  CHKERRQ(PetscMalloc1(Mn,&rows));
  CHKERRQ(MatNestGetISs(A,rows,NULL));
  CHKERRQ(MatNestPermonGetColumnISs(A,&cols));

  for (i=0; i<Mn; i++) {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,0,&risi));

    for (j=0; j<Nn; j++) {
      CHKERRQ(PermonMatGetLocalMat(mats_in[i][j],&Aloc));

      CHKERRQ(MatFindNonzeroRows(Aloc,&risij));
      if (!risij) {
        CHKERRQ(MatGetOwnershipIS(Aloc,&risij,NULL));
      }

      /* risi = union(risi,risij) */
      tempis = NULL;
      CHKERRQ(ISSum(risi,risij,&tempis));
      CHKERRQ(ISDestroy(&risij));
      if (tempis) {
        CHKERRQ(ISDestroy(&risi) );
        risi = tempis;
      }

      mats_out[i*Nn+j] = Aloc;
    }

    for (j=0; j<Nn; j++) {
      Aloc = mats_out[i*Nn+j];

      CHKERRQ(MatCreateSubMatrix(Aloc,risi,NULL,MAT_INITIAL_MATRIX,&As));
      CHKERRQ(MatDestroy(&Aloc));

      CHKERRQ(MatTranspose(As,MAT_INITIAL_MATRIX,&Ast));
      CHKERRQ(MatDestroy(&As));

      mats_out[i*Nn+j] = Ast;
    }

    CHKERRQ(ISLocalToGlobalMappingCreateIS(rows[i],&l2g));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(l2g,risi,&ris_arr[i]));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g));

    CHKERRQ(ISDestroy(&risi));
  }

  CHKERRQ(ISConcatenate(PETSC_COMM_SELF,Mn,ris_arr,&ris));
  for (i=0; i<Mn; i++) {
    CHKERRQ(ISDestroy(&ris_arr[i]));
  }
  CHKERRQ(PetscFree(ris_arr));

  for (j=0; j<Nn; j++) {
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,0,0,0,&cis_arr[j]));

    for (i=0; i<Mn; i++) {
      Ast = mats_out[i*Nn+j];

      CHKERRQ(MatFindNonzeroRows(Ast,&cisij));
      if (!cisij) {
        CHKERRQ(MatGetOwnershipIS(Ast,&cisij,NULL));
      }

      /* cis_arr[j] = union(cis_arr[j],cisij) */
      tempis = NULL;
      CHKERRQ(ISSum(cis_arr[j],cisij,&tempis));
      CHKERRQ(ISDestroy(&cisij));
      if (tempis) {
        CHKERRQ(ISDestroy(&cis_arr[j]) );
        cis_arr[j] = tempis;
      }
    }

    for (i=0; i<Mn; i++) {
      Ast = mats_out[i*Nn+j];

      CHKERRQ(MatCreateSubMatrix(Ast,cis_arr[j],NULL,MAT_INITIAL_MATRIX,&Asts));
      CHKERRQ(MatDestroy(&Ast));

      CHKERRQ(MatTranspose(Asts,MAT_INITIAL_MATRIX,&As));
      CHKERRQ(MatDestroy(&Asts));

      CHKERRQ(FllopPetscObjectInheritName((PetscObject)As,(PetscObject)mats_in[i][j],"_cond"));
      mats_out[i*Nn+j] = As;
    }

    CHKERRQ(ISLocalToGlobalMappingCreateIS(cols[j],&l2g));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(l2g,cis_arr[j],&tempis));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g));
    CHKERRQ(ISDestroy(&cis_arr[j]));
    cis_arr[j] = tempis;
  }

  CHKERRQ(ISConcatenate(PETSC_COMM_SELF,Nn,cis_arr,&cis));

  for (j=0; j<Nn; j++) {
    CHKERRQ(ISDestroy(&cis_arr[j]));
  }
  CHKERRQ(PetscFree(cis_arr));
  
  CHKERRQ(MatCreateNestPermon(PETSC_COMM_SELF,Mn,NULL,Nn,NULL,mats_out,&As));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)As,(PetscObject)A,"_cond"));
  CHKERRQ(MatCreateExtension(comm,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,As,ris,PETSC_TRUE,cis,&B));

  for (i=0; i<Mn; i++) {
    for (j=0; j<Nn; j++) {
      CHKERRQ(MatDestroy(&mats_out[i*Nn+j]));
    }
  }
  CHKERRQ(PetscFree(mats_out));

  CHKERRQ(ISDestroy(&ris));
  CHKERRQ(ISDestroy(&cis));
  CHKERRQ(MatDestroy(&As));
  CHKERRQ(PetscFree(rows));

  for (j=0; j<Nn; j++) {
    CHKERRQ(ISDestroy(&cols[j]));
  }
  CHKERRQ(PetscFree(cols));

  if (reuse == MAT_INPLACE_MATRIX) {
#if PETSC_VERSION_MINOR < 7
    CHKERRQ(MatHeaderReplace(A,B));
#else
    CHKERRQ(MatHeaderReplace(A,&B));
#endif
  } else {
    *newmat = B;
  }
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Extension"
PetscErrorCode MatMult_Extension(Mat TA, Vec c, Vec r) {
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(VecZeroEntries(r));
  CHKERRQ(VecScatterBegin(data->cscatter,c,data->cwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  data->cscatter,c,data->cwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(MatMult(data->A,data->cwork,data->rwork));
  CHKERRQ(VecScatterBegin(data->rscatter,data->rwork,r,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  data->rscatter,data->rwork,r,ADD_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_Extension"
PetscErrorCode MatMultAdd_Extension(Mat TA, Vec c, Vec r1, Vec r) {
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(VecCopy(r1,r));
  CHKERRQ(VecScatterBegin(data->cscatter,c,data->cwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  data->cscatter,c,data->cwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(MatMult(data->A,data->cwork,data->rwork));
  CHKERRQ(VecScatterBegin(data->rscatter,data->rwork,r,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  data->rscatter,data->rwork,r,ADD_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Extension"
PetscErrorCode MatMultTranspose_Extension(Mat TA, Vec r, Vec c) {
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(VecZeroEntries(c));
  CHKERRQ(VecScatterBegin(data->rscatter,r,data->rwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(  data->rscatter,r,data->rwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(data->A,data->rwork,data->cwork));
  CHKERRQ(VecScatterBegin(data->cscatter,data->cwork,c,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(  data->cscatter,data->cwork,c,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_Extension"
PetscErrorCode MatMultTransposeAdd_Extension(Mat TA, Vec r, Vec c1, Vec c) {
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(VecCopy(c1,c));
  CHKERRQ(VecScatterBegin(data->rscatter,r,data->rwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(  data->rscatter,r,data->rwork,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(data->A,data->rwork,data->cwork));
  CHKERRQ(VecScatterBegin(data->cscatter,data->cwork,c,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(  data->cscatter,data->cwork,c,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_Extension_2extension"
PetscErrorCode MatTransposeMatMult_BlockDiag_Extension_2extension(Mat B, Mat TA, MatReuse scall, PetscReal fill, Mat *C) {
  Mat_Extension *data = (Mat_Extension*) TA->data;
  Mat B_loc, C_loc, C_out;
  PetscInt i, n, rnnz,M_loc, N_loc, M, N;
  Vec *cols;
  Vec colExt;
  PetscScalar *row;
  const PetscInt *isCols;
  IS is_self, is_cols, is_rows;

  //B = R
  //TA = B^T

  PetscFunctionBegin;
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(MatGetDiagonalBlock(B, &B_loc));
  CHKERRQ(MatGetSize(B, NULL, &M));
  CHKERRQ(MatGetSize(TA, NULL, &N));
  CHKERRQ(MatGetLocalSize(B, NULL, &M_loc));
  CHKERRQ(MatGetLocalSize(TA, NULL, &N_loc));
  CHKERRQ(ISGetLocalSize(data->cis, &rnnz));
  CHKERRQ(ISOnComm(data->ris_local,PETSC_COMM_SELF,PETSC_USE_POINTER,&is_self));
  CHKERRQ(MatGetColumnVectors(B_loc,&n,&cols));
 
  CHKERRQ(MatCreateDensePermon(PETSC_COMM_SELF, M_loc, rnnz, M_loc, rnnz, NULL, &C_loc));
  
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, M_loc, 0, 1, &is_rows)); //all rows of B
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, rnnz, 0, 1, &is_cols)); //all cols of TA
  CHKERRQ(ISGetIndices(is_cols, &isCols));
  if (rnnz) {
    for (i = 0; i < n; i++){
      CHKERRQ(VecGetSubVector(cols[i], is_self, &colExt));
      CHKERRQ(MatMultTranspose(data->A, colExt, data->cwork));
      CHKERRQ(VecRestoreSubVector(cols[i], is_self, &colExt));
      CHKERRQ(VecGetArray(data->cwork, &row));
      CHKERRQ(MatSetValues(C_loc, 1, &i, rnnz, isCols, row, INSERT_VALUES));
      CHKERRQ(VecRestoreArray(data->cwork, &row));
    }
  }
  CHKERRQ(ISRestoreIndices(is_cols, &isCols));
  CHKERRQ(MatRestoreColumnVectors(B_loc,&n,&cols));
  CHKERRQ(MatAssemblyBegin(C_loc, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C_loc, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateExtension(PetscObjectComm((PetscObject)B),M_loc,N_loc,M,N,C_loc,is_rows,PETSC_FALSE,data->cis,&C_out));
  
  CHKERRQ(ISDestroy(&is_self));
  CHKERRQ(ISDestroy(&is_rows));
  CHKERRQ(ISDestroy(&is_cols));
  CHKERRQ(MatDestroy(&C_loc));
  *C = C_out;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_BlockDiag_Extension_2MPIAIJ"
PetscErrorCode MatTransposeMatMult_BlockDiag_Extension_2MPIAIJ(Mat B, Mat TA, MatReuse scall, PetscReal fill, Mat *C) {
  Mat_Extension *data = (Mat_Extension*) TA->data;
  Mat B_loc, C_out;
  PetscInt i, n, rnnz, rnnz_diag, M_loc, N_loc, M, N, ilo, rInd, nHigh;
  Vec *cols;
  Vec colExt;
  PetscScalar *row;
  const PetscInt *isCols;
  IS is_self;
  MPI_Comm comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)B, &comm));
  CHKERRQ(MatExtensionSetUp(TA));
  CHKERRQ(MatGetDiagonalBlock(B, &B_loc));
  CHKERRQ(MatGetSize(B, NULL, &M));
  CHKERRQ(MatGetSize(TA, NULL, &N));
  CHKERRQ(MatGetLocalSize(B, NULL, &M_loc));
  CHKERRQ(MatGetLocalSize(TA, NULL, &N_loc));
  CHKERRQ(ISGetLocalSize(data->cis, &rnnz));
  CHKERRQ(ISGetIndices(data->cis, &isCols));
  
  //compute number of nnz in DIAG part
  CHKERRQ(MPI_Scan(&N_loc, &nHigh, 1, MPIU_INT, MPI_SUM, comm));
  rnnz_diag = 0;
  for (i = 0; i < rnnz; i++) {
    if ( isCols[i] >= nHigh - N_loc && isCols[i] < nHigh ){
      rnnz_diag += 1;
    }
  }

  if ( scall == MAT_INITIAL_MATRIX){
    CHKERRQ(MatCreate(comm, &C_out ));
    CHKERRQ(MatSetSizes(C_out, M_loc, N_loc, M, N));
    CHKERRQ(MatSetFromOptions(C_out));
    CHKERRQ(MatSeqAIJSetPreallocation(C_out, rnnz_diag, NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(C_out, rnnz_diag, NULL, rnnz - rnnz_diag, NULL));
  } else{
    SETERRQ(comm,PETSC_ERR_ARG_WRONG,"scall must be MAT_INITIAL_MATRIX");
  }
  CHKERRQ(MatGetOwnershipRange(C_out, &ilo, NULL));

  CHKERRQ(ISOnComm(data->ris_local,PETSC_COMM_SELF,PETSC_USE_POINTER,&is_self));
    
  CHKERRQ(MatGetColumnVectors(B_loc,&n, &cols));
  for (i = 0; i < n; i++){
    CHKERRQ(VecGetSubVector(cols[i], is_self, &colExt));
    CHKERRQ(MatMultTranspose(data->A, colExt, data->cwork));
    CHKERRQ(VecRestoreSubVector(cols[i], is_self, &colExt));
    CHKERRQ(VecGetArray(data->cwork, &row));
    rInd = i + ilo;
    CHKERRQ(MatSetValues(C_out, 1, &rInd, rnnz, isCols, row, INSERT_VALUES));
    CHKERRQ(VecRestoreArray(data->cwork, &row));
  }
  CHKERRQ(MatRestoreColumnVectors(B_loc,&n, &cols));

  CHKERRQ(MatAssemblyBegin(C_out, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C_out, MAT_FINAL_ASSEMBLY));

  *C = C_out;
  CHKERRQ(ISRestoreIndices(data->cis, &isCols));
  CHKERRQ(ISDestroy(&is_self));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_BlockDiag_Extension"
static PetscErrorCode MatProductNumeric_BlockDiag_Extension(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  Mat            newmat;
  PetscBool      flg = PETSC_FALSE;

  switch (product->type) {
  case MATPRODUCT_AtB:
    PetscObjectOptionsBegin((PetscObject)C);
    CHKERRQ(PetscOptionsBool("-MatTrMatMult_2extension","MatTransposeMatMult_BlockDiag_Extension_2extension","Mat type of resulting matrix will be extension",flg,&flg,NULL));
    PetscOptionsEnd();
    if (flg){
      CHKERRQ(MatTransposeMatMult_BlockDiag_Extension_2extension(A, B, MAT_INITIAL_MATRIX, product->fill, &newmat));
    }else{
      CHKERRQ(MatTransposeMatMult_BlockDiag_Extension_2MPIAIJ(A, B, MAT_INITIAL_MATRIX, product->fill, &newmat));
    }
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MATPRODUCT type is not supported");
  }
  C->product = NULL;
  CHKERRQ(MatHeaderReplace(C,&newmat));
  C->product = product;
  C->ops->productnumeric = MatProductNumeric_BlockDiag_Extension;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_BlockDiag_Extension"
static PetscErrorCode MatProductSymbolic_BlockDiag_Extension(Mat C) {

  PetscFunctionBegin;
  C->ops->productnumeric  = MatProductNumeric_BlockDiag_Extension;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_BlockDiag_Extension"
static PetscErrorCode MatProductSetFromOptions_BlockDiag_Extension(Mat C) {

  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_BlockDiag_Extension;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMult_Extension_Extension_same"
PetscErrorCode MatMatTransposeMult_Extension_Extension_same(Mat A, Mat B, MatReuse scall, PetscReal fill, PetscInt mattype, Mat *C) {
  Mat_Extension *dataA = (Mat_Extension*)A->data;
  Mat A_loc,C_loc,C_out;
  Mat *submats,*submatsLoc;
  PetscBool isSym=PETSC_FALSE;
  PetscInt i,j,k,l,m,nnz,M_loc,M;
  PetscInt *nnzNeighbors,*nElem,*nElemR;
  PetscInt nnzAllLtNeighbors = 0; /* number of indices from all lower neighbors */
  PetscInt nnzElem = 0;
  PetscInt allNeighbors = 1; /* number of neighbors (inc self) */
  PetscInt neighborsGt = 0; /* number of neighbors with higher mpiRank */
  PetscInt neighborsLt = 0; /* number of neighbors with lower mpiRank */
  PetscMPIInt mpiRank;
  PetscInt *iNeighbor,*iNeighborRemote,*iNeighborInter,*iElem,*iRow,*iCol,*iIntersection,*iLocSort,*iRemSort;
  const PetscInt *iLocCol,*iNeighbors;
  PetscBLASInt bm,bk;
  PetscScalar _DOne = 1.0,_DZero = 0.0;
  PetscScalar **data,*dataElem,*arr,**dataLoc;
  IS isrow,myneighbors;
  IS *iscol,*iscolLoc;
  MPI_Request *mpiRequests,*mpiRequests2;
  MPI_Comm comm;

  PetscFunctionBeginI;
  CHKERRQ(PetscObjectGetComm((PetscObject)B,&comm));
  CHKERRQ(MPI_Comm_rank(comm,&mpiRank));
  CHKERRQ(MatExtensionSetUp(A));//TODO remove?
  CHKERRQ(MatGetSize(A,&M,NULL));
  CHKERRQ(MatGetLocalSize(A,&M_loc,NULL));
  CHKERRQ(ISGetLocalSize(dataA->cis,&nnz));
  CHKERRQ(ISGetIndices(dataA->cis,&iLocCol));
  
  CHKERRQ(PetscObjectQuery((PetscObject)A,"myneighbors",(PetscObject*)&myneighbors));
  if (myneighbors) {
    CHKERRQ(ISGetLocalSize(myneighbors,&allNeighbors));
    CHKERRQ(ISGetIndices(myneighbors,&iNeighbors)); /* ranks of all neighbors */
    for (i = 0; iNeighbors[i] < mpiRank && i < mpiRank; i++);
    neighborsGt = allNeighbors-i-1;
    neighborsLt = i;
  }

  if ( scall == MAT_INITIAL_MATRIX) {
#if defined(PETSC_USE_DEBUG)
    int M_max;
    CHKERRQ(MPI_Allreduce(&M_loc,&M_max,1,MPIU_INT,MPI_MAX,comm));
    if (M_loc != M_max) {
      SETERRQ(comm,PETSC_ERR_ARG_SIZ,"implemented only for matrices with same local row dimension");
    }
#endif
    if (!mattype) {
      CHKERRQ(MatCreate(comm,&C_out));
      CHKERRQ(MatSetSizes(C_out,M_loc,M_loc,M,M));
      CHKERRQ(MatSetFromOptions(C_out));
      CHKERRQ(MatSeqAIJSetPreallocation(C_out,M_loc,NULL));
      CHKERRQ(MatMPIAIJSetPreallocation(C_out,M_loc,NULL,M_loc*(allNeighbors-1),NULL));
    } else if(mattype == 1) {
        CHKERRQ(MatCreateBAIJ(comm,M_loc,M_loc,M_loc,M,M,M_loc,NULL,M_loc*(allNeighbors-1),NULL,&C_out));
    } else {
      CHKERRQ(MatCreateSBAIJ(comm,M_loc,M_loc,M_loc,M,M,M_loc,NULL,M_loc*neighborsGt,NULL,&C_out));
      isSym = PETSC_TRUE;
    }
  } else {
    SETERRQ(comm,PETSC_ERR_ARG_WRONG,"scall must be MAT_INITIAL_MATRIX");
  }
  
  /* get number of indices to recv from neigbours */
  CHKERRQ(PetscMalloc1(allNeighbors-1,&mpiRequests));
  CHKERRQ(PetscMalloc1(neighborsLt,&nnzNeighbors));
  CHKERRQ(PetscMalloc1(neighborsLt,&nElemR));
  for (i = 0; i < neighborsLt; i++) {
    CHKERRQ(MPI_Irecv(&nnzNeighbors[i],1,MPIU_INT,iNeighbors[i],0,comm,&mpiRequests[i]));
  }
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(MPI_Isend(&nnz,1,MPIU_INT,iNeighbors[i],0,comm,&mpiRequests[i-1]));
  }
  CHKERRQ(MPI_Waitall(allNeighbors-1,mpiRequests,MPI_STATUSES_IGNORE));

  /* recv indices from Lt neighbors */
  for (i = 0; i < neighborsLt; i++) {
    nnzAllLtNeighbors += nnzNeighbors[i];
  }
  CHKERRQ(PetscMalloc1(nnzAllLtNeighbors, &iNeighbor));
  for (i = 0; i < neighborsLt; i++) {
    CHKERRQ(MPI_Irecv(iNeighbor,nnzNeighbors[i],MPIU_INT,iNeighbors[i],1,comm,&mpiRequests[i])); 
    iNeighbor += nnzNeighbors[i];
  }
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(MPI_Isend((PetscInt*)iLocCol,nnz,MPIU_INT,iNeighbors[i],1,comm,&mpiRequests[i-1]));
  }
  CHKERRQ(MPI_Waitall(allNeighbors-1,mpiRequests,MPI_STATUSES_IGNORE));
  iNeighbor -= nnzAllLtNeighbors;

  /* compute indices intersection and send number of elements */
  CHKERRQ(PetscMalloc1(neighborsLt,&iscol));
  CHKERRQ(PetscMalloc1(nnzAllLtNeighbors, &iNeighborRemote));
  CHKERRQ(PetscMalloc1(nnz,&iIntersection));
  CHKERRQ(PetscMalloc1(nnz,&iLocSort));
  CHKERRQ(PetscMemcpy(iLocSort,iLocCol,nnz*sizeof(PetscInt)));
  CHKERRQ(PetscSortInt(nnz,iLocSort));
  for (i = 0; i < neighborsLt; i++) {

    CHKERRQ(PetscMalloc1(nnzNeighbors[i],&iRemSort));
    CHKERRQ(PetscMemcpy(iRemSort,iNeighbor,nnzNeighbors[i]*sizeof(PetscInt)));
    CHKERRQ(PetscSortInt(nnzNeighbors[i],iRemSort));
    CHKERRQ(PetscMalloc1(nnzNeighbors[i],&iNeighborInter));
    j = 0; 
    k = 0;
    l = 0;
    while(j < nnzNeighbors[i] && k < nnz) {
        if (iRemSort[j] < iLocSort[k]) {
          j += 1;
        } else if (iRemSort[j] > iLocSort[k]) {
          k += 1;
        } else {
          iIntersection[l] = iRemSort[j];
          j += 1;
          k += 1;
          l += 1;
      }
    }
    nElemR[i] = l;
    l = 0;

    for (j = 0; j < nElemR[i]; j++) {
      k = 0;
      while (iIntersection[j] != iNeighbor[k]) {
        k += 1;
      }
      iNeighborRemote[l] = k;
      k = 0;
      while (iIntersection[j] != iLocCol[k]) {
        k += 1;
      }
      iNeighborInter[l] = k;
      l += 1;
    }
    iNeighbor += nnzNeighbors[i];
    iNeighborRemote += nnzNeighbors[i];
    CHKERRQ(MPI_Isend(&nElemR[i],1,MPIU_INT,iNeighbors[i],2,comm,&mpiRequests[i]));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nElemR[i],iNeighborInter,PETSC_COPY_VALUES,&iscol[i]));
    CHKERRQ(PetscFree(iNeighborInter));
    CHKERRQ(PetscFree(iRemSort));
  }
  iNeighborRemote -= nnzAllLtNeighbors;
  CHKERRQ(PetscFree(iIntersection));
  CHKERRQ(PetscFree(iLocSort));
  CHKERRQ(ISRestoreIndices(dataA->cis,&iLocCol));

  /* recv number of elements */
  CHKERRQ(PetscMalloc1(neighborsGt,&nElem));
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(MPI_Irecv(&nElem[j],1,MPIU_INT,iNeighbors[i],2,comm,&mpiRequests[i-1]));
    j += 1;
  }

  iNeighbor -= nnzAllLtNeighbors;
  CHKERRQ(PetscFree(iNeighbor));
  
  /* get submats */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,M_loc,0,1,&isrow));
  CHKERRQ(PetscMalloc1(neighborsLt,&submats));
  CHKERRQ(PetscMalloc1(neighborsLt,&data));
  for (i=0; i < neighborsLt; i++) {
    CHKERRQ(MatCreateSubMatrix(dataA->A,isrow,iscol[i],MAT_INITIAL_MATRIX,&submats[i]));
    CHKERRQ(MatDenseGetArray(submats[i], &data[i]));
    CHKERRQ(ISDestroy(&iscol[i]));
  }
  CHKERRQ(PetscFree(iscol));
  CHKERRQ(MPI_Waitall(allNeighbors-1,mpiRequests,MPI_STATUSES_IGNORE)); /* wait for number of elements/indices */
  
  /* send remote ind */
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    nnzElem += nElem[j];
    j += 1;
  }
  CHKERRQ(PetscMalloc1(nnzElem, &iElem));
  for (i = 0; i < neighborsLt; i++){
    CHKERRQ(MPI_Isend(iNeighborRemote,nElemR[i],MPIU_INT,iNeighbors[i],3,comm,&mpiRequests[i]));
    iNeighborRemote += nnzNeighbors[i];
  }
  iNeighborRemote -= nnzAllLtNeighbors;
  CHKERRQ(PetscFree(nnzNeighbors));
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(MPI_Irecv(iElem,nElem[j],MPIU_INT,iNeighbors[i],3,comm,&mpiRequests[i-1])); 
    iElem += nElem[j];
    j += 1;
  }
  iElem -= nnzElem;

  /* send data */
  CHKERRQ(PetscMalloc1(allNeighbors-1,&mpiRequests2));
  CHKERRQ(PetscMalloc1(M_loc*nnzElem, &dataElem));
  for (i = 0; i < neighborsLt; i++) {
    CHKERRQ(MPI_Isend(data[i],M_loc*nElemR[i],MPIU_SCALAR,iNeighbors[i],4,comm,&mpiRequests2[i]));
  }
  CHKERRQ(PetscFree(nElemR));
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(MPI_Irecv(dataElem,M_loc*nElem[j],MPIU_SCALAR,iNeighbors[i],4,comm,&mpiRequests2[i-1])); 
    dataElem += M_loc*nElem[j];
    j += 1;
  }
  dataElem -= M_loc*nnzElem;
  

  /* compute local product */
  CHKERRQ(PetscMalloc1(M_loc,&iRow));
  CHKERRQ(PetscMalloc1(M_loc,&iCol));
  for (i = 0; i<M_loc; i++){
    iRow[i] = mpiRank*M_loc +i;
    iCol[i] = mpiRank*M_loc +i;
  }
  if (nnz) {
    /* PETSC BUG (MatMatTransposeMult) workaround
    CHKERRQ(MatMatTransposeMult(dataA->A,dataA->A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C_loc));
    CHKERRQ(MatDenseGetArray(C_loc,&arr));
    CHKERRQ(MatSetValuesBlockedLocal(C_out,M_loc,iRow,M_loc,iCol,arr,INSERT_VALUES));
    CHKERRQ(MatDenseRestoreArray(C_loc,&arr));
    CHKERRQ(MatDestroy(&C_loc));*/

    CHKERRQ(PermonMatTranspose(dataA->A,MAT_TRANSPOSE_CHEAPEST,&A_loc));
    CHKERRQ(PermonMatMatMult(dataA->A,A_loc,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C_loc));
    CHKERRQ(MatDenseGetArray(C_loc,&arr));
    /* PETSC BUG
    CHKERRQ(MatSetValuesBlocked(C_out,M_loc,iRow,M_loc,iCol,arr,INSERT_VALUES)); */
    CHKERRQ(MatSetValues(C_out,M_loc,iRow,M_loc,iCol,arr,INSERT_VALUES));
    CHKERRQ(MatDenseRestoreArray(C_loc,&arr));
    CHKERRQ(MatDestroy(&C_loc));
    CHKERRQ(MatDestroy(&A_loc));
  }

  /* compute off-diag products */
  CHKERRQ(PetscMalloc1(neighborsGt,&iscolLoc));
  CHKERRQ(PetscMalloc1(neighborsGt,&submatsLoc));
  CHKERRQ(PetscMalloc1(neighborsGt,&dataLoc));
  CHKERRQ(MPI_Waitall(allNeighbors-1,mpiRequests,MPI_STATUSES_IGNORE)); /* wait for indices */
  CHKERRQ(PetscFree(iNeighborRemote));
  CHKERRQ(PetscFree(mpiRequests));
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nElem[j],iElem,PETSC_USE_POINTER,&iscolLoc[j]));
    iElem += nElem[j];
    CHKERRQ(MatCreateSubMatrix(dataA->A,isrow,iscolLoc[j],MAT_INITIAL_MATRIX,&submatsLoc[j]));
    CHKERRQ(MatDenseGetArray(submatsLoc[j],&dataLoc[j]));
    CHKERRQ(ISDestroy(&iscolLoc[j]));
    j += 1;
  }
  iElem -= nnzElem;
  CHKERRQ(PetscFree(iElem));
  CHKERRQ(PetscFree(iscolLoc));
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(PetscMalloc1(M_loc*M_loc,&arr));
  CHKERRQ(MPI_Waitall(allNeighbors-1,mpiRequests2,MPI_STATUSES_IGNORE)); /* wait for data */

  CHKERRQ(PetscFree(mpiRequests2));
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    for (k = 0; k<M_loc; k++){
      iCol[k] = iNeighbors[i]*M_loc +k;
    }
    CHKERRQ(PetscBLASIntCast(M_loc,&bm));
    CHKERRQ(PetscBLASIntCast(nElem[j],&bk));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&bm,&bm,&bk,&_DOne,dataLoc[j],&bm,dataElem,&bm,&_DZero,arr,&bm));
    dataElem += nElem[j]*M_loc;
    if (!isSym){
      CHKERRQ(MatSetValues(C_out,M_loc,iCol,M_loc,iRow,arr,INSERT_VALUES));
    }
    PetscScalar val;
    for (l=0;l<M_loc;l++) {
      for (m=0;m<l;m++){
        val = arr[l*M_loc+m];
        arr[l*M_loc+m] = arr[m*M_loc+l];
        arr[m*M_loc+l] = val;
      }
    }
    CHKERRQ(MatSetValues(C_out,M_loc,iRow,M_loc,iCol,arr,INSERT_VALUES));
    /* PETSC BUG
    CHKERRQ(MatSetValuesBlocked(C_out, M_loc, iRow, M_loc, &i, arr, INSERT_VALUES)); */
    j += 1;
  }
  dataElem -= nnzElem*M_loc;

  CHKERRQ(PetscFree(dataElem));
  CHKERRQ(PetscFree(arr));
  CHKERRQ(PetscFree(iRow));
  CHKERRQ(PetscFree(iCol));
  if(myneighbors){
    CHKERRQ(ISRestoreIndices(myneighbors,&iNeighbors));
  }
  j = 0;
  for (i = neighborsLt+1; i <allNeighbors; i++) {
    CHKERRQ(MatDenseRestoreArray(submatsLoc[j],&dataLoc[j]));
    CHKERRQ(MatDestroy(&submatsLoc[j]));
    j += 1;
  }
  CHKERRQ(PetscFree(submatsLoc));
  CHKERRQ(PetscFree(dataLoc));
  for (i=0; i < neighborsLt; i++) {
    CHKERRQ(MatDenseRestoreArray(submats[i],&data[i]));
    CHKERRQ(MatDestroy(&submats[i]));
  }
  CHKERRQ(PetscFree(submats));
  CHKERRQ(PetscFree(data));
  CHKERRQ(PetscFree(nElem));

  CHKERRQ(MatAssemblyBegin(C_out,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C_out,MAT_FINAL_ASSEMBLY));
  *C = C_out;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_Extension"
static PetscErrorCode MatProductNumeric_Extension(Mat C)
{
  Mat_Product *product = C->product;
  Mat         A=product->A,B=product->B;
  Mat         newmat;
  PetscInt    mattype = 0; /* make aij default type */
  const char  *allowedMats[3] = {"aij","baij","sbaij"};

  /* TODO add general mult, resulting mat MPIAIJ || extension */
  switch (product->type) {
  case MATPRODUCT_ABt:
    if (A != B) {
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"implemented only for A=B");
    }
    PetscObjectOptionsBegin((PetscObject)C);
    CHKERRQ(PetscOptionsEList("-MatMatMultExt_mattype","MatMatMultExt_mattype","Set type of resulting matrix when assembling from extension type",allowedMats,3,MATAIJ,&mattype,NULL));
    PetscOptionsEnd();
    CHKERRQ(MatMatTransposeMult_Extension_Extension_same(A,B,MAT_INITIAL_MATRIX,product->fill,mattype,&newmat));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MATPRODUCT type is not supported");
  }
  C->product = NULL;
  CHKERRQ(MatHeaderReplace(C,&newmat));
  C->product = product;
  C->ops->productnumeric = MatProductNumeric_Extension;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_Extension"
PetscErrorCode MatProductSymbolic_Extension(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric  = MatProductNumeric_Extension;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_Extension"
static PetscErrorCode MatProductSetFromOptions_Extension(Mat C) {
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_Extension;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Extension"
PetscErrorCode MatDestroy_Extension(Mat TA) {
  Mat_Extension *data = (Mat_Extension*) TA->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&data->A));
  CHKERRQ(ISDestroy(&data->cis));
  CHKERRQ(ISDestroy(&data->ris));
  CHKERRQ(ISDestroy(&data->ris_local));
  CHKERRQ(VecDestroy(&data->cwork));
  CHKERRQ(VecDestroy(&data->rwork));
  CHKERRQ(VecScatterDestroy(&data->cscatter));
  CHKERRQ(VecScatterDestroy(&data->rscatter));
  CHKERRQ(PetscFree(data));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Extension"
FLLOP_EXTERN PetscErrorCode MatCreate_Extension(Mat TA)
{
  Mat_Extension *data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)TA,MATEXTENSION));

  CHKERRQ(PetscNewLog(TA,&data));

  /* initialize general inner data */
  TA->data                    = (void*) data;
  TA->assembled               = PETSC_TRUE;
  TA->preallocated            = PETSC_TRUE;

  /* initialize type-specific inner data */
  data->A                     = NULL;
  data->cis                   = NULL;
  data->ris                   = NULL;
  data->ris_local             = NULL;
  data->rows_use_global_numbering = PETSC_TRUE;
  data->cwork                 = NULL;
  data->rwork                 = NULL;
  data->cscatter              = NULL;
  data->rscatter              = NULL;
  data->setupcalled           = PETSC_FALSE;

  /* set type-specific implementations of general Mat methods */
  TA->ops->destroy            = MatDestroy_Extension;
  TA->ops->mult               = MatMult_Extension;
  TA->ops->multtranspose      = MatMultTranspose_Extension;
  TA->ops->multadd            = MatMultAdd_Extension;
  TA->ops->multtransposeadd   = MatMultTransposeAdd_Extension;
  TA->ops->convertfrom        = MatConvertFrom_Extension;
  TA->ops->productsetfromoptions = MatProductSetFromOptions_Extension;

  /* set type-specific methods */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatConvert_nestpermon_extension_C", MatConvert_NestPermon_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionCreateCondensedRows_Extension_C",MatExtensionCreateCondensedRows_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionCreateLocalMat_Extension_C",MatExtensionCreateLocalMat_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionGetColumnIS_Extension_C",MatExtensionGetColumnIS_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionSetColumnIS_Extension_C",MatExtensionSetColumnIS_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionGetRowIS_Extension_C",MatExtensionGetRowIS_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionGetRowISLocal_Extension_C",MatExtensionGetRowISLocal_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionSetRowIS_Extension_C",MatExtensionSetRowIS_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionGetCondensed_Extension_C",MatExtensionGetCondensed_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionSetCondensed_Extension_C",MatExtensionSetCondensed_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatExtensionSetUp_Extension_C",MatExtensionSetUp_Extension));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)TA,"MatProductSetFromOptions_blockdiag_extension_C",MatProductSetFromOptions_BlockDiag_Extension));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateExtension"
PetscErrorCode MatCreateExtension(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, Mat A, IS ris, PetscBool rows_use_global_numbering, IS cis, Mat *TA_new) {
  Mat TA;
  PetscInt rnnz,cnnz;
  Mat A_empty;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (ris) PetscValidHeaderSpecific(ris,IS_CLASSID,4);
  if (cis) PetscValidHeaderSpecific(cis,IS_CLASSID,6);
  PetscValidPointer(TA_new,7);
  CHKERRQ(MatCreate(comm,&TA));
  CHKERRQ(MatSetType(TA,MATEXTENSION));
  CHKERRQ(MatSetSizes(TA,m,n,M,N));

  /* (Permon) seqdense empty matrix MatMult workaround */
  CHKERRQ(MatGetSize(A,&cnnz,&rnnz));
  if (!rnnz || !cnnz) {
    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF, cnnz, rnnz, 0, NULL, &A_empty));
    CHKERRQ(MatAssemblyBegin(A_empty, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A_empty, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatExtensionSetCondensed(TA,A_empty));
    CHKERRQ(MatDestroy(&A_empty));
  } else {
    CHKERRQ(MatExtensionSetCondensed(TA,A));
  }

  CHKERRQ(MatExtensionSetRowIS(TA,ris,rows_use_global_numbering));
  CHKERRQ(MatExtensionSetColumnIS(TA,cis));
  CHKERRQ(MatExtensionSetUp(TA));
  *TA_new = TA;
  PetscFunctionReturn(0);
}

