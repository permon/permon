
#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVectors_NestPermon"
static PetscErrorCode MatGetColumnVectors_NestPermon(Mat A, Vec *cols_new[])
{
  MPI_Comm comm;
  PetscInt II,J,j,N,N1,N1_last,Mn,Nn;
  Mat **mats, mat;
  Vec *cols,*col_arr;
  Vec **cols_for_each_row_block;
  PetscContainer container;
  IS *is_glob_col,is_seq;
  const PetscInt *is_arr;

  PetscFunctionBegin;
  N = A->cmap->N;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));

  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats));
  PetscCall(PetscMalloc(sizeof(Vec*)*Mn,&cols_for_each_row_block));
  PetscCall(PetscMalloc(sizeof(Vec)*N,&cols););
  PetscCall(PetscMalloc(sizeof(Vec)*Mn,&col_arr));
  PetscCall(PetscMalloc(sizeof(IS)*Nn,&is_glob_col));
  PetscCall(MatNestGetISs(A,NULL,is_glob_col));

  for (J=0; J<Nn; J++) {
    N1_last = mats[0][J]->cmap->N;
    for (II=0; II<Mn; II++) {
      mat = mats[II][J];
      if (!mat) SETERRQ(comm,PETSC_ERR_SUP,"block (%d, %d) is null but null blocks not currently supported",II,J);
      PetscCall(MatGetColumnVectors(mat,&N1,&cols_for_each_row_block[II]));
      if (N1 != N1_last) SETERRQ(comm,PETSC_ERR_ARG_SIZ,"block (%d, %d) has different number of columns than block (%d, %d)",II,J,II-1,J);
      N1_last = N1;

      PetscCall(PetscContainerCreate(comm, &container));
      PetscCall(PetscContainerSetPointer(container,cols_for_each_row_block[II]));
      PetscCall(PetscObjectCompose((PetscObject)mat,"MatGetColumnVectors_Nest_cols",(PetscObject)container));
      PetscCall(PetscContainerDestroy(&container));
    }

    /* IS is_glob_col[J] holds indices of columns of the J-th column block in the global numbering of global matrix A */
    PetscCall(ISAllGather(is_glob_col[J],&is_seq));
    PetscCall(ISGetIndices(is_seq,&is_arr));
    
    for (j=0; j<N1; j++) {
      for (II=0; II<Mn; II++) { col_arr[II] = cols_for_each_row_block[II][j]; }
      PetscCall(VecCreateNest(comm,Mn,NULL,col_arr,&cols[is_arr[j]]));
    }
    PetscCall(ISRestoreIndices(is_seq,&is_arr));
    PetscCall(ISDestroy(&is_seq));
  }

  PetscCall(PetscFree(is_glob_col));
  PetscCall(PetscFree(col_arr));
  PetscCall(PetscFree(cols_for_each_row_block));
  *cols_new=cols;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreColumnVectors_NestPermon"
static PetscErrorCode MatRestoreColumnVectors_NestPermon(Mat A, Vec *cols[])
{
  PetscInt II,J,N,N1,Mn,Nn;
  Mat **mats, mat;
  Vec *block_cols;
  PetscContainer container;

  PetscFunctionBegin;
  N = A->cmap->N;
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats));

  for (II=0; II<Mn; II++) for (J=0; J<Nn; J++) {
    mat = mats[II][J];
    PetscCall(PetscObjectQuery((PetscObject)mat,"MatGetColumnVectors_Nest_cols",(PetscObject*)&container));
    PetscCall(PetscContainerGetPointer(container,(void**)&block_cols));
    PetscCall(MatRestoreColumnVectors(mat,&N1,&block_cols));
    PetscCall(PetscObjectCompose((PetscObject)mat,"MatGetColumnVectors_Nest_cols",NULL));
  }

  /* all cols should be destroyed already */
  PetscCall(VecDestroyVecs(N,cols));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFilterZeros_NestPermon"
static PetscErrorCode MatFilterZeros_NestPermon(Mat A, PetscReal tol, Mat *Af_new)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));
  MnNn = Mn*Nn;
  PetscCall(PetscMalloc(MnNn*sizeof(Mat),&mats_out));
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    PetscCall(MatFilterZeros(mats_in[i][j],tol,&mats_out[i*Nn+j]));
  }
  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)A),Mn,NULL,Nn,NULL,mats_out,Af_new));
  for (i=0; i<MnNn; i++) PetscCall(MatDestroy(&mats_out[i]));
  PetscCall(PetscFree(mats_out));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatCreateDenseProductMatrix_NestPermon"
static PetscErrorCode PermonMatCreateDenseProductMatrix_NestPermon(Mat A, PetscBool A_transpose, Mat B, Mat *C_new)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**A_mats,**B_mats,*B_p;
  PetscBool B_nest;
  
  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&B_nest,MATNEST,MATNESTPERMON,""));
  
  if (A_transpose) {
    PetscCall(MatNestGetSubMats(A,NULL,&Mn,&A_mats));
  } else {
    PetscCall(MatNestGetSubMats(A,&Mn,NULL,&A_mats));
  }

  if (B_nest) {
    PetscCall(MatNestGetSubMats(B,NULL,&Nn,&B_mats));
  } else {
    B_p = &B;
    B_mats=&B_p;
    Nn=1;
  }

  if (Mn==1 && Nn==1) {
    PetscCall(PermonMatCreateDenseProductMatrix(A_mats[0][0],A_transpose,B_mats[0][0],C_new));
    PetscFunctionReturn(0);
  }

  MnNn = Mn*Nn;
  PetscCall(PetscMalloc(MnNn*sizeof(Mat),&mats_out));
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    PetscCall(PermonMatCreateDenseProductMatrix(A_transpose?A_mats[0][i]:A_mats[i][0],A_transpose,B_mats[0][j],&mats_out[i*Nn+j]));
  }
  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)A),Mn,NULL,Nn,NULL,mats_out,C_new));
  for (i=0; i<MnNn; i++) PetscCall(MatDestroy(&mats_out[i]));
  PetscCall(PetscFree(mats_out));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatTranspose_NestPermon"
static PetscErrorCode PermonMatTranspose_NestPermon(Mat A,MatTransposeType type,Mat *At_out)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));
  MnNn = Mn*Nn;
  PetscCall(PetscMalloc(MnNn*sizeof(Mat),&mats_out));
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    if (mats_in[i][j]) {
      PetscCall(PermonMatTranspose(mats_in[i][j],type,&mats_out[j*Mn+i]));
    } else {
      mats_out[j*Mn+i] = NULL;
    }
  }
  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)A),Nn,NULL,Mn,NULL,mats_out,At_out));
  for (i=0; i<MnNn; i++) PetscCall(MatDestroy(&mats_out[i]));
  PetscCall(PetscFree(mats_out));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatGetLocalMat_NestPermon"
static PetscErrorCode PermonMatGetLocalMat_NestPermon(Mat A,Mat *Aloc)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;
  
  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));
  MnNn = Mn*Nn;
  PetscCall(PetscMalloc(MnNn*sizeof(Mat),&mats_out));
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    PetscCall(PermonMatGetLocalMat(mats_in[i][j],&mats_out[i*Nn+j]));
  }
  PetscCall(MatCreateNestPermon(PETSC_COMM_SELF,Mn,NULL,Nn,NULL,mats_out,Aloc));
  for (i=0; i<MnNn; i++) PetscCall(MatDestroy(&mats_out[i]));
  PetscCall(PetscFree(mats_out));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMergeAndDestroy_NestPermon"
static PetscErrorCode MatMergeAndDestroy_NestPermon(MPI_Comm comm, Mat *local_in, Vec x, Mat *global_out)
{
  PetscInt i,j,Mn,Nn,MnNn,n,N;
  Mat *mats_out=NULL,**mats_in=NULL;
  Vec *vecs_x=NULL, x_block=NULL;
  PetscBool flg;
  Mat A = *local_in;
  Mat_Nest *data = (Mat_Nest*)A->data;
  
  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));
  
  if (x) {
    PetscInt Nn_cl;
    PetscCall(VecGetLocalSize(x,&n));
    PetscCall(VecGetSize(x,&N));
    if (A->cmap->N != N) SETERRQ(comm,PETSC_ERR_ARG_SIZ,"global length of Vec #3 must be equal to the global number of columns of Mat #2");
    PetscCall(PetscObjectTypeCompare((PetscObject)x,VECNEST,&flg));
    if (!flg) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec #3 has to be nest");
    PetscCall(VecNestGetSubVecs(x,&Nn_cl,&vecs_x));
    if (Nn != Nn_cl) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"number of nested vectors of Vec #3 must be equal to the number of nested columns of Mat #2");
  }

  MnNn = Mn*Nn;
  PetscCall(PetscMalloc(MnNn*sizeof(Mat),&mats_out));

  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    x_block = x ? vecs_x[j] : NULL;
    PetscCall(MatMergeAndDestroy(comm,&data->m[i][j],x_block,&mats_out[i*Nn+j]));
  }
  PetscCall(MatCreateNestPermon(comm,Mn,NULL,Nn,NULL,mats_out,global_out));

  if (x) PERMON_ASSERT((*global_out)->cmap->n == n, "number of local columns requested equals actual number of local columns of the generated matrix (%d != %d)",(*global_out)->cmap->n,n);

  for (i=0; i<MnNn; i++) PetscCall(MatDestroy(&mats_out[i]));
  PetscCall(PetscFree(mats_out));

  PetscCall(MatDestroy(local_in));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PermonMatConvertBlocks_NestPermon"
static PetscErrorCode PermonMatConvertBlocks_NestPermon(Mat A, MatType newtype, MatReuse reuse, Mat* B)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;
  Mat block,cblock;
  Mat B_;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));
  MnNn = Mn*Nn;
  PetscCall(PetscMalloc(MnNn*sizeof(Mat),&mats_out));
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    block = mats_in[i][j];
    cblock = (reuse == MAT_INPLACE_MATRIX) ? block : NULL;
    PetscCall(PermonMatConvertBlocks(block,newtype,reuse,&cblock));
    mats_out[i*Nn+j] = cblock;
  }
  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)A),Mn,NULL,Nn,NULL,mats_out,&B_));
  if (reuse != MAT_INPLACE_MATRIX) {
    for (i=0; i<MnNn; i++) PetscCall(MatDestroy(&mats_out[i]));
    *B = B_;
  } else {
#if PETSC_VERSION_MINOR < 7
    PetscCall(MatHeaderReplace(A,B_));
#else
    PetscCall(MatHeaderReplace(A,&B_));
#endif
  }
  PetscCall(PetscFree(mats_out));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNestPermonGetVecs_NestPermon"
static PetscErrorCode MatNestPermonGetVecs_NestPermon(Mat A,Vec *right,Vec *left)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *L,*R;
  MPI_Comm       comm;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  if (right) {
    /* allocate R */
    PetscCall(PetscMalloc(sizeof(Vec) * bA->nc, &R));
    /* Create the right vectors */
    for (j=0; j<bA->nc; j++) {
      for (i=0; i<bA->nr; i++) {
        if (bA->m[i][j]) {
          PetscCall(MatNestPermonGetVecs(bA->m[i][j],&R[j],NULL));
          break;
        }
      }
      if (i==bA->nr) {
        /* have an empty column */
        SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null column.");
      }
    }
    PetscCall(VecCreateNest(comm,bA->nc,bA->isglobal.col,R,right));
    /* hand back control to the nest vector */
    for (j=0; j<bA->nc; j++) {
      PetscCall(VecDestroy(&R[j]));
    }
    PetscCall(PetscFree(R));
  }

  if (left) {
    /* allocate L */
    PetscCall(PetscMalloc(sizeof(Vec) * bA->nr, &L));
    /* Create the left vectors */
    for (i=0; i<bA->nr; i++) {
      for (j=0; j<bA->nc; j++) {
        if (bA->m[i][j]) {
          PetscCall(MatCreateVecs(bA->m[i][j],NULL,&L[i]));
          break;
        }
      }
      if (j==bA->nc) {
        /* have an empty row */
        SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null row.");
      }
    }

    PetscCall(VecCreateNest(comm,bA->nr,bA->isglobal.row,L,left));
    for (i=0; i<bA->nr; i++) {
      PetscCall(VecDestroy(&L[i]));
    }

    PetscCall(PetscFree(L));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNestSetVecType_NestPermon"
static PetscErrorCode  MatNestSetVecType_NestPermon(Mat A,VecType vtype)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(vtype,VECNEST,&flg));
  /* In reality, this only distinguishes VECNEST and "other" */
  if (flg) A->ops->getvecs = MatNestPermonGetVecs_NestPermon;
  else A->ops->getvecs = (PetscErrorCode (*)(Mat,Vec*,Vec*)) 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNestPermonGetColumnISs_NestPermon"
static PetscErrorCode MatNestPermonGetColumnISs_NestPermon(Mat A,IS **is_new)
{
  PetscInt j,r,Mn,Nn;
  PetscInt lo,m;
  IS *cols, **is_jr;
  const PetscInt *ranges_j;
  PetscMPIInt size;
  Mat **mats_in;
  Mat Aij;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats_in));
  PetscCall(PetscMalloc1(Nn,&cols));
  PetscCall(PetscMalloc1(Nn,&is_jr));
  for (j=0; j<Nn; j++) {
    PetscCall(PetscMalloc1(size, &is_jr[j]));
  }

  lo = 0;
  for (r=0; r<size; r++) {
    for (j=0; j<Nn; j++) {
      Aij = mats_in[0][j];
      PetscCall(MatGetOwnershipRangesColumn(Aij,&ranges_j));
      m = ranges_j[r+1]-ranges_j[r];
      PetscCall(ISCreateStride(PETSC_COMM_SELF,m,lo,1,&is_jr[j][r]));
      lo += m;
    }
  }

  for (j=0; j<Nn; j++) {
    PetscCall(ISConcatenate(PETSC_COMM_SELF,size,is_jr[j],&cols[j]));
    for (r=0; r<size; r++) {
      PetscCall(ISDestroy(&is_jr[j][r]));
    }
    PetscCall(PetscFree(is_jr[j]));
  }
  PetscCall(PetscFree(is_jr));

  *is_new = cols;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNestPermonGetColumnISs"
PetscErrorCode MatNestPermonGetColumnISs(Mat A,IS **is_new)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(is_new,2);
  PetscUseMethod(A,"MatNestPermonGetColumnISs_NestPermon_C",(Mat,IS**),(A,is_new));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNestPermonGetVecs"
PetscErrorCode MatNestPermonGetVecs(Mat A,Vec *x,Vec *y)
{
  PetscErrorCode (*f)(Mat,Vec*,Vec*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  if (x) PetscValidPointer(x,2);
  if (y) PetscValidPointer(y,3);
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatNestPermonGetVecs_C",&f));
  if (!f) f = MatCreateVecs;
  PetscCall((*f)(A,x,y));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_NestPermon_NestPermon"
static PetscErrorCode MatMatMult_NestPermon_NestPermon(Mat A,Mat B,PetscReal fill,Mat *AB_new)
{
  PetscInt i,j,k,M,K1,K2,N,MN;
  Mat *mats_row,*mats_out,**A_mats_in,**B_mats_in;
  Mat AB;

  PetscFunctionBeginI;
  PetscCall(MatNestGetSubMats(A,&M,&K1,&A_mats_in));
  PetscCall(MatNestGetSubMats(B,&K2,&N,&B_mats_in));
  PERMON_ASSERT(K1==K2,"# nest columns of A  =  # nest rows of B");

  MN = M*N;
  PetscCall(PetscMalloc(K1*sizeof(Mat),&mats_row));
  PetscCall(PetscMalloc(MN*sizeof(Mat),&mats_out));
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      for (k=0; k<K1; k++) {
        if (A_mats_in[i][k]) {
          PetscCall(PermonMatMatMult(A_mats_in[i][k],B_mats_in[k][j],MAT_INITIAL_MATRIX,fill,&AB));
        } else {
          AB = NULL;
        }
        mats_row[k] = AB;
      }
      PetscCall(MatCreateComposite(PetscObjectComm((PetscObject)A),K1,mats_row,&AB));
      PetscCall(MatCompositeSetType(AB,MAT_COMPOSITE_ADDITIVE));
      PetscCall(MatCompositeMerge(AB));
      mats_out[i*N+j] = AB;
      for (k=0; k<K1; k++) PetscCall(MatDestroy(&mats_row[k]));
    }
  }
  PetscCall(PetscFree(mats_row));
  
  if (MN==1) {
    /* handle 1x1 nest as normal matrix */
    AB = *mats_out;
  } else {
    PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)A),M,NULL,N,NULL,mats_out,&AB));
    for (i=0; i<MN; i++) PetscCall(MatDestroy(&mats_out[i]));
  }
  PetscCall(PetscFree(mats_out));
  *AB_new = AB;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductNumeric_NestPermon"
static PetscErrorCode MatProductNumeric_NestPermon(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  Mat            newmat;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatMatMult_NestPermon_NestPermon(A,B,product->fill,&newmat));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"MATPRODUCT type is not supported");
  }
  C->product = NULL;
  PetscCall(MatHeaderReplace(C,&newmat));
  C->product = product;
  C->ops->productnumeric = MatProductNumeric_NestPermon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSymbolic_NestPermon"
static PetscErrorCode MatProductSymbolic_NestPermon(Mat C)
{
  PetscFunctionBegin;
  C->ops->productnumeric  = MatProductNumeric_NestPermon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatProductSetFromOptions_NestPermon"
static PetscErrorCode MatProductSetFromOptions_NestPermon(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_NestPermon;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_NestPermon"
static PetscErrorCode MatDuplicate_NestPermon(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Mat            *b;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nr*nc,&b));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (bA->m[i][j]) {
        PetscCall(MatDuplicate(bA->m[i][j],op,&b[i*nc+j]));
        PetscCall(FllopPetscObjectInheritName((PetscObject)b[i*nc+j],(PetscObject)bA->m[i][j],NULL));
      } else {
        b[i*nc+j] = NULL;
      }
    }
  }
  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)A),nr,bA->isglobal.row,nc,bA->isglobal.col,b,B));
  /* Give the new MatNest exclusive ownership */
  for (i=0; i<nr*nc; i++) {
    PetscCall(MatDestroy(&b[i]));
  }
  PetscCall(PetscFree(b));

  PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_NestPermon"
static PetscErrorCode MatAXPY_NestPermon(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  PetscInt i,j,Mn,Nn,Mn1,Nn1;
  Mat **X_mats_in,**Y_mats_in;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(Y,&Mn1,&Nn1,&Y_mats_in));
  PetscCall(MatNestGetSubMats(X,&Mn,&Nn,&X_mats_in));
  PERMON_ASSERT(Mn=Mn1,"Mn=Mn1");
  PERMON_ASSERT(Nn=Nn1,"Nn=Nn1");
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    PetscCall(MatAXPY(Y_mats_in[i][j],a,X_mats_in[i][j],str));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatExtensionCreateCondensedRows_NestPermon"
static PetscErrorCode MatExtensionCreateCondensedRows_NestPermon(Mat TA,Mat *A,IS *ris_local)
{
  PetscInt j,Mn,Nn,m;
  Mat **mats_in,*mats_out;
  IS cis,ris_block,*ris_block_orig,ris_union;
  Mat block,block_A;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(TA,&Mn,&Nn,&mats_in));
  PERMON_ASSERT(Mn==1,"Mn==1");
  PetscCall(PetscMalloc1(Nn,&mats_out));
  PetscCall(PetscMalloc1(Nn,&ris_block_orig));

  for (j=0; j<Nn; j++) {
    block = mats_in[0][j];
    PetscCall(PetscObjectTypeCompare((PetscObject)block,MATEXTENSION,&flg));
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)TA),PETSC_ERR_ARG_WRONG,"nested block %d,0 is not extension",j);
    PetscCall(MatExtensionGetRowISLocal(block,&ris_block_orig[j]));
  }

  PetscCall(ISConcatenate(PETSC_COMM_SELF,Nn,ris_block_orig,&ris_union));
  PetscCall(ISSortRemoveDups(ris_union));
  PetscCall(ISGetLocalSize(ris_union,&m));

  for (j=0; j<Nn; j++) {
    block = mats_in[0][j];
    PetscCall(ISEmbed(ris_block_orig[j],ris_union,PETSC_TRUE,&ris_block));
    PetscCall(MatExtensionGetCondensed(block,&block_A));
    PetscCall(MatExtensionGetColumnIS(block,&cis));
    PetscCall(MatCreateExtension(PetscObjectComm((PetscObject)TA),m,block->cmap->n,PETSC_DECIDE,block->cmap->N,block_A,ris_block,PETSC_FALSE,cis,&mats_out[j]));
    PetscCall(ISDestroy(&ris_block));
  }

  PetscCall(MatCreateNestPermon(PetscObjectComm((PetscObject)TA),1,NULL,Nn,NULL,mats_out,A));
  if (ris_local) {
    *ris_local = ris_union;
  } else {
    PetscCall(ISDestroy(&ris_union));
  }

  PetscCall(PetscFree(mats_out));
  PetscCall(PetscFree(ris_block_orig));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_Nest_NestPermon"
PETSC_EXTERN PetscErrorCode MatConvert_Nest_NestPermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATNESTPERMON));

  B->ops->duplicate             = MatDuplicate_NestPermon;
  B->ops->axpy                  = MatAXPY_NestPermon;
  B->ops->productsetfromoptions = MatProductSetFromOptions_NestPermon;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFilterZeros_C",MatFilterZeros_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"PermonMatCreateDenseProductMatrix_C",PermonMatCreateDenseProductMatrix_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"PermonMatTranspose_C",PermonMatTranspose_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"PermonMatGetLocalMat_C",PermonMatGetLocalMat_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMergeAndDestroy_C",MatMergeAndDestroy_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"PermonMatConvertBlocks_C",PermonMatConvertBlocks_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatNestPermonGetVecs_C",MatNestPermonGetVecs_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatNestPermonGetColumnISs_NestPermon_C",MatNestPermonGetColumnISs_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatNestSetVecType_C",  MatNestSetVecType_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_nest_nestpermon_C", MatConvert_Nest_NestPermon));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatExtensionCreateCondensedRows_Extension_C",MatExtensionCreateCondensedRows_NestPermon));

  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_NestPermon"
PETSC_EXTERN PetscErrorCode MatCreate_NestPermon(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(mat,MATNEST));
  PetscCall(MatConvert_Nest_NestPermon(mat,MATNESTPERMON,MAT_INPLACE_MATRIX,&mat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateNestPermon"
PetscErrorCode MatCreateNestPermon(MPI_Comm comm,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[],Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreateNest(comm,nr,is_row,nc,is_col,a,B));
  PetscCall(MatConvert_Nest_NestPermon(*B,MATNESTPERMON,MAT_INPLACE_MATRIX,B));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateNestPermonVerticalMerge_Extract_Private"
static PetscErrorCode MatCreateNestPermonVerticalMerge_Extract_Private(PetscInt nmats_in,Mat *mats_in,PetscInt *nmats_out,Mat **mats_out)
{
  PetscBool nest;
  PetscInt i,nmats_i;
  PetscInt Mn,Nn;
  Mat A;
  Mat *mats, *mats_out_, *mats_out_p;
  Mat **mats2d;

  PetscFunctionBegin;
  if (!nmats_in) {
    *nmats_out = 0;
    if (mats_out) *mats_out = NULL;
    PetscFunctionReturn(0);
  }

  if (nmats_in == 1) {
    A = mats_in[0];
    PetscCall(PetscObjectTypeCompareAny((PetscObject)A,&nest,MATNEST,MATNESTPERMON,""));
    if (nest) {
      PetscCall(MatNestGetSize(A,&Mn,NULL));
      if (Mn==1) nest = PETSC_FALSE;
    }
    if (!nest) {
      *nmats_out = 1;
      if (mats_out) {
        PetscCall(PetscMalloc1(1,&mats_out_));
        mats_out_[0] = A;
        *mats_out = mats_out_;
      }
    } else {
      PetscCall(MatNestGetSubMats(A,&Mn,&Nn,&mats2d));
      if (Nn > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"number of nested column blocks exceeds 1");
      PetscCall(PetscMalloc1(Mn,&mats));
      for (i=0; i<Mn; i++) mats[i] = mats2d[i][0];
      PetscCall(MatCreateNestPermonVerticalMerge_Extract_Private(Mn,mats,nmats_out,mats_out));
      PetscCall(PetscFree(mats));
    }
    PetscFunctionReturn(0);
  }

  *nmats_out = 0;
  for (i=0; i<nmats_in; i++) {
    PetscCall(MatCreateNestPermonVerticalMerge_Extract_Private(1,&mats_in[i],&nmats_i,NULL));
    *nmats_out += nmats_i;
  }

  if (!mats_out) PetscFunctionReturn(0);
  
  PetscCall(PetscMalloc1(*nmats_out,&mats_out_));
  mats_out_p = mats_out_;
  for (i=0; i<nmats_in; i++) {
    PetscCall(MatCreateNestPermonVerticalMerge_Extract_Private(1,&mats_in[i],&nmats_i,&mats));
    PetscCall(PetscMemcpy(mats_out_p,mats,nmats_i*sizeof(Mat)));
    PetscCall(PetscFree(mats));
    mats_out_p += nmats_i;
  }
  *mats_out = mats_out_;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateNestPermonVerticalMerge"
PetscErrorCode MatCreateNestPermonVerticalMerge(MPI_Comm comm,PetscInt nmats,Mat mats[],Mat *merged)
{
  PetscInt i,nmats_out;
  Mat *mats_out;


  PetscFunctionBegin;
  PetscValidPointer(mats,2);
  for (i=0; i<nmats; i++) PetscValidHeaderSpecific(mats[i],MAT_CLASSID,2);
  PetscValidPointer(merged,3);
  if (!nmats) {
    *merged = NULL;
    PetscFunctionReturn(0);
  }
  PetscCall(MatCreateNestPermonVerticalMerge_Extract_Private(nmats,mats,&nmats_out,&mats_out));
  PetscCall(MatCreateNestPermon(comm,nmats_out,NULL,1,NULL,mats_out,merged));
  PetscCall(PetscFree(mats_out));
  PetscFunctionReturn(0);
}
