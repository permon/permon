
#include <private/fllopmatimpl.h>
#include <private/petscimpl.h>

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
  TRY( PetscObjectGetComm((PetscObject)A,&comm) );

  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats) );
  TRY( PetscMalloc(sizeof(Vec*)*Mn,&cols_for_each_row_block) );
  TRY( PetscMalloc(sizeof(Vec)*N,&cols); );
  TRY( PetscMalloc(sizeof(Vec)*Mn,&col_arr) );
  TRY( PetscMalloc(sizeof(IS)*Nn,&is_glob_col) );
  TRY( MatNestGetISs(A,NULL,is_glob_col) );

  for (J=0; J<Nn; J++) {
    N1_last = mats[0][J]->cmap->N;
    for (II=0; II<Mn; II++) {
      mat = mats[II][J];
      if (!mat) FLLOP_SETERRQ2(comm,PETSC_ERR_SUP,"block (%d, %d) is null but null blocks not currently supported",II,J);
      TRY( MatGetColumnVectors(mat,&N1,&cols_for_each_row_block[II]) );
      if (N1 != N1_last) FLLOP_SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"block (%d, %d) has different number of columns than block (%d, %d)",II,J,II-1,J);
      N1_last = N1;

      TRY( PetscContainerCreate(comm, &container) );
      TRY( PetscContainerSetPointer(container,cols_for_each_row_block[II]) );
      TRY( PetscObjectCompose((PetscObject)mat,"MatGetColumnVectors_Nest_cols",(PetscObject)container) );
      TRY( PetscContainerDestroy(&container) );
    }

    /* IS is_glob_col[J] holds indices of columns of the J-th column block in the global numbering of global matrix A */
    TRY( ISAllGather(is_glob_col[J],&is_seq) );
    TRY( ISGetIndices(is_seq,&is_arr) );
    
    for (j=0; j<N1; j++) {
      for (II=0; II<Mn; II++) { col_arr[II] = cols_for_each_row_block[II][j]; }
      TRY( VecCreateNest(comm,Mn,NULL,col_arr,&cols[is_arr[j]]) );
    }
    TRY( ISRestoreIndices(is_seq,&is_arr) );
    TRY( ISDestroy(&is_seq) );
  }

  TRY( PetscFree(is_glob_col) );
  TRY( PetscFree(col_arr) );
  TRY( PetscFree(cols_for_each_row_block) );
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
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats) );

  for (II=0; II<Mn; II++) for (J=0; J<Nn; J++) {
    mat = mats[II][J];
    TRY( PetscObjectQuery((PetscObject)mat,"MatGetColumnVectors_Nest_cols",(PetscObject*)&container) );
    TRY( PetscContainerGetPointer(container,(void**)&block_cols) );
    TRY( MatRestoreColumnVectors(mat,&N1,&block_cols) );
    TRY( PetscObjectCompose((PetscObject)mat,"MatGetColumnVectors_Nest_cols",NULL) );
  }

  /* all cols should be destroyed already */
  TRY( VecDestroyVecs(N,cols) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFilterZeros_NestPermon"
static PetscErrorCode MatFilterZeros_NestPermon(Mat A, PetscReal tol, Mat *Af_new)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;

  PetscFunctionBegin;
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats_in) );
  MnNn = Mn*Nn;
  TRY( PetscMalloc(MnNn*sizeof(Mat),&mats_out) );
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    TRY( MatFilterZeros(mats_in[i][j],tol,&mats_out[i*Nn+j]) );
  }
  TRY( MatCreateNestPermon(PetscObjectComm((PetscObject)A),Mn,NULL,Nn,NULL,mats_out,Af_new) );
  for (i=0; i<MnNn; i++) TRY( MatDestroy(&mats_out[i]) );
  TRY( PetscFree(mats_out) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatCreateDenseProductMatrix_NestPermon"
static PetscErrorCode FllopMatCreateDenseProductMatrix_NestPermon(Mat A, PetscBool A_transpose, Mat B, Mat *C_new)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**A_mats,**B_mats,*B_p;
  PetscBool B_nest;
  
  PetscFunctionBegin;
  TRY( PetscObjectTypeCompareAny((PetscObject)B,&B_nest,MATNEST,MATNESTPERMON,"") );
  
  if (A_transpose) {
    TRY( MatNestGetSubMats(A,NULL,&Mn,&A_mats) );
  } else {
    TRY( MatNestGetSubMats(A,&Mn,NULL,&A_mats) );
  }

  if (B_nest) {
    TRY( MatNestGetSubMats(B,NULL,&Nn,&B_mats) );
  } else {
    B_p = &B;
    B_mats=&B_p;
    Nn=1;
  }

  if (Mn==1 && Nn==1) {
    TRY( FllopMatCreateDenseProductMatrix(A_mats[0][0],A_transpose,B_mats[0][0],C_new) );
    PetscFunctionReturn(0);
  }

  MnNn = Mn*Nn;
  TRY( PetscMalloc(MnNn*sizeof(Mat),&mats_out) );
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    TRY( FllopMatCreateDenseProductMatrix(A_transpose?A_mats[0][i]:A_mats[i][0],A_transpose,B_mats[0][j],&mats_out[i*Nn+j]) );
  }
  TRY( MatCreateNestPermon(PetscObjectComm((PetscObject)A),Mn,NULL,Nn,NULL,mats_out,C_new) );
  for (i=0; i<MnNn; i++) TRY( MatDestroy(&mats_out[i]) );
  TRY( PetscFree(mats_out) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatTranspose_NestPermon"
static PetscErrorCode FllopMatTranspose_NestPermon(Mat A,MatTransposeType type,Mat *At_out)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;

  PetscFunctionBegin;
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats_in) );
  MnNn = Mn*Nn;
  TRY( PetscMalloc(MnNn*sizeof(Mat),&mats_out) );
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    if (mats_in[i][j]) {
      TRY( FllopMatTranspose(mats_in[i][j],type,&mats_out[j*Mn+i]) );
    } else {
      mats_out[j*Mn+i] = NULL;
    }
  }
  TRY( MatCreateNestPermon(PetscObjectComm((PetscObject)A),Nn,NULL,Mn,NULL,mats_out,At_out) );
  for (i=0; i<MnNn; i++) TRY( MatDestroy(&mats_out[i]) );
  TRY( PetscFree(mats_out) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatGetLocalMat_NestPermon"
static PetscErrorCode FllopMatGetLocalMat_NestPermon(Mat A,Mat *Aloc)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;
  
  PetscFunctionBegin;
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats_in) );
  MnNn = Mn*Nn;
  TRY( PetscMalloc(MnNn*sizeof(Mat),&mats_out) );
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    TRY( FllopMatGetLocalMat(mats_in[i][j],&mats_out[i*Nn+j]) );
  }
  TRY( MatCreateNestPermon(PETSC_COMM_SELF,Mn,NULL,Nn,NULL,mats_out,Aloc) );
  for (i=0; i<MnNn; i++) TRY( MatDestroy(&mats_out[i]) );
  TRY( PetscFree(mats_out) );
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
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats_in) );
  
  if (x) {
    PetscInt Nn_cl;
    TRY( VecGetLocalSize(x,&n) );
    TRY( VecGetSize(x,&N) );
    if (A->cmap->N != N) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_SIZ,"global length of Vec #3 must be equal to the global number of columns of Mat #2");
    TRY( PetscObjectTypeCompare((PetscObject)x,VECNEST,&flg) );
    if (!flg) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Vec #3 has to be nest");
    TRY( VecNestGetSubVecs(x,&Nn_cl,&vecs_x) );
    if (Nn != Nn_cl) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"number of nested vectors of Vec #3 must be equal to the number of nested columns of Mat #2");
  }

  MnNn = Mn*Nn;
  TRY( PetscMalloc(MnNn*sizeof(Mat),&mats_out) );

  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    x_block = x ? vecs_x[j] : NULL;
    TRY( MatMergeAndDestroy(comm,&data->m[i][j],x_block,&mats_out[i*Nn+j]) );
  }
  TRY( MatCreateNestPermon(comm,Mn,NULL,Nn,NULL,mats_out,global_out) );

  if (x) FLLOP_ASSERT2((*global_out)->cmap->n == n, "number of local columns requested equals actual number of local columns of the generated matrix (%d != %d)",(*global_out)->cmap->n,n);

  for (i=0; i<MnNn; i++) TRY( MatDestroy(&mats_out[i]) );
  TRY( PetscFree(mats_out) );

  TRY( MatDestroy(local_in) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FllopMatConvertBlocks_NestPermon"
static PetscErrorCode FllopMatConvertBlocks_NestPermon(Mat A, MatType newtype, MatReuse reuse, Mat* B)
{
  PetscInt i,j,Mn,Nn,MnNn;
  Mat *mats_out,**mats_in;
  Mat block,cblock;
  Mat B_;

  PetscFunctionBegin;
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats_in) );
  MnNn = Mn*Nn;
  TRY( PetscMalloc(MnNn*sizeof(Mat),&mats_out) );
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    block = mats_in[i][j];
    cblock = (reuse == MAT_INPLACE_MATRIX) ? block : NULL;
    TRY( FllopMatConvertBlocks(block,newtype,reuse,&cblock) );
    mats_out[i*Nn+j] = cblock;
  }
  TRY( MatCreateNestPermon(PetscObjectComm((PetscObject)A),Mn,NULL,Nn,NULL,mats_out,&B_) );
  if (reuse != MAT_INPLACE_MATRIX) {
    for (i=0; i<MnNn; i++) TRY( MatDestroy(&mats_out[i]) );
    *B = B_;
  } else {
#if PETSC_VERSION_MINOR < 7
    TRY( MatHeaderReplace(A,B_) );
#else
    TRY( MatHeaderReplace(A,&B_) );
#endif
  }
  TRY( PetscFree(mats_out) );
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (right) {
    /* allocate R */
    ierr = PetscMalloc(sizeof(Vec) * bA->nc, &R);CHKERRQ(ierr);
    /* Create the right vectors */
    for (j=0; j<bA->nc; j++) {
      for (i=0; i<bA->nr; i++) {
        if (bA->m[i][j]) {
          ierr = MatNestPermonGetVecs(bA->m[i][j],&R[j],NULL);CHKERRQ(ierr);
          break;
        }
      }
      if (i==bA->nr) {
        /* have an empty column */
        SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null column.");
      }
    }
    ierr = VecCreateNest(comm,bA->nc,bA->isglobal.col,R,right);CHKERRQ(ierr);
    /* hand back control to the nest vector */
    for (j=0; j<bA->nc; j++) {
      ierr = VecDestroy(&R[j]);CHKERRQ(ierr);
    }
    ierr = PetscFree(R);CHKERRQ(ierr);
  }

  if (left) {
    /* allocate L */
    ierr = PetscMalloc(sizeof(Vec) * bA->nr, &L);CHKERRQ(ierr);
    /* Create the left vectors */
    for (i=0; i<bA->nr; i++) {
      for (j=0; j<bA->nc; j++) {
        if (bA->m[i][j]) {
          ierr = MatCreateVecs(bA->m[i][j],NULL,&L[i]);CHKERRQ(ierr);
          break;
        }
      }
      if (j==bA->nc) {
        /* have an empty row */
        SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null row.");
      }
    }

    ierr = VecCreateNest(comm,bA->nr,bA->isglobal.row,L,left);CHKERRQ(ierr);
    for (i=0; i<bA->nr; i++) {
      ierr = VecDestroy(&L[i]);CHKERRQ(ierr);
    }

    ierr = PetscFree(L);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNestSetVecType_NestPermon"
static PetscErrorCode  MatNestSetVecType_NestPermon(Mat A,VecType vtype)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscStrcmp(vtype,VECNEST,&flg);CHKERRQ(ierr);
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
  TRY( MPI_Comm_size(PetscObjectComm((PetscObject)A),&size) );
  TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats_in) );
  TRY( PetscMalloc1(Nn,&cols) );
  TRY( PetscMalloc1(Nn,&is_jr) );
  for (j=0; j<Nn; j++) {
    TRY( PetscMalloc1(size, &is_jr[j]) );
  }

  lo = 0;
  for (r=0; r<size; r++) {
    for (j=0; j<Nn; j++) {
      Aij = mats_in[0][j];
      TRY( MatGetOwnershipRangesColumn(Aij,&ranges_j) );
      m = ranges_j[r+1]-ranges_j[r];
      TRY( ISCreateStride(PETSC_COMM_SELF,m,lo,1,&is_jr[j][r]) );
      lo += m;
    }
  }

  for (j=0; j<Nn; j++) {
    TRY( ISConcatenate(PETSC_COMM_SELF,size,is_jr[j],&cols[j]) );
    for (r=0; r<size; r++) {
      TRY( ISDestroy(&is_jr[j][r]) );
    }
    TRY( PetscFree(is_jr[j]) );
  }
  TRY( PetscFree(is_jr) );

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
  TRY( PetscUseMethod(A,"MatNestPermonGetColumnISs_NestPermon_C",(Mat,IS**),(A,is_new)) );
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
  TRY( PetscObjectQueryFunction((PetscObject)A,"MatNestPermonGetVecs_C",&f) );
  if (!f) f = MatCreateVecs;
  TRY( (*f)(A,x,y) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_NestPermon_NestPermon"
static PetscErrorCode MatMatMult_NestPermon_NestPermon(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *AB_new)
{
  PetscInt i,j,k,M,K1,K2,N,MN;
  Mat *mats_row,*mats_out,**A_mats_in,**B_mats_in;
  Mat AB;

  PetscFunctionBeginI;
  if (scall != MAT_INITIAL_MATRIX) FLLOP_SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"only MAT_INITIAL_MATRIX supported for MATNEST");
  TRY( MatNestGetSubMats(A,&M,&K1,&A_mats_in) );
  TRY( MatNestGetSubMats(B,&K2,&N,&B_mats_in) );
  FLLOP_ASSERT(K1==K2,"# nest columns of A  =  # nest rows of B");

  MN = M*N;
  TRY( PetscMalloc(K1*sizeof(Mat),&mats_row) );
  TRY( PetscMalloc(MN*sizeof(Mat),&mats_out) );
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      for (k=0; k<K1; k++) {
        if (A_mats_in[i][k]) {
          TRY( FllopMatMatMult(A_mats_in[i][k],B_mats_in[k][j],MAT_INITIAL_MATRIX,fill,&AB) );
        } else {
          AB = NULL;
        }
        mats_row[k] = AB;
      }
      TRY( MatCreateComposite(PetscObjectComm((PetscObject)A),K1,mats_row,&AB) );
      TRY( MatCompositeSetType(AB,MAT_COMPOSITE_ADDITIVE) );
      TRY( MatCompositeMerge(AB) );
      mats_out[i*N+j] = AB;
      for (k=0; k<K1; k++) TRY( MatDestroy(&mats_row[k]) );
    }
  }
  TRY( PetscFree(mats_row) );
  
  if (MN==1) {
    /* handle 1x1 nest as normal matrix */
    AB = *mats_out;
  } else {
    TRY( MatCreateNestPermon(PetscObjectComm((PetscObject)A),M,NULL,N,NULL,mats_out,&AB) );
    for (i=0; i<MN; i++) TRY( MatDestroy(&mats_out[i]) );
  }
  TRY( PetscFree(mats_out) );
  *AB_new = AB;
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_NestPermon"
static PetscErrorCode MatDuplicate_NestPermon(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Mat            *b;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nr*nc,&b);CHKERRQ(ierr);
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (bA->m[i][j]) {
        ierr = MatDuplicate(bA->m[i][j],op,&b[i*nc+j]);CHKERRQ(ierr);
        ierr = FllopPetscObjectInheritName((PetscObject)b[i*nc+j],(PetscObject)bA->m[i][j],NULL);CHKERRQ(ierr);
      } else {
        b[i*nc+j] = NULL;
      }
    }
  }
  ierr = MatCreateNestPermon(PetscObjectComm((PetscObject)A),nr,bA->isglobal.row,nc,bA->isglobal.col,b,B);CHKERRQ(ierr);
  /* Give the new MatNest exclusive ownership */
  for (i=0; i<nr*nc; i++) {
    ierr = MatDestroy(&b[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_NestPermon"
static PetscErrorCode MatAXPY_NestPermon(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  PetscInt i,j,Mn,Nn,Mn1,Nn1;
  Mat **X_mats_in,**Y_mats_in;

  PetscFunctionBegin;
  TRY( MatNestGetSubMats(Y,&Mn1,&Nn1,&Y_mats_in) );
  TRY( MatNestGetSubMats(X,&Mn,&Nn,&X_mats_in) );
  FLLOP_ASSERT(Mn=Mn1,"Mn=Mn1");
  FLLOP_ASSERT(Nn=Nn1,"Nn=Nn1");
  for (i=0; i<Mn; i++) for (j=0; j<Nn; j++) {
    TRY( MatAXPY(Y_mats_in[i][j],a,X_mats_in[i][j],str) );
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatConvert_Nest_NestPermon"
PETSC_EXTERN PetscErrorCode MatConvert_Nest_NestPermon(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    TRY( MatDuplicate(A,MAT_COPY_VALUES,&B) );
  }
  
  TRY( PetscObjectChangeTypeName((PetscObject)B,MATNESTPERMON) );

  B->ops->duplicate          = MatDuplicate_NestPermon;
  B->ops->axpy               = MatAXPY_NestPermon;
  B->ops->matmult            = MatMatMult_NestPermon_NestPermon;
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatGetColumnVectors_C",MatGetColumnVectors_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatRestoreColumnVectors_C",MatRestoreColumnVectors_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatFilterZeros_C",MatFilterZeros_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"FllopMatCreateDenseProductMatrix_C",FllopMatCreateDenseProductMatrix_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"FllopMatTranspose_C",FllopMatTranspose_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"FllopMatGetLocalMat_C",FllopMatGetLocalMat_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatMergeAndDestroy_C",MatMergeAndDestroy_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"FllopMatConvertBlocks_C",FllopMatConvertBlocks_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatNestPermonGetVecs_C",MatNestPermonGetVecs_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatNestPermonGetColumnISs_NestPermon_C",MatNestPermonGetColumnISs_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatNestSetVecType_C",  MatNestSetVecType_NestPermon) );
  TRY( PetscObjectComposeFunction((PetscObject)B,"MatConvert_nest_nestpermon_C", MatConvert_Nest_NestPermon) );

  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_NestPermon"
PETSC_EXTERN PetscErrorCode MatCreate_NestPermon(Mat mat)
{
  PetscFunctionBegin;
  TRY( MatSetType(mat,MATNEST) );
  TRY( MatConvert_Nest_NestPermon(mat,MATNESTPERMON,MAT_INPLACE_MATRIX,&mat) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateNestPermon"
PetscErrorCode MatCreateNestPermon(MPI_Comm comm,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[],Mat *B)
{
  PetscFunctionBegin;
  TRY( MatCreateNest(comm,nr,is_row,nc,is_col,a,B) );
  TRY( MatConvert_Nest_NestPermon(*B,MATNESTPERMON,MAT_INPLACE_MATRIX,B) );
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
    TRY( PetscObjectTypeCompareAny((PetscObject)A,&nest,MATNEST,MATNESTPERMON,"") );
    if (nest) {
      TRY( MatNestGetSize(A,&Mn,NULL) );
      if (Mn==1) nest = PETSC_FALSE;
    }
    if (!nest) {
      *nmats_out = 1;
      if (mats_out) {
        TRY( PetscMalloc1(1,&mats_out_) );
        mats_out_[0] = A;
        *mats_out = mats_out_;
      }
    } else {
      TRY( MatNestGetSubMats(A,&Mn,&Nn,&mats2d) );
      if (Nn > 1) FLLOP_SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"number of nested column blocks exceeds 1");
      TRY( PetscMalloc1(Mn,&mats) );
      for (i=0; i<Mn; i++) mats[i] = mats2d[i][0];
      TRY( MatCreateNestPermonVerticalMerge_Extract_Private(Mn,mats,nmats_out,mats_out) );
      TRY( PetscFree(mats) );
    }
    PetscFunctionReturn(0);
  }

  *nmats_out = 0;
  for (i=0; i<nmats_in; i++) {
    TRY( MatCreateNestPermonVerticalMerge_Extract_Private(1,&mats_in[i],&nmats_i,NULL) );
    *nmats_out += nmats_i;
  }

  if (!mats_out) PetscFunctionReturn(0);
  
  TRY( PetscMalloc1(*nmats_out,&mats_out_) );
  mats_out_p = mats_out_;
  for (i=0; i<nmats_in; i++) {
    TRY( MatCreateNestPermonVerticalMerge_Extract_Private(1,&mats_in[i],&nmats_i,&mats) );
    TRY( PetscMemcpy(mats_out_p,mats,nmats_i*sizeof(Mat)) );
    TRY( PetscFree(mats) );
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
  TRY( MatCreateNestPermonVerticalMerge_Extract_Private(nmats,mats,&nmats_out,&mats_out) );
  TRY( MatCreateNestPermon(comm,nmats_out,NULL,1,NULL,mats_out,merged) );
  TRY( PetscFree(mats_out) );
  PetscFunctionReturn(0);
}
