
#include <permon/private/permonmatimpl.h>
#include <permon/private/petscimpl.h>
#include <petscsf.h>

//#define TAG_firstElemGlobIdx 198533
 
#undef __FUNCT__
#define __FUNCT__ "FllopMatGetLocalMat_Gluing"
static PetscErrorCode FllopMatGetLocalMat_Gluing(Mat A,Mat *Aloc)
{

  Mat_Gluing *data = (Mat_Gluing*) A->data;
  PetscSF SF;
  PetscInt i, N_col, n_row, n_col, start_col;
  PetscInt *leafdata, *rootdata;
  PetscLayout links;

  PetscFunctionBegin;

  TRY( MatGetSize(A, NULL, &N_col) );
  TRY( MatGetLocalSize(A, &n_row, &n_col) );
  TRY( MatGetOwnershipRangeColumn(A, &start_col, NULL));
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscInt), &leafdata) );
  TRY( PetscMalloc(n_col*sizeof(PetscInt), &rootdata) );

  for (i=0; i<n_col; i++) {
    rootdata[i]=start_col + i;
  }
  TRY(PetscSFBcastBegin(data->SF, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  TRY(PetscSFBcastEnd(data->SF, MPIU_INT, rootdata, leafdata, MPI_REPLACE));

  TRY( PetscLayoutCreate(PETSC_COMM_SELF, &links) );
  TRY( PetscLayoutSetBlockSize(links, 1) );
  TRY( PetscLayoutSetSize(links, N_col) );
  TRY( PetscLayoutSetUp(links) );

  TRY( PetscSFCreate(PETSC_COMM_SELF, &SF) );
  TRY( PetscSFSetGraphLayout(SF, links, data->n_leaves, NULL, PETSC_COPY_VALUES, leafdata) );
  TRY( PetscSFSetRankOrder(SF, PETSC_TRUE) );

  TRY(  MatCreateGluing(PETSC_COMM_SELF, n_row, data->n_nonzeroRow, N_col, data->leaves_row, data->leaves_sign, SF, Aloc));

  TRY( PetscFree(leafdata) );
  TRY( PetscFree(rootdata) );
  TRY( PetscLayoutDestroy(&links) );

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_Gluing"
PetscErrorCode MatMult_Gluing(Mat mat, Vec right, Vec left)
{

  Mat_Gluing *data = (Mat_Gluing*) mat->data;
  PetscScalar      *x_onleaves, *lambda_root, *lambda_onleaves;
  PetscInt i, start, *idxX;

  PetscFunctionBegin;
  //right=lambda left=x

  TRY( VecGetArray(right, &lambda_root));
  TRY( VecGetOwnershipRange(left, &start,NULL));
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves) );

  TRY( PetscSFBcastBegin(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE) );
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscInt), &idxX) );
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscScalar), &x_onleaves) );
  TRY( VecZeroEntries(left) );
  TRY( PetscSFBcastEnd(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE) );
  TRY( VecRestoreArray(right,  &lambda_root));

  for (i=0; i<data->n_leaves; i++) {
    idxX[i]= start + data->leaves_row[i];
    x_onleaves[i ] = lambda_onleaves[i] *data->leaves_sign[i];
  }

  TRY( VecZeroEntries( left ));
  TRY( VecSetValues(left, data->n_leaves, idxX, x_onleaves, ADD_VALUES); );
  TRY( VecAssemblyBegin(left) );
  TRY( VecAssemblyEnd(left) );

  TRY( PetscFree(idxX) );
  TRY( PetscFree(x_onleaves) );
  TRY( PetscFree(lambda_onleaves) );

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_Gluing"
PetscErrorCode MatMultAdd_Gluing(Mat mat, Vec right, Vec add, Vec left) {

 Mat_Gluing *data = (Mat_Gluing*) mat->data;
  PetscScalar      *x_onleaves, *lambda_root, *lambda_onleaves;
  PetscInt i, start, *idxX;

  PetscFunctionBegin; 
   //right=lambda left=x

  TRY( VecGetArray(right, &lambda_root));
  TRY( VecGetOwnershipRange(left, &start,NULL));    
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves) );
  
  TRY( PetscSFBcastBegin(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE) );
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscInt), &idxX) );
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscScalar), &x_onleaves) );
  TRY( VecZeroEntries(left) );
  TRY( PetscSFBcastEnd(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE) );
  TRY( VecRestoreArray(right,  &lambda_root));
  
  for (i=0; i<data->n_leaves; i++) {
    idxX[i]= start + data->leaves_row[i];
    x_onleaves[i ] = lambda_onleaves[i] *data->leaves_sign[i];
  } 
  
  TRY( VecZeroEntries( left ));
  TRY( VecSetValues(left, data->n_leaves, idxX, x_onleaves, ADD_VALUES); );
  TRY( VecAssemblyBegin(left) );
  TRY( VecAssemblyEnd(left) ); 
  
  TRY( PetscFree(idxX) ); 
  TRY( PetscFree(x_onleaves) ); 
  TRY( PetscFree(lambda_onleaves) );  
  
  TRY( VecAXPY(left, 1, add) );
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_Gluing"
PetscErrorCode MatMultTranspose_Gluing(Mat mat, Vec right, Vec left)
{
  Mat_Gluing *data = (Mat_Gluing*) mat->data;
  PetscScalar     *x, *lambda_onroot, *lambda_onleaves;
  PetscInt i, start, n_col, *idxL;

  PetscFunctionBegin;
  //right=x left=lambda
  TRY( VecGetArray(right, &x));
  TRY( VecGetOwnershipRange(left, &start,NULL));
  TRY( MatGetLocalSize(mat, NULL, &n_col));

  TRY( PetscMalloc(n_col*sizeof(PetscScalar), &lambda_onroot) );
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves) );

  for (i=0; i<n_col; i++) {
    lambda_onroot[i]=0;
  }

  for ( i=0; i<data->n_leaves; i++) {
    lambda_onleaves[i]= x[ data->leaves_row[i] ] * data->leaves_sign[i];
  }

  TRY( PetscSFReduceBegin(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM) );
  TRY( PetscMalloc(n_col*sizeof(PetscInt), &idxL) );
  for (i=0; i<n_col; i++) {
    idxL[i]=start+i;
  }
  TRY( PetscSFReduceEnd(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM) );

  TRY( VecZeroEntries( left ));
  TRY( VecSetValues(left, n_col, idxL, lambda_onroot, INSERT_VALUES) );
  TRY( VecAssemblyBegin(left) );
  TRY( VecAssemblyEnd(left) );

  TRY( VecRestoreArray(right, &x));
  TRY( PetscFree(lambda_onroot) );
  TRY( PetscFree(lambda_onleaves) );
  TRY( PetscFree(idxL) );

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_Gluing"
PetscErrorCode MatMultTransposeAdd_Gluing(Mat mat, Vec right, Vec add, Vec left)
{

  Mat_Gluing *data = (Mat_Gluing*) mat->data;
  PetscScalar     *x, *lambda_onroot, *lambda_onleaves;
  PetscInt i, start, n_col, *idxL;

  PetscFunctionBegin;
  //right=x left=lambda
  TRY( VecGetArray(right, &x));
  TRY( VecGetOwnershipRange(left, &start,NULL));
  TRY( MatGetLocalSize(mat, NULL, &n_col));

  TRY( PetscMalloc(n_col*sizeof(PetscScalar), &lambda_onroot) );
  TRY( PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves) );

  for (i=0; i<n_col; i++) {
    lambda_onroot[i]=0;
  }

  for ( i=0; i<data->n_leaves; i++) {
    lambda_onleaves[i]= x[ data->leaves_row[i] ] * data->leaves_sign[i];
  }

  TRY( PetscSFReduceBegin(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM) );
  TRY( PetscMalloc(n_col*sizeof(PetscInt), &idxL) );
  for (i=0; i<n_col; i++) {
    idxL[i]=start+i;
  }
  TRY( PetscSFReduceEnd(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM) );
  TRY( VecZeroEntries( left ));
  TRY( VecSetValues(left, n_col, idxL, lambda_onroot, ADD_VALUES) );
  TRY( VecAssemblyBegin(left) );
  TRY( VecAssemblyEnd(left) );

  TRY( VecRestoreArray(right, &x));
  TRY( PetscFree(lambda_onroot) );
  TRY( PetscFree(lambda_onleaves) );
  TRY( PetscFree(idxL) );

  TRY( VecAXPY(left, 1, add) );

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Gluing"
PetscErrorCode MatDestroy_Gluing(Mat mat)
{
  PetscFunctionBegin;
  Mat_Gluing *data = (Mat_Gluing*) mat->data;
  TRY( PetscSFDestroy(&data->SF) );  
  TRY( PetscFree(data->leaves_row) );
  TRY( PetscFree(data->leaves_sign) );
  TRY( PetscFree(data) );
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateGluing"
PetscErrorCode MatCreateGluing(MPI_Comm comm, PetscInt n_x_localRow, PetscInt n_nonzeroRow, PetscInt n_l_localcol,  const PetscInt *leaves_row,	const PetscReal *leaves_sign, PetscSF SF, Mat *B_out)
{
  Mat_Gluing *data;
  PetscInt rlo,rhi,clo,chi, n_l;
  PetscInt *lr;
  PetscReal *ls;
  Mat B;

  PetscFunctionBegin;
  PetscValidPointer(B_out,7);

  /* Create matrix. */
  TRY( MatCreate(comm, &B) );
  TRY( MatSetType(B, MATGLUING) );
  data = (Mat_Gluing*) B->data;
  
 TRY( PetscSFGetLeafRange(SF,NULL,&n_l));    
 
  TRY( PetscMalloc1(n_l+1,&lr) );
  TRY( PetscMemcpy(lr,leaves_row,(n_l+1)*sizeof(PetscInt)) );
  TRY( PetscMalloc1(n_l+1,&ls) );
  TRY( PetscMemcpy(ls,leaves_sign,(n_l+1)*sizeof(PetscReal)) );  
  TRY( PetscObjectReference((PetscObject)SF) );
  
  data->n_leaves=n_l+1;
  data->n_nonzeroRow=n_nonzeroRow;
  data->SF = SF;
  data->leaves_row = lr;
  data->leaves_sign = ls;  
   
  /* Set up row layout */
  TRY( PetscLayoutSetLocalSize(B->rmap, n_x_localRow) );
  TRY( PetscLayoutSetUp(B->rmap) );
  TRY( PetscLayoutGetRange(B->rmap,&rlo,&rhi) );

  /* Set up column layout */
  TRY( PetscLayoutSetLocalSize(B->cmap,n_l_localcol) );
  TRY( PetscLayoutSetUp(B->cmap) );
  TRY( PetscLayoutGetRange(B->cmap,&clo,&chi) );
    
  *B_out = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Gluing"
FLLOP_EXTERN PetscErrorCode MatCreate_Gluing(Mat B) {
  
  Mat_Gluing *data;

  PetscFunctionBegin;
  TRY( PetscNew(&data) );
  TRY( PetscObjectChangeTypeName((PetscObject)B, MATGLUING) );
  B->data                = (void*) data;
  B->assembled           = PETSC_TRUE;
  B->preallocated        = PETSC_TRUE;
  
  data->SF               = NULL; 
  data->leaves_row      = NULL;
  data->leaves_sign      = NULL; 
  data->n_leaves=0; 
  data->n_nonzeroRow=0;
  
  /* Set operations of matrix. */  
  B->ops->destroy            = MatDestroy_Gluing;
  B->ops->mult               = MatMult_Gluing;
  B->ops->multtranspose      = MatMultTranspose_Gluing;
  B->ops->multadd            = MatMultAdd_Gluing;
  B->ops->multtransposeadd   = MatMultTransposeAdd_Gluing;
  TRY( PetscObjectComposeFunction((PetscObject)B,"FllopMatGetLocalMat_C",FllopMatGetLocalMat_Gluing) );
 
  PetscFunctionReturn(0);
} 
