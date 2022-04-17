
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

  CHKERRQ(MatGetSize(A, NULL, &N_col));
  CHKERRQ(MatGetLocalSize(A, &n_row, &n_col));
  CHKERRQ(MatGetOwnershipRangeColumn(A, &start_col, NULL));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscInt), &leafdata));
  CHKERRQ(PetscMalloc(n_col*sizeof(PetscInt), &rootdata));

  for (i=0; i<n_col; i++) {
    rootdata[i]=start_col + i;
  }
  CHKERRQ(PetscSFBcastBegin(data->SF, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(data->SF, MPIU_INT, rootdata, leafdata, MPI_REPLACE));

  CHKERRQ(PetscLayoutCreate(PETSC_COMM_SELF, &links));
  CHKERRQ(PetscLayoutSetBlockSize(links, 1));
  CHKERRQ(PetscLayoutSetSize(links, N_col));
  CHKERRQ(PetscLayoutSetUp(links));

  CHKERRQ(PetscSFCreate(PETSC_COMM_SELF, &SF));
  CHKERRQ(PetscSFSetGraphLayout(SF, links, data->n_leaves, NULL, PETSC_COPY_VALUES, leafdata));
  CHKERRQ(PetscSFSetRankOrder(SF, PETSC_TRUE));

  CHKERRQ( MatCreateGluing(PETSC_COMM_SELF, n_row, data->n_nonzeroRow, N_col, data->leaves_row, data->leaves_sign, SF, Aloc));

  CHKERRQ(PetscFree(leafdata));
  CHKERRQ(PetscFree(rootdata));
  CHKERRQ(PetscLayoutDestroy(&links));

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

  CHKERRQ(VecGetArray(right, &lambda_root));
  CHKERRQ(VecGetOwnershipRange(left, &start,NULL));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves));

  CHKERRQ(PetscSFBcastBegin(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscInt), &idxX));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscScalar), &x_onleaves));
  CHKERRQ(VecZeroEntries(left));
  CHKERRQ(PetscSFBcastEnd(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  CHKERRQ(VecRestoreArray(right,  &lambda_root));

  for (i=0; i<data->n_leaves; i++) {
    idxX[i]= start + data->leaves_row[i];
    x_onleaves[i ] = lambda_onleaves[i] *data->leaves_sign[i];
  }

  CHKERRQ(VecZeroEntries( left ));
  CHKERRQ(VecSetValues(left, data->n_leaves, idxX, x_onleaves, ADD_VALUES););
  CHKERRQ(VecAssemblyBegin(left));
  CHKERRQ(VecAssemblyEnd(left));

  CHKERRQ(PetscFree(idxX));
  CHKERRQ(PetscFree(x_onleaves));
  CHKERRQ(PetscFree(lambda_onleaves));

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

  CHKERRQ(VecGetArray(right, &lambda_root));
  CHKERRQ(VecGetOwnershipRange(left, &start,NULL));    
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves));
  
  CHKERRQ(PetscSFBcastBegin(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscInt), &idxX));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscScalar), &x_onleaves));
  CHKERRQ(VecZeroEntries(left));
  CHKERRQ(PetscSFBcastEnd(data->SF, MPIU_SCALAR, lambda_root, lambda_onleaves, MPI_REPLACE));
  CHKERRQ(VecRestoreArray(right,  &lambda_root));
  
  for (i=0; i<data->n_leaves; i++) {
    idxX[i]= start + data->leaves_row[i];
    x_onleaves[i ] = lambda_onleaves[i] *data->leaves_sign[i];
  } 
  
  CHKERRQ(VecZeroEntries( left ));
  CHKERRQ(VecSetValues(left, data->n_leaves, idxX, x_onleaves, ADD_VALUES););
  CHKERRQ(VecAssemblyBegin(left));
  CHKERRQ(VecAssemblyEnd(left)); 
  
  CHKERRQ(PetscFree(idxX)); 
  CHKERRQ(PetscFree(x_onleaves)); 
  CHKERRQ(PetscFree(lambda_onleaves));  
  
  CHKERRQ(VecAXPY(left, 1, add));
  
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
  CHKERRQ(VecGetArray(right, &x));
  CHKERRQ(VecGetOwnershipRange(left, &start,NULL));
  CHKERRQ(MatGetLocalSize(mat, NULL, &n_col));

  CHKERRQ(PetscMalloc(n_col*sizeof(PetscScalar), &lambda_onroot));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves));

  for (i=0; i<n_col; i++) {
    lambda_onroot[i]=0;
  }

  for ( i=0; i<data->n_leaves; i++) {
    lambda_onleaves[i]= x[ data->leaves_row[i] ] * data->leaves_sign[i];
  }

  CHKERRQ(PetscSFReduceBegin(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));
  CHKERRQ(PetscMalloc(n_col*sizeof(PetscInt), &idxL));
  for (i=0; i<n_col; i++) {
    idxL[i]=start+i;
  }
  CHKERRQ(PetscSFReduceEnd(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));

  CHKERRQ(VecZeroEntries( left ));
  CHKERRQ(VecSetValues(left, n_col, idxL, lambda_onroot, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(left));
  CHKERRQ(VecAssemblyEnd(left));

  CHKERRQ(VecRestoreArray(right, &x));
  CHKERRQ(PetscFree(lambda_onroot));
  CHKERRQ(PetscFree(lambda_onleaves));
  CHKERRQ(PetscFree(idxL));

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
  CHKERRQ(VecGetArray(right, &x));
  CHKERRQ(VecGetOwnershipRange(left, &start,NULL));
  CHKERRQ(MatGetLocalSize(mat, NULL, &n_col));

  CHKERRQ(PetscMalloc(n_col*sizeof(PetscScalar), &lambda_onroot));
  CHKERRQ(PetscMalloc(data->n_leaves*sizeof(PetscScalar), &lambda_onleaves));

  for (i=0; i<n_col; i++) {
    lambda_onroot[i]=0;
  }

  for ( i=0; i<data->n_leaves; i++) {
    lambda_onleaves[i]= x[ data->leaves_row[i] ] * data->leaves_sign[i];
  }

  CHKERRQ(PetscSFReduceBegin(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));
  CHKERRQ(PetscMalloc(n_col*sizeof(PetscInt), &idxL));
  for (i=0; i<n_col; i++) {
    idxL[i]=start+i;
  }
  CHKERRQ(PetscSFReduceEnd(data->SF, MPIU_SCALAR, lambda_onleaves, lambda_onroot, MPI_SUM));
  CHKERRQ(VecZeroEntries( left ));
  CHKERRQ(VecSetValues(left, n_col, idxL, lambda_onroot, ADD_VALUES));
  CHKERRQ(VecAssemblyBegin(left));
  CHKERRQ(VecAssemblyEnd(left));

  CHKERRQ(VecRestoreArray(right, &x));
  CHKERRQ(PetscFree(lambda_onroot));
  CHKERRQ(PetscFree(lambda_onleaves));
  CHKERRQ(PetscFree(idxL));

  CHKERRQ(VecAXPY(left, 1, add));

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Gluing"
PetscErrorCode MatDestroy_Gluing(Mat mat)
{
  PetscFunctionBegin;
  Mat_Gluing *data = (Mat_Gluing*) mat->data;
  CHKERRQ(PetscSFDestroy(&data->SF));  
  CHKERRQ(PetscFree(data->leaves_row));
  CHKERRQ(PetscFree(data->leaves_sign));
  CHKERRQ(PetscFree(data));
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
  CHKERRQ(MatCreate(comm, &B));
  CHKERRQ(MatSetType(B, MATGLUING));
  data = (Mat_Gluing*) B->data;
  
 CHKERRQ(PetscSFGetLeafRange(SF,NULL,&n_l));    
 
  CHKERRQ(PetscMalloc1(n_l+1,&lr));
  CHKERRQ(PetscMemcpy(lr,leaves_row,(n_l+1)*sizeof(PetscInt)));
  CHKERRQ(PetscMalloc1(n_l+1,&ls));
  CHKERRQ(PetscMemcpy(ls,leaves_sign,(n_l+1)*sizeof(PetscReal)));  
  CHKERRQ(PetscObjectReference((PetscObject)SF));
  
  data->n_leaves=n_l+1;
  data->n_nonzeroRow=n_nonzeroRow;
  data->SF = SF;
  data->leaves_row = lr;
  data->leaves_sign = ls;  
   
  /* Set up row layout */
  CHKERRQ(PetscLayoutSetLocalSize(B->rmap, n_x_localRow));
  CHKERRQ(PetscLayoutSetUp(B->rmap));
  CHKERRQ(PetscLayoutGetRange(B->rmap,&rlo,&rhi));

  /* Set up column layout */
  CHKERRQ(PetscLayoutSetLocalSize(B->cmap,n_l_localcol));
  CHKERRQ(PetscLayoutSetUp(B->cmap));
  CHKERRQ(PetscLayoutGetRange(B->cmap,&clo,&chi));
    
  *B_out = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Gluing"
FLLOP_EXTERN PetscErrorCode MatCreate_Gluing(Mat B) {
  
  Mat_Gluing *data;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&data));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B, MATGLUING));
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
  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"FllopMatGetLocalMat_C",FllopMatGetLocalMat_Gluing));
 
  PetscFunctionReturn(0);
} 
