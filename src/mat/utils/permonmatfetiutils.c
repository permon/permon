
#include <permonmat.h>
#include <petsc/private/matimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatRemoveGluingOfDirichletDofs_old"
PetscErrorCode MatRemoveGluingOfDirichletDofs_old(Mat Bgt, Vec cg, Mat Bdt, Mat *Bgt_new, Vec *cg_new, IS *is_new)
{
  MPI_Comm comm;
  PetscInt n, lo, hi, jlog, jhig, mg, Mg;
  PetscInt i,j,k,ncolsg,ncolsd;
  const PetscInt *colsg=NULL, *colsd=NULL;
  const PetscScalar *valsg=NULL, *valsd=NULL;
  PetscInt *idx=NULL;
  PetscBool *remove=NULL;
  IS isrow=NULL,iscol=NULL;

  PetscFunctionBeginI;
  n = Bgt->rmap->n;
  lo = Bgt->rmap->rstart;
  hi = Bgt->rmap->rend;
  jlog = Bgt->cmap->rstart;
  jhig = Bgt->cmap->rend;
  mg = Bgt->cmap->n;
  Mg = Bgt->cmap->N;
  TRY( PetscObjectGetComm((PetscObject)Bgt,&comm) );

  TRY( PetscMalloc(Mg*sizeof(PetscBool),&remove) );
  TRY( PetscMemzero(remove,Mg*sizeof(PetscBool)) );

  for (i=lo; i<hi; i++) {
    TRY( MatGetRow(Bdt,i,&ncolsd,&colsd,&valsd) );
    k=0;
    for (j=0; j<ncolsd; j++) {
      if (valsd[j]) k++;
      if (k>1) FLLOP_SETERRQ1(comm,PETSC_ERR_PLIB,"more than one nonzero in Bd row %d",i);
    }
    TRY( MatRestoreRow(Bdt,i,&ncolsd,&colsd,&valsd) );
    if (k) {
      TRY( MatGetRow(Bgt,i,&ncolsg,&colsg,&valsg) );
      for (j=0; j<ncolsg; j++) {
        if (valsg[j]) {
          remove[colsg[j]] = PETSC_TRUE;
        }
      }
      TRY( MatRestoreRow(Bgt,i,&ncolsg,&colsg,&valsg) );
    }
  }

  TRY( MPI_Allreduce(MPI_IN_PLACE,remove,Mg,MPIU_BOOL,MPI_LOR,comm) );

  TRY( PetscMalloc(mg*sizeof(PetscInt),&idx) );
  k=0;
  for (j=jlog; j<jhig; j++) {
    if (!remove[j]) {
      idx[k]=j;
      k++;
    }
  }
  TRY( ISCreateGeneral(comm,k,idx,PETSC_COPY_VALUES,&iscol) );
  TRY( ISCreateStride(comm,n,lo,1,&isrow) );     /* all rows */
  TRY( MatGetSubMatrix(Bgt,isrow,iscol,MAT_INITIAL_MATRIX,Bgt_new) );
  TRY( FllopPetscObjectInheritName((PetscObject)*Bgt_new,(PetscObject)Bgt,NULL) );
  if (cg_new) {
    FLLOP_ASSERT(cg,"cg vector specified");
    TRY( VecGetSubVector(cg,iscol,cg_new) );
    TRY( FllopPetscObjectInheritName((PetscObject)*cg_new,(PetscObject)cg,NULL) );
  }

  if (is_new) {
    *is_new = iscol;
  } else {
    TRY( ISDestroy(&iscol) );
  }
  TRY( PetscFree(remove) );
  TRY( PetscFree(idx) );
  PetscFunctionReturnI(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRemoveGluingOfDirichletDofs"
PetscErrorCode MatRemoveGluingOfDirichletDofs(Mat Bgt, Vec cg, Mat Bdt, Mat *Bgt_new, Vec *cg_new, IS *is_new)
{
  MPI_Comm comm;
  const PetscInt *mg_arr;
  PetscInt lo, hi, jlog, jhig, mg;
  PetscInt i,j,jj,k,ncolsg,ncolsd;
  PetscMPIInt rank,r,commsize;
  const PetscInt *colsg=NULL, *colsd=NULL;
  const PetscScalar *valsg=NULL, *valsd=NULL;
  PetscInt *idx=NULL;
  PetscBool *remove=NULL;
  IS isrow=NULL,iscol=NULL,iscol_self=NULL,ispair[2];

  PetscFunctionBeginI;
  lo = Bgt->rmap->rstart;
  hi = Bgt->rmap->rend;
  TRY( PetscObjectGetComm((PetscObject)Bgt,&comm) );
  TRY( MPI_Comm_size(comm,&commsize) );
  TRY( MPI_Comm_rank(comm,&rank) );

  TRY( MatGetOwnershipRangesColumn(Bgt,&mg_arr) );

  TRY( ISCreateStride(comm,0,0,0,&ispair[0]) );

  for (r=0; r<commsize; r++)
  {
    jlog = mg_arr[r];
    jhig = mg_arr[r+1];
    mg = jhig - jlog;
    
    TRY( PetscMalloc(mg*sizeof(PetscBool),&remove) );
    TRY( PetscMemzero(remove,mg*sizeof(PetscBool)) );
    
    for (i=lo; i<hi; i++) {
      TRY( MatGetRow(Bdt,i,&ncolsd,&colsd,&valsd) );
      k=0;
      for (jj=0; jj<ncolsd; jj++) {
        if (valsd[jj] > PETSC_MACHINE_EPSILON) k++;
        if (k>1) FLLOP_SETERRQ1(comm,PETSC_ERR_PLIB,"more than one nonzero in Bd row %d",i);
      }
      TRY( MatRestoreRow(Bdt,i,&ncolsd,&colsd,&valsd) );
      if (k) {
        TRY( MatGetRow(Bgt,i,&ncolsg,&colsg,&valsg) );
        for (jj=0; jj<ncolsg; jj++) {
          if (valsg[jj] > PETSC_MACHINE_EPSILON) {
            j = colsg[jj];
            if (j>=jlog && j<jhig) remove[j-jlog] = PETSC_TRUE;
          }
        }
        TRY( MatRestoreRow(Bgt,i,&ncolsg,&colsg,&valsg) );
      }
    }
    
    if (r == rank) {
      TRY( MPI_Reduce(MPI_IN_PLACE,remove,mg,MPIU_BOOL,MPI_LOR,r,comm) );
      TRY( PetscMalloc(mg*sizeof(PetscInt),&idx) );
      k=0;
      for (j=jlog; j<jhig; j++) {
        if (!remove[j-jlog]) {
          idx[k]=j;
          k++;
        }
      }
      TRY( ISCreateGeneral(PETSC_COMM_SELF,k,idx,PETSC_OWN_POINTER,&iscol_self) );
    } else {
      TRY( MPI_Reduce(remove,NULL,mg,MPIU_BOOL,MPI_LOR,r,comm) );
    }
    
    TRY( PetscFree(remove) );
  }

  TRY( ISOnComm(iscol_self,comm,PETSC_COPY_VALUES,&iscol) );
  TRY( ISDestroy(&iscol_self) );

  TRY( MatGetOwnershipIS(Bgt,&isrow,NULL) );
  TRY( MatGetSubMatrix(Bgt,isrow,iscol,MAT_INITIAL_MATRIX,Bgt_new) );
  TRY( FllopPetscObjectInheritName((PetscObject)*Bgt_new,(PetscObject)Bgt,NULL) );
  if (cg_new) {
    FLLOP_ASSERT(cg,"cg vector specified");
    TRY( VecGetSubVector(cg,iscol,cg_new) );
    TRY( FllopPetscObjectInheritName((PetscObject)*cg_new,(PetscObject)cg,NULL) );
  }

  if (is_new) {
    *is_new = iscol;
  } else {
    TRY( ISDestroy(&iscol) );
  }
  PetscFunctionReturnI(0);
}
