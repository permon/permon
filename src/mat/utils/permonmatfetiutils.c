
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
  CHKERRQ(PetscObjectGetComm((PetscObject)Bgt,&comm));

  CHKERRQ(PetscMalloc(Mg*sizeof(PetscBool),&remove));
  CHKERRQ(PetscMemzero(remove,Mg*sizeof(PetscBool)));

  for (i=lo; i<hi; i++) {
    CHKERRQ(MatGetRow(Bdt,i,&ncolsd,&colsd,&valsd));
    k=0;
    for (j=0; j<ncolsd; j++) {
      if (valsd[j]) k++;
      if (k>1) SETERRQ(comm,PETSC_ERR_PLIB,"more than one nonzero in Bd row %d",i);
    }
    CHKERRQ(MatRestoreRow(Bdt,i,&ncolsd,&colsd,&valsd));
    if (k) {
      CHKERRQ(MatGetRow(Bgt,i,&ncolsg,&colsg,&valsg));
      for (j=0; j<ncolsg; j++) {
        if (valsg[j]) {
          remove[colsg[j]] = PETSC_TRUE;
        }
      }
      CHKERRQ(MatRestoreRow(Bgt,i,&ncolsg,&colsg,&valsg));
    }
  }

  CHKERRQ(MPI_Allreduce(MPI_IN_PLACE,remove,Mg,MPIU_BOOL,MPI_LOR,comm));

  CHKERRQ(PetscMalloc(mg*sizeof(PetscInt),&idx));
  k=0;
  for (j=jlog; j<jhig; j++) {
    if (!remove[j]) {
      idx[k]=j;
      k++;
    }
  }
  CHKERRQ(ISCreateGeneral(comm,k,idx,PETSC_COPY_VALUES,&iscol));
  CHKERRQ(ISCreateStride(comm,n,lo,1,&isrow));     /* all rows */
  CHKERRQ(MatCreateSubMatrix(Bgt,isrow,iscol,MAT_INITIAL_MATRIX,Bgt_new));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)*Bgt_new,(PetscObject)Bgt,NULL));
  if (cg_new) {
    PERMON_ASSERT(cg,"cg vector specified");
    CHKERRQ(VecGetSubVector(cg,iscol,cg_new));
    CHKERRQ(FllopPetscObjectInheritName((PetscObject)*cg_new,(PetscObject)cg,NULL));
  }

  if (is_new) {
    *is_new = iscol;
  } else {
    CHKERRQ(ISDestroy(&iscol));
  }
  CHKERRQ(PetscFree(remove));
  CHKERRQ(PetscFree(idx));
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
  CHKERRQ(PetscObjectGetComm((PetscObject)Bgt,&comm));
  CHKERRQ(MPI_Comm_size(comm,&commsize));
  CHKERRQ(MPI_Comm_rank(comm,&rank));

  CHKERRQ(MatGetOwnershipRangesColumn(Bgt,&mg_arr));

  CHKERRQ(ISCreateStride(comm,0,0,0,&ispair[0]));

  for (r=0; r<commsize; r++)
  {
    jlog = mg_arr[r];
    jhig = mg_arr[r+1];
    mg = jhig - jlog;
    
    CHKERRQ(PetscMalloc(mg*sizeof(PetscBool),&remove));
    CHKERRQ(PetscMemzero(remove,mg*sizeof(PetscBool)));
    
    for (i=lo; i<hi; i++) {
      CHKERRQ(MatGetRow(Bdt,i,&ncolsd,&colsd,&valsd));
      k=0;
      for (jj=0; jj<ncolsd; jj++) {
        if (valsd[jj] > PETSC_MACHINE_EPSILON) k++;
        if (k>1) SETERRQ(comm,PETSC_ERR_PLIB,"more than one nonzero in Bd row %d",i);
      }
      CHKERRQ(MatRestoreRow(Bdt,i,&ncolsd,&colsd,&valsd));
      if (k) {
        CHKERRQ(MatGetRow(Bgt,i,&ncolsg,&colsg,&valsg));
        for (jj=0; jj<ncolsg; jj++) {
          if (valsg[jj] > PETSC_MACHINE_EPSILON) {
            j = colsg[jj];
            if (j>=jlog && j<jhig) remove[j-jlog] = PETSC_TRUE;
          }
        }
        CHKERRQ(MatRestoreRow(Bgt,i,&ncolsg,&colsg,&valsg));
      }
    }
    
    if (r == rank) {
      CHKERRQ(MPI_Reduce(MPI_IN_PLACE,remove,mg,MPIU_BOOL,MPI_LOR,r,comm));
      CHKERRQ(PetscMalloc(mg*sizeof(PetscInt),&idx));
      k=0;
      for (j=jlog; j<jhig; j++) {
        if (!remove[j-jlog]) {
          idx[k]=j;
          k++;
        }
      }
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,k,idx,PETSC_OWN_POINTER,&iscol_self));
    } else {
      CHKERRQ(MPI_Reduce(remove,NULL,mg,MPIU_BOOL,MPI_LOR,r,comm));
    }
    
    CHKERRQ(PetscFree(remove));
  }

  CHKERRQ(ISOnComm(iscol_self,comm,PETSC_COPY_VALUES,&iscol));
  CHKERRQ(ISDestroy(&iscol_self));

  CHKERRQ(MatGetOwnershipIS(Bgt,&isrow,NULL));
  CHKERRQ(MatCreateSubMatrix(Bgt,isrow,iscol,MAT_INITIAL_MATRIX,Bgt_new));
  CHKERRQ(FllopPetscObjectInheritName((PetscObject)*Bgt_new,(PetscObject)Bgt,NULL));
  if (cg_new) {
    PERMON_ASSERT(cg,"cg vector specified");
    CHKERRQ(VecGetSubVector(cg,iscol,cg_new));
    CHKERRQ(FllopPetscObjectInheritName((PetscObject)*cg_new,(PetscObject)cg,NULL));
  }

  if (is_new) {
    *is_new = iscol;
  } else {
    CHKERRQ(ISDestroy(&iscol));
  }
  PetscFunctionReturnI(0);
}
