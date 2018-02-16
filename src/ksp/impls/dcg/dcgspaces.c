#include "dcgimpl.h"

#if defined(HAVE_SLEPC)
#include <slepceps.h>
#include <slepcbv.h>
#endif

const char *const KSPDCGSpaceTypes[] = {
  "haar",
  "jh",
  "slepc",
  "user",
  "DCGSpaceType",
  "DCG_SPACE_",
  0
};

#undef __FUNCT__
#define __FUNCT__ "KSPDCGDeflationSpaceCreateJacketHaar"
static PetscErrorCode KSPDCGDeflationSpaceCreateJacketHaar(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscBool jacket,Mat *H)
{
  PetscErrorCode ierr;
  Mat defl;
  PetscInt i,j,ilo,ihi,alloc=2,*Iidx;
  PetscReal val,*row;

  PetscFunctionBegin;
  if (jacket) alloc = 3;
  ierr = PetscMalloc1(alloc,&row);CHKERRQ(ierr);
  ierr = PetscMalloc1(alloc,&Iidx);CHKERRQ(ierr);

  val = 1./pow(2,0.5);
  row[0] = val;
  row[1] = val;

  /* TODO pass A instead of KSP? */
  ierr = MatCreate(comm,&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetUp(defl);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(defl,alloc,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(defl,alloc,NULL,alloc,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  
  ierr = MatGetOwnershipRange(defl,&ilo,&ihi);CHKERRQ(ierr);
  for (i=0; i<2; i++) Iidx[i] = i+ilo*2;
  if (jacket && ihi==M) ihi -=2;
  if (ihi<ilo) SETERRQ1(comm,PETSC_ERR_ARG_WRONG,"To many cores to assemble Jacket Haar matrix with %d rows",M);
  for (i=ilo; i<ihi; i++) {
    ierr = MatSetValues(defl,1,&i,2,Iidx,row,INSERT_VALUES);CHKERRQ(ierr);
    for (j=0; j<2; j++) Iidx[j] += 2;
  }
  if (jacket && ihi == M-2) {
    for (i=0; i<3; i++) Iidx[i] = i+ilo*2;
    row[0] = 0.5; row[1] = 0.5; row[2] = val;
    ierr = MatSetValues(defl,1,&ihi,3,Iidx,row,INSERT_VALUES);CHKERRQ(ierr);
    ihi += 1;
    row[2] = -row[2];
    ierr = MatSetValues(defl,1,&ihi,3,Iidx,row,INSERT_VALUES);CHKERRQ(ierr);
  }
    
  ierr = MatAssemblyBegin(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscFree(row);CHKERRQ(ierr);
  ierr = PetscFree(Iidx);CHKERRQ(ierr);
  *H = defl;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGGetDeflationSpaceHaar"
PetscErrorCode KSPDCGGetDeflationSpaceHaar(KSP ksp,Mat *W,PetscInt size)
{
  PetscErrorCode ierr;
  Mat A,defl;
  PetscInt i,j,len,ilo,ihi,*Iidx,m,M;
  PetscReal *col,val;

  PetscFunctionBegin;
  /* Haar basis wavelet, level=size */
  len = pow(2,size);
  ierr = PetscMalloc1(len,&col);CHKERRQ(ierr);
  ierr = PetscMalloc1(len,&Iidx);CHKERRQ(ierr);
  val = 1./pow(2,size/2.);
  for (i=0; i<len; i++) col[i] = val;

  /* TODO pass A instead of KSP? */
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* NOTE: Get Pmat instead? */
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,m,PETSC_DECIDE,M,ceil(M/(float)len));CHKERRQ(ierr);
  ierr = MatSetUp(defl);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(defl,size,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(defl,size,NULL,size,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  
  ierr = MatGetOwnershipRangeColumn(defl,&ilo,&ihi);CHKERRQ(ierr);
  for (i=0; i<len; i++) Iidx[i] = i+ilo*len;
  if (M%len && ihi == (int)ceil(M/(float)len)) ihi -= 1;
  for (i=ilo; i<ihi; i++) {
    ierr = MatSetValues(defl,len,Iidx,1,&i,col,INSERT_VALUES);CHKERRQ(ierr);
    for (j=0; j<len; j++) Iidx[j] += len;
  }
  if (M%len && ihi+1 == ceil(M/(float)len)) {
    len = M%len;
    val = 1./pow(pow(2,len),0.5);
    for (i=0; i<len; i++) col[i] = val;
    ierr = MatSetValues(defl,len,Iidx,1,&ihi,col,INSERT_VALUES);CHKERRQ(ierr);
  }
    
  ierr = MatAssemblyBegin(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscFree(col);CHKERRQ(ierr);
  ierr = PetscFree(Iidx);CHKERRQ(ierr);
  *W = defl;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGGetDeflationSpacetHaarOrig"
PetscErrorCode KSPDCGGetDeflationSpacetHaarOrig(KSP ksp,Mat *W,PetscInt size)
{
  PetscErrorCode ierr;
  Mat A,defl;
  PetscInt i,j,len,start=2,ilo,ihi,*Iidx,m,M;
  PetscReal *col,val;

  PetscFunctionBegin;
  /* Haar basis wavelet, level=size */
  len = pow(2,size);
  ierr = PetscMalloc1(len,&col);CHKERRQ(ierr);
  ierr = PetscMalloc1(len,&Iidx);CHKERRQ(ierr);
  val = 1./pow(2,size/2.);
  /* mat G
  col[0] = val;
  col[1] = -val;
  while (2*start <= len) {
    for (i=start; i<2*start; i++) {
      col[i] = -col[i-start];
    }
    start = 2*start;
  }
  */
  for (i=0; i<len; i++) {
      col[i] = val;
  }

  /* TODO pass A instead of KSP? */
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* NOTE: Get Pmat instead? */
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,m,PETSC_DECIDE,M,M/len);CHKERRQ(ierr);
  ierr = MatSetUp(defl);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(defl,size,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(defl,size,NULL,size,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(defl,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  
  /* TODO change to transpose assembly, add MAT_IGNORE_OFF_PROC_ENTRIES */
  ierr = MatGetOwnershipRangeColumn(defl,&ilo,&ihi);CHKERRQ(ierr);
  for (i=0; i<len; i++) Iidx[i] = i+ilo*len;
  for (i=ilo; i<ihi; i++) {
    ierr = MatSetValues(defl,len,Iidx,1,&i,col,INSERT_VALUES);CHKERRQ(ierr);
    for (j=0; j<len; j++) Iidx[j] += len;
  }
  ierr = MatAssemblyBegin(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(defl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscFree(col);CHKERRQ(ierr);
  ierr = PetscFree(Iidx);CHKERRQ(ierr);
  *W = defl;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGGetDeflationSpaceJacketHaar"
PetscErrorCode KSPDCGGetDeflationSpaceJacketHaar(KSP ksp,Mat *W,PetscInt size)
{
  PetscErrorCode ierr;
  Mat A,*H,defl;
  PetscInt i,j,len,start=2,ilo,ihi,*Iidx,m,M,Mdefl,Ndefl;
  PetscBool jh;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&H);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* NOTE: Get Pmat instead? */
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  Mdefl = M;
  Ndefl = M;
  for (i=0; i<size; i++) {
    if (Mdefl%2)  {
      jh=PETSC_TRUE;
      Mdefl = Mdefl/2 +1;
    } else {
      jh=PETSC_FALSE;
      Mdefl = Mdefl/2;
    }
    printf("mdefl %d x ndefl %d\n",Mdefl,Ndefl);
    ierr = KSPDCGDeflationSpaceCreateJacketHaar(comm,PETSC_DECIDE,m,Mdefl,Ndefl,jh,&H[i]);CHKERRQ(ierr);
    ierr = MatGetLocalSize(H[i],&m,NULL);CHKERRQ(ierr);
    Ndefl = Mdefl;
  }
  //ierr = MatCreateProd(comm,size,H,&defl);CHKERRQ(ierr);
  //ierr = MatCreateComposite(comm,size,H,&defl);CHKERRQ(ierr);
  //ierr = MatCompositeSetType(defl,MAT_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
  /* TODO allow implicit */
  //ierr = MatCompositeMerge(defl);CHKERRQ(ierr);
  Mat newmat;
  defl = H[0];
  for (i=0; i<size-1; i++) {
    ierr = MatMatMult(H[i+1],defl,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr);
    ierr = MatDestroy(&defl);CHKERRQ(ierr);
    defl = newmat ;
  }

  ierr = MatTranspose(defl,MAT_INITIAL_MATRIX,W);CHKERRQ(ierr);
  
  ierr = MatDestroy(&defl);CHKERRQ(ierr);
  for (i=1; i<size; i++) {
    ierr = MatDestroy(&H[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGGetDeflationSpaceSLEPc"
PetscErrorCode KSPDCGGetDeflationSpaceSLEPc(KSP ksp,Mat *W,PetscInt size)
{
#if defined(HAVE_SLEPC)
  PetscErrorCode ierr;
  Mat A,defl;
  Vec vec;
  EPS eps;
  PetscScalar *data;
  PetscInt i,nconv,m,M;
  PetscBool slepcinit;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = SlepcInitialized(&slepcinit);CHKERRQ(ierr);
  if (!slepcinit) {
    ierr = SlepcInitialize(NULL,NULL,(char*)0,(char*)0);CHKERRQ(ierr);
    slepcinit = PETSC_TRUE;
  }
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* NOTE: Get Pmat instead? */
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  ierr = EPSCreate(comm,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr); /* Implemented only for CG */
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetDimensions(eps,size,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  if (!nconv) SETERRQ(comm,PETSC_ERR_CONV_FAILED,"SLEPc: Number of converged eigenpairs is 0");
  ierr = MatCreateVecs(A,NULL,&vec);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(m*nconv,&data);CHKERRQ(ierr);
  /* TODO check that eigenvalue is not 0 -> vec is not in Ker A */
  for (i=0; i<nconv; i++) {
    ierr = VecPlaceArray(vec,&data[i*m]);CHKERRQ(ierr);
    ierr = EPSGetEigenvector(eps,i,vec,NULL);CHKERRQ(ierr);
    ierr = VecResetArray(vec);CHKERRQ(ierr);
  }
  ierr = MatCreateDense(comm,m,PETSC_DECIDE,M,nconv,data,&defl);CHKERRQ(ierr);

  //ierr = EPSGetBV(eps,&bv);CHKERRQ(ierr);
  //ierr = BVCreateMat(bv,&defl);CHKERRQ(ierr);
  *W = defl;

  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  if (slepcinit) ierr = SlepcFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_CONV_FAILED,"Not compiled with SLEPc support (call make HAVE_SLEPC)");
#endif
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGComputeDeflationSpace"
PetscErrorCode KSPDCGComputeDeflationSpace(KSP ksp)
{
  PetscErrorCode ierr;
  Mat defl;
  KSP_DCG *cg = (KSP_DCG*)ksp->data;

  /* TODO valid header */
  PetscFunctionBegin;
  if (cg->spacesize < 1) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"Wrong DCG Space size specified: %d",cg->spacesize);
  switch (cg->spacetype) {
    case DCG_SPACE_HAAR:
      ierr = KSPDCGGetDeflationSpaceHaar(ksp,&defl,cg->spacesize);CHKERRQ(ierr);break;
    case DCG_SPACE_JACKET_HAAR:
      ierr = KSPDCGGetDeflationSpaceJacketHaar(ksp,&defl,cg->spacesize);CHKERRQ(ierr);break;
    case DCG_SPACE_SLEPC:
      ierr = KSPDCGGetDeflationSpaceSLEPc(ksp,&defl,cg->spacesize);CHKERRQ(ierr);break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"Wrong DCG Space Type specified");
  }
  
  ierr = KSPDCGSetDeflationSpace(ksp,defl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




