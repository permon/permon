#include "dcgimpl.h"

#if defined(HAVE_SLEPC)
#include <slepceps.h>
#include <slepcbv.h>
#endif

const char *const KSPDCGSpaceTypes[] = {
  "haar",
  "slepc",
  "user",
  "DCGSpaceType",
  "DCG_SPACE_",
  0
};

#undef __FUNCT__
#define __FUNCT__ "KSPDCGGeDeflationSpacetHaar"
PetscErrorCode KSPDCGGeDeflationSpacetHaar(KSP ksp,Mat *W,PetscInt size)
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
#define __FUNCT__ "KSPDCGGeDeflationSpacetHaarExtended"
PetscErrorCode KSPDCGGeDeflationSpacetHaarExtended(KSP ksp,Mat *W,PetscInt size)
{
  PetscErrorCode ierr;
  Mat A,defl;
  PetscInt i,j,len,start=2,ilo,ihi,*Iidx,m,M,Mdefl;
  PetscReal *col,val;
  PetscBool jh=PETSC_FALSE;

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
  Mdefl = M/len;
  if (Mdefl*len < M) {
    Mdefl += 1; /* M is odd -> extra row for Jacket-Haar */
    size += 1; /* TODO improve prealloc */

  }
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* NOTE: Get Pmat instead? */
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&defl);CHKERRQ(ierr);
  ierr = MatSetSizes(defl,m,PETSC_DECIDE,M,Mdefl);CHKERRQ(ierr);
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
#define __FUNCT__ "KSPDCGGeDeflationSpacetSLEPc"
PetscErrorCode KSPDCGGeDeflationSpaceSLEPc(KSP ksp,Mat *W,PetscInt size)
{
#if defined(HAVE_SLEPC)
  PetscErrorCode ierr;
  Mat A,defl;
  EPS eps;
  BV bv;
  PetscInt nconv;
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
  
  ierr = EPSGetBV(eps,&bv);CHKERRQ(ierr);
  ierr = BVCreateMat(bv,&defl);CHKERRQ(ierr);
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
      ierr = KSPDCGGeDeflationSpacetHaar(ksp,&defl,cg->spacesize);CHKERRQ(ierr);break;
    case DCG_SPACE_SLEPC:
      ierr = KSPDCGGeDeflationSpaceSLEPc(ksp,&defl,cg->spacesize);CHKERRQ(ierr);break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONG,"Wrong DCG Space Type specified");
  }
  
  ierr = KSPDCGSetDeflationSpace(ksp,defl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




