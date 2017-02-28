#include <fllopksp.h>
#include <private/fllopimpl.h>

#undef __FUNCT__
#define __FUNCT__ "KSPViewBriefInfo"
PetscErrorCode KSPViewBriefInfo(KSP ksp, PetscViewer viewer)
{
  KSPType                 ksptype;
  PCType                  pctype;
  const MatSolverPackage  pcpkg;
  MatType                 mattype;
  MPI_Comm                comm;
  PC                      pc;
  Mat                     mat;
  PetscReal               rtol,dtol,atol;
  PetscInt                maxit;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  TRY( PetscObjectGetComm((PetscObject)ksp, &comm) );
  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_(comm);
  } else {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(ksp,1,viewer,2);
  }

  TRY( KSPGetTolerances(ksp, &rtol,&dtol,&atol,&maxit ) );
  TRY( KSPGetType(ksp, &ksptype) );
  TRY( KSPGetPC(ksp, &pc) );
  TRY( PCGetType(pc, &pctype) );
  TRY( PCFactorGetMatSolverPackage(pc, &pcpkg) );
  TRY( PCGetOperators(pc, &mat, NULL) );
  TRY( MatGetType(mat, &mattype) );
  
  TRY( PetscObjectPrintClassNamePrefixType((PetscObject)ksp, viewer) );
  TRY( PetscViewerASCIIPushTab(viewer) );
  TRY( PetscViewerASCIIPrintf(viewer,"KSPType:          %s\n",ksptype) );
  TRY( PetscViewerASCIIPrintf(viewer,"PCType:           %s\n",pctype) );
  TRY( PetscViewerASCIIPrintf(viewer,"MatSolverPackage: %s\n",pcpkg) );
  TRY( PetscViewerASCIIPrintf(viewer,"MatType:          %s\n",mattype) );
  TRY( PetscViewerASCIIPrintf(viewer,"(rtol, dtol, atol, maxit) = (%.1e, %.1e, %.1e, %d)\n",rtol,dtol,atol,maxit) );
  TRY( PetscViewerASCIIPopTab(viewer) );
  PetscFunctionReturn(0);
}

