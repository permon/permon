#include <permonksp.h>
#include <permon/private/permonimpl.h>

#undef __FUNCT__
#define __FUNCT__ "KSPViewBriefInfo"
PetscErrorCode KSPViewBriefInfo(KSP ksp, PetscViewer viewer)
{
  KSPType                 ksptype;
  PCType                  pctype;
  MatSolverType           pcpkg;
  MatType                 mattype;
  MPI_Comm                comm;
  PC                      pc;
  Mat                     mat;
  PetscReal               rtol,dtol,atol;
  PetscInt                maxit;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)ksp, &comm));
  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_(comm);
  } else {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(ksp,1,viewer,2);
  }

  CHKERRQ(KSPGetTolerances(ksp, &rtol,&dtol,&atol,&maxit ));
  CHKERRQ(KSPGetType(ksp, &ksptype));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCGetType(pc, &pctype));
  CHKERRQ(PCFactorGetMatSolverType(pc, &pcpkg));
  CHKERRQ(PCGetOperators(pc, &mat, NULL));
  CHKERRQ(MatGetType(mat, &mattype));
  
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)ksp, viewer));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"KSPType:          %s\n",ksptype));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"PCType:           %s\n",pctype));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"MatSolverType:    %s\n",pcpkg));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"MatType:          %s\n",mattype));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"(rtol, dtol, atol, maxit) = (%.1e, %.1e, %.1e, %d)\n",rtol,dtol,atol,maxit));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

