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
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_(comm);
  } else {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(ksp,1,viewer,2);
  }

  PetscCall(KSPGetTolerances(ksp, &rtol,&dtol,&atol,&maxit ));
  PetscCall(KSPGetType(ksp, &ksptype));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &pctype));
  PetscCall(PCFactorGetMatSolverType(pc, &pcpkg));
  PetscCall(PCGetOperators(pc, &mat, NULL));
  PetscCall(MatGetType(mat, &mattype));
  
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ksp, viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer,"KSPType:          %s\n",ksptype));
  PetscCall(PetscViewerASCIIPrintf(viewer,"PCType:           %s\n",pctype));
  PetscCall(PetscViewerASCIIPrintf(viewer,"MatSolverType:    %s\n",pcpkg));
  PetscCall(PetscViewerASCIIPrintf(viewer,"MatType:          %s\n",mattype));
  PetscCall(PetscViewerASCIIPrintf(viewer,"(rtol, dtol, atol, maxit) = (%.1e, %.1e, %.1e, %d)\n",rtol,dtol,atol,maxit));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

