#if !defined(__FLLOPPETSCRETRO_H)
#define	__FLLOPPETSCRETRO_H
#include <petscsys.h>

#if PETSC_VERSION_MAJOR!=3 || (PETSC_VERSION_MAJOR==3 && (PETSC_VERSION_MINOR<5 || PETSC_VERSION_MINOR>7))
#error "FLLOP requires PETSc version 3.5 - 3.7"
#endif

/* ----------------------- */
/* petsc <= 3.6 compatibility */
/* ----------------------- */
#if PETSC_VERSION_MINOR<=6
/* include headers that use the functions overriden with the macros below so that they use the original functions */
#include <petsctao.h>
#include <petsclog.h>
#if PETSC_VERSION_MINOR<=5
#include <petsc-private/matimpl.h>
#else
#include <petsc/private/matimpl.h>
#endif

#define MAT_INPLACE_MATRIX MAT_REUSE_MATRIX
#define PetscOptionItems PetscOptions

#define PetscObjectProcessOptionsHandlers(options,obj)                PetscObjectProcessOptionsHandlers(obj)
#define PetscOptionsClearValue(options,iname)                         PetscOptionsClearValue(iname)
#define PetscOptionsGetBool(options,pre,name,ivalue,set)              PetscOptionsGetBool(pre,name,ivalue,set)
#define PetscOptionsGetEnum(options,pre,opt,list,value,set)           PetscOptionsGetEnum(pre,opt,list,value,set)
#define PetscOptionsGetInt(options,pre,name,ivalue,set)               PetscOptionsGetInt(pre,name,ivalue,set)
#define PetscOptionsGetReal(options,pre,name,dvalue,set)              PetscOptionsGetReal(pre,name,dvalue,set)
#define PetscOptionsGetString(options,pre,name,string,len,set)        PetscOptionsGetString(pre,name,string,len,set)
#define PetscOptionsGetStringArray(options,pre,name,strings,nmax,set) PetscOptionsGetStringArray(pre,name,strings,nmax,set)
#define PetscOptionsHasName(options,pre,name,set)                     PetscOptionsHasName(pre,name,set)
#define PetscOptionsInsert(options,argc,args,file)                    PetscOptionsInsert(argc,args,file)
#define PetscOptionsInsertFile(comm,options,file,require)             PetscOptionsInsertFile(comm,file,require)
#define PetscOptionsInsertString(options,in_str)                      PetscOptionsInsertString(in_str)
#define PetscOptionsSetValue(options,iname,value)                     PetscOptionsSetValue(iname,value)
#define PetscOptionsView(options,viewer)                              PetscOptionsView(viewer)
#define PetscViewerGetSubViewer(viewer,comm,outviewer)                PetscViewerGetSubcomm(viewer,comm,outviewer)
#define PetscViewerRestoreSubViewer(viewer,comm,outviewer)            PetscViewerRestoreSubcomm(viewer,comm,outviewer)
#define TaoSetTolerances(tao,gatol,grtol,gttol)                       TaoSetTolerances(tao,1e-50,1e-50,gatol,grtol,gttol)

#undef  PetscPreLoadBegin
#define PetscPreLoadBegin(flag,name) \
do {\
  PetscBool      PetscPreLoading = flag;\
  int            PetscPreLoadMax,PetscPreLoadIt;\
  PetscLogStage  _stageNum;\
  PetscErrorCode _3_ierr; \
  _3_ierr = PetscOptionsGetBool(NULL,NULL,"-preload",&PetscPreLoading,NULL);CHKERRQ(_3_ierr);\
  PetscPreLoadMax = (int)(PetscPreLoading);\
  PetscPreLoadingUsed = PetscPreLoading ? PETSC_TRUE : PetscPreLoadingUsed;\
  for (PetscPreLoadIt=0; PetscPreLoadIt<=PetscPreLoadMax; PetscPreLoadIt++) {\
    PetscPreLoadingOn = PetscPreLoading;\
    _3_ierr = PetscBarrier(NULL);CHKERRQ(_3_ierr);\
    if (PetscPreLoadIt>0) {\
      _3_ierr = PetscLogStageGetId(name,&_stageNum);CHKERRQ(_3_ierr);\
    } else {\
      _3_ierr = PetscLogStageRegister(name,&_stageNum);CHKERRQ(_3_ierr); \
    }\
    _3_ierr = PetscLogStageSetActive(_stageNum,(PetscBool)(!PetscPreLoadMax || PetscPreLoadIt));\
    _3_ierr = PetscLogStagePush(_stageNum);CHKERRQ(_3_ierr);
#endif


/* ----------------------- */
/* petsc <= 3.5 compatibility */
/* ----------------------- */
#if PETSC_VERSION_MINOR<=5
#undef MatSolverPackage
typedef const char* MatSolverPackage;
/* fix MatGetFactorAvailable for version <= 3.5 */
FLLOP_EXTERN PetscErrorCode MatGetFactorAvailable_permonfix(Mat mat, const MatSolverPackage type,MatFactorType ftype,PetscBool  *flg);
#define MatGetFactorAvailable(mat,type,ftype,flg) MatGetFactorAvailable_permonfix(mat,type,ftype,flg)
PETSC_EXTERN PetscOptionsObjectType PetscOptionsObject;
PETSC_STATIC_INLINE PetscErrorCode MatCreateVecs(Mat mat,Vec *x,Vec *y) {return MatGetVecs(mat,x,y);}
PETSC_STATIC_INLINE PetscErrorCode MatCreateRedundantMatrix(Mat mat,PetscInt nsubcomm,MPI_Comm subcomm,MatReuse reuse,Mat *matredundant) {return MatGetRedundantMatrix(mat,nsubcomm,subcomm,reuse,matredundant);}
#define PetscOptionsHead(obj,str) PetscOptionsHead(str)

#endif

#endif
