
#include <permonqps.h>
#include <permonmat.h>
#include <permonpc.h>
#include <permonksp.h>
#include <permon/private/permonimpl.h>

PetscBool FllopInitializeCalled = PETSC_FALSE;
PetscBool FllopBeganPetsc = PETSC_FALSE;

static int    fllop_one=1;
static char*  fllop_executable = (char*)"fllop";
static char** fllop_executablePtr = &fllop_executable;

PetscClassId  FLLOP_CLASSID;
FLLOP   fllop;

#undef __FUNCT__
#define __FUNCT__ "PermonInitialize"
/*@
   FllopInitialize - Initializes PETSc (if not already initialized), init FLLOP, register functions, etc.
   This function must be called in order to use the FLLOP library!

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
-  file - [optional] FLLOP database file, also checks in following order petsrc files from PetscInitialize, ~/.flloprc, ./flloprc, ./.flloprc and file;  use NULL to not check for code specific file 

   Options Database Keys:
+  -skip_flloprc  - skip reading flloprc file
-  -options_table - print table of options

   Level: beginner

.seealso FllopFinalize()
@*/
PetscErrorCode PermonInitialize(int *argc, char ***args, const char file[], const char help[])
{
  PetscErrorCode ierr;
  PetscBool flg=PETSC_FALSE;
  char pfile[PETSC_MAX_PATH_LEN];

  if (FllopInitializeCalled) {
    CHKERRQ(PetscInfo(0,"FLLOP already initialized, skipping initialization.\n"));
    return(0);
  }

  if (!PetscInitializeCalled) {
    if (argc&&args) {
      CHKERRQ(PetscInitialize(argc,args,file,help));
    } else {
      CHKERRQ(PetscInitialize(&fllop_one,&fllop_executablePtr,file,help));
    }
    FllopBeganPetsc=PETSC_TRUE;
    CHKERRQ(PetscInfo(0,"FLLOP successfully started PETSc.\n"));
  }
  
  if (!PetscInitializeCalled) {
    printf("Error initializing PETSc -- aborting.\n");
    exit(1);
  }

  CHKERRQ(PetscClassIdRegister("FLLOP",&FLLOP_CLASSID));
  CHKERRQ(FllopCreate(PETSC_COMM_WORLD,&fllop));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-skip_flloprc",&flg,NULL));
  if (!flg) {
    CHKERRQ(PetscGetHomeDirectory(pfile,PETSC_MAX_PATH_LEN-16));
    /* warning: assumes all processes have a home directory or none, but nothing in between */
    if (pfile[0]) {
      CHKERRQ(PetscStrcat(pfile,"/.flloprc"));
      CHKERRQ(PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL,pfile,PETSC_FALSE));
    }
    CHKERRQ(PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL, "flloprc", PETSC_FALSE));
    CHKERRQ(PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL, ".flloprc", PETSC_FALSE));
    /* override by petsc options - flloprc currently takes the lowest precedence */
    CHKERRQ(PetscOptionsInsert(NULL,argc,args,file));
  } else {
    CHKERRQ(PetscInfo(fllop,"skipping flloprc due to -skip_flloprc\n"));
  }

  CHKERRQ(FllopSetFromOptions());

  flg = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-options_view",&flg,NULL));
  if (!flg)CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-options_table",&flg,NULL));
  if (flg) CHKERRQ(PetscOptionsView(NULL,PETSC_VIEWER_STDOUT_WORLD));

  /* register all PERMON implementations of PETSc classes */
  CHKERRQ(PermonMatRegisterAll());
  CHKERRQ(FllopPCRegisterAll());
  CHKERRQ(PermonKSPRegisterAll());
  
  FllopInitializeCalled = PETSC_TRUE;
  CHKERRQ(PetscInfo(fllop,"FLLOP successfully initialized.\n"));
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PermonFinalize"
/*@
   FllopFinalize - Fllop cleanup, PetscFinalize() (if FLLOP started petsc), etc.

   Level: beginner

.seealso FllopInitialize()
@*/
PetscErrorCode PermonFinalize()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!FllopInitializeCalled) {
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscInfo(fllop,"FllopFinalize() called\n"));  
  CHKERRQ(FllopDestroy(&fllop));

  if (FllopBeganPetsc) {
    PetscFinalize();
  } 
  FllopInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister_fllop - This function is called when the dynamic library
  it is in is opened.

  This one registers all the QP methods in the libfllop.a library.
 */
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_permon"
PetscErrorCode PetscDLLibraryRegister_permon()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(QPPFInitializePackage());
  CHKERRQ(QPInitializePackage());
  CHKERRQ(QPSInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
