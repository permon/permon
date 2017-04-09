
#include <fllopqps.h>
#include <fllopmat.h>
#include <flloppc.h>
#include <permon/private/fllopimpl.h>

PetscBool FllopInitializeCalled = PETSC_FALSE;
PetscBool FllopBeganPetsc = PETSC_FALSE;

static int    fllop_one=1;
static char*  fllop_executable = (char*)"fllop";
static char** fllop_executablePtr = &fllop_executable;

PetscClassId  FLLOP_CLASSID;
FLLOP   fllop;

#undef __FUNCT__
#define __FUNCT__ "FllopInitialize"
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
PetscErrorCode FllopInitialize(int *argc, char ***args, const char file[])
{
  PetscErrorCode ierr;
  PetscBool flg=PETSC_FALSE;
  char pfile[PETSC_MAX_PATH_LEN];

  if (FllopInitializeCalled) {
    ierr=PetscInfo(0,"FLLOP already initialized, skipping initialization.\n");CHKERRQ(ierr);
    return(0);
  }

  if (!PetscInitializeCalled) {
    if (argc&&args) {
      ierr = PetscInitialize(argc,args,file,(char*)0);CHKERRQ(ierr);
    } else {
      ierr = PetscInitialize(&fllop_one,&fllop_executablePtr,file,(char*)0);CHKERRQ(ierr);
    }
    FllopBeganPetsc=PETSC_TRUE;
    ierr=PetscInfo(0,"FLLOP successfully started PETSc.\n");CHKERRQ(ierr);
  }
  
  if (!PetscInitializeCalled) {
    printf("Error initializing PETSc -- aborting.\n");
    exit(1);
  }

  ierr = PetscClassIdRegister("FLLOP",&FLLOP_CLASSID);CHKERRQ(ierr);
  ierr = FllopCreate(PETSC_COMM_WORLD,&fllop);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-skip_flloprc",&flg,NULL);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscGetHomeDirectory(pfile,PETSC_MAX_PATH_LEN-16);CHKERRQ(ierr);
    /* warning: assumes all processes have a home directory or none, but nothing in between */
    if (pfile[0]) {
      ierr = PetscStrcat(pfile,"/.flloprc");CHKERRQ(ierr);
      ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL,pfile,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL, "flloprc", PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL, ".flloprc", PETSC_FALSE);CHKERRQ(ierr);
    /* override by petsc options - flloprc currently takes the lowest precedence */
    ierr = PetscOptionsInsert(NULL,argc,args,file);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(fllop,"skipping flloprc due to -skip_flloprc\n");CHKERRQ(ierr);
  }

  ierr = FllopSetFromOptions();CHKERRQ(ierr);

  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-options_view",&flg,NULL);CHKERRQ(ierr);
  if (!flg){ ierr = PetscOptionsGetBool(NULL,NULL,"-options_table",&flg,NULL);CHKERRQ(ierr); }
  if (flg) { ierr = PetscOptionsView(NULL,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  /* register all PERMON implementations of PETSc classes */
  ierr = FllopMatRegisterAll();CHKERRQ(ierr);
  ierr = FllopPCRegisterAll();CHKERRQ(ierr);
  
  FllopInitializeCalled = PETSC_TRUE;
  ierr = PetscInfo(fllop,"FLLOP successfully initialized.\n");CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FllopFinalize"
/*@
   FllopFinalize - Fllop cleanup, PetscFinalize() (if FLLOP started petsc), etc.

   Level: beginner

.seealso FllopInitialize()
@*/
PetscErrorCode FllopFinalize()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!FllopInitializeCalled) {
    PetscFunctionReturn(0);
  }
  ierr = PetscInfo(fllop,"FllopFinalize() called\n");CHKERRQ(ierr);  
  ierr = FllopDestroy(&fllop);CHKERRQ(ierr);

  if (FllopBeganPetsc) {
    PetscFinalize();
  } 
  FllopInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister_fllop - This function is called when the dynamic library
  it is in is opened.

  This one registers all the QP methods in the libfllop.a library.
 */
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_fllop"
PetscErrorCode PetscDLLibraryRegister_fllop()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QPPFInitializePackage();CHKERRQ(ierr);
  ierr = QPInitializePackage();CHKERRQ(ierr);
  ierr = QPSInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
