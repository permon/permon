#include <permonqps.h>
#include <permonmat.h>
#include <permonpc.h>
#include <permonksp.h>
#include <permon/private/permonimpl.h>

PetscBool PermonInitializeCalled = PETSC_FALSE;
PetscBool PermonBeganPetsc       = PETSC_FALSE;

static int    permon_one           = 1;
static char  *permon_executable    = (char *)"permon";
static char **permon_executablePtr = &permon_executable;

PetscClassId PERMON_CLASSID;
PERMON       permon;

#undef __FUNCT__
#define __FUNCT__ "PermonInitialize"
/*@
   PermonInitialize - Initializes PETSc (if not already initialized), init PERMON, register functions, etc.
   This function must be called in order to use the PERMON library!

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
-  file - [optional] PERMON database file, also checks in following order petsrc files from PetscInitialize, ~/.permonrc, ./permonrc, ./.permonrc and file;  use NULL to not check for code specific file

   Options Database Keys:
+  -skip_permonrc  - skip reading permonrc file
-  -options_table - print table of options

   Level: beginner

.seealso PermonFinalize()
@*/
PetscErrorCode PermonInitialize(int *argc, char ***args, const char file[], const char help[])
{
  PetscBool flg = PETSC_FALSE;
  char      pfile[PETSC_MAX_PATH_LEN];

  if (PermonInitializeCalled) {
    PetscCall(PetscInfo(0, "PERMON already initialized, skipping initialization.\n"));
    return (0);
  }

  if (!PetscInitializeCalled) {
    if (argc && args) {
      PetscCall(PetscInitialize(argc, args, file, help));
    } else {
      PetscCall(PetscInitialize(&permon_one, &permon_executablePtr, file, help));
    }
    PermonBeganPetsc = PETSC_TRUE;
    PetscCall(PetscInfo(0, "PERMON successfully started PETSc.\n"));
  }

  if (!PetscInitializeCalled) {
    printf("Error initializing PETSc -- aborting.\n");
    exit(1);
  }

  PetscCall(PetscClassIdRegister("PERMON", &PERMON_CLASSID));
  PetscCall(PermonCreate(PETSC_COMM_WORLD, &permon));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-skip_permonrc", &flg, NULL));
  if (!flg) {
    PetscCall(PetscGetHomeDirectory(pfile, PETSC_MAX_PATH_LEN - 16));
    /* warning: assumes all processes have a home directory or none, but nothing in between */
    if (pfile[0]) {
      PetscCall(PetscStrlcat(pfile, "/.permonrc", sizeof(pfile)));
      PetscCall(PetscOptionsInsertFile(PETSC_COMM_WORLD, NULL, pfile, PETSC_FALSE));
    }
    PetscCall(PetscOptionsInsertFile(PETSC_COMM_WORLD, NULL, "permonrc", PETSC_FALSE));
    PetscCall(PetscOptionsInsertFile(PETSC_COMM_WORLD, NULL, ".permonrc", PETSC_FALSE));
    /* override by petsc options - permonrc currently takes the lowest precedence */
    PetscCall(PetscOptionsInsert(NULL, argc, args, file));
  } else {
    PetscCall(PetscInfo(permon, "skipping permonrc due to -skip_permonrc\n"));
  }

  PetscCall(PermonSetFromOptions());

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-options_view", &flg, NULL));
  if (!flg) PetscCall(PetscOptionsGetBool(NULL, NULL, "-options_table", &flg, NULL));
  if (flg) PetscCall(PetscOptionsView(NULL, PETSC_VIEWER_STDOUT_WORLD));

  /* register all PERMON implementations of PETSc classes */
  PetscCall(PermonMatRegisterAll());
  PetscCall(PermonPCRegisterAll());
  PetscCall(PermonKSPRegisterAll());

  PermonInitializeCalled = PETSC_TRUE;
  PetscCall(PetscInfo(permon, "PERMON successfully initialized.\n"));
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PermonFinalize"
/*@
   PermonFinalize - Permon cleanup, PetscFinalize() (if PERMON started petsc), etc.

   Level: beginner

.seealso PermonInitialize()
@*/
PetscErrorCode PermonFinalize()
{
  PetscFunctionBegin;
  if (!PermonInitializeCalled) { PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(PetscInfo(permon, "PermonFinalize() called\n"));
  PetscCall(PermonDestroy(&permon));

  if (PermonBeganPetsc) { PetscCall(PetscFinalize()); }
  PermonInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  /*
  PetscDLLibraryRegister_permon - This function is called when the dynamic library
  it is in is opened.

  This one registers all the QP methods in the libpermon.a library.
 */
  #undef __FUNCT__
  #define __FUNCT__ "PetscDLLibraryRegister_permon"
PetscErrorCode PetscDLLibraryRegister_permon()
{
  PetscFunctionBegin;
  PetscCall(QPPFInitializePackage());
  PetscCall(QPInitializePackage());
  PetscCall(QPSInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
