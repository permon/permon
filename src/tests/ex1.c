
#include <permonqps.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;

  ierr = PermonInitialize(&argc,&args,(char *)0,(char *)0);if (ierr) return ierr;
  ierr = PermonFinalize();
  return ierr;
}

/*TEST
  test:
    nsize: {{1 2}}
TEST*/

