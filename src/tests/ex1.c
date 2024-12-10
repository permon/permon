#include <permonqps.h>

int main(int argc,char **args)
{
  PetscCall(PermonInitialize(&argc,&args,(char *)0,(char *)0));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  test:
    nsize: {{1 2}}
TEST*/

