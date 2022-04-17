
#include <permonqps.h>

int main(int argc,char **args)
{
  CHKERRQ(PermonInitialize(&argc,&args,(char *)0,(char *)0));
  CHKERRQ(PermonFinalize());
  return 0;
}

/*TEST
  test:
    nsize: {{1 2}}
TEST*/

