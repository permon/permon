static char help[] = "This example illustrates the use of FETI/TFETI with 2D/3D DMDA.\n\
It solves the constant coefficient Poisson problem or the Elasticity problem \n\
on a uniform grid of [0,cells_x] x [0,cells_y] x [0,cells_z]\n\n";

/* This example is an adapted version of PETSc KSP example ex71
* Contributed to PETSc by Wim Vanroose <wim@vanroo.se> */

#include <permonksp.h>
#include <petscksp.h>
#include <petscpc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmplex.h>

static PetscScalar poiss_1D_emat[] = {1.0000000000000000e+00, -1.0000000000000000e+00, -1.0000000000000000e+00, 1.0000000000000000e+00};
static PetscScalar poiss_2D_emat[] = {6.6666666666666674e-01,  -1.6666666666666666e-01, -1.6666666666666666e-01, -3.3333333333333337e-01, -1.6666666666666666e-01, 6.6666666666666674e-01,  -3.3333333333333337e-01, -1.6666666666666666e-01,
                                      -1.6666666666666666e-01, -3.3333333333333337e-01, 6.6666666666666674e-01,  -1.6666666666666666e-01, -3.3333333333333337e-01, -1.6666666666666666e-01, -1.6666666666666666e-01, 6.6666666666666674e-01};
static PetscScalar poiss_3D_emat[] = {3.3333333333333348e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -8.3333333333333343e-02, 0.0000000000000000e+00,  -8.3333333333333343e-02, -8.3333333333333343e-02, -8.3333333333333356e-02,
                                      0.0000000000000000e+00,  3.3333333333333337e-01,  -8.3333333333333343e-02, 0.0000000000000000e+00,  -8.3333333333333343e-02, 0.0000000000000000e+00,  -8.3333333333333356e-02, -8.3333333333333343e-02,
                                      0.0000000000000000e+00,  -8.3333333333333343e-02, 3.3333333333333337e-01,  0.0000000000000000e+00,  -8.3333333333333343e-02, -8.3333333333333356e-02, 0.0000000000000000e+00,  -8.3333333333333343e-02,
                                      -8.3333333333333343e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  3.3333333333333348e-01,  -8.3333333333333356e-02, -8.3333333333333343e-02, -8.3333333333333343e-02, 0.0000000000000000e+00,
                                      0.0000000000000000e+00,  -8.3333333333333343e-02, -8.3333333333333343e-02, -8.3333333333333356e-02, 3.3333333333333337e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -8.3333333333333343e-02,
                                      -8.3333333333333343e-02, 0.0000000000000000e+00,  -8.3333333333333356e-02, -8.3333333333333343e-02, 0.0000000000000000e+00,  3.3333333333333337e-01,  -8.3333333333333343e-02, 0.0000000000000000e+00,
                                      -8.3333333333333343e-02, -8.3333333333333356e-02, 0.0000000000000000e+00,  -8.3333333333333343e-02, 0.0000000000000000e+00,  -8.3333333333333343e-02, 3.3333333333333337e-01,  0.0000000000000000e+00,
                                      -8.3333333333333356e-02, -8.3333333333333343e-02, -8.3333333333333343e-02, 0.0000000000000000e+00,  -8.3333333333333343e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  3.3333333333333337e-01};
static PetscScalar elast_1D_emat[] = {3.0000000000000000e+00, -3.0000000000000000e+00, -3.0000000000000000e+00, 3.0000000000000000e+00};
static PetscScalar elast_2D_emat[] = {1.3333333333333335e+00,  5.0000000000000000e-01,  -8.3333333333333337e-01, 0.0000000000000000e+00,  1.6666666666666671e-01,  0.0000000000000000e+00,  -6.6666666666666674e-01, -5.0000000000000000e-01,
                                      5.0000000000000000e-01,  1.3333333333333335e+00,  0.0000000000000000e+00,  1.6666666666666671e-01,  0.0000000000000000e+00,  -8.3333333333333337e-01, -5.0000000000000000e-01, -6.6666666666666674e-01,
                                      -8.3333333333333337e-01, 0.0000000000000000e+00,  1.3333333333333335e+00,  -5.0000000000000000e-01, -6.6666666666666674e-01, 5.0000000000000000e-01,  1.6666666666666674e-01,  0.0000000000000000e+00,
                                      0.0000000000000000e+00,  1.6666666666666671e-01,  -5.0000000000000000e-01, 1.3333333333333335e+00,  5.0000000000000000e-01,  -6.6666666666666674e-01, 0.0000000000000000e+00,  -8.3333333333333337e-01,
                                      1.6666666666666671e-01,  0.0000000000000000e+00,  -6.6666666666666674e-01, 5.0000000000000000e-01,  1.3333333333333335e+00,  -5.0000000000000000e-01, -8.3333333333333337e-01, 0.0000000000000000e+00,
                                      0.0000000000000000e+00,  -8.3333333333333337e-01, 5.0000000000000000e-01,  -6.6666666666666674e-01, -5.0000000000000000e-01, 1.3333333333333335e+00,  0.0000000000000000e+00,  1.6666666666666674e-01,
                                      -6.6666666666666674e-01, -5.0000000000000000e-01, 1.6666666666666674e-01,  0.0000000000000000e+00,  -8.3333333333333337e-01, 0.0000000000000000e+00,  1.3333333333333335e+00,  5.0000000000000000e-01,
                                      -5.0000000000000000e-01, -6.6666666666666674e-01, 0.0000000000000000e+00,  -8.3333333333333337e-01, 0.0000000000000000e+00,  1.6666666666666674e-01,  5.0000000000000000e-01,  1.3333333333333335e+00};
static PetscScalar elast_3D_emat[] =
  {5.5555555555555558e-01,  1.6666666666666666e-01,  1.6666666666666666e-01,  -2.2222222222222232e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  1.1111111111111113e-01,  0.0000000000000000e+00,  8.3333333333333356e-02,
   -1.9444444444444442e-01, -1.6666666666666669e-01, 0.0000000000000000e+00,  1.1111111111111112e-01,  8.3333333333333356e-02,  0.0000000000000000e+00,  -1.9444444444444445e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01,
   -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -1.3888888888888887e-01, -8.3333333333333356e-02, -8.3333333333333356e-02, 1.6666666666666666e-01,  5.5555555555555558e-01,  1.6666666666666666e-01,
   0.0000000000000000e+00,  1.1111111111111113e-01,  8.3333333333333356e-02,  0.0000000000000000e+00,  -2.2222222222222232e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01, -1.9444444444444442e-01, 0.0000000000000000e+00,
   8.3333333333333356e-02,  1.1111111111111112e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -1.9444444444444445e-01, -1.6666666666666669e-01,
   -8.3333333333333356e-02, -1.3888888888888887e-01, -8.3333333333333356e-02, 1.6666666666666666e-01,  1.6666666666666666e-01,  5.5555555555555558e-01,  0.0000000000000000e+00,  8.3333333333333356e-02,  1.1111111111111112e-01,
   8.3333333333333356e-02,  0.0000000000000000e+00,  1.1111111111111112e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222229e-01,
   -1.6666666666666669e-01, 0.0000000000000000e+00,  -1.9444444444444445e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01, -1.9444444444444445e-01, -8.3333333333333356e-02, -8.3333333333333356e-02, -1.3888888888888887e-01,
   -2.2222222222222232e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  5.5555555555555558e-01,  -1.6666666666666666e-01, -1.6666666666666666e-01, -1.9444444444444442e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,
   1.1111111111111113e-01,  0.0000000000000000e+00,  -8.3333333333333356e-02, -1.9444444444444445e-01, 0.0000000000000000e+00,  1.6666666666666669e-01,  1.1111111111111113e-01,  -8.3333333333333356e-02, 0.0000000000000000e+00,
   -1.3888888888888887e-01, 8.3333333333333356e-02,  8.3333333333333356e-02,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.1111111111111113e-01,  8.3333333333333356e-02,
   -1.6666666666666666e-01, 5.5555555555555558e-01,  1.6666666666666669e-01,  1.6666666666666669e-01,  -1.9444444444444442e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222229e-01, 0.0000000000000000e+00,
   0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  -8.3333333333333356e-02, 1.1111111111111112e-01,  0.0000000000000000e+00,  8.3333333333333356e-02,  -1.3888888888888887e-01, -8.3333333333333356e-02,
   0.0000000000000000e+00,  -1.9444444444444448e-01, -1.6666666666666666e-01, 0.0000000000000000e+00,  8.3333333333333356e-02,  1.1111111111111112e-01,  -1.6666666666666666e-01, 1.6666666666666669e-01,  5.5555555555555558e-01,
   0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, -8.3333333333333356e-02, 0.0000000000000000e+00,  1.1111111111111112e-01,  1.6666666666666669e-01,  0.0000000000000000e+00,  -1.9444444444444445e-01,
   0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01, 8.3333333333333356e-02,  -8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00,  -1.6666666666666666e-01, -1.9444444444444448e-01,
   1.1111111111111113e-01,  0.0000000000000000e+00,  8.3333333333333356e-02,  -1.9444444444444442e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,  5.5555555555555569e-01,  -1.6666666666666666e-01, 1.6666666666666669e-01,
   -2.2222222222222229e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -1.3888888888888887e-01, 8.3333333333333356e-02,  -8.3333333333333356e-02,
   1.1111111111111112e-01,  -8.3333333333333343e-02, 0.0000000000000000e+00,  -1.9444444444444448e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01, 0.0000000000000000e+00,  -2.2222222222222232e-01, 0.0000000000000000e+00,
   1.6666666666666669e-01,  -1.9444444444444442e-01, 0.0000000000000000e+00,  -1.6666666666666666e-01, 5.5555555555555558e-01,  -1.6666666666666669e-01, 0.0000000000000000e+00,  1.1111111111111113e-01,  -8.3333333333333343e-02,
   0.0000000000000000e+00,  -1.9444444444444445e-01, 1.6666666666666669e-01,  8.3333333333333356e-02,  -1.3888888888888887e-01, 8.3333333333333356e-02,  -8.3333333333333343e-02, 1.1111111111111113e-01,  0.0000000000000000e+00,
   0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  8.3333333333333356e-02,  0.0000000000000000e+00,  1.1111111111111112e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02,
   1.6666666666666669e-01,  -1.6666666666666669e-01, 5.5555555555555558e-01,  0.0000000000000000e+00,  -8.3333333333333343e-02, 1.1111111111111112e-01,  0.0000000000000000e+00,  1.6666666666666669e-01,  -1.9444444444444445e-01,
   -8.3333333333333356e-02, 8.3333333333333356e-02,  -1.3888888888888887e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01, -1.6666666666666669e-01, 0.0000000000000000e+00,  -1.9444444444444448e-01,
   -1.9444444444444442e-01, -1.6666666666666669e-01, 0.0000000000000000e+00,  1.1111111111111113e-01,  0.0000000000000000e+00,  -8.3333333333333356e-02, -2.2222222222222229e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,
   5.5555555555555558e-01,  1.6666666666666669e-01,  -1.6666666666666666e-01, -1.3888888888888887e-01, -8.3333333333333356e-02, 8.3333333333333356e-02,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,
   -1.9444444444444448e-01, 0.0000000000000000e+00,  1.6666666666666669e-01,  1.1111111111111112e-01,  8.3333333333333343e-02,  0.0000000000000000e+00,  -1.6666666666666669e-01, -1.9444444444444442e-01, 0.0000000000000000e+00,
   0.0000000000000000e+00,  -2.2222222222222229e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  1.1111111111111113e-01,  -8.3333333333333343e-02, 1.6666666666666669e-01,  5.5555555555555558e-01,  -1.6666666666666669e-01,
   -8.3333333333333356e-02, -1.3888888888888887e-01, 8.3333333333333356e-02,  0.0000000000000000e+00,  -1.9444444444444448e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,
   8.3333333333333343e-02,  1.1111111111111112e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, -8.3333333333333356e-02, 0.0000000000000000e+00,  1.1111111111111112e-01,
   0.0000000000000000e+00,  -8.3333333333333343e-02, 1.1111111111111112e-01,  -1.6666666666666666e-01, -1.6666666666666669e-01, 5.5555555555555558e-01,  8.3333333333333356e-02,  8.3333333333333356e-02,  -1.3888888888888887e-01,
   0.0000000000000000e+00,  1.6666666666666669e-01,  -1.9444444444444448e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,  -1.9444444444444448e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01,
   1.1111111111111112e-01,  8.3333333333333356e-02,  0.0000000000000000e+00,  -1.9444444444444445e-01, 0.0000000000000000e+00,  1.6666666666666669e-01,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,
   -1.3888888888888887e-01, -8.3333333333333356e-02, 8.3333333333333356e-02,  5.5555555555555569e-01,  1.6666666666666669e-01,  -1.6666666666666669e-01, -2.2222222222222227e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,
   1.1111111111111112e-01,  0.0000000000000000e+00,  -8.3333333333333343e-02, -1.9444444444444448e-01, -1.6666666666666669e-01, 0.0000000000000000e+00,  8.3333333333333356e-02,  1.1111111111111112e-01,  0.0000000000000000e+00,
   0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -1.9444444444444445e-01, 1.6666666666666669e-01,  -8.3333333333333356e-02, -1.3888888888888887e-01, 8.3333333333333356e-02,
   1.6666666666666669e-01,  5.5555555555555558e-01,  -1.6666666666666669e-01, 0.0000000000000000e+00,  1.1111111111111112e-01,  -8.3333333333333343e-02, 0.0000000000000000e+00,  -2.2222222222222227e-01, 0.0000000000000000e+00,
   -1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222229e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,  -1.9444444444444445e-01,
   0.0000000000000000e+00,  1.6666666666666669e-01,  -1.9444444444444445e-01, 8.3333333333333356e-02,  8.3333333333333356e-02,  -1.3888888888888887e-01, -1.6666666666666669e-01, -1.6666666666666669e-01, 5.5555555555555558e-01,
   0.0000000000000000e+00,  -8.3333333333333343e-02, 1.1111111111111113e-01,  -8.3333333333333343e-02, 0.0000000000000000e+00,  1.1111111111111113e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02,
   -1.9444444444444445e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01, 1.1111111111111113e-01,  -8.3333333333333356e-02, 0.0000000000000000e+00,  -1.3888888888888887e-01, 8.3333333333333356e-02,  -8.3333333333333356e-02,
   -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  5.5555555555555558e-01,  -1.6666666666666669e-01, 1.6666666666666669e-01,
   -1.9444444444444448e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,  1.1111111111111112e-01,  0.0000000000000000e+00,  8.3333333333333343e-02,  0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,
   -8.3333333333333356e-02, 1.1111111111111112e-01,  0.0000000000000000e+00,  8.3333333333333356e-02,  -1.3888888888888887e-01, 8.3333333333333356e-02,  0.0000000000000000e+00,  -1.9444444444444448e-01, 1.6666666666666669e-01,
   0.0000000000000000e+00,  1.1111111111111112e-01,  -8.3333333333333343e-02, -1.6666666666666669e-01, 5.5555555555555558e-01,  -1.6666666666666666e-01, 1.6666666666666669e-01,  -1.9444444444444448e-01, 0.0000000000000000e+00,
   0.0000000000000000e+00,  -2.2222222222222227e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01, 0.0000000000000000e+00,  -1.9444444444444445e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01,
   -8.3333333333333356e-02, 8.3333333333333356e-02,  -1.3888888888888887e-01, 0.0000000000000000e+00,  1.6666666666666669e-01,  -1.9444444444444448e-01, 0.0000000000000000e+00,  -8.3333333333333343e-02, 1.1111111111111113e-01,
   1.6666666666666669e-01,  -1.6666666666666666e-01, 5.5555555555555558e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, 8.3333333333333343e-02,  0.0000000000000000e+00,  1.1111111111111113e-01,
   -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -1.3888888888888887e-01, 8.3333333333333356e-02,  8.3333333333333356e-02,  1.1111111111111112e-01,  -8.3333333333333343e-02, 0.0000000000000000e+00,
   -1.9444444444444448e-01, 0.0000000000000000e+00,  1.6666666666666669e-01,  1.1111111111111112e-01,  0.0000000000000000e+00,  -8.3333333333333343e-02, -1.9444444444444448e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,
   5.5555555555555558e-01,  -1.6666666666666669e-01, -1.6666666666666669e-01, -2.2222222222222227e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  -1.9444444444444445e-01, -1.6666666666666669e-01,
   8.3333333333333356e-02,  -1.3888888888888887e-01, -8.3333333333333356e-02, -8.3333333333333343e-02, 1.1111111111111113e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,
   0.0000000000000000e+00,  -2.2222222222222227e-01, 0.0000000000000000e+00,  1.6666666666666669e-01,  -1.9444444444444448e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01, 5.5555555555555558e-01,  1.6666666666666669e-01,
   0.0000000000000000e+00,  1.1111111111111112e-01,  8.3333333333333343e-02,  0.0000000000000000e+00,  -1.6666666666666669e-01, -1.9444444444444445e-01, 8.3333333333333356e-02,  -8.3333333333333356e-02, -1.3888888888888887e-01,
   0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01, 1.6666666666666669e-01,  0.0000000000000000e+00,  -1.9444444444444448e-01, -8.3333333333333343e-02, 0.0000000000000000e+00,  1.1111111111111113e-01,
   0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02, -1.6666666666666669e-01, 1.6666666666666669e-01,  5.5555555555555558e-01,  0.0000000000000000e+00,  8.3333333333333343e-02,  1.1111111111111113e-01,
   -1.3888888888888887e-01, -8.3333333333333356e-02, -8.3333333333333356e-02, -2.7777777777777769e-02, 0.0000000000000000e+00,  0.0000000000000000e+00,  -1.9444444444444448e-01, 0.0000000000000000e+00,  -1.6666666666666669e-01,
   1.1111111111111112e-01,  8.3333333333333343e-02,  0.0000000000000000e+00,  -1.9444444444444448e-01, -1.6666666666666669e-01, 0.0000000000000000e+00,  1.1111111111111112e-01,  0.0000000000000000e+00,  8.3333333333333343e-02,
   -2.2222222222222227e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  5.5555555555555558e-01,  1.6666666666666669e-01,  1.6666666666666669e-01,  -8.3333333333333356e-02, -1.3888888888888887e-01, -8.3333333333333356e-02,
   0.0000000000000000e+00,  -1.9444444444444448e-01, -1.6666666666666666e-01, 0.0000000000000000e+00,  -2.7777777777777769e-02, 0.0000000000000000e+00,  8.3333333333333343e-02,  1.1111111111111112e-01,  0.0000000000000000e+00,
   -1.6666666666666669e-01, -1.9444444444444448e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  1.1111111111111112e-01,  8.3333333333333343e-02,
   1.6666666666666669e-01,  5.5555555555555558e-01,  1.6666666666666669e-01,  -8.3333333333333356e-02, -8.3333333333333356e-02, -1.3888888888888887e-01, 0.0000000000000000e+00,  -1.6666666666666666e-01, -1.9444444444444448e-01,
   -1.6666666666666669e-01, 0.0000000000000000e+00,  -1.9444444444444448e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.2222222222222227e-01, 0.0000000000000000e+00,  0.0000000000000000e+00,  -2.7777777777777769e-02,
   8.3333333333333343e-02,  0.0000000000000000e+00,  1.1111111111111113e-01,  0.0000000000000000e+00,  8.3333333333333343e-02,  1.1111111111111113e-01,  1.6666666666666669e-01,  1.6666666666666669e-01,  5.5555555555555558e-01};

typedef enum {
  PDE_POISSON,
  PDE_ELASTICITY
} PDEType;

typedef struct {
  PDEType      pde;
  PetscInt     dim;
  PetscInt     dof;
  PetscInt     cells[3];
  PetscBool    useglobal;
  PetscBool    dirbc;
  PetscBool    per[3];
  PetscBool    test;
  PetscScalar *elemMat;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char *pdeTypes[2] = {"Poisson", "Elasticity"};
  PetscInt    n, pde;

  PetscFunctionBeginUser;
  options->pde       = PDE_POISSON;
  options->elemMat   = NULL;
  options->dim       = 2;
  options->cells[0]  = 8;
  options->cells[1]  = 6;
  options->cells[2]  = 4;
  options->useglobal = PETSC_FALSE;
  options->dirbc     = PETSC_TRUE;
  options->test      = PETSC_FALSE;
  options->per[0]    = PETSC_FALSE;
  options->per[1]    = PETSC_FALSE;
  options->per[2]    = PETSC_FALSE;

  PetscOptionsBegin(comm, NULL, "Problem Options", NULL);
  pde = options->pde;
  PetscCall(PetscOptionsEList("-pde_type", "The PDE type", __FILE__, pdeTypes, 2, pdeTypes[options->pde], &pde, NULL));
  options->pde = (PDEType)pde;
  PetscCall(PetscOptionsInt("-dim", "The topological mesh dimension", __FILE__, options->dim, &options->dim, NULL));
  PetscCall(PetscOptionsIntArray("-cells", "The mesh division", __FILE__, options->cells, (n = 3, &n), NULL));
  PetscCall(PetscOptionsBoolArray("-periodicity", "The mesh periodicity", __FILE__, options->per, (n = 3, &n), NULL));
  PetscCall(PetscOptionsBool("-use_global", "Test MatSetValues", __FILE__, options->useglobal, &options->useglobal, NULL));
  PetscCall(PetscOptionsBool("-dirichlet", "Use dirichlet BC", __FILE__, options->dirbc, &options->dirbc, NULL));
  PetscCall(PetscOptionsBool("-test_assembly", "Test MATIS assembly", __FILE__, options->test, &options->test, NULL));
  PetscOptionsEnd();

  for (n = options->dim; n < 3; n++) options->cells[n] = 0;
  if (options->per[0]) options->dirbc = PETSC_FALSE;

  /* element matrices */
  switch (options->pde) {
  case PDE_ELASTICITY:
    options->dof = options->dim;
    switch (options->dim) {
    case 1:
      options->elemMat = elast_1D_emat;
      break;
    case 2:
      options->elemMat = elast_2D_emat;
      break;
    case 3:
      options->elemMat = elast_3D_emat;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, options->dim);
    }
    break;
  case PDE_POISSON:
    options->dof = 1;
    switch (options->dim) {
    case 1:
      options->elemMat = poiss_1D_emat;
      break;
    case 2:
      options->elemMat = poiss_2D_emat;
      break;
    case 3:
      options->elemMat = poiss_3D_emat;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, options->dim);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported PDE %" PetscInt_FMT, options->pde);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  AppCtx user;
  KSP    ksp;
  //PC                     pc;
  Mat                    A;
  DM                     da;
  Vec                    x, b, xcoor, xcoorl;
  ISLocalToGlobalMapping map;
  MatNullSpace           nullsp = NULL;
  PetscInt               i;
  PetscInt               nel, nen;     /* Number of elements & element nodes */
  const PetscInt        *e_loc;        /* Local indices of element nodes (in local element order) */
  PetscInt              *e_glo = NULL; /* Global indices of element nodes (in local element order) */
  PetscBool              ismatis;
  PetscErrorCode         ierr;

  PetscCall(PermonInitialize(&argc, &args, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  switch (user.dim) {
  case 3:
    ierr = DMDACreate3d(PETSC_COMM_WORLD, user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, user.per[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, user.per[2] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, user.cells[0] + 1, user.cells[1] + 1, user.cells[2] + 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        user.dof, 1, NULL, NULL, NULL, &da);
    PetscCall(ierr);
    break;
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD, user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, user.per[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, user.cells[0] + 1, user.cells[1] + 1, PETSC_DECIDE, PETSC_DECIDE, user.dof, 1, NULL, NULL, &da);
    PetscCall(ierr);
    break;
  case 1:
    ierr = DMDACreate1d(PETSC_COMM_WORLD, user.per[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE, user.cells[0] + 1, user.dof, 1, NULL, &da);
    PetscCall(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, user.dim);
  }
  PetscCall(DMSetMatType(da, MATIS));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetUp(da));
  {
    PetscInt M, N, P;
    PetscCall(DMDAGetInfo(da, 0, &M, &N, &P, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    switch (user.dim) {
    case 3:
      user.cells[2] = P - 1;
    case 2:
      user.cells[1] = N - 1;
    case 1:
      user.cells[0] = M - 1;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, user.dim);
    }
  }
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0 * user.cells[0], 0.0, 1.0 * user.cells[1], 0.0, 1.0 * user.cells[2]));
  PetscCall(DMGetCoordinates(da, &xcoor));

  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(DMGetLocalToGlobalMapping(da, &map));
  PetscCall(DMDAGetElements(da, &nel, &nen, &e_loc));
  if (user.useglobal) {
    PetscCall(PetscMalloc1(nel * nen, &e_glo));
    PetscCall(ISLocalToGlobalMappingApplyBlock(map, nen * nel, e_loc, e_glo));
  }

  /* we reorder the indices since the element matrices are given in lexicographic order,
     whereas the elements indices returned by DMDAGetElements follow the usual FEM ordering
     i.e., element matrices     DMDA ordering
               2---3              3---2
              /   /              /   /
             0---1              0---1
  */
  for (i = 0; i < nel; ++i) {
    PetscInt ord[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    PetscInt j, idxs[8];

    PetscCheck(nen <= 8, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Not coded");
    if (!e_glo) {
      for (j = 0; j < nen; j++) idxs[j] = e_loc[i * nen + ord[j]];
      PetscCall(MatSetValuesBlockedLocal(A, nen, idxs, nen, idxs, user.elemMat, ADD_VALUES));
    } else {
      for (j = 0; j < nen; j++) idxs[j] = e_glo[i * nen + ord[j]];
      PetscCall(MatSetValuesBlocked(A, nen, idxs, nen, idxs, user.elemMat, ADD_VALUES));
    }
  }
  PetscCall(DMDARestoreElements(da, &nel, &nen, &e_loc));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Boundary conditions */
  if (user.dirbc) { /* fix one side of DMDA */
    Vec          nat, glob;
    IS           zero;
    PetscScalar *vals;
    PetscInt     n, *idx, j, st;

    n = PetscGlobalRank ? 0 : (user.cells[1] + 1) * (user.cells[2] + 1);
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, 0, user.cells[0] + 1, &zero));
    if (user.dof > 1) { /* zero all components */
      const PetscInt *idx;
      IS              bzero;

      PetscCall(ISGetIndices(zero, (const PetscInt **)&idx));
      PetscCall(ISCreateBlock(PETSC_COMM_WORLD, user.dof, n, idx, PETSC_COPY_VALUES, &bzero));
      PetscCall(ISRestoreIndices(zero, (const PetscInt **)&idx));
      PetscCall(ISDestroy(&zero));
      zero = bzero;
    }
    /* map indices from natural to global */
    PetscCall(DMDACreateNaturalVector(da, &nat));
    PetscCall(ISGetLocalSize(zero, &n));
    PetscCall(PetscMalloc1(n, &vals));
    for (i = 0; i < n; i++) vals[i] = 1.0;
    PetscCall(ISGetIndices(zero, (const PetscInt **)&idx));
    PetscCall(VecSetValues(nat, n, idx, vals, INSERT_VALUES));
    PetscCall(ISRestoreIndices(zero, (const PetscInt **)&idx));
    PetscCall(PetscFree(vals));
    PetscCall(VecAssemblyBegin(nat));
    PetscCall(VecAssemblyEnd(nat));
    PetscCall(DMCreateGlobalVector(da, &glob));
    PetscCall(DMDANaturalToGlobalBegin(da, nat, INSERT_VALUES, glob));
    PetscCall(DMDANaturalToGlobalEnd(da, nat, INSERT_VALUES, glob));
    PetscCall(VecDestroy(&nat));
    PetscCall(ISDestroy(&zero));
    PetscCall(VecGetLocalSize(glob, &n));
    PetscCall(PetscMalloc1(n, &idx));
    PetscCall(VecGetOwnershipRange(glob, &st, NULL));
    PetscCall(VecGetArray(glob, &vals));
    for (i = 0, j = 0; i < n; i++)
      if (PetscRealPart(vals[i]) == 1.0) idx[j++] = i + st;
    PetscCall(VecRestoreArray(glob, &vals));
    PetscCall(VecDestroy(&glob));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, j, idx, PETSC_OWN_POINTER, &zero));
    PetscCall(MatZeroRowsColumnsIS(A, zero, 1.0, NULL, NULL));
    PetscCall(ISDestroy(&zero));
  } else {
    switch (user.pde) {
    case PDE_POISSON:
      PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullsp));
      break;
    case PDE_ELASTICITY:
      PetscCall(MatNullSpaceCreateRigidBody(xcoor, &nullsp));
      break;
    }
    /* with periodic BC and Elasticity, just the displacements are in the nullspace
       this is no harm since we eliminate all the components of the rhs */
    PetscCall(MatSetNullSpace(A, nullsp));
  }

  if (user.test) {
    Mat AA;

    PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &AA));
    PetscCall(MatViewFromOptions(AA, NULL, "-assembled_view"));
    PetscCall(MatDestroy(&AA));
  }

  /* Attach near null space for elasticity */
  if (user.pde == PDE_ELASTICITY) {
    MatNullSpace nearnullsp;

    PetscCall(MatNullSpaceCreateRigidBody(xcoor, &nearnullsp));
    PetscCall(MatSetNearNullSpace(A, nearnullsp));
    PetscCall(MatNullSpaceDestroy(&nearnullsp));
  }

  /* we may want to use MG for the local solvers: attach local nearnullspace to the local matrices */
  PetscCall(DMGetCoordinatesLocal(da, &xcoorl));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (ismatis) {
    MatNullSpace lnullsp = NULL;
    Mat          lA;

    PetscCall(MatISGetLocalMat(A, &lA));
    if (user.pde == PDE_ELASTICITY) {
      Vec                    lc;
      ISLocalToGlobalMapping l2l;
      IS                     is;
      const PetscScalar     *a;
      const PetscInt        *idxs;
      PetscInt               n, bs;

      /* when using a DMDA, the local matrices have an additional local-to-local map
         that maps from the DA local ordering to the ordering induced by the elements */
      PetscCall(MatCreateVecs(lA, &lc, NULL));
      PetscCall(MatGetLocalToGlobalMapping(lA, &l2l, NULL));
      PetscCall(VecSetLocalToGlobalMapping(lc, l2l));
      PetscCall(VecSetOption(lc, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
      PetscCall(VecGetLocalSize(xcoorl, &n));
      PetscCall(VecGetBlockSize(xcoorl, &bs));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, n / bs, 0, 1, &is));
      PetscCall(ISGetIndices(is, &idxs));
      PetscCall(VecGetArrayRead(xcoorl, &a));
      PetscCall(VecSetValuesBlockedLocal(lc, n / bs, idxs, a, INSERT_VALUES));
      PetscCall(VecAssemblyBegin(lc));
      PetscCall(VecAssemblyEnd(lc));
      PetscCall(VecRestoreArrayRead(xcoorl, &a));
      PetscCall(ISRestoreIndices(is, &idxs));
      PetscCall(ISDestroy(&is));
      PetscCall(MatNullSpaceCreateRigidBody(lc, &lnullsp));
      PetscCall(VecDestroy(&lc));
    } else if (user.pde == PDE_POISSON) {
      PetscCall(MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, NULL, &lnullsp));
    }
    PetscCall(MatSetNearNullSpace(lA, lnullsp));
    PetscCall(MatNullSpaceDestroy(&lnullsp));
    PetscCall(MatISRestoreLocalMat(A, &lA));
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPFETI));
  //PetscCall(KSPGetPC(ksp,&pc));
  //PetscCall(PCSetType(pc,PCBDDC));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  PetscCall(DMGetGlobalVector(da, &x));
  PetscCall(DMGetGlobalVector(da, &b));
  PetscCall(VecSet(b, 1.0));
  //PetscCall(VecSetRandom(b,NULL));
  if (nullsp) { PetscCall(MatNullSpaceRemove(nullsp, b)); }
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(DMRestoreGlobalVector(da, &x));
  PetscCall(DMRestoreGlobalVector(da, &b));

  /* cleanup */
  PetscCall(PetscFree(e_glo));
  PetscCall(MatNullSpaceDestroy(&nullsp));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PermonFinalize());
  return 0;
}

/*TEST
  build:
    requires: mumps
  testset:
    args: -qps_view_convergence -qp_chain_view_kkt
    filter: grep -e CONVERGED -e "r ="
    test:
      nsize: 6
      suffix: 1
      args: -pde_type Poisson -cells 7,8,9 -dim 3 -feti_gluing_type {{nonred full orth}separate output}
    test:
      nsize: 7
      suffix: 2
      args: -pde_type Elasticity -dim 3 -qps_rtol 1e-6 -dual_pc_dual_type {{none lumped}separate output}
 TEST*/
