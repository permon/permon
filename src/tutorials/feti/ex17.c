static char help[] = "Linear elasticity in 2d and 3d with finite elements.\n\
We solve the elasticity problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

/*
  adapted from PETSc snes/tutorials/ex17.c
*/
/*
* mpir -n 6 ./ex17 -displacement_petscspace_degree 1 -dm_refine 3 -ksp_monitor -dm_view hdf5:sol.h5 -displacement_view hdf5:sol.h5::append -sol_type elas_uniform_strain -ksp_type cg -ksp_converged_reason -pc_type none -dim 2 -ksp_type feti -dm_mat_type is -qp_chain_view_kkt -log_view       -petscpartitioner_type parmetis -petscpartitioner_view_graph -options_left*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>
#include <petscsf.h> /* For SplitFaces() */
#include <permonqps.h>
#include <permonksp.h>

typedef enum {SOL_VLAP_QUADRATIC, SOL_ELAS_QUADRATIC, SOL_VLAP_TRIG, SOL_ELAS_TRIG, SOL_ELAS_AXIAL_DISP, SOL_ELAS_UNIFORM_STRAIN, NUM_SOLUTION_TYPES} SolutionType;
const char *solutionTypes[NUM_SOLUTION_TYPES+1] = {"vlap_quad", "elas_quad", "vlap_trig", "elas_trig", "elas_axial_disp", "elas_uniform_strain", "unknown"};

typedef struct {
  /* Domain and mesh definition */
  char         dmType[256]; /* DM type for the solve */
  PetscInt     dim;         /* The topological mesh dimension */
  PetscBool    simplex;     /* Simplicial mesh */
  PetscInt     cells[3];    /* The initial domain division */
  PetscInt     numSplitFaces;/*  */
  PetscInt     numDupVertices;/*  */
  PetscBool    shear;       /* Shear the domain */
  PetscBool    solverksp;   /* Use KSP as solver */
  PetscBool    solverqps;    /* Use QPS as solver */
  /* Problem definition */
  SolutionType solType;     /* Type of exact solution */
  /* Solver definition */
  PetscBool    useNearNullspace; /* Use the rigid body modes as a near nullspace for AMG */
  PetscReal *fracCoord;
  PetscInt *oldpoints,*newpoints;
  PetscInt nFrac;
  PetscInt n;
  PetscInt x_n;
  PetscInt y_n;
  Vec       coeff;
  PetscReal *x;
  PetscReal *y;
  PetscReal *xPoints;
  Mat Bineq;
  Vec cineq;
  PetscSubcomm psubcomm;
  PetscMPIInt sizeL;
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static PetscErrorCode cnst(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 4.0;
  return 0;
}

static PetscErrorCode quadratic_2d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0];
  u[1] = x[1]*x[1] - 2.0*x[0]*x[1];
  return 0;
}

static PetscErrorCode quadratic_3d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0];
  u[1] = x[1]*x[1] - 2.0*x[0]*x[1];
  u[2] = x[2]*x[2] - 2.0*x[1]*x[2];
  return 0;
}

static void f0_push_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[d] = 0.0;
  f0[dim-1] = 2.;
  //f0[dim-2] = .5;
  //f0[dim-2] = -2.;
}

/*
  u = x^2
  v = y^2 - 2xy
  Delta <u,v> - f = <2, 2> - <2, 2>

  u = x^2
  v = y^2 - 2xy
  w = z^2 - 2yz
  Delta <u,v,w> - f = <2, 2, 2> - <2, 2, 2>
*/
static void f0_vlap_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[d] += 2.0;
}

/*
  u = x^2
  v = y^2 - 2xy
  \varepsilon = / 2x     -y    \
                \ -y   2y - 2x /
  Tr(\varepsilon) = div u = 2y
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j (2y) + 2\mu < 2-1, 2 >
    = \lambda < 0, 2 > + \mu < 2, 4 >

  u = x^2
  v = y^2 - 2xy
  w = z^2 - 2yz
  \varepsilon = / 2x     -y       0   \
                | -y   2y - 2x   -z   |
                \  0     -z    2z - 2y/
  Tr(\varepsilon) = div u = 2z
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j (2z) + 2\mu < 2-1, 2-1, 2 >
    = \lambda < 0, 0, 2 > + \mu < 2, 2, 4 >
*/
static void f0_elas_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal mu     = 1.0;
  const PetscReal lambda = 1.0;
  PetscInt        d;

  for (d = 0; d < dim-1; ++d) f0[d] += 2.0*mu;
  f0[dim-1] += 2.0*lambda + 4.0*mu;
}

static PetscErrorCode trig_2d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(2.0*PETSC_PI*x[0]);
  u[1] = PetscSinReal(2.0*PETSC_PI*x[1]) - 2.0*x[0]*x[1];
  return 0;
}

static PetscErrorCode trig_3d_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(2.0*PETSC_PI*x[0]);
  u[1] = PetscSinReal(2.0*PETSC_PI*x[1]) - 2.0*x[0]*x[1];
  u[2] = PetscSinReal(2.0*PETSC_PI*x[2]) - 2.0*x[1]*x[2];
  return 0;
}

/*
  u = sin(2 pi x)
  v = sin(2 pi y) - 2xy
  Delta <u,v> - f = <-4 pi^2 u, -4 pi^2 v> - <-4 pi^2 sin(2 pi x), -4 pi^2 sin(2 pi y)>

  u = sin(2 pi x)
  v = sin(2 pi y) - 2xy
  w = sin(2 pi z) - 2yz
  Delta <u,v,2> - f = <-4 pi^2 u, -4 pi^2 v, -4 pi^2 w> - <-4 pi^2 sin(2 pi x), -4 pi^2 sin(2 pi y), -4 pi^2 sin(2 pi z)>
*/
static void f0_vlap_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[d] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
}

/*
  u = sin(2 pi x)
  v = sin(2 pi y) - 2xy
  \varepsilon = / 2 pi cos(2 pi x)             -y        \
                \      -y          2 pi cos(2 pi y) - 2x /
  Tr(\varepsilon) = div u = 2 pi (cos(2 pi x) + cos(2 pi y)) - 2 x
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j 2 pi (cos(2 pi x) + cos(2 pi y)) + 2\mu < -4 pi^2 sin(2 pi x) - 1, -4 pi^2 sin(2 pi y) >
    = \lambda < -4 pi^2 sin(2 pi x) - 2, -4 pi^2 sin(2 pi y) > + \mu < -8 pi^2 sin(2 pi x) - 2, -8 pi^2 sin(2 pi y) >

  u = sin(2 pi x)
  v = sin(2 pi y) - 2xy
  w = sin(2 pi z) - 2yz
  \varepsilon = / 2 pi cos(2 pi x)            -y                     0         \
                |         -y       2 pi cos(2 pi y) - 2x            -z         |
                \          0                  -z         2 pi cos(2 pi z) - 2y /
  Tr(\varepsilon) = div u = 2 pi (cos(2 pi x) + cos(2 pi y) + cos(2 pi z)) - 2 x - 2 y
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j (2 pi (cos(2 pi x) + cos(2 pi y) + cos(2 pi z)) - 2 x - 2 y) + 2\mu < -4 pi^2 sin(2 pi x) - 1, -4 pi^2 sin(2 pi y) - 1, -4 pi^2 sin(2 pi z) >
    = \lambda < -4 pi^2 sin(2 pi x) - 2, -4 pi^2 sin(2 pi y) - 2, -4 pi^2 sin(2 pi z) > + 2\mu < -4 pi^2 sin(2 pi x) - 1, -4 pi^2 sin(2 pi y) - 1, -4 pi^2 sin(2 pi z) >
*/
static void f0_elas_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal mu     = 1.0;
  const PetscReal lambda = 1.0;
  const PetscReal fact   = 4.0*PetscSqr(PETSC_PI);
  PetscInt        d;

  for (d = 0; d < dim; ++d) f0[d] += -(2.0*mu + lambda) * fact*PetscSinReal(2.0*PETSC_PI*x[d]) - (d < dim-1 ? 2.0*(mu + lambda) : 0.0);
}

static PetscErrorCode axial_disp_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal mu     = 1.0;
  const PetscReal lambda = 1.0;
  const PetscReal N      = 1.0;
  PetscInt d;

  u[0] = (3.*lambda*lambda + 8.*lambda*mu + 4*mu*mu)/(4*mu*(3*lambda*lambda + 5.*lambda*mu + 2*mu*mu))*N*x[0];
  u[1] = -0.25*lambda/mu/(lambda+mu)*N*x[1];
  for (d = 2; d < dim; ++d) u[d] = 0.0;
  return 0;
}

/*
  We will pull/push on the right side of a block of linearly elastic material. The uniform traction conditions on the
  right side of the box will result in a uniform strain along x and y. The Neumann BC is given by

     n_i \sigma_{ij} = t_i

  u = (1/(2\mu) - 1) x
  v = -y
  f = 0
  t = <4\mu/\lambda (\lambda + \mu), 0>
  \varepsilon = / 1/(2\mu) - 1   0 \
                \ 0             -1 /
  Tr(\varepsilon) = div u = 1/(2\mu) - 2
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j (1/(2\mu) - 2) + 2\mu < 0, 0 >
    = \lambda < 0, 0 > + \mu < 0, 0 > = 0
  NBC =  <1,0> . <4\mu/\lambda (\lambda + \mu), 0> = 4\mu/\lambda (\lambda + \mu)

  u = x - 1/2
  v = 0
  w = 0
  \varepsilon = / x  0  0 \
                | 0  0  0 |
                \ 0  0  0 /
  Tr(\varepsilon) = div u = x
  div \sigma = \partial_i \lambda \delta_{ij} \varepsilon_{kk} + \partial_i 2\mu\varepsilon_{ij}
    = \lambda \partial_j x + 2\mu < 1, 0, 0 >
    = \lambda < 1, 0, 0 > + \mu < 2, 0, 0 >
*/
static void f0_elas_axial_disp_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal N = -1.0;

  f0[0] = N;
}

static PetscErrorCode uniform_strain_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal eps_xx = 0.1;
  const PetscReal eps_xy = 0.3;
  const PetscReal eps_yy = 0.25;
  PetscInt d;

  u[0] = eps_xx*x[0] + eps_xy*x[1];
  u[1] = eps_xy*x[0] + eps_yy*x[1];
  for (d = 2; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static void f1_vlap_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt Nc = dim;
  PetscInt       c, d;

  for (c = 0; c < Nc; ++c) for (d = 0; d < dim; ++d) f1[c*dim+d] += u_x[c*dim+d];
}

static void f1_elas_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt  Nc     = dim;
  const PetscReal mu     = 1.0;
  const PetscReal lambda = 1.0;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] += mu*(u_x[c*dim+d] + u_x[d*dim+c]);
      f1[c*dim+c] += lambda*u_x[d*dim+d];
    }
  }
}

static void g3_vlap_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt Nc = dim;
  PetscInt       c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc + c)*dim + d)*dim + d] = 1.0;
    }
  }
}

/*
  \partial_df \phi_fc g_{fc,gc,df,dg} \partial_dg \phi_gc

  \partial_df \phi_fc \lambda \delta_{fc,df} \sum_gc \partial_dg \phi_gc \delta_{gc,dg}
  = \partial_fc \phi_fc \sum_gc \partial_gc \phi_gc
*/
static void g3_elas_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscInt  Nc     = dim;
  const PetscReal mu     = 1.0;
  const PetscReal lambda = 1.0;
  PetscInt        c, d;
  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc + c)*dim + d)*dim + d] += mu;
      g3[((c*Nc + d)*dim + d)*dim + c] += mu;
      g3[((c*Nc + d)*dim + c)*dim + d] += lambda;
    }
  }
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3, sol;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->cells[0]         = 1;
  options->cells[1]         = 1;
  options->cells[2]         = 1;
  options->simplex          = PETSC_TRUE;
  options->shear            = PETSC_FALSE;
  options->solverksp        = PETSC_TRUE;
  options->solverqps        = PETSC_TRUE;
  options->solType          = SOL_VLAP_QUADRATIC;
  options->useNearNullspace = PETSC_TRUE;
  options->n                = 10;
  options->x_n              = 3;
  ierr = PetscStrncpy(options->dmType, DMPLEX, 256);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Linear Elasticity Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex17.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex17.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex17.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-shear", "Shear the domain", "ex17.c", options->shear, &options->shear, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_ksp", "Use KSP as solver", "ex17.c", options->solverksp, &options->solverksp, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_qps", "Use KSP as solver", "ex17.c", options->solverqps, &options->solverqps, NULL);CHKERRQ(ierr);
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of exact solution", "ex17.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsBool("-near_nullspace", "Use the rigid body modes as an AMG near nullspace", "ex17.c", options->useNearNullspace, &options->useNearNullspace, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type", "Convert DMPlex to another format", "ex17.c", DMList, options->dmType, options->dmType, 256, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(options->x_n,&options->x);CHKERRQ(ierr);
  ierr = PetscMalloc1(options->x_n,&options->y);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-x", "x coords to fit the slope separation interface", "ex17.c", options->x, &options->x_n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-y", "y coords to fit the slope separation interface", "ex17.c", options->y, &options->x_n, &flg);CHKERRQ(ierr);
  if (!flg) options->x_n = 0;
  ierr = PetscOptionsInt("-n", "number of points on slope separation interface in initial mesh", "ex17.c", options->n, &options->n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateCubeBoundary(DM dm, const PetscReal lower[], const PetscReal upper[], const PetscInt faces[])
{
  PetscInt       vertices[3], numVertices;
  PetscInt       numFaces    = 2*faces[0]*faces[1] + 2*faces[1]*faces[2] + 2*faces[0]*faces[2]+2;
  Vec            coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscInt       coordSize;
  PetscMPIInt    rank;
  PetscInt       v, vx, vy, vz;
  PetscInt       voffset, iface=0, cone[4];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((faces[0] < 1) || (faces[1] < 1) || (faces[2] < 1)) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Must have at least 1 face per side");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  vertices[0] = faces[0]+1; vertices[1] = faces[1]+1; vertices[2] = faces[2]+1;
  numVertices = vertices[0]*vertices[1]*vertices[2]+8;
  if (!rank) {
    PetscInt f;

    ierr = DMPlexSetChart(dm, 0, numFaces+numVertices);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      ierr = DMPlexSetConeSize(dm, f, 4);CHKERRQ(ierr);
    }
    ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */

    /* Side 0 (Top) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vertices[0]*vertices[1]*(vertices[2]-1) + vy*vertices[0] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]+1; cone[3] = voffset+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        printf(" %d, cone %d %d %d %d\n",iface,cone[0],cone[1],cone[2],cone[3]);
				iface++;
      }
    }

    /* Side 1 (Bottom) */
    for (vy = 0; vy < faces[1]; vy++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vy*(faces[0]+1) + vx;
        cone[0] = voffset+1; cone[1] = voffset; cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]+1;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
				iface++;
      }
    }

    /* Side 2 (Front) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vx;
        cone[0] = voffset; cone[1] = voffset+1; cone[2] = voffset+vertices[0]*vertices[1]+1; cone[3] = voffset+vertices[0]*vertices[1];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
				iface++;
      }
    }

    /* Side 3 (Back) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vx = 0; vx < faces[0]; vx++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vertices[0]*(vertices[1]-1) + vx;
        cone[0] = voffset+vertices[0]*vertices[1]; cone[1] = voffset+vertices[0]*vertices[1]+1;
        cone[2] = voffset+1; cone[3] = voffset;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
				iface++;
      }
    }

    /* Side 4 (Left) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vy = 0; vy < faces[1]; vy++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vy*vertices[0];
        cone[0] = voffset; cone[1] = voffset+vertices[0]*vertices[1];
        cone[2] = voffset+vertices[0]*vertices[1]+vertices[0]; cone[3] = voffset+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
				iface++;
      }
    }

    /* Side 5 (Right) */
    for (vz = 0; vz < faces[2]; vz++) {
      for (vy = 0; vy < faces[1]; vy++) {
        voffset = numFaces + vz*vertices[0]*vertices[1] + vy*vertices[0] + faces[0];
        cone[0] = voffset+vertices[0]*vertices[1]; cone[1] = voffset;
        cone[2] = voffset+vertices[0]; cone[3] = voffset+vertices[0]*vertices[1]+vertices[0];
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
				iface++;
      }
    }
    {
        voffset = numFaces + numVertices-8;
        cone[0] = voffset; cone[1] = voffset+1;
        cone[2] = voffset+2; cone[3] = voffset+3;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        iface++;
        voffset = numFaces + numVertices-4;
        cone[0] = voffset; cone[1] = voffset+1;
        cone[2] = voffset+2; cone[3] = voffset+3;
        ierr    = DMPlexSetCone(dm, iface, cone);CHKERRQ(ierr);
        iface++;
    }
  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMSetCoordinateDim(dm, 3);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numFaces, numFaces + numVertices);CHKERRQ(ierr);
  for (v = numFaces; v < numFaces+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 3);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, 3);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (vz = 0; vz <= faces[2]; ++vz) {
    for (vy = 0; vy <= faces[1]; ++vy) {
      for (vx = 0; vx <= faces[0]; ++vx) {
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+0] = lower[0] + ((upper[0] - lower[0])/faces[0])*vx;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+1] = lower[1] + ((upper[1] - lower[1])/faces[1])*vy;
        coords[((vz*(faces[1]+1)+vy)*(faces[0]+1)+vx)*3+2] = lower[2] + ((upper[2] - lower[2])/faces[2])*vz;
      }
    }
  }
  // 1st
  coords[3*(numVertices-8)] = .2;
  coords[3*(numVertices-8)+1] = .2;
  coords[3*(numVertices-8)+2] = .5;
  //2 
  coords[3*(numVertices-5)] = .5;
  coords[3*(numVertices-5)+1] = .2;
  coords[3*(numVertices-5)+2] = .5;
  //3
  coords[3*(numVertices-6)] = .5;
  coords[3*(numVertices-6)+1] = .6;
  coords[3*(numVertices-6)+2] = .5;
  //4
  coords[3*(numVertices-7)] = .2;
  coords[3*(numVertices-7)+1] = .6;
  coords[3*(numVertices-7)+2] = .5;
  // 1st
  coords[3*(numVertices-4)] = .3;
  coords[3*(numVertices-4)+1] = .5;
  coords[3*(numVertices-4)+2] = .2;
  //2 
  coords[3*(numVertices-1)] = .4;
  coords[3*(numVertices-1)+1] = .5;
  coords[3*(numVertices-1)+2] = .2;
  //3
  coords[3*(numVertices-2)] = .4;
  coords[3*(numVertices-2)+1] = .5;
  coords[3*(numVertices-2)+2] = .6;
  //4
  coords[3*(numVertices-3)] = .3;
  coords[3*(numVertices-3)+1] = .5;
  coords[3*(numVertices-3)+2] = .6;
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = ax^2 +bx +c, coeff = [c,b,a] */
static PetscErrorCode ParabolaEval(Vec coeff,PetscReal x,PetscReal *y)
{
  PetscScalar *arr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(coeff,&arr);CHKERRQ(ierr);
  *y = arr[2]*x*x +arr[1]*x +arr[0];
  ierr = VecRestoreArrayRead(coeff,&arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscReal EvalParabolaArcLength(PetscReal a,PetscReal b,PetscReal c,PetscReal val)
{ 
  PetscReal aux  = 2.*a*val+b;
  PetscReal aux2 = sqrt(aux*aux+1.);
  return (aux2*aux + log(aux2 +aux))/(4.*a);
}

/* y = ax^2 +bx +c, coeff = [c,b,a] */
static PetscErrorCode ParabolaArcLength(Vec coeff,PetscReal start, PetscReal end,PetscReal *l)
{
  PetscScalar *arr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(coeff,&arr);CHKERRQ(ierr);
  *l = EvalParabolaArcLength(arr[2],arr[1],arr[0],end)-EvalParabolaArcLength(arr[2],arr[1],arr[0],start);
  ierr = VecRestoreArrayRead(coeff,&arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscReal EvalParabolaArea(PetscReal a,PetscReal b,PetscReal c,PetscReal val)
{ 
  PetscReal aux = val*val;
  return (aux*val*a/3. + .5*b*aux + c*val);
}

/* y = ax^2 +bx +c, coeff = [c,b,a] */
static PetscErrorCode ParabolaArea(Vec coeff,PetscReal start, PetscReal end,PetscReal *a)
{
  PetscScalar *arr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(coeff,&arr);CHKERRQ(ierr);
  *a = EvalParabolaArea(arr[2],arr[1],arr[0],end)-EvalParabolaArea(arr[2],arr[1],arr[0],start);
  ierr = VecRestoreArrayRead(coeff,&arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = ax^2 +bx +c, coeff = [c,b,a] */
static PetscErrorCode ParabolaEqPoints(Vec coeff,PetscReal start,PetscReal end,PetscInt n,PetscReal **a)
{
  PetscScalar    *arr;
  PetscReal      *pts;
  PetscReal      x,l,l1,l2,aux;
  PetscInt       i,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&pts);CHKERRQ(ierr);
  ierr = ParabolaArcLength(coeff,start,end,&l);
  ierr = VecGetArrayRead(coeff,&arr);CHKERRQ(ierr);
  x = start+(end-start)/(n-1.);
  for (i=1; i<n-1; i++) {
    l2 = l*i/(n-1);
    for (k=0; k<=5; k++) {
      ierr = ParabolaArcLength(coeff,start,x,&l1);
      aux = 2.*arr[2]*x+arr[1];
      x = x - (l1-l2)/sqrt(1.+aux*aux);
    }
    pts[i] = x;
  }
  pts[0] = start;
  pts[i] = end;
  ierr = VecRestoreArrayRead(coeff,&arr);CHKERRQ(ierr);
  *a = pts;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBoundaryMesh(MPI_Comm comm, AppCtx *user, DM *boundary)
{
  DM dm;
  PetscInt numVertices    = 0;
  PetscInt numEdges       = 0;
  const char     *bdname = "marker"; /* only "marker" vertices are copied by all generators */
  Vec            x,b,coordinates;
  PetscSection   coordSection;
  PetscScalar    *coords;
  PetscReal      ifLength;
  PetscInt       i,j,coordSize;
  PetscMPIInt    rank,size,sizeL,sizeU;
  MPI_Comm       subcomm;
  PetscInt       v, vx, vy;
  KSP ksp;
  PC pc;
  Mat A; 
  PetscScalar *data;
  PetscReal S=375.,Saux,Sl,Su;
  PetscReal y;
  PetscInt vert,vertB,vertR,vertLu,vertL,vertU,vertUl;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (user->x_n) {
    if (size == 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Need at least two ranks for FETI");
    /* compute interface function using quadratic least squares fit */
    ierr = PetscMalloc1(3*user->x_n,&data);CHKERRQ(ierr);
    for (i=0; i<user->x_n; i++) {
      data[i] = 1.;
    }
    for (j=1; j<3; j++) {
      for (i=0; i<user->x_n; i++) {
        data[j*user->x_n+i] = data[(j-1)*user->x_n+i]*user->x[i];
      }
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,user->x_n,3,data,&A);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
    ierr = VecPlaceArray(b,user->y);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    user->coeff = x;
    /* interface length */
    ierr = ParabolaArcLength(x,user->x[0],user->x[user->x_n-1],&ifLength);CHKERRQ(ierr);
    /* eq points */
    ierr = ParabolaEqPoints(x,user->x[0],user->x[user->x_n-1],user->n,&user->xPoints);CHKERRQ(ierr);
    /* domains area */
    ierr = ParabolaArea(x,user->x[0],user->x[user->x_n-1],&Saux);CHKERRQ(ierr);
    Su = S;
    if (user->y[0] != 15.) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented computation for given area");
    }
    if (user->y[user->x_n-1] != 10.) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented computation for given area");
    }
    Su = Su -Saux -15.*user->x[0] -10*(30-user->x[user->x_n-1]);
    /* domain ranks */
    Sl = S-Su;
    sizeU = PetscMax(1,size*Su/S);
    sizeL = size-sizeU;
    user->sizeL = sizeL;
    ierr = PetscSubcommCreate(comm,&user->psubcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetTypeGeneral(user->psubcomm,rank<sizeL ? 0:1,rank<sizeL ? rank:sizeL-rank);CHKERRQ(ierr);
    ierr = PetscSubcommGetChild(user->psubcomm,&subcomm);CHKERRQ(ierr);
    ierr = DMCreate(subcomm,&dm);CHKERRQ(ierr);                                 
  } else {
    ierr = DMCreate(comm,&dm);CHKERRQ(ierr);                                 
  }

  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);                            
  ierr = DMSetDimension(dm,user->dim-1);CHKERRQ(ierr);                   
  ierr = DMSetCoordinateDim(dm,user->dim);CHKERRQ(ierr);
  if (!user->x_n) {
    if (!rank) {
      PetscInt e;
      PetscInt vertex,cone[2];

      ierr = DMPlexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
      for (e = 0; e < numEdges; ++e) {
        ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
      vertex = numEdges;
      for (e = 0; e < numEdges; ++e) {
        cone[0] = vertex; cone[1] = vertex+1;
        if (cone[1] == numVertices+numEdges) cone[1] = numEdges;
        printf("e %d, c: %d %d\n",e,cone[0],cone[1]);
        ierr    = DMPlexSetCone(dm, e, cone);CHKERRQ(ierr);
        vertex += 1;
      }
    }
  } else {
    PetscInt e;
    PetscInt vertex,cone[2];
    PetscReal edgeLength = ifLength/(double)user->n;
    /* lower domain */
    if (!rank) {
      /* doesn't handle all configs */
      vertB  = floor(30./edgeLength);
      vertR  = PetscMax(floor(10./edgeLength),2); /*right and upper right*/
      vertLu = PetscMax(floor(user->x[0]/edgeLength),2);
      vertL  = PetscMax(floor(15./edgeLength),2);

      numVertices = vertB +2*vertR +vertLu +vertL +user->n -6;
      numEdges = numVertices;
      ierr = DMPlexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
      for (e = 0; e < numEdges; ++e) {
        ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
      vertex = numEdges;
      for (e = 0; e < numEdges; ++e) {
        cone[0] = vertex; cone[1] = vertex+1;
        if (cone[1] == numVertices+numEdges) cone[1] = numEdges;
        ierr    = DMPlexSetCone(dm, e, cone);CHKERRQ(ierr);
        vertex += 1;
      }
      /* upper domain */
    } else if (rank == sizeL) {
      vertU  = PetscMax(floor(11.18/edgeLength),2);
      vertUl  = PetscMax(floor((10.-(user->x[0]))/edgeLength),2);
      numVertices = vertU +vertUl +user->n -3;
      numEdges = numVertices;
      ierr = DMPlexSetChart(dm, 0, numEdges+numVertices);CHKERRQ(ierr);
      for (e = 0; e < numEdges; ++e) {
        ierr = DMPlexSetConeSize(dm, e, 2);CHKERRQ(ierr);
      }
      ierr = DMSetUp(dm);CHKERRQ(ierr); /* Allocate space for cones */
      vertex = numEdges;
      for (e = 0; e < numEdges; ++e) {
        cone[0] = vertex; cone[1] = vertex+1;
        if (cone[1] == numVertices+numEdges) cone[1] = numEdges;
        ierr    = DMPlexSetCone(dm, e, cone);CHKERRQ(ierr);
        vertex += 1;
      }
    }
  }
    
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  /* Build coordinates */
  ierr = DMSetCoordinateDim(dm, 2);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numEdges, numEdges + numVertices);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, 2);CHKERRQ(ierr);
  for (v = numEdges; v < numEdges+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, 2);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, 2);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, 2);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);

  if (!user->x_n) {
    coords[0] = 0.;
    coords[1] = 0.;

    coords[2] = 30.;
    coords[3] = 0.;

    coords[4] = 30.;
    coords[5] = 10.;

    coords[6] = 20.;
    coords[7] = 10.;

    coords[8] = 10.;
    coords[9] = 15.;

    coords[10] = 0.;
    coords[11] = 15.;
  } else {
    if (!rank) {
      PetscReal len = 30./(vertB-1);
      vert = 2*vertB-2;
      j = 0;
      for (i=0; i<vert; i+=2) {
        coords[i] = j*len;
        coords[i+1] = 0.;
        j++;
      }

      len = 10./(vertR-1);
      vert += 2*vertR-2;
      j = 0;
      for (; i<vert; i+=2) {
        coords[i] = 30.;
        coords[i+1] = j*len;
        j++;
      }
      
      vert += 2*vertR-2;
      j = 0;
      for (; i<vert; i+=2) {
        coords[i] = 30.-j*len;
        coords[i+1] = 10.;
        j++;
      }

      vert += 2*user->n -2;
      j = user->n-1;
      for (; i<vert; i+=2) {
        coords[i] = user->xPoints[j];
        ierr = ParabolaEval(x,user->xPoints[j],&y);CHKERRQ(ierr);
        coords[i+1] = y;
        j--;
      }

      len = user->x[0]/(vertLu-1);
      vert += 2*vertLu-2;
      j = 0;
      for (; i<vert; i+=2) {
        coords[i] = user->x[0]-j*len;
        coords[i+1] = 15.;
        j++;
      }

      len = 15./(vertL-1);
      vert += 2*vertL-2;
      j = 0;
      for (; i<vert; i+=2) {
        coords[i] = 0.;
        coords[i+1] = 15-j*len;
        j++;
      }
    } else if (rank==sizeL) {
      PetscReal len = 11.88/(vertU-1);
      vert = 2*user->n -2;
      j = 0;
      for (i=0; i<vert; i+=2) {
        coords[i] = user->xPoints[j];
        ierr = ParabolaEval(x,user->xPoints[j],&y);CHKERRQ(ierr);
        coords[i+1] = y;
        j++;
      }
      vert += 2*vertU-2;
      j = 0;
      PetscReal isq = 1./PetscSqrtReal(5);
      for (; i<vert; i+=2) {
        coords[i] = 20.-j*len*2.*isq;
        coords[i+1] = 10. +j*len*isq;
        j++;
      }
      len = 4.5/(vertUl-1);
      vert += 2*vertUl-2;
      j = 0;
      for (; i<vert; i+=2) {
        coords[i] = 10.-j*len;
        coords[i+1] = 15.;
        j++;
      }
    }
  }

  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
	if(rank==sizeL)ierr = DMViewFromOptions(dm, NULL, "-dm_boundary_view");CHKERRQ(ierr);
  *boundary = dm;
  PetscFunctionReturn(0);
}

static PetscErrorCode findInt(PetscInt val,PetscInt size,PetscInt *arr,PetscInt *idx)
{
  PetscInt i;

  PetscFunctionBeginUser;
  *idx = -1;
  for (i=0; i<size; i++) {
    if (arr[i] == val) {
      *idx = i;
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ViewClosure(DM dm, PetscInt p,PetscBool useCone)
{
  PetscInt          i,np,*points;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("tc %d: ",p);
  ierr = DMPlexGetTransitiveClosure(dm,p,useCone,&np,&points);CHKERRQ(ierr);
  for (i=0; i<2*np; i++) {
    printf("%d, ",points[i]);
  }
  printf("\n");
  ierr = DMPlexRestoreTransitiveClosure(dm,p,useCone,&np,&points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Right now, I have just added duplicate faces, which see both cells. We can
- Add duplicate vertices and decouple the face cones
- Disconnect faces from cells across the rotation gap
*/
static PetscErrorCode SplitFaces(DM *dmSplit, const char labelName[], AppCtx *user)
{
  DM             dm = *dmSplit, sdm;
  PetscSF        sfPoint, gsfPoint;
  PetscSection   coordSection, newCoordSection;
  Vec            coordinates;
  IS             idIS;
  const PetscInt *ids;
  PetscInt       *newpoints,*oldpoints;
  PetscInt       dim, depth, maxConeSize, maxSupportSize, numLabels, numGhostCells;
  PetscInt       numFS, fs, pStart, pEnd, p, cEnd, cEndInterior, vStart, vEnd, v, fStart, fEnd, newf, d, l;
  PetscBool      hasLabel;
  PetscInt depthShift[3]; /* fix for 3d*/
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMHasLabel(dm, labelName, &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) PetscFunctionReturn(0);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(sdm, dim);CHKERRQ(ierr);

  ierr = DMGetLabelIdIS(dm, labelName, &idIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);

  user->numSplitFaces = 0;
  for (fs = 0; fs < numFS; ++fs) {
    PetscInt numBdFaces;

    ierr = DMGetStratumSize(dm, labelName, ids[fs], &numBdFaces);CHKERRQ(ierr);
    user->numSplitFaces += numBdFaces;
  }
  user->numDupVertices = user->numSplitFaces -user->nFrac; //TODO fix for 3d, edpoints
  printf("dup: %d %d\n",user->numSplitFaces,user->numDupVertices);
  ierr  = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  pEnd += user->numSplitFaces + user->numDupVertices;
  ierr  = DMPlexSetChart(sdm, pStart, pEnd);CHKERRQ(ierr);
  /* Set cone and support sizes */
  depthShift[0] = 0; /* vertices */
  depthShift[1] = user->numDupVertices;
  depthShift[2] = 0; /* cells */
  // Inherit everything
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
  printf("dep %d %d %d\n",depth,pStart,pEnd);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt newp = p+depthShift[d];
      PetscInt size;

      ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, newp, size);CHKERRQ(ierr);
      //ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
      //ierr = DMPlexSetSupportSize(sdm, newp, size);CHKERRQ(ierr);
    }
  }
  // Connect new facets to a single cell
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  newf = fEnd+depthShift[1]; /* TODO fix 3d */
  for (fs = 0; fs < numFS; ++fs) {
    IS             faceIS;
    const PetscInt *faces;
    PetscInt       numFaces, f;

    ierr = DMGetStratumIS(dm, labelName, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f, ++newf) {
      PetscInt size;

      /* facets have the same amount of vertices */
      ierr = DMPlexGetConeSize(dm, faces[f], &size);CHKERRQ(ierr);
      if (size != 2) printf("errr\n");
      ierr = DMPlexSetConeSize(sdm, newf, size);CHKERRQ(ierr);
      ///* but only a single cell (size should be eq 1) */
      //ierr = DMPlexGetSupportSize(dm, faces[f], &size);CHKERRQ(ierr);
      //ierr = DMPlexSetSupportSize(sdm, newf, size-1);CHKERRQ(ierr);
      ///* the old face support size is also 1 */
      //ierr = DMPlexSetSupportSize(sdm, faces[f]+depthShift[1], size-1);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = DMSetUp(sdm);CHKERRQ(ierr);
  /* Set cones and supports */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(maxConeSize, maxSupportSize), &newpoints);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    PetscInt shiftl=0,shiftu=0;
    if (d) shiftl = depthShift[d-1];
    if (d!=depth) shiftu = depthShift[d+1];
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      const PetscInt *points, *orientations;
      PetscInt       size, i, newp = p+depthShift[d];

      ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, p, &points);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &orientations);CHKERRQ(ierr);
      for (i = 0; i < size; ++i) {
        newpoints[i] = points[i]+shiftl;
        //printf("%d o:%d n:%d, (%d,%d)\n",d,p,newp,points[i],newpoints[i]);
      }
      ierr = DMPlexSetCone(sdm, newp, newpoints);CHKERRQ(ierr);
      //if (d != depth) { //cells don't have orientation? 3D?
			ierr = DMPlexSetConeOrientation(sdm, newp, orientations);CHKERRQ(ierr);
			//}
      //ierr = DMPlexGetSupportSize(sdm, newp, &size);CHKERRQ(ierr);
      //ierr = DMPlexGetSupport(dm, p, &points);CHKERRQ(ierr);
      //for (i = 0; i < size; ++i) { 
      //  newpoints[i] = points[i]+shiftu;
      //}
      //ierr = DMPlexSetSupport(sdm, newp, newpoints);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(newpoints);CHKERRQ(ierr);
  newf = fEnd+depthShift[1]; /* TODO fix 3d */
  printf("%d %d\n",fEnd,depthShift[1]);
  ierr = PetscMalloc2(user->numDupVertices+2*user->nFrac,&newpoints,user->numDupVertices+2*user->nFrac,&oldpoints);CHKERRQ(ierr);
  user->newpoints = newpoints;
  user->oldpoints = oldpoints;
  for (fs = 0; fs < user->numDupVertices+2*user->nFrac; fs++) {
    newpoints[fs] = -1;
    oldpoints[fs] = -1;
  }
  ierr = DMPlexGetDepthStratum(dm, 0,NULL, &vEnd);CHKERRQ(ierr);
  PetscInt pts = 0,pts2=0,idx,size;
  for (fs = 0; fs < numFS; ++fs) {
    IS             faceIS,endIS;
    const PetscInt *faces,*endpoints;
    PetscInt       numFaces, f,i,j;

    ierr = DMGetStratumIS(dm, labelName, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm, "fracEnd", ids[fs], &endIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISGetIndices(endIS, &endpoints);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f, ++newf) {
      const PetscInt *points,*fpoints,*orientations;
      PetscInt i,cone[2],cell;

      ierr = DMPlexGetConeSize(dm, faces[f], &size);CHKERRQ(ierr); /* should be 2 */
      if (size != 2) printf("err\n");; /* there are vertices for some reason */
      ierr = DMPlexGetCone(dm, faces[f], &points);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, faces[f], &orientations);CHKERRQ(ierr);
      for (i=0; i<2; i++) {
        ierr = findInt(points[i],pts,oldpoints,&idx);CHKERRQ(ierr);
        printf("%d,%d,%d\n",points[i],pts,idx);
        if (idx > -1) {
          cone[i] = newpoints[idx];
        } else {
          oldpoints[pts] = points[i];
          newpoints[pts] = vEnd+pts2;
          /* do not duplicate endpoints */
          if (points[i] == endpoints[0] || points[i] == endpoints[1]) {
            newpoints[pts] = oldpoints[pts];
          } else { 
            pts2++;
          }
          cone[i] = newpoints[pts];
          printf("%d->%d\n",oldpoints[pts],newpoints[pts]);
          pts++;
        }
      }   
      printf("face %d: %d %d -> %d: %d %d\n",faces[f],points[0],points[1],newf,cone[0],cone[1]);
      ierr = DMPlexSetCone(sdm, newf, cone);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(sdm, newf, orientations);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, faces[f], &points);CHKERRQ(ierr);
      ierr = DMGetLabelValue(dm,"fracCell",points[0],&idx);CHKERRQ(ierr);
      cell = points[1];
      if (idx == fs) {
        //ierr = DMPlexSetSupport(sdm, faces[f]+depthShift[1], &points[1]);CHKERRQ(ierr);
        cell = points[0];
      }
      //ierr = DMPlexSetSupport(sdm, newf, &cell);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cell, &fpoints);CHKERRQ(ierr);
      //ierr = DMPlexGetConeOrientation(dm, cell, &orientations);CHKERRQ(ierr);
      printf("c %d \n",cell);
      for (i=0; i<3; i++) {
        if (fpoints[i] == faces[f]) {
          ierr = DMPlexInsertCone(sdm,cell,i,newf);CHKERRQ(ierr);
          continue;
        }
        ierr = DMPlexGetCone(dm, fpoints[i], &points);CHKERRQ(ierr);
          printf("f %d (%d,%d)\n",fpoints[i],points[0],points[1]);
        for (j=0; j<2; j++) {
          ierr = findInt(points[j],pts,oldpoints,&idx);CHKERRQ(ierr);
          if (idx > -1) {
            ierr = DMPlexInsertCone(sdm,fpoints[i]+depthShift[1],j,newpoints[idx]);CHKERRQ(ierr);
             printf("fc %d, %d\n",fpoints[i],newpoints[idx]);
          }
        }
      }
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  //ierr = DMPlexGetDepthStratum(dm, 2, &pStart, &pEnd);CHKERRQ(ierr);
  //for (p = pStart; p < pEnd; ++p) {
  //  PetscInt          i,j,*edges,*points;

  //  printf("c %d: ",p);
  //  ierr = DMPlexGetCone(sdm,p,&edges);CHKERRQ(ierr);
  //  for (i=0; i<3; i++) {
  //    ierr = DMPlexGetCone(sdm,edges[i],&points);CHKERRQ(ierr);
  //    printf("(%d,%d) ",points[0]-pEnd,points[1]-pEnd);
  //    }
  //  printf("\n");
  //}
  ierr = DMPlexSymmetrize(sdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(sdm);CHKERRQ(ierr);
  ierr = DMView(sdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Convert coordinates */
  ierr = DMPlexGetDepthStratum(sdm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(newCoordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(newCoordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newCoordSection, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(newCoordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(newCoordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(newCoordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(sdm, PETSC_DETERMINE, newCoordSection);CHKERRQ(ierr);
  //ierr = PetscSectionDestroy(&newCoordSection);CHKERRQ(ierr); /* relinquish our reference */
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates,&size);CHKERRQ(ierr);
  Vec newCoordinates;
  const PetscScalar *arr;
  ierr = VecGetArrayRead(coordinates,&arr);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,(vEnd-vStart)*dim,&newCoordinates);CHKERRQ(ierr);
  for (v = 0; v < size; ++v) {
    ierr = VecSetValue(newCoordinates,v,arr[v],INSERT_VALUES);CHKERRQ(ierr);
  }
  PetscInt i;
  for (v = vStart+size/dim,i=size; v < vEnd;v++,i+=dim) {
    ierr = findInt(v,pts,newpoints,&idx);CHKERRQ(ierr);
    printf("v %d %d %d\n",v,pts,idx);
    ierr = VecSetValue(newCoordinates,i,arr[2*(oldpoints[idx]-vStart)],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(newCoordinates,i+1,arr[2*(oldpoints[idx]-vStart)+1]-.05,INSERT_VALUES);CHKERRQ(ierr);
    //printf("i %d %f %f\n",i,arr[2*(oldpoints[idx]-vStart)],arr[2*(oldpoints[idx]-vStart)+1]);
  }
  //VecView(coordinates,PETSC_VIEWER_STDOUT_WORLD);
  //VecView(newCoordinates,PETSC_VIEWER_STDOUT_WORLD);
  ierr = DMSetCoordinatesLocal(sdm, newCoordinates);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&newCoordSection);CHKERRQ(ierr); /* relinquish our reference */
  /* Copy labels */
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    const char *lname;
    PetscBool  isDepth, isDim;

    ierr = DMGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = PetscStrcmp(lname, "dim", &isDim);CHKERRQ(ierr);
    if (isDim) continue;
    ierr = DMCreateLabel(sdm, lname);CHKERRQ(ierr);
    ierr = DMGetLabelIdIS(dm, lname, &idIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
    ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);
    for (fs = 0; fs < numFS; ++fs) {
      IS             pointIS;
      const PetscInt *points;
      PetscInt       numPoints;

      ierr = DMGetStratumIS(dm, lname, ids[fs], &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints; ++p) {
        PetscInt newpoint = points[p];
        PetscInt depth;
        ierr = DMPlexGetPointDepth(dm,newpoint,&depth);CHKERRQ(ierr);
        newpoint += depthShift[depth];
        /* TODO mark new points */
        ierr = DMSetLabelValue(sdm, lname, newpoint, ids[fs]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
    ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  }
  {
    /* Convert pointSF */
    const PetscSFNode *remotePoints;
    PetscSFNode       *gremotePoints;
    const PetscInt    *localPoints;
    PetscInt          *glocalPoints,*newLocation,*newRemoteLocation;
    PetscInt          numRoots, numLeaves;
    PetscMPIInt       size;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(sdm, &gsfPoint);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    if (numRoots >= 0) {
      ierr = PetscMalloc2(numRoots,&newLocation,pEnd-pStart,&newRemoteLocation);CHKERRQ(ierr);
      for (l=0; l<numRoots; l++) newLocation[l] = l; /* + (l >= cEnd ? numGhostCells : 0); */
      ierr = PetscSFBcastBegin(sfPoint, MPIU_INT, newLocation, newRemoteLocation,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfPoint, MPIU_INT, newLocation, newRemoteLocation,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeaves,    &glocalPoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeaves, &gremotePoints);CHKERRQ(ierr);
      for (l = 0; l < numLeaves; ++l) {
        glocalPoints[l]        = localPoints[l]; /* localPoints[l] >= cEnd ? localPoints[l] + numGhostCells : localPoints[l]; */
        gremotePoints[l].rank  = remotePoints[l].rank;
        gremotePoints[l].index = newRemoteLocation[localPoints[l]];
      }
      ierr = PetscFree2(newLocation,newRemoteLocation);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(gsfPoint, numRoots+numGhostCells, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
    ierr     = DMDestroy(dmSplit);CHKERRQ(ierr);
    *dmSplit = sdm;
    DMLabel label;
    ierr = DMGetLabel(sdm,labelName,&label);CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MarkVertices2D(DM dm, PetscInt nFrac,const PetscReal coord[], PetscReal eps)
{
  PetscInt          c, cdim, i, j,k, o, p, vStart, vEnd,val,size;
  Vec               allCoordsVec;
  PetscScalar       min[2],max[2];
  const PetscScalar *allCoords;
  const PetscInt     *supp,*cone;
  PetscReal         area;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (eps < 0) eps = PETSC_SQRT_MACHINE_EPSILON;
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &allCoordsVec);CHKERRQ(ierr);
  //VecView(allCoordsVec,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecGetArrayRead(allCoordsVec, &allCoords);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
	for (p = vStart,j=0; p < vEnd; p++,j+=cdim) { //for each vertex
		for (i=0; i < nFrac*4; i+=4) { //for each fracture
      area  = (coord[i]-coord[i+2])*(coord[i+3]-allCoords[j+1]); 
      area -= (coord[i+1]-coord[i+3])*(coord[i+2]-allCoords[j]);
      if (PetscAbsReal(area) <= eps) {
        /* check for straight line */
        min[0] = PetscMin(coord[i],coord[i+2]);
        min[1] = PetscMin(coord[i+1],coord[i+3]);
        max[0] = PetscMax(coord[i],coord[i+2]);
        max[1] = PetscMax(coord[i+1],coord[i+3]);
        if (min[0] > allCoords[j] || min[1] > allCoords[j+1] || max[0] < allCoords[j] || max[1] < allCoords[j+1]) continue;
        printf("%d frac: %f %f, %d\n",i/4,allCoords[j],allCoords[j+1],p);
        ierr = DMSetLabelValue(dm,"fracVert",p,i/4);
        /* mark fracture endpoints */
        ierr = DMGetLabelValue(dm,"marker",p,&val); /* check from coords? */
        if (val>-1) {
          ierr = DMSetLabelValue(dm,"fracEnd",p,i/4);
        }
        /* mark adjacent vetices on + side */
        ierr = DMPlexGetSupport(dm,p,&supp);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm,p,&size);CHKERRQ(ierr);
        for (k=0; k<size; k++) { //for each support
          ierr = DMPlexGetCone(dm,supp[k],&cone);CHKERRQ(ierr);
          val = cone[0];
          if (val == p) val = cone[1];
          area =  (coord[i]-coord[i+2])*(coord[i+3]-allCoords[cdim*(val-vStart)+1]); 
          area -= (coord[i+1]-coord[i+3])*(coord[i+2]-allCoords[cdim*(val-vStart)]);
          if (area>0) {
            ierr = DMSetLabelValue(dm,"fracVertOrient",val,i/4);
            printf("+ %d: %d (%f,%f) \n",p-vStart,val-vStart,allCoords[(val-vStart)*cdim],allCoords[cdim*(val-vStart)+1]);
          }
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(allCoordsVec, &allCoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode MarkFractures(DM dm,PetscInt nFrac)
{
  const char     *bdname = "fracture";
  IS             is,isFace;
  const PetscInt *points,*pts,*faces,*faces2,*cells,*orientPts;
  PetscInt       i,j,k,l,m,n,val,depth,nCovered,numPoints,numOrientPts,pt[2],labv;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLabel(dm,"fracVert",&label);CHKERRQ(ierr);
  ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,"fracVertOrient",&label);CHKERRQ(ierr);
  ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, bdname);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,bdname,&label);CHKERRQ(ierr);

  for (i=0; i<nFrac; i++) {
    ierr = DMGetStratumIS(dm,"fracVert",i,&is);CHKERRQ(ierr);
    ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&points);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm,"fracVertOrient",i,&is);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&orientPts);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&numOrientPts);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&orientPts);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    for (j=0; j<numPoints; j++) {
      pt[0] = points[j];
      for (k=0; k<numPoints; k++) {
        if (k==j) continue;
        pt[1] = points[k];
        ierr = DMPlexGetJoin(dm,2,pt,&nCovered,&faces);CHKERRQ(ierr);
        if (nCovered==1) {
          ierr = DMSetLabelValue(dm,bdname,faces[0],i);CHKERRQ(ierr);
          /* mark adjacent '+' cells */
          ierr = DMPlexGetSupport(dm,faces[0],&cells);CHKERRQ(ierr);
          for (l=0; l<2;l++) {
            ierr = DMPlexGetCone(dm,cells[l],&faces2);CHKERRQ(ierr);
            for (m=0; m<3; m++) {
              ierr = DMPlexGetCone(dm,faces2[m],&pts);CHKERRQ(ierr);
              /* take vertex that is not in the fracture */
              val = pts[0];
              if (pt[0] == pts[0] || pt[1] == pts[0]) val = pts[1]; 
              for (n=0; n<numPoints; n++) {
                if (val == points[n]) {
                  val = -1;
                  break;
                }
              }
              if (val < 0) continue; /* this face is makes part of the fracture */
              for (n=0; n<numOrientPts; n++) {
                if (orientPts[n] == val) {
                  printf("val %d, pt %d %d\n",val,pt[0],pt[1]);
                  ierr = DMSetLabelValue(dm,"fracCell",cells[l],i);CHKERRQ(ierr);
                  printf("cell %d, %d, %d,%d\n",cells[l],faces2[0],pts[0],pts[1]);
                  break;
                }
              }
            }
          }
        }
        ierr = DMPlexRestoreJoin(dm,2,pt,&nCovered,&faces);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMGetLabel(dm,bdname,&label);CHKERRQ(ierr);
  ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscBool      flg;
  DM             boundary;
  PetscMPIInt    rank;
  DMLabel        label;

  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = CreateBoundaryMesh(comm,user,&boundary);CHKERRQ(ierr);
  ierr = DMPlexTriangleSetOptions(boundary,"pqezQ a1");CHKERRQ(ierr);
  ierr = DMPlexGenerate(boundary, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  //ierr = DMView(*dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //DM dmint;
  //ierr = DMPlexInterpolate(*dm,&dmint);CHKERRQ(ierr);
  //ierr = DMView(dmint,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (rank<user->sizeL) {
    ierr = DMViewFromOptions(*dm, NULL, "-dml_view");CHKERRQ(ierr);
  }
  if (rank>=user->sizeL) ierr = DMViewFromOptions(*dm, NULL, "-dmu_view");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  if (rank<user->sizeL) {
    ierr = DMViewFromOptions(*dm, NULL, "-dmls_view");CHKERRQ(ierr);
  }
  if (rank>=user->sizeL) ierr = DMViewFromOptions(*dm, NULL, "-dmus_view");CHKERRQ(ierr);
  //ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  //ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* Mark boundary in sequence by their distinguishing component:
  * x: 1 = left,   2 = right
  * y: 3 = bottom, 4 = top
  * z: 5 = front,  6 = back
  */
  {
    DMLabel         label;
    IS              is;
    /* Get all facets */
    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    ierr = DMGetStratumIS(*dm, "boundary", 1,  &is);CHKERRQ(ierr);
    ierr = DMCreateLabel(*dm,"Faces");CHKERRQ(ierr);
    if (is) {
    //ierr = ISView(is,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      PetscInt        d, f, Nf;
      const PetscInt *faces;
      PetscInt        csize, i;
      PetscSection    cs;
      Vec             coordinates ;
      DM              cdm;
      ierr = ISGetLocalSize(is, &Nf);CHKERRQ(ierr);
      ierr = ISGetIndices(is, &faces);CHKERRQ(ierr);
      //ISView(is,PETSC_VIEWER_STDOUT_WORLD);
      ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetLocalSection(cdm, &cs);CHKERRQ(ierr);
      /* Check for each boundary facet if both endpoints are on the interface */
      for (f = 0; f < Nf; ++f) {
        PetscReal   tol = 10*PETSC_MACHINE_EPSILON;
        PetscReal   y1,y2;
        PetscInt    b,v;
        PetscScalar *coords = NULL;
        PetscInt    Nv;
        /* Get closure of the facet (vertices in 2D, edges in 3D) */
        ierr = DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
        ierr = ParabolaEval(user->coeff,coords[0],&y1);CHKERRQ(ierr);
        ierr = ParabolaEval(user->coeff,coords[2],&y2);CHKERRQ(ierr);
        if (PetscAbsReal(coords[1] - y1) < tol && PetscAbsReal(coords[3] - y2) < tol) {
          ierr = DMSetLabelValue(*dm,"Faces",faces[f],9);CHKERRQ(ierr);
        }

        if (PetscAbsReal(coords[0] - 0.) < tol && PetscAbsReal(coords[2] - .0) < tol) {
          ierr = DMSetLabelValue(*dm,"Faces",faces[f],1);CHKERRQ(ierr);
        }
          
        ierr = DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(is, &faces);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    //if (rank<user->sizeL) {
    //ierr = DMGetLabel(*dm, "Faces", &label);CHKERRQ(ierr);
    //DMLabelView(label,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)*dm)));
    ////ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    //DMLabelView(label,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)*dm)));
    //DMLabelView(label,PETSC_VIEWER_STDOUT_WORLD);
    //}
  //MPI_Barrier(PETSC_COMM_WORLD);exit(0);
    //DM subdm;
    //DMLabel labelfrac,labelb;
    //ierr = DMGetLabel(*dm, "marker", &labelfrac);CHKERRQ(ierr);
    //DMLabelView(labelfrac,PETSC_VIEWER_STDOUT_WORLD);
    //ierr = DMPlexCreateSubmesh(*dm,labelfrac,9,PETSC_FALSE,&subdm);CHKERRQ(ierr);
    //ierr = DMGetLabel(subdm, "marker", &labelb);CHKERRQ(ierr);
    //ierr = DMView(subdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    //ierr = DMViewFromOptions(subdm, NULL, "-dmx_view");CHKERRQ(ierr);
    //DMLabelView(labelb,PETSC_VIEWER_STDOUT_WORLD);
  

    //ierr = DMPlexLabelCohesiveComplete(*dm, labelfrac,NULL,PETSC_FALSE,boundary);CHKERRQ(ierr);
    //DMLabelView(labelfrac,PETSC_VIEWER_STDOUT_WORLD);
    //exit(0);
    //DM dmsplit;
    //DMLabel labelsplit;
    //ierr = DMPlexConstructCohesiveCells(*dm,labelfrac,labelsplit,&dmsplit);CHKERRQ(ierr);
    //ierr = DMGetStratumIS(*dm, "Faces", 1,  &is);CHKERRQ(ierr);
    //if (is) ierr = ISView(is,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  /* Partition */
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }

  /* Convert DM type */
  ierr = PetscStrcmp(user->dmType, DMPLEX, &flg);CHKERRQ(ierr);
  if (!flg) {
    DM ndm;

    ierr = DMConvert(*dm, user->dmType, &ndm);CHKERRQ(ierr);
    if (ndm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = ndm;
    }
  }

  /* ??? */
  if (user->shear) {ierr = DMPlexShearGeometry(*dm, DM_X, NULL);CHKERRQ(ierr);}
  /* Create local coordinates for cells having periodic faces */
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr); /* refine,... */
  /* duplicate vertices on fracture boundaries */
  DMView(*dm,PETSC_VIEWER_STDOUT_WORLD);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = MarkVertices2D(*dm,user->nFrac,user->fracCoord,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = MarkFractures(*dm,user->nFrac);CHKERRQ(ierr);
  ierr = SplitFaces(dm,"fracture",user);CHKERRQ(ierr);
  //ierr = DMSetFromOptions(*dm);CHKERRQ(ierr); /* refine,... */
  DMView(*dm,PETSC_VIEWER_STDOUT_WORLD);
  ierr = DMViewFromOptions(*dm, NULL, "-dms_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscErrorCode (*exact)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  PetscDS        prob;
  PetscInt       id;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetSpatialDimension(prob, &dim);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_push_u, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_elas_uu);CHKERRQ(ierr);
    //exact = uniform_strain_u;
  //ierr = PetscDSSetExactSolution(prob, 0, exact, user);CHKERRQ(ierr);
  id = 1;
  ierr = DMAddBoundary(dm,   DM_BC_ESSENTIAL, "bottom", "Faces", 0, 0, NULL, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateElasticityNullSpace(DM dm, PetscInt origField, PetscInt field, MatNullSpace *nullspace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateRigidBody(dm, origField, nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupFE(DM dm, PetscInt Nc, PetscBool simplex, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             cdm  = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create finite element */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, Nc, simplex, name ? prefix : NULL,PETSC_DETERMINE, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  /* Copy fields and discrete systems to the coarse DMs, optionally setup null space */
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    if (user->useNearNullspace) {ierr = DMSetNearNullSpaceConstructor(cdm, 0, CreateElasticityNullSpace);CHKERRQ(ierr);}
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeRHS(DM dm,Vec F,void *ctx)
{
  Vec            X,Xloc,Floc;
  PetscBool      transform;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(dm,&X);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
	ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecZeroEntries(Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
  /* Non-conforming routines needs boundary values before G2L */
  ierr = DMPlexSNESComputeBoundaryFEM(dm,Xloc,ctx);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  /* Need to reset boundary values if we transformed */
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  if (transform) {
		ierr = DMPlexSNESComputeBoundaryFEM(dm,Xloc,ctx);CHKERRQ(ierr);
	}
  CHKMEMQ;
  ierr = DMPlexSNESComputeResidualFEM(dm,Xloc,Floc,ctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = VecScale(F,-1.);CHKERRQ(ierr);
  //{
  //  char        name[PETSC_MAX_PATH_LEN];
  //  char        oldname[PETSC_MAX_PATH_LEN];
  //  const char *tmp;
  //  PetscInt    it;

  //  ierr = SNESGetIterationNumber(snes, &it);CHKERRQ(ierr);
  //  ierr = PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Solution, Iterate %d", (int) it);CHKERRQ(ierr);
  //  ierr = PetscObjectGetName((PetscObject) X, &tmp);CHKERRQ(ierr);
  //  ierr = PetscStrncpy(oldname, tmp, PETSC_MAX_PATH_LEN-1);CHKERRQ(ierr);
  //  ierr = PetscObjectSetName((PetscObject) X, name);CHKERRQ(ierr);
  //  ierr = VecViewFromOptions(X, (PetscObject) snes, "-dmsnes_solution_vec_view");CHKERRQ(ierr);
  //  ierr = PetscObjectSetName((PetscObject) X, oldname);CHKERRQ(ierr);
  //  ierr = PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Residual, Iterate %d", (int) it);CHKERRQ(ierr);
  //  ierr = PetscObjectSetName((PetscObject) F, name);CHKERRQ(ierr);
  //  ierr = VecViewFromOptions(F, (PetscObject) snes, "-dmsnes_residual_vec_view");CHKERRQ(ierr);
  //}
  PetscFunctionReturn(0);
}

static PetscErrorCode SolverSNES(DM dm,Vec u)
{
  SNES           snes; /* Nonlinear solver */
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);

  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SolverKSP(DM dm,AppCtx *user,Vec u)
{
  Mat A;
  Vec b,z;
  KSP ksp;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&z,&b);CHKERRQ(ierr);
  ierr = VecSet(z,0.0);CHKERRQ(ierr);

  ierr = DMPlexSNESComputeJacobianFEM(dm,z,A,A,NULL);CHKERRQ(ierr);
  ierr = ComputeRHS(dm,b,user);CHKERRQ(ierr);
  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SolverQPS(DM dm,AppCtx *user,Vec u)
{
  Mat A,R,Bineq;
  Vec b,cineq,z;
  QP  qp;
  QPS qps;
  PetscBool converged;
  PetscInt i,j,l,n;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&z,&b);CHKERRQ(ierr);
  ierr = VecSet(z,0.0);CHKERRQ(ierr);

  ierr = DMPlexSNESComputeJacobianFEM(dm,z,A,A,NULL);CHKERRQ(ierr);
  ierr = ComputeRHS(dm,b,user);CHKERRQ(ierr);
  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = QPCreate(PETSC_COMM_WORLD,&qp);CHKERRQ(ierr);
  ierr = QPSetOperator(qp,A);CHKERRQ(ierr);
  ierr = QPSetRhs(qp,b);CHKERRQ(ierr);
  //ierr = VecView(b,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = QPSetInitialVector(qp,u);CHKERRQ(ierr);
  if (user->numDupVertices) {
    ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
    /* Create ineq mat */
    ierr = MatCreateAIJ(PetscObjectComm((PetscObject)dm),PETSC_DECIDE,PETSC_DECIDE,user->numDupVertices*user->dim,n,2,NULL,2,NULL,&Bineq);CHKERRQ(ierr);CHKERRQ(ierr);
    l = 0;
    for (i=0; i<user->numDupVertices+2*user->nFrac; i++) {
      PetscInt s1,s2,e1;
      if (user->oldpoints[i] == user->newpoints[i]) continue;
      printf("bineq: %d %d\n",user->oldpoints[i],user->newpoints[i]);
      ierr = DMPlexGetPointGlobal(dm,user->oldpoints[i],&s1,&e1);CHKERRQ(ierr);
      ierr = DMPlexGetPointGlobal(dm,user->newpoints[i],&s2,NULL);CHKERRQ(ierr);
      for (j = 0; j<e1-s1; j++) {
        PetscScalar values[2] = {1.,-1.};
        PetscInt idx[2] = {s1+j,s2+j};
        ierr = MatSetValues(Bineq,1,&l,2,idx,values,INSERT_VALUES);CHKERRQ(ierr);
        l++;
      }
    }
    ierr = MatAssemblyBegin(Bineq,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Bineq,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(Bineq,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatCreateVecs(Bineq,NULL,&cineq);CHKERRQ(ierr);
  ierr = MatScale(Bineq,-1.);CHKERRQ(ierr);
    ierr = VecZeroEntries(cineq);CHKERRQ(ierr); /* TODO set aparature */
    ierr = VecSetValue(cineq,1,0.05,INSERT_VALUES);CHKERRQ(ierr); /* TODO set aparature */
    ierr = VecAssemblyBegin(cineq);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(cineq);CHKERRQ(ierr);
    ierr = QPSetIneq(qp,Bineq,cineq);CHKERRQ(ierr);
    
    /* empty nullspace mat */
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,0,0,NULL,0,NULL,&R);CHKERRQ(ierr);                   
    ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);                                                          
    ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);    
    ierr = QPSetOperatorNullSpace(qp,R);CHKERRQ(ierr);                                                                    
    ierr = PetscOptionsInsertString(NULL,"-qpt_dualize_B_nest_extension 0 -qpt_dualize_G_explicit 0");CHKERRQ(ierr); /* workaround for empty nullspace */
    ierr = QPTDualize(qp,MAT_INV_MONOLITHIC,MAT_REG_NONE);CHKERRQ(ierr);
  }
  ierr = QPSetFromOptions(qp);CHKERRQ(ierr);

  ierr = QPSCreate(PetscObjectComm((PetscObject)qp),&qps);CHKERRQ(ierr);
  ierr = QPSSetQP(qps,qp);CHKERRQ(ierr);
  ierr = QPSSetFromOptions(qps);CHKERRQ(ierr);

  ierr = QPSSolve(qps);CHKERRQ(ierr);                                                                                   
  ierr = QPIsSolved(qp,&converged);CHKERRQ(ierr);                                                                       
  if (!converged) PetscPrintf(PETSC_COMM_WORLD,"QPS did not converge!\n"); 

  /* check the constraint */
  ierr = MatMult(Bineq,u,cineq);CHKERRQ(ierr);
  ierr = VecView(cineq,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = QPSDestroy(&qps);CHKERRQ(ierr);
  ierr = QPDestroy(&qp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscViewer       viewer;                                                     
  PetscViewerFormat format; 
  PetscErrorCode ierr;

  /* Init PERMON */
  ierr = PermonInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupFE(dm, user.dim, user.simplex, "displacement", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "displacement");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ISLocalToGlobalMapping ltogm;
  DMGetLocalToGlobalMapping(dm,&ltogm);
ISLocalToGlobalMappingViewFromOptions(ltogm,NULL,"-view_l2g");
  
  if (user.solverqps) {
    ierr = SolverQPS(dm,&user,u);CHKERRQ(ierr);
  } else if (user.solverksp) {
    ierr = SolverKSP(dm,&user,u);CHKERRQ(ierr);
  } else {
    ierr = SolverSNES(dm,u);CHKERRQ(ierr);
  }

  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)u),NULL,NULL, "-displacement_view", &viewer, &format, NULL);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr); 
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PermonFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1_quad_vlap
    requires: triangle
    args: -displacement_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_quad_vlap
    requires: triangle
    args: -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_p3_quad_vlap
    requires: triangle
    args: -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_q1_quad_vlap
    args: -simplex 0 -displacement_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_quad_vlap
    args: -simplex 0 -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_q3_quad_vlap
    requires: !single
    args: -simplex 0 -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001
  test:
    suffix: 2d_p1_quad_elas
    requires: triangle
    args: -sol_type elas_quad -displacement_petscspace_degree 1 -dm_refine 2 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_quad_elas
    requires: triangle
    args: -sol_type elas_quad -displacement_petscspace_degree 2 -dmsnes_check .0001
  test:
    suffix: 2d_p3_quad_elas
    requires: triangle
    args: -sol_type elas_quad -displacement_petscspace_degree 3 -dmsnes_check .0001
  test:
    suffix: 2d_q1_quad_elas
    args: -sol_type elas_quad -simplex 0 -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_quad_elas_shear
    args: -sol_type elas_quad -simplex 0 -shear -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_quad_elas
    args: -sol_type elas_quad -simplex 0 -displacement_petscspace_degree 2 -dmsnes_check .0001
  test:
    suffix: 2d_q2_quad_elas_shear
    args: -sol_type elas_quad -simplex 0 -shear -displacement_petscspace_degree 2 -dmsnes_check
  test:
    suffix: 2d_q3_quad_elas
    args: -sol_type elas_quad -simplex 0 -displacement_petscspace_degree 3 -dmsnes_check .0001
  test:
    suffix: 2d_q3_quad_elas_shear
    requires: !single
    args: -sol_type elas_quad -simplex 0 -shear -displacement_petscspace_degree 3 -dmsnes_check

  test:
    suffix: 3d_p1_quad_vlap
    requires: ctetgen
    args: -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_quad_vlap
    requires: ctetgen
    args: -dim 3 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_p3_quad_vlap
    requires: ctetgen
    args: -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001
  test:
    suffix: 3d_q1_quad_vlap
    args: -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_quad_vlap
    args: -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_q3_quad_vlap
    args: -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001
  test:
    suffix: 3d_p1_quad_elas
    requires: ctetgen
    args: -sol_type elas_quad -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_quad_elas
    requires: ctetgen
    args: -sol_type elas_quad -dim 3 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_p3_quad_elas
    requires: ctetgen
    args: -sol_type elas_quad -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001
  test:
    suffix: 3d_q1_quad_elas
    args: -sol_type elas_quad -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_quad_elas
    args: -sol_type elas_quad -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -dmsnes_check .0001
  test:
    suffix: 3d_q3_quad_elas
    requires: !single
    args: -sol_type elas_quad -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -dmsnes_check .0001

  test:
    suffix: 2d_p1_trig_vlap
    requires: triangle
    args: -sol_type vlap_trig -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_trig_vlap
    requires: triangle
    args: -sol_type vlap_trig -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p3_trig_vlap
    requires: triangle
    args: -sol_type vlap_trig -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_trig_vlap
    args: -sol_type vlap_trig -simplex 0 -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_trig_vlap
    args: -sol_type vlap_trig -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_trig_vlap
    args: -sol_type vlap_trig -simplex 0 -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p1_trig_elas
    requires: triangle
    args: -sol_type elas_trig -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p2_trig_elas
    requires: triangle
    args: -sol_type elas_trig -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_p3_trig_elas
    requires: triangle
    args: -sol_type elas_trig -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_trig_elas
    args: -sol_type elas_trig -simplex 0 -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q1_trig_elas_shear
    args: -sol_type elas_trig -simplex 0 -shear -displacement_petscspace_degree 1 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_trig_elas
    args: -sol_type elas_trig -simplex 0 -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q2_trig_elas_shear
    args: -sol_type elas_trig -simplex 0 -shear -displacement_petscspace_degree 2 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_trig_elas
    args: -sol_type elas_trig -simplex 0 -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate
  test:
    suffix: 2d_q3_trig_elas_shear
    args: -sol_type elas_trig -simplex 0 -shear -displacement_petscspace_degree 3 -dm_refine 1 -convest_num_refine 3 -snes_convergence_estimate

  test:
    suffix: 3d_p1_trig_vlap
    requires: ctetgen
    args: -sol_type vlap_trig -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_trig_vlap
    requires: ctetgen
    args: -sol_type vlap_trig -dim 3 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_p3_trig_vlap
    requires: ctetgen
    args: -sol_type vlap_trig -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q1_trig_vlap
    args: -sol_type vlap_trig -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_trig_vlap
    args: -sol_type vlap_trig -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q3_trig_vlap
    requires: !__float128
    args: -sol_type vlap_trig -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_p1_trig_elas
    requires: ctetgen
    args: -sol_type elas_trig -dim 3 -cells 2,2,2 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_p2_trig_elas
    requires: ctetgen
    args: -sol_type elas_trig -dim 3 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_p3_trig_elas
    requires: ctetgen
    args: -sol_type elas_trig -dim 3 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q1_trig_elas
    args: -sol_type elas_trig -dim 3 -cells 2,2,2 -simplex 0 -displacement_petscspace_degree 1 -convest_num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q2_trig_elas
    args: -sol_type elas_trig -dim 3 -simplex 0 -displacement_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate
  test:
    suffix: 3d_q3_trig_elas
    requires: !__float128
    args: -sol_type elas_trig -dim 3 -simplex 0 -displacement_petscspace_degree 3 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate

  test:
    suffix: 2d_p1_axial_elas
    requires: triangle
    args: -sol_type elas_axial_disp -displacement_petscspace_degree 1 -dm_plex_separate_marker -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p2_axial_elas
    requires: triangle
    args: -sol_type elas_axial_disp -displacement_petscspace_degree 2 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p3_axial_elas
    requires: triangle
    args: -sol_type elas_axial_disp -displacement_petscspace_degree 3 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q1_axial_elas
    args: -sol_type elas_axial_disp -simplex 0 -displacement_petscspace_degree 1 -dm_plex_separate_marker -dm_refine 1 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q2_axial_elas
    args: -sol_type elas_axial_disp -simplex 0 -displacement_petscspace_degree 2 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q3_axial_elas
    args: -sol_type elas_axial_disp -simplex 0 -displacement_petscspace_degree 3 -dm_plex_separate_marker -dmsnes_check .0001 -pc_type lu

  test:
    suffix: 2d_p1_uniform_elas
    requires: triangle
    args: -sol_type elas_uniform_strain -displacement_petscspace_degree 1 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p2_uniform_elas
    requires: triangle
    args: -sol_type elas_uniform_strain -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_p3_uniform_elas
    requires: triangle
    args: -sol_type elas_uniform_strain -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q1_uniform_elas
    args: -sol_type elas_uniform_strain -simplex 0 -displacement_petscspace_degree 1 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q2_uniform_elas
    requires: !single
    args: -sol_type elas_uniform_strain -simplex 0 -displacement_petscspace_degree 2 -dm_refine 2 -dmsnes_check .0001 -pc_type lu
  test:
    suffix: 2d_q3_uniform_elas
    requires: !single
    args: -sol_type elas_uniform_strain -simplex 0 -displacement_petscspace_degree 3 -dm_refine 2 -dmsnes_check .0001 -pc_type lu

TEST*/
