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
  PetscBool    shear;       /* Shear the domain */
  PetscBool    solverksp;   /* Use KSP as solver */
  /* Problem definition */
  SolutionType solType;     /* Type of exact solution */
  /* Solver definition */
  PetscBool    useNearNullspace; /* Use the rigid body modes as a near nullspace for AMG */
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
  f0[dim-1] = 1.;
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->cells[0]         = 1;
  options->cells[1]         = 1;
  options->cells[2]         = 1;
  options->simplex          = PETSC_TRUE;
  options->shear            = PETSC_FALSE;
  options->solverksp        = PETSC_TRUE;
  options->solType          = SOL_VLAP_QUADRATIC;
  options->useNearNullspace = PETSC_TRUE;
  ierr = PetscStrncpy(options->dmType, DMPLEX, 256);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Linear Elasticity Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex17.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex17.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex17.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-shear", "Shear the domain", "ex17.c", options->shear, &options->shear, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_ksp", "Use KSP as solver", "ex17.c", options->solverksp, &options->solverksp, NULL);CHKERRQ(ierr);
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of exact solution", "ex17.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsBool("-near_nullspace", "Use the rigid body modes as an AMG near nullspace", "ex17.c", options->useNearNullspace, &options->useNearNullspace, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type", "Convert DMPlex to another format", "ex17.c", DMList, options->dmType, options->dmType, 256, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
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
  printf("abc\n");
    if (is) {
  printf("abc2\n");
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
      /* Check for each boundary facet if any component of its centroid is either 0.0 or 1.0 */
      for (f = 0; f < Nf; ++f) {
        PetscReal   faceCoord;
        PetscInt    b,v;
        PetscScalar *coords = NULL;
        PetscInt    Nv;
        /* Get closure of the facet (vertices in 2D, edges in 3D) */
        ierr = DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
        for (i=0; i<csize; i++) {
          printf("%f, ", coords[i]);
        }
        printf("\n");
        Nv   = csize/user->dim;
        /* Calculate mean coordinate vector: sum[xi,yi,zi]/dim */
        for (d = 0; d < user->dim; ++d) {
          faceCoord = 0.0;
          for (v = 0; v < Nv; ++v) faceCoord += PetscRealPart(coords[v*user->dim+d]);
          faceCoord /= Nv;
          printf("%f\n", faceCoord);
          for (b = 0; b < 2; ++b) {
            /* assuming [0,1]^dim */
            if (PetscAbs(faceCoord - b) < PETSC_SMALL) {              ierr = DMSetLabelValue(*dm, "Faces", faces[f], d*2+b+1);CHKERRQ(ierr);
              printf("%d %d %d\n", b, f,d*2+b+1);
            }
          }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(is, &faces);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "Faces", &label);CHKERRQ(ierr);
    DMLabelView(label,PETSC_VIEWER_STDOUT_WORLD);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    DMLabelView(label,PETSC_VIEWER_STDOUT_WORLD);
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
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  switch (user->solType) {
  case SOL_VLAP_QUADRATIC:
    ierr = PetscDSSetResidual(prob, 0, f0_vlap_quadratic_u, f1_vlap_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_vlap_uu);CHKERRQ(ierr);
    switch (dim) {
    case 2: exact = quadratic_2d_u;break;
    case 3: exact = quadratic_3d_u;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid dimension: %D", dim);
    }
    break;
  case SOL_ELAS_QUADRATIC:
    ierr = PetscDSSetResidual(prob, 0, f0_elas_quadratic_u, f1_elas_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_elas_uu);CHKERRQ(ierr);
    switch (dim) {
    case 2: exact = quadratic_2d_u;break;
    case 3: exact = quadratic_3d_u;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid dimension: %D", dim);
    }
    break;
  case SOL_VLAP_TRIG:
    ierr = PetscDSSetResidual(prob, 0, f0_vlap_trig_u, f1_vlap_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_vlap_uu);CHKERRQ(ierr);
    switch (dim) {
    case 2: exact = trig_2d_u;break;
    case 3: exact = trig_3d_u;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid dimension: %D", dim);
    }
    break;
  case SOL_ELAS_TRIG:
    ierr = PetscDSSetResidual(prob, 0, f0_elas_trig_u, f1_elas_u);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_elas_uu);CHKERRQ(ierr);
    switch (dim) {
    case 2: exact = trig_2d_u;break;
    case 3: exact = trig_3d_u;break;
    default: SETERRQ1(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid dimension: %D", dim);
    }
    break;
  case SOL_ELAS_AXIAL_DISP:
    ierr = PetscDSSetResidual(prob, 0, NULL, f1_elas_u);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob, 0, f0_elas_axial_disp_bd_u, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_elas_uu);CHKERRQ(ierr);
    exact = axial_disp_u;
    break;
  case SOL_ELAS_UNIFORM_STRAIN:
   //ierr = PetscDSSetResidual(prob, 0, NULL, f1_elas_u);CHKERRQ(ierr);
  //ierr = PetscDSSetResidual(prob, 0, f0_push_u, f1_elas_u);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 0, f0_push_u, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_elas_uu);CHKERRQ(ierr);
    exact = uniform_strain_u;
    break;
  default: SETERRQ2(PetscObjectComm((PetscObject) prob), PETSC_ERR_ARG_WRONG, "Invalid solution type: %s (%D)", solutionTypes[PetscMin(user->solType, NUM_SOLUTION_TYPES)], user->solType);
  }
  ierr = PetscDSSetExactSolution(prob, 0, exact, user);CHKERRQ(ierr);
//  //if (user->solType == SOL_ELAS_AXIAL_DISP) {
    PetscInt cmp;

    id   = dim == 3 ? 5 : 2;
    //ierr = DMAddBoundary(dm,   DM_BC_NATURAL,   "right",  "marker", 0, 0, NULL, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
    id   = dim == 3 ? 6 : 4;
    id = 1;
    //cmp  = 0;
    ierr = DMAddBoundary(dm,   DM_BC_ESSENTIAL, "left",   "Faces", 0, 0, NULL, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
    DMLabel label;
    ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
    //ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    DMLabelView(label,PETSC_VIEWER_STDOUT_WORLD);
    ierr = DMGetLabel(dm, "Faces", &label);CHKERRQ(ierr);
    //ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    DMLabelView(label,PETSC_VIEWER_STDOUT_WORLD);
    //cmp  = dim == 3 ? 2 : 1;
    id = 3;
    ierr = DMAddBoundary(dm,   DM_BC_ESSENTIAL, "bottom", "Faces", 0, 0, NULL, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
    if (dim == 3) {
      cmp  = 1;
    id   = dim == 3 ? 5 : 3;
      //ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "front",  "marker", 0, 1, &cmp, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
      }
  //  ierr = DMAddBoundary(dm,   DM_BC_NATURAL, "bottom", "marker", 0, 1, &cmp, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
    //if (dim == 3) {
    //  cmp  = 1;
    //  id   = 3;
   //   ierr = DMAddBoundary(dm, DM_BC_NATURAL, "front",  "marker", 0, 1, &cmp, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
//    }
//  //} else {
 // ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, &cmp, (void (*)(void)) exact, NULL, 1, &id, user);CHKERRQ(ierr);
  //ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) zero, NULL, 1, &id, user);CHKERRQ(ierr);
   // PetscInt cmp;

   // id   = dim == 3 ? 5 : 2;
   // ierr = DMAddBoundary(dm,   DM_BC_ESSENTIAL,   "right",  "marker", 0, 0, NULL, (void (*)(void)) cnst, NULL, 1, &id, user);CHKERRQ(ierr);
  //id   = dim == 3 ? 6 : 4;
  //cmp  = 0;
  //ierr = DMAddBoundary(dm,   DM_BC_ESSENTIAL, "left",   "marker", 0, 1, &cmp, (void (*)(void)) cnst, NULL, 1, &id, user);CHKERRQ(ierr);
  //  cmp  = dim == 3 ? 2 : 1;
  //  id   = dim == 3 ? 1 : 1;
  //  ierr = DMAddBoundary(dm,   DM_BC_NATURAL, "bottom", "marker", 0, 1, &cmp, (void (*)(void)) cnst, NULL, 1, &id, user);CHKERRQ(ierr);
  //  if (dim == 3) {
  //    cmp  = 1;
  //    id   = 3;
  //    ierr = DMAddBoundary(dm, DM_BC_NATURAL, "front",  "marker", 0, 1, &cmp, (void (*)(void)) cnst, NULL, 1, &id, user);CHKERRQ(ierr);
  //}
//PetscSection section;
//DMLabel label;
//PetscInt       n, k, p, d;
//  PetscInt       dof, off, num_bc_dofs;
//  IS             is;
//const PetscInt* points;
//PetscInt* bc_indices;
//  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
//  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
//  ierr = DMLabelGetStratumSize(label, 1, &n);CHKERRQ(ierr);
//  ierr = DMLabelGetStratumIS(label, 1, &is);CHKERRQ(ierr);
//  ierr = ISGetIndices(is, &points);CHKERRQ(ierr);
//  num_bc_dofs = 0;
//  for (p = 0; p < n; ++p) {
//    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
//    num_bc_dofs += dof;
//  }
//  ierr = PetscMalloc1(num_bc_dofs, &bc_indices);CHKERRQ(ierr);
//  for (p = 0, k = 0; p < n; ++p) {
//    ierr = PetscSectionGetDof(section, points[p], &dof);CHKERRQ(ierr);
//    ierr = PetscSectionGetOffset(section, points[p], &off);CHKERRQ(ierr);
//    for (d = 0; d < dof; ++d) bc_indices[k++] = off+d;
//		printf("off+d: %d\n",off+d);
//  }
//  ierr = ISRestoreIndices(is, &points);CHKERRQ(ierr);
//  ierr = ISDestroy(&is);CHKERRQ(ierr);
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
  
  if (user.solverksp) {
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
