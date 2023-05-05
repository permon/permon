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
  ///* duplicate vertices on fracture boundaries */
  //DMView(*dm,PETSC_VIEWER_STDOUT_WORLD);
  //ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  //ierr = MarkVertices2D(*dm,user->nFrac,user->fracCoord,PETSC_DEFAULT);CHKERRQ(ierr);
  //ierr = MarkFractures(*dm,user->nFrac);CHKERRQ(ierr);
  //ierr = SplitFaces(dm,"fracture",user);CHKERRQ(ierr);
  ////ierr = DMSetFromOptions(*dm);CHKERRQ(ierr); /* refine,... */
  //DMView(*dm,PETSC_VIEWER_STDOUT_WORLD);
  //ierr = DMViewFromOptions(*dm, NULL, "-dms_view");CHKERRQ(ierr);
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

TEST*/
