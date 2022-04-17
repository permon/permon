/*
  Include "petsctao.h" so we can use TAO solvers
  Include "petscdmda.h" so that we can use distributed arrays (DMs) for managing
  Include "petscksp.h" so we can set KSP type
  the parallel mesh.
*/

#include <petsctao.h>
#include <petscdmda.h>
#include <permonqps.h>

static  char help[]=
"This example demonstrates use of the TAO package to \n\
solve a bound constrained minimization problem.  This example is based on \n\
the problem DPJB from the MINPACK-2 test suite.  This pressure journal \n\
bearing problem is an example of elliptic variational problem defined over \n\
a two dimensional rectangle.  By discretizing the domain into triangular \n\
elements, the pressure surrounding the journal bearing is defined as the \n\
minimum of a quadratic function whose variables are bounded below by zero.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
 \n";

/*T
   Concepts: TAO^Solving a bound constrained minimization problem
   Routines: TaoCreate();
   Routines: TaoSetType(); TaoSetObjectiveAndGradientRoutine();
   Routines: TaoSetHessianRoutine();
   Routines: TaoSetVariableBounds();
   Routines: TaoSetMonitor(); TaoSetConvergenceTest();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve();
   Routines: TaoDestroy();
   Processors: n
T*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunctionGradient(),
   FormHessian().
*/
typedef struct {
  /* problem parameters */
  PetscReal      ecc;          /* test problem parameter */
  PetscReal      b;            /* A dimension of journal bearing */
  PetscInt       nx,ny;        /* discretization in x, y directions */

  /* Working space */
  DM          dm;           /* distributed array data structure */
  Mat         A;            /* Quadratic Objective term */
  Vec         B;            /* Linear Objective term */
  Vec         xl,xu;        /* bounds vectors */
} AppCtx;

/* User-defined routines */
static PetscReal p(PetscReal xi, PetscReal ecc);
static PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *,Vec,void *);
static PetscErrorCode FormHessian(Tao,Vec,Mat, Mat, void *);
static PetscErrorCode ComputeB(AppCtx*);
static PetscErrorCode Monitor(Tao, void*);
static PetscErrorCode ConvergenceTest(Tao, void*);
extern PetscErrorCode CallPermonAndCompareResults(Tao, void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode     ierr;            /* used to check for functions returning nonzeros */
  PetscInt           Nx, Ny;          /* number of processors in x- and y- directions */
  PetscInt           m;               /* number of local elements in vectors */
  Vec                x;               /* variables vector */
  PetscReal          d1000 = 1000;
  PetscBool          flg,testgetdiag; /* A return variable when checking for user options */
  Tao                tao;             /* Tao solver context */
  KSP                ksp;
  AppCtx             user;               /* user-defined work context */
  PetscReal          zero=0.0;           /* lower bound on all variables */
  TaoConvergedReason reason;

  /* Initialize PETSC and PERMON */
  PermonInitialize(&argc, &argv, (char*)0, help);

  /* Set the default values for the problem parameters */
  user.nx = 50; user.ny = 50; user.ecc = 0.1; user.b = 10.0;
  testgetdiag = PETSC_FALSE;

  /* Check for any command line arguments that override defaults */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mx",&user.nx,&flg));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-my",&user.ny,&flg));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-ecc",&user.ecc,&flg));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-b",&user.b,&flg));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_getdiagonal",&testgetdiag,NULL));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n---- Journal Bearing Problem SHB-----\n"));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"mx: %D,  my: %D,  ecc: %g \n\n",user.nx,user.ny,(double)user.ecc));

  /* Let Petsc determine the grid division */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  /*
     A two dimensional distributed array will help define this problem,
     which derives from an elliptic PDE on two dimensional domain.  From
     the distributed array, Create the vectors.
  */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.nx,user.ny,Nx,Ny,1,1,NULL,NULL,&user.dm));
  CHKERRQ(DMSetFromOptions(user.dm));
  CHKERRQ(DMSetUp(user.dm));

  /*
     Extract global and local vectors from DM; the vector user.B is
     used solely as work space for the evaluation of the function,
     gradient, and Hessian.  Duplicate for remaining vectors that are
     the same types.
  */
  CHKERRQ(DMCreateGlobalVector(user.dm,&x)); /* Solution */
  CHKERRQ(VecDuplicate(x,&user.B)); /* Linear objective */


  /*  Create matrix user.A to store quadratic, Create a local ordering scheme. */
  CHKERRQ(VecGetLocalSize(x,&m));
  CHKERRQ(DMCreateMatrix(user.dm,&user.A));

  if (testgetdiag) {
    CHKERRQ(MatSetOperation(user.A,MATOP_GET_DIAGONAL,NULL));
  }

  /* User defined function -- compute linear term of quadratic */
  CHKERRQ(ComputeB(&user));

  /* The TAO code begins here */

  /*
     Create the TAO optimization solver
     Suitable methods: TAOGPCG, TAOBQPIP, TAOTRON, TAOBLMVM
  */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));

  /* Set the initial vector */
  CHKERRQ(VecSet(x, zero));
  CHKERRQ(TaoSetSolution(tao,x));

  /* Set the user function, gradient, hessian evaluation routines and data structures */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void*) &user));

  CHKERRQ(TaoSetHessian(tao,user.A,user.A,FormHessian,(void*)&user));

  /* Set a routine that defines the bounds */
  CHKERRQ(VecDuplicate(x,&user.xl));
  CHKERRQ(VecDuplicate(x,&user.xu));
  CHKERRQ(VecSet(user.xl, zero));
  CHKERRQ(VecSet(user.xu, d1000));
  CHKERRQ(TaoSetVariableBounds(tao,user.xl,user.xu));

  CHKERRQ(TaoGetKSP(tao,&ksp));
  if (ksp) {
    CHKERRQ(KSPSetType(ksp,KSPCG));
  }

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-testmonitor",&flg));
  if (flg) {
    CHKERRQ(TaoSetMonitor(tao,Monitor,&user,NULL));
  }
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-testconvergence",&flg));
  if (flg) {
    CHKERRQ(TaoSetConvergenceTest(tao,ConvergenceTest,&user));
  }

  /* Set default. Check for any TAO command line options. */
  /* Note PERMON/KSP rtol is equivalent to TAO gttol. */
  /* TAO grtol currently has not counterpart in PERMON, so we deactivate it by setting ridiculously small value. */
  CHKERRQ(TaoSetType(tao,TAOTRON));
  CHKERRQ(TaoSetTolerances(tao,1e-8,1e-50,1e-8));
  CHKERRQ(TaoSetFromOptions(tao));

  /* Solve the bound constrained problem */
  CHKERRQ(TaoSolve(tao));
  CHKERRQ(TaoGetConvergedReason(tao, &reason));
  if (reason < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "TAO diverged, reason %D (%s)", reason, TaoConvergedReasons[reason]);
  
  /* Call PERMON solver and compare results */
  CHKERRQ(CallPermonAndCompareResults(tao, &user));

  /* Free PETSc data structures */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&user.xl));
  CHKERRQ(VecDestroy(&user.xu));
  CHKERRQ(MatDestroy(&user.A));
  CHKERRQ(VecDestroy(&user.B));
  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(DMDestroy(&user.dm));

  ierr = PermonFinalize();
  return ierr;
}


static PetscReal p(PetscReal xi, PetscReal ecc)
{
  PetscReal t=1.0+ecc*PetscCosScalar(xi);
  return (t*t*t);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeB"
PetscErrorCode ComputeB(AppCtx* user)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscInt       nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal      two=2.0, pi=4.0*atan(1.0);
  PetscReal      hx,hy,ehxhy;
  PetscReal      temp,*b;
  PetscReal      ecc=user->ecc;

  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  ehxhy = ecc*hx*hy;


  /*
     Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));
  CHKERRQ(DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL));

  /* Compute the linear term in the objective function */
  CHKERRQ(VecGetArray(user->B,&b));
  for (i=xs; i<xs+xm; i++){
    temp=PetscSinScalar((i+1)*hx);
    for (j=ys; j<ys+ym; j++){
      k=xm*(j-ys)+(i-xs);
      b[k]=  - ehxhy*temp;
    }
  }
  CHKERRQ(VecRestoreArray(user->B,&b));
  CHKERRQ(PetscLogFlops(5*xm*ym+3*xm));

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *fcn,Vec G,void *ptr)
{
  AppCtx*        user=(AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k,kk;
  PetscInt       col[5],row,nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal      one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal      hx,hy,hxhy,hxhx,hyhy;
  PetscReal      xi,v[5];
  PetscReal      ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal      vmiddle, vup, vdown, vleft, vright;
  PetscReal      tt,f1,f2;
  PetscReal      *x,*g,zero=0.0;
  Vec            localX;

  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  CHKERRQ(DMGetLocalVector(user->dm,&localX));

  CHKERRQ(DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX));

  CHKERRQ(VecSet(G, zero));
  /*
    Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));
  CHKERRQ(DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL));

  CHKERRQ(VecGetArray(localX,&x));
  CHKERRQ(VecGetArray(G,&g));

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);

    for (j=ys; j<ys+ym; j++){

      row=(j-gys)*gxm + (i-gxs);
       v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;

       k=0;
       if (j>gys){
         v[k]=vdown; col[k]=row - gxm; k++;
       }

       if (i>gxs){
         v[k]= vleft; col[k]=row - 1; k++;
       }

       v[k]= vmiddle; col[k]=row; k++;

       if (i+1 < gxs+gxm){
         v[k]= vright; col[k]=row+1; k++;
       }

       if (j+1 <gys+gym){
         v[k]= vup; col[k] = row+gxm; k++;
       }
       tt=0;
       for (kk=0;kk<k;kk++){
         tt+=v[kk]*x[col[kk]];
       }
       row=(j-ys)*xm + (i-xs);
       g[row]=tt;

     }

  }

  CHKERRQ(VecRestoreArray(localX,&x));
  CHKERRQ(VecRestoreArray(G,&g));

  CHKERRQ(DMRestoreLocalVector(user->dm,&localX));

  CHKERRQ(VecDot(X,G,&f1));
  CHKERRQ(VecDot(user->B,X,&f2));
  CHKERRQ(VecAXPY(G, one, user->B));
  *fcn = f1/2.0 + f2;


  CHKERRQ(PetscLogFlops((91 + 10*ym) * xm));
  return 0;

}


#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/*
   FormHessian computes the quadratic term in the quadratic objective function
   Notice that the objective function in this problem is quadratic (therefore a constant
   hessian).  If using a nonquadratic solver, then you might want to reconsider this function
*/
PetscErrorCode FormHessian(Tao tao,Vec X,Mat hes, Mat Hpre, void *ptr)
{
  AppCtx*        user=(AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscInt       col[5],row,nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal      one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal      hx,hy,hxhy,hxhx,hyhy;
  PetscReal      xi,v[5];
  PetscReal      ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal      vmiddle, vup, vdown, vleft, vright;
  PetscBool      assembled;

  nx=user->nx;
  ny=user->ny;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  /*
    Get local grid boundaries
  */
  CHKERRQ(DMDAGetCorners(user->dm,&xs,&ys,NULL,&xm,&ym,NULL));
  CHKERRQ(DMDAGetGhostCorners(user->dm,&gxs,&gys,NULL,&gxm,&gym,NULL));
  CHKERRQ(MatAssembled(hes,&assembled));
  if (assembled)CHKERRQ(MatZeroEntries(hes));

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);
    v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;

    for (j=ys; j<ys+ym; j++){
      row=(j-gys)*gxm + (i-gxs);

      k=0;
      if (j>gys){
        v[k]=vdown; col[k]=row - gxm; k++;
      }

      if (i>gxs){
        v[k]= vleft; col[k]=row - 1; k++;
      }

      v[k]= vmiddle; col[k]=row; k++;

      if (i+1 < gxs+gxm){
        v[k]= vright; col[k]=row+1; k++;
      }

      if (j+1 <gys+gym){
        v[k]= vup; col[k] = row+gxm; k++;
      }
      CHKERRQ(MatSetValuesLocal(hes,1,&row,k,col,v,INSERT_VALUES));

    }

  }

  /*
     Assemble matrix, using the 2-step process:
     MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  CHKERRQ(MatAssemblyBegin(hes,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(hes,MAT_FINAL_ASSEMBLY));

  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  CHKERRQ(MatSetOption(hes,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  CHKERRQ(MatSetOption(hes,MAT_SYMMETRIC,PETSC_TRUE));

  CHKERRQ(PetscLogFlops(9*xm*ym+49*xm));
  CHKERRQ(MatNorm(hes,NORM_1,&hx));
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(Tao tao, void *ctx)
{
  PetscErrorCode     ierr;
  PetscInt           its;
  PetscReal          f,gnorm,cnorm,xdiff;
  TaoConvergedReason reason;

  PetscFunctionBegin;
  CHKERRQ(TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason));
  if (!(its%5)) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\tf=%g\n",its,(double)f));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConvergenceTest"
PetscErrorCode ConvergenceTest(Tao tao, void *ctx)
{
  PetscErrorCode     ierr;
  PetscInt           its;
  PetscReal          f,gnorm,cnorm,xdiff;
  TaoConvergedReason reason;

  PetscFunctionBegin;
  CHKERRQ(TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason));
  if (its == 100) {
    CHKERRQ(TaoSetConvergedReason(tao,TAO_DIVERGED_MAXITS));
  }
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "CallPermonAndCompareResults"
PetscErrorCode CallPermonAndCompareResults(Tao tao, void *ctx)
{
  PetscErrorCode     ierr;
  AppCtx*        user=(AppCtx*)ctx;
  QP             qp;
  QPS            qps;
  Vec            x_qp;      /* approx solution for qp */
  Vec            x_tao;     /* approx solution for tao */
  Vec            x_diff;
  PetscReal      x_diff_norm;
  PetscReal      rtol,atol,dtol;
  PetscInt       maxit;
  PetscReal      gatol, grtol, gttol;
  PetscReal      rhs_norm;
  PetscReal      tao_diff_tol = 1e2*PETSC_SQRT_MACHINE_EPSILON;
  KSPConvergedReason reason;
  
  PetscFunctionBeginI;
  /* Compute Hessian */
  CHKERRQ(FormHessian(tao, NULL, user->A, NULL, (void*) user));
  
  /* Prescribe the QP problem. */
  CHKERRQ(QPCreate(PETSC_COMM_WORLD, &qp));
  CHKERRQ(QPSetOperator(qp, user->A));
  CHKERRQ(QPSetRhsPlus(qp, user->B));
  CHKERRQ(QPSetBox(qp, NULL, user->xl, user->xu));
  CHKERRQ(VecDuplicate(user->B,&x_diff));
  
  /* Create the QP solver (QPS). */
  CHKERRQ(QPSCreate(PETSC_COMM_WORLD, &qps));
  
  /* Set QPS type to TAO within QPS TAO wrapper. */
  CHKERRQ(QPSSetType(qps,QPSTAO));
  CHKERRQ(QPSTaoSetType(qps,TAOBLMVM));

  /* Insert the QP problem into the solver. */
  CHKERRQ(QPSSetQP(qps, qp));
  
  /* Get Tao tolerances */
  CHKERRQ(TaoGetTolerances(tao, &gatol, &grtol, &gttol));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"TAO tolerances are gatol = %e, grtol =  %e, gttol = %e\n",gatol, grtol, gttol));

  /* Set default QPS options. */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Setting PERMON rtol = gttol, atol = gatol\n"));
  rtol  = gttol;
  atol  = gatol;
  dtol  = PETSC_DEFAULT;
  maxit = PETSC_DEFAULT;
  CHKERRQ(QPSSetTolerances(qps, rtol, atol, dtol, maxit));  

  /* Set the QPS monitor */
  CHKERRQ(QPSMonitorSet(qps,QPSMonitorDefault,NULL,0));
          
  /* Set QPS options from the options database (overriding the defaults). */
  CHKERRQ(QPSSetFromOptions(qps));  

  /* Solve the QP */
  CHKERRQ(QPSSolve(qps));  
  CHKERRQ(QPSGetConvergedReason(qps, &reason));
  if (reason < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED, "QPS diverged, reason %D (%s)", reason, KSPConvergedReasons[reason]);

  /* Get the solution vector */
  CHKERRQ(QPGetSolutionVector(qp, &x_qp));
  CHKERRQ(TaoGetSolution(tao,&x_tao));
  
  /* Difference of results from TAO and QP */
  CHKERRQ(VecNorm(user->B, NORM_2, &rhs_norm));
  tao_diff_tol = 1e1 * PetscMax(rtol*rhs_norm,atol); /* 10 times the tolerance for the default convergence test */
  CHKERRQ(PetscOptionsGetReal(NULL, NULL, "-tao_diff_tol", &tao_diff_tol, NULL));
  CHKERRQ(VecCopy(x_tao, x_diff));
  CHKERRQ(VecAXPY(x_diff, -1.0, x_qp));
  CHKERRQ(VecNorm(x_diff, NORM_2, &x_diff_norm));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)qps),"Norm of difference of results from TAO and QP = %e %s %e = tolerance\n",x_diff_norm, (x_diff_norm <= tao_diff_tol) ? "<=" : ">", tao_diff_tol));
  if (x_diff_norm > tao_diff_tol) SETERRQ(PetscObjectComm((PetscObject)qps), PETSC_ERR_PLIB, "PERMON and TAO yield different results!");
  CHKERRQ(QPSDestroy(&qps));
  CHKERRQ(QPDestroy(&qp));
  CHKERRQ(VecDestroy(&x_diff));
  PetscFunctionReturnI(0);
  
}


/*TEST
  build:
    requires: !complex !single

  testset:
    args: -tao_gttol 1e-6 -qps_view_convergence
    test:
      args: -mx 8 -my 12 
    test:
      suffix: 2
      nsize: 2
      args: -mx 10 -my 16
    test:
      suffix: 3
      nsize: 3
      args: -mx 30 -my 30
    
  testset:
    args: -tao_gttol 1e-6 -qps_view_convergence -qps_type mpgp
    test:
      suffix: 4
      args: -mx 8 -my 12 
    test:
      suffix: 5
      nsize: 2
      args: -mx 10 -my 16
    test:
      suffix: 6
      nsize: 3
      args: -mx 30 -my 30
TEST*/
