
/*
  TODO 
  KSP_MatMult


    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines are
    are actually called via the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPView_XXX()            - Prints details of solver being used.

    Detailed notes:
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option
    -ksp_cg_type symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
    cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include "dcgimpl.h"
//#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/
//extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
//extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

PetscLogEvent KSPDCG_APPLY;

#undef __FUNCT__
#define __FUNCT__ "KSPDCGSetDeflationSpace"
PetscErrorCode KSPDCGSetDeflationSpace(KSP ksp,Mat W,PetscBool transp,PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod((ksp),"KSPDCGSetDeflationSpace_C",(KSP,Mat,PetscBool,PetscInt),(ksp,W,transp,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGSetDeflationSpace_DCG"
static PetscErrorCode KSPDCGSetDeflationSpace_DCG(KSP ksp,Mat W,PetscBool transp,PetscInt n)
{
  KSP_DCG        *cgP = (KSP_DCG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (transp) {
    cgP->Wt = W;
    cgP->W = NULL;
  } else {
    cgP->W = W;
  }
  cgP->spacesize = n;
  ierr = PetscObjectReference((PetscObject)W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGSetMaxNestLvl"
PetscErrorCode KSPDCGSetMaxNestLvl(KSP ksp,PetscInt max)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod((ksp),"KSPDCGSetNestLvl_C",(KSP,PetscInt,PetscInt),(ksp,0,max));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGSetNestLvl_DCG"
static PetscErrorCode KSPDCGSetNestLvl_DCG(KSP ksp,PetscInt current,PetscInt max)
{
  KSP_DCG        *cgP = (KSP_DCG*)ksp->data;

  PetscFunctionBegin;
  cgP->nestedlvl = current;
  cgP->maxnestedlvl = max;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDCGConvergedAdaptive_DCG"
static PetscErrorCode  KSPDCGConvergedAdaptive_DCG(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      scale = *((PetscReal*)ctx);
  KSPNormType    normtype;
  KSP_DCG        *cg = (KSP_DCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,4);
  *reason = KSP_CONVERGED_ITERATING;

  ierr = KSPGetNormType(ksp,&normtype);CHKERRQ(ierr);
  if (normtype == KSP_NORM_NONE) PetscFunctionReturn(0);

  if (!n) {
    /* if user gives initial guess need to compute norm of b */
    if (!ksp->guess_zero) {
      PetscReal snorm = 0.0;
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED || ksp->pc_side == PC_RIGHT) {
        ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of RHS\n");CHKERRQ(ierr);
        ierr = VecNorm(ksp->vec_rhs,NORM_2,&snorm);CHKERRQ(ierr);        /*     <- b'*b */
      } else {
        Vec z;
        /* Should avoid allocating the z vector each time but cannot stash it in cctx because if KSPReset() is called the vector size might change */
        ierr = VecDuplicate(ksp->vec_rhs,&z);CHKERRQ(ierr);
        ierr = KSP_PCApply(ksp,ksp->vec_rhs,z);CHKERRQ(ierr);
        if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
          ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of preconditioned RHS\n");CHKERRQ(ierr);
          ierr = VecNorm(z,NORM_2,&snorm);CHKERRQ(ierr);                 /*    dp <- b'*B'*B*b */
        } else if (ksp->normtype == KSP_NORM_NATURAL) {
          PetscScalar norm;
          ierr  = PetscInfo(ksp,"user has provided nonzero initial guess, computing natural norm of RHS\n");CHKERRQ(ierr);
          ierr  = VecDot(ksp->vec_rhs,z,&norm);CHKERRQ(ierr);
          snorm = PetscSqrtReal(PetscAbsScalar(norm));                            /*    dp <- b'*B*b */
        }
        ierr = VecDestroy(&z);CHKERRQ(ierr);
      }
      /* handle special case of zero RHS and nonzero guess */
      if (!snorm) {
        ierr  = PetscInfo(ksp,"Special case, user has provided nonzero initial guess and zero RHS\n");CHKERRQ(ierr);
        snorm = rnorm;
      }
      else ksp->rnorm0 = snorm;
    } else {
      ksp->rnorm0 = rnorm;
    }
    ksp->ttol = PetscMax(cg->adaptiveconst*ksp->rtol*ksp->rnorm0/scale,ksp->abstol);
    ksp->ttol = PetscMax(ksp->ttol,PETSC_MACHINE_EPSILON);
  }

  if (n <= ksp->chknorm) PetscFunctionReturn(0);

  if (PetscIsInfOrNanReal(rnorm)) {
    PCFailedReason pcreason;
    PetscInt       sendbuf,pcreason_max;
    ierr = PCGetSetUpFailedReason(ksp->pc,&pcreason);CHKERRQ(ierr);
    sendbuf = (PetscInt)pcreason;
    ierr = MPI_Allreduce(&sendbuf,&pcreason_max,1,MPIU_INT,MPIU_MAX,PetscObjectComm((PetscObject)ksp));CHKERRQ(ierr);
    if (pcreason_max) {
      *reason = KSP_DIVERGED_PCSETUP_FAILED;
      ierr    = VecSetInf(ksp->vec_sol);CHKERRQ(ierr);
      ierr    = PetscInfo(ksp,"Linear solver pcsetup fails, declaring divergence \n");CHKERRQ(ierr);
    } else {
      *reason = KSP_DIVERGED_NANORINF;
      ierr    = PetscInfo(ksp,"Linear solver has created a not a number (NaN) as the residual norm, declaring divergence \n");CHKERRQ(ierr);
    }
  } else if (rnorm <= ksp->ttol) {
    if (rnorm < ksp->abstol) {
      ierr    = PetscInfo3(ksp,"Linear solver has converged. Residual norm %14.12e is less than absolute tolerance %14.12e at iteration %D\n",(double)rnorm,(double)ksp->abstol,n);CHKERRQ(ierr);
      *reason = KSP_CONVERGED_ATOL;
    } else {
       ierr = PetscInfo4(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial right hand side norm %14.12e at iteration %D\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n);CHKERRQ(ierr);
      *reason = KSP_CONVERGED_RTOL;
    }
  } else if (rnorm >= ksp->divtol*ksp->rnorm0) {
    ierr    = PetscInfo3(ksp,"Linear solver is diverging. Initial right hand size norm %14.12e, current residual norm %14.12e at iteration %D\n",(double)ksp->rnorm0,(double)rnorm,n);CHKERRQ(ierr);
    *reason = KSP_DIVERGED_DTOL;
  }
  PetscFunctionReturn(0);
}

/*
     A macro used in the following KSPSolve_DCG and KSPSolve_DCG_SingleReduction routines
*/
#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))


/*
    InitCG - Choose initial guess orthogonal to the Krylov subspace of redsiduals
    
    Reference: Erhel and Guyomarc'H, An Augmented Conjugate Gradient Method for Solving
    Consecutive Symmetric Positive Definite Linear Systems, 2000
*/
#undef __FUNCT__
#define __FUNCT__ "KSPDCGInitCG"
static PetscErrorCode KSPDCGInitCG(KSP ksp)
{
  KSP_DCG        *cg = (KSP_DCG*)ksp->data;
  PetscErrorCode ierr;
  Mat            Amat;
  Vec            X,B,R,Z,W1,W2;
  PetscScalar    beta;
  PetscReal      dp;

  PetscFunctionBegin;
  X =  ksp->vec_sol;
  B =  ksp->vec_rhs;
  R =  ksp->work[0];
  Z =  ksp->work[1];
  W1 = cg->work[0];
  W2 = cg->work[1];
  ierr = KSPGetOperators(ksp,&Amat,NULL);CHKERRQ(ierr);
  
  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,R);CHKERRQ(ierr);            /*    r <- b - Ax                      */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                 /*    r <- b (x is 0)                  */
  }
  
  if ( cg->truenorm ) {
    switch (ksp->normtype) {
      case KSP_NORM_PRECONDITIONED:
        ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
        ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
        break;
      case KSP_NORM_UNPRECONDITIONED:
        ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r = e'*A'*A*e            */
        break;
      case KSP_NORM_NATURAL:
        ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
        ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*    beta <- z'*r                      */
        KSPCheckDot(ksp,beta);
        dp = PetscSqrtReal(PetscAbsScalar(beta));                /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
        break;
      case KSP_NORM_NONE:
        dp = 0.0;
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }
    ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
    ksp->rnorm = dp;

    ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);     /* test for convergence */
    if (ksp->reason) PetscFunctionReturn(0);
  }

  ierr = MatMultTranspose(cg->W,R,W1);CHKERRQ(ierr);   /*    x <- x + W*(W'*A*W)^{-1}*W'*r    */ 
  if (cg->adaptiveconv) {
    cg->WtAWinv->rtol = ksp->ttol;
    *((PetscReal*)cg->WtAWinv->cnvP) = ksp->rnorm;
  }
  ierr = KSPSolve(cg->WtAWinv,W1,W2);CHKERRQ(ierr);
  ierr = MatMult(cg->W,W2,R);CHKERRQ(ierr);
  ierr = VecAYPX(X,1.0,R);CHKERRQ(ierr);

  ierr = MatMult(Amat,X,R);CHKERRQ(ierr);              /*    r <- b - Ax                      */
  ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    DeflationApply - 
*/
#undef __FUNCT__
#define __FUNCT__ "KSPDCGDeflationApply"
static PetscErrorCode KSPDCGDeflationApply(KSP ksp)
{
  KSP_DCG        *cgP = (KSP_DCG*)ksp->data;
  PetscErrorCode ierr;
  Mat            Amat;
  Vec            R,Z,P,U,W1,W2;

  PetscFunctionBegin;
  R =  ksp->work[0];
  Z =  ksp->work[1];
  P =  ksp->work[2];
  U =  ksp->work[3];
  W1 = cgP->work[0];
  W2 = cgP->work[1];
  ierr = KSPGetOperators(ksp,&Amat,NULL);CHKERRQ(ierr);
  
  ierr = PetscLogEventBegin(KSPDCG_APPLY,ksp,0,0,0);CHKERRQ(ierr);
  if (!cgP->AW) {
    ierr = MatMult(Amat,Z,U);CHKERRQ(ierr);                   /*    p <- p - W*(W'*A*W)^{-1}*W'*A*z      */ 
    if (cgP->correct) ierr = VecAXPY(U,-1.0,R);CHKERRQ(ierr); /*    p <- p - W*(W'*A*W)^{-1}*W'*(A*z -r) */
    ierr = MatMultTranspose(cgP->W,U,W1);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(cgP->AW,Z,W1);CHKERRQ(ierr);      /*    p <- p - W*(W'*A*W)^{-1}*W'*A*z      */ 
    if (cgP->correct) {                                       /*    p <- p - W*(W'*A*W)^{-1}*W'*(A*z -r) */
      ierr = MatMultTranspose(cgP->W,R,W2);CHKERRQ(ierr);
      ierr = VecAXPY(W1,-1.0,W2);CHKERRQ(ierr);
    }
  }
  if (cgP->adaptiveconv) {
    *((PetscReal*)cgP->WtAWinv->cnvP) = ksp->rnorm;
  }
  ierr = KSPSolve(cgP->WtAWinv,W1,W2);CHKERRQ(ierr);
  ierr = MatMult(cgP->W,W2,Z);CHKERRQ(ierr);
  ierr = VecAXPY(P,-1.0,Z);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(KSPDCG_APPLY,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPSetUp_DCG - Sets up the workspace needed by the DCG method.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_DCG"
static PetscErrorCode KSPSetUp_DCG(KSP ksp)
{
  KSP_DCG        *cgP = (KSP_DCG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       i,maxit = ksp->max_it,nwork = 4,commsize,red,m,size;
  Mat            Amat;
  Mat            W,Wt;
  Mat            nextDefl=NULL,*mats,newmat;
  PC             pc;
  KSP            innerksp;
  PetscBool      match,transp=PETSC_FALSE;
  const char     *prefix;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  if (cgP->W || cgP->Wt) {
    cgP->spacetype = DCG_SPACE_USER;
  } else {
    ierr = KSPDCGComputeDeflationSpace(ksp);CHKERRQ(ierr);
  }

  /* nested DCG */
  if (cgP->W) {
    ierr = PetscObjectTypeCompare((PetscObject)cgP->W,MATPROD,&match);CHKERRQ(ierr);
  } else {
    ierr = MatCreateTranspose(cgP->Wt,&cgP->W);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)cgP->Wt,MATPROD,&match);CHKERRQ(ierr);
    transp = PETSC_TRUE;
  }

  if (match) {
    size = cgP->spacesize;
    ierr = PetscMalloc1(size,&mats);CHKERRQ(ierr);
    for (i=0; i <size; i++) {
      if (transp) {
        ierr = MatProdGetMat(cgP->Wt,i,&mats[i]);CHKERRQ(ierr);
      } else {
        ierr = MatProdGetMat(cgP->W,i,&mats[i]);CHKERRQ(ierr);
      }
      ierr = PetscObjectReference((PetscObject)mats[i]);CHKERRQ(ierr);
    }
    if (cgP->nestedlvl < cgP->maxnestedlvl) {
      if (!transp) {
        W = mats[size-1];
        ierr = PetscObjectReference((PetscObject)mats[size-1]);CHKERRQ(ierr);
      } else {
        Wt = mats[0];
        ierr = PetscObjectReference((PetscObject)mats[0]);CHKERRQ(ierr);
        ierr = MatTranspose(Wt,MAT_INITIAL_MATRIX,&W);CHKERRQ(ierr);
      }
    }
    if (cgP->nestedlvl >= cgP->maxnestedlvl) { /* assemble W */
      if (transp) {
        ierr = MatTranspose(mats[size-1],MAT_INITIAL_MATRIX,&W);CHKERRQ(ierr); 
        for (i=size-1; i>0; i--) { 
          ierr = MatTransposeMatMult(mats[i-1],W,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr); 
          ierr = MatDestroy(&mats[i-1]);CHKERRQ(ierr);
          ierr = MatDestroy(&W);CHKERRQ(ierr); 
          W = newmat; 
        }
      } else {
        W = mats[0];
        for (i=1; i<size; i++) { 
          ierr = MatMatMult(mats[i],W,MAT_INITIAL_MATRIX,PETSC_DECIDE,&newmat);CHKERRQ(ierr); 
          ierr = MatDestroy(&mats[i]);CHKERRQ(ierr);
          ierr = MatDestroy(&W);CHKERRQ(ierr); 
          W = newmat; 
        } 
      }
    } else {
      size -= 1;
      i = 0;
      if (transp) i = 1;
      if (size > 1) { 
        ierr = MatCreateProd(comm,size,&mats[i],&nextDefl);CHKERRQ(ierr);
        for (i=0; i<size+1; i++) ierr = MatDestroy(&mats[i]);CHKERRQ(ierr);
      } else {
        nextDefl = mats[i];
      }
      ierr = MatDestroy(&cgP->W);CHKERRQ(ierr);
      ierr = MatDestroy(&cgP->Wt);CHKERRQ(ierr);
      if (transp) {
        cgP->Wt = Wt;
        ierr = MatCreateTranspose(cgP->Wt,&cgP->W);CHKERRQ(ierr);
      } else {
        cgP->W = W;
      }
    }
    ierr = PetscFree(mats);CHKERRQ(ierr);
  } else {
    if (transp) {
      Wt = cgP->Wt;
      ierr = MatTranspose(cgP->Wt,MAT_INITIAL_MATRIX,&W);CHKERRQ(ierr);
    } else {
      W = cgP->W;
    }
  }  

  /* get work vectors needed by CG */
  if (cgP->singlereduction) nwork += 2;
  ierr = KSPSetWorkVecs(ksp,nwork);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit+1,&cgP->e,maxit+1,&cgP->d,maxit+1,&cgP->ee,maxit+1,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);

    //ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    //ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  ierr = MatGetSize(W,NULL,&m);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Deflation Space size: %d\n",m);
  if (!cgP->WtAWinv) {
    if (!cgP->WtAW) {
      ierr = KSPGetOperators(ksp,&Amat,NULL);CHKERRQ(ierr);
      /* TODO add implicit product version */
      ierr = PetscObjectTypeCompareAny((PetscObject)Amat,&match,MATSEQAIJ,MATMPIAIJ,"");CHKERRQ(ierr);
      if (match) ierr = PetscObjectTypeCompareAny((PetscObject)W,&match,MATSEQAIJ,MATMPIAIJ,"");CHKERRQ(ierr);
      if (!match) {
        if (!cgP->AW) ierr = MatMatMult(Amat,W,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&cgP->AW);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(W,cgP->AW,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&cgP->WtAW);CHKERRQ(ierr);
      } else {
        ierr = MatPtAP(Amat,W,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&cgP->WtAW);CHKERRQ(ierr);
        ierr = MatSetOption(cgP->WtAW,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_DEBUG)
      /* Check WtAW is not sigular */
      PetscReal *norms;
      ierr = PetscMalloc1(m,&norms);CHKERRQ(ierr);
      ierr = MatGetColumnNorms(cgP->WtAW,NORM_INFINITY,norms);CHKERRQ(ierr);
      for (i=0; i<m; i++) {
        if (norms[i] < 10*PETSC_MACHINE_EPSILON) {
          MatView(cgP->W, PETSC_VIEWER_STDOUT_WORLD);
          SETERRQ1(comm,PETSC_ERR_SUP,"Column %d of W is in kernel of A.",i);
        }
      }
      ierr = PetscFree(norms);CHKERRQ(ierr);
#endif
    }
    ierr = KSPCreate(comm,&cgP->WtAWinv);CHKERRQ(ierr);
    ierr = KSPSetOperators(cgP->WtAWinv,cgP->WtAW,cgP->WtAW);CHKERRQ(ierr);
    ierr = KSPGetPC(cgP->WtAWinv,&pc);CHKERRQ(ierr);
    if (nextDefl) {
      ierr = KSPSetType(cgP->WtAWinv,KSPDCG);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
      ierr = KSPDCGSetDeflationSpace(cgP->WtAWinv,nextDefl,transp,size);CHKERRQ(ierr);
      ierr = KSPDCGSetNestLvl_DCG(cgP->WtAWinv,cgP->nestedlvl+1,cgP->maxnestedlvl);CHKERRQ(ierr);
      ((KSP_DCG*)(cgP->WtAWinv->data))->correct = cgP->correct;
      ((KSP_DCG*)(cgP->WtAWinv->data))->adaptiveconv = cgP->adaptiveconv;
      ((KSP_DCG*)(cgP->WtAWinv->data))->adaptiveconst = cgP->adaptiveconst;
      ierr = MatDestroy(&nextDefl);CHKERRQ(ierr);
      innerksp = cgP->WtAWinv;
    } else {
      ierr = KSPSetType(cgP->WtAWinv,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
      /* Redundancy choice */

      red = cgP->redundancy;
      if (red < 0) {
        ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
        red  = ceil((float)commsize/ceil((float)m/commsize));
        ierr = PetscObjectTypeCompareAny((PetscObject)(cgP->WtAW),&match,MATSEQDENSE,MATMPIDENSE,MATDENSE,"");CHKERRQ(ierr);
        if (match) red = commsize;
        PetscPrintf(PETSC_COMM_WORLD,"Auto choosing redundancy %d\n",red);
      }
      ierr = PCRedundantSetNumber(pc,red);CHKERRQ(ierr);
      ierr = PCRedundantGetKSP(pc,&innerksp);CHKERRQ(ierr);
      ierr = KSPGetPC(innerksp,&pc);CHKERRQ(ierr);
      ierr = KSPSetType(innerksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      //TODO remove explicit matSolverPackage
      if (commsize == red) {
        ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU);CHKERRQ(ierr);
      } else {
        ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
      }
    }
    ierr = KSPGetOptionsPrefix(ksp,&prefix);CHKERRQ(ierr);
    if (prefix) {
      ierr = KSPSetOptionsPrefix(innerksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(innerksp,"dcg_");CHKERRQ(ierr);
    } else {
      ierr = KSPSetOptionsPrefix(innerksp,"dcg_");CHKERRQ(ierr);
    }
    /* TODO: check if WtAWinv is KSP and move following from this if */
    ierr = KSPSetFromOptions(cgP->WtAWinv);CHKERRQ(ierr);
    if (cgP->adaptiveconv) {
      PetscReal *rnorm;
      PetscNew(&rnorm);
      ierr = KSPSetConvergenceTest(cgP->WtAWinv,KSPDCGConvergedAdaptive_DCG,rnorm,NULL);CHKERRQ(ierr);
    }
    ierr = KSPSetUp(cgP->WtAWinv);CHKERRQ(ierr);

    Mat WtAW;
    ierr = KSPGetOperators(innerksp,&WtAW,NULL);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"spd %d set %d \n",WtAW->spd,WtAW->spd_set);
  }
  //TODO may not cover all cases
  ierr = KSPCreateVecs(cgP->WtAWinv,2,&cgP->work,0,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*
     KSPSolve_CG - This routine actually applies the conjugate gradient method

     Note : this routine can be replaced with another one (see below) which implements
            another variant of CG.

   Input Parameter:
.     ksp - the Krylov space object that was set to use deflated conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPDCG);
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_DCG"
static PetscErrorCode KSPSolve_DCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi = 0.0,a = 1.0,beta,betaold = 1.0,b = 0,*e = 0,*d = 0,dpiold;
  PetscReal      dp  = 0.0;
  Vec            X,B,Z,R,P,S;
  KSP_DCG        *cg;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  // TODO diag scaling?
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  cg            = (KSP_DCG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  S             = Z;

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  ierr = KSPDCGInitCG(ksp);CHKERRQ(ierr);
  if ( cg->truenorm ) {
    if (ksp->reason) PetscFunctionReturn(0);
    ksp->its = 1;
  }

  switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z = e'*A'*B'*B*A'*e'     */
      break;
    case KSP_NORM_UNPRECONDITIONED:
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r = e'*A'*A*e            */
      break;
    case KSP_NORM_NATURAL:
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*    z <- Br                           */
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*    beta <- z'*r                      */
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));                /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
      break;
    case KSP_NORM_NONE:
      dp = 0.0;
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,ksp->its,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,ksp->its,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);     /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  if (ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) {
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                /*     z <- Br                           */
  }
  if (ksp->normtype != KSP_NORM_NATURAL) {
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                  /*     beta <- z'*r                      */
    KSPCheckDot(ksp,beta);
  }

  i = 0;
  do {
    ksp->its += 1;
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (beta*betaold < 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);                       /*     p <- z                           */
      b    = 0.0;
    } else {
      b = beta/betaold;
      if (eigs) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);                     /*     p <- z + b* p                    */
    }
    if (!cg->initcg) {
      ierr = KSPDCGDeflationApply(ksp);CHKERRQ(ierr);
    }
    dpiold = dpi;
    ierr = KSP_MatMult(ksp,Amat,P,S);CHKERRQ(ierr);            /*     s <- Ap                          */
    ierr = VecXDot(P,S,&dpi);CHKERRQ(ierr);                    /*     dpi <- p's                       */
    KSPCheckDot(ksp,dpi);
    betaold = beta;

    if ((dpi == 0.0) || ((i > 0) && (PetscRealPart(dpi*dpiold) <= 0.0))) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    a = beta/dpi;                                              /*     a = beta/p's                     */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);                       /*     x <- x + ap                      */
    ierr = VecAXPY(R,-a,S);CHKERRQ(ierr);                      /*     r <- r - as                      */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i+2) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br                          */
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*     dp <- z'*z                       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*     dp <- r'*r                       */
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br                          */
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*     beta <- r'*z                     */
      KSPCheckDot(ksp,beta);
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    if (eigs) cg->ned = ksp->its;
    ierr = KSPMonitor(ksp,ksp->its,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,ksp->its,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_PRECONDITIONED && (ksp->normtype != KSP_NORM_NATURAL)) || (ksp->chknorm >= i+2)) {
      ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);               /*     z <- Br                          */
    }
    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i+2)) {
      ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);                 /*     beta <- z'*r                     */
      KSPCheckDot(ksp,beta);
    }

    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*
     KSPDestroy_CG - Frees resources allocated in KSPSetup_CG and clears function
                     compositions from KSPCreate_CG. If adding your own KSP implementation,
                     you must be sure to free all allocated resources here to prevent
                     leaks.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_DCG"
PetscErrorCode KSPDestroy_DCG(KSP ksp)
{
  KSP_DCG         *cg = (KSP_DCG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    ierr = PetscFree4(cg->e,cg->d,cg->ee,cg->dd);CHKERRQ(ierr);
  }
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used.
                  If your Krylov method has special options or flags that information 
                  should be printed here.
*/

#undef __FUNCT__
#define __FUNCT__ "KSPView_DCG"
PetscErrorCode KSPView_DCG(KSP ksp,PetscViewer viewer)
{
  KSP_DCG         *cg = (KSP_DCG*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;
  PetscInt       nIter;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (cg->singlereduction) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CG: using single-reduction variant\n");CHKERRQ(ierr);
    }
    if (cg->adaptiveconv) {
      ierr = PetscViewerASCIIPrintf(viewer,"  DCG: using adaptive precision for inner solve with C=%.1e\n",cg->adaptiveconst);CHKERRQ(ierr);
    }
    if (cg->correct) {
      ierr = PetscViewerASCIIPrintf(viewer,"  DCG: using CP correction\n");CHKERRQ(ierr);
    }
    if (!cg->nestedlvl) {
      ierr = PetscViewerASCIIPrintf(viewer,"  DCG deflation space type: %s\n",KSPDCGSpaceTypes[cg->spacetype]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  DCG %s\n",cg->extendsp ? "extended" : "truncated");CHKERRQ(ierr);
    }
    ierr = KSPGetTotalIterations(ksp,&nIter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  DCG iters: %d\n",nIter);CHKERRQ(ierr);
    ierr = KSPView(cg->WtAWinv,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_DCG"
PetscErrorCode KSPSetFromOptions_DCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode  ierr;
  KSP_DCG         *cg = (KSP_DCG*)ksp->data;

  PetscFunctionBegin;
  /* TODO change to PetscOptionsBegin */
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP DCG options");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsEnum("-ksp_cg_type","Matrix is Hermitian or complex symmetric","KSPCGSetType",KSPCGTypes,(PetscEnum)cg->type,
                          (PetscEnum*)&cg->type,NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsEnum("-ksp_dcg_compute_space","Compute deflation space","KSPDCGSetDeflationSpace",KSPDCGSpaceTypes,(PetscEnum)cg->spacetype,(PetscEnum*)&cg->spacetype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_dcg_compute_space_size","Set size of the deflation space to compute","KSPDCGSetDeflationSpace",cg->spacesize,&cg->spacesize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_dcg_space_extend","extend deflation space instead of truncating","KSPDCG",cg->extendsp,&cg->extendsp,NULL);CHKERRQ(ierr);
//TODO add set function and fix manpages
  ierr = PetscOptionsBool("-ksp_dcg_initcg","Use only initialization step - InitCG","KSPDCG",cg->initcg,&cg->initcg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_dcg_correct","Add Qr to descent direction","KSPDCG",cg->correct,&cg->correct,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_dcg_adaptive","Adaptive stopping criteria","KSPDCG",cg->adaptiveconv,&cg->adaptiveconv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_dcg_adaptive_const","Adaptive stopping criteria constant","KSPDCG",cg->adaptiveconst,&cg->adaptiveconst,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_dcg_rnorm0","set rnorm0 as initcg rnorm","KSPDCG",cg->truenorm,&cg->truenorm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_dcg_redundancy","Number of subgroups for coarse problem solution","KSPDCG",cg->redundancy,&cg->redundancy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_dcg_max_nested_lvl","Maximum of nested DCGs","KSPDCG",cg->maxnestedlvl,&cg->maxnestedlvl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the
                      routine KSPCGSetType() (see the file cgtype.c).
*/
PetscErrorCode  KSPCGSetType_DCG(KSP ksp,KSPCGType type)
{
  KSP_DCG *cg = (KSP_DCG*)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(0);
}

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/
/*MC
     KSPCG - The preconditioned conjugate gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see KSPCGSetType()
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single MPIU_Allreduce() call, see KSPCGUseSingleReduction()

   Level: beginner

   Notes: The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite
          Only left preconditioning is supported.

   For complex numbers there are two different CG methods. One for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   KSPCGSetType() to indicate which type you are using.

   Developer Notes: KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the KSP object.

   References:
.   1. - Magnus R. Hestenes and Eduard Stiefel, Methods of Conjugate Gradients for Solving Linear Systems,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPCGUseSingleReduction(), KSPPIPECG, KSPGROPPCG

M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_DCG"
PETSC_EXTERN PetscErrorCode KSPCreate_DCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_DCG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type = KSP_CG_SYMMETRIC;
#else
  cg->type = KSP_CG_HERMITIAN;
#endif
  cg->initcg        = PETSC_FALSE;
  cg->correct       = PETSC_FALSE;
  cg->truenorm      = PETSC_TRUE;
  cg->redundancy    = -1;
  cg->spacetype     = DCG_SPACE_HAAR;
  cg->spacesize     = 1;
  cg->extendsp      = PETSC_FALSE;
  cg->nestedlvl     = 0;
  cg->maxnestedlvl  = 0;
  cg->adaptiveconv  = PETSC_FALSE;
  cg->adaptiveconst = 1.0;
  ksp->data         = (void*)cg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);

  PetscLogEventRegister("KSPDCG_Apply",KSP_CLASSID,&KSPDCG_APPLY);

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_DCG;
  ksp->ops->solve          = KSPSolve_DCG;
  ksp->ops->destroy        = KSPDestroy_DCG;
  ksp->ops->view           = KSPView_DCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_DCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",KSPCGSetType_DCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPDCGSetDeflationSpace_C",KSPDCGSetDeflationSpace_DCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPDCGSetNestLvl_C",KSPDCGSetNestLvl_DCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
