#include <permonqps.h>
#include <permonqpfeti.h>
#include <permon/private/permonkspimpl.h>

typedef struct {
  QP qp;
  QPS qps;
  Vec b,x;
  IS isDir;
  QPFetiNumberingType dirNumType;
  PetscBool dirEnforceExt;
  /* TODO bool setup */
} KSP_FETI;

#undef __FUNCT__
#define __FUNCT__ "KSPFETISetDirichlet_FETI"
static PetscErrorCode KSPFETISetDirichlet_FETI(KSP ksp,IS isDir,QPFetiNumberingType numtype,PetscBool enforce_by_B)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  feti->isDir = isDir;
  feti->dirNumType = numtype;
  feti->dirEnforceExt = enforce_by_B;
  PetscFunctionReturn(0);
}

/*@
 KSPFETISetDirichlet - Sets the Dirichlet boundary conditions.

   Input Parameters:
+  ksp          - the FETI Krylov solver
.  isDir        - index set of dirichlet DOFs
.  numtype      - what numbering is isDir in. One of FETI_LOCAL, FETI_GLOBAL_DECOMPOSED, FETI_GLOBAL_UNDECOMPOSED
-  enforce_by_B - If true, the Dirichlet BC are enforced by equality constraint (TFETI approach - recommended), else the Hessian is modified using MatZeroRowsColumnsIS (classic FETI approach).

   Level: intermediate

.seealso: QPFETISetDirichlet
@*/
#undef __FUNCT__
#define __FUNCT__ "KSPFETISetDirichlet"
PetscErrorCode KSPFETISetDirichlet(KSP ksp,IS isDir,QPFetiNumberingType numtype,PetscBool enforce_by_B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (isDir) PetscValidHeaderSpecific(isDir,IS_CLASSID,2);
  /* TODO valid numtype */
  PetscValidLogicalCollectiveBool(ksp,enforce_by_B,4);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = PetscTryMethod(ksp,"KSPFETISetDirichlet_C",(KSP,IS,QPFetiNumberingType,PetscBool),(ksp,isDir,numtype,enforce_by_B));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPQPSSetUp"
static PetscErrorCode KSPQPSSetUp(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QPSCreate(PetscObjectComm((PetscObject)ksp),&feti->qps);CHKERRQ(ierr);
  ierr = QPSSetQP(feti->qps,feti->qp);CHKERRQ(ierr);
  ierr = QPSSetFromOptions(feti->qps);CHKERRQ(ierr);
  ierr = QPSSetUp(feti->qps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETISetUp"
static PetscErrorCode KSPFETISetUp(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = VecDuplicate(ksp->vec_sol,&feti->x);CHKERRQ(ierr);
  //ierr = VecCopy(ksp->vec_sol,feti->x);CHKERRQ(ierr);
  //ierr = VecDuplicate(ksp->vec_rhs,&feti->b);CHKERRQ(ierr);
  //ierr = VecCopy(ksp->vec_rhs,feti->b);CHKERRQ(ierr);
  feti->x = ksp->vec_sol;
  feti->b = ksp->vec_rhs;
  ierr = QPSetRhs(feti->qp,feti->b);CHKERRQ(ierr);
  ierr = QPSetInitialVector(feti->qp,feti->x);CHKERRQ(ierr);
  /*ierr = QPSetInitialVector(feti->qp,ksp->vec_sol);CHKERRQ(ierr);
  ierr = QPSetRhs(feti->qp,ksp->vec_rhs);CHKERRQ(ierr);*/
  ierr = QPTMatISToBlockDiag(feti->qp);CHKERRQ(ierr);
  /* FETI chain needs blockDiag */
  ierr = QPGetChild(feti->qp,&feti->qp);CHKERRQ(ierr);
  if (feti->isDir) ierr = QPFetiSetDirichlet(feti->qp,feti->isDir,feti->dirNumType,feti->dirEnforceExt);CHKERRQ(ierr);
  ierr = QPFetiSetUp(feti->qp);CHKERRQ(ierr);
  ierr = QPTFromOptions(feti->qp);CHKERRQ(ierr);
  ierr = QPGetParent(feti->qp,&feti->qp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_FETI"
PetscErrorCode KSPSetUp_FETI(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  Mat A;
  PetscBool ismatis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Amat should be of type MATIS");

  ierr = PetscOptionsInsertString(NULL,"-feti");CHKERRQ(ierr);

  ierr = QPCreate(PetscObjectComm((PetscObject)ksp),&feti->qp);CHKERRQ(ierr);
  ierr = QPSetOperator(feti->qp,A);CHKERRQ(ierr);

  /* TODO allow full FETI setup before KSPSolve
  if (feti->b && feti->x) {
    KSPFETISetUp(ksp);
  }
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_FETI"
PetscErrorCode KSPDestroy_FETI(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = VecDestroy(&feti->b);CHKERRQ(ierr);
  //ierr = VecDestroy(&feti->x);CHKERRQ(ierr);
  ierr = QPSDestroy(&feti->qps);CHKERRQ(ierr);
  ierr = QPDestroy(&feti->qp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETISetDirichlet_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//TODO implement preconditiong
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_FETI"
PetscErrorCode KSPSolve_FETI(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPFETISetUp(ksp);CHKERRQ(ierr);
  ierr = KSPQPSSetUp(ksp);CHKERRQ(ierr);
  ierr = QPSSolve(feti->qps);CHKERRQ(ierr);
  ierr = QPSGetConvergedReason(feti->qps,&ksp->reason);CHKERRQ(ierr);
  ierr = QPSGetIterationNumber(feti->qps,&ksp->its);CHKERRQ(ierr);
  ierr = QPGetSolutionVector(feti->qp,&ksp->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  KSPFETI - The FETI and Total FETI (TFETI) method.

  Thin KSP wrapper for PermonFLLOP implementation of (T)FETI.
  The matrix for the KSP must be of type MATIS.

  Options Database Keys:

  Level: beginner

  References:
. 1. - Farhat
M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_FETI"
FLLOP_EXTERN PetscErrorCode KSPCreate_FETI(KSP ksp)
{
  KSP_FETI *feti;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&feti);CHKERRQ(ierr);
  ksp->data = (void*)feti;

  //TODO norms
  TRY( KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2) );

  ksp->ops->setup          = KSPSetUp_FETI;
  ksp->ops->solve          = KSPSolve_FETI;
  ksp->ops->destroy        = KSPDestroy_FETI;

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPFETISetDirichlet_C",KSPFETISetDirichlet_FETI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
