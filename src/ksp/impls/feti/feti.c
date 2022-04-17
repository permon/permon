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
  CHKERRQ(KSPSetUp(ksp));
  PetscTryMethod(ksp,"KSPFETISetDirichlet_C",(KSP,IS,QPFetiNumberingType,PetscBool),(ksp,isDir,numtype,enforce_by_B));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPQPSSetUp"
static PetscErrorCode KSPQPSSetUp(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(QPSCreate(PetscObjectComm((PetscObject)ksp),&feti->qps));
  CHKERRQ(QPSSetQP(feti->qps,feti->qp));
  CHKERRQ(QPSSetFromOptions(feti->qps));
  CHKERRQ(QPSSetUp(feti->qps));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETISetUp"
static PetscErrorCode KSPFETISetUp(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //CHKERRQ(VecDuplicate(ksp->vec_sol,&feti->x));
  //CHKERRQ(VecCopy(ksp->vec_sol,feti->x));
  //CHKERRQ(VecDuplicate(ksp->vec_rhs,&feti->b));
  //CHKERRQ(VecCopy(ksp->vec_rhs,feti->b));
  feti->x = ksp->vec_sol;
  feti->b = ksp->vec_rhs;
  CHKERRQ(QPSetRhs(feti->qp,feti->b));
  CHKERRQ(QPSetInitialVector(feti->qp,feti->x));
  /*CHKERRQ(QPSetInitialVector(feti->qp,ksp->vec_sol));
  CHKERRQ(QPSetRhs(feti->qp,ksp->vec_rhs));*/
  CHKERRQ(QPTMatISToBlockDiag(feti->qp));
  /* FETI chain needs blockDiag */
  CHKERRQ(QPGetChild(feti->qp,&feti->qp));
  if (feti->isDir) CHKERRQ(QPFetiSetDirichlet(feti->qp,feti->isDir,feti->dirNumType,feti->dirEnforceExt));
  CHKERRQ(QPFetiSetUp(feti->qp));
  CHKERRQ(QPTFromOptions(feti->qp));
  CHKERRQ(QPGetParent(feti->qp,&feti->qp));
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
  CHKERRQ(KSPGetOperators(ksp,&A,NULL));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis));
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Amat should be of type MATIS");

  CHKERRQ(PetscOptionsInsertString(NULL,"-feti"));

  CHKERRQ(QPCreate(PetscObjectComm((PetscObject)ksp),&feti->qp));
  CHKERRQ(QPSetOperator(feti->qp,A));

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
  //CHKERRQ(VecDestroy(&feti->b));
  //CHKERRQ(VecDestroy(&feti->x));
  CHKERRQ(QPSDestroy(&feti->qps));
  CHKERRQ(QPDestroy(&feti->qp));
  CHKERRQ(KSPDestroyDefault(ksp));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETISetDirichlet_C",NULL));
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
  CHKERRQ(KSPFETISetUp(ksp));
  CHKERRQ(KSPQPSSetUp(ksp));
  CHKERRQ(QPSSolve(feti->qps));
  CHKERRQ(QPSGetConvergedReason(feti->qps,&ksp->reason));
  CHKERRQ(QPSGetIterationNumber(feti->qps,&ksp->its));
  CHKERRQ(QPGetSolutionVector(feti->qp,&ksp->vec_sol));
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
  CHKERRQ(PetscNewLog(ksp,&feti));
  ksp->data = (void*)feti;

  //TODO norms
  TRY( KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2) );

  ksp->ops->setup          = KSPSetUp_FETI;
  ksp->ops->solve          = KSPSolve_FETI;
  ksp->ops->destroy        = KSPDestroy_FETI;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETISetDirichlet_C",KSPFETISetDirichlet_FETI));
  PetscFunctionReturn(0);
}
