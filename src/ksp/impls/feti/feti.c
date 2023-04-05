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
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (isDir) PetscValidHeaderSpecific(isDir,IS_CLASSID,2);
  /* TODO valid numtype */
  PetscValidLogicalCollectiveBool(ksp,enforce_by_B,4);
  PetscCall(KSPSetUp(ksp));
  PetscTryMethod(ksp,"KSPFETISetDirichlet_C",(KSP,IS,QPFetiNumberingType,PetscBool),(ksp,isDir,numtype,enforce_by_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "KSPQPSSetUp"
static PetscErrorCode KSPQPSSetUp(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  PetscCall(QPSCreate(PetscObjectComm((PetscObject)ksp),&feti->qps));
  PetscCall(QPSSetQP(feti->qps,feti->qp));
  PetscCall(QPSSetFromOptions(feti->qps));
  PetscCall(QPSSetUp(feti->qps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "KSPFETISetUp"
static PetscErrorCode KSPFETISetUp(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  //PetscCall(VecDuplicate(ksp->vec_sol,&feti->x));
  //PetscCall(VecCopy(ksp->vec_sol,feti->x));
  //PetscCall(VecDuplicate(ksp->vec_rhs,&feti->b));
  //PetscCall(VecCopy(ksp->vec_rhs,feti->b));
  feti->x = ksp->vec_sol;
  feti->b = ksp->vec_rhs;
  PetscCall(QPSetRhs(feti->qp,feti->b));
  PetscCall(QPSetInitialVector(feti->qp,feti->x));
  /*PetscCall(QPSetInitialVector(feti->qp,ksp->vec_sol));
  PetscCall(QPSetRhs(feti->qp,ksp->vec_rhs));*/
  PetscCall(QPTMatISToBlockDiag(feti->qp));
  /* FETI chain needs blockDiag */
  PetscCall(QPGetChild(feti->qp,&feti->qp));
  if (feti->isDir) PetscCall(QPFetiSetDirichlet(feti->qp,feti->isDir,feti->dirNumType,feti->dirEnforceExt));
  PetscCall(QPFetiSetUp(feti->qp));
  PetscCall(QPTFromOptions(feti->qp));
  PetscCall(QPGetParent(feti->qp,&feti->qp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_FETI"
PetscErrorCode KSPSetUp_FETI(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;
  Mat A;
  PetscBool ismatis;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp,&A,NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATIS,&ismatis));
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_USER,"Amat should be of type MATIS");

  PetscCall(PetscOptionsInsertString(NULL,"-feti"));

  PetscCall(QPCreate(PetscObjectComm((PetscObject)ksp),&feti->qp));
  PetscCall(QPSetOperator(feti->qp,A));

  /* TODO allow full FETI setup before KSPSolve
  if (feti->b && feti->x) {
    KSPFETISetUp(ksp);
  }
  */
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_FETI"
PetscErrorCode KSPDestroy_FETI(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  //PetscCall(VecDestroy(&feti->b));
  //PetscCall(VecDestroy(&feti->x));
  PetscCall(QPSDestroy(&feti->qps));
  PetscCall(QPDestroy(&feti->qp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETISetDirichlet_C",NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO implement preconditiong
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_FETI"
PetscErrorCode KSPSolve_FETI(KSP ksp)
{
  KSP_FETI *feti = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPFETISetUp(ksp));
  PetscCall(KSPQPSSetUp(ksp));
  PetscCall(QPSSolve(feti->qps));
  PetscCall(QPSGetConvergedReason(feti->qps,&ksp->reason));
  PetscCall(QPSGetIterationNumber(feti->qps,&ksp->its));
  PetscCall(QPGetSolutionVector(feti->qp,&ksp->vec_sol));
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionBegin;
  PetscCall(PetscNew(&feti));
  ksp->data = (void*)feti;

  //TODO norms
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));

  ksp->ops->setup          = KSPSetUp_FETI;
  ksp->ops->solve          = KSPSolve_FETI;
  ksp->ops->destroy        = KSPDestroy_FETI;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,"KSPFETISetDirichlet_C",KSPFETISetDirichlet_FETI));
  PetscFunctionReturn(PETSC_SUCCESS);
}
