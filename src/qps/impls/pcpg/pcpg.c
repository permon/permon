#include <../src/qps/impls/pcpg/pcpgimpl.h>

#undef __FUNCT__
#define __FUNCT__ "QPSIsQPCompatible_PCPG"
/*
QPSIsQPCompatible_PCPG - verify if the algorithm is able to solve given QP problem

Parameters:
+ qps - QP solver
. qp - quadratic programming problem
- flg - the pointer to result
*/
PetscErrorCode QPSIsQPCompatible_PCPG(QPS qps,QP qp,PetscBool *flg){
    PetscFunctionBegin;
    if (qp->qpc || qp->BI || !qp->BE) {
      *flg = PETSC_FALSE;
    } else {
      *flg = PETSC_TRUE;
    }
    PetscFunctionReturn(PETSC_SUCCESS);

}



#undef __FUNCT__
#define __FUNCT__ "QPSSetup_PCPG"
/*
 * QPSSetup_PCPG - the setup function of PCPG algorithm
 *
 * Parameters:
 * . qps - QP solver
 * */
PetscErrorCode QPSSetup_PCPG(QPS qps){
  PetscFunctionBegin;
  PetscCall(QPSSetWorkVecs(qps,6));
  if (qps->solQP->cE) {
    PetscCall(QPTHomogenizeEq(qps->solQP));
    PetscCall(QPChainGetLast(qps->solQP, &qps->solQP));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSSolve_PCPG"
/*
 * QPSSolve_PCPG - the solver; solve the problem using PCPG algorithm
 *
 * Parameters:
 * . qps - QP solver
 * */
PetscErrorCode QPSSolve_PCPG(QPS qps){
  QP qp;
  QPPF cp;
  PC pc;
  Mat Amat;
  Vec lm; // solution
  Vec p; // search dir
  Vec r; // grad
  Vec w; // proj grad
  Vec z; // precond w
  Vec y; // proj z
  Vec Ap;
  Vec rhs;
  PetscScalar alpha, alpha1, beta, beta1=0, beta2;
  PetscBool pcnone;

  PetscFunctionBegin;
  PetscCall(QPSGetSolvedQP(qps,&qp));

  PetscCall(QPGetOperator(qp, &Amat));
  PetscCall(QPGetQPPF(qp, &cp));
  PetscCall(QPGetPC(qp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCNONE, &pcnone));
  if (!pcnone) {
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCDUAL, &pcnone));
    if (pcnone) {
      PCDualType type;
      PetscCall(PCDualGetType(pc, &type));
      if (type == PC_DUAL_NONE) {
        pcnone = PETSC_TRUE;
      }else{
        pcnone = PETSC_FALSE;
      }
    }
  }

  //initialize
  PetscCall(QPGetSolutionVector(qp, &lm));
  PetscCall(QPGetRhs(qp, &rhs));
  p = qps->work[0];
  r = qps->work[1];
  w = qps->work[2];
  z = qps->work[3];
  y = qps->work[4];
  Ap = qps->work[5];

  PetscCall(MatMult(Amat, lm, r));
  PetscCall(VecAYPX(r, -1.0, rhs));

  qps->iteration = 0;
  do {
    PetscCall(QPPFApplyP(cp, r, w));

    //convergence test
    PetscCall(VecNorm(w, NORM_2, &qps->rnorm));
    PetscCall((*qps->convergencetest)(qps,&qps->reason));
    if (qps->reason) break;

    if (pcnone) {
      y = w;
    }else{
      PetscCall(PCApply(pc, w, z));
      PetscCall(QPPFApplyP(cp, z, y));
    }
    beta2 = beta1;
    PetscCall(VecDot(y, w, &beta1)); // beta1 = (y_{i-1},w_{i-1})
    if (!qps->iteration){
      beta = 0;
      PetscCall(VecCopy(y, p));
    }else{
      beta = beta1/beta2; //beta = (y_{i-1},w_{i-1})/(y_{i-2},w_{i-2})
      PetscCall(VecAYPX(p, beta, y)); //p= y + beta*p
    }
    PetscCall(MatMult(Amat,p, Ap));
    PetscCall(VecDot(p, Ap, &alpha1));
    alpha = beta1/alpha1;
    PetscCall(VecAXPY(lm, alpha, p));
    PetscCall(VecAXPY(r, -alpha, Ap ));

    qps->iteration++;
  } while (qps->iteration < qps->max_it);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "QPSCreate_PCPG"
FLLOP_EXTERN PetscErrorCode QPSCreate_PCPG(QPS qps)
{
  PetscFunctionBegin;
  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  qps->ops->setup = QPSSetup_PCPG;
  qps->ops->solve = QPSSolve_PCPG;
  qps->ops->isqpcompatible = QPSIsQPCompatible_PCPG;
  PetscFunctionReturn(PETSC_SUCCESS);
}
