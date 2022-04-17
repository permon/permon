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
    PetscFunctionReturn(0);

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
  CHKERRQ(QPSSetWorkVecs(qps,6));
  if (qps->solQP->cE) {
    CHKERRQ(QPTHomogenizeEq(qps->solQP));
    CHKERRQ(QPChainGetLast(qps->solQP, &qps->solQP));
  }
  PetscFunctionReturn(0);
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
  CHKERRQ(QPSGetSolvedQP(qps,&qp));
 
  CHKERRQ(QPGetOperator(qp, &Amat));
  CHKERRQ(QPGetQPPF(qp, &cp));
  CHKERRQ(QPGetPC(qp, &pc));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc, PCNONE, &pcnone));
  if (!pcnone) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc, PCDUAL, &pcnone));
    if (pcnone) {
      PCDualType type;
      CHKERRQ(PCDualGetType(pc, &type));
      if (type == PC_DUAL_NONE) {
        pcnone = PETSC_TRUE;
      }else{
        pcnone = PETSC_FALSE;
      }
    }
  }

  //initialize
  CHKERRQ(QPGetSolutionVector(qp, &lm));
  CHKERRQ(QPGetRhs(qp, &rhs));
  p = qps->work[0];
  r = qps->work[1];
  w = qps->work[2];
  z = qps->work[3];
  y = qps->work[4];
  Ap = qps->work[5];

  CHKERRQ(MatMult(Amat, lm, r));
  CHKERRQ(VecAYPX(r, -1.0, rhs));
  
  qps->iteration = 0;
  do {
    CHKERRQ(QPPFApplyP(cp, r, w));
    
    //convergence test
    CHKERRQ(VecNorm(w, NORM_2, &qps->rnorm));
    CHKERRQ((*qps->convergencetest)(qps,&qps->reason));
    if (qps->reason) break;
    
    if (pcnone) {
      y = w;
    }else{
      CHKERRQ(PCApply(pc, w, z));
      CHKERRQ(QPPFApplyP(cp, z, y));
    }
    beta2 = beta1;
    CHKERRQ(VecDot(y, w, &beta1)); // beta1 = (y_{i-1},w_{i-1})
    if (!qps->iteration){
      beta = 0;
      CHKERRQ(VecCopy(y, p));
    }else{
      beta = beta1/beta2; //beta = (y_{i-1},w_{i-1})/(y_{i-2},w_{i-2})
      CHKERRQ(VecAYPX(p, beta, y)); //p= y + beta*p
    }
    CHKERRQ(MatMult(Amat,p, Ap));
    CHKERRQ(VecDot(p, Ap, &alpha1));
    alpha = beta1/alpha1;
    CHKERRQ(VecAXPY(lm, alpha, p));
    CHKERRQ(VecAXPY(r, -alpha, Ap ));
    
    qps->iteration++;
  } while (qps->iteration < qps->max_it);
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}
