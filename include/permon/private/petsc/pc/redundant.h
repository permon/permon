
#ifndef __REDUNDANT_H
#define	__REDUNDANT_H

#include <petsc/private/pcimpl.h>
#include <petscksp.h>

typedef struct {
  KSP                ksp;
  PC                 pc;                    /* actual preconditioner used on each processor */
  Vec                xsub, ysub;            /* vectors of a subcommunicator to hold parallel vectors of PetscObjectComm((PetscObject)pc) */
  Vec                xdup, ydup;            /* parallel vector that congregates xsub or ysub facilitating vector scattering */
  Mat                pmats;                 /* matrix and optional preconditioner matrix belong to a subcommunicator */
  VecScatter         scatterin, scatterout; /* scatter used to move all values to each processor group (subcommunicator) */
  PetscBool          useparallelmat;
  PetscSubcomm       psubcomm;
  PetscInt           nsubcomm; /* num of data structure PetscSubcomm */
  PetscBool          shifttypeset;
  MatFactorShiftType shifttype;
} PC_Redundant;

#endif
