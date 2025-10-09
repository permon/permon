/* This file is a stripped-down version of
   src/ksp/pc/impls/redundant/redundant.c
   found in the PETSc source code.

   The original PETSc code is licensed under the BSD 2-Clause "Simplified" License.
   See the LICENSE file in this directory for full terms:
   ./LICENSE or https://gitlab.com/petsc/petsc/-/blob/main/LICENSE
*/

#pragma once

#include <petsc/private/pcimpl.h>
#include <petscksp.h>

typedef struct {
  KSP                ksp;
  PC                 pc;                    /* actual preconditioner used on each processor */
  Vec                xsub, ysub;            /* vectors of a subcommunicator to hold parallel vectors of PetscObjectComm((PetscObject)pc) */
  Vec                xdup, ydup;            /* parallel vector that congregates xsub or ysub facilitating vector scattering */
  Mat                pmats;                 /* matrices belong to a subcommunicator */
  VecScatter         scatterin, scatterout; /* scatter used to move all values to each processor group (subcommunicator) */
  PetscBool          useparallelmat;
  PetscSubcomm       psubcomm;
  PetscInt           nsubcomm; /* num of data structure PetscSubcomm */
  PetscBool          shifttypeset;
  MatFactorShiftType shifttype;
} PC_Redundant;
