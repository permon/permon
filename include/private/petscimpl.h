#if !defined(__FLLOPMATSTRUCT_H)
#define	__FLLOPMATSTRUCT_H

#include "petscpc.h"

#if PETSC_VERSION_MINOR==6
#include "private/petsc/3.6/mat/aij.h"
#include "private/petsc/3.6/mat/mpiaij.h"
#include "private/petsc/3.6/mat/matnestimpl.h"
#include "private/petsc/3.6/mat/sbaij.h"
#include "private/petsc/3.6/mat/shell.h"
#include "private/petsc/3.6/pc/redundant.h"
#elif PETSC_VERSION_MINOR==7
#include "private/petsc/3.7/mat/aij.h"
#include "private/petsc/3.7/mat/mpiaij.h"
#include "private/petsc/3.7/mat/matnestimpl.h"
#include "private/petsc/3.7/mat/sbaij.h"
#include "private/petsc/3.7/mat/shell.h"
#include "private/petsc/3.7/pc/redundant.h"
#else
#error "unsupported PETSc version"
#endif

typedef struct {
  Mat A;
} Mat_Transpose;

#endif
