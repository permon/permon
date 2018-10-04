#if !defined(__PETSCIMPL_H)
#define	__PETSCIMPL_H

#include "petscpc.h"

#if PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>=9
#include "permon/private/petsc/mat/aij.h"
#include "permon/private/petsc/mat/dense.h"
#include "permon/private/petsc/mat/mpiaij.h"
#include "permon/private/petsc/mat/matnestimpl.h"
#include "permon/private/petsc/mat/normm.h"
#include "permon/private/petsc/mat/sbaij.h"
#include "permon/private/petsc/mat/shell.h"
#include "permon/private/petsc/mat/transm.h"
#include "permon/private/petsc/mat/matis.h"
#include "permon/private/petsc/pc/redundant.h"
#else
#error "unsupported PETSc version"
#endif

#endif
