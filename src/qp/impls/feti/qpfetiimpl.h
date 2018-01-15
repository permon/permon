#if !defined(__QPFETIIMPL_H)
#define	__QPFETIIMPL_H
#include <permon/private/qpimpl.h>
#include <permonqpfeti.h>

typedef struct _n_QPFetiDirichlet *QPFetiDirichlet;
struct _n_QPFetiDirichlet {
  IS is;
  QPFetiNumberingType numtype;
  PetscBool enforce_by_B;
};

typedef struct _n_QPFetiCtx *QPFetiCtx;
struct _n_QPFetiCtx {
  /* gluing data */
  IS i2g, l2g;
  ISLocalToGlobalMapping i2g_map, l2g_map;
  
  /* Dirichlet B.C. */
  QPFetiDirichlet dbc;

  PetscBool setupcalled;
};

FLLOP_INTERN PetscErrorCode QPFetiAssembleDirichlet(QP qp);
FLLOP_INTERN PetscErrorCode QPFetiAssembleGluing(QP qp, FetiGluingType type, PetscBool exclude_dir, Mat *Bg_new);
FLLOP_INTERN PetscErrorCode QPFetiGetGlobalDir(QP qp, IS dbc, QPFetiNumberingType numtype, IS *dbc_g);

FLLOP_INTERN PetscErrorCode QPFetiCtxCreate(QPFetiCtx *ctxout);
FLLOP_INTERN PetscErrorCode QPFetiCtxDestroy(QPFetiCtx ctx);
FLLOP_INTERN PetscErrorCode QPFetiGetCtx(QP qp,QPFetiCtx *ctxout);

FLLOP_INTERN PetscErrorCode QPFetiDirichletCreate(IS dbcis, QPFetiNumberingType numtype, PetscBool enforce_by_B, QPFetiDirichlet *dbc);
FLLOP_INTERN PetscErrorCode QPFetiDirichletDestroy(QPFetiDirichlet *dbc);

FLLOP_INTERN PetscErrorCode QPFetiCreateMapMatrix(MPI_Comm comm, PetscInt Nu, PetscInt ni, PetscInt mm_size, IS i2g, Mat *MapMatrix_new);  
FLLOP_INTERN PetscErrorCode QPFetiGetI2Lmapping(MPI_Comm comm, IS l2g, IS i2g, IS *i2l_new);   
FLLOP_INTERN PetscErrorCode QPFetiGetNotOrthoBgtSF(MPI_Comm comm, IS i2g, PetscInt Nu, IS i2l, PetscInt nl, PetscBool full_red, Mat *Bgt_out);
FLLOP_INTERN PetscErrorCode QPFetiGetBgtSF(MPI_Comm comm, IS i2g, PetscInt Nu, IS i2l, PetscInt nl, FetiGluingType type, Mat *Bgt_out);
FLLOP_INTERN PetscErrorCode QPFetiGetOrthonorBgtSF(MPI_Comm comm, IS i2g, PetscInt Nu, IS i2l, PetscInt nl, Mat *Bgt_out);
   
FLLOP_INTERN PetscLogEvent QP_Feti_SetUp, QP_Feti_AssembleGluingFromNeighbors, QP_Feti_AssembleDirichlet;
FLLOP_INTERN PetscLogEvent QP_Feti_AssemGluing, QP_Feti_GetBgtSF, QP_Feti_GetOrthoBgtSF, QP_Feti_GetNotOrthoBgtSF, QP_Feti_GetI2Lmapping, QP_AddEq; 
#endif
