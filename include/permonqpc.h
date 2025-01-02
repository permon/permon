#pragma once

#include "permonqppf.h"

typedef struct _p_QPC *QPC;

PERMON_EXTERN PetscClassId QPC_CLASSID;
#define QPC_CLASS_NAME "qpc"

#define QPCType char *
#define QPCBOX  "box"

PERMON_EXTERN PetscErrorCode QPCInitializePackage(void);
PERMON_EXTERN PetscErrorCode QPCFinalizePackage(void);

PERMON_EXTERN PetscFunctionList QPCList;
PERMON_EXTERN PetscBool         QPCRegisterAllCalled;
PERMON_EXTERN PetscErrorCode    QPCRegisterAll(void);
PERMON_EXTERN PetscErrorCode    QPCRegister(const char[], PetscErrorCode (*)(QPC));

PERMON_EXTERN PetscErrorCode QPCCreate(MPI_Comm comm, QPC *qpc);
PERMON_EXTERN PetscErrorCode QPCView(QPC qpc, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPCViewKKT(QPC qpc, Vec x, PetscReal normb, PetscViewer v);
PERMON_EXTERN PetscErrorCode QPCDestroy(QPC *qpc);
PERMON_EXTERN PetscErrorCode QPCSetFromOptions(QPC qpc);
PERMON_EXTERN PetscErrorCode QPCSetUp(QPC qpc);
PERMON_EXTERN PetscErrorCode QPCReset(QPC qpc);

PERMON_EXTERN PetscErrorCode QPCSetType(QPC qpc, const QPCType type);
PERMON_EXTERN PetscErrorCode QPCSetIS(QPC, IS is);

PERMON_EXTERN PetscErrorCode QPCGetType(QPC qpc, const QPCType *type);
PERMON_EXTERN PetscErrorCode QPCGetIS(QPC, IS *is);

PERMON_EXTERN PetscErrorCode QPCGetSubvector(QPC, Vec x, Vec *xc);
PERMON_EXTERN PetscErrorCode QPCRestoreSubvector(QPC, Vec x, Vec *xc);

PERMON_EXTERN PetscErrorCode QPCGetConstraintFunction(QPC, Vec x, Vec *vals);
PERMON_EXTERN PetscErrorCode QPCRestoreConstraintFunction(QPC, Vec x, Vec *vals);

PERMON_EXTERN PetscErrorCode QPCGetBlockSize(QPC, PetscInt *bs);
PERMON_EXTERN PetscErrorCode QPCIsLinear(QPC, PetscBool *linear);
PERMON_EXTERN PetscErrorCode QPCIsSubsymmetric(QPC, PetscBool *subsym);
PERMON_EXTERN PetscErrorCode QPCGetNumberOfConstraints(QPC, PetscInt *num);

PERMON_EXTERN PetscErrorCode QPCProject(QPC, Vec x, Vec Px);
PERMON_EXTERN PetscErrorCode QPCGrads(QPC, Vec x, Vec g, Vec gf, Vec gc);
PERMON_EXTERN PetscErrorCode QPCGradReduced(QPC qpc, Vec x, Vec gf, PetscReal alpha, Vec gr);
PERMON_EXTERN PetscErrorCode QPCFeas(QPC, Vec x, Vec d, PetscScalar *alpha);
PERMON_EXTERN PetscErrorCode QPCOuterNormal(QPC, PetscScalar *n_a, PetscScalar *xconstr_a, PetscInt local_idx);

/* BOX */
PERMON_EXTERN PetscErrorCode QPCCreateBox(MPI_Comm comm, IS is, Vec lb, Vec ub, QPC *qpc);
PERMON_EXTERN PetscErrorCode QPCBoxSet(QPC qpc, Vec lb, Vec ub);
PERMON_EXTERN PetscErrorCode QPCBoxGet(QPC qpc, Vec *lb, Vec *ub);
PERMON_EXTERN PetscErrorCode QPCBoxGetMultipliers(QPC qpc, Vec *llb, Vec *lub);
