#if !defined(__PERMONQPC_H)
#define	__PERMONQPC_H
#include "permonqppf.h"

typedef struct _p_QPC* QPC;

FLLOP_EXTERN PetscClassId QPC_CLASSID;
#define QPC_CLASS_NAME  "qpc"

#define QPCType char*
#define QPCBOX    "box"

FLLOP_EXTERN PetscErrorCode QPCInitializePackage(void);
FLLOP_EXTERN PetscErrorCode QPCFinalizePackage(void);

FLLOP_EXTERN PetscFunctionList QPCList;
FLLOP_EXTERN PetscBool  QPCRegisterAllCalled;
FLLOP_EXTERN PetscErrorCode QPCRegisterAll(void);
FLLOP_EXTERN PetscErrorCode QPCRegister(const char[],PetscErrorCode (*)(QPC));

FLLOP_EXTERN PetscErrorCode QPCCreate(MPI_Comm comm,QPC *qpc);
FLLOP_EXTERN PetscErrorCode QPCView(QPC qpc,PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPCViewKKT(QPC qpc, Vec x, PetscReal normb, PetscViewer v);
FLLOP_EXTERN PetscErrorCode QPCDestroy(QPC *qpc);
FLLOP_EXTERN PetscErrorCode QPCSetFromOptions(QPC qpc);
FLLOP_EXTERN PetscErrorCode QPCSetUp(QPC qpc);
FLLOP_EXTERN PetscErrorCode QPCReset(QPC qpc);

FLLOP_EXTERN PetscErrorCode QPCSetType(QPC qpc,const QPCType type);
FLLOP_EXTERN PetscErrorCode QPCGetType(QPC qpc,const QPCType *type);

FLLOP_EXTERN PetscErrorCode QPCSetIS(QPC qpc,IS is);
FLLOP_EXTERN PetscErrorCode QPCGetIS(QPC qpc,IS *is);

FLLOP_EXTERN PetscErrorCode QPCSetChangedActiveSet(QPC qpc,PetscBool changed);
FLLOP_EXTERN PetscErrorCode QPCGetActiveSet(QPC qpc,PetscBool global,IS *is);
FLLOP_EXTERN PetscErrorCode QPCGetFreeSet(QPC qpc,PetscBool global,Vec x,IS *is);

FLLOP_EXTERN PetscErrorCode QPCGetSubvector(QPC,Vec x,Vec *xc);
FLLOP_EXTERN PetscErrorCode QPCRestoreSubvector(QPC,Vec x,Vec *xc);

FLLOP_EXTERN PetscErrorCode QPCGetConstraintFunction(QPC,Vec x,Vec *vals);
FLLOP_EXTERN PetscErrorCode QPCRestoreConstraintFunction(QPC,Vec x,Vec *vals);

FLLOP_EXTERN PetscErrorCode QPCGetBlockSize(QPC,PetscInt *bs);
FLLOP_EXTERN PetscErrorCode QPCIsLinear(QPC,PetscBool *linear);
FLLOP_EXTERN PetscErrorCode QPCIsSubsymmetric(QPC,PetscBool *subsym);
FLLOP_EXTERN PetscErrorCode QPCGetNumberOfConstraints(QPC,PetscInt *num);

FLLOP_EXTERN PetscErrorCode QPCProject(QPC,Vec x,Vec Px);
FLLOP_EXTERN PetscErrorCode QPCGrads(QPC,Vec x,Vec g,Vec gf,Vec gc);
FLLOP_EXTERN PetscErrorCode QPCGradReduced(QPC qpc, Vec x, Vec gf, PetscReal alpha, Vec gr);
FLLOP_EXTERN PetscErrorCode QPCFeas(QPC,Vec x,Vec d,PetscScalar *alpha);
FLLOP_EXTERN PetscErrorCode QPCOuterNormal(QPC,PetscScalar *n_a,PetscScalar *xconstr_a,PetscInt local_idx);

/* BOX */
FLLOP_EXTERN PetscErrorCode QPCCreateBox(MPI_Comm comm,IS is,Vec lb,Vec ub,QPC *qpc);
FLLOP_EXTERN PetscErrorCode QPCBoxSet(QPC qpc,Vec lb,Vec ub);
FLLOP_EXTERN PetscErrorCode QPCBoxGet(QPC qpc,Vec *lb,Vec *ub);
FLLOP_EXTERN PetscErrorCode QPCBoxGetMultipliers(QPC qpc,Vec *llb,Vec *lub);

#endif

