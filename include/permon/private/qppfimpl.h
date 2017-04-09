#if !defined(__QPPFIMPL_H)
#define	__QPPFIMPL_H
#include <fllopqppf.h>
#include <permon/private/fllopimpl.h>

struct _p_QPPF {
    PETSCHEADER(int);
    
    Mat G, Gt, GGtinv;
    PetscInt Gm,Gn,GM,GN;

    PetscBool explicitInv, G_has_orthonormal_rows_explicitly, G_has_orthonormal_rows_implicitly;
    PetscInt  redundancy;
    PetscReal GGt_relative_fill;

    PetscBool setupcalled, dataChange, variantChange, explicitInvChange, GChange;
    PetscInt setfromoptionscalled;

    /* measurements */
    PetscInt it_GGtinvv;
    KSPConvergedReason conv_GGtinvv;
    
    /* preallocated stuff */
    Vec Gt_right;
    Vec G_left;
    Vec alpha_tilde;

    Vec QPPFApplyQ_last_v;
    Vec QPPFApplyQ_last_Qv;
    PetscObjectState QPPFApplyQ_last_v_state;
};


FLLOP_EXTERN PetscLogEvent QPPF_SetUp, QPPF_SetUp_Gt, QPPF_SetUp_GGt, QPPF_SetUp_GGtinv;
FLLOP_EXTERN PetscLogEvent QPPF_ApplyCP, QPPF_ApplyCP_gt, QPPF_ApplyCP_sc;
FLLOP_EXTERN PetscLogEvent QPPF_ApplyP, QPPF_ApplyQ, QPPF_ApplyHalfQ, QPPF_ApplyG, QPPF_ApplyGt;

#endif
