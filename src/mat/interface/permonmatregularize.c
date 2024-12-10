#include <permon/private/permonmatimpl.h>

PetscLogEvent Mat_Regularize;

#undef __FUNCT__
#define __FUNCT__ "MatRegularize_GetPivots_Private"
static PetscErrorCode MatRegularize_GetPivots_Private(Mat R, IS *pivots) {

  PetscInt p, npivots, maxdp;
  PetscInt *idx_arr;
  PetscScalar *v1, *v2, *v1p, *v2p;
  PetscScalar vpivot, alpha, t;
  PetscInt i, j, II, J, ipivot, jpivot;
  PetscInt *all, *perm;
  Mat R_work,R_work1;

  PetscFunctionBeginI;
  /*TODO: fix this workaround for prealloc check in MatDenseGetLDA */
  PetscCall(MatConvert(R,MATDENSE,MAT_INITIAL_MATRIX,&R_work1));
  PetscCall(MatConvert(R_work1,MATDENSEPERMON,MAT_INITIAL_MATRIX,&R_work));
  PetscCall(MatDestroy(&R_work1));
  PetscCall(MatGetSize(R_work, &p, &npivots));
  maxdp = PetscMax(npivots, p);
  PetscCall(PetscMalloc(maxdp * sizeof (PetscInt),    &all));
  PetscCall(PetscMalloc(p     * sizeof (PetscInt),    &perm));
  PetscCall(PetscMalloc(p     * sizeof (PetscScalar), &v1));
  PetscCall(PetscMalloc(p     * sizeof (PetscScalar), &v2));

  /* all = 0 : max(d,p)-1 */
  for (i = 0; i < maxdp; i++)
    all[i] = i;

  /* perm = 0 : p-1 */
  for (i = 0; i < p; i++)
    perm[i] = i;

  /* main iteration through all columns from the last to the first */
  for (J = npivots - 1, II = p - 1; J >= 0; J--, II--) {

    /* find i,j,v of a pivot */
    vpivot = 0.0;
    for (j = 0; j <= J; j++) {
      PetscCall(MatGetValues(R_work, p, all, 1, &j, v1));

      /* [vpivot,ipivot] = max(abs(R(0:p-1, j))) */
      for (i = 0; i <= II; i++) {
        if (PetscAbsScalar(v1[i]) > PetscAbsScalar(vpivot)) {
          ipivot = i;
          jpivot = j;
          vpivot = v1[i];
        }
      }
    }

    /* swap rows ipivot and II:
     * v1 = R(ipivot,:);  R(ipivot,:) = R(II,:);  R(II,:) = v1;
     * t  = perm(ipivot); perm(ipivot)= perm(II); perm(II) = t;
     */
    PetscCall(MatGetValues(R_work, 1, &ipivot, J + 1, all, v1));
    PetscCall(MatGetValues(R_work, 1, &II,      J + 1, all, v2));
    PetscCall(MatSetValues(R_work, 1, &ipivot, J + 1, all, v2, INSERT_VALUES));
    PetscCall(MatSetValues(R_work, 1, &II,      J + 1, all, v1, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(R_work, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(  R_work, MAT_FINAL_ASSEMBLY));
    t = perm[ipivot];
    perm[ipivot] = perm[II];
    perm[II] = t;

    /* swap columns jpivot and J:
     * v1 = R(:,jpivot);  R(:,jpivot) = R(:,J); R(:,J) = v1;
     */
    PetscCall(MatGetValues(R_work, II + 1, all, 1, &jpivot, v1));
    PetscCall(MatGetValues(R_work, II + 1, all, 1, &J,      v2));
    PetscCall(MatSetValues(R_work, II + 1, all, 1, &J,      v1, INSERT_VALUES));
    PetscCall(MatSetValues(R_work, II + 1, all, 1, &jpivot, v2, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(R_work, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(  R_work, MAT_FINAL_ASSEMBLY));

    /* columnwise elimination of the row II */
    for (j = 0; j <= J - 1; j++) {
      /* v2 = R(:,j) */
      PetscCall(MatGetValues(R_work, II + 1, all, 1, &j, v2));

      if (PetscAbsScalar(v2[II]) < PETSC_MACHINE_EPSILON)
        continue;

      /* v2 = -(vpivot/v2(II)) * v2 */
      alpha = -vpivot / v2[II];
      v1p = v1;
      v2p = v2;
      for (i = 0; i <= II; i++) {
        *v2p *= alpha;
        *v2p += *v1p;
        v1p++;
        v2p++;
      }

      /* R(:,j) = v2 */
      PetscCall(MatSetValues(R_work, II + 1, all, 1, &j, v2, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(R_work, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(  R_work, MAT_FINAL_ASSEMBLY));
    }
  }

  /* idx_arr points to last d entries of perm */
  idx_arr = &perm[p - npivots];

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, npivots, idx_arr, PETSC_COPY_VALUES, pivots));
  PetscCall(ISSort(*pivots));

  PetscCall(MatDestroy(&R_work));
  PetscCall(PetscFree(all));
  PetscCall(PetscFree(perm));
  PetscCall(PetscFree(v1));
  PetscCall(PetscFree(v2));
  PetscFunctionReturnI(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatRegularize_GetRegularization_Private"
static PetscErrorCode MatRegularize_GetRegularization_Private(Mat K_loc, Mat R_loc, IS pivots, Mat *newQ)
{
  Mat Q_loc_condensed=NULL, Q_loc=NULL;
  PetscInt prim_loc, defect_loc, npivots, i, j, ncols;
  const PetscInt *pivarr, *cols;
  const PetscScalar *vals;
  PetscInt *Q_allocation;
  MatType K_type;

  PetscFunctionBegin;
  PetscCall(MatGetSize(R_loc, &prim_loc, &defect_loc));
  PetscCall(ISGetSize(pivots, &npivots));

  if (npivots) {
    Mat RI, RIt, RItRI, iRItRI, invRItRI, RI_invRItRI, Q_loc_condensed_filtered;
    IS R_all_cols;

    /* R_all_cols = 0:1:defect_loc-1 */
    PetscCall(ISCreateStride(PETSC_COMM_SELF, defect_loc, 0, 1, &R_all_cols));

    PetscCall(MatCreateSubMatrix(R_loc, pivots, R_all_cols, MAT_INITIAL_MATRIX, &RI));
    PetscCall(MatConvert(RI,MATDENSE,MAT_INPLACE_MATRIX,&RI));
    PetscCall(ISDestroy(&R_all_cols));

    PetscCall(MatTranspose(RI, MAT_INITIAL_MATRIX, &RIt));
    PetscCall(MatMatMult(RIt, RI, MAT_INITIAL_MATRIX, 1.0, &RItRI));

    PetscCall(MatCreateInv(RItRI, MAT_INV_MONOLITHIC, &invRItRI));
    PetscCall(MatInvExplicitly(invRItRI, PETSC_FALSE, MAT_INITIAL_MATRIX, &iRItRI));
    PetscCall(MatDestroy(&invRItRI));
    invRItRI = iRItRI;

    PetscCall(MatMatMult(RI, invRItRI, MAT_INITIAL_MATRIX, 1.0, &RI_invRItRI));
    PetscCall(MatMatMult(RI_invRItRI, RIt, MAT_INITIAL_MATRIX, 1.0, &Q_loc_condensed));

    PetscCall(MatFilterZeros(Q_loc_condensed, PETSC_MACHINE_EPSILON*10, &Q_loc_condensed_filtered));
    PetscCall(MatDestroy(&Q_loc_condensed));
    Q_loc_condensed = Q_loc_condensed_filtered;

    PetscCall(MatDestroy(&RI));
    PetscCall(MatDestroy(&RIt));
    PetscCall(MatDestroy(&invRItRI));
    PetscCall(MatDestroy(&RI_invRItRI));
  }

  /* prepare preallocation array for Q_loc */
  PetscCall(PetscMalloc(prim_loc*sizeof(PetscInt), &Q_allocation));
  PetscCall(PetscMemzero(Q_allocation,prim_loc*sizeof(PetscInt)));
  PetscCall(ISGetIndices(pivots, &pivarr));
  for (i=0; i<npivots; i++) {
    PetscCall(MatGetRow(    Q_loc_condensed,i,&ncols,NULL,NULL));
    Q_allocation[pivarr[i]] = ncols;
    PetscCall(MatRestoreRow(Q_loc_condensed,i,&ncols,NULL,NULL));
  }
  PetscCall(ISRestoreIndices(pivots, &pivarr));

  /* allocate Q_loc */
  PetscCall(MatCreate(PETSC_COMM_SELF, &Q_loc));
  PetscCall(MatSetSizes(Q_loc, prim_loc, prim_loc, prim_loc, prim_loc));
  PetscCall(MatGetType(K_loc,&K_type));
  PetscCall(MatSetType(Q_loc, K_type));
  PetscCall(MatSeqAIJSetPreallocation(Q_loc, -1, Q_allocation));
  PetscCall(MatSeqSBAIJSetPreallocation(Q_loc, 1, -1, Q_allocation));
  PetscCall(MatSetUp(Q_loc));
  PetscCall(PetscFree(Q_allocation));

  /* insert values of Q_loc_condensed to appropriate positions in Q_loc */
  for (i=0; i<npivots; i++) {
    PetscCall(MatGetRow(    Q_loc_condensed,i,&ncols,&cols,&vals));
    for (j=0; j<ncols; j++) {
      PetscCall(MatSetValue(Q_loc, pivarr[i], pivarr[cols[j]], vals[j], INSERT_VALUES));
    }
    PetscCall(MatRestoreRow(Q_loc_condensed,i,&ncols,&cols,&vals));
  }
  PetscCall(MatAssemblyBegin(Q_loc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(  Q_loc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatInheritSymmetry(K_loc,Q_loc));
  PetscCall(MatDestroy(&Q_loc_condensed));
  *newQ = Q_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef __FUNCT__
#define __FUNCT__ "MatRegularize"
PetscErrorCode MatRegularize(Mat K, Mat R, MatRegularizationType type, Mat *newKreg) {
  static PetscBool      registered = PETSC_FALSE;
  static PetscInt       regularized_id;
  MPI_Comm              comm;
  IS                    pivots;
  Mat                   Q_loc, Kreg;
  PetscScalar           rho;
  Mat                   K_loc, R_loc;
  PETSC_UNUSED PetscInt regularized_int;
  PetscBool             regularized = PETSC_FALSE;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(K,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(K,type,3);
  PetscAssertPointer(newKreg,4);

  if (type == MAT_REG_NONE) {
    PetscCall(PetscInfo(K,"MatRegularizationType set to MAT_REG_NONE, returning input matrix\n"));
    *newKreg = K;
    PetscCall(PetscObjectReference((PetscObject)K));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (!registered) {
    PetscCall(PetscLogEventRegister(__FUNCT__, MAT_CLASSID, &Mat_Regularize));
    registered = PETSC_TRUE;
    PetscCall(PetscObjectComposedDataRegister(&regularized_id));
  }

  PetscCall(PetscObjectComposedDataGetInt((PetscObject)K,regularized_id,regularized_int,regularized));
  if (regularized) {
    PetscCall(PetscInfo(K,"matrix marked as regularized, returning input matrix\n"));
    *newKreg = K;
    PetscCall(PetscObjectReference((PetscObject)K));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  PetscCheckSameComm(K,1,R,2);
  PetscCall(PetscObjectGetComm((PetscObject)K,&comm));
  if (K->rmap->n!=K->cmap->n) SETERRQ(comm,PETSC_ERR_ARG_SIZ,"Matrix #1 must be locally square");
  if (K->rmap->n!=R->rmap->n) SETERRQ(comm,PETSC_ERR_ARG_SIZ,"Matrices #1 and #2 don't have the same row layout, %" PetscInt_FMT " != %" PetscInt_FMT "",K->rmap->n,R->rmap->n);

  FllopTraceBegin;
  PetscCall(PetscLogEventBegin(Mat_Regularize,K,R,0,0));

  /* this should work at least for MATMPIAIJ and MATBLOCKDIAG */
  PetscCall(MatGetDiagonalBlock(K,&K_loc));
  PetscCall(MatGetDiagonalBlock(R,&R_loc));

  PetscCall(MatRegularize_GetPivots_Private(R_loc, &pivots));
  PetscCall(MatRegularize_GetRegularization_Private(K_loc, R_loc, pivots, &Q_loc));

  /* Kreg_loc = K_loc + rho*Q_loc */
  //TODO parametrize
  PetscCall(MatGetMaxEigenvalue(K_loc, NULL, &rho, 1, 20));
  PetscCall(MatScale(Q_loc, rho));
  if (type == MAT_REG_EXPLICIT)
  {
    Mat Kreg_loc;

    PetscCall(MatDuplicate(K, MAT_COPY_VALUES, &Kreg));
    PetscCall(MatGetDiagonalBlock(Kreg,&Kreg_loc));

    //TODO avoid adding new nonzeros - do preallocation of Kreg
    PetscCall(MatSetOption(Kreg_loc, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
    PetscCall(MatAXPY(Kreg_loc, rho, Q_loc, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAssemblyBegin(Kreg_loc, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(  Kreg_loc, MAT_FINAL_ASSEMBLY));
  }
  else  /* type == MAT_REG_IMPLICIT */
  {
    Mat Q, Kreg_arr[2];

    PetscCall(MatCreateBlockDiag(comm,Q_loc,&Q));
    Kreg_arr[0]=Q; Kreg_arr[1]=K;
    PetscCall(MatCreateComposite(comm,2,Kreg_arr,&Kreg));
    PetscCall(MatDestroy(&Q));
  }
  PetscCall(MatInheritSymmetry(K,Kreg));
  PetscCall(MatDestroy(&Q_loc));
  PetscCall(ISDestroy(&pivots));

  /* mark the matrix as regularized */
  PetscCall(PetscObjectComposedDataSetInt((PetscObject)Kreg,regularized_id,PETSC_TRUE));
  *newKreg = Kreg;
  PetscCall(PetscLogEventEnd(Mat_Regularize,K,R,0,0));
  PetscFunctionReturnI(PETSC_SUCCESS);
}
