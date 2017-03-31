
#include <private/fllopmatimpl.h>

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
  Mat R_work;

  PetscFunctionBeginI;
  TRY( MatConvert(R,MATDENSEPERMON,MAT_INITIAL_MATRIX,&R_work) );
  TRY( MatGetSize(R_work, &p, &npivots) );
  maxdp = PetscMax(npivots, p);
  TRY( PetscMalloc(maxdp * sizeof (PetscInt),    &all) );
  TRY( PetscMalloc(p     * sizeof (PetscInt),    &perm) );
  TRY( PetscMalloc(p     * sizeof (PetscScalar), &v1) );
  TRY( PetscMalloc(p     * sizeof (PetscScalar), &v2) );

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
      TRY( MatGetValues(R_work, p, all, 1, &j, v1) );

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
    TRY( MatGetValues(R_work, 1, &ipivot, J + 1, all, v1) );
    TRY( MatGetValues(R_work, 1, &II,      J + 1, all, v2) );
    TRY( MatSetValues(R_work, 1, &ipivot, J + 1, all, v2, INSERT_VALUES) );
    TRY( MatSetValues(R_work, 1, &II,      J + 1, all, v1, INSERT_VALUES) );
    TRY( MatAssemblyBegin(R_work, MAT_FINAL_ASSEMBLY) );
    TRY( MatAssemblyEnd(  R_work, MAT_FINAL_ASSEMBLY) );
    t = perm[ipivot];
    perm[ipivot] = perm[II];
    perm[II] = t;

    /* swap columns jpivot and J:
     * v1 = R(:,jpivot);  R(:,jpivot) = R(:,J); R(:,J) = v1;
     */
    TRY( MatGetValues(R_work, II + 1, all, 1, &jpivot, v1) );
    TRY( MatGetValues(R_work, II + 1, all, 1, &J,      v2) );
    TRY( MatSetValues(R_work, II + 1, all, 1, &J,      v1, INSERT_VALUES) );
    TRY( MatSetValues(R_work, II + 1, all, 1, &jpivot, v2, INSERT_VALUES) );
    TRY( MatAssemblyBegin(R_work, MAT_FINAL_ASSEMBLY) );
    TRY( MatAssemblyEnd(  R_work, MAT_FINAL_ASSEMBLY) );

    /* columnwise elimination of the row II */
    for (j = 0; j <= J - 1; j++) {
      /* v2 = R(:,j) */
      TRY( MatGetValues(R_work, II + 1, all, 1, &j, v2) );

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
      TRY( MatSetValues(R_work, II + 1, all, 1, &j, v2, INSERT_VALUES) );
      TRY( MatAssemblyBegin(R_work, MAT_FINAL_ASSEMBLY) );
      TRY( MatAssemblyEnd(  R_work, MAT_FINAL_ASSEMBLY) );
    }
  }

  /* idx_arr points to last d entries of perm */
  idx_arr = &(perm[p - npivots]);

  TRY( ISCreateGeneral(PETSC_COMM_SELF, npivots, idx_arr, PETSC_COPY_VALUES, pivots) );
  TRY( ISSort(*pivots) );

  TRY( MatDestroy(&R_work) );
  TRY( PetscFree(all) );
  TRY( PetscFree(perm) );
  TRY( PetscFree(v1) );
  TRY( PetscFree(v2) );
  PetscFunctionReturnI(0);
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
  TRY( MatGetSize(R_loc, &prim_loc, &defect_loc) );
  TRY( ISGetSize(pivots, &npivots) );

  if (npivots) {
    Mat RI, RIt, RItRI, invRItRI, RI_invRItRI, Q_loc_condensed_filtered;
    IS R_all_cols;

    /* R_all_cols = 0:1:defect_loc-1 */
    TRY( ISCreateStride(PETSC_COMM_SELF, defect_loc, 0, 1, &R_all_cols) );
    
    TRY( MatGetSubMatrix(R_loc, pivots, R_all_cols, MAT_INITIAL_MATRIX, &RI) );
    TRY( MatConvert(RI,MATDENSE,MAT_INPLACE_MATRIX,&RI) );
    TRY( ISDestroy(&R_all_cols) );

    TRY( MatTranspose(RI, MAT_INITIAL_MATRIX, &RIt) );
    TRY( MatMatMult(RIt, RI, MAT_INITIAL_MATRIX, 1.0, &RItRI) );
  
    TRY( MatCreateInv(RItRI, MAT_INV_MONOLITHIC, &invRItRI) );
    TRY( MatInvExplicitly(invRItRI, PETSC_FALSE, MAT_REUSE_MATRIX, &RItRI) );
    TRY( MatDestroy(&invRItRI) );
    invRItRI = RItRI;

    TRY( MatMatMult(RI, invRItRI, MAT_INITIAL_MATRIX, 1.0, &RI_invRItRI) );
    TRY( MatMatMult(RI_invRItRI, RIt, MAT_INITIAL_MATRIX, 1.0, &Q_loc_condensed) );

    TRY( MatFilterZeros(Q_loc_condensed, PETSC_MACHINE_EPSILON*10, &Q_loc_condensed_filtered) );
    TRY( MatDestroy(&Q_loc_condensed) );
    Q_loc_condensed = Q_loc_condensed_filtered;

    TRY( MatDestroy(&RI) );
    TRY( MatDestroy(&RIt) );
    TRY( MatDestroy(&invRItRI) );
    TRY( MatDestroy(&RI_invRItRI) );
  }

  /* prepare preallocation array for Q_loc */
  TRY( PetscMalloc(prim_loc*sizeof(PetscInt), &Q_allocation) );
  TRY( PetscMemzero(Q_allocation,prim_loc*sizeof(PetscInt)) );
  TRY( ISGetIndices(pivots, &pivarr) );
  for (i=0; i<npivots; i++) {
    TRY( MatGetRow(    Q_loc_condensed,i,&ncols,NULL,NULL) );
    Q_allocation[pivarr[i]] = ncols;
    TRY( MatRestoreRow(Q_loc_condensed,i,&ncols,NULL,NULL) );
  }
  TRY( ISRestoreIndices(pivots, &pivarr) );

  /* allocate Q_loc */
  TRY( MatCreate(PETSC_COMM_SELF, &Q_loc) );
  TRY( MatSetSizes(Q_loc, prim_loc, prim_loc, prim_loc, prim_loc) );
  TRY( MatGetType(K_loc,&K_type) );  
  TRY( MatSetType(Q_loc, K_type) );
  TRY( MatSeqAIJSetPreallocation(Q_loc, -1, Q_allocation) );
  TRY( MatSeqSBAIJSetPreallocation(Q_loc, 1, -1, Q_allocation) );
  TRY( MatSetUp(Q_loc) );
  TRY( PetscFree(Q_allocation) );

  /* insert values of Q_loc_condensed to appropriate positions in Q_loc */
  for (i=0; i<npivots; i++) {
    TRY( MatGetRow(    Q_loc_condensed,i,&ncols,&cols,&vals) );
    for (j=0; j<ncols; j++) {
      TRY( MatSetValue(Q_loc, pivarr[i], pivarr[cols[j]], vals[j], INSERT_VALUES) );
    }
    TRY( MatRestoreRow(Q_loc_condensed,i,&ncols,&cols,&vals) );
  }
  TRY( MatAssemblyBegin(Q_loc, MAT_FINAL_ASSEMBLY) );
  TRY( MatAssemblyEnd(  Q_loc, MAT_FINAL_ASSEMBLY) );
  TRY( MatInheritSymmetry(K_loc,Q_loc) ); 
  TRY( MatDestroy(&Q_loc_condensed) );
  *newQ = Q_loc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRegularize"
PetscErrorCode MatRegularize(Mat K, Mat R, MatRegularizationType type, Mat *newKreg) {
  static PetscBool  registered = PETSC_FALSE;
  static PetscInt   regularized_id;
  MPI_Comm          comm;
  IS                pivots;
  Mat               Q_loc, Kreg;
  PetscScalar       rho;
  Mat               K_loc, R_loc;
  PetscBool         regularized = PETSC_FALSE;

  FllopTracedFunctionBegin;
  PetscValidHeaderSpecific(K,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(K,type,3);
  PetscValidPointer(newKreg,4);

  if (type == MAT_REG_NONE) {
    TRY( PetscInfo(K,"MatRegularizationType set to MAT_REG_NONE, returning input matrix\n") );
    *newKreg = K;
    TRY( PetscObjectReference((PetscObject)K) );
    PetscFunctionReturn(0);
  }

  if (!registered) {
    TRY( PetscLogEventRegister(__FUNCT__, MAT_CLASSID, &Mat_Regularize) );
    registered = PETSC_TRUE;
    TRY( PetscObjectComposedDataRegister(&regularized_id) );
  }

  TRY( PetscObjectComposedDataGetInt((PetscObject)K,regularized_id,regularized,regularized) );
  if (regularized) {
    TRY( PetscInfo(K,"matrix marked as regularized, returning input matrix\n") );
    *newKreg = K;
    TRY( PetscObjectReference((PetscObject)K) );
    PetscFunctionReturn(0);
  }
  
  PetscValidHeaderSpecific(R,MAT_CLASSID,2);
  PetscCheckSameComm(K,1,R,2);
  TRY( PetscObjectGetComm((PetscObject)K,&comm) );
  if (K->rmap->n!=K->cmap->n) FLLOP_SETERRQ(comm,PETSC_ERR_ARG_SIZ,"Matrix #1 must be locally square");
  if (K->rmap->n!=R->rmap->n) FLLOP_SETERRQ2(comm,PETSC_ERR_ARG_SIZ,"Matrices #1 and #2 don't have the same row layout, %D != %D",K->rmap->n,R->rmap->n);

  FllopTraceBegin;
  TRY( PetscLogEventBegin(Mat_Regularize,K,R,0,0) );
  
  /* this should work at least for MATMPIAIJ and MATBLOCKDIAG */
  TRY( MatGetDiagonalBlock(K,&K_loc) );
  TRY( MatGetDiagonalBlock(R,&R_loc) );

  TRY( MatRegularize_GetPivots_Private(R_loc, &pivots) );
  TRY( MatRegularize_GetRegularization_Private(K_loc, R_loc, pivots, &Q_loc) );

  /* Kreg_loc = K_loc + rho*Q_loc */
  //TODO parametrize
  TRY( MatGetMaxEigenvalue(K_loc, NULL, &rho, 1, 20) );
  TRY( MatScale(Q_loc, rho) );
  if (type == MAT_REG_EXPLICIT)
  {
    Mat Kreg_loc;
    
    TRY( MatDuplicate(K, MAT_COPY_VALUES, &Kreg) );
    TRY( MatGetDiagonalBlock(Kreg,&Kreg_loc) );
    
    //TODO avoid adding new nonzeros - do preallocation of Kreg
    TRY( MatSetOption(Kreg_loc, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE) );
    TRY( MatAXPY(Kreg_loc, rho, Q_loc, DIFFERENT_NONZERO_PATTERN) );
    TRY( MatAssemblyBegin(Kreg_loc, MAT_FINAL_ASSEMBLY) );
    TRY( MatAssemblyEnd(  Kreg_loc, MAT_FINAL_ASSEMBLY) );
  }
  else  /* type == MAT_REG_IMPLICIT */
  {
    Mat Q, Kreg_arr[2];
    
    TRY( MatCreateBlockDiag(comm,Q_loc,&Q) );
    Kreg_arr[0]=Q; Kreg_arr[1]=K;
    TRY( MatCreateSum(comm,2,Kreg_arr,&Kreg) );
    TRY( MatDestroy(&Q) );
  }
  TRY( MatInheritSymmetry(K,Kreg) );
  TRY( MatDestroy(&Q_loc) );
  TRY( ISDestroy(&pivots) );
  
  /* mark the matrix as regularized */
  TRY( PetscObjectComposedDataSetInt((PetscObject)Kreg,regularized_id,PETSC_TRUE) );
  *newKreg = Kreg;
  TRY( PetscLogEventEnd(Mat_Regularize,K,R,0,0) );
  PetscFunctionReturnI(0);
}
