static char help[] = "3D linear elasticity using QP directly with optional contact\n\
Options:\n\
  -contact          : Contact on the right face (default: True)\n\
  -dir_in_hess      : if true enforce dirichlet in Hessian, use Eq. constraint otherwise; default False.\n\
  -young <val>      : Young's modulus (default: 2.0e4 Pa)\n\
  -poisson <val>    : Poisson's ratio (default: 0.33)\n\
  -force <val>      : Total force on top surface (default: -465 N)\n\
  -lx, -ly, -lz     : Domain dimensions (default: 1x1x1)\n\n";

#include <permonqps.h>
#include <permonqpfeti.h>
#include <petscdm.h>
#include <petscdmda.h>

typedef struct {
  PetscReal    young;
  PetscReal    poisson;
  PetscReal    force;
  PetscReal    lx, ly, lz;
  PetscInt     cells[3];
  PetscInt     dof;
  PetscBool    contact;
  PetscBool    dirInHess;
  PetscScalar *elemMat;
} AppCtx;

PetscErrorCode ComputeElementStiffness(PetscReal E, PetscReal nu, PetscReal dx, PetscReal dy, PetscReal dz, PetscScalar *Ke)
{
  PetscReal lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
  PetscReal mu     = E / (2.0 * (1.0 + nu));
  PetscReal c1     = lambda + 2 * mu;

  PetscReal J_inv_00 = 2.0 / dx;
  PetscReal J_inv_11 = 2.0 / dy;
  PetscReal J_inv_22 = 2.0 / dz;
  PetscReal detJ     = (dx * dy * dz) / 8.0;

  PetscReal pts[2] = {-0.577350269189626, 0.577350269189626};
  //PetscReal wts[2] = {1.0, 1.0};
  PetscReal node_xi[8]   = {-1, 1, 1, -1, -1, 1, 1, -1};
  PetscReal node_eta[8]  = {-1, -1, 1, 1, -1, -1, 1, 1};
  PetscReal node_zeta[8] = {-1, -1, -1, -1, 1, 1, 1, 1};

  PetscReal B[6][24];
  PetscReal D[6][6];
  PetscReal DB[6][24];

  PetscReal xi, eta, zeta;
  PetscReal dN_dx, dN_dy, dN_dz;

  PetscFunctionBeginUser;
  PetscMemzero(Ke, 24 * 24 * sizeof(PetscScalar));
  for (PetscInt k = 0; k < 2; k++) {
    for (PetscInt j = 0; j < 2; j++) {
      for (PetscInt i = 0; i < 2; i++) {
        xi   = pts[i];
        eta  = pts[j];
        zeta = pts[k];
        PetscMemzero(B, 144 * sizeof(PetscReal));
        for (PetscInt n = 0; n < 8; n++) {
          dN_dx = 0.125 * node_xi[n] * (1 + node_eta[n] * eta) * (1 + node_zeta[n] * zeta) * J_inv_00;
          dN_dy = 0.125 * node_eta[n] * (1 + node_xi[n] * xi) * (1 + node_zeta[n] * zeta) * J_inv_11;
          dN_dz = 0.125 * node_zeta[n] * (1 + node_xi[n] * xi) * (1 + node_eta[n] * eta) * J_inv_22;

          B[0][n * 3 + 0] = dN_dx;
          B[1][n * 3 + 1] = dN_dy;
          B[2][n * 3 + 2] = dN_dz;
          B[3][n * 3 + 0] = dN_dy;
          B[3][n * 3 + 1] = dN_dx;
          B[4][n * 3 + 1] = dN_dz;
          B[4][n * 3 + 2] = dN_dy;
          B[5][n * 3 + 0] = dN_dz;
          B[5][n * 3 + 2] = dN_dx;
        }

        /* Constitutive matrix */
        PetscMemzero(D, 36 * sizeof(PetscReal));
        D[0][0] = c1;
        D[1][1] = c1;
        D[2][2] = c1;
        D[0][1] = lambda;
        D[0][2] = lambda;
        D[1][0] = lambda;
        D[1][2] = lambda;
        D[2][0] = lambda;
        D[2][1] = lambda;
        D[3][3] = mu;
        D[4][4] = mu;
        D[5][5] = mu;

        PetscMemzero(DB, 144 * sizeof(PetscReal));
        for (PetscInt r = 0; r < 6; r++) {
          for (PetscInt c = 0; c < 24; c++) {
            for (PetscInt m = 0; m < 6; m++) DB[r][c] += D[r][m] * B[m][c];
          }
        }
        for (PetscInt r = 0; r < 24; r++) {
          for (PetscInt c = 0; c < 24; c++) {
            PetscReal val = 0;
            for (PetscInt m = 0; m < 6; m++) val += B[m][r] * DB[m][c];
            Ke[r * 24 + c] += val * detJ; // *wts[i]*wts[j]*wts[k], which is 1.0
          }
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt n;

  PetscFunctionBeginUser;
  options->young     = 2.0e4;
  options->poisson   = 0.33;
  options->force     = -465.0;
  options->lx        = 1.0;
  options->ly        = 1.0;
  options->lz        = 1.0;
  options->cells[0]  = 10;
  options->cells[1]  = 10;
  options->cells[2]  = 10;
  options->dof       = 3;
  options->contact   = PETSC_TRUE;
  options->dirInHess = PETSC_FALSE;

  PetscOptionsBegin(comm, NULL, "Contact linear elasticity options", NULL);
  PetscCall(PetscOptionsBool("-contact", "Contact problem", NULL, options->contact, &options->contact, NULL));
  PetscCall(PetscOptionsBool("-dir_in_hess", "Dirichlet BC enforced by Hessian", NULL, options->dirInHess, &options->dirInHess, NULL));
  PetscCall(PetscOptionsReal("-young", "Young's modulus", NULL, options->young, &options->young, NULL));
  PetscCall(PetscOptionsReal("-poisson", "Poisson's ratio", NULL, options->poisson, &options->poisson, NULL));
  PetscCall(PetscOptionsReal("-force", "Top force", NULL, options->force, &options->force, NULL));
  PetscCall(PetscOptionsReal("-lx", "Length X", NULL, options->lx, &options->lx, NULL));
  PetscCall(PetscOptionsReal("-ly", "Length Y", NULL, options->ly, &options->ly, NULL));
  PetscCall(PetscOptionsReal("-lz", "Length Z", NULL, options->lz, &options->lz, NULL));
  PetscCall(PetscOptionsIntArray("-cells", "Mesh division", NULL, options->cells, (n = 3, &n), NULL));
  PetscOptionsEnd();

  PetscReal dx = options->lx / options->cells[0];
  PetscReal dy = options->ly / options->cells[1];
  PetscReal dz = options->lz / options->cells[2];

  PetscCall(PetscMalloc1(24 * 24, &options->elemMat));
  PetscCall(ComputeElementStiffness(options->young, options->poisson, dx, dy, dz, options->elemMat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  AppCtx          user;
  QP              qp, qpd;
  Mat             A;
  DM              da;
  Vec             x, b, xcoor;
  const PetscInt *e_loc;
  PetscInt        nel, nen;
  PetscInt        M, N, P, m, n, p;
  PetscInt        xs, ys, zs, xm, ym, zm;
  PetscInt        gxs, gys, gzs, gxm, gym, gzm;

  PetscCall(PermonInitialize(&argc, &args, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  /* Create mesh, matrices and vectors */
  M = user.cells[0] + 1;
  N = user.cells[1] + 1;
  P = user.cells[2] + 1;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, M, N, P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, user.dof, 1, NULL, NULL, NULL, &da));
  PetscCall(DMSetMatType(da, MATIS));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMDASetElementType(da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, user.lx, 0.0, user.ly, 0.0, user.lz));
  PetscCall(DMDAGetInfo(da, NULL, NULL, NULL, NULL, &m, &n, &p, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscPrintf(PETSC_COMM_WORLD, "#subdomains      %12d  (Nx Ny Nz)=(%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT ")\n", m * n * p, m, n, p);
  PetscPrintf(PETSC_COMM_WORLD, "#elements        %12d  (nx ny nz)=(%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT ")\n", M * N * P - 3, M - 1, N - 1, P - 1);
  PetscPrintf(PETSC_COMM_WORLD, "#DOFs undecomp.  %12d\n", M * N * P * user.dof);
  PetscPrintf(PETSC_COMM_WORLD, "#DOFs decomp.    %12d\n", (M + m - 1) * (N + n - 1) * (P + p - 1) * user.dof);
  PetscPrintf(PETSC_COMM_WORLD, "Contact          %12d\n", user.contact);
  PetscPrintf(PETSC_COMM_WORLD, "Dir. BC in Hess. %12d\n", user.dirInHess);

  PetscCall(DMGetCoordinates(da, &xcoor));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(DMCreateGlobalVector(da, &b));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecSet(b, 0.0));
  PetscCall(VecSet(x, 0.0)); // this is the value of Dirichlet BC on the bottom face

  /* Assemble stiffness matrix */
  {
    PetscInt *idx_dof;
    PetscCall(PetscMalloc1(24, &idx_dof));
    PetscCall(DMDAGetElements(da, &nel, &nen, &e_loc));
    for (PetscInt i = 0; i < nel; i++) {
      const PetscInt *idx = &e_loc[i * nen];
      for (PetscInt j = 0; j < 8; j++) {
        idx_dof[3 * j + 0] = idx[j] * 3 + 0;
        idx_dof[3 * j + 1] = idx[j] * 3 + 1;
        idx_dof[3 * j + 2] = idx[j] * 3 + 2;
      }
      PetscCall(MatSetValuesLocal(A, 24, idx_dof, 24, idx_dof, user.elemMat, ADD_VALUES));
    }
    PetscCall(DMDARestoreElements(da, &nel, &nen, &e_loc));
    PetscCall(PetscFree(idx_dof));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  /* Set force from top */
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gxm, &gym, &gzm));
  {
    /* force per node = 1/4 * force / number of faces */
    PetscReal val = 0.25 * user.force / (PetscReal)((M - 1) * (N - 1));

    /* ensure we own an element on the top layer (P-1) */
    PetscInt k = P - 1;
    if (k >= zs && k < zs + zm) {
      /* loop over elements on the top surface
           loop through the bottom-left corner of elements we own */
      for (PetscInt j = ys; j < ys + ym; j++) {
        for (PetscInt i = xs; i < xs + xm; i++) {
          /* check if (i,j) defines valid domain bounds */
          if (i < M - 1 && j < N - 1) {
            PetscInt    indices[4];
            PetscScalar values[4] = {val, val, val, val};
            /* compute local indices for the 4 nodes of this face
                           formula: (i_loc + j_loc * gxm + k_loc * gxm * gym) * dof + component
                           we are loading the z-component, i.e., component = 2 */
            indices[0] = ((i - gxs) + (j - gys) * gxm + (k - gzs) * gxm * gym) * user.dof + 2;
            indices[1] = ((i + 1 - gxs) + (j - gys) * gxm + (k - gzs) * gxm * gym) * user.dof + 2;
            indices[2] = ((i - gxs) + (j + 1 - gys) * gxm + (k - gzs) * gxm * gym) * user.dof + 2;
            indices[3] = ((i + 1 - gxs) + (j + 1 - gys) * gxm + (k - gzs) * gxm * gym) * user.dof + 2;
            PetscCall(VecSetValuesLocal(b, 4, indices, values, ADD_VALUES));
          }
        }
      }
    }
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }

  /* Create QP */
  PetscCall(QPCreate(PETSC_COMM_WORLD, &qp));
  PetscCall(QPSetOperator(qp, A));
  PetscCall(QPSetRhs(qp, b));
  PetscCall(QPSetInitialVector(qp, x));
  /* Transform to decomposed numbering (only A, b and x are currently supported) */
  PetscCall(QPTMatISToBlockDiag(qp)); /* this also computes and sets l2g and i2g mappings in the child QP */
  PetscCall(QPGetChild(qp, &qpd));

  /* Dirichlet BC on the bottom face */
  {
    PetscInt              *idx_list = NULL;
    PetscInt               c = 0, cnt = 0;
    PetscInt              *local;
    PetscInt               nlocal;
    IS                     dirichletIS;
    ISLocalToGlobalMapping ltog;

    if (zs == 0) {
      cnt = xm * ym * user.dof;
      PetscCall(PetscMalloc1(cnt, &idx_list));
      for (PetscInt j = ys; j < ys + ym; j++) {
        for (PetscInt i = xs; i < xs + xm; i++) {
          for (int d = 0; d < user.dof; d++) { idx_list[c++] = ((i - gxs) + (j - gys) * gxm + (0 - gzs) * gxm * gym) * user.dof + d; }
        }
      }
    }
    PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
    if (cnt > 0) PetscCall(ISLocalToGlobalMappingApply(ltog, cnt, idx_list, idx_list));
    PetscCall(MatISIndicesGlobalToLocal(A, cnt, idx_list, &nlocal, &local)); /* this communicates undecomposed global DOFs into local FETI numbering */
    //PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, cnt, idx_list, PETSC_OWN_POINTER, &dirichletIS));
    //MatZeroRowsColumnsIS(A, dirichletIS, 2481.69 ,x , b); /* 8 rank 2x2x2 cells test case */
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nlocal, local, PETSC_OWN_POINTER, &dirichletIS));
    PetscCall(QPFetiSetDirichlet(qpd, dirichletIS, FETI_LOCAL, PetscNot(user.dirInHess)));
    PetscCall(ISDestroy(&dirichletIS));
  }

  /* Create contact constraint, i.e., Bx <= c,
     which constraints displacement in X-component of right-hand face <= c */
  if (user.contact) {
    Mat             Adecomposed, B;
    Vec             c;
    IS              islocal, isglobald;
    PetscInt        num_found = 0;
    PetscInt        num_local = 0;
    PetscInt        cnt       = 0;
    PetscInt        size, row_offset;
    PetscInt       *allnodes;
    PetscInt       *ghost_to_local;
    PetscInt       *idxGhost;
    PetscInt       *idxl;
    const PetscInt *idxdg;

    /* Identify nodes on the right-hand face in the local numbering
         This could be done in the same way as above for the Dirichlet BC */
    PetscCall(DMDAGetElements(da, &nel, &nen, &e_loc));
    PetscCall(PetscMalloc1(nel * nen, &allnodes));
    PetscCall(PetscMalloc1(gxm * gym * gzm, &ghost_to_local));

    /* all nodes in local numbering */
    for (PetscInt i = 0; i < nel * nen; i++) allnodes[i] = e_loc[i];
    /* make a map of ghost nodes to local nodes */
    PetscCall(PetscSortInt(nel * nen, allnodes));
    if (nel * nen > 0) {
      ghost_to_local[allnodes[0]] = num_local++;
      for (PetscInt i = 1; i < nel * nen; i++) {
        if (allnodes[i] > allnodes[i - 1]) { ghost_to_local[allnodes[i]] = num_local++; }
      }
    }
    PetscCall(PetscFree(allnodes));

    /* get all nodes in ghost numbering on the right-hand boundary */
    PetscCall(PetscMalloc1(num_local, &idxGhost));
    for (PetscInt e = 0; e < nel; e++) {
      const PetscInt *nodes = &e_loc[e * nen];
      for (PetscInt n = 0; n < nen; n++) {
        if (nodes[n] % gxm + gxs == M - 1) { idxGhost[num_found++] = nodes[n]; }
      }
    }

    /* map ghost right-hand boundary nodes to local numbering */
    PetscCall(PetscMalloc1(num_local, &idxl));
    PetscCall(PetscSortInt(num_found, idxGhost));
    if (num_found > 0) {
      idxl[cnt++] = ghost_to_local[idxGhost[0]] * user.dof + 0;
      for (PetscInt i = 1; i < num_found; i++) {
        if (idxGhost[i] > idxGhost[i - 1]) {
          PetscInt c_idx = ghost_to_local[idxGhost[i]];
          idxl[cnt++]    = c_idx * user.dof + 0;
        }
      }
    }
    PetscCall(DMDARestoreElements(da, &nel, &nen, &e_loc));
    PetscCall(PetscFree(ghost_to_local));
    PetscCall(PetscFree(idxGhost));

    /* convert local numbering to global undecomposed */
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, cnt, idxl, PETSC_USE_POINTER, &islocal));
    PetscCall(QPFetiConvertNumberingIS(qpd, FETI_LOCAL, islocal, FETI_GLOBAL_DECOMPOSED, &isglobald));
    PetscCall(ISDestroy(&islocal));
    if (cnt > 0) PetscCall(PetscFree(idxl));

    PetscCall(QPGetOperator(qpd, &Adecomposed));
    PetscCall(MatGetLocalSize(Adecomposed, &size, NULL));
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, cnt, size, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &B)); /* o_nz can probably be zero */

    /* fill the constraint matrix */
    PetscCallMPI(MPI_Scan(&cnt, &row_offset, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    row_offset -= cnt;
    PetscCall(ISGetIndices(isglobald, &idxdg));
    for (PetscInt i = 0; i < cnt; i++) { PetscCall(MatSetValue(B, row_offset + i, idxdg[i], 1.0, INSERT_VALUES)); }
    PetscCall(ISRestoreIndices(isglobald, &idxdg));
    PetscCall(ISDestroy(&isglobald));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

    /* set the constraint vector */
    PetscCall(MatCreateVecs(B, NULL, &c));
    PetscCall(VecSet(c, 0.001));

    /* Set the QP constraint */
    PetscCall(QPSetIneq(qpd, B, c));
    PetscCall(MatDestroy(&B));
    PetscCall(VecDestroy(&c));
  }

  /* Setup FETI */
  PetscCall(QPFetiSetUp(qpd));
  PetscCall(PetscOptionsInsertString(NULL, "-feti -qpt_dualize_B_nest_extension"));
  PetscCall(QPTFromOptions(qpd));

  /* Solve */
  QPS qps;
  PetscCall(QPSCreate(PETSC_COMM_WORLD, &qps));
  PetscCall(QPSSetQP(qps, qp));
  PetscCall(QPSSetFromOptions(qps));
  PetscCall(QPSSetUp(qps));
  PetscCall(QPSSolve(qps));

  /* View solution
     For ParaView: -dm_view vtk:solution.vts -sol_view vtk:solution.vts::append */
  PetscCall(DMViewFromOptions(da, NULL, "-dm_view"));
  PetscCall(VecViewFromOptions(x, NULL, "-sol_view"));

  PetscCall(QPSDestroy(&qps));
  PetscCall(QPDestroy(&qp)); // destroys the whole QP chain
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFree(user.elemMat));
  PetscCall(PermonFinalize());
  return 0;
}
