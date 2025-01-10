#include <permon/private/petscimpl.h>

#if defined(MatCreateSeqSBAIJWithArrays)
  #undef MatCreateSeqSBAIJWithArrays
#endif

#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqSBAIJWithArrays_permonfix"
PetscErrorCode MatCreateSeqSBAIJWithArrays_permonfix(MPI_Comm comm, PetscInt bs, PetscInt m, PetscInt n, PetscInt *i, PetscInt *j, PetscScalar *a, Mat *mat)
{
  Mat_SeqSBAIJ *sbaij;

  PetscFunctionBegin;
  PetscCall(MatCreateSeqSBAIJWithArrays(comm, bs, m, n, i, j, a, mat));
  sbaij                 = (Mat_SeqSBAIJ *)(*mat)->data;
  sbaij->free_imax_ilen = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define MatCreateSeqSBAIJWithArrays(a, b, c, d, e, f, g, h) MatCreateSeqSBAIJWithArrays_permonfix(a, b, c, d, e, f, g, h)
