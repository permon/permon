
#if !defined(__NORMM_H)
#define	__NORMM_H

typedef struct {
  Mat         A;
  Vec         w,left,right,leftwork,rightwork;
  PetscScalar scale;
} Mat_Normal;

#endif
