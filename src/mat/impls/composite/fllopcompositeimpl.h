#if !defined(__COMPOSITEMIMPL_H)
#define __COMPOSITEMIMPL_H
#include <permon/private/permonmatimpl.h>

typedef struct _Mat_CompositeLink *Mat_CompositeLink;
struct _Mat_CompositeLink {
  Mat               mat;
  Vec               work;
  Mat_CompositeLink next,prev;
};
  
typedef struct {
  MatCompositeType  type;
  Mat_CompositeLink head,tail;
  Vec               work;
  PetscScalar       scale;        /* scale factor supplied with MatScale() */
  Vec               left,right;   /* left and right diagonal scaling provided with MatDiagonalScale() */
  Vec               leftwork,rightwork;
} Mat_Composite;

#endif
