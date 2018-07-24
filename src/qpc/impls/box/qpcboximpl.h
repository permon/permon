#if !defined(__BOXIMPL_H)
#define __BOXIMPL_H
#include <permon/private/qpcimpl.h>

typedef struct {
  Vec lb;
  Vec ub;
  Vec llb;
  Vec lub;
} QPC_Box;

#endif
