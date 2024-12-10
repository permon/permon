#pragma once

#include <permon/private/qpcimpl.h>

typedef struct {
  Vec lb;
  Vec ub;
  Vec llb;
  Vec lub;
} QPC_Box;
