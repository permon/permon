ALL: ex1 ex71

CFLAGS      = -I.
FFLAGS      =
CPPFLAGS    =
FPPFLAGS    =
TUTORIALS   = ex1.c ex71.c
EXAMPLESC   = ex1.c ex71.c
EXAMPLECH   =
EXAMPLESF   =
EXAMPLESFH  =
LOCDIR      = src/examples/feti
MANSEC      =
DOCS        =
DIRS        =
CLEANFILES  = ex1 ex71

include ${PERMON_DIR}/lib/permon/conf/permon_base

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} ex1.o

ex71: ex71.o chkopts
	-${CLINKER} -o ex71 ex71.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} ex71.o

