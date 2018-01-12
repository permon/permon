ALL: ex1

CFLAGS      = -I.
FFLAGS      =
CPPFLAGS    =
FPPFLAGS    =
TUTORIALS   = ex1.c
EXAMPLESC   = ex1.c
EXAMPLECH   =
EXAMPLESF   =
EXAMPLESFH  =
LOCDIR      = src/examples/feti
MANSEC      =
DOCS        =
DIRS        =
CLEANFILES  = ex1

include ${PERMON_DIR}/lib/permon/conf/permon_base

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} ex1.o

