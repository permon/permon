ALL: ex1

CFLAGS      = -I.
FFLAGS      =
CPPFLAGS    =
FPPFLAGS    =
TUTORIALS   = ex1.c ex2.c jbearing2.c
EXAMPLESC   = ex1.c ex2.c jbearing2.c
EXAMPLECH   =
EXAMPLESF   =
EXAMPLESFH  =
LOCDIR      = src/examples
MANSEC      =
DOCS        =
DIRS        =
CLEANFILES  = ex1 ex2 jbearing2

include ${PERMON_DIR}/lib/permon/conf/permon_base

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} ex2.o

jbearing2: jbearing2.o chkopts
	-${CLINKER} -o jbearing2 jbearing2.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} jbearing2.o

