ALL: jbearing2

CFLAGS		 = -I.
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
TUTORIALS        = jbearing2.c
EXAMPLESC        = jbearing2.c
EXAMPLECH        =
EXAMPLESF        =
EXAMPLESFH       =
LOCDIR		 = src/tao/bound/examples/tutorials/
MANSEC		 =
DOCS		 =
DIRS		 =
CLEANFILES       =  jbearing2

include ${PERMON_DIR}/lib/permon/conf/permon_base


jbearing2: jbearing2.o chkopts
	-${CLINKER} -o jbearing2 jbearing2.o ${PETSC_TAO_LIB} ${PERMON_LIB}
	${RM} jbearing2.o

