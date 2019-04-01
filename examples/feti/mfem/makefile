include ${PERMON_DIR}/lib/permon/conf/permon_variables
ifeq ($(with_mfem),1) 

ALL: ex1p

CFLAGS      = -I.
FFLAGS      =
CPPFLAGS    =
FPPFLAGS    =
TUTORIALS   = ex1p.cpp
EXAMPLESC   = ex1p.cpp
EXAMPLECH   =
EXAMPLESF   =
EXAMPLESFH  =
LOCDIR      = src/examples/feti/mfem
MANSEC      =
DOCS        =
DIRS        =
CLEANFILES  = ex1p

include ${PERMON_DIR}/lib/permon/conf/permon_rules

ex1p: ex1p.o chkopts
	-${CXXLINKER} -o ex1p ex1p.o ${PERMON_LIB}
	${RM} ex1p.o

else
$(warning "To use PERMON with MFEM, calling 'make with_mfem=1' in PERMON_DIR is required.")
.SUFFIXES:
endif

