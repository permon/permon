#
# This is the makefile for installing PERMON <http://permon.it4i.eu/>.
# Adopted from SLEPc <http://slepc.upv.es/> makefile.

ALL: all
LOCDIR = .
DIRS   = src include docs

# Include the rest of makefiles
-include ${PERMON_DIR}/lib/permon/conf/permon_variables
-include ${PERMON_DIR}/lib/permon/conf/permon_rules

#
# Basic targets to build PERMON library
all: chk_all
	@mkdir -p ./${PETSC_ARCH}/lib/permon/conf
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} all-gnumake-local 2>&1 | tee ./${PETSC_ARCH}/lib/permon/conf/make.log; \
	 else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} all-legacy-local 2>&1 | tee ./${PETSC_ARCH}/lib/permon/conf/make.log | ${GREP} -v "has no symbols"; \
	 fi
	@egrep -i "( error | error: |no such file or directory)" ${PETSC_ARCH}/lib/permon/conf/make.log | tee ./${PETSC_ARCH}/lib/permon/conf/error.log > /dev/null
	@if test -s ./${PETSC_ARCH}/lib/permon/conf/error.log; then \
           printf ${PETSC_TEXT_HILIGHT}"*******************************ERROR************************************\n" 2>&1 | tee -a ./${PETSC_ARCH}/lib/permon/conf/make.log; \
           echo "  Error during compile, check ./${PETSC_ARCH}/lib/permon/conf/make.log" 2>&1 | tee -a ./${PETSC_ARCH}/lib/permon/conf/make.log; \
           echo "  Send all contents of ./${PETSC_ARCH}/lib/permon/conf to vaclav.hapla@vsb.cz" 2>&1 | tee -a ./${PETSC_ARCH}/lib/permon/conf/make.log;\
           printf "************************************************************************"${PETSC_TEXT_NORMAL}"\n" 2>&1 | tee -a ./${PETSC_ARCH}/lib/permon/conf/make.log; \
					 exit 1; \
	 else \
		echo "Completed building libraries in ${PERMON_DIR}/${PETSC_ARCH}"; \
			echo "========================================="; \
	 fi

all-gnumake: chk_all
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
          ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} PERMON_BUILD_USING_CMAKE="" all;\
        else printf ${PETSC_TEXT_HILIGHT}"Build not configured for GNUMAKE. Quiting"${PETSC_TEXT_NORMAL}"\n"; exit 1; fi

all-legacy: chk_all
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} PERMON_BUILD_USING_CMAKE="" MAKE_IS_GNUMAKE="" all

all-gnumake-local: permon_info permon_gnumake

all-legacy-local:  permon_info deletelibs build permon_shared permon_hginfo

chk_all: chk_permon_dir chk_permon_petsc_dir chklib_dir chk_makej

#
# Check if PETSC_DIR variable specified is valid
#
#
chk_permon_petsc_dir: chk_petscdir
	@if [ -z ${PETSC_DIR} ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "PETSC_DIR not specified!"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
  false; fi
	@if [ ! -f ${PETSC_DIR}/include/petscversion.h ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}!"; \
	  echo "You need to use / to separate directories, not \\!"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
    false; fi
	@if [ ! -f ${PETSC_DIR}/${PETSC_ARCH}/lib/lib*petsc*.a ] && [ ! -f ${PETSC_DIR}/${PETSC_ARCH}/lib/lib*petsc*.so ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Given PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH} does not contain compiled PETSc library (libpetsc.so or libpetsc.a)!"; \
	  echo "You may need to run configure and make in ${PETSC_DIR}"; \
	  echo "Use / to separate directories, not \\!"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi

#
# Check if PERMON_DIR variable specified is valid
#
#TODO replace fllopsys.h with permonsys.h
chk_permon_dir: true_PERMON_DIR := $(realpath $(PERMON_DIR))
chk_permon_dir: mypwd := $(realpath .)
chk_permon_dir:
	@if [ -z ${PERMON_DIR} ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "PERMON_DIR not specified!"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
    false; fi
	@if [ ! -f ${PERMON_DIR}/include/fllopsys.h ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Incorrect PERMON_DIR specified: ${PERMON_DIR}!"; \
	  echo "Note: You need to use / to separate directories, not \\."; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi
	@if [ "${true_PERMON_DIR}" != "${mypwd}" ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Your PERMON_DIR does not match the directory you are in"; \
	  echo "PERMON_DIR: "${true_PERMON_DIR}; \
	  echo "Current directory: "${mypwd}; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi

#
# Prints information about the system and version of PERMON being compiled
#
permon_info: chk_makej
	-@echo "=========================================="
	-@echo Starting on `hostname` at `date`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using PERMON directory: ${PERMON_DIR}"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "-----------------------------------------"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc configure options: ${CONFIGURE_OPTIONS}"
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ include paths: ${PERMON_CC_INCLUDES}"
	-@echo "Using C/C++ compiler: ${PCC} ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS}"
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${PCC_LINKER}"
	-@echo "Using C/C++ flags: ${PCC_LINKER_FLAGS}"
	-@echo "-----------------------------------------"
	-@echo "Using libraries: ${PERMON_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpiexec: ${MPIEXEC}"
	-@echo "=========================================="

#
# Builds the PERMON library
#
build: chk_makej
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} ACTION=libfast tree
	-@${RANLIB} ${PERMON_LIB_DIR}/*.${AR_LIB_SUFFIX}  > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
	-@echo "Completed building libraries"
	-@echo "========================================="

#                                                                                                                       
# builds the PERMON shared library                                                                                       
#                                                                                                                       
permon_shared:                                                                                                           
	-@echo "BEGINNING TO LINK THE SHARED LIBRARY"                                                                         
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} OTHERSHAREDLIBS="${PETSC_KSP_LIB}" shared_nomesg

# Ranlib on the library
ranlib:
	${RANLIB} ${PERMON_LIB_DIR}/*.${AR_LIB_SUFFIX}

# Deletes PERMON library
deletelibs: chk_makej
	-@${RM} -r ${PERMON_LIB_DIR}/libpermon*.*

deletebins:                          
	-@${RM} ${PERMON_DIR}/${PETSC_ARCH}/bin/*      

# Cleans up build
allclean-legacy: deletelibs
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} ACTION=clean tree
allclean-gnumake:
	-@${OMAKE} -f gmakefile clean

allclean: deletebins
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} allclean-gnumake; \
	else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} allclean-legacy; \
	fi

clean:: allclean


# ------------------------------------------------------------------
#
# All remaining rules are intended for PERMON developers.
# PERMON users should not generally need to use these commands.
#

# Builds all the documentation
alldoc: alldoc1 alldoc2

# Build everything that goes into 'doc' dir except html sources
alldoc1: chk_loc deletemanualpages
	-${OMAKE} ACTION=manualpages_buildcite tree_basic LOC=${LOC}
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > ${LOC}/docs/manualpages/htmlmap
	-@cat ${PETSC_DIR}/src/docs/mpi.www.index >> ${LOC}/docs/manualpages/htmlmap
	-${OMAKE} ACTION=permon_manualpages tree_basic LOC=${LOC}
	-${PYTHON} ${PETSC_DIR}/bin/maint/wwwindex.py ${PERMON_DIR} ${LOC}
	-${OMAKE} ACTION=permon_manexamples tree_basic LOC=${LOC}

# Builds .html versions of the source
alldoc2: chk_loc
	-${OMAKE} ACTION=permon_html PETSC_DIR=${PETSC_DIR} alltree LOC=${LOC}

# Deletes documentation
alldocclean: deletemanualpages allcleanhtml
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/docs/manualpages ]; then \
          find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/docs/manualpages/manualpages.cit ;\
        fi
allcleanhtml:
	-${OMAKE} ACTION=cleanhtml PETSC_DIR=${PETSC_DIR} alltree

