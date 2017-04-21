#
# This is the makefile for installing PERMON <http://permon.it4i.eu/>.
# Adopted from SLEPc <http://slepc.upv.es/> makefile.

ALL: all
LOCDIR = .
DIRS   = src include
LOG    = ${PERMON_DIR}/${PETSC_ARCH}/lib/permon/conf/make.log
ERRLOG = ${PERMON_DIR}/${PETSC_ARCH}/lib/permon/conf/error.log
HGLOG  = ${PERMON_DIR}/${PETSC_ARCH}/lib/permon/conf/hg.log
MODLOG = ${PERMON_DIR}/${PETSC_ARCH}/lib/permon/conf/modules.log

#  Escape codes to change the text color on xterms and terminals
PETSC_TEXT_HILIGHT = "\033[1;31m"
PETSC_TEXT_NORMAL = "\033[0;39m\033[0;49m"

# Include the rest of makefiles
-include ${PERMON_DIR}/lib/permon/conf/permon_variables
-include ${PERMON_DIR}/lib/permon/conf/permon_rules

#
# Basic targets to build PERMON library
all: chk_all
	@mkdir -p ./${PETSC_ARCH}/lib/permon/conf
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} all-gnumake-local 2>&1 | tee ${LOG}; \
	 else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} all-legacy-local 2>&1 | tee ${LOG} | ${GREP} -v "has no symbols"; \
	 fi
	@egrep -i "( error | error: |no such file or directory)" ${PETSC_ARCH}/lib/permon/conf/make.log | tee ${ERRLOG} > /dev/null
	@if test -s ${ERRLOG}; then \
           printf ${PETSC_TEXT_HILIGHT}"*******************************ERROR************************************\n" 2>&1 | tee -a ${LOG}; \
           echo "  Error during compile, check ./${PETSC_ARCH}/lib/permon/conf/make.log" 2>&1 | tee -a ${LOG}; \
           echo "  Send all contents of ./${PETSC_ARCH}/lib/permon/conf to vaclav.hapla@vsb.cz" 2>&1 | tee -a ${LOG};\
           printf "************************************************************************"${PETSC_TEXT_NORMAL}"\n" 2>&1 | tee -a ${LOG}; \
					 exit 1; \
	 else \
		 echo "Completed building libraries in ${PERMON_DIR}/${PETSC_ARCH}" | tee -a ${LOG}; \
 echo "=========================================" | tee -a ${LOG}; \
	 fi
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} permon_hginfo

all-gnumake: chk_all
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
          ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} PERMON_BUILD_USING_CMAKE="" all;\
        else printf ${PETSC_TEXT_HILIGHT}"Build not configured for GNUMAKE. Quiting"${PETSC_TEXT_NORMAL}"\n"; exit 1; fi

all-legacy: chk_all
	@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} PERMON_BUILD_USING_CMAKE="" MAKE_IS_GNUMAKE="" all

all-gnumake-local: permon_info permon_build_gnumake

all-legacy-local:  permon_info deletelibs permon_build_legacy permon_shared

chk_all: chk_permon_dir chk_permon_petsc_dir chklib_dir chk_makej

#
# Check if PETSC_DIR variable specified is valid
#
#
chk_permon_petsc_dir:
	@if [ -z ${PETSC_DIR} ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "PETSC_DIR not specified!"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
  false; fi
	@if [ ! -f ${PETSC_DIR}/include/petscversion.h ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}!"; \
    echo 'File ${PETSC_DIR}/include/petscversion.h does not exist.'; \
	  echo "Note you need to use / to separate directories, not \\"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
    false; fi
	@if [ ! -f ${PETSC_DIR}/${PETSC_ARCH}/lib/lib*petsc*.a ] && [ ! -f ${PETSC_DIR}/${PETSC_ARCH}/lib/lib*petsc*.so ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Given PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"; \
    echo "  (directory ${PETSC_DIR}/${PETSC_ARCH}/lib)"; \
    echo "  does not contain compiled PETSc library (libpetsc.so or libpetsc.a)!"; \
	  echo "You may need to run configure and make in ${PETSC_DIR}"; \
	  echo "Note you need to use / to separate directories, not \\"; \
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
	  echo "Note: You need to use / to separate directories, not \\"; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi
	@if [ "${true_PERMON_DIR}" != "${mypwd}" ]; then \
          printf ${PETSC_TEXT_HILIGHT}"*************************ERROR**************************************\n"; \
	  echo "Your PERMON_DIR does not match the directory you are in"; \
	  echo "PERMON_DIR: "${true_PERMON_DIR}; \
	  echo "Current directory: "${mypwd}; \
          printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n"; \
	  false; fi

permon_hginfo:

#
# Prints information about the system and version of PERMON being compiled
#
permon_info:
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
# Builds the PERMON library using PETSc legacy buildsystem
#
permon_build_legacy: chk_makej
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PERMON_DIR=${PERMON_DIR} ACTION=libfast tree
	-@${RANLIB} ${PERMON_LIB_DIR}/*.${AR_LIB_SUFFIX}  > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
	-@echo "Completed building libraries"
	-@echo "========================================="

#
# Builds the PERMON library using PETSc GNU Make parallel buildsystem
#
permon_build_gnumake: chk_makej
	@echo "Building PERMON using GNU Make with ${MAKE_NP} build threads"           
	@echo "=========================================="
	@cd ${PERMON_DIR} && ${OMAKE_PRINTDIR} -f gmakefile -j ${MAKE_NP} V=${V}
	@echo "========================================="

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

