#
# This is the makefile for installing PERMON <http://permon.it4i.cz/>.
#

ALL: permon-all
all: permon-all
LOCDIR = .
DIRS   = src include

# Include the rest of makefiles
include lib/permon/conf/permon_variables
include lib/permon/conf/permon_rules
include lib/permon/conf/permon_test

cleanbin:                          
	-@${RM} ${PERMON_DIR}/${PETSC_ARCH}/bin/*      

clean:: allclean
