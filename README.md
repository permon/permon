PERMON toolbox
================================
PERMON (Parallel Efficient Robust Modular Object Numerical) is a software toolbox for quadratic programming (QP) based on [PETSc](http://www.mcs.anl.gov/petsc/). It also includes domain decomposition methods (FETI and Total FETI) and Support Vector Machines (SVM) machine learning methods.

Project homepage: <http://permon.vsb.cz>  

PERMON contains following modules:

* [PermonQP](http://permon.vsb.cz/permonqp.htm) - contains QP solvers and (Total) FETI domain decomposition methods
* [PermonSVM](http://permon.vsb.cz/permonsvm.htm) - SVM machine learning implementation based on PermonQP - [separate repository](https://github.com/permon/permonsvm)

Please use [GitHub](https://github.com/permon/permon) for issues and pull requests.

Quick guide to PERMON installation
------------------------------------

1. set environment variables `PETSC_DIR` and `PETSC_ARCH` pointing to a PETSc instance
   - `PETSC_ARCH` is empty in case of "prefix" PETSc installation
   - for more details about [PETSc](http://www.mcs.anl.gov/petsc/) installation and the two environment variables, see [PETSc documentation](http://www.mcs.anl.gov/petsc/documentation/installation.html)
2. set `PERMON_DIR` variable pointing to the PERMON directory (probably this file's parent directory)
3. build PERMON simply using makefile (makes use of PETSc buildsystem):

     `make`
4. if the build is successful, there is a new subdirectory named `$PETSC_ARCH`, the program library is `$PETSC_ARCH/lib/libpermon.{so,a}`
   - shared library (.so) is built only if PETSc has been configured with option `--with-shared-libraries`
   - all compiler settings are inherited from PETSc

Documentation and examples
----------------------------------
The documentation of the routines is available at <http://permon.vsb.cz/documentation.htm>. There are several examples in the [tutorials](https://github.com/permon/permon/tree/master/src/tutorials) directory illustrating the basic usage of both modules.

Supported PETSc versions
----------------------------------
PERMON tries to support newest versions of PETSc as soon as possible. The [releases](https://github.com/permon/permon/releases) are tagged with major.minor.sub-minor numbers. The major.minor numbers correspond to the major.minor release numbers of the supported PETSc version.
