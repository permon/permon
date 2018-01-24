PERMON toolbox
================================
PERMON (Parallel Efficient Robust Modular Object Numerical) is a software toolbox for quadratic programming (QP) based on [PETSc](http://www.mcs.anl.gov/petsc/). It also includes domain decomposition methods (FETI and Total FETI) as a specific QP implementation.

Project homepage: <http://permon.it4i.cz>  

PERMON contains following modules:

* [PermonQP](http://permon.it4i.cz/permonqp.htm) - contains QP transformations and solvers
* [PermonFLLOP](http://permon.it4i.cz/permonfllop.htm) - (FETI Light Layer On top of PETSc) implements (Total) FETI

Support Vector Machines (SVMs) implementation [PermonSVM](http://permon.it4i.cz/permonsvm.htm) based on PermonQP can be found at <https://github.com/It4innovations/permonsvm>.


Please use [GitHub](https://github.com/It4innovations/permon) for issues and pull requests.

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
The documentation of the routines is available at <http://permon.it4i.cz/documentation.htm>. There are several examples in the [examples](https://github.com/It4innovations/permon/tree/master/examples) directory illustrating the basic usage of both modules.

Supported PETSc versions
----------------------------------
PERMON tries to support newest versions of PETSc as soon as possible. The [releases](https://github.com/It4innovations/permon/releases) are tagged with major.minor.sub-minor numbers. The major.minor numbers correspond to the major.minor release numbers of the supported PETSc version.

