#!/usr/bin/env python
# gmake file list ($PERMON_DIR/$PETSC_ARCH/lib/permon/conf/files) generator
# adopted from SLEPc <http://slepc.upv.es/>.

import os
from distutils.sysconfig import parse_makefile
import sys
import logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config'))

PKGS = 'sys vec mat ksp qp qpc qppf pc qps interface'.split()
LANGS = dict(c='C', cxx='CXX', cu='CU', F='F')

AUTODIRS = set('ftn-auto ftn-custom f90-custom'.split()) # Automatically recurse into these, if they exist
SKIPDIRS = set('benchmarks'.split())                     # Skip these during the build
NOWARNDIRS = set('tests tutorials'.split())              # Do not warn about mismatch in these

# compatibility code for python-2.4 from http://code.activestate.com/recipes/523034-emulate-collectionsdefaultdict/
try:
    from collections import defaultdict
except:
    class defaultdict(dict):
        def __init__(self, default_factory=None, *a, **kw):
            if (default_factory is not None and
                not hasattr(default_factory, '__call__')):
                raise TypeError('first argument must be callable')
            dict.__init__(self, *a, **kw)
            self.default_factory = default_factory
        def __getitem__(self, key):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return self.__missing__(key)
        def __missing__(self, key):
            if self.default_factory is None:
                raise KeyError(key)
            self[key] = value = self.default_factory()
            return value
        def __reduce__(self):
            if self.default_factory is None:
                args = tuple()
            else:
                args = self.default_factory,
            return type(self), args, None, None, self.items()
        def copy(self):
            return self.__copy__()
        def __copy__(self):
            return type(self)(self.default_factory, self)
        def __deepcopy__(self, memo):
            import copy
            return type(self)(self.default_factory,
                              copy.deepcopy(self.items()))
        def __repr__(self):
            return 'defaultdict(%s, %s)' % (self.default_factory,
                                            dict.__repr__(self))

try:
    all([True, True])
except NameError:               # needs python-2.5
    def all(iterable):
        for i in iterable:
            if not i:
                return False
        return True

try:
    os.path.relpath             # needs python-2.6
except AttributeError:
    def _relpath(path, start=os.path.curdir):
        """Return a relative version of a path"""

        from os.path import curdir, abspath, commonprefix, sep, pardir, join
        if not path:
            raise ValueError("no path specified")

        start_list = [x for x in abspath(start).split(sep) if x]
        path_list = [x for x in abspath(path).split(sep) if x]

        # Work out how much of the filepath is shared by start and path.
        i = len(commonprefix([start_list, path_list]))

        rel_list = [pardir] * (len(start_list)-i) + path_list[i:]
        if not rel_list:
            return curdir
        return join(*rel_list)
    os.path.relpath = _relpath

class debuglogger(object):
    def __init__(self, log):
        self._log = log

    def write(self, string):
        self._log.debug(string)

def pathsplit(path):
    """Recursively split a path, returns a tuple"""
    stem, basename = os.path.split(path)
    if stem == '':
        return (basename,)
    if stem == path:            # fixed point, likely '/'
        return (path,)
    return pathsplit(stem) + (basename,)

class Mistakes(object):
    def __init__(self, log, verbose=False):
        self.mistakes = []
        self.verbose = verbose
        self.log = log

    def compareDirLists(self,root, mdirs, dirs):
        if NOWARNDIRS.intersection(pathsplit(root)):
            return
        smdirs = set(mdirs)
        sdirs  = set(dirs).difference(AUTODIRS)
        if not smdirs.issubset(sdirs):
            self.mistakes.append('Makefile contains directory not on filesystem: %s: %r' % (root, sorted(smdirs - sdirs)))
        if not self.verbose: return
        if smdirs != sdirs:
            from sys import stderr
            stderr.write('Directory mismatch at %s:\n\t%s: %r\n\t%s: %r\n\t%s: %r\n'
                         % (root,
                            'in makefile   ',sorted(smdirs),
                            'on filesystem ',sorted(sdirs),
                            'symmetric diff',sorted(smdirs.symmetric_difference(sdirs))))

    def compareSourceLists(self, root, msources, files):
        if NOWARNDIRS.intersection(pathsplit(root)):
            return
        smsources = set(msources)
        ssources  = set(f for f in files if os.path.splitext(f)[1] in ['.c', '.cxx', '.cc', '.cu', '.cpp', '.F'])
        if not smsources.issubset(ssources):
            self.mistakes.append('Makefile contains file not on filesystem: %s: %r' % (root, sorted(smsources - ssources)))
        if not self.verbose: return
        if smsources != ssources:
            from sys import stderr
            stderr.write('Source mismatch at %s:\n\t%s: %r\n\t%s: %r\n\t%s: %r\n'
                         % (root,
                            'in makefile   ',sorted(smsources),
                            'on filesystem ',sorted(ssources),
                            'symmetric diff',sorted(smsources.symmetric_difference(ssources))))

    def summary(self):
        for m in self.mistakes:
            self.log.write(m + '\n')
        if self.mistakes:
            raise RuntimeError('SLEPc makefiles contain mistakes or files are missing on filesystem.\n%s\nPossible reasons:\n\t1. Files were deleted locally, try "hg revert filename" or "git checkout filename".\n\t2. Files were deleted from repository, but were not removed from makefile. Send mail to permon-maint@upv.es.\n\t3. Someone forgot to "add" new files to the repository. Send mail to permon-maint@upv.es.' % ('\n'.join(self.mistakes)))

def stripsplit(line):
  return line[len('#requires'):].replace("'","").split()

class Permon(object):
    def __init__(self, permon_dir=None, petsc_dir=None, petsc_arch=None, installed_petsc=False, verbose=False):
        if permon_dir is None:
            permon_dir = os.environ.get('PERMON_DIR')
            if permon_dir is None:
                raise RuntimeError('Could not determine PERMON_DIR, please set in environment')
        if petsc_dir is None:
            petsc_dir = os.environ.get('PETSC_DIR')
            if petsc_dir is None:
                raise RuntimeError('Could not determine PETSC_DIR, please set in environment')
        if petsc_arch is None:
            petsc_arch = os.environ.get('PETSC_ARCH')
            if petsc_arch is None:
                raise RuntimeError('Could not determine PETSC_ARCH, please set in environment')
        self.permon_dir = permon_dir
        self.petsc_dir = petsc_dir
        self.petsc_arch = petsc_arch
        self.installed_petsc = installed_petsc
        self.read_conf()
        logging.basicConfig(filename=self.arch_path('lib','permon','conf', 'gmake.log'), level=logging.DEBUG)
        self.log = logging.getLogger('gmakegen')
        self.mistakes = Mistakes(debuglogger(self.log), verbose=verbose)
        self.gendeps = []

    def petsc_path(self, *args):
        if self.installed_petsc:
            return os.path.join(self.petsc_dir, *args)
        else:
            return os.path.join(self.petsc_dir, self.petsc_arch, *args)

    def arch_path(self, *args):
        return os.path.join(self.permon_dir, self.petsc_arch, *args)

    def read_conf(self):
        self.conf = dict()
        for line in open(self.petsc_path('include', 'petscconf.h')):
            if line.startswith('#define '):
                define = line[len('#define '):]
                space = define.find(' ')
                key = define[:space]
                val = define[space+1:]
                self.conf[key] = val
        self.conf.update(parse_makefile(self.petsc_path('lib','petsc','conf', 'petscvariables')))
        """for line in open(self.arch_path('include', 'permonconf.h')):
            if line.startswith('#define '):
                define = line[len('#define '):]
                space = define.find(' ')
                key = define[:space]
                val = define[space+1:]
                self.conf[key] = val
        self.conf.update(parse_makefile(self.arch_path('lib','permon','conf', 'permonvariables')))"""
        self.have_fortran = int(self.conf.get('PETSC_HAVE_FORTRAN', '0'))

    def inconf(self, key, val):
        if key in ['package', 'function', 'define']:
            return self.conf.get(val)
        elif key == 'precision':
            return val == self.conf['PETSC_PRECISION']
        elif key == 'scalar':
            return val == self.conf['PETSC_SCALAR']
        elif key == 'language':
            return val == self.conf['PETSC_LANGUAGE']
        raise RuntimeError('Unknown conf check: %s %s' % (key, val))

    def relpath(self, root, src):
        return os.path.relpath(os.path.join(root, src), self.permon_dir)

    def get_sources(self, makevars):
        """Return dict {lang: list_of_source_files}"""
        source = dict()
        for lang, sourcelang in LANGS.items():
            source[lang] = [f for f in makevars.get('SOURCE'+sourcelang,'').split() if f.endswith(lang)]
        return source

    def gen_pkg(self, pkg):
        pkgsrcs = dict()
        for lang in LANGS:
            pkgsrcs[lang] = []
        for root, dirs, files in os.walk(os.path.join(self.permon_dir, 'src', pkg)):
            makefile = os.path.join(root,'makefile')
            if not os.path.exists(makefile):
                dirs[:] = []
                continue
            mklines = open(makefile)
            conditions = set(tuple(stripsplit(line)) for line in mklines if line.startswith('#requires'))
            mklines.close()
            if not all(self.inconf(key, val) for key, val in conditions):
                dirs[:] = []
                continue
            makevars = parse_makefile(makefile)
            mdirs = makevars.get('DIRS','').split() # Directories specified in the makefile
            self.mistakes.compareDirLists(root, mdirs, dirs) # diagnostic output to find unused directories
            candidates = set(mdirs).union(AUTODIRS).difference(SKIPDIRS)
            dirs[:] = list(candidates.intersection(dirs))
            allsource = []
            def mkrel(src):
                return self.relpath(root, src)
            source = self.get_sources(makevars)
            for lang, s in source.items():
                pkgsrcs[lang] += map(mkrel, s)
                allsource += s
            self.mistakes.compareSourceLists(root, allsource, files) # Diagnostic output about unused source files
            self.gendeps.append(self.relpath(root, 'makefile'))
        return pkgsrcs

    def gen_gnumake(self, fd):
        def write(stem, srcs):
            fd.write('%s :=\n' % stem)
            for lang in LANGS:
                fd.write('%(stem)s.%(lang)s := %(srcs)s\n' % dict(stem=stem, lang=lang, srcs=' '.join(srcs[lang])))
                fd.write('%(stem)s += $(%(stem)s.%(lang)s)\n' % dict(stem=stem, lang=lang))
        for pkg in PKGS:
            srcs = self.gen_pkg(pkg)
            write('srcs-' + pkg, srcs)
        return self.gendeps

    def summary(self):
        self.mistakes.summary()

def WriteGnuMake(permon):
    arch_files = permon.arch_path('lib','permon','conf', 'files')
    fd = open(arch_files, 'w')
    gendeps = permon.gen_gnumake(fd)
    fd.write('\n')
    fd.write('# Dependency to regenerate this file\n')
    fd.write('%s : %s %s\n' % (os.path.relpath(arch_files, permon.permon_dir),
                               os.path.relpath(__file__, os.path.realpath(permon.permon_dir)),
                               ' '.join(gendeps)))
    fd.write('\n')
    fd.write('# Dummy dependencies in case makefiles are removed\n')
    fd.write(''.join([dep + ':\n' for dep in gendeps]))
    fd.close()

def main(permon_dir=None, petsc_dir=None, petsc_arch=None, installed_petsc=False, output=None, verbose=False):
    if output is None:
        output = 'gnumake'
    writer = dict(gnumake=WriteGnuMake)
    permon = Permon(permon_dir=permon_dir, petsc_dir=petsc_dir, petsc_arch=petsc_arch, installed_petsc=installed_petsc, verbose=verbose)
    writer[output](permon)
    permon.summary()

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', action='store_true', default=False)
    parser.add_option('--petsc-arch', help='Set PETSC_ARCH different from environment', default=os.environ.get('PETSC_ARCH'))
    parser.add_option('--installed-petsc', help='Using a prefix PETSc installation', default=False)
    parser.add_option('--output', help='Location to write output file', default=None)
    opts, extra_args = parser.parse_args()
    if extra_args:
        import sys
        sys.stderr.write('Unknown arguments: %s\n' % ' '.join(extra_args))
        exit(1)
    main(petsc_arch=opts.petsc_arch, installed_petsc=opts.installed_petsc, output=opts.output, verbose=opts.verbose)
