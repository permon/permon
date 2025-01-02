Contributions are appreciated and encouraged.
They can be sumitted via GitHub pull request to

  https://github.com/permon/permon

If you are planning a large contribution, we
encourage you to discuss the concept by creating an issue and interact
with us frequently to ensure that your effort is well-directed.

## Guidelines
PERMON follows PETSc code standards.
Please read the code standards chapter of the PETSc developer guide https://petsc.org/release/developers/style/. The code style can be checked and mostly enforced by running
```
make checkbadsource
make permonclangformat
```

Before filing a pull request (PR):

- If your contribution can be logically decomposed into 2 or more separate contributions, submit them in sequence with different branches instead of all at once.
- Include tests which cover any changes to the source code.
- Before submitting any PR, run the full test suite - i.e `make alltests TIMEOUT=600` on your machine

## Certificate of Origin

PERMON is distributed under a 2-clause BSD license (see LICENSE).  The
act of submitting a pull request or patch (with or without an explicit
Signed-off-by tag) will be understood as an affirmation of the
following:

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
