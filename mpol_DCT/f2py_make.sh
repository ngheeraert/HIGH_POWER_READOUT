rm fortran_module.cpython-37m-darwin.so
rm -r fortran_module.cpython-37m-darwin.so.dSYM
f2py -c -m fortran_module fortran_module.f90
