# Unified Framework

Unified Framework is a 2D/3D higher order (Hybridized) Discontinuous Galerkin solver implemented in C++.

The solver uses geometry and mesh using NGSolve. The basis functions and quartarure rules are also generated using NGSolve. The solver can handle mixed elements in the mesh.

In addition, the software can perform isotropic (element splitting) as well as anisotropic (metric based) adaptation. 

Metric based adaptation can be performed using multiple (external) mesh generators like BAMG, Madlib, Angener90, MMG2D as well as refine and MMG3D.

## Dependencies

1. NGSolve (required) - Used for handling mesh, geometry and generating basis functions.
2. PETSc (required) - Used for the linear solver.

## Optional tools for mesh adaptation in 2D
1. BAMG
2. Angener90
3. Madlib
4. MMG2D

## Optional tools for mesh adaptation in 3D
1. refine
2. MMG3D
3. Madlib

## Instructions to compile the dependencies
The instructions to compile and configure both the required and optional dependencies are given below. 
### Installing NGSolve
1. Follow the instructions to install [NGSolve](https://ngsolve.org/) either from source or directly using the binaries.
2. Replace autodiff.hpp (found in the `include` folder of the NGSolve directory Ex: `$NETGENDIR../include/`) with the one found in the folder `unifiedframework/Extra/ngsolve/`
3. Add the `bin` directory of NGSolve into `$PATH` (`ngs` should be callable from the terminal)

For additional options about compiling NGSolve with MPI or OpenCascade see `unifiedframework/Extra/ngsolve/README.md`.

Note: Replacing autodiff.hpp is important only when solving Euler and Navier-Stokes. This is because by default NGSolve does not define the function `pow(AutoDiff x, SCAL y)`. 


### Installing PETSc
1. Follow the instructions to compile [PETSc](https://www.mcs.anl.gov/petsc/index.html)
2. Configure with the following options `./configure --with-fc=gfortran --with-mpi=0`
3. Create the variables `PETSC_DIR` and `PETSC_ARCH` as per instructions in the installation. (`PETSC_DIR` should point to the directory where PETSc was installed and `PETSC_ARCH` should be specified as shown on screen during installation)


### Installing BAMG
1. Download [FreeFem++](https://doc.freefem.org/introduction/installation.html) and compile from source
2. Replace FreeFem-sources/src/bamglib/MeshGeom.cpp with the hacked version of MeshGeom.cpp
3. Compile as usual
4. Add the `bin` directory to `$PATH` (`bamg` should be callable from the terminal)

An older version of BAMG is saved in `unifiedframework/Extra/dependencies`


### Using MMG2D/MMG3D
1. Download the binary for [MMG](https://www.mmgtools.org/mmg-remesher-downloads) for the appropriate OS
2. Add the `bin` directory to `$PATH` (`mmg2d_O3` and `mmg3d_O3` should be callable from the terminal)

An older version of MMG is saved in `unifiedframework/Extra/dependencies`


### Installing Madlib
1. Download the [Madlib](https://sites.uclouvain.be/madlib/) branch called [MeshAdapt.8.9.2017](https://svn.cenaero.be/MAdLib/branches/MeshAdapt.8.09.2017/) using the command
`svn co https://svn.cenaero.be/MAdLib/branches/MeshAdapt.8.09.2017/`
2. Inside create two folders `MeshAdapt.8.09.2017/install` and `MeshAdapt.8.09.2017/build`
3. Place the bash script (in `unifiedframework/Extra/madlib/`) file in `MeshAdapt.8.09.2017/build` folder
4. Edit the bash script to set the compilers for g++, gcc and fortran
5. Within `MeshAdapt.8.09.2017/Testcases` folder copy the file (`unifiedframework/Extra/madlib/CMakeLists.txt`) and folders `unifiedframework/Extra/madlib/CustomAdaptation` and `unifiedframework/Extra/madlib/CustomAdaptation3d` (`CMakeLists.txt` has to be replaced)
6. Run the bash script
7. Add `MeshAdapt.8.09.2017/build/Testcases` to `$PATH` (`CustomAdaptation` and `CustomAdaptation3D` should be callable from the terminal)

An older version of madlib is saved in `unifiedframework/Extra/dependencies`


### Installing Angener 3.0
1. Download [Angener](https://www2.karlin.mff.cuni.cz/~dolejsi/angen/angen.htm)
2. Follow the instructions on the website to install
3. Add the installed directory to `$PATH` (`side` has to be callable in the terminal. However, within the code one calles `Angener90` so in the bashrc or zshrc file one needs to have `alias Angener90='side'`)

Note: For Angener meshes, only one edge of an element can be a boundary edge. 

An older version of Angener is saved in `unifiedframework/Extra/dependencies`


### Installing refine
1. Download [refine](https://github.com/nasa/refine)
2. Install following the instructions given on the website
3. Add `bin` to `$PATH` (`ref` should be callable from the terminal)

For additional options about compiling refine with egads or OpenCascade see `unifiedframework/Extra/refine/README.md`

## Installing Unified Framework
1. Have all the required dependencies installed and added to `$PATH`
2. Clone the master
3. Change to the branch `git checkout ml_adapt`
4. Set the following in the `Makefile`
	1. Comment or uncomment `CFLAGS=-DLAXFRIEDRICH` or `CFLAGS=-DROE` to choose between Lax-Friedrich and Roe convective flux
		-	Lax-Friedrich flux is to be preferred for scalar test cases
		-	Roe flux performs better for Euler and Navier-Stokes
	2. Comment or uncomment `CFLAGS=-DLOCALDG` or `CFLAGS=-DBR2` to choose between Local-LDG and BR2 viscous flux
		-	Local-LDG flux is to be preferred for scalar test cases
		-	BR2 flux performs better for Euler and Navier-Stokes
	3. Comment or uncomment `LDFLAGS` based on the OS being used
5. Compile the wrapper used for TecPlot post-processing `make wrapper` (Should be done when compiling the first time)
6. Compile the code using `make`

Note that on the RWTH cluster one could encounter the error
> /usr/bin/ld: cannot find -lngsolve

In that case, change the flag `-lngsolve` when compiling `main.so` to `-lsolve`.

## Usage
To run a simple adaptation use the script `run.py`. If all the libraries have been installed and the code has been compiled once then this should run anisotropic adaptation with increasing number of degrees-of-freedom ($`DoF`$).