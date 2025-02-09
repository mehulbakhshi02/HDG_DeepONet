# Installing refine with extra options

## Installing EGAD

To install refine with EGADS one has to first install EGADS

1. Download [EGADS](https://acdl.mit.edu/ESP/)
2. Follow the instruction to install it.


To compile refine with EGADS and MPI use the following command during `configure`:\
`../configure --with-EGADS=/directory/where/EngSketchPad/is/installed --with-OpenCASCADE=/usr/local/Cellar/opencascade/7.x.x CC=mpicc FC=mpif90 CFLAGS=-DHAVE_MPI --prefix=/preferred/location/refine/build`

Note: This has been tested only on MacOS (10.14)