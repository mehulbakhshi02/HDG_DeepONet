# Installing NGSolve with extra options

## Installing NGSolve with MPI

1. To install NGSolve on either MacOS (tested on 10.14) or Ubuntu 18.04.5 LTS with MPI support, build NGSolve from sources as given [here](https://ngsolve.org/docu/latest/install/install_sources.html). 
2. Call `cmake` with the following options
`cmake -DUSE_MPI=mpicc -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install ${BASEDIR}/ngsolve-src`
3. Proceed as usual for the next steps

## Installing NGSolve with OpenCascade support
This is important if we wish to generate mesh and geometry from CAD. 

### Steps to install Open Cascade (tested on 7.3 and 7.4) on MacOS (tested on 10.14)

1. Install `homebrew`
2. Call `brew install opencascade`
3. This should install Open Cascade in the folder `/usr/local/Cellar/opencascade/7.x.x`

### Steps to install Open Cascade on Ubuntu 18.04.5 LTS

1. `sudo apt-add-repository universe`
2. `sudo apt-get update`
3. `sudo apt-get install liboce-ocaf-dev`
4. This should install Open Cascade in the folder `/usr/include/oce/`

To install NGSolve with Open Cascade follow the instructions to compile from source. The `cmake` call on MacOS has to be as follows:\
`cmake -DUSE_OCC=ON -DOCC_INCLUDE_DIR=/usr/local/Cellar/opencascade/7.x.x/include/opencascade/ -DOCC_LIB_DIR=/usr/local/Cellar/opencascade/7.x.x/lib/ $NGROOT/ngsolve-src -DCMAKE_INSTALL_PREFIX=$NGROOT/ngsolve-install`

On Ubuntu the following can be used to compile\
`cmake -DUSE_OCC=ON -DOCC_INCLUDE_DIR=/usr/include/oce/ $NGROOT/ngsolve-src -DCMAKE_INSTALL_PREFIX=$NGROOT/ngsolve-install`