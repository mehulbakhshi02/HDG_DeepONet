#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include "netgen2bamg_gmsh.h"

int main() {

//  Netgen2Bamg("netgen.in2d", "bamg.geo", "netgen.vol", "bamg.mesh");
  Netgen2Bamg("../netgen.in2d", "../bamg.geo", "../netgen.vol", "../bamg_gmsh.mesh");
  return 0;

}