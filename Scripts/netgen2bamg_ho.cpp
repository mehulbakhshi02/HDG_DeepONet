#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include <cmath>
#include "netgen2bamg_ho.h"

int main() {

  Netgen2Bamg("netgen.in2d", "bamg.geo", "netgen.vol", "bamg.mesh");
//  Netgen2Bamg("../netgen.in2d", "../bamg.geo", "../netgen.vol", "../bamg.mesh");
  return 0;

}
