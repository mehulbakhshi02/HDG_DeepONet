#include <vector>
#include <fstream>
#include <iomanip>
#include<iostream>

#include "mmg3d2netgen.h"
using namespace std;
int main() {

  Mmg2Netgen3D("../adapted.mesh", "../netgen.vol");
  
  return 0;

}