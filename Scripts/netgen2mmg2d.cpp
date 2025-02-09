#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<math.h>
#include "netgen2mmg2d.h"
using namespace std;

int main() {
  Netgen2Mmg2D("../netgen.vol", "../adapted.mesh");
  return 0;
}