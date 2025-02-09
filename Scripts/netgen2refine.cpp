#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<math.h>
#include "netgen2refine.h"

using namespace std;


int main() {
  Netgen2Refine("../netgen.vol", "../adapted.mesh");
  return 0;
}