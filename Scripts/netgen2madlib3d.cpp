#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<math.h>
#include "netgen2madlib3d.h"
using namespace std;

int main() {
  Netgen2Madlib3D("../netgen.vol", "../adapted.msh");
  return 0;
}