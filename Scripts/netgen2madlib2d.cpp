#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<math.h>
#include "netgen2madlib2d.h"
using namespace std;

int main() {
  Netgen2Madlib2D("../netgen.vol", "../adapted.msh");
  return 0;
}