#include <vector>
#include <fstream>
#include <iomanip>
#include<iostream>

#include "madlib3d2netgen.h"
using namespace std;
int main() {

  Madlib3D2Netgen("../adapted.mesh", "../netgen.vol");
  
  return 0;

}