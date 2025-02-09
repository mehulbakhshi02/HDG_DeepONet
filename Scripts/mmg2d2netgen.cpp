#include <vector>
#include <fstream>
#include <iomanip>
#include<iostream>

#include "mmg2d2netgen.h"
using namespace std;
int main() {

  Mmg2Netgen2D("../adapted.mesh", "../netgen.vol");
  
  return 0;

}