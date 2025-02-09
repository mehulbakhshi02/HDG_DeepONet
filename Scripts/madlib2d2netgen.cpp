#include <vector>
#include <fstream>
#include <iomanip>
#include<iostream>

#include "madlib2d2netgen.h"
using namespace std;
int main() {

  Mmg2D2Netgen("../adapted.msh", "../netgen.vol");
  
  return 0;

}