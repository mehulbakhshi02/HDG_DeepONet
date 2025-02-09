#include <vector>
#include <fstream>
#include <iomanip>
#include "refine2netgen.h"

int main() {

  Refine2Netgen("../adapted.mesh", "../netgen.vol");
  
  return 0;

}