#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "bamg2netgen.h"

int main() {

  Bamg2Netgen("../bamg.mesh", "../netgen.vol");

  return 0;

}