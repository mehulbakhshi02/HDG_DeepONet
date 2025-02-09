#include <vector>
#include <fstream>

#include "angener2netgen.h"

int main() {

  Angener2Netgen("angener.mesh", "netgen.vol");

  return 0;

}