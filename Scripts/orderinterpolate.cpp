#include <vector>
#include <fstream>
#include<iostream>
#include <stdlib.h>
#include<cmath>
#include "orderinterpolate.h"

int main() {

  OrderInterpolate("netgen_prev.vol", "nodeorder.bb", "netgen.vol", "nodeorder_new.bb");
  
  return 0;

}