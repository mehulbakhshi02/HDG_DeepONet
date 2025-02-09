template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::SetPseudoCFL(const int i, const double resL2, const double resLInf) const {

  switch (pcfl_strategy) {
  case 0: { 
    // Bassi et al.
    const double f = max(resL2 / resL20, resLInf / resLInf0);
    pcfl = min(pcfl_min * pow(f, -pcfl_beta), pcfl_max);
    break;
  }
  case 1: {
    // Francesca
    double res = pcfl;
    if (i <= pcfl_reset) res = pcfl_min * (-2. * pow((i+1.)/pcfl_reset, 3) + 3. * pow((i+1.)/pcfl_reset, 2));
    if (i > pcfl_reset) res = pcfl * (1 + pcfl_beta * log10(resL20 / resL2));
    if (resL2 < pcfl_full_newton && resL20 > resL2) res = pcfl * (1 + pcfl_beta * log10(resL20 / resL2));
    if (resL20 < resL2 && i > pcfl_reset) res = pcfl;
    pcfl = res;
    break;
  }
  }
}
