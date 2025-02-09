#ifndef SHOCKRESIDUAL_H
#define SHOCKRESIDUAL_H

//const double eps_viscous = 0.2;

template<int D, int COMP>
class ShockResidual
{
public:	
  template <typename SCAL>
  static void evaluate(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SCAL & res) {

    const double gamma = 1.4;

    SCAL rho       = state(0);
    Vec<D, SCAL> m = state.Rows(1, D+1);
    SCAL E         = state(D+1);

    SCAL rhoinv    = 1. / rho;
    Vec<D, SCAL> U = rhoinv * m;
    
    SCAL U2(0.);
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    SCAL p = (gamma - 1.) * (E - 0.5 * rho * U2);
    if (COMP == D+4)
      p += (5. / 3. - gamma) * state(D+2);
    
    Mat<D, D, SCAL> gradU = rhoinv * (grad.Rows(1, D+1) - U * Trans(grad.Row(0)));
    Vec<D, SCAL> gradp    = (gamma - 1.) * (grad.Row(D+1) - rho * Trans(gradU) * U - 0.5 * U2 * grad.Row(0));
    if (COMP == D+4)
      gradp += (5. / 3. - gamma) * grad.Row(D+2);
    
    SCAL divu(0.);
    for (int dd = 0; dd < D; ++dd)
      divu += gradU(dd, dd);
    
    Vec<COMP, SCAL> resi(0.);
    for (int dd = 0; dd < D; ++dd)
      resi(0) += grad(dd, dd);
    resi.Rows(1, D+1) = gradp + divu * m + grad.Rows(1, D+1) * U;
    resi(D+1) = (E + p) * divu;
    for (int dd = 0; dd < D; ++dd)
      resi(D+1) += U(dd) * (grad(D+1, dd) + gradp(dd));

    // Vec<COMP, SCAL> Dp(0.);

    // Dp(0)           = .5 * (gamma - 1.) * U2;
    // Dp.Rows(1, D+1) = -(gamma - 1.) * U;
    // Dp(D+1)         = gamma - 1.;

    // for (int ll = 0; ll < D+2; ++ll)
    //   resi(ll) *= Dp(ll);
    
    res = 0.;
    for (int ll = 0; ll < D+2; ++ll)
      if (resi(ll) < 0.)
        res -= resi(ll);
      else
        res += resi(ll);
    
    // SCAL gradp2(0.);
    // for (int dd = 0; dd < D; ++dd)
    //   gradp2 += gradp(dd) * gradp(dd);
    // if (gradp2 > 0.)
    //   gradp2 = sqrt(gradp2);

    // res *= gradp2 / (p + 1.e-12);

  }
	
  void LoadParameters(PDE & apde)
  {
  }
};
#endif
