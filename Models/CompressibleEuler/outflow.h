#ifndef OUTFLOW_H
#define OUTFLOW_H

template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalOutflow(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

  SCAL rho       = state(0);
  SCAL rhoinv    = 1. / rho;
  Vec<D, SCAL> U = rhoinv * state.Rows(1, D+1);
  SCAL E         = state(D+1);

  double gm1     = gamma - 1.;
  double gm1inv  = 1. / gm1;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  SCAL Un = 0.;
  for (int dd = 0; dd < D; ++dd)
    Un += U(dd) * sparam.normal(dd);

  SCAL p     = gm1 * (E - 0.5 * rho * U2);
  SCAL c2    = gamma * p * rhoinv;
  SCAL c     = sqrt(c2);

  if (U2 - c2 > 0.) {
    res = state;
    return;
  }

  SCAL GAMINV = 1./gamma;
  SCAL rhob  = rho * pow(p_infty/p,GAMINV);
  SCAL cb2   = gamma * p_infty / rhob;
  SCAL dUnb  = 2. * gm1inv * (c - sqrt(cb2));
    
  Vec<D, SCAL> Ub = U + dUnb * sparam.normal;
  SCAL U2b = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2b += Ub(dd) * Ub(dd);

  res(0)           = rhob;
  res.Rows(1, D+1) = rhob * Ub;
  res(D+1)         = gm1inv * p_infty + 0.5 * rhob * U2b;

  // Passive scalars
  if (COMP > D+2)
    res.Rows(D+2, COMP) = (rhob / state(0)) * state.Rows(D+2, COMP);
}

#endif
