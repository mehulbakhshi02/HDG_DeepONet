template<int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
  ::LimitUpdate(double & alpha, Solution & delta, Solution & sol) {

  double limit = SolverData::max_change;

  if (COMP < 4)
    return;

  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];
    
    int offset_w = ed.offset_w;
    int ndof_w   = ed.ndof_w;
    int nip      = ed.nip;
    
    mygemm('t', 'n', COMP, nip, ndof_w, 1., &sol.vecW[COMP*offset_w], &ed.phi[0], 0., &temp[0]);
    mygemm('t', 'n', COMP, nip, ndof_w, 1., &delta.vecW[COMP*offset_w], &ed.phi[0], 0., &temp2[0]);
    
    for (int j = 0; j < nip; ++j) {
      const double * pw  = &temp[COMP * j];
      const double * pdw = &temp2[COMP * j];
      const double * pq  = &ed.qp[D * j];

      // Check for valid update step wrt density
      double alpha_rho = -1. * limit * pw[0] / pdw[0];
      if (alpha_rho < 0. || alpha_rho > 1.)
        alpha_rho = 1.;
      
      // Check for valid update step wrt pressure
      double alpha_p = alpha_rho;

      Vec<COMP> state, state2;
      for (int ll = 0; ll < COMP; ++ll)
        state(ll) = pw[ll];

      SpatialParams<D> sparam;
      for (int dd = 0; dd < D; ++dd)
        sparam.pos(dd) = pq[dd];
      
      double p = Model::EvalConstrPrimVar(0, state, sparam);

      for (int ll = 0; ll < COMP; ++ll)
        state2(ll) = state(ll) + alpha_p * pdw[ll];

      double p2 = Model::EvalConstrPrimVar(0, state2, sparam);

      while (p2 < (1. - limit) * p) {
        alpha_p *= 0.75;

        for (int ll = 0; ll < COMP; ++ll)
          state2(ll) = state(ll) + alpha_p * pdw[ll];

        p2 = Model::EvalConstrPrimVar(0, state2, sparam);
      }
      if(!isnan(alpha_p) && !isnan(alpha_rho))      
        alpha = min(min(alpha_rho, alpha_p), alpha);
    }
  }

  //  return;
  
  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;
    FacetData<D, COMP> & fd = *fadata[i];
    
    int offset_l = fd.offset_l;
    int ndof_l   = fd.ndof_l;
    int nip      = fd.nip;

    if (ndof_l == 0) continue;
    
    mygemm('t', 'n', COMP, nip, ndof_l, 1., &sol.vecL[COMP*offset_l], &fd.mu[0], 0., &temp[0]);
    mygemm('t', 'n', COMP, nip, ndof_l, 1., &delta.vecL[COMP*offset_l], &fd.mu[0], 0., &temp2[0]);
    
    for (int j = 0; j < nip; ++j) {
      const double * pl  = &temp[COMP * j];
      const double * pdl = &temp2[COMP * j];
      const double * pq  = &fd.qp[D * j];
      
      // Check for valid update step wrt density
      double alpha_rho = -1. * limit * pl[0] / pdl[0];
      if (alpha_rho < 0. || alpha_rho > 1.)
        alpha_rho = 1.;
      
      // Check for valid update step wrt pressure
      double alpha_p = alpha_rho;

      Vec<COMP> state, state2;
      for (int ll = 0; ll < COMP; ++ll)
        state(ll) = pl[ll];

      SpatialParams<D> sparam;
      for (int dd = 0; dd < D; ++dd)
        sparam.pos(dd) = pq[dd];
      
      double p = Model::EvalConstrPrimVar(0, state, sparam);

      for (int ll = 0; ll < COMP; ++ll)
        state2(ll) = state(ll) + alpha_p * pdl[ll];

      double p2 = Model::EvalConstrPrimVar(0, state2, sparam);

      while (p2 < (1. - limit) * p) {
        alpha_p *= 0.75;

        for (int ll = 0; ll < COMP; ++ll)
          state2(ll) = state(ll) + alpha_p * pdl[ll];

        p2 = Model::EvalConstrPrimVar(0, state2, sparam);
      }
      if(!isnan(alpha_p) && !isnan(alpha_rho))      
        alpha = min(min(alpha_rho, alpha_p), alpha);

    }
  }  
}
