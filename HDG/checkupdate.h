template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::CheckUpdate(double & alpha, Residual & res, Solution & delta, Solution & sol) {
 
  Solution sol_try;
  double res_w = 0.;
  	 
  while (alpha >= 0.01) {

    sol_try = sol;

    // Compute the new coefficient vector of lambda
    int size = COMP * ndof_l_total;
    int ione = 1;
    daxpy_(&size, &alpha, &delta.vecL[0], &ione, &sol_try.vecL[0], &ione);
    
    // Compute the new coefficient vector of w and q
    int size_w = COMP * ndof_w_total;
    int size_q = COMP * ndof_q_total;
    daxpy_(&size_w, &alpha, &delta.vecW[0], &ione, &sol_try.vecW[0], &ione);
    if (Model::Diffusion || Model::Source)
      daxpy_(&size_q, &alpha, &delta.vecQ[0], &ione, &sol_try.vecQ[0], &ione);
    
    ReconstructSolution(sol_try);

    // Check physical constraints (density, pressure)
    bool physical = true;
    for (int i = 0; i < ne; ++i) {
      ElementData<D, COMP> & ed = *eldata[i];

      for (int j = 0; j < ed.nip; ++j) {
        const double * pw = &ed.w[COMP * j];
        const double * pq = &ed.qp[D * j];

        if (pw[0] < 0.) {
          physical = false;
          alpha *= 0.75;
          break;
        }

        Vec<COMP> state;
        for (int ll = 0; ll < COMP; ++ll)
          state(ll) = pw[ll];

        SpatialParams<D> sparam;
        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = pq[dd];
      
        double pressure = Model::EvalConstrPrimVar(0, state, sparam);

        if (pressure < 0.) {
          physical = false;
          alpha *= 0.75;
          break;
        }

      }
      
      if (!physical) break;
    }

    if (!physical) continue;

    for (int i = 0; i < nf; ++i) {

      if (!fadata[i]) continue;
      
      FacetData<D, COMP> & fd = *fadata[i];

      if (fd.bcnr == -2) continue; // no zero-measure faces
      
      for (int j = 0; j < fd.nip; ++j) {
        const double * pw1 = &fd.w1[COMP * j];
        const double * pq  = &fd.qp[D * j];

        if (pw1[0] < 0.) {
          physical = false;
          alpha *= 0.75;
          break;
        }

        Vec<COMP> state;
        for (int ll = 0; ll < COMP; ++ll)
          state(ll) = pw1[ll];

        SpatialParams<D> sparam;
        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = pq[dd];
      
        double pressure = Model::EvalConstrPrimVar(0, state, sparam);

        if (pressure < 0.) {
          physical = false;
          alpha *= 0.75;
          break;
        }

        if (fd.ndof_l != 0) {
          const double * pw2 = &fd.w2[COMP * j];
          const double * pl  = &fd.lambda[COMP * j];
          
          {
            if (pw2[0] < 0.) {
              physical = false;
              alpha *= 0.75;
              break;
            }

            for (int ll = 0; ll < COMP; ++ll)
              state(ll) = pw2[ll];

            pressure = Model::EvalConstrPrimVar(0, state, sparam);
            
            if (pressure < 0.) {
              physical = false;
              alpha *= 0.75;
              break;
            }

          }

          {
            if (pl[0] < 0.) {
              physical = false;
              alpha *= 0.75;
              break;
            }

            for (int ll = 0; ll < COMP; ++ll)
              state(ll) = pl[ll];

            pressure = Model::EvalConstrPrimVar(0, state, sparam);
            
            if (pressure < 0.) {
              physical = false;
              alpha *= 0.75;
              break;
            }
          }
        }
      }
      
      if (!physical) break;
    }

    if (!physical) continue;
    // If we do not use the line search, we can stop here
    if (!SolverData::line_search) break;

#ifdef BR2
    vector<double> dummy(0);
#endif
    
    res_w = 0.;
    for (int i = 0; i < ne; ++i) {
      ElementData<D, COMP> & ed = *eldata[i];

      int ndof_q  = ed.ndof_q;
      int ndof_w  = ed.ndof_w;
      int nip     = ed.nip;

      fill(&matQ[0], &matQ[0] + COMP * ndof_q, 0.0);
      fill(&matW[0], &matW[0] + COMP * ndof_w, 0.0);

#ifdef BR2
      AssembleBR2(i, false, 0);
#endif
      AssembleLocalResidual(i, matQ, matW);

      // Augment residual with the unsteady term
      double ev = 0.;

      // Integrate the maximum eigenvalue along the element boundary
      for (int ff = 0; ff < ed.nf; ++ff) {
        FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];
      
        if (fd.bcnr == -2) continue; // zero-measure face
      
        // On which side of the facet are we?
        bool side = ed.offset_w == fd.offset_w1;
      
        int nip    = fd.nip;

        for (int j = 0; j < nip; ++j) {

          const double intweights = fd.qw[j];

          const double * pw   = side ? &fd.w1[j * COMP] : &fd.w2[j * COMP];
          const double * pn   = side ? &fd.n1[j * D] : &fd.n2[j * D];
          const double * ppos = &fd.qp[j * D];

          Vec<COMP> state(0.);
          SpatialParams<D> sparam(ppos, pn);
          
          for (int ll = 0; ll < COMP; ++ll)
            state(ll) = pw[ll];

          double lambda_c = 0.;
          Model::EvalMaxEigenvalue(state, sparam, lambda_c);

          ev += intweights * lambda_c;
        }
      }
    
      double dt = pcfl * ed.vol / ev;      

      mygemm('n', 'n', ndof_w, COMP, ndof_w, alpha/dt, &ed.mass[0], &delta.vecW[COMP*ed.offset_w], -1., &matW[0]);
      
      for (int ll = 0, index = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_w; ++pp, ++index)
          res_w += pow(residual_scale[ll] * matW[index], 2.);
    } 

    res_w = sqrt(res_w);

    if (res_w > (1. + SolverData::res_inc) * res.resW)
      alpha *= 0.75;
    else 
      break;
  }
}
