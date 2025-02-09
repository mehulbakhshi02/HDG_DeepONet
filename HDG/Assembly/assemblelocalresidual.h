#include "../../helper_blas.h"

template <int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
  ::AssembleLocalResidual(const int i, vector<double> & vecF, vector<double> & vecG) {

  ElementData<D, COMP> & ed = *eldata[i];
  
  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;
  int nip     = ed.nip;
  
  double eps_mean = 0.0;

  if (ed.order > 0 && shock_capturing > 0) {

    // Shock-Capturing
    double p = ed.order;
    double hh = ed.h / p; //ed.nf * ed.vol / (ed.surf * p); //ed.h / p;    
    double eps0 = eps_art_visc * pow(hh, 2. - 0.0);

    Vec<COMP> state;
    Mat<COMP, D> grad;

    for (int j = 0; j < ed.nip; ++j) {
      const double qw = ed.qw[j];

      // Get pointers to arrays containing shape functions etc
      const double * pw = &ed.w[COMP*j];           // [COMP]
      const double * pq = &ed.q[COMP*D*j];         // [COMP,D]

      for (int ll = 0; ll < COMP; ++ll)
        state(ll) = pw[ll];
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          grad(ll, dd) = pq[dd+D*ll];

      double eps_av;
      ShockResidual<D, COMP>::evaluate(state, grad, eps_av);
      eps_mean += qw * eps_av;
    }
    eps_mean *= eps0 / ed.vol;


    FlatVector<Vec<1>> eps = gf_const->GetVector().FV<Vec<1>>();   
    eps(i) = log10(eps_mean);
    if (isnan(log10(eps_mean)))
      eps(i) = -30.;
  }
  
  // Volume contributions
  for (int j = 0; j < nip; ++j) {
    Vec<COMP> w(0.), sf(0.);
    Mat<COMP, D> q(0.), fc(0.), fv(0.), f(0.);

    const double intweights = ed.qw[j];
    
    const double * pq   = &ed.q[j * D * COMP];
    const double * pw   = &ed.w[j * COMP];
    const double * ppos = &ed.qp[j * D];
    
    // Fill helper variables
    for (int ll = 0; ll < COMP; ++ll)
      w(ll) = pw[ll];
    
    for (int ll = 0; ll < COMP; ++ll)
      for (int dd = 0; dd < D; ++dd)
        q(ll, dd) = pq[dd+D*ll];

    SpatialParams<D> sparam(ppos, ed.wd[j]);

    // Computing volume functionals
    if (compute_vol_monitors) {
      Vec<Model::NumVolFunctionals> monitors(0.);
      Model::EvalVolFunctionals(w, sparam, monitors);
      for (int ll = 0; ll < Model::NumVolFunctionals; ++ll)
        vol_monitors[ll] += intweights * monitors(ll);    
    }
    
    if (Model::Source)
      Model::EvalSource(w, q, sparam, sf);

    if (ed.order > 0) {
      // Evaluate fluxes
      if (Model::Convection)
        Model::EvalConvFlux(w, sparam, fc);
      if (Model::Diffusion)
        Model::EvalDiffFlux(w, q, sparam, fv);

      f = fc - fv;

      if (shock_capturing > 0)
        f -= eps_mean * q;
      
      double * tmp = &temp[j * COMP * D];
      for (int dd = 0; dd < D; ++dd)
        for (int ll = 0; ll < COMP; ++ll)
          tmp[ll+COMP*dd] = intweights * f(ll, dd);

      if (Model::Diffusion || Model::Source) {
        double * tmp2 = &temp2[j * COMP];
        for (int ll = 0; ll < COMP; ++ll)
          tmp2[ll] = intweights * w(ll);
      }
    }
    
    if (Model::Diffusion || Model::Source) {
      double * tmp3 = &temp3[j * D * COMP];
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          tmp3[dd+D*ll] = intweights * q(ll, dd);
    }

    if (Model::Source) {
      double * tmp4 = &temp4[j * COMP];
      for (int ll = 0; ll < COMP; ++ll)
        tmp4[ll] = intweights * sf(ll);
    }
  }
  
  // int_{K} dphi * f    
  if (ed.order > 0)
    mygemm('n', 't', ndof_w, COMP, nip * D, 1., &ed.dphi[0], &temp[0], 1., &vecG[0]);

  // int_{K} phi * s
  if (Model::Source)
    mygemm('n', 't', ndof_w, COMP, nip, 1., &ed.phi[0], &temp4[0], 1., &vecG[0]);

  // int_{K} w div(tau)
  if (Model::Diffusion || Model::Source)
    if (ed.order > 0)
      mygemm('n', 't', ndof_q, COMP, nip, -1., &ed.dphi[0], &temp2[0], 1., &vecF[0]);

  // int_{K} q * tau
  if (Model::Diffusion || Model::Source)
    mygemm('n', 't', ndof_w, COMP * D, nip, -1., &ed.phi[0], &temp3[0], 1., &vecF[0]);

#ifdef BR2
  int offset_br2 = 0;
#endif
  
  // Facet contributions
  for (int ff = 0; ff < ed.nf; ++ff) {
    FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];

    if (fd.bcnr == -2) continue; // zero-measure face
    
    // On which side of the facet are we?
    bool side = ed.offset_w == fd.offset_w1;
    
    int ndof_l = fd.ndof_l;
    int nip    = fd.nip;
    int width_br2 = COMP * D * (1 + COMP * (ndof_w + ndof_l));

    for (int j = 0; j < nip; ++j) {

      Vec<COMP> lambda(0.), w(0.), fcn(0.), fvn(0.), fn(0.);
      Mat<COMP, D> q(0.), fv(0.);

      const double intweights = fd.qw[j];

      const double * pq   = side ? &fd.q1[j * D * COMP] : &fd.q2[j * D * COMP];
      const double * pw   = side ? &fd.w1[j * COMP] : &fd.w2[j * COMP];
      const double * pl   = &fd.lambda[j * COMP];
      const double * pn   = side ? &fd.n1[j * D] : &fd.n2[j * D];
      const double * ppos = &fd.qp[j * D];

      SpatialParams<D> sparam(ppos, pn);      
      
      for (int ll = 0; ll < COMP; ++ll)
        lambda(ll) = pl[ll];
      for (int ll = 0; ll < COMP; ++ll)
        w(ll) = pw[ll];
      if (Model::Diffusion)
        for (int ll = 0; ll < COMP; ++ll)
          for (int dd = 0; dd < D; ++dd)
            q(ll, dd) = pq[dd+D*ll];

      if (ndof_l == 0) {
        min_y = ed.vol / fd.h;

        if (Model::Convection)
          Model::EvalBdryConvFlux(fd.bcnr, w, sparam, fcn);
        
        if (Model::Diffusion)
          Model::EvalBdryDiffFlux(fd.bcnr, w, q, sparam, fvn);

        if (Model::Diffusion || Model::Source)
          for (int ll = 0; ll < COMP; ++ll)
            lambda(ll) = fd.w2[ll+COMP*j];
        
      } else {
        
        if (Model::Convection)
          NumConvFlux(lambda, w, sparam, fcn);
        if (Model::Diffusion)
          NumViscFlux(lambda, q, w, q, sparam, fvn);
      }

#ifdef BR2
      if (Model::Diffusion) {
        // Augment viscous flux with lifted flux
        const double * tmp_lift = &temp_lift2[offset_br2 + j * width_br2];
        for (int ll = 0; ll < COMP; ++ll)
          for (int dd = 0; dd < D; ++dd)
            fvn(ll) += tmp_lift[dd+D*ll] * sparam.normal(dd);
      }
#endif

      fn = fcn - fvn;

      // Compute boundary monitors
      if (ndof_l == 0 && compute_bdry_monitors) {

        // Locals
        Vec<Model::NumBdryCoeffs> coeffs(0.);
        Model::EvalBdryCoefficients(fd.bcnr, fcn, fvn, sparam, coeffs);

        fcoeffs << fd.bcnr;
        for (int dd = 0; dd < D; ++dd)
          fcoeffs << setw(24) << sparam.pos(dd);
        for (int kk = 0; kk < Model::NumBdryCoeffs; ++kk)
          fcoeffs << setw(24) << coeffs(kk);
        fcoeffs << endl;
        
        // Integrals
        Mat<Model::NumBdryFluxWeight, COMP> weights(0.);
        Vec<Model::NumBdryFluxWeight>       monitors(0.);
        Model::EvalBdryFluxWeight(fd.bcnr, sparam, weights);
        monitors = intweights * weights * fn;
        for (int ll = 0; ll < Model::NumBdryFluxWeight; ++ll)
          bdry_monitors[ll] += monitors(ll);
      }     
      
      double * tmp = &temp[j * COMP];
      for (int ll = 0; ll < COMP; ++ll)
        tmp[ll] = intweights * fn(ll);

      if (Model::Diffusion || Model::Source) {
        double * tmp2 = &temp2[j * D * COMP];
        for (int ll = 0; ll < COMP; ++ll)
          for (int dd = 0; dd < D; ++dd)
            tmp2[dd+D*ll] = intweights * lambda(ll) * sparam.normal(dd);
      }
    }// end of loop over integration points

    // int_{dK} phi * fn
    double * phi = side ? &fd.phi1[0] : &fd.phi2[0];
    mygemm('n', 't', ndof_w, COMP, nip, -1., phi, &temp[0], 1., &vecG[0]);

    // int_{dK} l * (n * tau)
    if (Model::Diffusion || Model::Source)
      mygemm('n', 't', ndof_w, COMP * D, nip, 1., phi, &temp2[0], 1., &vecF[0]);

    if (ndof_l != 0) {

      // int_{dK} mu * fn
      if (assemblyMode == AM_AssemblyPrimal || assemblyMode == AM_ErrorEstimation)
        mygemm('n', 't', ndof_l, COMP, nip, -1., &fd.mu[0], &temp[0], 0., &insert[0]);

      // adj_rhs_l contains the pure hybrid residual
      // this is necessary for weighting      
      if (assemblyMode == AM_ErrorEstimation)
        for (int ll = 0, index = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < ndof_l; ++pp, ++index)
            adj_rhs_l[index+COMP*fd.offset_l] += insert[index];
      
      if (assemblyMode == AM_AssemblyPrimal) {
        // Apply scaling
        for (int ll = 0, index = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < ndof_l; ++pp, ++index)
            insert[index] *= residual_scale[ll];
        
        petsc_add_vector(COMP * ndof_l, COMP * fd.offset_l, &insert[0]);
      }
    }

#ifdef BR2
    offset_br2 += max(nip, ndof_w) * width_br2; 
#endif
  }// end of loop over faces
}
