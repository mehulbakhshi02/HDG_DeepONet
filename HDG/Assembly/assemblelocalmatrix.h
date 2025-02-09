#include "../../helper_blas.h"

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::AssembleLocalMatrix(const int i, vector<double> & matB, vector<double> & matC, vector<double> & matD, 
		                vector<double> & matR, vector<double> & matS,
		                vector<double> & matL, vector<double> & matM) {

  bool assembly = assemblyMode == AM_AssemblyPrimal || assemblyMode == AM_AssemblyAdjoint;
  bool adjoint  = assemblyMode == AM_AssemblyAdjoint || assemblyMode == AM_BackSolveAdjoint;
  
  ElementData<D, COMP> & ed = *eldata[i];

  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;
  int nip     = ed.nip;
  
  // Volume contributions

  // Pseudo-time stepping (for stabilization)
  if (!testing) {

    double ev = 0.;

    // Integrate the maximum eigenvalue along the element boundary
    for (int ff = 0; ff < ed.nf; ++ff) {
      FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];
      
      if (fd.bcnr == -2) continue; // zero-measure face
      
      // On which side of the facet are we?
      bool side = ed.offset_w == fd.offset_w1;
      
      int nip = fd.nip;

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
      
    for (int pp = 0; pp < ndof_w; ++pp) {
      for (int qq = 0; qq < ndof_w; ++qq) {
        double mass = ed.mass[pp*ndof_w+qq];
        for (int ll = 0; ll < COMP; ++ll)
          matD[qq+ndof_w*(ll+COMP*(pp+ndof_w*ll))] = 1./dt * mass;
      }
    }
  }

  // Check whether the gradient plays a role
  const int dsize = (Model::Diffusion || Model::Source) ? D+1 : 1;

  // Shock-Capturing
  double p = max(1, ed.order);
  double hh = ed.h / p; //ed.nf * ed.vol / (ed.surf * p); //ed.h / p;
  double eps0 = eps_art_visc * pow(hh, 2. - 0.0) ;
  double eps_mean = 0.0;

  if (shock_capturing > 0 && ed.order > 0) {

    deps.assign(COMP * ndof_w * dsize, 0.);
    
    Vec<COMP, AutoDiff<COMP*(D+1)> > ad_state;
    Mat<COMP, D, AutoDiff<COMP*(D+1)> > ad_grad;
    AutoDiff<COMP*(D+1)> ad_eps;

    for (int j = 0; j < nip; ++j) {

      const double qw     = ed.qw[j];
      const double * phi  = &ed.phi[j * ndof_w];
      const double * dphi = &ed.dphi[j * D * ndof_w];
      const double * pq   = &ed.q[j * COMP * D];
      const double * pw   = &ed.w[j * COMP];

      for (int ll = 0; ll < COMP; ++ll)
        ad_state(ll) = AutoDiff<COMP*(D+1)> (pw[ll], ll);

      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          ad_grad(ll, dd) = AutoDiff<COMP*(D+1)> (pq[dd+D*ll], ll+COMP*(dd+1));

      ShockResidual<D, COMP>::evaluate(ad_state, ad_grad, ad_eps);

      eps_mean += qw * ad_eps.Value();

      for (int ll = 0; ll < COMP; ++ll) {
        for (int pp = 0; pp < ndof_w; ++pp) {
          double val = ad_eps.DValue(ll) * phi[pp];
          if (Model::Diffusion || Model::Source)
            for (int dd = 0; dd < D; ++dd)
              deps[pp+ndof_w*(ll+COMP*(dd+1))] += qw * ad_eps.DValue(ll+COMP*(dd+1)) * phi[pp];
          else
            for (int dd = 0; dd < D; ++dd)
              val += ad_eps.DValue(ll+COMP*(dd+1)) * dphi[pp+ndof_w*dd];
	  
          deps[pp+ndof_w*ll] += qw * val;
        }
      }
    }

    eps_mean *= eps0 / ed.vol;
    for (int dd = 0; dd < dsize; ++dd)
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_w; ++pp)
          deps[pp+ndof_w*(ll+COMP*dd)] *= eps0 / ed.vol;
  }
  // End Shock-Capturing
      
  if (Model::Diffusion || Model::Source)
    fill(&temp3[0], &temp3[0] + nip * COMP * ndof_w * COMP, 0.);
    
  for (int j = 0; j < nip; ++j) {

    Vec<COMP, AutoDiff<COMP*dsize> > ad_state(0.), ad_sf(0.);
    Mat<COMP, D, AutoDiff<COMP*dsize> > ad_grad(0.), ad_fc(0.), ad_fv(0.), ad_f(0.);
    
    const double intweights = ed.qw[j];
    
    const double * phi  = &ed.phi[j * ndof_w];
    const double * dphi = &ed.dphi[j * ndof_w * D];
    const double * pq   = &ed.q[j * COMP * D];
    const double * pw   = &ed.w[j * COMP];
    const double * ppos = &ed.qp[j * D];
    
    // Fill helper variables
    for (int ll = 0; ll < COMP; ++ll)
      ad_state(ll) = AutoDiff<COMP*dsize> (pw[ll], ll);
    
    if (Model::Diffusion || Model::Source)
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          ad_grad(ll, dd) = AutoDiff<COMP*dsize> (pq[dd+D*ll], ll+COMP*(dd+1));
    else if (shock_capturing > 0)
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          ad_grad(ll, dd) = pq[dd+D*ll];

    SpatialParams<D> sparam(ppos, ed.wd[j]);
    
    if (Model::Source)
      Model::EvalSource(ad_state, ad_grad, sparam, ad_sf);

    if (ed.order > 0) { 
      // Evaluate fluxes
      if (Model::Convection)
        Model::EvalConvFlux(ad_state, sparam, ad_fc);
      if (Model::Diffusion)
        Model::EvalDiffFlux(ad_state, ad_grad, sparam, ad_fv);

      ad_f = ad_fc - ad_fv;
    
      // df/dw * phi
      double * tmp = &temp[j * COMP * ndof_w * COMP * D];
      for (int dd = 0; dd < D; ++dd)
        for (int mm = 0; mm < COMP; ++mm)
          for (int qq = 0; qq < ndof_w; ++qq)
            for (int ll = 0; ll < COMP; ++ll)
              tmp[ll+COMP*(qq+ndof_w*(mm+COMP*dd))] = intweights * ad_f(ll, dd).DValue(mm) * phi[qq];
    
      if (Model::Diffusion) {
        double * tmp2 = &temp2[j * COMP * ndof_w * D * COMP * D];
        for (int dd = 0; dd < D; ++dd)
          for (int mm = 0; mm < COMP; ++mm)
            for (int dd2 = 0; dd2 < D; ++dd2)
              for (int qq = 0; qq < ndof_w; ++qq)
                for (int ll = 0; ll < COMP; ++ll)
                  tmp2[ll+COMP*(qq+ndof_w*(dd2+D*(mm+COMP*dd)))] = intweights * ad_f(ll, dd).DValue(mm+COMP*(dd2+1)) * phi[qq];
      }

      if (shock_capturing > 0) {
        for (int dd = 0; dd < D; ++dd)
          for (int mm = 0; mm < COMP; ++mm)
            for (int qq = 0; qq < ndof_w; ++qq)
              for (int ll = 0; ll < COMP; ++ll)
                tmp[ll+COMP*(qq+ndof_w*(mm+COMP*dd))] -= intweights * deps[qq+ndof_w*mm] * ad_grad(ll, dd).Value();

        if (Model::Diffusion) {
          double * tmp2 = &temp2[j * COMP * ndof_w * D * COMP * D];          
          for (int dd = 0; dd < D; ++dd)
            for (int mm = 0; mm < COMP; ++mm)
              for (int qq = 0; qq < ndof_w; ++qq)
                tmp2[mm+COMP*(qq+ndof_w*(dd+D*(mm+COMP*dd)))] -= intweights * eps_mean * phi[qq];

          for (int dd = 0; dd < D; ++dd)
            for (int mm = 0; mm < D+2; ++mm)
              for (int dd2 = 0; dd2 < D; ++dd2)
                for (int qq = 0; qq < ndof_w; ++qq)
                  for (int ll = 0; ll < D+2; ++ll)
                    tmp2[ll+COMP*(qq+ndof_w*(dd2+D*(mm+COMP*dd)))] -= intweights * deps[qq+ndof_w*(mm+COMP*(dd2+1))] * ad_grad(ll, dd).Value();
          
        } else {
          for (int dd = 0; dd < D; ++dd)
            for (int mm = 0; mm < COMP; ++mm)
              for (int qq = 0; qq < ndof_w; ++qq)
                tmp[mm+COMP*(qq+ndof_w*(mm+COMP*dd))] -= intweights * eps_mean * dphi[qq+ndof_w*dd];          
        }
      }
      
      if (Model::Diffusion || Model::Source) {
        // int_{K} phi div(tau)
        double * tmp3 = &temp3[j * COMP * ndof_w * COMP];
        for (int qq = 0; qq < ndof_w; ++qq)
          for (int ll = 0; ll < COMP; ++ll)
            tmp3[ll+COMP*(qq+ndof_w*ll)] = intweights * phi[qq];
      }
    }

    if (Model::Source) {
      double * tmp4 = &temp4[j * COMP * ndof_w * COMP];
      for (int mm = 0; mm < COMP; ++mm)
        for (int qq = 0; qq < ndof_w; ++qq)
          for (int ll = 0; ll < COMP; ++ll)
            tmp4[ll+COMP*(qq+ndof_w*mm)] = intweights * ad_sf(ll).DValue(mm) * phi[qq];      

      double * tmp5 = &temp5[j * COMP * ndof_w * D * COMP];
      for (int mm = 0; mm < COMP; ++mm)
        for (int dd = 0; dd < D; ++dd)
          for (int qq = 0; qq < ndof_w; ++qq)
            for (int ll = 0; ll < COMP; ++ll)
              tmp5[ll+COMP*(qq+ndof_w*(dd+D*mm))] = intweights * ad_sf(ll).DValue(mm+COMP*(dd+1)) * phi[qq];      
    }

  }

  if (Model::Source) {
    // int_{K} phi * ds/dw * phi
    mygemm('n', 't', ndof_w, COMP * ndof_w * COMP, nip, -1., &ed.phi[0], &temp4[0], 1., &matD[0]);
    // int_{K} phi * ds/dq * tau
    mygemm('n', 't', ndof_w, COMP * ndof_w * D * COMP, nip, -1., &ed.phi[0], &temp5[0], 1., &matC[0]);
  }
  
  if (ed.order > 0) { 
    // int_{K} dphi * df/dw * phi
    mygemm('n', 't', ndof_w, COMP * ndof_w * COMP, nip * D, -1., &ed.dphi[0], &temp[0], 1., &matD[0]);
    
    // int_{K} dphi * df/dq * tau
    if (Model::Diffusion)
      mygemm('n', 't', ndof_w, COMP * ndof_w * D * COMP, nip * D, -1., &ed.dphi[0], &temp2[0], 1., &matC[0]);
  
    // int_{K} phi div(tau)    
    if (Model::Diffusion || Model::Source)
      mygemm('n', 't', ndof_w * D, COMP * ndof_w * COMP, nip, 1., &ed.dphi[0], &temp3[0], 1., &matB[0]);
  }

  // Facet contributions
  int offset_lq  = 0;
  int offset_lw  = 0;
  int offset_br2 = 0;

  for (int ff = 0; ff < ed.nf; ++ff) {
    FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];

    if (fd.bcnr == -2) continue; // zero-measure face
    
    // On which side of the facet are we?
    bool side = ed.offset_w == fd.offset_w1;
    
    int ndof_l = fd.ndof_l;
    int nip    = fd.nip;

    int width_br2 = COMP * D * (1 + COMP * (ndof_w + ndof_l));
    
    const int dsize = (Model::Diffusion) ? D+2 : 2;

    if (assembly)
      fill(&insert[0], &insert[0] + COMP * ndof_l * COMP * ndof_l, 0.);
    
    if (Model::Diffusion)
      fill(&temp4[0], &temp4[0] + COMP * ndof_w * D * COMP * nip, 0.);

    double * tmpR = &matR[COMP*ndof_q+offset_lq];
    double * tmpS = &matS[COMP*ndof_w+offset_lw];
    double * tmpL = &matL[offset_lq];
    double * tmpM = &matM[offset_lw];

    for (int j = 0; j < nip; ++j) {

      Vec<COMP, AutoDiff<COMP*dsize> > ad_lambda(0.), ad_state(0.);
      Vec<COMP, AutoDiff<COMP*dsize> > ad_fcn(0.), ad_fvn(0.), ad_fn(0.);
      Mat<COMP, D, AutoDiff<COMP*dsize> > ad_grad(0.);

      const double intweights = fd.qw[j];

      const double * phi  = side ? &fd.phi1[j * ndof_w] : &fd.phi2[j * ndof_w];
      const double * pq   = side ? &fd.q1[j * D * COMP] : &fd.q2[j * D * COMP];
      const double * pw   = side ? &fd.w1[j * COMP] : &fd.w2[j * COMP];
      const double * mu   = &fd.mu[j * ndof_l];
      const double * pl   = &fd.lambda[j * COMP];
      const double * pn   = side ? &fd.n1[j * D] : &fd.n2[j * D];
      const double * ppos = &fd.qp[j * D];

      SpatialParams<D> sparam(ppos, pn);
      
      for (int ll = 0; ll < COMP; ++ll)
        ad_lambda(ll) = AutoDiff<COMP*dsize> (pl[ll], ll);
      for (int ll = 0; ll < COMP; ++ll)
        ad_state(ll) = AutoDiff<COMP*dsize> (pw[ll], ll+COMP);
      if (Model::Diffusion)
        for (int ll = 0; ll < COMP; ++ll)
          for (int dd = 0; dd < D; ++dd)
            ad_grad(ll, dd) = AutoDiff<COMP*dsize> (pq[dd+D*ll], ll+COMP*(dd+2));
      
      if (ndof_l == 0) {

        min_y = ed.vol / fd.h;

        if (Model::Convection)
          Model::EvalBdryConvFlux(fd.bcnr, ad_state, sparam, ad_fcn);
        if (Model::Diffusion)
          Model::EvalBdryDiffFlux(fd.bcnr, ad_state, ad_grad, sparam, ad_fvn);
      } else {
        if (Model::Convection)
          NumConvFlux(ad_lambda, ad_state, sparam, ad_fcn);
        if (Model::Diffusion)
          NumViscFlux(ad_lambda, ad_grad, ad_state, ad_grad, sparam, ad_fvn);
      }
      
      ad_fn = ad_fcn - ad_fvn;

      // int_{dK} mu * dfn/dl * mu (row-oriented)
      if (assembly)
        for (int ll = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < ndof_l; ++pp)
            for (int mm = 0; mm < COMP; ++mm)
              for (int qq = 0; qq < ndof_l; ++qq)
                insert[qq+ndof_l*(mm+COMP*(pp+ndof_l*ll))] += intweights * mu[pp] * ad_fn(ll).DValue(mm) * mu[qq];
      
      // dfn/dw * phi
      double * tmp = &temp[j * COMP * ndof_w * COMP];
      for (int mm = 0; mm < COMP; ++mm)
        for (int qq = 0; qq < ndof_w; ++qq)
          for (int ll = 0; ll < COMP; ++ll)
            tmp[ll+COMP*(qq+ndof_w*mm)] = intweights * ad_fn(ll).DValue(mm+COMP) * phi[qq];

      // dfn/dl * mu
      double * tmp2 = &temp2[j * COMP * ndof_l * COMP];
      for (int mm = 0; mm < COMP; ++mm)
        for (int qq = 0; qq < ndof_l; ++qq)
          for (int ll = 0; ll < COMP; ++ll)
            tmp2[ll+COMP*(qq+ndof_l*mm)] = intweights * ad_fn(ll).DValue(mm) * mu[qq];

      // dfn/dq * tau
      if (Model::Diffusion) {
        double * tmp3 = &temp3[j * COMP * ndof_w * D * COMP];
        for (int mm = 0; mm < COMP; ++mm)
          for (int dd = 0; dd < D; ++dd)
            for (int qq = 0; qq < ndof_w; ++qq)
              for (int ll = 0; ll < COMP; ++ll)
                tmp3[ll+COMP*(qq+ndof_w*(dd+D*mm))] = intweights * ad_fn(ll).DValue(mm+COMP*(dd+2)) * phi[qq];
      }

#ifdef BR2
      if (Model::Diffusion) {
        double * tmp_lift = &temp_lift2[offset_br2 + j * width_br2 + COMP * D];

        // int_{dK} mu * (dr/dl * n) * mu (row-oriented)        
        if (assembly)
          for (int ll = 0; ll < COMP; ++ll)
            for (int pp = 0; pp < ndof_l; ++pp)
              for (int mm = 0; mm < COMP; ++mm)
                for (int qq = 0; qq < ndof_l; ++qq) {
                  double val = 0.;
                  for (int dd = 0; dd < D; ++dd)
                    val += tmp_lift[dd+D*(ll+COMP*(qq+ndof_l*mm))] * sparam.normal(dd);
                  insert[qq+ndof_l*(mm+COMP*(pp+ndof_l*ll))] -= intweights * mu[pp] * val;
                }

        // (dr/dl * n) * mu
        double * tmp2 = &temp2[j * COMP * ndof_l * COMP];
        for (int mm = 0; mm < COMP; ++mm)
          for (int qq = 0; qq < ndof_l; ++qq)
            for (int ll = 0; ll < COMP; ++ll) {
              double val = 0.;
              for (int dd = 0; dd < D; ++dd)
                val += tmp_lift[dd+D*(ll+COMP*(qq+ndof_l*mm))] * sparam.normal(dd);
              tmp2[ll+COMP*(qq+ndof_l*mm)] -= intweights * val;
            }        

        tmp_lift = &temp_lift2[offset_br2 + j * width_br2 + COMP * D * (1 + COMP * ndof_l)];
        
        // (dr/dw * n) * phi
        double * tmp = &temp[j * COMP * ndof_w * COMP];
        for (int mm = 0; mm < COMP; ++mm)
          for (int qq = 0; qq < ndof_w; ++qq)
            for (int ll = 0; ll < COMP; ++ll) {
              double val = 0.;
              for (int dd = 0; dd < D; ++dd)
                val += tmp_lift[dd+D*(ll+COMP*(qq+ndof_w*mm))] * sparam.normal(dd);
              tmp[ll+COMP*(qq+ndof_w*mm)] -= intweights * val;
            }
      }
#endif

      if (Model::Diffusion || Model::Source) {
        if (ndof_l == 0) {
          Vec<COMP, AutoDiff<COMP> > ad_w(0.), ad_w_bc(0.);
          for (int ll = 0; ll < COMP; ++ll)
            ad_w(ll) = AutoDiff<COMP> (pw[ll], ll);
          Model::EvalBdryState(fd.bcnr, ad_w, sparam, ad_w_bc);

          double * tmp4 = &temp4[j * COMP * ndof_w * D * COMP];
          for (int mm = 0; mm < COMP; ++mm)
            for (int qq = 0; qq < ndof_w; ++qq)
              for (int ll = 0; ll < COMP; ++ll)
                for (int dd = 0; dd < D; ++dd)
                  tmp4[dd+D*(ll+COMP*(qq+ndof_w*mm))] = intweights * sparam.normal(dd) * ad_w_bc(ll).DValue(mm) * phi[qq];
        } else {
          double * tmp4 = &temp4[j * COMP * ndof_l * D * COMP];
          for (int ll = 0; ll < COMP; ++ll)
            for (int qq = 0; qq < ndof_l; ++qq)
              for (int dd = 0; dd < D; ++dd)
                tmp4[dd+D*(ll+COMP*(qq+ndof_l*ll))] = intweights * sparam.normal(dd) * mu[qq];
        }
      }
    }

    // int_{dK} phi * dfn/dw * phi
    double * phi = side ? &fd.phi1[0] : &fd.phi2[0];
    mygemm('n', 't', ndof_w, COMP * ndof_w * COMP, nip, 1., phi, &temp[0], 1., &matD[0]);

    // int_{dK} phi * dfn/dq * tau
    if (Model::Diffusion)
      mygemm('n', 't', ndof_w, COMP * ndof_w * D * COMP, nip, 1., phi, &temp3[0], 1., &matC[0]);

    if (ndof_l != 0) {
      // int_{dK} tau * n * mu
      if (Model::Diffusion || Model::Source)
        mygemm('n', 't', ndof_w, COMP * ndof_l * D * COMP, nip, 1., phi, &temp4[0], 1., &tmpR[0]);
      
      // int_{dK} phi * dfn/dl * mu
      mygemm('n', 't', ndof_w, COMP * ndof_l * COMP, nip, -1., phi, &temp2[0], 1., &tmpS[0]);
    
      if (assembly || assemblyMode == AM_BackSolveAdjoint) {
        // int_{dK} mu * dfn/dq * tau
        if (Model::Diffusion)
          mygemm('n', 't', ndof_l, COMP * ndof_w * D * COMP, nip, 1., &fd.mu[0], &temp3[0], 1., &tmpL[0]);

        // int_{dK} mu * dfn/dw * phi
        mygemm('n', 't', ndof_l, COMP * ndof_w * COMP, nip, 1., &fd.mu[0], &temp[0], 1., &tmpM[0]);
      }
      
      // int_{dK} mu * dfn/dl * mu
      if (assembly) {

        // Apply scaling
        int nn;
        for (int ll = 0, index = 0; ll < COMP; ++ll) {
          if (assemblyMode == AM_AssemblyPrimal) nn = ll;
          for (int pp = 0; pp < ndof_l; ++pp)
            for (int mm = 0; mm < COMP; ++mm) {
              if (assemblyMode == AM_AssemblyAdjoint) nn = mm;
              for (int qq = 0; qq < ndof_l; ++qq, ++index)
                insert[index] *= residual_scale[nn];
            }
        }
        
        petsc_add_submatrix(COMP * ndof_l, COMP * ndof_l, COMP * fd.offset_l, COMP * fd.offset_l, &insert[0]);
      }
      
      offset_lw += COMP * ndof_l * COMP * ndof_w;
      offset_lq += COMP * ndof_l * COMP * ndof_q;
    } else {
      // int_{dK} tau * dwo/dw * n * phi
      if (Model::Diffusion || Model::Source)
        mygemm('n', 't', ndof_w, COMP * ndof_w * D * COMP, nip, -1., phi, &temp4[0], 1., &matB[0]);      
    }

    offset_br2 += max(ndof_w, nip) * width_br2;
  }
}

