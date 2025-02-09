#include "../../helper_blas.h"

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::AssembleBR2(const int i, bool derivative, int order_dec) {

  if (!Model::Diffusion) return;
  
  ElementData<D, COMP> & ed = *eldata[i];

  int ndof_w  = ed.ndof_w;
  int offset = 0;

  vector<int> conv_loc_w;
  
  if (order_dec > 0) {
    conv_loc_w.assign(ndof_w, 0);
    transfer_ndof(ed.order, ed.order - order_dec, ed.type, &conv_loc_w[0]);
  }
  
  for (int ff = 0; ff < ed.nf; ++ff) {
    FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];
    
    int nip    = fd.nip;
    int ndof_l = fd.ndof_l;
    int bcnr   = fd.bcnr;

    if (bcnr == -2) continue; // skip zero-measure faces

    bool side = ed.offset_w == fd.offset_w1;
    
    int width = COMP * D * (1 + COMP * (ndof_w + ndof_l));

    // Assemble the right hand side for the lifting operator
    for (int j = 0; j < nip; ++j) {
      const double qw = fd.qw[j];      
      const double * phi  = side ? &fd.phi1[j * ndof_w] : &fd.phi2[j * ndof_w];
      const double * pq   = side ? &fd.q1[j * D * COMP] : &fd.q2[j * D * COMP];
      const double * pw   = side ? &fd.w1[j * COMP] : &fd.w2[j * COMP];
      const double * mu   = &fd.mu[j * ndof_l];
      const double * pl   = &fd.lambda[j * COMP];
      const double * pn   = side ? &fd.n1[j * D] : &fd.n2[j * D];
      const double * ppos = &fd.qp[j * D];

      Vec<COMP, AutoDiff<COMP*2> >    ad_state(0.), ad_lambda(0.);
      Mat<COMP, D, AutoDiff<COMP*2> > ad_grad(0.),  ad_jump(0.), ad_flux(0.);

      SpatialParams<D> sparam(ppos, pn);
  
      if (derivative) {
        for (int ll = 0; ll < COMP; ++ll)
          ad_state(ll) = AutoDiff<COMP*2> (pw[ll], ll+COMP);

        if (ndof_l != 0)
          for (int ll = 0; ll < COMP; ++ll)
            ad_lambda(ll) = AutoDiff<COMP*2> (pl[ll], ll);
      } else {
        for (int ll = 0; ll < COMP; ++ll)
          ad_state(ll) = pw[ll];

        if (ndof_l != 0)
          for (int ll = 0; ll < COMP; ++ll)
            ad_lambda(ll) = pl[ll];
      }

      // On boundary faces, lambda contains the boundary state (evaluated with w)
      if (ndof_l == 0) {
        min_y = ed.vol / fd.h;
        Model::EvalBdryState(bcnr, ad_state, sparam, ad_lambda);
      }
      
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          ad_grad(ll, dd) = pq[dd+D*ll];

      ad_jump = stab_visc * ed.nf * (ad_lambda - ad_state) * Trans(sparam.normal);

      if (ndof_l == 0 && bcnr == 1 && COMP > 4) {
        Vec<COMP, AutoDiff<COMP*2> > ghoststate(2. * ad_lambda - ad_state);
        Mat<COMP, D, AutoDiff<COMP*2> > ad_flux1(0.), ad_flux2(0.);
        Model::ApplyK(ad_state, ad_grad, ad_jump, sparam, ad_flux1);
        Model::ApplyK(ghoststate, ad_grad, ad_jump, sparam, ad_flux2);
        ad_flux = 0.5 * (ad_flux1 + ad_flux2);
      } else {
        Model::ApplyK(ad_lambda, ad_grad, ad_jump, sparam, ad_flux);
      }

      // The following is a hack for adiabatic walls (has to change in the future);
      if (bcnr == 1) {
        if (COMP == D+4)
          ad_flux.Row(D+1) = ad_flux.Row(D+2);
        else if (COMP >= 4)
          ad_flux.Row(D+1) = 0.;
      }

      // Lifting operator
      double * tmp = &temp_lift[offset + j * width];
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          tmp[dd+D*ll] = qw * ad_flux(ll, dd).Value();

      if (!derivative) continue;
      
      // Derivative wrt lambda (this gets skipped for ndof_l=0)
      tmp = &temp_lift[offset + j * width + COMP * D];
      for (int mm = 0; mm < COMP; ++mm)
        for (int qq = 0; qq < ndof_l; ++qq)
          for (int ll = 0; ll < COMP; ++ll)
            for (int dd = 0; dd < D; ++dd)
              tmp[dd+D*(ll+COMP*(qq+ndof_l*mm))] = qw * ad_flux(ll, dd).DValue(mm) * mu[qq];

      // Derivative wrt state
      tmp = &temp_lift[offset + j * width + COMP * D * (1 + COMP * ndof_l)];
      for (int mm = 0; mm < COMP; ++mm)
        for (int qq = 0; qq < ndof_w; ++qq)
          for (int ll = 0; ll < COMP; ++ll)
            for (int dd = 0; dd < D; ++dd)
              tmp[dd+D*(ll+COMP*(qq+ndof_w*mm))] = qw * ad_flux(ll, dd).DValue(COMP+mm) * phi[qq];
    }

    double * phi = side ? &fd.phi1[0] : &fd.phi2[0];    
    // Multiply with test functions
    mygemm('n', 't', ndof_w, width, nip, 1., phi, &temp_lift[offset], 0., &temp_lift2[offset]);    

    // Compute coefficients for lifting operator and its derivatives
    mygemm('n', 'n', ndof_w, width, ndof_w, 1., &ed.inv_mass[0], &temp_lift2[offset], 0., &temp_lift[offset]);

    // Blend out (p+1) modes
    if (assemblyMode == AM_ErrorEstimation && order_dec > 0) {
      // inefficient!!!!

      // Make a copy
      copy(&temp_lift[offset], &temp_lift[offset] + width * ndof_w, &temp_lift2[offset]);

      // Zero out the original
      fill(&temp_lift[offset], &temp_lift[offset] + width * ndof_w, 0.0);
      
      // And then copy only the decreased order parts back
      int ndof_w_dec = get_ndof(ed.order - order_dec, ed.type);
      for (int qq = 0; qq < width; ++qq)
        for (int pp = 0; pp < ndof_w_dec; ++pp) {
          int new_pp = conv_loc_w[pp];
          temp_lift[offset+new_pp+ndof_w*qq] = temp_lift2[offset+new_pp+ndof_w*qq];
        }
    }    

    // Reconstruct the lifting operator and its derivatives
    mygemm('t', 'n', width, nip, ndof_w, 1., &temp_lift[offset], phi, 0., &temp_lift2[offset]);

    offset += max(nip, ndof_w) * width;
  }
}
