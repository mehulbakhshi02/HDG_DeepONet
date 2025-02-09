#include "../../helper_blas.h"
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::AssembleLocalAdjointRhs(const int i, vector<double> & vecF, vector<double> & vecG) {

  const int dsize = (Model::Diffusion) ? D+1 : 1;

  ElementData<D, COMP> & ed = *eldata[i];
  
  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;

  fill(&vecF[0], &vecF[0] + COMP * ndof_q, 0.);
  fill(&vecG[0], &vecG[0] + COMP * ndof_w, 0.);
  
  if (Model::NumVolFunctionals > 0) {

    for (int j = 0; j < ed.nip; ++j) {

      Vec<COMP, AutoDiff<COMP*dsize> > ad_state(0.);
      Vec<Model::NumVolFunctionals, AutoDiff<COMP*dsize> > ad_funct(0.);

      const double intweights = ed.qw[j];

      const double * pw   = &ed.w[j * COMP];
      const double * ppos = &ed.qp[j * D];

      SpatialParams<D> sparam(ppos);        
      
      for (int ll = 0; ll < COMP; ++ll)
        ad_state(ll) = AutoDiff<COMP*dsize> (pw[ll], ll);      

      Model::EvalVolFunctionals(ad_state, sparam, ad_funct);

      const AutoDiff<COMP*dsize> & ad_j = ad_funct(0); // hack for drag
      
      double * tmp = &temp[j * COMP];
      for (int ll = 0; ll < COMP; ++ll)
        tmp[ll] = intweights * ad_j.DValue(ll);

    }

    // int_{dK} phi * dJ/dw
    mygemm('n', 't', ndof_w, COMP, ed.nip, 1., &ed.phi[0], &temp[0], 1., &vecG[0]);
  }


  if (Model::NumBdryFluxWeight > 0) {
  
    int offset_br2 = 0;

    // Facet contributions
    for (int ff = 0; ff < ed.nf; ++ff) {
      FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];

      int ndof_l = fd.ndof_l;
      int bcnr   = fd.bcnr;
      int nip    = fd.nip;    
    
      int width_br2 = COMP * D * (1 + COMP * (ndof_w + ndof_l));

      if (ndof_l != 0) { // interior faces
#ifdef BR2      
        offset_br2 += max(ndof_w, nip) * width_br2;
#endif      
        continue;  
      }

      if (bcnr == -2) continue; // zero measure faces

      for (int j = 0; j < nip; ++j) {

        Vec<COMP, AutoDiff<COMP*dsize> > ad_state(0.), ad_fcn(0.), ad_fvn(0.), ad_fn(0.);
        Mat<COMP, D, AutoDiff<COMP*dsize> > ad_grad(0.);

        Mat<Model::NumBdryFluxWeight, COMP> weights(0.);
        Vec<Model::NumBdryFluxWeight, AutoDiff<COMP*dsize> > monitors(0.);

        SpatialParams<D> sparam;

        const double intweights = fd.qw[j];

        const double * pq   = &fd.q1[j * D * COMP];
        const double * pw   = &fd.w1[j * COMP];
        const double * pn   = &fd.n1[j * D];
        const double * ppos = &fd.qp[j * D];

        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = ppos[dd];
        for (int dd = 0; dd < D; ++dd)
          sparam.normal(dd) = pn[dd];
      
        for (int ll = 0; ll < COMP; ++ll)
          ad_state(ll) = AutoDiff<COMP*dsize> (pw[ll], ll);
        if (Model::Diffusion)
          for (int ll = 0; ll < COMP; ++ll)
            for (int dd = 0; dd < D; ++dd)
              ad_grad(ll, dd) = AutoDiff<COMP*dsize> (pq[dd+D*ll], ll+COMP*(dd+1));

        min_y = ed.vol / fd.h;
      
        if (Model::Convection)
          Model::EvalBdryConvFlux(bcnr, ad_state, sparam, ad_fcn);
        if (Model::Diffusion)
          Model::EvalBdryDiffFlux(bcnr, ad_state, ad_grad, sparam, ad_fvn);

        ad_fn = ad_fcn - ad_fvn;

        Model::EvalBdryFluxWeight(bcnr, sparam, weights);

        monitors = weights * ad_fn;

        const AutoDiff<COMP*dsize> & ad_j = monitors(0); // hack for drag
      
        double * tmp = &temp[j * COMP];
        for (int ll = 0; ll < COMP; ++ll)
          tmp[ll] = intweights * ad_j.DValue(ll);

        if (Model::Diffusion) {
          double * tmp2 = &temp2[j * COMP * D];
          for (int ll = 0; ll < COMP; ++ll)
            for (int dd = 0; dd < D; ++dd)
              tmp2[dd+D*ll] = intweights * ad_j.DValue(ll+COMP*(dd+1));

#ifdef BR2
          const double * tmp_lift = &temp_lift2[offset_br2 + j * width_br2 + COMP * D];

          for (int mm = 0; mm < COMP; ++mm)
            for (int qq = 0; qq < ndof_w; ++qq) {
              double val = 0.;
              for (int ll = 0; ll < COMP; ++ll)
                for (int dd = 0; dd < D; ++dd)
                  val += weights(0, ll) * tmp_lift[dd+D*(ll+COMP*(qq+ndof_w*mm))] * sparam.normal(dd);
              vecG[qq+ndof_w*mm] -= intweights * val;
            }
#endif
        }      
      }
    
      // int_{dK} phi * dJ/dw
      mygemm('n', 't', ndof_w, COMP, nip, 1., &fd.phi1[0], &temp[0], 1., &vecG[0]);

      // int_{dK} phi * dJ/dq    
      if (Model::Diffusion)
        mygemm('n', 't', ndof_w, COMP*D, nip, 1., &fd.phi1[0], &temp2[0], 1., &vecF[0]);


#ifdef BR2
      offset_br2 += max(ndof_w, nip) * width_br2;
#endif
    }
  }


}
