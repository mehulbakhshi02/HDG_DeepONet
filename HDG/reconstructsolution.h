#include "../helper_blas.h"
/** @file
 * @brief In this file we take the coefficients from the varialbe \c sol and write it into the quadrature points of element data. This has to be used
 * carefully because it overwrites the global variable eldata. 
 * @param[in] - sol - Contains the current coeffiencts in the variable of type \c Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double 
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::ReconstructSolution(Solution & sol) {

  ReconstructSolution(sol, fadata, eldata);
  
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::ReconstructSolution(Solution & sol, vector<FacetData<D, COMP> *> & fadata, vector<ElementData<D, COMP> *> & eldata) {
  // Elemental data
  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];
    int offset_q = ed.offset_q;
    int offset_w = ed.offset_w;
    int ndof_w   = ed.ndof_w;
    int nip      = ed.nip;

    mygemm('t', 'n', COMP, nip, ndof_w, 1., &sol.vecW[COMP*offset_w], &ed.phi[0], 0., &ed.w[0]);

    if (Model::Diffusion || Model::Source)
      mygemm('t', 'n', COMP * D, nip, ndof_w, 1., &sol.vecQ[COMP*offset_q], &ed.phi[0], 0., &ed.q[0]);
    else {
      // Should be rewritten with gemm (might not be possible in the requested order)
      for (int j = 0; j < nip; ++j) {
        double * dphi = &ed.dphi[j * ndof_w * D];
        double * pq   = &ed.q[j * COMP * D];
        
        for (int ll = 0; ll < COMP; ++ll)
          for (int dd = 0; dd < D; ++dd) {
            double val = 0.;
            for (int pp = 0; pp < ndof_w; ++pp)
              val += sol.vecW[COMP*offset_w+pp+ndof_w*ll] * dphi[pp+ndof_w*dd];
            pq[dd+D*ll] = val;
          }
      }
    }
  }

  // Facet Data
  Vec<COMP> state, state_bc;
  SpatialParams<D> sparam;

  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;
    FacetData<D, COMP> & fd = *fadata[i];

    if (fd.bcnr == -2) continue; // zero-measure face
    
    int offset_q1 = fd.offset_q1, offset_q2 = fd.offset_q2;
    int offset_w1 = fd.offset_w1, offset_w2 = fd.offset_w2;
    int offset_l  = fd.offset_l;
    int ndof_w1   = fd.ndof_w1,   ndof_w2   = fd.ndof_w2;
    int ndof_l    = fd.ndof_l;
    int nip       = fd.nip;

    mygemm('t', 'n', COMP, nip, ndof_w1, 1., &sol.vecW[COMP*offset_w1], &fd.phi1[0], 0., &fd.w1[0]);

    if (Model::Diffusion)
      mygemm('t', 'n', COMP * D, nip, ndof_w1, 1., &sol.vecQ[COMP*offset_q1], &fd.phi1[0], 0., &fd.q1[0]);

    if (fd.ndof_l != 0) {
      mygemm('t', 'n', COMP, nip, ndof_w2, 1., &sol.vecW[COMP*offset_w2], &fd.phi2[0], 0., &fd.w2[0]);
      mygemm('t', 'n', COMP, nip, ndof_l, 1., &sol.vecL[COMP*offset_l], &fd.mu[0], 0., &fd.lambda[0]);
      if (Model::Diffusion)
        mygemm('t', 'n', COMP * D, nip, ndof_w2, 1., &sol.vecQ[COMP*offset_q2], &fd.phi2[0], 0., &fd.q2[0]);
    } else {
      min_y =  eldata[fd.elnr1]->vol / fd.h;
      for (int j = 0; j < nip; ++j) {
        const double * pw1  = &fd.w1[j * COMP];
        double * pw2  = &fd.w2[j * COMP];
        const double * ppos = &fd.qp[j * D];
        const double * pn   = &fd.n1[j * D];
      
        for (int ll = 0; ll < COMP; ++ll)
          state(ll) = pw1[ll];
        for (int dd = 0; dd < D; ++dd)
          sparam.normal(dd) = pn[dd];
        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = ppos[dd];
	
        state_bc = 0.;
        Model::EvalBdryState(fd.bcnr, state, sparam, state_bc);

        for (int ll = 0; ll < COMP; ++ll)
          pw2[ll] = state_bc(ll);
      }
    }
  }
}


