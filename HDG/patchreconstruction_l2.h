template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::PatchReconstruction(const Solution & sol, Solution & sol_rec, int numlayers, LocalHeap & lh) {

  if (D != 2) {
    cout << "[PatchReconstruction] So far implemented for 2d only." << endl;
    exit(0);
  }

  cout << string(32, '*') << endl;
  cout << "Patch reconstruction... " << endl;
  cout << string(32, '*') << endl;
  

  GetElementInformation(fspacedual, fadata, eldata, ma, order_array, lh);

  // Prolongate old solution into p+1 arrays
  Solution sol_prol;
  ProlongateOrder(sol, sol_prol);

  // ... and reconstruct it at new integration points
  ReconstructSolution(sol_prol);

  sol_rec = sol_prol;

  int ps_max = pow(nf_max + 1, numlayers);

  vector<double> weights(nip_max * ps_max);

  vector<double> H1basis(ndof_w_max * nip_max * ps_max);
  // vector<double> dxH1basis(ndof_w_max * nip_max * ps_max);
  // vector<double> dyH1basis(ndof_w_max * nip_max * ps_max);

  vector<double> vecWH1(ndof_w_max * COMP);

  vector<int>    piv(ndof_w_max);
    
  // Main loop over elements
  for (int i = 0; i < ne; ++i) {
    
    ElementData<D, COMP> & ed_cntr = *eldata[i];

    // Determine patch elements
    vector<int> patch;
    patch.push_back(i);

    for (int n = 0; n < numlayers; ++n) {

      int ps = patch.size();
      for (int p = 0; p < ps; ++p) {

        int ii = patch[p];
        ElementData<D, COMP> & ed = *eldata[ii];

        for (int f = 0; f < ed.nf; ++f) {
          FacetData<D, COMP> & fd = *fadata[ed.faces[f]];
          int el = -1;
          if (fd.elnr1 == ii && fd.elnr2 != -1)
            el = fd.elnr2;
          else
            el = fd.elnr1;
          if (el != -1 && find(patch.begin(), patch.end(), el) == patch.end())
            patch.push_back(el);
        }
      }
    }
    
    // Collect quadrature weights
    int nip_total = 0;
    for (int k = 0, index1 = 0; k < patch.size(); ++k) {

      ElementData<D, COMP> & ed = *eldata[patch[k]];

      nip_total += ed.nip;
      for (int j = 0; j < ed.nip; ++j, ++index1)
        weights[index1] = ed.qw[j];
    }
    
    // Compute monomial basis functions and their derivatives at integration points
    for (int m = 0, index2 = 0; m <= ed_cntr.order; ++m) {
      for (int n = 0; n <= ed_cntr.order-m; ++n, ++index2) {

        for (int k = 0, index1 = 0; k < patch.size(); ++k) {

          ElementData<D, COMP> & ed = *eldata[patch[k]];
            
          for (int j = 0; j < ed.nip; ++j, ++index1) {

            int index3 = index2 * nip_total + index1;
            
            double x = ed.qp[j*D];
            double y = ed.qp[j*D+1];

            // Some coordinate transformation
            x -= ed_cntr.cntr[0];
            y -= ed_cntr.cntr[1];
            
            // x^n*y^m;
            H1basis[index3] = pow(x, n) * pow(y, m);

            // // d/dx (x^n*y^m) = n*x^(n-1)*y^m
            // dxH1basis[index3] = n > 0 ? n * pow(x, n-1) * pow(y, m) : 0.;

            // // d/dy (x^n*y^m) = m*x^n*y^(m-1)
            // dyH1basis[index3] = m > 0 ? m * pow(x, n) * pow(y, m-1) : 0.;
          }
        }
      }
    }

    // Actual MGS (wrt H1-norm)
    int ndof = (ed_cntr.order + 1) * (ed_cntr.order + 2) / 2;
    
    for (int pp = 0; pp < ndof; ++pp) {
      
      // Orthogonalize twice
      for (int n = 0; n < 2; ++n) {
      
        // Projection
        for (int qq = 0; qq < pp; ++qq) {

          double rpq = 0.;

          // Compute dot-product (phi_m, phi_l)_H1
          for (int j = 0; j < nip_total; ++j) {

            int idx_p = pp * nip_total + j;
            int idx_q = qq * nip_total + j;
            
            rpq += weights[j] *   H1basis[idx_p] *   H1basis[idx_q];
            // rpq += weights[j] * dxH1basis[idx_p] * dxH1basis[idx_q];
            // rpq += weights[j] * dyH1basis[idx_p] * dyH1basis[idx_q];
          }

          // Subtract projection
          for (int j = 0; j < nip_total; ++j) {

            int idx_p = pp * nip_total + j;
            int idx_q = qq * nip_total + j;
            
            H1basis[idx_p]   -= rpq *   H1basis[idx_q];
            // dxH1basis[idx_p] -= rpq * dxH1basis[idx_q];
            // dyH1basis[idx_p] -= rpq * dyH1basis[idx_q];
          }
        }
      }

      // Compute norm aka dot-product (phi_m, phi_m)_H1
      double rpp = 0.;
      
      for (int j = 0; j < nip_total; ++j) {

        int idx_p = pp * nip_total + j;

        rpp += weights[j] *   H1basis[idx_p] *   H1basis[idx_p];
        // rpp += weights[j] * dxH1basis[idx_p] * dxH1basis[idx_p];
        // rpp += weights[j] * dyH1basis[idx_p] * dyH1basis[idx_p];          
      }

      rpp = sqrt(rpp);

      // Normalization      
      for (int j = 0; j < nip_total; ++j) {

        int idx_p = pp * nip_total + j;

        H1basis[idx_p]   /= rpp;
        // dxH1basis[idx_p] /= rpp;
        // dyH1basis[idx_p] /= rpp;
      }
    }

    // Finally: Patch reconstruction
    for (int ll = 0; ll < COMP; ++ll) {
      for (int pp = 0; pp < ndof; ++pp) {

        double val = 0.;
        
        for (int k = 0, index1 = 0; k < patch.size(); ++k) {

          ElementData<D, COMP> & ed = *eldata[patch[k]];
            
          for (int j = 0; j < ed.nip; ++j, ++index1) {

            const double   qw =  ed.qw[j];
            const double * pw = &ed.w[j * COMP];
            const double * pq = &ed.q[j * COMP * D];

            int idx_pp = pp * nip_total + index1;
            
            val += qw *   H1basis[idx_pp] * pw[ll];
            // val += qw * dxH1basis[idx_pp] * pq[D*ll];
            // val += qw * dyH1basis[idx_pp] * pq[D*ll+1];            
          }
        }

        vecWH1[pp+ndof*ll] = val;
      }
    }

    // Transform to original basis again
    int ndof_w = ed_cntr.ndof_w;
    int nip    = ed_cntr.nip;

    double * vecW = &sol_rec.vecW[COMP * ed_cntr.offset_w];

    fill(vecW, vecW + COMP * ndof_w, 0.0); 
    
    for (int j = 0; j < nip; ++j) {

      const double    qw = ed_cntr.qw[j];
      const double * phi = &ed_cntr.phi[j * ndof_w];

      double pw[COMP] = {0.0};

      // Reconstruction of the solution using H1-basis
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof; ++pp)
          pw[ll] += H1basis[pp*nip_total+j] * vecWH1[pp+ndof*ll];

      // Assemble right hand side
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_w; ++pp)
          vecW[pp+ndof_w*ll] += qw * pw[ll] * phi[pp];
    }

    // Solve system
    int size = ndof_w;
    int nrhs = COMP;
    int info;
    dgesv_(&size, &nrhs, &ed_cntr.mass[0], &size, &piv[0], vecW, &size, &info);
  }
}
