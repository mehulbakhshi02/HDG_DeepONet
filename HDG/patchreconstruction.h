/** @file
 * @brief In this file we compute the reconstructed solution using the monomial basis. This is the procedure used
 *        - Store all the gauss points of a patch
 *        - Compute the monomials upto order k on this patch
 *        - H1 orthonormalize these monomials on this patch using Gram-Schmidt
 *        - Compute the H1 projection of the solution from solution order p to k on this patch
 *        - Compute the solution in the original basis using the coefficients of the monomial basis for order k
 * @param[int] - sol - Contains the initial solution of type \c Solution upto order p cellwise. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[out] - sol_rec - Contains the reconstructed solution of type \c Solution upto order k cellwise. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[in] - add_order - Number of orders by which we want to increase from original. Size = 1 int
 * @param[in] - numlayers - Number of layers of cells that form the patch. Size = 1 int
 * @param[in] - lh - Local heap used to generate the finite element spaces. 
*/
template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::PatchReconstruction(const Solution & sol, Solution & sol_rec, const int add_order, const int numlayers, LocalHeap & lh) {

    // Approximate a p+1 solution via H1 patch reconstruction
  max_order = max_order + add_order;
  // Seems to be useful only in the case where we write out the solution
  order = order + add_order;
  for (int i = 0; i < ne; ++i)
    order_array[i] = order_array[i] + add_order;

  int zk = 1;
  if(D==2)
  {
    zk = 0;
  }
  // // Higher than dual space
  Space fspace_loc;
  // If add_order is 0 then we have the space space as the solution
  if(add_order == 0)
    fspace_loc = fspace;
  else if(add_order == 1)
    fspace_loc = fspacedual;// If add_order is 1 then we have one order higher space
  else if(add_order == 2)
    fspace_loc = fspacehp;// If add_order is 1 then we have two orders higher space
  else
  {
    cout<<"Error in choosing the correct order space for patch reconstruction!"<<endl;
  }
  cout << string(32, '*') << endl;
  cout << "Patch reconstruction... " << endl;
  cout << string(32, '*') << endl;

  GetElementInformation(fspace_loc, fadata, eldata, ma, order_array, lh);

  // Prolongate old solution into order_array arrays
  Solution sol_prol;
  ProlongateOrder(sol, sol_prol, add_order);

  // ... and reconstruct it at new integration points
  ReconstructSolution(sol_prol);
  sol_rec = sol_prol;

  int ps_max = pow(nf_max + 1, numlayers);

  vector<double> weights(nip_max * ps_max);

  vector<double> H1basis(ndof_w_max * nip_max * ps_max);

  vector<double> dxH1basis(ndof_w_max * nip_max * ps_max);
  vector<double> dyH1basis(ndof_w_max * nip_max * ps_max);
  vector<double> dzH1basis(ndof_w_max * nip_max * ps_max);

  // L2 basis for reconstructing q
  vector<double> L2basis;
  if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
    L2basis.resize(ndof_w_max * nip_max * ps_max);

  vector<double> vecWH1(ndof_w_max * COMP);

  vector<double> vecQL2;
  if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
    vecQL2.resize(ndof_q_max * COMP);

  vector<int>    piv(ndof_w_max);
  
  vector<int> pivq;
  if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
    pivq.resize(ndof_q_max);
  
    // run mygemm instead of dgesv
  vector<double> rhs_w(COMP * ndof_w_max, 0.);
  vector<double> rhs_q(COMP * ndof_q_max, 0.);   
  // Main loop over elements
  // #pragma omp parallel for firstprivate(weights, H1basis, dxH1basis, dyH1basis, dzH1basis, L2basis, vecWH1, vecQL2, piv, pivq)
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
      {
        weights[index1] = ed.qw[j];
      }
    }
    // remember ed_cntr = order_array[i]    
    // Compute monomial basis functions and their derivatives at integration points
    for (int m = 0, index2 = 0; m <= ed_cntr.order; ++m) 
    {
      for (int n = 0; n <= ed_cntr.order-m; ++n) 
      {
        for(int l=0;l<=(ed_cntr.order-m-n)*zk;++l,++index2)
        {
          for (int k = 0, index1 = 0; k < patch.size(); ++k) 
          {

            ElementData<D, COMP> & ed = *eldata[patch[k]];
              
            for (int j = 0; j < ed.nip; ++j, ++index1) {

              int index3 = index2 * nip_total + index1;
              
              double x = ed.qp[j*D];
              double y = ed.qp[j*D+1];
              double z;
              // Some coordinate transformation
              x -= ed_cntr.cntr[0];
              y -= ed_cntr.cntr[1];
              if(zk == 1)
              {
                z = ed.qp[j*D+2];
                z -= ed_cntr.cntr[2];
              }
              else
              {
                z = 1.0;
              }
              // // // // Trying to scale the coordinates between -1 and 1
              // x = x/pow(ed_cntr.vol, 1.0/(double)D);
              // y = y/pow(ed_cntr.vol, 1.0/(double)D);

              // x^n*y^m;
              H1basis[index3]=pow(x,n)*pow(y,m)*pow(z,l);

              // x^n*y^m;
              if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
                L2basis[index3] = pow(x,n) * pow(y,m) * pow(z,l);

              // d/dx (x^n*y^m) = n*x^(n-1)*y^m
              dxH1basis[index3] = n > 0 ? n * pow(x, n-1) * pow(y, m)*pow(z,l) : 0.;

              // d/dy (x^n*y^m) = m*x^n*y^(m-1)
              dyH1basis[index3] = m > 0 ? m * pow(x, n) * pow(y, m-1)*pow(z,l) : 0.;

              if(zk == 1)
              {
                dzH1basis[index3] = l > 0 ? l * pow(x, n) * pow(y, m)*pow(z,l-1) : 0.;
              }
            }
          }
        }
      }
    }

    // Actual MGS (wrt H1-norm)
    int ndof = (ed_cntr.order + 1) * (ed_cntr.order + 2) / 2;
    if(zk == 1)
    {
      ndof=(ed_cntr.order + 1) * (ed_cntr.order + 2)*(ed_cntr.order+3)/6;
    }
    for (int pp = 0; pp < ndof; ++pp) {
      
      // Orthogonalize twice
      for (int n = 0; n < 2; ++n) {
      
        // Projection
        for (int qq = 0; qq < pp; ++qq) {

          double rpq = 0.;
          double rpq_l2 = 0.;

          // Compute dot-product (phi_m, phi_l)_H1
          for (int j = 0; j < nip_total; ++j) {

            int idx_p = pp * nip_total + j;
            int idx_q = qq * nip_total + j;
            
            rpq += weights[j] *   H1basis[idx_p] *   H1basis[idx_q];
            rpq += weights[j] * dxH1basis[idx_p] * dxH1basis[idx_q];
            rpq += weights[j] * dyH1basis[idx_p] * dyH1basis[idx_q];
            if(zk == 1)
              rpq += weights[j] * dzH1basis[idx_p] * dzH1basis[idx_q];

            // l2 basis for reconstructing q
            if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
              rpq_l2 += weights[j] *   L2basis[idx_p] *   L2basis[idx_q];
          }

          // Subtract projection
          for (int j = 0; j < nip_total; ++j) {

            int idx_p = pp * nip_total + j;
            int idx_q = qq * nip_total + j;
            
            H1basis[idx_p]   -= rpq *   H1basis[idx_q];
            dxH1basis[idx_p] -= rpq * dxH1basis[idx_q];
            dyH1basis[idx_p] -= rpq * dyH1basis[idx_q];
            if(zk == 1)
              dzH1basis[idx_p] -= rpq * dzH1basis[idx_q];

            //l2 basis for q
            if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
              L2basis[idx_p]   -= rpq_l2 *   L2basis[idx_q];

          }
        }
      }

      // Compute norm aka dot-product (phi_m, phi_m)_H1
      double rpp = 0.;
      // Compute norm aka dot-product (phi_m, phi_m)_l2 for the reconstruction of q
      double rpp_l2 = 0.;

      for (int j = 0; j < nip_total; ++j) {

        int idx_p = pp * nip_total + j;

        rpp += weights[j] *   H1basis[idx_p] *   H1basis[idx_p];
        rpp += weights[j] * dxH1basis[idx_p] * dxH1basis[idx_p];
        rpp += weights[j] * dyH1basis[idx_p] * dyH1basis[idx_p];
        if(zk == 1)
          rpp += weights[j] * dzH1basis[idx_p] * dzH1basis[idx_p];
        // used to reconstruct q
        if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
          rpp_l2 += weights[j] *   L2basis[idx_p] *   L2basis[idx_p];
      }

      rpp = sqrt(rpp);

      rpp_l2 = sqrt(rpp_l2);

      // Normalization      
      for (int j = 0; j < nip_total; ++j) {

        int idx_p = pp * nip_total + j;

        H1basis[idx_p]   /= rpp;
        dxH1basis[idx_p] /= rpp;
        dyH1basis[idx_p] /= rpp;
        if(zk == 1)
          dzH1basis[idx_p] /= rpp;
        
        if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
          L2basis[idx_p]   /= rpp_l2;
      }
    }

    // Finally: Patch reconstruction
    for (int ll = 0; ll < COMP; ++ll) {
      for (int pp = 0; pp < ndof; ++pp) {
        // This contains the coeffients of the monomial basis defined on the patch
        // for w, qx and qy
        double val = 0.;
        double val_q[D] = {0.};
        
        for (int k = 0, index1 = 0; k < patch.size(); ++k) {

          ElementData<D, COMP> & ed = *eldata[patch[k]];
            
          for (int j = 0; j < ed.nip; ++j, ++index1) {

            const double   qw =  ed.qw[j];
            const double * pw = &ed.w[j * COMP];
            const double * pq = &ed.q[j * COMP * D];

            int idx_pp = pp * nip_total + index1;
            
            val += qw *   H1basis[idx_pp] * pw[ll];
            val += qw * dxH1basis[idx_pp] * pq[D*ll];
            val += qw * dyH1basis[idx_pp] * pq[D*ll+1];  
            if(zk == 1)
              val += qw * dzH1basis[idx_pp] * pq[D*ll+2];
            // For qx and qy we can only do L2 projection
            if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
            {            
              for(int dd = 0; dd < D; ++dd)
              {
                val_q[dd] += qw * L2basis[idx_pp] * pq[dd+D*ll];
              }
            }
          }
        }
        vecWH1[pp+ndof*ll] = val;
        if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
          for(int dd = 0; dd < D; ++dd)
          {
            vecQL2[pp+ndof*(dd+D*ll)] = val_q[dd];
          }

      }
    }

    // Transform to original basis again
    int ndof_w = ed_cntr.ndof_w;
    int ndof_q = ed_cntr.ndof_q;
    int nip    = ed_cntr.nip;

    double * vecW = &sol_rec.vecW[COMP * ed_cntr.offset_w];
    double * vecQ;
    if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
      vecQ = &sol_rec.vecQ[COMP * ed_cntr.offset_q];

    // fill(vecW, vecW + COMP * ndof_w, 0.0); 
    // if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
    //   fill(vecQ, vecQ + COMP * ndof_w * D, 0.0); 

      rhs_w.assign(ndof_w*COMP, 0.);
      rhs_q.assign(ndof_q*COMP, 0.);

    for (int j = 0; j < nip; ++j) {

      const double    qw = ed_cntr.qw[j];
      const double * phi = &ed_cntr.phi[j * ndof_w];

      double pw[COMP] = {0.0};
      double pq[COMP * D] = {0.0};

      // Reconstruction of the solution using H1-basis (the orthogonalized monomial basis)
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof; ++pp)
        {          
          pw[ll] += H1basis[pp*nip_total+j] * vecWH1[pp+ndof*ll];
          if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
          {         
            for(int dd = 0; dd < D; ++dd)
            {
                pq[dd+ll*D] += L2basis[pp*nip_total+j] * vecQL2[pp+ndof*(dd+D*ll)];
            }
          }
        }

      // Assemble right hand side
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_w; ++pp)
        {          
          rhs_w[pp+ndof_w*ll] += qw * pw[ll] * phi[pp];
          if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
          {
            for(int dd = 0; dd < D; ++dd)
            {
              rhs_q[pp+ndof_w*(dd+D*ll)] += qw * pq[dd+ll*D] * phi[pp];
            }
          }
        }

    }

    mygemm('n', 'n', ndof_w, COMP, ndof_w, 1., &ed_cntr.inv_mass[0], &rhs_w[0], 0., vecW);
    if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
    {
        mygemm('n', 'n', ndof_w, COMP * D, ndof_w, 1., &ed_cntr.inv_mass[0], &rhs_q[0], 0., vecQ);
    }

    // // Solve system
    // int size = ndof_w;
    // int nrhs = COMP;
    // int info;
    // dgesv_(&size, &nrhs, &ed_cntr.mass[0], &size, &piv[0], vecW, &size, &info);
    // if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
    // {
    //   size = ndof_w;
    //   nrhs = COMP * D;
    //   dgesv_(&size, &nrhs, &ed_cntr.mass[0], &size, &pivq[0], vecQ, &size, &info);      
    // }
  }// end of loop over elements

}
