#ifndef INITIAL_H
#define INITIAL_H
/** @file
 * @brief In this file we setup the initial conditions for the problem. We have three options to setup
 *  - Zero for all variables used by setting \c IT_Zero
 *  - Constant for all variables used by setting \c IT_Constant
 *  - Function for all one variable used by setting \c IT_Function
 * @param[out] - sol - Contains the initial conditions of type \c Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
*/

template <int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
  ::Initial(Solution & sol) const {

  sol.vecQ.assign(ndof_q_total * COMP, 0.0);
  sol.vecW.assign(ndof_w_total * COMP, 0.0);
  sol.vecL.assign(ndof_l_total * COMP, 0.0);
  
  if (Model::inittype == IT_Zero) {
      
  } else if (Model::inittype == IT_Constant) {
    
    // Obtain constant state
    Vec<COMP> state;
    SpatialParams<D> sparam; // dummy (not necessary as state is constant)
    Model::EvalInitial(sparam, state);
      
    // Only initialize mean values of w
    for (int i = 0; i < ne; ++i) {
      ElementData<D, COMP> & ed = *eldata[i];

      double * vecW = &sol.vecW[COMP * ed.offset_w];
      for (int ll = 0; ll < COMP; ++ll)
        vecW[ll * ed.ndof_w] = state(ll);      
    }

    // Only initialize mean values of lambda
    for (int i = 0; i < nf; ++i) {
      if (!fadata[i]) continue;
      FacetData<D, COMP> & fd = *fadata[i];
      if (fd.ndof_l == 0) continue;

      double * vecL = &sol.vecL[COMP * fd.offset_l];
      for (int ll = 0; ll < COMP; ++ll)
        vecL[ll * fd.ndof_l] = state(ll);      
    }
    
  } else if (Model::inittype == IT_Function) {

    // So far least-squares with QR factorization
    vector<double> rhs_w(COMP * ndof_w_max, 0.);
    vector<double> rhs_q(COMP * ndof_q_max, 0.);    
  
    for (int i = 0; i < ne; ++i) {
      ElementData<D, COMP> & ed = *eldata[i];

      int ndof_w = ed.ndof_w;
      int ndof_q = ed.ndof_q;      
      int nip    = ed.nip;

      double * vecW = &sol.vecW[COMP * ed.offset_w];
      double * vecQ = &sol.vecQ[COMP * ed.offset_q];

      rhs_w.assign(ndof_w*COMP, 0.);
      rhs_q.assign(ndof_q*COMP, 0.);
      
      for (int j = 0; j < nip; ++j) {
        double intweight = ed.qw[j];
        const double * phi = &ed.phi[j * ndof_w];

        SpatialParams<D> sparam;
        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = ed.qp[D*j+dd];

        Vec<COMP> sol;
        Model::EvalInitial(sparam, sol);

        for (int ll = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < ndof_w; ++pp)
            rhs_w[pp+ndof_w*ll] += intweight * sol(ll) * phi[pp];
      }

      mygemm('n', 'n', ndof_w, COMP, ndof_w, 1., &ed.inv_mass[0], &rhs_w[0], 0., vecW);

      for (int j = 0; j < nip; ++j) {
        const double intweight = ed.qw[j];
        const double * phi = &ed.phi[j * ndof_w];
        const double * dphi = &ed.dphi[ndof_q * j];

        // Reconstruct solution gradient
        double dsol[D] = {0.0};
        for (int dd = 0; dd < D; ++dd)
          for (int pp = 0; pp < ndof_w; ++pp)
            dsol[dd] += vecW[pp] * dphi[pp+ndof_w*dd];

        for (int ll = 0; ll < COMP; ++ll)
          for (int dd = 0; dd < D; ++dd)
            for (int pp = 0; pp < ndof_w; ++pp)
              rhs_q[pp+ndof_w*(dd+D*ll)] += intweight * dsol[dd] * phi[pp];
      }

      mygemm('n', 'n', ndof_w, COMP * D, ndof_w, 1., &ed.inv_mass[0], &rhs_q[0], 0., vecQ);
    }

    vector<double> rhs_l(COMP * ndof_l_max, 0.);

    for (int i = 0; i < nf; ++i) {
      if (!fadata[i]) continue;
      FacetData<D, COMP> & fd = *fadata[i];
      if (fd.ndof_l == 0) continue;

      int ndof_l = fd.ndof_l;
      int nip    = fd.nip;

      rhs_l.assign(COMP*ndof_l, 0.0);

      double * vecL = &sol.vecL[COMP * fd.offset_l];

      for (int j = 0; j < nip; ++j) {
        double intweight = fd.qw[j];
        const double * mu = &fd.mu[j * ndof_l];

        SpatialParams<D> sparam;
        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = fd.qp[D*j+dd];

        Vec<COMP> sol;
        Model::EvalInitial(sparam, sol);

        for (int ll = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < ndof_l; ++pp)
            rhs_l[pp+ndof_l*ll] += intweight * sol(ll) * mu[pp];
      }

      mygemm('n', 'n', ndof_l, COMP, ndof_l, 1., &fd.inv_mass[0], &rhs_l[0], 0., vecL);
    }

    
  } 
  else if(Model::inittype == IT_File)
  {
      // So far least-squares with QR factorization
    vector<double> rhs_w(COMP * ndof_w_max, 0.);   
    vector<double> rhs_q(COMP * ndof_q_max, 0.); 
    cout << string(32, '*') << endl;
    cout << "Initial condition from file... " << endl;
    cout << string(32, '*') << endl;
    ifstream file("solution_trial.txt", ios::in);
    if (!file.is_open())
    {
      cout<<"File not found!"<<endl;
      exit(1);
    }
    int ne_inp = 0;
    int D_inp = 0;
    file >> ne_inp >> D_inp;
    if(ne!=ne_inp || D!=D_inp)
    {
      cout<<"Incorrect number of cells or incorrect dimensions!!"<<endl;

      exit(1);
    }
    for (int i = 0; i < ne; ++i) {
      ElementData<D, COMP> & ed = *eldata[i];

      int ndof_w = ed.ndof_w;
      int ndof_q = ed.ndof_q;      
      int nip    = ed.nip;

      double * vecW = &sol.vecW[COMP * ed.offset_w];
      double * vecQ = &sol.vecQ[COMP * ed.offset_q];

      rhs_w.assign(ndof_w*COMP, 0.);
      rhs_q.assign(ndof_q*COMP, 0.);
      // Check if the polynomial order and number of quadrature points are the same
      int i_inp = 0;
      int p_inp = 0;
      int nip_inp = 0;
      file >> i_inp >> p_inp >> nip_inp;
      if(ed.order!=p_inp || ed.nip!=nip_inp)
      {
        cout<<"Incorrect polynomial order or incorrect order of quadrature in cell number "<<i<<endl;
        exit(1);
      }
      for (int j = 0; j < nip; ++j) {
        double intweight = ed.qw[j];
        const double * phi = &ed.phi[j * ndof_w];

        SpatialParams<D> sparam;
        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = ed.qp[D*j+dd];

        double x, y, qw, u;
        double tol = 1e-15;
        file >> qw >> x >> y >> u;
        if(fabs(x-sparam.pos(0))>tol || fabs(y-sparam.pos(1))>tol || fabs(qw-intweight)>tol)
        {
          cout<<"Incorrect quadrature position or quadrature weight in cell number "<<i<<endl;
          cout<<fabs(x-sparam.pos(0))<<" "<<fabs(y-sparam.pos(1))<<" "<<fabs(qw-intweight)<<endl;
          exit(1);
        }
        Vec<COMP> sol;
        // Model::EvalInitial(sparam, sol);
        sol(0) = u;
        for (int ll = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < ndof_w; ++pp)
            rhs_w[pp+ndof_w*ll] += intweight * sol(ll) * phi[pp];
      }

      mygemm('n', 'n', ndof_w, COMP, ndof_w, 1., &ed.inv_mass[0], &rhs_w[0], 0., vecW);
      // Reconstruct the gradient varaible
      if(Model::Diffusion || Model::Source)
      {
        for (int j = 0; j < nip; ++j) 
        {
          const double intweight = ed.qw[j];
          const double * phi = &ed.phi[j * ndof_w];
          const double * dphi = &ed.dphi[ndof_q * j];

          // Reconstruct solution gradient
          for (int ll = 0; ll < COMP; ++ll)
          {
            double dsol[D] = {0.0};
            for (int dd = 0; dd < D; ++dd)
              for (int pp = 0; pp < ndof_w; ++pp)
                dsol[dd] += vecW[pp+ll*ndof_w] * dphi[pp+ndof_w*dd];

              for (int dd = 0; dd < D; ++dd)
                for (int pp = 0; pp < ndof_w; ++pp)
                  rhs_q[pp+ndof_w*(dd+D*ll)] += intweight * dsol[dd] * phi[pp];          
          }

        }

        mygemm('n', 'n', ndof_w, COMP * D, ndof_w, 1., &ed.inv_mass[0], &rhs_q[0], 0., vecQ);  
      }
      
    }
  }
  else {
    cout << "Unknown init type." << endl;
    exit(0);
  }


    
}

#endif
