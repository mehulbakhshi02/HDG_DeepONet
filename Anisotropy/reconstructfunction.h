/** @file 
 * @brief This file reconstructs the coefficients for any given function so that it can
 * be used in patch reconstruction and subsequently in anisotropy calculations
 * @param[in] - sol - Contains the data of type Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[out] - sol_func - Contains the function f(u) and its gradients. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[in] - lh - Local heap.
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ReconstructFunction(const Solution & sol, Solution & sol_func, LocalHeap & lh) {

	int max_order = *max_element(order_array.begin(), order_array.end());

	Array<int> vnums; 

	vector<int> piv(ndof_w_max);

	sol_func = sol;
    vector<double> rhs_w(COMP * ndof_w_max, 0.);
    vector<double> rhs_q(COMP * ndof_q_max, 0.);    

	for (int i = 0; i < ne; ++i) {

		HeapReset hr(lh);
      ElementData<D, COMP> & ed = *eldata[i];

      int ndof_w = ed.ndof_w;
      int ndof_q = ed.ndof_q;      
      int nip    = ed.nip;

      double * vecW = &sol_func.vecW[COMP * ed.offset_w];
      double * vecQ = &sol_func.vecQ[COMP * ed.offset_q];

      rhs_w.assign(ndof_w*COMP, 0.);
      rhs_q.assign(ndof_q*COMP, 0.);

		for (int j = 0; j < ed.nip; ++j) {

			const double   qw  = ed.qw[j];
			const double * qp  = &ed.qp[j * D];
			const double * pw  = &ed.w[j * COMP];
			const double * phi = &ed.phi[j * ndof_w];

			SpatialParams<D> sparam;
			for (int dd = 0; dd < D; ++dd)
			  sparam.pos(dd) = qp[dd];

			Vec<COMP> state;
			for (int ll = 0; ll < COMP; ++ll)
			  state(ll) = pw[ll];

	        Vec<Model::NumDerVar> dervar;
			Model::EvalDerVar(state, sparam, dervar);

			// Assemble right hand side
	        for (int ll = 0; ll < COMP; ++ll)
	          for (int pp = 0; pp < ndof_w; ++pp)
	            rhs_w[pp+ndof_w*ll] += qw * dervar(ll) * phi[pp];

		}

		mygemm('n', 'n', ndof_w, COMP, ndof_w, 1., &ed.inv_mass[0], &rhs_w[0], 0., vecW);

		// Compute gradient of solution and fill vecQ
		if (Model::Diffusion || Model::Source)
		{
			for (int j = 0; j < ed.nip; ++j) {
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
					{
						//cout << "phi " << phi[pp] << endl; 	        
						rhs_q[pp+ndof_w*(dd+D*ll)] += intweight * dsol[dd] * phi[pp];}

      }

      mygemm('n', 'n', ndof_w, COMP * D, ndof_w, 1., &ed.inv_mass[0], &rhs_q[0], 0., vecQ);

		}

	}

	vector<double> rhs_l(COMP * ndof_l_max, 0.);
	for (int i = 0; i < nf; ++i) {
		if (!fadata[i]) continue;
			FacetData<D, COMP> & fd = *fadata[i];
		if (fd.ndof_l == 0) continue;

		int ndof_l = fd.ndof_l;
		int nip    = fd.nip;

		rhs_l.assign(COMP*ndof_l, 0.0);

		double * vecL = &sol_func.vecL[COMP * fd.offset_l];

		for (int j = 0; j < nip; ++j) {
			double intweight = fd.qw[j];
			const double * pw1  = &fd.w1[j * COMP];
			const double * pw2  = &fd.w2[j * COMP];
			const double * plambda  = &fd.lambda[j * COMP];

			const double * mu = &fd.mu[j * ndof_l];

			SpatialParams<D> sparam;
			for (int dd = 0; dd < D; ++dd)
				sparam.pos(dd) = fd.qp[D*j+dd];

			// We take the state as the average of the two states
			Vec<COMP> state;
			for (int ll = 0; ll < COMP; ++ll)
			{
			  // state(ll) = 0.5*(pw1[ll]+pw2[ll]);
  			  state(ll) = plambda[ll];
			}

	        Vec<Model::NumDerVar> dervar;
			Model::EvalDerVar(state, sparam, dervar);

			for (int ll = 0; ll < COMP; ++ll)
				for (int pp = 0; pp < ndof_l; ++pp)
					rhs_l[pp+ndof_l*ll] += intweight * dervar(ll) * mu[pp];
		}

		mygemm('n', 'n', ndof_l, COMP, ndof_l, 1., &fd.inv_mass[0], &rhs_l[0], 0., vecL);
	}

}
