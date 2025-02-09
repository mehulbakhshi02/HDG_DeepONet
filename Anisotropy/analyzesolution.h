/** @file
 * @brief Computes the L2 error of the solution or the error in target functional based on the solution that is there in the
 * global array ed. Should be modified to incorporate the solution at the current
 * state. Should also contain the details of the adjoint based error estimate and
 * error bounds. We decrese the order of the solution to work at the order p
*/
template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::AnalyzeSolution(const Solution &sol, const Solution &adj, LocalHeap & lh) {

	// // // Setup the correct HDG spaces for the new order
	// GetElementInformation(fspacedual, fadata, eldata, ma, order_array, lh);
	// // // Save the const to normal so that it can be reconstructed
	// Solution sol_current = sol;
	// ReconstructSolution(sol_current);

	double q = AnisotropyData::norm;

	// Reconstruct the solution that has been passed into the integration points
    Vec<COMP> state_err = 0.0;
    Mat<COMP, D> grad_err = 0.0;
	Vec<Model::NumVolFunctionals> vol_target_err = 0.0;
	Vec<Model::NumBdryFluxWeight> bdry_target_err = 0.0;

	// hack for cell wise
    // fstream outnodeorder;
    // outnodeorder.open("cell_wise_error_exact.csv", ios::out);
    // outnodeorder.precision(16);
    // outnodeorder.setf(ios::scientific, ios::floatfield);
    // end hack
	// testing
	double tot_vol = 0.0;
	for (int i = 0; i < ne; ++i) 
	{
		// testing
		double loc_vol = 0.0;
		double loc_err = 0.0;
		//testing ends
	    ElementData<D, COMP> & ed = *eldata[i];
        int ndof_w = ed.ndof_w;
		double adj_vol_loc = 0.0;
		for (int j = 0; j < ed.nip; ++j) 
		{
			// Compute the values of solution at integration points
	        const double   qw  = ed.qw[j];// Quadrature weights
	        const double * qp  = &ed.qp[j * D];// Quadrature points
	        const double * pw  = &ed.w[j * COMP];// Numerical solution w
			const double * pq = &ed.q[COMP * D * j];// Gradiend of solution \nabla w

			SpatialParams<D> sparam;
			for (int dd = 0; dd < D; ++dd)
				sparam.pos(dd) = qp[dd];
			// Contains the exact solution if it exists if not then it contains a zero
		    Vec<COMP> w(0.);
		    Mat<COMP, D> q_ex(0.);
			Model::EvalAnalyticSol(w, q_ex, sparam);
	        for (int ll = 0; ll < COMP; ++ll)
			{
				state_err(ll) += qw * pow((pw[ll]-w(ll)), q);// Compute the sum of errors in the state variable
				loc_err += qw * pow(pw[ll]-w(ll), q);//
			}

			if (AnisotropyData::adjoint_based)
			{
				Vec<COMP> state;
				Vec<COMP> state_exact;
				for (int ll = 0; ll < COMP; ++ll)
				{
					state(ll) = pw[ll];
					state_exact(ll) = w(ll);
				}
				Vec<Model::NumVolFunctionals> monitors(0.), monitors_exact(0.);
				Model::EvalVolFunctionals(state, sparam, monitors);
				Model::EvalVolFunctionals(state_exact, sparam, monitors_exact);
				for (int ll = 0; ll < Model::NumVolFunctionals; ++ll)
				{				
					vol_target_err(ll) += qw * (monitors(ll) - monitors_exact(ll));
					adj_vol_loc += qw * (monitors(ll) - monitors_exact(ll));
				}
			}

	        for (int ll = 0; ll < COMP; ++ll)
	        {
				for (int dd = 0; dd < D; ++dd)
				{
					// The gradient solution is shifted by one column because the first column has 
					// the solution itself
					grad_err(ll, dd) += qw * pow(fabs(pq[dd+D*ll]-q_ex(ll, dd)), q);
				}
	        }

	        loc_vol += qw;
	    }// end of loop over integration points
	    tot_vol += loc_vol;
	}// end of loop over elements
	if(AnisotropyData::adjoint_based && Model::NumBdryFluxWeight > 0)
	{
		for (int k = 0; k < nf; ++k) {
	      if (!fadata[k]) continue;
	      FacetData<D, COMP> & fd = *fadata[k];

	      if (fd.ndof_l != 0) continue; // only boundary faces
	      if (fd.bcnr   == -2) continue; // no zero-measure faces
	    
	      for (int j = 0; j < fd.nip; ++j) {

	        const double * pw = &fd.w1[j * COMP];
	        const double * pq = &fd.q1[j * COMP * D];
	        const double * qp = &fd.qp[j * D];
	        const double * pn = &fd.n1[j * D];
	      
	        SpatialParams<D> sparam;
	        Vec<COMP> state;
	        Mat<COMP, D> grad;
	        Vec<COMP> cfn(0.), dfn(0.);
	        // The exact convective and diffusive fluxes
	        Vec<COMP> cfn_ex(0.), dfn_ex(0.);

	        for (int dd = 0; dd < D; ++dd)
	          sparam.pos(dd) = qp[dd];
	        for (int dd = 0; dd < D; ++dd)
	          sparam.normal(dd) = pn[dd];


		    Vec<COMP> w(0.);
		    Mat<COMP, D> q_ex(0.);
			Model::EvalAnalyticSol(w, q_ex, sparam);

	        Mat<Model::NumBdryFluxWeight, COMP> weights;
	        Model::EvalBdryFluxWeight(fd.bcnr, sparam, weights);
	      
	        for (int ll = 0; ll < COMP; ++ll)
			{       
			   state(ll) = pw[ll];
			}

	        if (Model::Convection)
			{	        
				Model::EvalBdryConvFlux(fd.bcnr, state, sparam, cfn);
				Model::EvalBdryConvFlux(fd.bcnr, w, sparam, cfn_ex);
			}
	      
	        if (Model::Diffusion) {
	          for (int ll = 0; ll < COMP; ++ll)
	            for (int dd = 0; dd < D; ++dd)
	              grad(ll, dd) = pq[dd+D*ll];
	          
	          Model::EvalBdryDiffFlux(fd.bcnr, state, grad, sparam, dfn);
	          Model::EvalBdryDiffFlux(fd.bcnr, w, q_ex, sparam, dfn_ex);
	        }

	        bdry_target_err += fd.qw[j] * weights * ((cfn - dfn) - (cfn_ex - dfn_ex));
	      }
	    }//end of loop over faces		
	}
    for (int ll = 0; ll < COMP; ++ll)
    {
    	state_err(ll) = pow(state_err(ll), 1.0/q);
		for (int dd = 0; dd < D; ++dd)
		{
			grad_err(ll, dd) = pow(grad_err(ll,dd), 1.0/q);
		}
    	cout<<"L2 error: "<<setprecision(16)<<state_err(ll)<<endl;
    }

 //    if(AnisotropyData::adjoint_based)
	// {	
	// 	for (int ll = 0; ll < Model::NumVolFunctionals; ++ll)
	// 	{	
	// 		cout<<"Volume target function error: "<<vol_target_err(ll)<<endl;
	// 	}
	// 	for (int ll = 0; ll < Model::NumBdryFluxWeight; ++ll)
	// 	{	
	// 		cout<<"Flux target function error: "<<bdry_target_err(ll)<<endl;
	// 	}
	// }

	// Write adjoint error into file
	if (AnisotropyData::sol_write_error) {
	    for (int ll = 0; ll < COMP; ++ll)
	    {
			err_monitors.push_back(state_err(ll));
			for (int dd = 0; dd < D; ++dd)
			{
				err_monitors.push_back(grad_err(ll, dd));
			}
	    }
		if(AnisotropyData::adjoint_based)
		{	
			for (int ll = 0; ll < Model::NumVolFunctionals; ++ll)
				err_monitors.push_back(vol_target_err(ll));
			for (int ll = 0; ll < Model::NumBdryFluxWeight; ++ll)
				err_monitors.push_back(bdry_target_err(ll));
		
		}
	}
}