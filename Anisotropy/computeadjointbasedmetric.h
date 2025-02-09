/** @file
 * @brief This function takes the adjoint, \c adj, the array of anisotropies
 * and reconciles them to one final anisotropy called aniso_sol
 * @param[in] - adj - Contains the data of type Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[in] - add_order - The order to be added to order_array. Size = 1 int
 * @param[in] - aniso_sol_trial - Local anisotropy for all components and their gradients. Size = COMP*(D+1)*D*(D+1)/2*ne double
 * @param[out] - aniso_sol - Contains the reconciled anisotropy based on the adjoint gradients and so on. Size = D*(D+1)/2*ne double
*/

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeAdjointBasedMetric(const Solution & adj, const int add_order, vector< SolAniso<D> > & aniso_sol_vec, SolAniso<D> &aniso_sol){
	cout << string(32, '*') << endl;
	cout << "Compute continuous adjoint based estimates... " << endl;
	cout << string(32, '*') << endl;

	int no_aniso = COMP;
	if (Model::Diffusion || Model::Source)
	    no_aniso = COMP * (D+1);
		
	// Compute the current mesh metric
	MeshMetric<D> metric_implied(ne);
	if(!AnisotropyData::use_intersection)
	{	// Hack for 2D
		ComputeMeshMetric(metric_implied);
	}
	// double err_nume_adj = 0.0;
	double x_const = 1e20;
	for(int i = 0; i < ne; ++i)
	{
	    ElementData<D, COMP> & ed = *eldata[i];
		int offset_q = ed.offset_q;
		int offset_w = ed.offset_w;
		int ndof_w   = ed.ndof_w;
		int nip      = ed.nip;
		//Reconstruct the gradient of the adjoint
		// The adjoint gradient in each cell computed for each integration point
		Solution adj_temp = adj;
	    double ed_adj[nip * COMP * D];
	    memset( ed_adj, 0, nip*COMP*D*sizeof(double) );
		if (Model::Diffusion || Model::Source)
		{		
			mygemm('t', 'n', COMP * D, nip, ndof_w, 1., &adj_temp.vecQ[COMP*offset_q], &ed.phi[0], 0., &ed_adj[0]);
		}
		else 
		{
			// Should be rewritten with gemm (might not be possible in the requested order)
			for (int j = 0; j < nip; ++j) 
			{
				double * dphi = &ed.dphi[j * ndof_w * D];
				double * pq   = &ed_adj[j * COMP * D];

				for (int ll = 0; ll < COMP; ++ll)
					for (int dd = 0; dd < D; ++dd) 
					{
						double val = 0.;
						for (int pp = 0; pp < ndof_w; ++pp)
							val += adj.vecW[COMP*offset_w+pp+ndof_w*ll] * dphi[pp+ndof_w*dd];
						pq[dd+D*ll] = val;
					}
			}
		}
		// Check whether the gradient plays a role
		const int dsize = (Model::Diffusion || Model::Source) ? D+1 : 1;

		//Reconstruct the gradient of the solution for the diffusion part
		//Compute the derivative of the convective and diffusive flux
		vector<double> local_consts(no_aniso, 0.0);
		for (int j = 0; j < nip; ++j)
		{
			// Solution w
			const double * pw  = &ed.w[j * COMP];
			// Gradient q of the primal
            const double * pq = &ed.q[j * COMP * D];
            // Gradient of the adjoint
			double * padj   = &ed_adj[j * COMP * D];

			// Quadrature weights
			const double   qw  = ed.qw[j];
            // Quadrature points
			const double * qp  = &ed.qp[j * D];
			SpatialParams<D> sparam;
		    // Fill helper variables
	        for (int dd = 0; dd < D; ++dd)
	          sparam.pos(dd) = qp[dd];

			// Numerical solutions, gradients and fluxes filled into AD variables
			Vec<COMP, AutoDiff<COMP*dsize> > ad_state(0.);
			Mat<COMP, D, AutoDiff<COMP*dsize> > ad_grad(0.), ad_fc(0.), ad_fv(0.), ad_f(0.);

		    // Vec<COMP> w_h(0.);
		    // Mat<COMP, D> q_h(0.), fc_h(0.), fv_h(0.), f_h(0.);
		    // Filling the numerical solutions and gradients
			for (int ll = 0; ll < COMP; ++ll)
		        ad_state(ll) = AutoDiff<COMP*dsize> (pw[ll], ll);  
	        if (Model::Diffusion)
	          for (int ll = 0; ll < COMP; ++ll)
	            for (int dd = 0; dd < D; ++dd)
	              ad_grad(ll, dd) = AutoDiff<COMP*dsize> (pq[dd+D*ll], ll+COMP*(dd+1));

			if(Model::Convection)
			{
				//Computing the numerical convective flux
		        Model::EvalConvFlux(ad_state, sparam, ad_fc);
		        for(int mm = 0; mm < COMP; ++mm)// loop for each of the local_consts
		        {
					for(int dd = 0; dd < D; ++dd)// loop over each dimension for the flux derivative
					{
						for(int ll = 0; ll < COMP; ++ll)// loop for each component of the flux derivative
						{
							double adj_der = padj[dd+D*ll];

							double flux_der = ad_fc(ll, dd).DValue(mm);
							int index = dsize * mm; // The order to store local constants is u1, u1x, u1y, u1z, u2, u2x, u2y, u2z, u3 and so on
							local_consts[index] += adj_der * flux_der * qw;
						}
					}		        	
		        }

			}

			if(Model::Diffusion)
			{
				// Computing the numerical diffusive flux
		        Model::EvalDiffFlux(ad_state, ad_grad, sparam, ad_fv);
		        for(int mm = 0; mm < COMP; ++mm)// loop for each of the local_consts
		        {
					for(int dd = 0; dd < D; ++dd)// loop over each dimension for the flux derivative
					{
						for(int ll = 0; ll < COMP; ++ll)// loop for each component of the flux derivative
						{
							double adj_der = padj[dd+D*ll];
							double flux_der = ad_fv(ll, dd).DValue(mm);
							int index = dsize * mm; // The order to store local constants is u1, u1x, u1y, u1x, u2, u2x, u2y, u2z, u3 and so on
							local_consts[index] += adj_der * flux_der * qw;
						}
					}		        	
		        }
                for (int mm = 0; mm < COMP; ++mm)
				{
					for (int dd2 = 0; dd2 < D; ++dd2)
					{
			            for (int ll = 0; ll < COMP; ++ll)
		                {
							for (int dd = 0; dd < D; ++dd)
							{
								double adj_der = padj[dd+D*ll];// The derivative of the llth component against the ddth coordinate
								double flux_der = ad_fv(ll, dd).DValue(mm+COMP*(dd2+1));// We want to compute the derivative of the llth components, ddth coordinate of the diffusive flux against the mmth components, dd2th derivative 
								int index = dsize * mm + dd2 + 1; 
								local_consts[index] += adj_der * flux_der * qw;
							}
						}
					}
				}
			}


		}// end of loop over integration points

		int p = order_array[i] + add_order;//Computes the order of the original solution
		double c_p1 = (double)D/((double)p+1.0+(double)D);

		double total_weight = 0.0;
		vector<double> local_weight(no_aniso, 0.0);
		// We compute the total_weight that we need to divide to get local weights which is the size
		SolAniso<D> sol_aniso_local(no_aniso);// This contains the anisotropies for the cell i but all components
		if(AnisotropyData::use_intersection)
		{
			for(int dd2 = 0; dd2 < no_aniso; ++dd2)
			{
				vector<double> aniso_loc(D*(D+1), 0.0);
				aniso_sol_vec[dd2].GetComponents(i, aniso_loc);
				sol_aniso_local.SetComponents(dd2, aniso_loc);// We set the anisotropic components for the cell i only
				double A1 = aniso_loc[0];
				local_weight[dd2] = A1*fabs(local_consts[dd2])*c_p1;
				total_weight += local_weight[dd2];
			}
			// We consider total weight/local weight because if the local weight is high then we want the ellipse to be
			// as small as possible because intersect the metrics
			for(int dd2 = 0; dd2 < no_aniso; ++dd2)
			{
				local_weight[dd2] = total_weight/local_weight[dd2];
			}
			MeshMetric<D> cell_metric(no_aniso);
			// Takes the anisotropy of the solutions and gradients
			// Gives out the metric for each component i.e, the metric that optimizes that component
			ComputeSolutionMetric(sol_aniso_local, local_weight, p, cell_metric);
			vector<double> aniso_loc(D*(D+1), 0.0);
			IntersectMetric(cell_metric, total_weight, p, aniso_loc);
			aniso_sol.SetComponents(i, aniso_loc);
		}
		else
		{
			for(int dd2 = 0; dd2 < no_aniso; ++dd2)
			{
				vector<double> aniso_loc(D*(D+1), 0.0);
				aniso_sol_vec[dd2].GetComponents(i, aniso_loc);
				sol_aniso_local.SetComponents(dd2, aniso_loc);// We set the anisotropic components for the cell i only
				local_weight[dd2] = fabs(local_consts[dd2])*c_p1;
				total_weight += local_weight[dd2];
			}
			// // We consider total weight/local weight because if the local weight is high then we want the ellipse to be
			// // as small as possible because intersect the metrics
			// for(int dd2 = 0; dd2 < no_aniso; ++dd2)
			// {
			// 	local_weight[dd2] = total_weight/local_weight[dd2];
			// }
			double size = 0.0;
			vector<double> beta(D-1,0.0);
			vector<double> q_met(D*D,0.0);
			metric_implied.GetPrincipal(i, size, beta, q_met);
			double lambda = sqrt(size);	
			// Should be checked if this gives consistent results for all quadrants as we have aniso_sol_u to be between
			// 0 and PI
			double sigma = sqrt(beta[0]);
		    double theta = acos(q_met[D*(D-1)]);
			vector<double> aniso_loc(D*(D+1), 0.0);
			ComputeOptimumAnisotropy(sol_aniso_local, local_weight, p, lambda, sigma, theta, aniso_loc);
			aniso_sol.SetComponents(i, aniso_loc);
		}
	}// end of loop over elements

	if(AnisotropyData::regularize_anisotropy && !AnisotropyData::hp && !AnisotropyData::use_aposteriori)
		RegularizeAnisotropy(aniso_sol);
}