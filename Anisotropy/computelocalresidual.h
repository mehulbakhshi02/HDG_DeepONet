/** @file 
 * @brief This function calculates the cell wise residual and decomposes them like how we want for Prof. Dolejsi new way of doing.
 * @param[in] - i - Contains the index of the cell. Size = 1 int
 * @param[out] - residual - Contains the residuals for each triangle. Domain, boundary flux and domain values is the ordering. Size = ne x 3 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeLocalResidual(Solution & sol, const int i, vector<double> &residual) {

	ElementData<D, COMP> & ed = *eldata[i];
	int ndof_q  = ed.ndof_q;
	int ndof_w  = ed.ndof_w;
	int ndof_lt = ed.ndof_lt;
    int offset_w = ed.offset_w;
	int nip     = ed.nip;

	// Check whether the gradient plays a role
	const int dsize = (Model::Diffusion || Model::Source) ? D+1 : 1;

	// Shock-Capturing
	double p = max(1, ed.order);
	double hh = ed.h / p; //ed.nf * ed.vol / (ed.surf * p); //ed.h / p;
	double eps0 = eps_art_visc * pow(hh, 2. - 0.0) ;
	double eps_mean = 0.0;

	if (shock_capturing > 0 && ed.order > 0) 
	{

		deps.assign(COMP * ndof_w * dsize, 0.);

		Vec<COMP, AutoDiff<COMP*(D+1)> > ad_state;
		Mat<COMP, D, AutoDiff<COMP*(D+1)> > ad_grad;
		AutoDiff<COMP*(D+1)> ad_eps;

		for (int j = 0; j < nip; ++j) 
		{

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

			for (int ll = 0; ll < COMP; ++ll) 
			{
				for (int pp = 0; pp < ndof_w; ++pp) 
				{
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
	vector<double> tmp1(COMP, 0.0);
	// // Volume contributions
	for (int j = 0; j < nip; ++j) 
	{
		Vec<COMP, AutoDiff<COMP*dsize> > ad_state(0.), ad_sf(0.);
		Mat<COMP, D, AutoDiff<COMP*dsize> > ad_grad(0.), ad_fc(0.), ad_fv(0.), ad_f(0.);

		// Hack for scalar 
		Vec<Model::NumVolFunctionals, AutoDiff<COMP*dsize> > ad_funct(0.);
		Vec<COMP> weight(0.);

		const double intweights = ed.qw[j];

		const double * phi  = &ed.phi[j * ndof_w];
		const double * dphi = &ed.dphi[j * ndof_w * D];
		const double * pq   = &ed.q[j * COMP * D];
		const double * pw   = &ed.w[j * COMP];
		const double * ppos = &ed.qp[j * D];


		SpatialParams<D> sparam(ppos, ed.wd[j]);
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

		// Hack for now need to check if this is correct
		// Computing the gradient of the solution w and gradients sigma with respect to space variables
		Vec<COMP, AutoDiff<D> > ad_w(0.);
		Mat<COMP, D, AutoDiff<D> > ad_q(0.);
		// Fill helper variables
		for (int ll = 0; ll < COMP; ++ll)
			ad_w(ll).Value() = pw[ll];
        
		if (Model::Diffusion || Model::Source)
			for (int ll = 0; ll < COMP; ++ll)
				for (int dd = 0; dd < D; ++dd)
					ad_q(ll, dd).Value() = pq[dd+D*ll];
		else if (shock_capturing > 0)
			for (int ll = 0; ll < COMP; ++ll)
				for (int dd = 0; dd < D; ++dd)
					ad_q(ll, dd).Value() = pq[dd+D*ll];

        for (int ll = 0; ll < COMP; ++ll)
        {
			for (int dd = 0; dd < D; ++dd) 
			{
				double val = 0.;
				double dval[D] = {0.};
				for (int pp = 0; pp < ndof_w; ++pp)
				{				
					val += sol.vecW[COMP*offset_w+pp+ndof_w*ll] * dphi[pp+ndof_w*dd];
					for(int dd2 = 0; dd2 < D; ++dd2)
					{
						dval[dd2] += sol.vecQ[COMP*ed.offset_q+pp+ndof_w*(dd2+D*ll)] * dphi[pp+ndof_w*dd];
					}
				}
				ad_w(ll).DValue(dd) = val;
				ad_q(ll, dd).DValue(dd) = dval[dd];
			}
		}

		// hack for exact solution   
		// Vec<COMP> w_ex(0.);
		// Mat<COMP, D> q_ex(0.);
		// Model::EvalAnalyticSol(w_ex, q_ex, sparam);
		// for (int ll = 0; ll < COMP; ++ll)
		// 	ad_state(ll) = AutoDiff<COMP*dsize> (w_ex(ll), ll);

		// if (Model::Diffusion || Model::Source)
		// for (int ll = 0; ll < COMP; ++ll)
		// 	for (int dd = 0; dd < D; ++dd)
		// 		ad_grad(ll, dd) = AutoDiff<COMP*dsize> (q_ex(ll,dd), ll+COMP*(dd+1));
		// // Computing the gradient of the solution w and gradients sigma with respect to space variables
		// Vec<COMP, AutoDiff<D> > ad_w(0.);
		// Mat<COMP, D, AutoDiff<D> > ad_q(0.);
		// // Fill helper variables
		// for (int ll = 0; ll < COMP; ++ll)
		// 	ad_w(ll).Value() = w_ex(ll);
        
		// if (Model::Diffusion || Model::Source)
		// 	for (int ll = 0; ll < COMP; ++ll)
		// 		for (int dd = 0; dd < D; ++dd)
		// 			ad_q(ll, dd).Value() = q_ex(ll,dd);

  // //       for (int ll = 0; ll < COMP; ++ll)
  // //       {
		// // 	for (int dd = 0; dd < D; ++dd) 
		// // 	{
		// // 		ad_w(ll).DValue(dd) = q_ex(ll,dd);
		// // 	}
		// // hack for exact solution
		// double x = sparam.pos(0);
		// double y = sparam.pos(1);
		// double eps = Poisson<D>::epsilon;
		// // derivative of w wrt x
		// ad_w(0).DValue(0) = -(exp(x/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(y-(exp(y/eps)-1.0)/(exp(1.0/eps)-1.0));
		// // derivative of w wrt y
		// ad_w(0).DValue(1) = -(exp(y/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(x-(exp(x/eps)-1.0)/(exp(1.0/eps)-1.0));
		// // Derivative of qx wrt x
		// ad_q(0,0).DValue(0) = -(1.0/(eps*eps)*exp(x/eps)*(y-(exp(y/eps)-1.0)/(exp(1.0/eps)-1.0)))/(exp(1.0/eps)-1.0);
		// // Derivative of qx wrt y
		// ad_q(0,0).DValue(1) = (exp(x/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(exp(y/eps)/(eps*(exp(1.0/eps)-1.0))-1.0);;
		// // Derivative of qy wrt x
		// ad_q(0,1).DValue(0) = (exp(x/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(exp(y/eps)/(eps*(exp(1.0/eps)-1.0))-1.0);;
		// // Derivative of qy wrt y
		// ad_q(0,1).DValue(1) = -(1.0/(eps*eps)*exp(y/eps)*(x-(exp(x/eps)-1.0)/(exp(1.0/eps)-1.0)))/(exp(1.0/eps)-1.0);
		// // hack for exact solution
		

		// If primal computation then we have source
		if (Model::Source && AnisotropyData::residual_mode == 0)
			Model::EvalSource(ad_state, ad_grad, sparam, ad_sf);

		// If dual computation then we have the weight
		if(Model::NumVolFunctionals > 0 && AnisotropyData::residual_mode == 1)
		{			
			Model::EvalVolFunctionals(ad_state, sparam, ad_funct);
			// hack for scalar
			// We compute the product of solution and weights in the volume instead of weights 
			weight(0) = ad_funct(0).Value()/ad_state(0).Value();
		}

		vector<double> tmp_f(COMP, 0.0);
		if (ed.order > 0) 
		{
			// Evaluate fluxes
			if (Model::Convection)
				Model::EvalConvFlux(ad_state, sparam, ad_fc);
			if (Model::Diffusion)
				Model::EvalDiffFlux(ad_state, ad_grad, sparam, ad_fv);
			// Derivative with respect to the state variables
			if(AnisotropyData::residual_mode == 0)
				ad_f = ad_fc - ad_fv;
			// hack for scalar, linear, equations with 0 divergence in the convective field
			if(AnisotropyData::residual_mode == 1)
				ad_f = -ad_fc - ad_fv;
			// Hack for scalar

			for(int ll = 0; ll < COMP; ++ll)
			{
				double val_f = 0.0;
				for(int dd = 0; dd < D; ++dd)
				{
					for(int mm = 0; mm < COMP; ++mm)
						val_f += ad_f(ll,dd).DValue(mm)*ad_w(mm).DValue(dd);
				}
				tmp_f[ll] = val_f;
			}
			for(int ll = 0; ll < COMP; ++ll)
			{
				double val_f = 0.0;
				for(int dd = 0; dd < D; ++dd)
				{
					for(int mm = 0; mm < COMP; ++mm)	
						for(int dd2 = 0; dd2 < D; ++dd2)
							val_f += ad_f(ll,dd).DValue(mm+COMP*(dd2+1))*ad_q(mm, dd2).DValue(dd);
				}
				tmp_f[ll] += val_f;
			}
			// Hack for scalar even though it seems like it is written for vectors I have not checked this to 
			// work for vectors
			// 
			// Hack I am not sure how this shock capturing is supposed to work
			// if (shock_capturing > 0)
			// 	f -= eps_mean * q;
		}

		vector<double> tmp_s(COMP,0.0);
	    if (Model::Source && AnisotropyData::residual_mode == 0) 
	    {
			for (int ll = 0; ll < COMP; ++ll)
				tmp_s[ll] = ad_sf(ll).Value();
	    }
	    // hack for scalar
	    if (Model::NumVolFunctionals > 0 && AnisotropyData::residual_mode == 1) 
	    {
				tmp_s[0] = weight(0);
	    }
	 //    if(fabs(tmp_s[0]-tmp_f[0]) > 1e-13)
		// {	    
		// 	cout<<AnisotropyData::residual_mode<<" "<< fabs(tmp_s[0]-tmp_f[0])<<endl;
		// }
		for (int ll = 0; ll < COMP; ++ll)
			tmp1[ll] += intweights * fabs(tmp_s[ll] - tmp_f[ll]);

	}// end of loop over integration points

	#ifdef BR2
	int offset_br2 = 0;
	#endif

	vector<double> tmp2(COMP, 0.0);	
	vector<double> tmp3(COMP, 0.0);
	 // Facet contributions
	for (int ff = 0; ff < ed.nf; ++ff) 
	{
		FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];
		if (fd.bcnr == -2) continue; // zero-measure face
		// On which side of the facet are we?
		bool side = ed.offset_w == fd.offset_w1;

		int ndof_l = fd.ndof_l;
		int nip    = fd.nip;
		int width_br2 = COMP * D * (1 + COMP * (ndof_w + ndof_l));
	    for (int j = 0; j < nip; ++j) 
	    {
			Vec<COMP> lambda(0.), w(0.), fcn(0.), fvn(0.), fn(0.), fp(0.);//fp is the analytical flux evaluated using the solution on the inside
			Mat<COMP, D> q(0.), fcl, fvl, f;
			const double intweights = fd.qw[j];

			const double * pq   = side ? &fd.q1[j * D * COMP] : &fd.q2[j * D * COMP];
			const double * pw   = side ? &fd.w1[j * COMP] : &fd.w2[j * COMP];
			const double * pl   = &fd.lambda[j * COMP];
			const double * pn   = side ? &fd.n1[j * D] : &fd.n2[j * D];
			const double * ppos = &fd.qp[j * D];

			SpatialParams<D> sparam(ppos, pn);      
			for (int ll = 0; ll < COMP; ++ll)
			lambda(ll) = pl[ll];
			for (int ll = 0; ll < COMP; ++ll)
				w(ll) = pw[ll];
	
			if (Model::Diffusion)
			for (int ll = 0; ll < COMP; ++ll)
			  for (int dd = 0; dd < D; ++dd)
			    q(ll, dd) = pq[dd+D*ll];

			if (ndof_l == 0) 
			{
				min_y = ed.vol / fd.h;

				if (Model::Convection && AnisotropyData::residual_mode == 0)
				  Model::EvalBdryConvFlux(fd.bcnr, w, sparam, fcn);
				// hack needs to be checked
				if (Model::Convection && AnisotropyData::residual_mode == 1)
				{
			        Mat<Model::NumBdryFluxWeight, COMP> weights(0.);
			        Model::EvalBdryFluxWeight(fd.bcnr, sparam, weights);
			        Vec<COMP> w_bdry(0.);
			        w_bdry(0) = -weights(0,0);// Hack for scalar, dirichlet boundary only
					Model::EvalBdryConvFlux(fd.bcnr, w_bdry, sparam, fcn);
				}
				// hack needs to be checked if this even makes sense
				if (Model::Diffusion)
				  Model::EvalBdryDiffFlux(fd.bcnr, w, q, sparam, fvn);

				if (Model::Diffusion || Model::Source)
				  for (int ll = 0; ll < COMP; ++ll)
				    lambda(ll) = fd.w2[ll+COMP*j];

			} else {

				if (Model::Convection)
				  NumConvFlux(lambda, w, sparam, fcn);
				if (Model::Diffusion)
				  NumViscFlux(lambda, q, w, q, sparam, fvn);

			}

			#ifdef BR2
			if (Model::Diffusion) 
			{
				// Augment viscous flux with lifted flux
				const double * tmp_lift = &temp_lift2[offset_br2 + j * width_br2];
				for (int ll = 0; ll < COMP; ++ll)
				  for (int dd = 0; dd < D; ++dd)
				    fvn(ll) += tmp_lift[dd+D*ll] * sparam.normal(dd);
			}
			#endif
			if(AnisotropyData::residual_mode == 0)			
				fn = fcn - fvn;
			if(AnisotropyData::residual_mode == 1)
				fn = - fcn - fvn;
			// Evaluate fluxes
			if (Model::Convection)
				Model::EvalConvFlux(w, sparam, fcl);
			if (Model::Diffusion)
				Model::EvalDiffFlux(w, q, sparam, fvl);
			if(AnisotropyData::residual_mode == 0)
				f = fcl - fvl;
			// hack for scalar, linear, convective divergence free case
			if(AnisotropyData::residual_mode == 1)
				f = - fcl - fvl;
			fp = f * sparam.normal;

			for (int ll = 0; ll < COMP; ++ll)
			{				
				tmp2[ll] += intweights * fabs(fn(ll)-fp(ll));
				tmp3[ll] += intweights * fabs(lambda(ll)-w(ll));
			}

		}// end of loop over integration points
	}// end of loop over faces

	double loc_norm = 1.0;
	for (int ll = 0; ll < COMP; ++ll)
	{				
		residual[0] += pow(tmp1[ll], loc_norm);
		residual[1] += pow(tmp2[ll], loc_norm);
		residual[2] += pow(tmp3[ll], loc_norm);
	}

	residual[0] = pow(residual[0], 1.0/loc_norm);
	residual[1] = pow(residual[1], 1.0/loc_norm);
	residual[2] = pow(residual[2], 1.0/loc_norm);
}

