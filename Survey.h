template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::GenerateMLData(const Solution & sol, LocalHeap & lh) {

  // Currently hard coded for scalar equations
  // Save the coefficients for the error polynomial
  Solution sol_err;
  ReconstructFunction(sol, sol_err, lh);

  // Resize the ml_eldata vector
  ml_eldata.resize(ne);
  ml_fadata.resize(nf);
  // Resize the ml_outputdata vector
  ml_outputdata.resize(ne);
  // Resize the ml_deploy vector
  ml_deploydata.resize(ne);

  double q_norm = AnisotropyData::norm;


  // Compute the current mesh metric
  MeshMetric<D> metric_implied(ne);
  // Hack for 2D
  ComputeMeshMetric(metric_implied);


// KUNAL: I need the nip to know how many values are missing
// To store the nip values at the faces
int nip_universal;
// A marker to check that the value is acquired
int nip_acquired = 0; 

for (int i = 0; i<nf && nip_acquired == 0; i++)
{
	FacetData<D, COMP> & fd = *fadata[i];
	if (fd.ndof_l != 0)
	{
		nip_universal = fd.nip;
		nip_acquired = 1;
	}
}

cout<<"nip_universal: "<<nip_universal<<endl;



// Loop over faces
  for(int i = 0; i<nf; i++)
  {
    if (!fadata[i]) continue;
    FacetData<D, COMP> & fd = *fadata[i];
    int nip;
    int bc_marker = 0;
    
    if (fd.bcnr == -2) continue; // zero-measure face
    // KUNAL
    // if (fd.ndof_l == 0) continue; // skip boundary faces
    if (fd.ndof_l == 0)
    {
    	nip = nip_universal;
    	bc_marker = 1;
    }
    else
    {
    	nip = fd.nip;
    }
    
    
    // cout<< "bc_marker: " << bc_marker << endl;
    ml_fadata[i] = new ML_FacetData<D,COMP> (fd.facenr, fd.nip, fd.elnr1, fd.elnr2);
    // Take current instance of ML_ElementData
    ML_FacetData<D, COMP> & ml_fd = *ml_fadata[i];
      for (int j = 0; j < nip; ++j) { 
        // w1 and w2 at the quadrature points
        const double * pw1  = &fd.w1[j * COMP];
        const double * pw2  = &fd.w2[j * COMP];
        // q1 and q2 at the quadrature points
        const double * pq_1 = &fd.q1[j * COMP * D];
        const double * pq_2 = &fd.q2[j * COMP * D];
        // Jump in w on quarature points
        double *jump_w = &ml_fd.w_jump[j*COMP];
        // Jump in q on quadrature points
        double *jump_q = &ml_fd.q_jump[j * COMP * D];
	// cout<< "bc_marker: " << endl;
        for(int ll=0;ll<COMP;ll++)
        {
          if (bc_marker == 1)
          {
          	double * pw1  = &fd.lambda[j * COMP];
          	jump_w[ll] = fabs(pw1[ll]);
          	// cout<< "jump_w[ll]" << jump_w[ll] << endl;
  	  }
          else
          {
          	jump_w[ll] = fabs(pw1[ll]-pw2[ll]);
          }
          for(int dd=0;dd<D;dd++)
          {
          	if (bc_marker == 1)
          {
          	jump_q[dd+D*ll] = 0;
          }
          else
          {
          	    jump_q[dd+D*ll] = fabs(pq_1[dd+D*ll]-pq_2[dd+D*ll]);          
          }
          }
        }

      }
  }//loop over face ends
  // cout<< "bc_marker: " << endl;

  // Looping over all elements
  for(int i=0;i<ne;i++)
  {
    // Read in the original data 
    ElementData<D,COMP> & ed = *eldata[i];
    // Element offset in array
    int offset_w = ed.offset_w;
    // ndof for current element
    int ndof_w   = ed.ndof_w;
    int elnf = ed.nf;

    double size = 0.0;
    vector<double> beta(D-1,0.0);
    vector<double> q_met(D*D,0.0);
    metric_implied.GetPrincipal(i, size, beta, q_met);
    double theta = acos(q_met[D*(D-1)]);
      
    // Generate ML_ElementData
    ml_eldata[i] = new ML_ElementData<D,COMP> (i,  elnf, ed.nip, ndof_w);

    // ML deploy data to be initialized
    ml_deploydata[i] = new ML_DeployData<D,COMP> (i,  ed.nip, ndof_w);

    // Take current instance of ML_ElementData
    ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
    // Set variables
    // Hack only for 2D
    ml_elidata.beta = beta[0];
    ml_elidata.theta = theta;
    
    ml_elidata.h = ed.h;
    ml_elidata.surf = ed.surf;
    ml_elidata.vol = ed.vol;

    for (int j = 0; j<elnf; j++)
    {
      ml_elidata.faces[j] = ed.faces[j];
    }

    for (int dd=0;dd<D;dd++)
    {
      ml_elidata.cntr[dd] = ed.cntr[dd];
    }
    ml_elidata.nip = ed.nip;
    // Counting the non-boundary faces of the element
    int nf_bc = 0;
	for(int j = 0 ; j < ml_elidata.nf ; j ++)
	{
		int fcnr = ml_elidata.faces[j];	
	    
	    if (!fadata[fcnr]) continue;
	    FacetData<D, COMP> & fd = *fadata[fcnr];
	    if (fd.bcnr == -2) continue; // zero-measure face
	    if (fd.ndof_l != 0)
	    {
	    	nf_bc++;
	    } 
	    
	}
	ml_elidata.nf_bc = nf_bc;
    // Generating the input data for solution and gradient
    for (int ll = 0; ll < COMP; ++ll)
    {
      for (int pp = 0; pp < ndof_w; ++pp)
      {
        ml_elidata.vecW[pp+ll*ndof_w] = sol.vecW[COMP*offset_w+pp+ll*ndof_w];
        for(int dd = 0; dd < D; ++dd)
        {
          ml_elidata.vecQ[pp+ndof_w*(dd+D*ll)] = sol.vecQ[COMP*ed.offset_q+pp+ndof_w*(dd+D*ll)];  
        }
      }
    }
    
    // Generating the output data
    // Generate ML_ElementData
    // Currently the number of output variables are set by numdervar
    // This will have to be passed as a parameter to the output so that one can set the size of vecW
    ml_outputdata[i] = new ML_OutputData<D,COMP> (i, ed.nip, ndof_w);
     // Take current instance of ML_OutputData
    ML_OutputData<D, COMP> & ml_outdata = *ml_outputdata[i];
    for (int pp = 0; pp < ndof_w; ++pp)
    {
       // KUNAL
       // This needs to be changed
       ml_outdata.vecW[pp] = sol_err.vecW[COMP*offset_w+pp];
    }
    for (int j = 0; j < ed.nip; ++j) {

      const double   qw  = ed.qw[j];
      const double * qp  = &ed.qp[j * D];
      const double * pw  = &ed.w[j * COMP];
      const double * phi = &ed.phi[j * ndof_w];

      double * pw_out  = &ml_outdata.w[j * COMP];

      SpatialParams<D> sparam;
      for (int dd = 0; dd < D; ++dd)
        sparam.pos(dd) = qp[dd];

      Vec<COMP> state;
      for (int ll = 0; ll < COMP; ++ll)
        state(ll) = pw[ll];

      Vec<Model::NumDerVar> dervar;
      Model::EvalDerVar(state, sparam, dervar);

      for (int ll = 0; ll < COMP; ++ll)
        {
          double err = (dervar(ll));// u-uh
          // KUNAL
          // Some values are coming negative here in the variable err
          // cout << "err: " << fabs(err)<< endl;
          err = fabs(err);
          double scal = pow(ml_elidata.h, max_order+1); //h^p+1
          double err_scal = err/scal;
          double err_scal_pow = pow(err, q_norm);
          pw_out[ll] = log(err_scal_pow);
        }
      
    }


  }

  vector<double> tr_err(ne,0.0);
  vector<double> proj_err(ne,0.0);
  vector<double> ml_err(ne, 0.0);
	double err_proj = 0.0;
  double err_u = 0.0;
	// Elemental data
	Mat<COMP, D> grad_err = 0.0;

  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];
    int offset_q = ed.offset_q;
    int offset_w = ed.offset_w;
    int ndof_w   = ed.ndof_w;
    int nip      = ed.nip;

    //  // Take current instance of ML_OutputData
    ML_OutputData<D, COMP> & ml_outdata = *ml_outputdata[i];
    vector<double> w_err, w, q;

    w.resize(nip*COMP);
    w_err.resize(nip*COMP);
    mygemm('t', 'n', COMP, nip, ndof_w, 1., &ml_outdata.vecW[0], &ed.phi[0], 0., &w_err[0]);

    q.resize(nip*COMP*D);
    ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
    mygemm('t', 'n', COMP, nip, ndof_w, 1., &ml_elidata.vecW[0], &ed.phi[0], 0., &w[0]);


    mygemm('t', 'n', COMP * D, nip, ndof_w, 1., &ml_elidata.vecQ[0], &ed.phi[0], 0., &q[0]);

    double loc_err_proj = 0.0;
    double loc_u_err = 0.0;
    for (int j = 0; j < ed.nip; ++j) {

		const double   qw  = ed.qw[j];
		const double * qp  = &ed.qp[j * D];
		const double * pw_err  = &w_err[j * COMP];

    const double * pw  = &w[j * COMP];

		const double * pq = &q[COMP * D * j];// Gradiend of solution \nabla w

		SpatialParams<D> sparam;
		for (int dd = 0; dd < D; ++dd)
			sparam.pos(dd) = qp[dd];
		// Contains the exact solution if it exists if not then it contains a zero
	 Vec<COMP> w(0.);			
		Mat<COMP, D> q_ex(0.);
		Model::EvalAnalyticSol(w, q_ex, sparam);

		for (int ll = 0; ll < COMP; ++ll)
		{
      loc_u_err += qw * pow((pw[ll]-w(ll)), q_norm);
			for (int dd = 0; dd < D; ++dd)
			{
				// The gradient solution is shifted by one column because the first column has 
				// the solution itself
				// cout<<pq[dd+D*ll]<<" "<<q_ex(ll, dd)<<endl;
				grad_err(ll, dd) += qw * pow(fabs(pq[dd+D*ll]-q_ex(ll, dd)), q_norm);
			}
		}

      // At each quadrature point we have pw[0] = u-u_h
      loc_err_proj += (pw_err[0]*pw_err[0])*qw;
    }
    
    // KUNAL
    // Here only the internal elements are printed
    // Taking this out of the if statement
    /*
    if(ml_elidata.nf==ml_elidata.nf_bc)
    {
      double scal = pow(ml_elidata.h, q_norm*(max_order+1)); //h^r*(p+1)
      double err_scal = loc_u_err/scal;
      // cout<<loc_u_err<<" "<<log(loc_u_err)<<endl;
      ml_outdata.error = log(err_scal);
      err_proj += loc_err_proj;
      err_u += loc_u_err;
      tr_err[i] = loc_u_err;
      proj_err[i] = loc_err_proj;
    }
    */
    double scal = pow(ml_elidata.h, q_norm*(max_order+1)); //h^r*(p+1)
      double err_scal = loc_u_err/scal;
      // cout<<loc_u_err<<" "<<log(loc_u_err)<<endl;
      ml_outdata.error = log(err_scal);
      err_proj += loc_err_proj;
      err_u += loc_u_err;
      tr_err[i] = loc_u_err;
      proj_err[i] = loc_err_proj;
    
    
  }
  err_proj = pow(err_proj, 0.5);
  err_u = pow(err_u, 0.5);


double jump_err_u = 0;
// facet data
Mat<COMP, D> jump_grad_err = 0.0;
// Loop over faces
  for(int i = 0; i<nf; i++)
  {
    if (!fadata[i]) continue;
    FacetData<D, COMP> & fd = *fadata[i];
    int bc_marker = 0;
    // KUNAL
    // NO effect
    // if (fd.ndof_l == 0) continue; // skip boundary faces
    if (fd.ndof_l == 0)
    {
    	bc_marker = 1;
    }
    else
    {
    	bc_marker = 0;
    }
    
    if (fd.bcnr == -2) continue; // zero-measure face
    int nip = fd.nip;

    ML_FacetData<D, COMP> & ml_fd = *ml_fadata[i];

      for (int j = 0; j < nip; ++j) {
        double intweight = fd.qw[j];
        // Jump in w on quarature points
        double *jump_w = &ml_fd.w_jump[j*COMP];
        // Jump in q on quadrature points
        double *jump_q = &ml_fd.q_jump[j * COMP * D];

        for(int ll=0;ll<COMP;ll++)
        {
          jump_err_u += intweight * pow(jump_w[ll], q_norm);
          for(int dd=0;dd<D;dd++)
          {
              jump_grad_err(ll, dd) += pow(jump_q[dd+D*ll], q_norm);
          }
        }

      }
  }//loop over face ends

  jump_err_u = pow(jump_err_u, 1.0/q_norm);
  for (int ll = 0; ll < COMP; ++ll)
  {
    for (int dd = 0; dd < D; ++dd)
    {
      grad_err(ll, dd) = pow(grad_err(ll,dd), 1.0/q_norm);
      jump_grad_err(ll, dd) = pow(jump_grad_err(ll, dd), 1.0/q_norm);
    }
  }
  
  if(MLParameters::train)
    WriteMLData();
  if(MLParameters::deploy)
    DeployML();
  // KUNAL
  // Data file for ML is written before this
  
  double err_dep = 0.0;

  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];
    int offset_q = ed.offset_q;
    int offset_w = ed.offset_w;
    int ndof_w   = ed.ndof_w;
    int nip      = ed.nip;

    //  // Take current instance of ML_DeployML
    ML_DeployData<D, COMP> & ml_depldata = *ml_deploydata[i];
    // If learning from quadrature nodes directly
    // vector<double> w_err;

    // w_err.resize(nip*COMP);
    // mygemm('t', 'n', COMP, nip, ndof_w, 1., &ml_depldata.vecW[0], &ed.phi[0], 0., &w_err[0]);

    double loc_err = 0.0;
    for (int j = 0; j < ed.nip; ++j) {

    const double   qw  = ed.qw[j];
    // const double * pw_err  = &w_err[j * COMP];
    const double * pw_err  = &ml_depldata.w[j * COMP];
      
    // cout<<i<<" "<<j<<" "<<pw_err[0]<<endl;
    // At each quadrature point we have pw[0] = u-u_h from the deployed ML
    loc_err += (pw_err[0]*pw_err[0])*qw;
  }
    // loc_err = ml_depldata.error;
    err_dep += loc_err;
    ml_err[i] = loc_err;
  }
  err_dep = pow(err_dep, 0.5);


  stringstream err_oss1(" ");
  err_oss1 << "cell_wise_ml_err.csv";
  ofstream err_log1(err_oss1.str().c_str(), ios::app);
  err_log1.precision(16);
  err_log1.setf(ios::scientific, ios::floatfield);
  
  for(int i=0;i<ne;i++)
  {
    // cout<<"Number of elements: "<<ne<<endl;
    // cout<<"ML error for the element: "<<i<<" : "<<ml_err[i]<<endl;
    
    ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
    // KUNAL
    // Comment this to print the ML Predicted errors
    // if(ml_elidata.nf!=ml_elidata.nf_bc) continue;

    err_log1<<proj_err[i]<<","<<tr_err[i]<<","<<ml_err[i]<<endl;
  }
  // KUNAL
  // err_log1.close();
  stringstream err_oss(" ");
  err_oss << "ml_err_data.csv";
  
  ofstream err_log(err_oss.str().c_str(), ios::app);
  err_log.precision(16);
  err_log.setf(ios::scientific, ios::floatfield);
  
  // err_log << ne << "," << ndof_l_solve << "," << err <<","<<err_u<< "," <<grad_err(0, 0)<< "," << grad_err(0, 1) << "," << jump_err_u << "," << jump_grad_err(0,0) << "," << jump_grad_err(0,1) << "," << err_dep <<endl;
  err_log << ne << "," << ndof_l_solve << "," << err_proj <<","<<err_u<< "," << err_dep <<endl;
  // KUNAL
  // err_log.close();
}
