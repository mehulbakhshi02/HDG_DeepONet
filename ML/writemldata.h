/** @file
 * @brief Computes the L2 error of the solution or the error in target functional based on the solution that is there in the
 * global array ed. Should be modified to incorporate the solution at the current
 * state. Should also contain the details of the adjoint based error estimate and
 * error bounds. We decrese the order of the solution to work at the order p
*/

template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::WriteMLData(const Solution & new_sol, LocalHeap & lh, vector<double> &u_coordinates, vector<double> &u_sol) {

	Solution adj;
	vector<double> error(ne);
	int its;
	vector<double> adj_coordinates;
	vector<double> adj_sol;

	if(AnisotropyData::adjoint_based)
  	{
		SolveAdjointSystem(new_sol, adj, error, its, adj_coordinates, adj_sol, lh);
	}
	
	// Writing the error monitors
	stringstream err_oss(" ");
    // oss << "solution-" << ne << "-" << p;
	err_oss << "mldata-" << ne << "-" << max_order;
    Model::GetFilename(err_oss);
  	err_oss << ".txt";
	ofstream err_log(err_oss.str().c_str(), ios::app);
	err_log.precision(16);
	int ne_int = 0;
	for(int i=0;i<ne;i++)
	{
		ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
		
		// KUNAL
		//if(ml_elidata.nf==ml_elidata.nf_bc)
				ne_int++;
	}
	
	Model::GetFilename(err_oss);
	Vec<Model::NumParam> param;
	Model::GetParameters(param);
	// err_log << "starting line 1" << endl;
	err_log << ne_int << "," << D << "," << COMP <<endl;
	
	// KUNAL: I need the nip to know how many values are missing
	// To store the nip values at the faces
	int nip_universal;
	// A marker to check that the value is acquired
	int nip_acquired = 0; 
	// To debug the smallest 2 elements case
	// As a 6 faces are there in total for 2 triangular element
	int ctr = 0;
	for(int i = 0; i < ne && nip_acquired == 0; i++)
	{
		ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
		ElementData<D, COMP> & elidata = *eldata[i];
		for(int j = 0 ; j < ml_elidata.nf ; j++)
		{
			int fcnr = ml_elidata.faces[j];
			// vect[ctr] = fcnr;
			// int fcnr = elidata.faces[j];
			if(ml_elidata.nf==ml_elidata.nf_bc)
			{
				ML_FacetData<D, COMP> & ml_fd_1 = *ml_fadata[fcnr];
				nip_universal = ml_fd_1.nip;
				nip_acquired = 1; // Value acquired
			}
			// ctr = ctr + 1;
			// To fix the 2 elements case
			
		}
	}
	
	// To fix the 2 elements case
	if (ne == 2)
	{
		vector<int> vect(6);
		for(int i = 0; i < ne && nip_acquired == 0; i++)
		{
			ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
			for(int j = 0 ; j < ml_elidata.nf ; j++)
			{
				int fcnr = ml_elidata.faces[j];
				vect[ctr] = fcnr;
				ctr = ctr + 1;			
			}
		}
		// Check the first element
		for (int i = 0;i < 3;i++)
		{
			// Check the second element
			for (int j = 3;j < vect.size();j++)
			{
				if (vect[i] == vect[j])
				{
					int fcnr = vect[i];
					cout<< fcnr<<endl;
					ML_FacetData<D, COMP> & ml_fd_1 = *ml_fadata[fcnr];
					nip_universal = ml_fd_1.nip;
					nip_acquired = 1;
				}
			}
		}
	}

	//Solution coordinate transformation

	int m1 = 0;
	int m2 = 1;
	int coordinate_size = u_coordinates.size()/ne;

	// stringstream oss(" ");
    // // oss << "solution-" << ne << "-" << p;
	// oss << "u_element_transformation" << ne << "-" << max_order;
    // Model::GetFilename(oss);
  	// oss << ".txt";
	// ofstream cordi_log(oss.str().c_str(), ios::app);
	// cordi_log.precision(16);

	// for(int i=0; i<ne; i++)
	// {
	// 	vector<double> x(coordinate_size/2);
	// 	vector<double> y(coordinate_size/2);

	// 	for(int j=0; j<coordinate_size/2; j++)
	// 	{
	// 		x[j] = u_coordinates[m1];
	// 		y[j] = u_coordinates[m2];
	// 		m1=m1+2;
	// 		m2=m2+2;
	// 	}

	// 	double x1 = x[0];
	// 	double y1 = y[0];
	// 	double x2 = x[6];
	// 	double y2 = y[6];
	// 	double x3 = x[27];
	// 	double y3 = y[27];

	// 	double A11 = x2 - x1, A12 = x3 - x1;
    // 	double A21 = y2 - y1, A22 = y3 - y1;

	// 	double detA = A11 * A22 - A12 * A21;

	// 	double invA11 = A22 / detA, invA12 = -A12 / detA;
    // 	double invA21 = -A21 / detA, invA22 = A11 / detA;

	// 	cordi_log << "element: " << i+1 << endl;
	// 	cordi_log << "x, y, xi, eta" << endl;

	// 	for(int j=0; j<coordinate_size/2; j++)
	// 	{
	// 		double c1 = x[j] - x1, c2 = y[j] - y1;

	// 		double xi = invA11 * c1 + invA12 * c2;
	// 		double eta = invA21 * c1 + invA22 * c2;

	// 		cordi_log << x[j] << "," << y[j] << "," << xi << "," << eta << endl;
	// 	}
		
	// }

	//Adjoint coordinate transformation

	m1 = 0;
	m2 = 1;
	coordinate_size = adj_coordinates.size()/ne;

	vector<double> trans_adj_coordinates;

	// stringstream oss1(" ");
    // // oss << "solution-" << ne << "-" << p;
	// oss1 << "adj_element_transformation" << ne << "-" << max_order;
    // Model::GetFilename(oss1);
  	// oss1 << ".txt";
	// ofstream adj_cordi_log(oss1.str().c_str(), ios::app);
	// adj_cordi_log.precision(16);

	for(int i=0; i<ne; i++)
	{
		vector<double> x(coordinate_size/2);
		vector<double> y(coordinate_size/2);

		for(int j=0; j<coordinate_size/2; j++)
		{
			x[j] = adj_coordinates[m1];
			y[j] = adj_coordinates[m2];
			m1=m1+2;
			m2=m2+2;
		}

		double x1 = x[0];
		double y1 = y[0];
		double x2 = x[8];
		double y2 = y[8];
		double x3 = x[44];
		double y3 = y[44];

		double A11 = x2 - x1, A12 = x3 - x1;
    	double A21 = y2 - y1, A22 = y3 - y1;

		double detA = A11 * A22 - A12 * A21;

		double invA11 = A22 / detA, invA12 = -A12 / detA;
    	double invA21 = -A21 / detA, invA22 = A11 / detA;

		// adj_cordi_log << "element: " << i+1 << endl;
		// adj_cordi_log << "x, y, xi, eta" << endl;

		for(int j=0; j<coordinate_size/2; j++)
		{
			double c1 = x[j] - x1, c2 = y[j] - y1;

			double xi = invA11 * c1 + invA12 * c2;
			double eta = invA21 * c1 + invA22 * c2;

			//adj_cordi_log << x[j] << "," << y[j] << "," << xi << "," << eta << endl;

			trans_adj_coordinates.push_back(xi);
			trans_adj_coordinates.push_back(eta);
		}
		
	}

	int u_size = u_sol.size()/ne;
	int adj_size = adj_sol.size()/ne;

	int k=0;
	m1=0;
	m2=1;
	int n=0;

	for(int i = 0; i < ne; i++)
	{
		
		ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
	
		ElementData<D, COMP> & elidata = *eldata[i];
	
		if(ml_elidata.nf!=ml_elidata.nf_bc)  // if it is a boundary element, then goes in
		{
			
			
			int ndof_w = ml_elidata.ndof_w;
			
			int fcnr = ml_elidata.faces[0];
		
			ML_FacetData<D, COMP> & ml_fd_1 = *ml_fadata[fcnr];
		
			int nip_1 = 0;
			
			
			nip_1 = nip_universal;

			// err_log << "starting_line_2" << endl;
			err_log << u_size << "," << D << "," << adj_size << endl;
						
			
			// err_log << "parameters" << endl;
			// for(int pr = 0; pr<Model::NumParam;pr++)
			// {
			// 	err_log << param(pr) <<endl;
	    	// }
	    		
			// err_log << "ML Branch Input (boundary element)" << endl;

			for (int pp = 0; pp < u_size; ++pp)
			{
				err_log << log(abs(u_sol[k])) << endl;
				k = k+1;
			}

			// err_log << "ML Trunk Input (Boundary element)" << endl;

			for (int l = 0; l < adj_size; ++l)
			{

				err_log << trans_adj_coordinates[m1] << "," << trans_adj_coordinates[m2] << endl;
				m1=m1+2;
				m2=m2+2;
				
		    }
			
			// err_log << "ML Output Data (Boundary element)" << endl;

			for(int l=0; l<adj_size; l++)
			{
				err_log << log(abs(adj_sol[n])) << endl;
				n=n+1;
			}



			
		}
		else
		{
			// For the internal elements
			
			
			int ndof_w = ml_elidata.ndof_w;

			int fcnr = ml_elidata.faces[0];
	
			ML_FacetData<D, COMP> & ml_fd_1 = *ml_fadata[fcnr];
			int nip_1 = ml_fd_1.nip;

			// err_log << "starting_line_2" << endl;
			err_log << u_size << "," << D << "," << adj_size << endl;
			
			// err_log << "parameters" << endl;
			// for(int pr = 0; pr<Model::NumParam;pr++)
			// {
			// 	err_log << param(pr) <<endl;
		
	    	// }
	    		
			// err_log << "ML Branch Input (internal element)" << endl;

			for (int pp = 0; pp < u_size; ++pp)
			{
				err_log << log(abs(u_sol[k])) << endl;
				k = k+1;
			}

			// err_log << "ML Trunk Input (internal element)" << endl;

			for (int l = 0; l < adj_size; ++l)
			{

				err_log << trans_adj_coordinates[m1] << "," << trans_adj_coordinates[m2] << endl;
				m1=m1+2;
				m2=m2+2;
				
		    }

			// err_log << "ML Output Data (Internal element)" << endl;

			for(int l=0; l<adj_size; l++)
			{
				err_log << log(abs(adj_sol[n])) << endl;
				n=n+1;
			}
		}

	}
}