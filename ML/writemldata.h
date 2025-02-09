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

	int u_size = u_sol.size()/ne;
	int adj_size = adj_sol.size()/ne;

	int k=0;
	int m1=0;
	int m2=1;
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
			err_log << u_size << "," << adj_size << "," << adj_size << endl;
						
			
			// err_log << "parameters" << endl;
			// for(int pr = 0; pr<Model::NumParam;pr++)
			// {
			// 	err_log << param(pr) <<endl;
	    	// }
	    		
			// err_log << "ML Branch Input (boundary element)" << endl;

			for (int pp = 0; pp < u_size; ++pp)
			{
				err_log << u_sol[k] << endl;
				k = k+1;
			}

			// err_log << "ML Trunk Input (Boundary element)" << endl;

			for (int l = 0; l < adj_size; ++l)
			{

				err_log << adj_coordinates[m1] << "," << adj_coordinates[m2] << endl;
				m1=m1+2;
				m2=m2+2;
				
		    }
			
			// err_log << "ML Output Data (Boundary element)" << endl;

			for(int l=0; l<adj_size; l++)
			{
				err_log << adj_sol[n] << endl;
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
			err_log << u_size << "," << adj_size << "," << adj_size << endl;
			
			// err_log << "parameters" << endl;
			// for(int pr = 0; pr<Model::NumParam;pr++)
			// {
			// 	err_log << param(pr) <<endl;
		
	    	// }
	    		
			// err_log << "ML Branch Input (internal element)" << endl;

			for (int pp = 0; pp < u_size; ++pp)
			{
				err_log << u_sol[k] << endl;
				k = k+1;
			}

			// err_log << "ML Trunk Input (internal element)" << endl;

			for (int l = 0; l < adj_size; ++l)
			{

				err_log << adj_coordinates[m1] << "," << adj_coordinates[m2] << endl;
				m1=m1+2;
				m2=m2+2;
				
		    }

			// err_log << "ML Output Data (Internal element)" << endl;

			for(int l=0; l<adj_size; l++)
			{
				err_log << adj_sol[n] << endl;
				n=n+1;
			}
		}

	}
}