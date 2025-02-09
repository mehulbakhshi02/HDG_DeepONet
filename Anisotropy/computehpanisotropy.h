/** @file
 * @brief This function takes the three anisotrpy computed using the
 * p-1,p,p+1 either for the Lq norm or for the adjoint. We compare the anisotropies
 * using the error estimate and compute the optimal p. We also
 * compute the optimal anisotropy and store it in aniso_sol. We also populate
 * the array order_new_array with the new polynomial orders.
 * @param[in] - aniso_sol_pm1 - Contains the anisotropy for p-1. Size = ne double
 * @param[in] - aniso_sol_p - Contains the anisotropy for p-1. Size = ne double
 * @param[in] - aniso_sol_pp1 - Contains the anisotropy for p+1. Size = ne double
 * @param[out] - aniso_sol - Contains the consolidated anisotropy. Size = ne double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeHPAnisotropy(const SolAniso<D> &aniso_sol_pm1, const SolAniso<D> &aniso_sol_p, const SolAniso<D> &aniso_sol_pp1, SolAniso<D> &aniso_sol, vector<double> &error) {
	cout << string(32, '*') << endl;
	cout << "Computing optimal polynomial order" << endl;
	cout << string(32, '*') << endl;
	// Norm of the problem used to compute the continuous error
	// estimate. Must be set to 1 if we are using the adjoint
	// based adaptation. This check is done in
	// Anisotropy/analyzesolution.h
	double q = (double)AnisotropyData::norm;
	if(AnisotropyData::norm == -1)
	{
		cout<<"hp adaptation does not work with L infinity norm"<<endl;
		exit(1);
	}
	// We compute the local value of numerator for the bisection
	// method
	// g_pq = f_{pq}^{\frac{D}{q(p+1)+D}}\bar{A}^{\frac{qD}{q(p+1)+D}}\omega^{\frac{q(p+1)}{q(p+1)+D}}*elidata.vol
	// where f_{pq} = \frac{q(p+1)}{q(p+1)+D}
	// \bar{A} = (A_{1}A_{2}\cdots A_{D})^{\frac{1}{D}}
	vector<double> g_pq(ne, 0.0);
	double p_avg = 2.0;
    double p_fact = 1.0;
    for(int dd = 0; dd < D; ++dd)
    	p_fact *= (p_avg + (double)dd + 1.0)/((double)dd+1.0);
    
	if(D == 3)
	{
		AnisotropyData::adapt_const = 8.0/(9.0*sqrt(3.0));
	}
    double Nnodes = p_fact * AnisotropyData::dof_target * AnisotropyData::adapt_const;// The constant that determines the number of cells in the mesh
    // double Nnodes = AnisotropyData::dof_target * AnisotropyData::adapt_const;// The constant that determines the number of cells in the mesh
    // This constant is hard coded now. This has to be made
    // automatic. Can be choosen such that we always have 
    // a nice constant to find using our bisection method
	double x_const = 1.0e15;
	// #pragma omp parallel for
	for(int i = 0; i < ne; i++)
	{
		ElementData<D,COMP> & ed = *eldata[i];
		double p = (double)order_array[i];
		double pow1 = (double)D*q/(q*(p+1.0)+(double)D);//D*q/(q*(p+1)+D)
		double pow2 = (double)D/(q*(p+1.0)+(double)D);//D/(q*(p+1)+D)
		double pow3 = q*(p+1.0)/(q*(p+1.0)+(double)D);//q*(p+1)/(q*(p+1)+D)
		double fpq = pow3;// q*(p+1)/(q*(p+1)+D)
		// weight = the number of dofs per cell
		double weight = 1.0;
	    for(int dd = 0; dd < D; ++dd)
	    	weight *= (p+(double)dd+1.0)/((double)dd+1.0);

	    // a_bar is calculated for the current p
	    double a_prod = aniso_sol_p.GetProd(i);
	    double a_bar = pow(a_prod, 1.0/(double)D);
		g_pq[i] = pow(x_const, pow2)*pow(fpq, pow2) * pow(a_bar, pow1) * pow(weight, pow3) * ed.vol;
		double max_val = 1e16;
		if(g_pq[i]>max_val)
		{
			cout<<"Error! Value exceeded "<<g_pq[i]<<endl;
			// exit(1);
		}
	}

	double nume_sum_a = 0.0;
	double nume_sum_b = 0.0;
	double a_guess = 1e100, b_guess = 1e-100;
	for(int i=0;i<ne;++i)
	{
		double p = (double)order_array[i];
		double fi = (double)D/(q*(p+1.0)+(double)D);// D/(q*(p+1)+D)
		nume_sum_a += g_pq[i]/pow(a_guess, fi);
		nume_sum_b += g_pq[i]/pow(b_guess, fi);
	}

	double err_a = Nnodes-nume_sum_a;
	double err_b = Nnodes-nume_sum_b;
	double err = 1.0;
	double m = 0.0;
	if(err_a * err_b > 0.0)
	{
		cout<<"Error! We have same sign!"<<endl;
		cout<<err_a<<" "<<err_b<<endl;
		exit(1);
	}
	else
	{
		cout<<"Bisection method to compute constant"<<endl;
		int count = 0;
		while(fabs(err)>1e-10 && count<1000)
		{	
			m = pow(10.0,  0.5 * (log10(a_guess) + log10(b_guess)));
			double nume_sum = 0.0;
			
			for(int i=0;i<ne;++i)
			{
				double p = (double)order_array[i];
				double fi = (double)D/(q*(p+1.0)+(double)D);
				nume_sum += g_pq[i]/pow(m, fi);
			}

			err = Nnodes - nume_sum;
			
			if(err * err_a > 0. && err * err_b < 0.)
			{
					a_guess = m;
					err_a = err;
			}
			else if(err * err_a < 0. && err * err_b > 0.)
			{
				b_guess = m;
				err_b = err;
			}
	        else if(err*err_a == 0 || err*err_b == 0)
	        {
	          err = 1e-11;
	        }
			else
			{
				cout<<"The bisection algorithm error. Issue choosing the next value!"<<endl;
				cout<<err<<" "<<err_a<<" "<<err_b<<endl;
				exit(1);
			}
			count++;
			cout<<"Ite "<<count<<": "<<err<<endl;
		}

	}
	cout<<"Value of constant: "<<m<<endl;
	double tol = 1e-16;
	if(m<tol)
	{
		cout<<"Too low constant: "<<m<<endl;
		exit(1);
	}
	order_new_array.resize(ne, 0);
    // #pragma omp parallel for
    for(int i=0;i<ne;i++)
    {
		double p = (double)order_array[i];
		double pow1 = (double)D*q/(q*(p+1.0)+(double)D);//D*q/(q*(p+1)+D)
		double pow2 = (double)D/(q*(p+1.0)+(double)D);//D/(q*(p+1)+D)
		double fact = q*(p+1.0)/(q*(p+1.0)+(double)D);//q*(p+1)/(q*(p+1)+D)

		double weight = 1.0;
	    for(int dd = 0; dd < D; ++dd)
			weight *= (p+(double)dd+1.0)/((double)dd+1.0);

	    // a_bar is calculated for the current p
	    double a_prod = aniso_sol_p.GetProd(i);
	    double a_bar = pow(a_prod, 1.0/(double)D);
	    // This is the optimum area computed for 
	    // optimizing polynomial order p
	    double new_Area = pow(weight/fact, pow2) * pow(a_bar, -1.0 * pow1) * 1.0/pow(x_const, pow2) * pow(m, pow2);

	    double pow3 = q*(p+1.0)/((double)D);// q(p+1)/D
	    double cpq = 0.0;//D/(q(p+1)+D)
	    if(AnisotropyData::norm != -1)
		    cpq = (double)D/(q*(p+1.0)+(double)D);//(D/(q(p+1)+D))^{\frac{1}{q}}
		else
			cpq = 1.0;
		// This is e_d while in template framework we
		// calculate e_d^q
		// Remember that a_prod = A1*A2*...AD while 
		// we want a bar which is (A1*A2*...AD)^(1/D) = a_prod^(1/D) = a_bar
	    double cont_err = cpq * pow(a_bar, q) * pow(new_Area, pow3);

	    error[i] = cont_err;
	    // We now need to calculate the density that produces this
	    // error for p-1,p,p+1
		double cpq_pm1 = (double)D/(q*(p-1.0+1.0)+(double)D);//(D/(q(p-1+1)+D));
		double cpq_p = (double)D/(q*(p+1.0)+(double)D);//(D/(q(p+1)+D))
		double cpq_pp1 = (double)D/(q*(p+1.0+1.0)+(double)D);//(D/(q(p+1+1)+D))

	    // Product of anisotropies
	    double a_prod_pm1 = aniso_sol_pm1.GetProd(i);
	    double a_prod_p = aniso_sol_p.GetProd(i);
	    double a_prod_pp1 = aniso_sol_pp1.GetProd(i);
	    double a_bar_pm1 = pow(a_prod_pm1, 1.0/(double)D);
	    double a_bar_p = pow(a_prod_p, 1.0/(double)D);
	    double a_bar_pp1 = pow(a_prod_pp1, 1.0/(double)D);

	    // The optimum area for the various
	    // p such that the error is fixed to be
	    // cont_err(remeber that this is area and not density)
	    // In template framework we used density
		double area_pm1 = pow(cont_err/(cpq_pm1 * pow(a_bar_pm1, q)), (double)D/(q*(p-1.0+1.0)));// The area for p-1
		double area_p = pow(cont_err/(cpq_p * pow(a_bar_p, q)), (double)D/(q*(p+1.0)));// The area for p
		double area_pp1 = pow(cont_err/(cpq_pp1 * pow(a_bar_pp1, q)), (double)D/(q*(p+1.0+1.0)));// The area for p+1
		// The dofs computed after calculating the 
		// sizes for the given error tolerance
		if(fabs(new_Area-area_p)/area_p>1e-10)
			cout<<"Something is wrong "<<fabs(new_Area-area_p)<<endl;

		double dof_pm1 = 1.0;
		double dof_p = 1.0;
		double dof_pp1 = 1.0;

		for(int dd = 0; dd < D; dd++)
		{
			dof_pm1 *= (p-1.0+(double)dd+1.0)/((double)dd+1.0);//(p-1+1)*(p-1+2)*(p-1+3)/6
			dof_p *= (p+(double)dd+1.0)/((double)dd+1.0);//(p+1)*(p+2)*(p+3)/6
			dof_pp1 *= (p+1.0+(double)dd+1.0)/((double)dd+1.0);//(p+1+1)*(p+1+2)*(p+1+3)/6
		}
		dof_pm1 = dof_pm1/area_pm1;//dividing by the correct area
		dof_p = dof_p/area_p;
		dof_pp1 = dof_pp1/area_pp1;

		// After we have computed the dofs we now compare to
		// see which is the lowest and pick that particular 
		// polynomial order as the optimum one
		// p-1 is optimum
		// cout<<dof_pm1<<" "<<dof_p<<" "<<dof_pp1<<endl;
		if(dof_pm1 < dof_p && dof_pm1 < dof_pp1)
		{
			vector<double> aniso_loc(D*(D+1),0.0);
			aniso_sol_pm1.GetComponents(i, aniso_loc);
			aniso_sol.SetComponents(i, aniso_loc);
			order_new_array[i] = (int)p - 1;
		}
		else if(dof_p < dof_pm1 && dof_p < dof_pp1)
		{
			vector<double> aniso_loc(D*(D+1),0.0);
			aniso_sol_p.GetComponents(i, aniso_loc);
			aniso_sol.SetComponents(i, aniso_loc);
			order_new_array[i] = (int)p;

		} 
		else if(dof_pp1 < dof_pm1 && dof_pp1 < dof_p)
		{
			vector<double> aniso_loc(D*(D+1),0.0);
			aniso_sol_pp1.GetComponents(i,aniso_loc);
			aniso_sol.SetComponents(i,aniso_loc);
			order_new_array[i] = (int)p + 1;
		}
		else
		{
			cout<<"Error in polynomial optimization!"<<endl;
			exit(1);
		}
		if(order_new_array[i]>AnisotropyData::max_order_adap || order_new_array[i]<AnisotropyData::min_order_adap)
		{
			vector<double> aniso_loc(D*(D+1),0.0);
			aniso_sol_p.GetComponents(i,aniso_loc);
			aniso_sol.SetComponents(i,aniso_loc);
			order_new_array[i] = (int)p;
		}
	}// end of loop over elements

	if(AnisotropyData::regularize_anisotropy)
	{
		RegularizeAnisotropy(aniso_sol);
		vector<double> g_pq(ne, 0.0);
		double p_avg = 2.0;
	    double p_fact = 1.0;
	    for(int dd = 0; dd < D; ++dd)
	    	p_fact *= (p_avg + (double)dd + 1.0)/((double)dd+1.0);
	    
		if(D == 3)
		{
			AnisotropyData::adapt_const = 8.0/(9.0*sqrt(3.0));
		}
	    double Nnodes = p_fact * AnisotropyData::dof_target * AnisotropyData::adapt_const;// The constant that determines the number of cells in the mesh
	    // double Nnodes = AnisotropyData::dof_target * AnisotropyData::adapt_const;// The constant that determines the number of cells in the mesh
	    // This constant is hard coded now. This has to be made
	    // automatic. Can be choosen such that we always have 
	    // a nice constant to find using our bisection method
		double x_const = 1.0e15;
		// #pragma omp parallel for
		for(int i = 0; i < ne; i++)
		{
			ElementData<D,COMP> & ed = *eldata[i];
			double p = (double)order_new_array[i];
			double pow1 = (double)D*q/(q*(p+1.0)+(double)D);//D*q/(q*(p+1)+D)
			double pow2 = (double)D/(q*(p+1.0)+(double)D);//D/(q*(p+1)+D)
			double pow3 = q*(p+1.0)/(q*(p+1.0)+(double)D);//q*(p+1)/(q*(p+1)+D)
			double fpq = pow3;// q*(p+1)/(q*(p+1)+D)
			// weight = the number of dofs per cell
			double weight = 1.0;
		    for(int dd = 0; dd < D; ++dd)
		    	weight *= (p+(double)dd+1.0)/((double)dd+1.0);

		    // a_bar is calculated for the current p
		    double a_prod = aniso_sol.GetProd(i);
		    double a_bar = pow(a_prod, 1.0/(double)D);
			g_pq[i] = pow(x_const, pow2)*pow(fpq, pow2) * pow(a_bar, pow1) * pow(weight, pow3) * ed.vol;
			double max_val = 1e16;
			if(g_pq[i]>max_val)
			{
				cout<<"Error! Value exceeded "<<g_pq[i]<<endl;
				// exit(1);
			}
		}

		double nume_sum_a = 0.0;
		double nume_sum_b = 0.0;
		double a_guess = 1e100, b_guess = 1e-100;
		for(int i=0;i<ne;++i)
		{
			double p = (double)order_new_array[i];
			double fi = (double)D/(q*(p+1.0)+(double)D);// D/(q*(p+1)+D)
			nume_sum_a += g_pq[i]/pow(a_guess, fi);
			nume_sum_b += g_pq[i]/pow(b_guess, fi);
		}

		double err_a = Nnodes-nume_sum_a;
		double err_b = Nnodes-nume_sum_b;
		double err = 1.0;
		double m = 0.0;
		if(err_a * err_b > 0.0)
		{
			cout<<"Error! We have same sign!"<<endl;
			cout<<err_a<<" "<<err_b<<endl;
			exit(1);
		}
		else
		{
			cout<<"Bisection method to compute constant"<<endl;
			int count = 0;
			while(fabs(err)>1e-10 && count<1000)
			{	
				m = pow(10.0,  0.5 * (log10(a_guess) + log10(b_guess)));
				double nume_sum = 0.0;
				
				for(int i=0;i<ne;++i)
				{
					double p = (double)order_new_array[i];
					double fi = (double)D/(q*(p+1.0)+(double)D);
					nume_sum += g_pq[i]/pow(m, fi);
				}

				err = Nnodes - nume_sum;
				
				if(err * err_a > 0. && err * err_b < 0.)
				{
						a_guess = m;
						err_a = err;
				}
				else if(err * err_a < 0. && err * err_b > 0.)
				{
					b_guess = m;
					err_b = err;
				}
				        else if(err*err_a == 0 || err*err_b == 0)
				        {
				          err = 1e-11;
				        }
				else
				{
					cout<<"The bisection algorithm error. Issue choosing the next value!"<<endl;
					cout<<err<<" "<<err_a<<" "<<err_b<<endl;
					exit(1);
				}			
				count++;
				cout<<"Ite "<<count<<": "<<err<<endl;
			}

		}
		cout<<"Value of constant: "<<m<<endl;
		double tol = 1e-16;
		if(m<tol)
		{
			cout<<"Too low constant: "<<m<<endl;
			exit(1);
		}
	    // #pragma omp parallel for
	    for(int i=0;i<ne;i++)
	    {
			double p = (double)order_new_array[i];
			double pow1 = (double)D*q/(q*(p+1.0)+(double)D);//D*q/(q*(p+1)+D)
			double pow2 = (double)D/(q*(p+1.0)+(double)D);//D/(q*(p+1)+D)
			double fact = q*(p+1.0)/(q*(p+1.0)+(double)D);//q*(p+1)/(q*(p+1)+D)

			double weight = 1.0;
		    for(int dd = 0; dd < D; ++dd)
				weight *= (p+(double)dd+1.0)/((double)dd+1.0);

		    // a_bar is calculated for the current p
		    double a_prod = aniso_sol.GetProd(i);
		    double a_bar = pow(a_prod, 1.0/(double)D);
		    // This is the optimum area computed for 
		    // optimizing polynomial order p
		    double new_Area = pow(weight/fact, pow2) * pow(a_bar, -1.0 * pow1) * 1.0/pow(x_const, pow2) * pow(m, pow2);

		    double pow3 = q*(p+1.0)/((double)D);// q(p+1)/D
		    double cpq = 0.0;//D/(q(p+1)+D)
		    if(AnisotropyData::norm != -1)
			    cpq = (double)D/(q*(p+1.0)+(double)D);//(D/(q(p+1)+D))^{\frac{1}{q}}
			else
				cpq = 1.0;
			// This is e_d while in template framework we
			// calculate e_d^q
			// Remember that a_prod = A1*A2*...AD while 
			// we want a bar which is (A1*A2*...AD)^(1/D) = a_prod^(1/D) = a_bar
		    double cont_err = cpq * pow(a_bar, q) * pow(new_Area, pow3);
		    error[i] = cont_err;
		}// end of loop over elements
	}
}
