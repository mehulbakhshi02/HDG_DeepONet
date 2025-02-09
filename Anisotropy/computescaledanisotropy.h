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

::ComputeScaledAnisotropy(SolAniso<D> &aniso_sol, const vector<double> error, const int add_order){
	cout << string(32, '*') << endl;
	cout << "Compute size using adjoint based estimates... " << endl;
	cout << string(32, '*') << endl;
	  // Changing the anisotropic adaptation constant
	if(D == 3)
	{
		AnisotropyData::adapt_const = 8.0/(9.0*sqrt(3.0));
	}

  double q = (double)AnisotropyData::norm;
	  // #pragma omp parallel for
  for (int i = 0; i < ne; ++i)
  {
    // We are not assuming that the order is fixed
	ElementData<D,COMP> & ed = *eldata[i];
	double p = (double)order_array[i]+(double)add_order;
	double cpq = pow((double)D/(q*(p+1.)+(double)D), 1.0/q);
	double alp_power = pow(AnisotropyData::adapt_const, (p+1.)/((double)D));
	double vol_power = pow(ed.vol, (q*(p+1.)+(double)D)/((double)D*q));
	double abar = (fabs(error[i]) * alp_power)/(cpq * vol_power);
	double aprod = pow(abar, (double)D);
    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);

    vector<double> beta(D-1,0.0);
    double beta_prod = 1.0;
    // Beta should be stored in increasing order
    for(int ll = 0; ll < D-1; ++ll)
    {
      // We save the aspect ratios
      // Corrected beta calculation including the 3D version
      beta[ll] = fabs(aniso_loc[D-(ll+2)]/aniso_loc[D-1]);
      beta_prod *= beta[ll];
    }
    double AD_new = pow(aprod/beta_prod, 1.0/(double)D);// A_2 in 2D and A_3 in 3D = (new product/(beta_1))^(1/D)
    vector<double> a_new(D, 0.0);
    for(int ll=0; ll<D-1;++ll)
    {
    	a_new[ll] = AD_new * beta[D-(ll+2)];
    }
    a_new[D-1] = AD_new;

    for(int dd = 0; dd<D; ++dd)
    {
    	aniso_loc[dd] = a_new[dd];
    }

    aniso_sol.SetComponents(i, aniso_loc);
   	double prod_check = aniso_sol.GetProd(i);
    double tol = 1e-12;
   	if(fabs(prod_check-aprod)/(fabs(aprod))>tol)
   	{
   		cout<<"Error in scaling error"<<endl;
   	}
  }
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>

::ComputeScaledAnisotropy(SolAniso<D> &aniso_sol, vector<double> &error, const vector<double> adj_error, const int add_order){
  cout << string(32, '*') << endl;
  cout << "Compute size using adjoint based estimates... " << endl;
  cout << string(32, '*') << endl;
    // Changing the anisotropic adaptation constant
  if(D == 3)
  {
    AnisotropyData::adapt_const = 8.0/(9.0*sqrt(3.0));
  }

  double q = (double)AnisotropyData::norm;
    // #pragma omp parallel for
  for (int i = 0; i < ne; ++i)
  {
    // We are not assuming that the order is fixed as the equidistribution can be done for both h and hp
    ElementData<D,COMP> & ed = *eldata[i];
    double p = (double)order_new_array[i]+(double)add_order;
    double cpq = pow((double)D/(q*(p+1.)+(double)D), 1.0/q);
    double alp_power = pow(AnisotropyData::adapt_const, (p+1.)/((double)D));
    double vol_power = pow(ed.vol, (q*(p+1.)+(double)D)/((double)D*q));
    double abar = (fabs(adj_error[i]) * alp_power)/(cpq * vol_power);
    double aprod = pow(abar, (double)D);
    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);

    vector<double> beta(D-1,0.0);
    double beta_prod = 1.0;
    // Beta should be stored in increasing order
    for(int ll = 0; ll < D-1; ++ll)
    {
      // We save the aspect ratios
      // Corrected beta calculation including the 3D version
      beta[ll] = fabs(aniso_loc[D-(ll+2)]/aniso_loc[D-1]);
      beta_prod *= beta[ll];
    }
    double AD_new = pow(aprod/beta_prod, 1.0/(double)D);// A_2 in 2D and A_3 in 3D = (new product/(beta_1))^(1/D)
    vector<double> a_new(D, 0.0);
    for(int ll=0; ll<D-1;++ll)
    {
      a_new[ll] = AD_new * beta[D-(ll+2)];
    }
    a_new[D-1] = AD_new;

    for(int dd = 0; dd<D; ++dd)
    {
      aniso_loc[dd] = a_new[dd];
    }

    aniso_sol.SetComponents(i, aniso_loc);
    double prod_check = aniso_sol.GetProd(i);
    double tol = 1e-12;
    if(fabs(prod_check-aprod)/(fabs(aprod))>tol)
    {
      cout<<"Error in scaling error"<<endl;
    }
  }
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