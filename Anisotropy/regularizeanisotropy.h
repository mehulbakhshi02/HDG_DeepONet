/** @file
 * @brief This function regularizes the A_i's so that we have large numbers only
 * @param[in] - aniso_sol_trial - Local anisotropy for all components and their gradients. Size = COMP*(D+1)*D*(D+1)*ne double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::RegularizeAnisotropy(SolAniso<D> &aniso_sol){
  cout << string(32, '*') << endl;
  cout << "Regularizing anisotropy... " << endl;
  cout << string(32, '*') << endl;
  // We first compute the optimum cellwise error estimate assuming the current distribution is the best
  double q = (double)AnisotropyData::norm;
  double normalization = 0.0;
  double max_prod = -1.0;
  double min_prod = 1e300;
  vector<double> aniso_ori(ne, 0.0);
  for (int i = 0; i < ne; ++i) 
  {
    double aniso_product = aniso_sol.GetProd(i);
    aniso_ori[i] = aniso_product;
    if(aniso_product > max_prod)
      max_prod = aniso_product;
    if(aniso_product < min_prod)
      min_prod = aniso_product;
  }// end of loop over elements

  double fact = 1.0;
  double tol = 1e-200;
  if(min_prod < tol)
  {
    double max_tol = 1e15;
    fact = min(max_tol, max_tol/max_prod);
    fact = pow(fact, 1.0/(double)D);
  }

  // #pragma omp parallel for
  for (int i = 0; i < ne; ++i) 
  {
    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);
    for(int j = 0 ;j < D; ++j)
      aniso_loc[j] = aniso_loc[j] * fact;
    aniso_sol.SetComponents(i, aniso_loc);
  }// end of loop over elements

  if(!AnisotropyData::hp)
  {
    for (int i = 0; i < ne; ++i) 
    {
      ElementData<D,COMP> & ed = *eldata[i];
      double p = (double)order_array[i];
      double power = q/(q*(p+1.0)+(double)D);
      double aniso_product = aniso_sol.GetProd(i);
      normalization += pow(aniso_product, power) * ed.vol;
    }// end of loop over elements
    // #pragma omp parallel for
    int n_reg = 0;
    for (int i = 0; i < ne; ++i) 
    {

      ElementData<D,COMP> & ed = *eldata[i];
      double p = (double)order_array[i];
      double power = (q*(p+1.0)+(double)D)/(double)q;
      double dof_cell = ne;
      double min_prod = pow(normalization/(ed.vol*dof_cell), power);
      double tol_loc = 1e-200;
      min_prod = max(tol_loc, min_prod);
      double aniso_product = aniso_sol.GetProd(i);
      vector<double> aniso_loc(D*(D+1), 0.0);
      aniso_sol.GetComponents(i, aniso_loc);
      vector<double> aniso_ratio(D-1,0.0);
      // we compute the betas to preserve the anisotropic ratios after regulariation
      for(int j = 0; j < D-1; j++)
      {   
        aniso_ratio[j] = aniso_loc[0]/aniso_loc[j+1];
      }

      double factor = 1.0;

      if(aniso_product < factor*min_prod)
      {
        n_reg++;
        if(D==3)
        {
          // We set A3 then A2 then A1 as they are in increasing order
          double fact = aniso_ratio[0]/pow(aniso_ratio[1], 2.0);
          aniso_loc[2] = pow(min_prod, 1.0/(double)D) * pow(fact, 1.0/(double)D);
          aniso_loc[1] = aniso_loc[2] * aniso_ratio[1]/aniso_ratio[0];
          aniso_loc[0] = aniso_loc[2] * aniso_ratio[1];
        }
        if(D==2)
        {
            // We set A2 and then A1
          double fact = aniso_ratio[0];
          aniso_loc[1] = pow(min_prod, 1.0/(double)D)/pow(fact, 1.0/(double)D);
          aniso_loc[0] = aniso_loc[1] * aniso_ratio[0];
        }
        aniso_sol.SetComponents(i, aniso_loc);
      }  

    }// end of loop over elements
    cout<<"Regularized: "<<((double)n_reg/(double)ne)*100.0<<endl;
    cout<<"Not Regularized: "<<(((double)ne-(double)n_reg)/(double)ne)*100.0<<endl;    
  }
  else if(AnisotropyData::hp)
  {
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
      ElementData<D,COMP> & ed = *eldata[i];
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
      // This is the same as 1/d
      double new_Area = pow(weight/fact, pow2) * pow(a_bar, -1.0 * pow1) * 1.0/pow(x_const, pow2) * pow(m, pow2);
      double area_pow = ((double)q*((double)p+1.)+(double)D)/((double)q*(double)D);//(q*(p+1)+D)/Dq
      double area_factor = pow(new_Area * AnisotropyData::adapt_const/ed.vol, area_pow);
      double min_prod = area_factor * a_bar;
      double tol_loc = 1e-200;
      min_prod = max(tol_loc, min_prod);
      double aniso_product = aniso_sol.GetProd(i);
      vector<double> aniso_loc(D*(D+1), 0.0);
      aniso_sol.GetComponents(i, aniso_loc);
      vector<double> aniso_ratio(D-1,0.0);
      // we compute the betas to preserve the anisotropic ratios after regulariation
      for(int j = 0; j < D-1; j++)
      {   
        aniso_ratio[j] = aniso_loc[0]/aniso_loc[j+1];
      }

      double factor = 8.e-10;

      if(aniso_product < factor*min_prod)
      {
        if(D==3)
        {
          // We set A3 then A2 then A1 as they are in increasing order
          double fact = aniso_ratio[0]/pow(aniso_ratio[1], 2.0);
          aniso_loc[2] = pow(factor*min_prod, 1.0/(double)D) * pow(fact, 1.0/(double)D);
          aniso_loc[1] = aniso_loc[2] * aniso_ratio[1]/aniso_ratio[0];
          aniso_loc[0] = aniso_loc[2] * aniso_ratio[1];
        }
        if(D==2)
        {
          // We set A2 and then A1
          double fact = aniso_ratio[0];
          aniso_loc[1] = pow(factor*min_prod, 1.0/(double)D)/pow(fact, 1.0/(double)D);
          aniso_loc[0] = aniso_loc[1] * aniso_ratio[0];
        }
        aniso_sol.SetComponents(i, aniso_loc);
      }  
    }// end of loop over elements
  }

}
