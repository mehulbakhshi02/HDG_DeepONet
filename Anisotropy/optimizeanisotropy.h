/** @file
 * @brief We take the values required for the optimization of the anisotropy and compute this numerically
 * @param[in] - i - Index of the cell. Size = 1 int
 * @param[in] - lambda - Square root of the size of the cell. Size = 1 double
 * @param[in] - q1 - Contains the power required to calculate the integral. Size = 6 double
 * @param[in] - q2 - Contains the power required to calculate the integral. Size = 6 double
 * @param[in] - ap - Contains the Ap for the various value. Size = 6 double
 * @param[in] - rho - Contains the rho for the various value. Size = 6 double
 * @param[in] - phi - Contains the phi for the various value. Size = 6 double
 * @param[in] - residuals - Contains the residuals for the various value. Size = 6 double
 * @param[out] - sigma_opt - We return the optiumum sigma for the given set of parameters. Size = 1 double
 * @param[out] - theta_opt -We return the optiumum theta for the given set of parameters. Size = 1 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::OptimizeAnisotropy(const int i, const double lambda, const vector<double> &q1, const vector<double> &q2, const vector<double> &ap, const vector<double> &rho, const vector<double> &phi, const vector<double> &residuals, double &size_comp, double &sigma_opt, double &theta_opt, double &err_comp){
	int no_values = q1.size();
	double delta_sigma = 10000;
	double delta_theta = M_PI;
	int count = 0;
	double sigma_current = sigma_opt;
	double theta_current = theta_opt;
    double pk = (double)order_array[i];
	while((delta_theta > M_PI/180.0 || fabs(delta_sigma - 1.0) > 1e-3) && count<2000)
	{	
		if(count%2 == 0)
		{
			double theta_c1 = theta_current + delta_theta;
			double theta_c2 = theta_current - delta_theta;	
			double v1 = 0.0;
			double v2 = 0.0;
			vector<double> factor(no_values, 0.0);
			factor[0] = pow(ap[0], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
			factor[1] = pow(ap[1], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_current;
			factor[2] = ap[2] * pow(lambda, 2.0*pk+1.0) * sigma_current;
			factor[3] = pow(ap[3], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
			factor[4] = pow(ap[4], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_current;
			factor[5] = ap[5] * pow(lambda, 2.0*pk+1.0) * sigma_current;
			for(int ll = 0; ll < no_values; ++ll)
			{
				v1 += residuals[ll] * sqrt(factor[ll]*GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_current, theta_c1));
				v2 += residuals[ll] * sqrt(factor[ll]*GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_current, theta_c2));
			}
			if(v1 < v2)
			{
				theta_current = theta_c1;
			}
			else
			{
				theta_current = theta_c2;
			}
			delta_theta = delta_theta/2.0;
		}
		if(count%2 == 1)
		{
			double sigma_c1 = sigma_current * delta_sigma;
			double sigma_c2 = sigma_current / delta_sigma;
			double theta_c1 = theta_current;
			double theta_c2 = theta_current;
			if(sigma_c1 < 1.0)
			{
				sigma_c1 = 1.0/sigma_c1;
				theta_c1 = theta_c1 + M_PI/2.0;
			}
			if(sigma_c2 < 1.0)
			{
				sigma_c2 = 1.0/sigma_c2;
				theta_c2 = theta_c2 + M_PI/2.0;
			}

			double v1 = 0.0;
			double v2 = 0.0;
			vector<double> factor_c1(no_values, 0.0);
			factor_c1[0] = pow(ap[0], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
			factor_c1[1] = pow(ap[1], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_c1;
			factor_c1[2] = ap[2] * pow(lambda, 2.0*pk+1.0) * sigma_c1;
			factor_c1[3] = pow(ap[3], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
			factor_c1[4] = pow(ap[4], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_c1;
			factor_c1[5] = ap[5] * pow(lambda, 2.0*pk+1.0) * sigma_c1;

			vector<double> factor_c2(no_values, 0.0);
			factor_c2[0] = pow(ap[0], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
			factor_c2[1] = pow(ap[1], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_c2;
			factor_c2[2] = ap[2] * pow(lambda, 2.0*pk+1.0) *sigma_c2;
			factor_c2[3] = pow(ap[3], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
			factor_c2[4] = pow(ap[4], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_c2;
			factor_c2[5] = ap[5] * pow(lambda, 2.0*pk+1.0) * sigma_c2;
			for(int ll = 0; ll < no_values; ++ll)
			{
				v1 += residuals[ll] * sqrt(factor_c1[ll]*GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_c1, theta_c1));
				v2 += residuals[ll] * sqrt(factor_c2[ll]*GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_c2, theta_c2));
			}
			// cout<<"Count "<<count<<" "<<sigma_c1<<" "<<sigma_c2<<" "<<pow(rho[0], 1.0/(2.0*(order_array[i]+1.0)))<<" "<<delta_sigma<<endl;
			// cout<<v1<<" "<<v2<<endl;

			if(v1 < v2)
			{
				sigma_current = sigma_c1;
				theta_current = theta_c1;
			}
			else
			{
				sigma_current = sigma_c2;
				theta_current = theta_c2;
			}
			delta_sigma = sqrt(delta_sigma);
		}
		count++;
	}// end of while loop

	sigma_opt = sigma_current;
	theta_opt = theta_current;
	vector<double> factor(no_values, 0.0);
	factor[0] = pow(ap[0], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
	factor[1] = pow(ap[1], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_opt;
	factor[2] = ap[2] * pow(lambda, 2.0*pk+1.0) * sigma_opt;
	factor[3] = pow(ap[3], 2.0) * pow(lambda, 2.0*(pk+2.0))/(2.0*pk+4.0);
	factor[4] = pow(ap[4], 2.0) * pow(lambda, 2.0*pk+3.0) * sigma_opt;
	factor[5] = ap[5] * pow(lambda, 2.0*pk+1.0) * sigma_opt;
	for(int ll = 0; ll < no_values; ++ll)
	{
		size_comp += residuals[ll] * sqrt(factor[ll]);
		err_comp += residuals[ll] * sqrt(factor[ll] * GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_opt, theta_opt));
	}
}

/** @file
 * @brief We take the values required for the optimization of the anisotropy and compute this numerically
 * @param[in] - i - Index of the cell. Size = 1 int
 * @param[in] - lambda - Square root of the size of the cell. Size = 1 double
 * @param[in] - q1 - Contains the power required to calculate the integral. Size = 6 double
 * @param[in] - q2 - Contains the power required to calculate the integral. Size = 6 double
 * @param[in] - ap - Contains the Ap for the various value. Size = 6 double
 * @param[in] - rho - Contains the rho for the various value. Size = 6 double
 * @param[in] - phi - Contains the phi for the various value. Size = 6 double
 * @param[in] - residuals - Contains the residuals for the various value. Size = 6 double
 * @param[out] - sigma_opt - We return the optiumum sigma for the given set of parameters. Size = 1 double
 * @param[out] - theta_opt -We return the optiumum theta for the given set of parameters. Size = 1 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::OptimizeAnisotropy(const double p, const double lambda, const vector<double> &q1, const vector<double> &q2, const vector<double> &ap, const vector<double> &rho, const vector<double> &phi, const vector<double> &total_weight, double &size_comp, double &sigma_opt, double &theta_opt, double &err_comp){
	int no_values = q1.size();
	double delta_sigma = 10000;
	double delta_theta = M_PI;
	int count = 0;
	double sigma_current = sigma_opt;
	double theta_current = theta_opt;
    double pk = p;

	while((delta_theta > M_PI/180.0 || fabs(delta_sigma - 1.0) > 1e-3) && count<2000)
	{	
		if(count%2 == 0)
		{

			double theta_c1 = theta_current + delta_theta;
			double theta_c2 = theta_current - delta_theta;	
			double v1 = 0.0;
			double v2 = 0.0;
			for(int ll = 0; ll < no_values; ++ll)
			{

				double factor = ap[ll]*pow(lambda,q2[ll]+2.0)/(q2[ll] + 2.0);
				v1 += total_weight[ll] * (factor*GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_current, theta_c1));
				v2 += total_weight[ll] * (factor*GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_current, theta_c2));
			}

			if(v1 < v2)
			{
				theta_current = theta_c1;
			}
			else
			{
				theta_current = theta_c2;
			}
			delta_theta = delta_theta/2.0;
		}
		if(count%2 == 1)
		{
			double sigma_c1 = sigma_current * delta_sigma;
			double sigma_c2 = sigma_current / delta_sigma;
			double theta_c1 = theta_current;
			double theta_c2 = theta_current;
			if(sigma_c1 < 1.0)
			{
				sigma_c1 = 1.0/sigma_c1;
				theta_c1 = theta_c1 + M_PI/2.0;
			}
			if(sigma_c2 < 1.0)
			{
				sigma_c2 = 1.0/sigma_c2;
				theta_c2 = theta_c2 + M_PI/2.0;
			}

			double v1 = 0.0;
			double v2 = 0.0;
			for(int ll = 0; ll < no_values; ++ll)
			{
				double factor = ap[ll]*pow(lambda,q2[ll]+2.0)/(q2[ll] + 2.0);
				v1 += total_weight[ll] * (factor * GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_c1, theta_c1));
				v2 += total_weight[ll] * (factor * GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_c2, theta_c2));
			}

			if(v1 < v2)
			{
				sigma_current = sigma_c1;
				theta_current = theta_c1;
			}
			else
			{
				sigma_current = sigma_c2;
				theta_current = theta_c2;
			}
			delta_sigma = sqrt(delta_sigma);
		}
		count++;
	}// end of while loop

	sigma_opt = sigma_current;
	theta_opt = theta_current;
	for(int ll = 0; ll < no_values; ++ll)
	{
		double factor = ap[ll]*pow(lambda,q2[ll]+2.0)/(q2[ll] + 2.0);
		size_comp += total_weight[ll] * factor;
		err_comp += total_weight[ll] * (factor * GFunc(q1[ll], q2[ll], rho[ll], phi[ll], sigma_opt, theta_opt));
	}
}
