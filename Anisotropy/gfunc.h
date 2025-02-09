/** @file
 * @brief We calculate the g function at a given point.
 * @param[in] - q1 - Power that is to be given. Size = 1 double
 * @param[in] - q2 - Power of the ratio of A1 and A2 determined by polynomial order. Size = 1 double
 * @param[in] - rho - Ratio of A1/A2. Size = 1 double
 * @param[in] - phi - Angle of maximum direction derivative of the error model. Size = 1 double
 * @param[in] - sigma/beta - Current aspect ratio of the metric. Size = 1 double
 * @param[in] - theta - Current orientation of the metric. Size = 1 double
*/

template <int D, int COMP, class Model>
double UnifyingFramework<D, COMP, Model>
::GFunc(const double q1, const double q2, const double rho, const double phi, const double sigma, const double theta) {
	double integral = 0.0;
	// Calculate the quadrature points
	int Nquad = 100;
	points.resize(Nquad, 0.0);
	weights.resize(Nquad, 0.0);
	lwgt(0, M_PI, Nquad, points, weights);
	for(int i = 0; i<Nquad; ++i)
	{
		double g11 = pow(sigma,2.0)*(pow(cos(phi-theta),2.0)+pow(rho,-2.0/q2)*pow(sin(phi-theta),2.0));
		double g12 = -sin(phi-theta)*cos(phi-theta)*(1.0-pow(rho,-2.0/q2));
		double g22 = pow(sigma,-2.0)*(pow(sin(phi-theta),2.0)+pow(rho,-2.0/q2)*pow(cos(phi-theta),2.0));
		double t = points[i];
		double f = g11*pow(cos(t),2.0)+2.0*g12*sin(t)*cos(t)+g22*pow(sin(t),2.0);
		integral += weights[i]*pow(f, q1);
	}
	return integral;
}

