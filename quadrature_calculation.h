void lwgt(double a, double b, int Nquad, vector<double> &points, vector<double> &weights)
{
	int N = Nquad - 1;
	int N1 = N + 1;
	int N2 = N + 2;

	double *xu = (double*) calloc(N1, sizeof(double));//Initialize the array required for the calculation of wieghts
	for(int i = 0; i<N1; i++)
	{
		xu[i] = -1.0 + i*(2.0/(N1-1));//Create a linspace between [-1, 1]
	}

	double *y = (double*) calloc(Nquad, sizeof(double));
	for(int i = 0; i<=N; i++)
	{
		y[i] = cos(((2*i+1)*M_PI)/(2.0*N+2.0))+(0.27/N1)*sin(M_PI*xu[i]*(double)N/(double)N2);
	}

	double* Lp = (double*) calloc(N1, sizeof(double));

	double** L = (double**) calloc(N1, sizeof(*L));
	for(int i = 0; i < N1; i++)
	{
		L[i] = (double*) calloc(N2, sizeof(*(L[i])));
	}

	double *y0 = (double*) calloc(Nquad, sizeof(double));
	for(int i = 0; i < N1; i++)
	{
		y0[i] = 2;
	}
	double eps = 2.220446049250313e-16;

	double max_err = -1000000.0;

	for(int i = 0; i <= N; i++)
	{
		if(max_err < fabs(y[i]-y0[i]))
		{
			max_err = fabs(y[i]-y0[i]);
		}
	}

	while(max_err > eps)
	{
		max_err = -1000000.0;
		for(int i = 0; i < N1; i++)
		{
			L[i][0] = 1.0;
			Lp[i] = 0.0;
			L[i][1] = y[i];
		}
		for(int k = 1; k < N1; k++)
		{
			for(int i = 0; i < N1; i++)
			{
				L[i][k+1] = ( (2*(k+1)-1)*y[i]*L[i][k] - (k)*L[i][k-1] )/(k+1);
			}
		}

		for(int i = 0; i < N1; i++)
		{
			Lp[i] = N2*(L[i][N1-1] - y[i]*L[i][N2-1])/(1-y[i]*y[i]);
		}

		for(int i = 0; i < N1 ; i++)
		{
			y0[i] = y[i];
		}

		for(int i = 0; i < N1 ; i++)
		{
			y[i] = y0[i] - L[i][N2-1]/Lp[i];

		}

		for(int i = 0; i <= N; i++)
		{
			if(max_err < fabs(y[i]-y0[i]))
			{
				max_err = fabs(y[i]-y0[i]);
			}
		}
	}


	for (int i = 0; i < Nquad; ++i)
	{
		points[i] = (a*(1-y[i])+b*(1+y[i]))/2.0;// Calculating the points for integration
	}

	for (int i = 0; i < Nquad; ++i)
	{
		weights[i] = (b-a)/((1-y[i]*y[i])*Lp[i]*Lp[i])*pow((double)N2/(double)N1, 2.0);
	}
	free(xu);
	free(y);
	free(y0);
	for(int i = 0; i < N1; i++)
	{
		free(L[i]);
	}
	free(L);

	free(Lp);
}
