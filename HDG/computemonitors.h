#ifndef COMPUTEMONITORS_H
#define COMPUTEMONITORS_H

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::ComputeMonitors() {

  // Boundary monitors (integrated)
  if (Model::NumBdryFluxWeight > 0) {
  
    Vec<Model::NumBdryFluxWeight> bdry_mon(0.);
  
    for (int k = 0; k < nf; ++k) {
      if (!fadata[k]) continue;
      FacetData<D, COMP> & fd = *fadata[k];

      if (fd.ndof_l != 0) continue; // only boundary faces
      if (fd.bcnr   == -2) continue; // no zero-measure faces
    
      for (int j = 0; j < fd.nip; ++j) {

        const double * pw = &fd.w1[j * COMP];
        const double * pq = &fd.q1[j * COMP * D];
        const double * qp = &fd.qp[j * D];
        const double * pn = &fd.n1[j * D];
      
        SpatialParams<D> sparam;
        Vec<COMP> state;
        Mat<COMP, D> grad;
        Vec<COMP> cfn(0.), dfn(0.);

        for (int dd = 0; dd < D; ++dd)
          sparam.pos(dd) = qp[dd];
        for (int dd = 0; dd < D; ++dd)
          sparam.normal(dd) = pn[dd];

        Mat<Model::NumBdryFluxWeight, COMP> weights;
        Model::EvalBdryFluxWeight(fd.bcnr, sparam, weights);
      
        for (int ll = 0; ll < COMP; ++ll)
          state(ll) = pw[ll];

        if (Model::Convection)
          Model::EvalBdryConvFlux(fd.bcnr, state, sparam, cfn);
      
        if (Model::Diffusion) {
          for (int ll = 0; ll < COMP; ++ll)
            for (int dd = 0; dd < D; ++dd)
              grad(ll, dd) = pq[dd+D*ll];
          
          Model::EvalBdryDiffFlux(fd.bcnr, state, grad, sparam, dfn);
        }

        bdry_mon += fd.qw[j] * weights * (cfn - dfn);
      }
    }

    for (int i = 0; i < Model::NumBdryFluxWeight; ++i)
      cout << setw(16) << bdry_mon(i);
  }

  // Volume monitors
  
  /* double err = 0.; */
  /* //    double err_grad = 0.; */
    
  /* for (int i = 0; i < ne; ++i) { */
  /*   ElementData<D, COMP> & ed = *eldata[i]; */

  /*   for (int j = 0; j < ed.nip; ++j) { */

  /*     SpatialParams<D> sparam; */
  /*     for (int dd = 0; dd < D; ++dd) */
  /*       sparam.pos(dd) = ed.qp[D*j+dd]; */

  /*     Vec<COMP> sol; */
  /*     Model::EvalInitial(sparam, sol); */

  /*     for (int ll = 0; ll < COMP; ++ll) */
  /*       err += ed.qw[j] * pow(ed.w[COMP*j+ll] - sol(ll), 2.); */
  /*   } */
  /* } */

  /* double err_f = 0.; */
  /* for (int i = 0; i < nf; ++i) { */
  /*   if (!fadata[i]) continue; */
  /*   FacetData<D, COMP> & fd = *fadata[i]; */
  /*   if (fd.ndof_l == 0) continue; */

  /*   for (int j = 0; j < fd.nip; ++j) { */

  /*     SpatialParams<D> sparam; */
  /*     for (int dd = 0; dd < D; ++dd) */
  /*       sparam.pos(dd) = fd.qp[D*j+dd]; */

  /*     Vec<COMP> sol; */
  /*     Model::EvalInitial(sparam, sol); */

  /*     for (int ll = 0; ll < COMP; ++ll) */
  /*       err_f += fd.qw[j] * fd.h * pow(fd.lambda[COMP*j+ll] - sol(ll), 2.); */
  /*   } */
  /* } */

  /* cout << setw(16) << sqrt(err) << setw(16) << sqrt(err_f); // << setw(16) << sqrt(err_grad); */

  


    // HACK FOR FLATPLATE
    /*    stringstream oss;
    oss << "cf-" << ne << "-" << order;
    oss << "-"<< Euler::mach << "-" << Euler::alpha;
    if (Model::Diffusion)
      oss << "-" << NavierStokes::reynolds;
    oss << ".dat";

    ofstream filecf(oss.str().c_str(), ios::out);

    filecf.precision(8);
    filecf.setf(ios::scientific,ios::floatfield); 

    double ufr = 0.;
    double viscw = 0.;

    filecf << "x\ty\tcp\tcf\typlus" << endl;

    for (int k = 0; k < nf; ++k) {
      if (!fadata[k]) continue;
      FacetData<D, COMP> & fd = *fadata[k];
      if (fd.bcnr != 1) continue;

      int nip = fd.nip;
      for (int j = 0; j < nip; ++j) {
	const double * pw = &fd.w1[j * COMP];
	const double * pq = &fd.q1[j * COMP * D];

	// Compute wall shear stress
	Vec<COMP> state;
	Mat<COMP, D> grad;
	
	for (int ll = 0; ll < COMP; ++ll)
	  state(ll) = pw[ll];
	for (int ll = 0; ll < COMP; ++ll)
	  for (int dd = 0; dd < D; ++dd)
	    grad(ll, dd) = pq[dd+D*ll];

	double rho   = state(0);
	Vec<D> m = state.Rows(1, D+1);
	double rhoE  = state(D+1);       
	double rhok  = state(D+2);
	double rhoinv  = 1. / rho;

	Vec<D> U = rhoinv * m;
	double U2 = 0.;
	for (int dd = 0; dd < D; ++dd)
	  U2 += U(dd) * U(dd);

	double gm1  = Euler::gamma - 1.;
	double ggm1 = Euler::gamma / gm1;
	
	double p = gm1 * rhoE;
	double T = ggm1 * p * rhoinv;

	double cp = 2. * (p - Euler::p_infty) / (Euler::rho_infty * pow(Euler::mach * Euler::c_infty, 2.));
	
	Mat<D, D> gradU = rhoinv * grad.Rows(1, D+1);

	// Sutherland's law
	double TT   = NavierStokes::rr * p / rho;
	double rmu  = NavierStokes::suth_a1 * pow(TT, 1.5) / (TT + NavierStokes::suth_a2);
	// Some scaling due to non-dimensionalization
	double visc = NavierStokes::amu * rmu / NavierStokes::mu0;
	double cf   = 2. * rmu / (NavierStokes::mu0 * NavierStokes::reynolds * sqrt(Euler::gamma) * Euler::mach)  * gradU(0, 1);// / (Euler::rho_infty * pow(Euler::mach * Euler::c_infty, .2));

	if (abs(fd.qp[j*D] - 1.) < 1.e-1) {
	  viscw = visc;
	  ufr = sqrt(visc * gradU(0, 1) / rho);
	}

	double ypl  = rho * min_y * sqrt(visc * gradU(0, 1) / rho) / visc;
	
	filecf << fd.qp[j*D] << "\t" <<  fd.qp[j*D+1] << "\t" << cp << "\t" << cf << "\t" << ypl << endl;
      }
    }

    filecf.close();

    stringstream oss2;
    oss2 << "uplus-" << ne << "-" << order;
    oss2 << "-"<< Euler::mach << "-" << Euler::alpha;
    if (Model::Diffusion)
      oss2 << "-" << NavierStokes::reynolds;
    oss2 << ".dat";

    ofstream fileup(oss2.str().c_str(), ios::out);

    fileup << "x\typlus\tuplus\tkplus\tomegaplus\tnuplus" << endl;

    fileup.precision(8);
    fileup.setf(ios::scientific,ios::floatfield); 

    for (int k = 0; k < ne; ++k) {
      ElementData<D, COMP> & ed = *eldata[k];

      for (int j = 0; j < ed.nip; ++j) {
	const double * pw = &ed.w[COMP * j];
	const double * pq = &ed.q[COMP * D * j];

	Vec<COMP> state;
	Mat<COMP, D> grad;

	for (int ll = 0; ll < COMP; ++ll)
	  state(ll) = pw[ll];
	for (int ll = 0; ll < COMP; ++ll)
	  for (int dd = 0; dd < D; ++dd)
	    grad(ll, dd) = pq[dd+D*ll];

	double rho = state(0);
	Vec<D> m = state.Rows(1, D+1);
	double rhoE = state(D+1);       
	double rhok = state(D+2);
	double rhoinv  = 1. / rho;

	Vec<D> U = rhoinv * m;
	double U2 = 0.;
	for (int dd = 0; dd < D; ++dd)
	  U2 += U(dd) * U(dd);

	double gm1  = Euler::gamma - 1.;
	double ggm1 = Euler::gamma / gm1;
	
	double p = gm1 * (rhoE - 0.5 * rho * U2 - rhok);
	double T = ggm1 * p * rhoinv;
	
	Mat<D, D> gradU = rhoinv * (grad.Rows(1, D+1) - U * Trans(grad.Row(0)));

	// Sutherland's law
	double TT   = NavierStokes::rr * p / rho;
	double rmu  = NavierStokes::suth_a1 * pow(TT, 1.5) / (TT + NavierStokes::suth_a2);
	// Some scaling due to non-dimensionalization
	double visc = NavierStokes::amu * rmu / NavierStokes::mu0;

	if (abs(ed.qp[j*D] - 1.) < 1.e-1) {
	  double yplus = rho * ufr * ed.qp[j*D+1] / visc;
	  double uplus = U(0) / ufr;
	  double kplus = rhok / (rho * ufr * ufr); 
	  double omplus = exp(state(D+3)/rho) * viscw / (rho * ufr * ufr);
	  double nuplus = kplus / omplus;
	  fileup << ed.qp[j*D] << "\t" << yplus << "\t" << uplus << "\t" << kplus << "\t" << omplus << "\t" << nuplus << endl;
	}
      }
    }

    fileup.close(); */

    // END HACK FP



}

#endif
