#ifndef MONITORS_H
#define MONITORS_H

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ResetMonitors() {

  if (Model::NumBdryFluxWeight > 0) {
    compute_bdry_monitors = true;
    bdry_monitors.assign(Model::NumBdryFluxWeight, 0.);
  }

  if (Model::NumVolFunctionals > 0) {
    compute_vol_monitors = true;
    vol_monitors.assign(Model::NumVolFunctionals, 0.);
  }
    
  if (Model::NumBdryCoeffs > 0) {
    compute_bdry_monitors = true;
    stringstream oss(" ");
    oss << "coefficients-" << ne << "-" << order;
    Model::GetFilename(oss);
    oss << ".txt";
    fcoeffs.open(oss.str().c_str());
    fcoeffs.precision(16);
    fcoeffs.setf(ios::scientific, ios::floatfield);
  }

}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::OutputMonitors(ostream & os) {

  int len = os.precision() + 8;
  
  for (int ll = 0; ll < Model::NumBdryFluxWeight; ++ll)
    os << "," << bdry_monitors[ll];
  compute_bdry_monitors = false;

  for (int ll = 0; ll < Model::NumVolFunctionals; ++ll)
    os << "," << vol_monitors[ll];
  compute_vol_monitors = false;
    
  if (Model::NumBdryCoeffs > 0)
    if (fcoeffs.is_open())
      fcoeffs.close();
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::OutputErrorMonitors(ostream & os) {

  int sz = err_monitors.size();
  for (int ll = 0; ll < sz; ++ll)
    os << "," << fabs(err_monitors[ll]);
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::OutputTimes() {
  double solve_time = solve - starttime;
  double adapt_time = adapt - solve;
  double total_time = adapt - starttime;
  double adjoint_time = AnisotropyData::adjoint_time;
  double patchreconstruction_time = AnisotropyData::patchreconstruction_time;
  double computesolutionanisotropy_time = AnisotropyData::computesolutionanisotropy_time;
  double computeadjointbasedmetric_time = AnisotropyData::computeadjointbasedmetric_time;
  double hp_time = AnisotropyData::hp_time;
  double computesolutionsize_time = AnisotropyData::computesolutionsize_time;
  double augmentmetric_time = AnisotropyData::augmentmetric_time;
  // Printing the times
  cout<<setprecision(2)<<endl;
  cout<<"Work"<<setw(16)<<"Time(s)"<<setw(16)<<"% of total"<<endl;
  cout << string(40, '*') << endl;
  cout<<"Solve"<<setw(16)<<fixed<<solve_time<<setw(16)<<(solve_time/total_time)*100<<endl;
  cout<<"Adapt"<<setw(16)<<adapt_time<<setw(16)<<(adapt_time/total_time)*100<<endl;
  cout << string(40, '*') << endl;
  cout<<"Work"<<setw(18)<<"Time(s)"<<setw(16)<<"% of total"<<endl;
  cout << string(40, '*') << endl;
  if(AnisotropyData::adjoint_based)
    cout<<"Adjoint"<<setw(19)<<adjoint_time<<setw(16)<<(adjoint_time/adapt_time)*100<<endl;
  cout<<"Patchwise"<<setw(17)<<patchreconstruction_time<<setw(16)<<(patchreconstruction_time/adapt_time)*100<<endl;
  cout<<"Anisotropy"<<setw(16)<<computesolutionanisotropy_time<<setw(16)<<(computesolutionanisotropy_time/adapt_time)*100<<endl;
  if(AnisotropyData::adjoint_based)
    cout<<"Adj metric"<<setw(16)<<computeadjointbasedmetric_time<<setw(16)<<(computeadjointbasedmetric_time/adapt_time)*100<<endl;
  if(AnisotropyData::hp)
    cout<<"hp"<<setw(24)<<hp_time<<setw(16)<<(hp_time/adapt_time)*100<<endl;
  cout<<"Size"<<setw(22)<<computesolutionsize_time<<setw(16)<<(computesolutionsize_time/adapt_time)*100<<endl;
  cout<<"Metric"<<setw(20)<<augmentmetric_time<<setw(16)<<(augmentmetric_time/adapt_time)*100<<endl;
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::WriteLog() {

  stringstream oss(" ");
  oss << "log-" << order;
  Model::GetFilename(oss);
  oss << ".csv";
  
  ofstream log(oss.str().c_str(), ios::app);
  log.precision(16);
  log.setf(ios::scientific, ios::floatfield);
  
  log << ne << "," << ndof_w_total;
  // We have to use the solve ndof because the ndof_w_total is refilled due to p adaptation
  log << nf << "," << ndof_l_solve << "," << pow(ndof_l_solve, 1.0/(double)D);
  double now = get_time();
  double tcomp = now - starttime;

  log << "," << tcomp;

  OutputMonitors(log);
  log << endl;
  
  log.close();
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::WriteErrorTimeLog() {

  // Writing the error monitors
  stringstream err_oss(" ");
  err_oss << "error_data.csv";
  
  ofstream err_log(err_oss.str().c_str(), ios::app);
  err_log.precision(16);
  err_log.setf(ios::scientific, ios::floatfield);
  
  // err_log << ne << "," << ndof_w_solve << "," << pow(ndof_w_solve, 1.0/(double)D);
  err_log << ne << "," << ndof_l_solve << "," << pow(ndof_l_solve, 1.0/(double)D);
  OutputErrorMonitors(err_log);
  err_log << endl;
  err_log.close();
  OutputTimes();
}

#endif
