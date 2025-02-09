template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::Test(Solution & sol) {

  cout.precision(8);
  cout.setf(ios::scientific,ios::floatfield); 

  cout << setw(16) << "Eps" << setw(16) << "Error LS " << setw(16) << "Error" << endl;
  
  double eps = 1.e-2;
  while (eps > 1.e-10) {
    double err_ls = TestLocalSolves(sol, eps);
    double error  = TestJacobian(sol, eps);
    cout << setw(16) << eps << " " << setw(16) << err_ls << setw(16) << error << endl;
    eps /= 2.;
  }

}

