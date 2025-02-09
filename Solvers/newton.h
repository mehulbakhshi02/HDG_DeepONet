#include "../helper_blas.h"

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::Newton(Solution & sol) {

  cout.precision(8);
  cout.setf(ios::scientific,ios::floatfield); 

  Solution delta = sol;
  Residual res;
  res.resL = 1.;
  //  res.resW = 1.;

  sol_save = sol;

  ReconstructSolution(sol);

  pcfl = SolverData::pcfl_min;

  cout << "\t" << string(12*8, '*');
  if (Model::Diffusion || Model::Source)    
    cout << string(2*8, '*');
  cout << endl;
  cout << "\t" << setw(16) << "WallTime" << setw(8) << "#Newton" << setw(16) << "CFL" << setw(8) << "#GMRES" << setw(16) << "resL2" << setw(16) << "resW";
  if (Model::Diffusion || Model::Source)
    cout << setw(16) << "resQ";
  cout << endl;
  cout << "\t" << string(12*8, '*');
  if (Model::Diffusion || Model::Source)    
    cout << string(2*8, '*');
  cout << endl;

  for (int i = 0; i < SolverData::newton_n && res.resL > SolverData::newton_res; ++i) {

    timeval time;
    gettimeofday (&time, 0);
    double now = time.tv_sec + 1.e-6 * time.tv_usec;
    double tcomp = now - starttime;

    cout << "\t" << setw(16) << tcomp << setw(8) << i << setw(16) << pcfl << flush;

    for (int ll = 0; ll < COMP; ++ll)
      residual[ll] = 0.;

    ResetMonitors();
    
    int its = 0; // iteration count for linear solver
    SolveLinearizedSystem(delta, res, its);

    if (SolverData::show_residuals)
      for (int ll = 0; ll < COMP; ++ll)
        cout << setw(16) << sqrt(residual[ll]);

    OutputMonitors();
    
    if (res.resL > SolverData::newton_res) {

      UpdateSolution(delta, sol);
      
      ReconstructSolution(sol);
    }

    // Screen output
    TransferCoefficients(sol, gfunc);
    Ng_Redraw();

    cout << endl;
  }
}
