/** @file
 * @brief Contains the solver parameters used in the code. 
*/

/// The pseudo cfl number used in the pseudo time stepping. Size = 1 double
double pcfl;
/// The saved solution (not sure why it is to be declared in this file. Can possibly be a local variable).
Solution sol_save;

namespace SolverData {

  // Convergence tolerance
  /// The default tolerance after which the newton iteration stops. The value used is loaded from the PDE file. The newton iteration stops when number of 
  /// iteration hits either \c newton_n or if we hit this tolerance which ever is earlier.
  double newton_res = 1.e-10;
  /// The default number of iterations after which the newton iteration stops. The used value is loaded from the PDE file.The newton iteration stops when the tolerance
  ///  hits either \c newton_res or if we hit this number of iterations which ever is earlier.
  int newton_n      = 20;
  
  // Robustness
  /// The default minimum pseudo cfl number. The value used is loaded from the PDE file
  double pcfl_min    = 1.;
  /// The default maximum pseudo cfl number. The value used is loaded from the PDE file
  double pcfl_max    = 1.e30;
  /// The default scaling of the pseudo cfl number. The value used is loaded from the PDE file. pcfl = pcfl * pcfl * pcfl_beta
  double pcfl_beta   = 2.;
  /// The default reduction ratio of the pseudo cfl number. The value used is loaded from the PDE file. 
  double pcfl_reduce = 0.1;
  /// The default tolerance after which we shift from damped newton to full newton. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  double full_newton = 1.e-7;
  /// The default scaling of the pseudo cfl number in the full newton regime. The value used is loaded from the PDE file. 
  double pcfl_newton = 2.;
  /// The minimum update during the line search for updating the solution. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  double min_update  = 0.005;
  /// The maximum update during the line search for updating the solution. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  double max_change  = 0.2;
  /// DO NOT KNOW WHAT THIS IS. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  double res_inc     = 0.01; //0.05;
  /// The boolean to see if we want to limit the update of solution. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  bool limit_update  = true;
  /// Check the physics in real euler and ns cases. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  bool check_physics = false;
  /// Boolean to decide if we want to do a line search for solution update. The value used is loaded from the PDE file. (DO NOT KNOW EXACTLY)
  bool line_search   = true;
  /// Boolean to decide if we want to print the residual to screen. The value used is loaded from the PDE file. 
  bool show_residuals = false;
}
