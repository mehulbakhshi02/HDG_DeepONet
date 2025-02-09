/** @file
 * @brief This file contains the variables related to anisotropic adaptation. Many of them
 * are related to the input and output data. Some are also related to the way we do the 
 * scaling of the metric.
*/
#include "metric.h"
#include "solaniso.h"

namespace AnisotropyData {
  /// The data is loaded from the PDE file
  bool adaptation = false;

  /// If the adaptation has to use the adjoint to run the adaptation
  bool adjoint_based = false;
  bool adj_anisotropy = false;
  // Output
  /// Save the implied metric. Value is loaded from PDE file.
  bool save_mesh_metric   = true;
  /// Save the metric for adaptation. Data is loaded from PDE file.
  bool save_adj_metric    = true;
  /// Read the polynomial order at each node. Value is loaded from PDE file.
  bool read_bamg_order    = false;
  /// Read the solution interpolated by BAMG. Value is loaded from PDE file.
  bool read_bamg_solution = false;
  /// Saves the polynomial order at each cell for hp adaptation. Value is loaded from PDE file
  bool save_bamg_order    = false;
  /// Save the bamg solution used for interpolation. Value is loaded from PDE file
  bool save_bamg_solution = false;
  
  // Control
  /// Used to control the dofs either by scaling the metric elements directly 
  bool dof_control      = true;
  /// Used to fix the target dofs for the adaptation. In case of old adjoint based adaptation
  /// it is ne*(p+1)*(p+2)/2 while in case of the analytical optimization it is only ne
  int  dof_target       = 10000;
  /// Used to decide the number of derivatives whose derivatives need to be calculated. Used especially in the systems case to use for various
  /// functionals
  // [0..COMP-1]: Conservative variables, [COMP...]: Additional variables (Mach, Pressure, Entropy)
  int  comp_der         = 0;
  /// Used in the old \f$hp\f$ adaptation method
  double omega = 0.04;

  // Limiting metric
  /// Used in case we want to limit the metric size or anisotropy. Value taken form PDE file.
  /// DO NOT KNOW EXACTLY HOW THIS IS USED
  bool limit_metric = true;
  /// The mesh fraction where reference error is defined in the old adjoint based adaptation. Has to be between
  /// 0 and 1. Value taken form PDE file.
  double ref_err      = 0.3;
  /// The refinement factor in the old adjoint based adaptation. Value taken form PDE file.
  double r_max      = 3.;
  /// The coarsening factor in the old adjoint based adaptation. Value taken form PDE file.
  double c_max      = 6.;
  /// The limiting factor for limit metric. DO NOT KNOW HOW THIS WORKS. Taken from PDE file.
  double rlim       = 4.;
  /// The limiting factor for limit metric. DO NOT KNOW HOW THIS WORKS. Taken from PDE file.
  double clim       = 2.;
  /// The maximum ansiotropy ratio allowed for metrics. Taken from PDE file.
  // Original
  // double beta_max   = 1000000.;
  
  // KUNAL
  double beta_max = 1.;
  
  
  
  /// Used to select hp adaptation. Value taken from PDE file.
  bool hp               = false;
  int read_order        = 0;
  int max_order_adap    = 5;
  int min_order_adap    = 0;
  /// DO NOT KNOW WHAT THIS DOES. 
  double excoe_ani_tol = 100000.;
  //@{
  /// Chooses the optimization strategy for choosing the size
  /// - 0 - We use the anisotropy of the solution \f$u\f$ for scalar problems or some quantity for anisotropy. The size is decided by the adjoint
  /// - 1 - We the analytic optimization strategy where we use the anisotropy of the reconciled flux or \f$u\f$. The size is computed using the analytic optimization.
  int opt_strategy = 0;
  int mesh_fraction = 0;
  int analytic = 1;
  int numeric = 2;
  //@}
  /// Choose the norm in which we want to do the optimization. Used mostly in the analytical optimization
  /// strategy. Set to -1 to do \f$L^{\infty}\f$ adaptation
  int norm = 2;
  //@{
  /// Choosing mesh generator
  /// Defining the mesh generators so as to use in equality
  int mesh_generator = 1;
  int bamg = 1;
  int angener = 2;
  int mmg2d = 3;
  // int omega_h = 4;
  int madlib = 4;
  int mmg3d = 5;
  int refine = 6;
  //@}
  //@{
  /// Choosing basis to be used in calculating the highest order
  /// derivative
  int projection_basis = 2;
  int dubiner_basis = 1;
  int monomial_basis = 2;
  //@}
  /// Flag to write error to file
  int sol_write_error = 1;
  /// If mesh has to be curved then set in the pde file to be 1
  bool curved_mesh = false;
  /// Contains the constant that is to be set of the analytic
  /// optimization
  /// 3*sqrt(3)/4 for 2D and still unknown for 3D
  double adapt_const = 3.0*sqrt(3.0)/4.0;
  //@{
  /// Time calculation for each step of adaptation
  /// The variable solve is computed at the beginning of adaptation
  /// The name of the variable here shows the time computed AFTER the said function is performed.
  /// Ex: If the variable is called patchreconstruction_time then it is computed after patch reconstruction
  /// is performed
  double adjoint_time;
  double patchreconstruction_time;
  double computesolutionanisotropy_time;
  double computeadjointbasedmetric_time;
  double hp_time;
  double computesolutionsize_time;
  double augmentmetric_time;
  //@}
  /// THe factor to see if we should or should not regularize the anisotropy
  int regularize_anisotropy = 0;
  int use_aposteriori = 1;
  /// FOr the numeric optimization we need to see if we are computing the residual of the solution or the adjoint
  /// If residual_mode = 0 - primal residual being computed
  /// If residual_mode = 1 - dual residual being computed
  bool residual_mode = 0;
  bool use_intersection = true;
}
