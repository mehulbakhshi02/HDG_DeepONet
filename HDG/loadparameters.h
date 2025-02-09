/** @file
 * @brief This file contains the set of parameters that are read from the PDE file. We also setup the names of the parameters
 * that will be used along with the appropriate name spaces within the code. 
 * @param[in] - apde - A given PDE file which has all the parameters that we need for the code.  Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::LoadParameters(shared_ptr<PDE> apde) {

  min_order = max_order = order = 0;

  LoadConstant(apde, "order", order);
  min_order = max_order = order;

  LoadConstant(apde, "min_order", min_order);
  LoadConstant(apde, "max_order", max_order);

  if (min_order > max_order) {
    cout << "min_order > max_order. Setting min_order = max_order" << endl;
    min_order = max_order;
  }

  max_order = max(max_order, order);

  // Load parameters for nonlinear solver

  {
    using namespace SolverData;
  
    LoadConstant(apde, "pcfl_min", pcfl_min);
    LoadConstant(apde, "pcfl_max", pcfl_max);
    LoadConstant(apde, "pcfl_beta", pcfl_beta);
    LoadConstant(apde, "pcfl_newton", pcfl_newton);
    LoadConstant(apde, "full_newton", full_newton);
    LoadConstant(apde, "limit_update", limit_update);
    LoadConstant(apde, "check_physics", check_physics);
    LoadConstant(apde, "line_search", line_search);
    LoadConstant(apde, "res_inc", res_inc);
    LoadConstant(apde, "min_update", min_update);
    LoadConstant(apde, "show_residuals", show_residuals);

    LoadConstant(apde, "newton_n", newton_n);
    LoadConstant(apde, "newton_res", newton_res);
  }
  // Load parameters for ML 
  {
    using namespace MLParameters;
    LoadConstant(apde, "train", train);
    LoadConstant(apde, "deploy", deploy);
    if(train)
      deploy = false;
    else
      deploy = true;
     
    LoadConstant(apde, "quadadd", quadadd);
  }
  // Load parameters for anisotropic adaptation
  {
    using namespace AnisotropyData;

    LoadConstant(apde, "hp", hp);
    LoadConstant(apde, "read_order", read_order);
    LoadConstant(apde, "min_order_adap", min_order_adap);
    LoadConstant(apde, "max_order_adap", max_order_adap);
    // In the first hp adaptation we want it to run upto
    // the provided max order. For the case when we read in
    // the orders we want it to be the same as the max order
    if(hp && read_order)
      max_order = max_order_adap;

    LoadConstant(apde, "adaptation", adaptation);
    LoadConstant(apde, "adjoint_based", adjoint_based);
    LoadConstant(apde, "save_mesh_metric", save_mesh_metric);
    LoadConstant(apde, "save_adj_metric", save_adj_metric);
    LoadConstant(apde, "read_bamg_order", read_bamg_order);
    LoadConstant(apde, "read_bamg_solution", read_bamg_solution);
    LoadConstant(apde, "save_bamg_order", save_bamg_order);
    LoadConstant(apde, "save_bamg_solution", save_bamg_solution);

    LoadConstant(apde, "dof_control", dof_control);
    LoadConstant(apde, "dof_target", dof_target);
    LoadConstant(apde, "comp_der", comp_der);
    LoadConstant(apde, "limit_metric", limit_metric);
    LoadConstant(apde, "beta_max", beta_max);

    LoadConstant(apde, "rlim", rlim);
    LoadConstant(apde, "clim", clim);

    LoadConstant(apde, "ref_err", ref_err);

    LoadConstant(apde, "r_max", r_max);
    LoadConstant(apde, "c_max", c_max);

    LoadConstant(apde, "excoe_ani_tol", excoe_ani_tol);

    // Added for analytic optimization
    LoadConstant(apde, "opt_strategy", opt_strategy);
    LoadConstant(apde, "norm", norm);
    // if(adjoint_based)
    //   norm = 1;
    // Added to run various mesh generators
    LoadConstant(apde, "mesh_generator", mesh_generator);
    LoadConstant(apde, "projection_basis", projection_basis);
    LoadConstant(apde, "sol_write_error", sol_write_error);
    LoadConstant(apde, "curved_mesh", curved_mesh);
  }

  {
    using namespace AdjointData;

    LoadConstant(apde, "adjoint_tol", nonlin_tol);
    LoadConstant(apde, "adjoint_lin_tol", lin_tol);
    LoadConstant(apde, "adjoint_pcfl_min", pcfl_min);
    LoadConstant(apde, "adjoint_pcfl_beta", pcfl_beta);
    LoadConstant(apde, "adjoint_ksp_restart", ksp_restart);
    LoadConstant(apde, "adjoint_nit_lin", nit_lin);
    LoadConstant(apde, "adjoint_nit", nit);
    LoadConstant(apde, "adjoint_write_error", write_error);
  }
  
  // Petsc
  LoadConstant(apde, "show_monitor", show_monitor);
  LoadConstant(apde, "ksp_restart", ksp_restart);
  LoadConstant(apde, "pc_factor_levels", pc_factor_levels);
  LoadConstant(apde, "rel_tol", rel_tol);
  LoadConstant(apde, "max_steps", max_steps);
  
  LoadConstant(apde, "testing", testing);

  LoadConstant(apde, "shock_capturing", shock_capturing);
  if (shock_capturing > 0)
    LoadConstant(apde, "eps_art_visc", eps_art_visc);

  LoadConstant(apde, "stab_conv", stab_conv);
  LoadConstant(apde, "stab_visc", stab_visc);

  LoadConstant(apde, "tecplot_x", tecplot_x);
  LoadConstant(apde, "tecplot_y", tecplot_y);
  LoadConstant(apde, "tecplot_z", tecplot_z);  

  if (tecplot_x < 1.e-10) tecplot_x = 1.e10;
  if (tecplot_y < 1.e-10) tecplot_y = 1.e10;
  if (tecplot_z < 1.e-10) tecplot_z = 1.e10;

  LoadConstant(apde, "visualize_adjoint", visualize_adjoint);  
  
  LoadConstant(apde, "output", outputname);
  
 
  
  // if (outputname.compare("paraview") == 0 || outputname.compare("paraview_volume") == 0)
  //   outputType = output::paraview_volume;
  if (outputname.compare("tecplot") == 0 || outputname.compare("tecplot_volume") == 0)
    outputType = output::tecplot_volume;
  // else if (outputname.compare("paraview_surface") == 0)
  //   outputType = output::paraview_surface;
  // else if (outputname.compare("tecplot_surface") == 0)
  //   outputType = output::tecplot_surface;
  else    
    outputType = output::none;
}
