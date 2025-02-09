/** @file
 * @brief This function takes in the solution, \c sol, and adapts the mesh either using
 * the anisotropic adaptation where the anisotropy aligns with the solution \f$u\f$ for scalar
 * equations and with the \f$Mach\f$ number for systems.  
 * The size of the cells can determined by a few methods:
 *   - adjoint based error estimate with \c c_max and \c r_max
 *   - continuous mesh optimization to minimise \f$L^q\f$ norm of \f$u\f$ 
 * @param[in] - sol - Contains the data of type Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[in] - lh - Local heap that is used to generate the finite element space for patchwise
 * reconstruction
*/


// KUNAL
#include<algorithm>
#include<iostream>
#include<vector>
#include<fstream>
#include<cmath>
#include<string>
#include<unistd.h>
#include<functional>
#include<cassert>

vector<double> readCsvColumn(const string& filename, int column, int ne) 
{
    ifstream file(filename);
    vector<double> data;
    string line, cell;

    if (!file.is_open()) 
    {
        cerr << "Error opening file: " << filename << endl;
        cerr << "Current working directory: " << filesystem::current_path() << endl; // Debugging: print current working directory
        return data;
    }
    while (getline(file, line)) 
    {
        istringstream lineStream(line);
        int currentColumn = 0;

        while (getline(lineStream, cell, ',')) 
        {
            if (currentColumn == column) 
            {
                // cout<<"currentColumn: "<< currentColumn <<endl;
                // cout<<"column: "<< column <<endl;
                data.push_back(stod(cell));
                // break;
            }
            ++currentColumn;
        }
    }

    file.close();
    /*
    for (int i = 0; i < ne; ++i) 
    {
	cout << "data: "<< data[i] << std::endl;
    }
    */
    
    return data;
}


// To read ML input deploy option
vector<double> input_parameters(string file_name)
{
    // Vector to read the input parameters from a input file
    vector<double> input;
    string item_name;
    int i = 0;

    // Open the file
    ifstream file;
    file.open(file_name);

    // If the file is NOT opened
    if( !file )
    {
        // Error Message if the file couldn't be opened
        cerr << "Error: Input file could not be opened" << endl;
        exit(1);
    }

    cout<<"Input file "<<file_name<<" is opened."<<endl;

    string line;
    while (getline(file, line))
    {
        // Classifying a string as double if the first character is a numeric
        if(line[0]!= '/')
        {
            // To ensure that we are not reading white space characters
            if(isdigit(line[0]))
            {
                input.push_back(stod(line));
            }
        }
    }

    // Closing the input file
    file.close();

    cout<<"Input file "<<file_name<<" is closed."<<endl;
    cout<<endl;

    return input;
}


template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::AnisotropicAdaptation(const Solution & sol, LocalHeap & lh) {
  // Compute the adjoint and the error estimate
  double ts_adj = get_time();
  Solution adj;
  vector<double> error(ne);
  
  int its;
// AnisotropyData::adjoint_based

  vector<double> adj_coordinates;
  vector<double> adj_sol;
  if(AnisotropyData::adjoint_based)
  {
    SolveAdjointSystem(sol, adj, error, its, adj_coordinates, adj_sol, lh);
  }
  
  
  
        // KUNAL  
	vector<double> ml_err(ne);
	ml_err = readCsvColumn("cell_wise_ml_err.csv", 2, ne);
	/*
	for (int i = 0; i < ne; ++i) 
	{
		cout << "anisotropicadap.h: ml_err"<< ml_err[i] << std::endl;
	}
	for (int i = 0; i < ne; ++i) 
	{
		cout << "anisotropicadap.h: "<< error[i] << std::endl;
	}
	*/
	// Filling in ML error:
	
	vector<double> ML_based_adaptation = input_parameters("ML_Option.txt");


	// KUNAL
	// Feeding in ML errors
        cout << ML_based_adaptation[0] <<endl;
	if (ML_based_adaptation[0] == 1.0)
	{	
		for (int i = 0; i < ne; ++i) 
		{
			error[i] = ml_err[i];
		}
	}
	
  double te_adj = get_time();
  // Timing
  AnisotropyData::adjoint_time = te_adj-ts_adj;
  SolAniso<D> aniso_sol(ne);// Contains the anisotropy of the reconciled one
  if(!AnisotropyData::hp && (AnisotropyData::opt_strategy == AnisotropyData::analytic || AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction))
  {
    double ts_p = get_time();
    int no_aniso = COMP;
    if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
      no_aniso = COMP * (D+1);
    // Hack to use derived variables only for the anisotropy
    if(!AnisotropyData::adjoint_based && Model::NumDerVar>0)
      no_aniso = Model::NumDerVar;
    // Hack to use the der var only ends

    // We use the interger add_order to compute the order upto which we need to compute the derivatives
    int add_order = 1;
    int numlayers = 1;
    Solution sol_rec;
    PatchReconstruction(sol, sol_rec, add_order, numlayers, lh);// Increases the order of the arrays by 1
  if (AnisotropyData::save_bamg_solution)
    SaveSolutionBamg("nodesolution.bb", sol_rec, lh);

    ReconstructSolution(sol_rec);
     // We need to decrease the order again as it was increased in patchwise
    max_order = max_order - add_order;
    // Seems to be useful only in the case where we write out the solution
    order = order - add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] - add_order;
    // Order is reduced to p
    double te_p = get_time();
    // Timing
    AnisotropyData::patchreconstruction_time = te_p - ts_p;

    double ts_an = get_time();
    vector< SolAniso<D> > aniso_sol_vec(no_aniso, ne);
    ComputeSolutionAnisotropy(sol_rec, aniso_sol_vec, add_order, lh);
    double te_an = get_time();
    // Timing
    AnisotropyData::computesolutionanisotropy_time = te_an - ts_an;
    // The aniso_sol_vec vector that was passed can be reconciled here using metric intersections
    // We set the add order to 0 because we reduced the order_array to contain p now
    double ts_contadj = get_time();

    if(AnisotropyData::adjoint_based && !AnisotropyData::adj_anisotropy && !(AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction))
    {
      add_order = 0;
      ComputeAdjointBasedMetric(adj, add_order, aniso_sol_vec, aniso_sol);
      if(AnisotropyData::use_aposteriori)
      {
        ComputeScaledAnisotropy(aniso_sol, error, add_order);
        if(AnisotropyData::regularize_anisotropy && !AnisotropyData::hp)
          RegularizeAnisotropy(aniso_sol);

      }
    }
    else if((AnisotropyData::adjoint_based && AnisotropyData::adj_anisotropy && !(AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction)))
    {    
          // coeff_nu=0 is like the original case
    // coeff_nu=1 is the case considering only the anisotropy of the adjoint
      double coeff_nu = 0.5;
      vector< SolAniso<D> > aniso_dual_vec(no_aniso, ne);
      add_order = 1;
      ComputeSolutionAnisotropy(adj, aniso_dual_vec, add_order, lh);
      add_order = 0;
      vector<double> temp(ne, 0.0);
      ComputeAdjointBasedMetric(sol_rec, adj, add_order, aniso_sol_vec, aniso_dual_vec, aniso_sol, coeff_nu, temp);
    }
    else if(AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction)
    {
      aniso_sol = aniso_sol_vec[0];
    }
    else
    {
    // Hack for scalar test case and also to just make sure it doesnt randomly crash
      aniso_sol = aniso_sol_vec[0];
    }
    double te_contadj = get_time();
    // Timing
    AnisotropyData::computeadjointbasedmetric_time = te_contadj - ts_contadj;
  }
  else if(AnisotropyData::hp && AnisotropyData::opt_strategy == AnisotropyData::analytic)
  {
    double ts_p = get_time();
    // We need to call patchwise three times
    // Hack for scalar case - needs to be tested
    int no_aniso = COMP;
    if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
      no_aniso = COMP * (D+1);
    // Hack to use derived variables only for the anisotropy
    if(!AnisotropyData::adjoint_based && Model::NumDerVar>0)
      no_aniso = Model::NumDerVar;
    // Hack to use the der var only ends

    int numlayers = 1;
    // We use the interger add_order to compute the order upto which we need to compute the derivatives
    int add_order = 0;
    Solution sol_rec;
    PatchReconstruction(sol, sol_rec, add_order, numlayers, lh);
    double te_p = get_time();
    ReconstructSolution(sol_rec);
    AnisotropyData::patchreconstruction_time += te_p - ts_p;

    double ts_an = get_time();
    // We need to decrease the order again by 0
    max_order = max_order - add_order;
    // Seems to be useful only in the case where we write out the solution
    order = order - add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] - add_order;

    vector< SolAniso<D> > aniso_sol_vec_pm1(no_aniso, ne);
    ComputeSolutionAnisotropy(sol_rec, aniso_sol_vec_pm1, add_order, lh);

    double te_an = get_time();
    // Timing
    AnisotropyData::computesolutionanisotropy_time += te_an - ts_an;

    ts_p = get_time();
    add_order = 1;
    PatchReconstruction(sol, sol_rec, add_order, numlayers, lh);
    ReconstructSolution(sol_rec);
    // Storing the solution of order p+1 as we need this for the adjoint
    Solution sol_p = sol_rec;
    te_p = get_time();
    AnisotropyData::patchreconstruction_time += te_p - ts_p;

    ts_an = get_time();
    // // We need to decrease the order again
    // // Approximate a p solution via H1 patch reconstruction
    max_order = max_order - add_order;
    // Seems to be useful only in the case where we write out the solution
    order = order - add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] - add_order;
    vector< SolAniso<D> > aniso_sol_vec_p(no_aniso, ne);

    ComputeSolutionAnisotropy(sol_rec, aniso_sol_vec_p, add_order, lh);


    te_an = get_time();
    // Timing
    AnisotropyData::computesolutionanisotropy_time += te_an - ts_an;

    ts_p = get_time();
    add_order = 2;
    PatchReconstruction(sol, sol_rec, add_order, numlayers, lh);
    ReconstructSolution(sol_rec);
    te_p = get_time();
    AnisotropyData::patchreconstruction_time += te_p - ts_p;

    ts_an = get_time();
    // We reduce the order to make it one higher than one in which the solution was solved
    // This seems to be a bit random to me. If we have to edit this
    // we will have to edit it in a few more places
    max_order = max_order - add_order;
    // We reset the solution to have the original order with
    // which we solved the problem
    order = order - add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] - add_order;

    vector< SolAniso<D> > aniso_sol_vec_pp1(no_aniso, ne);

    ComputeSolutionAnisotropy(sol_rec, aniso_sol_vec_pp1, add_order, lh);

    te_an = get_time();
    // Timing
    AnisotropyData::computesolutionanisotropy_time += te_an - ts_an;

    double ts_contadj = get_time();
    SolAniso<D> aniso_sol_pm1(ne);
    SolAniso<D> aniso_sol_p(ne);
    SolAniso<D> aniso_sol_pp1(ne);

    if(AnisotropyData::adjoint_based)
    {
  // Here we have to do the adjoint computations
      // First we have to compute the the space of order p+1 which is the dual space
      // // We need to decrease the order again
      // // Approximate a p solution via H1 patch reconstruction
      add_order = 1;
      max_order = max_order + add_order;
      // Seems to be useful only in the case where we write out the solution
      order = order + add_order;
      for (int i = 0; i < ne; ++i)
        order_array[i] = order_array[i] + add_order;
      // Order_array had p now it has p+1
      GetElementInformation(fspacedual, fadata, eldata, ma, order_array, lh);
      ReconstructSolution(sol_p);
      max_order = max_order - add_order;
      // Seems to be useful only in the case where we write out the solution
      order = order - add_order;
      for (int i = 0; i < ne; ++i)
        order_array[i] = order_array[i] - add_order;
      // Now order_array conatains p
      // We make order_array=-1 which means we want compute with p-1
      add_order = -1;
      ComputeAdjointBasedMetric(adj, add_order, aniso_sol_vec_pm1, aniso_sol_pm1);
      add_order = 0;
      ComputeAdjointBasedMetric(adj, add_order, aniso_sol_vec_p, aniso_sol_p);
      add_order = 1;
      ComputeAdjointBasedMetric(adj, add_order, aniso_sol_vec_pp1, aniso_sol_pp1);
    }
    else
    {   
      aniso_sol_pm1 = aniso_sol_vec_pm1[0];
      aniso_sol_p = aniso_sol_vec_p[0];
      aniso_sol_pp1 = aniso_sol_vec_pp1[0];
    }

    double te_contadj = get_time();
    // Save original adjoint error
    vector<double> adj_error(ne);
    adj_error = error;
    // Timing
    AnisotropyData::computeadjointbasedmetric_time = te_contadj - ts_contadj;
    double ts_hp = get_time();
    ComputeHPAnisotropy(aniso_sol_pm1, aniso_sol_p, aniso_sol_pp1, aniso_sol, error);
    double te_hp = get_time();
    // Timing
    AnisotropyData::hp_time = te_hp - ts_hp;
    if(AnisotropyData::use_aposteriori && AnisotropyData::adjoint_based)
    {
      add_order = 0;
    	ComputeScaledAnisotropy(aniso_sol, error, adj_error, add_order);
    }

  }
  else if(AnisotropyData::opt_strategy == AnisotropyData::numeric)
  {

    double ts_p = get_time();
    // Hack for scalar case - needs to be tested
    int no_aniso = COMP;
    if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
      no_aniso = COMP * (D+1);
    // We use the interger add_order to compute the order upto which we need to compute the derivatives
    int add_order = 1;
    int numlayers = 1;
    Solution sol_rec;
    PatchReconstruction(sol, sol_rec, add_order, numlayers, lh);
    ReconstructSolution(sol_rec);
    double te_p = get_time();
    // Timing
    AnisotropyData::patchreconstruction_time = te_p - ts_p;

    double ts_an = get_time();
    // we will have to edit it in a few more places
    max_order = max_order - add_order;
    // We reset the solution to have the original order with
    // which we solved the problem
    order = order - add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] - add_order;

    vector< SolAniso<D> > aniso_sol_vec(no_aniso, ne);
    ComputeSolutionAnisotropy(sol_rec, aniso_sol_vec, add_order, lh);
    vector< SolAniso<D> > aniso_grad_vec(COMP, ne);
    ComputeGradAnisotropy(sol_rec, aniso_grad_vec, add_order, lh);

    add_order = 1;
    // Compute the anisotropies of the the dual and its gradient
    vector< SolAniso<D> > aniso_dual_vec(no_aniso, ne);
    ComputeSolutionAnisotropy(adj, aniso_dual_vec, add_order, lh);
    vector< SolAniso<D> > aniso_dual_grad_vec(COMP, ne);
    ComputeGradAnisotropy(adj, aniso_dual_grad_vec, add_order, lh);
    double te_an = get_time();
    // Timing
    AnisotropyData::computesolutionanisotropy_time = te_an - ts_an;
	// // // Compute the residual of the primal
    vector<double> res_primal(ne*3, 0.0);
    ComputePrimalResidual(sol, res_primal, lh);
    // Increase order as we want to compute the residual of the dual
    max_order = max_order + add_order;
    order = order + add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] + add_order;
    vector<double> res_dual(ne*3, 0.0);
    ComputeDualResidual(adj, res_dual, lh);
    // // Compute optimum anisotropy
    // Hack for scalar problems
    SolAniso<D> aniso_sol_u(ne);
    SolAniso<D> aniso_grad(ne);
    SolAniso<D> aniso_dual(ne);
    SolAniso<D> aniso_dual_grad(ne);

    aniso_sol_u = aniso_sol_vec[0];
    aniso_grad = aniso_grad_vec[0];
    aniso_dual = aniso_dual_vec[0];
    aniso_dual_grad = aniso_dual_grad_vec[0];
    max_order = max_order - add_order;
    // Seems to be useful only in the case where we write out the solution
    order = order - add_order;
    for (int i = 0; i < ne; ++i)
      order_array[i] = order_array[i] - add_order;
    vector<double> res_err(ne, 0.0);
    double coeff_nu = 0.5;

    for (int i = 0; i < ne; ++i)
    {
      res_dual[i*3+0] = coeff_nu * res_dual[i*3+0];
      res_dual[i*3+1] = coeff_nu * res_dual[i*3+1];
      res_dual[i*3+2] = coeff_nu * res_dual[i*3+2];

      res_primal[i*3+0] = (1.0 - coeff_nu) * res_primal[i*3+0];
      res_primal[i*3+1] = (1.0 - coeff_nu) * res_primal[i*3+1];
      res_primal[i*3+2] = (1.0 - coeff_nu) * res_primal[i*3+2];
    }
    ComputeOptimumAnisotropy(aniso_sol_u, aniso_grad, aniso_dual, aniso_dual_grad, res_primal, res_dual, aniso_sol, res_err);
  } 
  double ts_sz = get_time();
  vector<double> metric_size(ne);

  if(AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction)
  {
    ComputeSolutionSize(error, metric_size);
  }
  else if((AnisotropyData::opt_strategy == AnisotropyData::analytic || AnisotropyData::opt_strategy == AnisotropyData::numeric ) && !AnisotropyData::hp)// && 
  {
    ComputeSolutionSize(aniso_sol, metric_size);
  }
  else if(AnisotropyData::opt_strategy == AnisotropyData::analytic && AnisotropyData::hp)
  {
    ComputeSolutionSize(aniso_sol, error, metric_size);
  }

  double te_sz = get_time();
 //  // Timing
  AnisotropyData::computesolutionsize_time = te_sz - ts_sz;
  double ts_au = get_time();


  AugmentMetric(aniso_sol, metric_size, lh);

  double te_au = get_time();
  // Timingcom
  AnisotropyData::augmentmetric_time = te_au - ts_au;

}
