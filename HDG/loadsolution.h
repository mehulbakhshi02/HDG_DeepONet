/** @file
 * @brief In this file we setup read the solution from a file. Mostly this is used as initial conditions for the solver.
 * @param[in] - filename - Contains the name of the file from which we need to read in the solution. 
 * @param[out] - sol - Contains the solution read from file conditions of type \c Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double 
*/
template <int D, int COMP, class Model>
bool UnifyingFramework<D, COMP, Model>
::LoadSolution(const string & filename, Solution & sol) const {	                 

  string fname_out(filename);
  fname_out.append(".out");
  
  if (Model::Diffusion || Model::Source)
    sol.vecQ.resize(ndof_q_total * COMP);
  sol.vecW.resize(ndof_w_total * COMP);
  sol.vecL.resize(ndof_l_total * COMP);
  
  ifstream file(fname_out.c_str(), ios::binary);

  if (file.is_open()) {
  
    file.read((char *)&sol.vecW[0], sizeof(double) * ndof_w_total * COMP);
    if (Model::Diffusion || Model::Source)
      file.read((char *)&sol.vecQ[0], sizeof(double) * ndof_q_total * COMP);
    file.read((char *)&sol.vecL[0], sizeof(double) * ndof_l_total * COMP);

    file.close();

    cout << "Solution loaded from " << fname_out << endl;

    return true;
  }

  return false;
}
