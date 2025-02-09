template <int D>
class SolAniso{

  vector<double> Aniso;

  public:
/*!
 * Set the size size of the solution anisotropy arrays
 * @param[in] n - This is the number of elements for which we want to calculate
 * the anisotropy. This is not the size of the Aniso array. The Aniso array is of size
 * n*(D)*(D+1)/2. Ex: For a 2D system we have 3 components and for a 3D system we have 6 components
 */
  SolAniso(int n) : Aniso(n*(D)*(D+1)) {}
  /*!
 * Return the value of each component of the solution
 * @param[in] i - The index of the position where we want the anisotropy. Size = 1 int
 * @param[out] aniso_loc - The first component of the anisotropy of the solution. Size = D*(D+1)/2 double
 * We should store the aniso_loc as follows: Ap1, Ap2, theta_x for 2D and Ap1, Ap2, Ap3, theta_x, theta_y, theta_z for 3D
 */
  void GetComponents(const int i, vector<double> & aniso_loc) const 
  {
    if(aniso_loc.size()!=D*(D+1))
    {
      cout<<"Incorrect size for anisotropy vector. Resizing..."<<endl;
      aniso_loc.resize(D*(D+1));
    }

  	for(int t = 0; t < D * (D+1); ++t)
  	{
      int index = i*D*(D+1) + t;
  		aniso_loc[t] = Aniso[index];
  	}
  }
  /*!
 * Set the value of each component of the solution ansiotropy
 * @param[in] i - The index of the position where we want the solution ansiotropy. Size = 1 int
 * @param[out] aniso_loc - The first component of the anisotropy of the solution. Size = D*(D+1)/2 double
 */
  void SetComponents(const int i, const vector<double> & aniso_loc) {
    if(aniso_loc.size()!=D*(D+1))
    {
      cout<<"Incorrect size for local metric vector"<<endl;
      exit(1);
    }
  	for(int t = 0; t < D * (D+1); ++t)
  	{
      int index = i*D*(D+1) + t;// Gives the location of the first position for the current cell i
  		Aniso[index] = aniso_loc[t];
  	}
  }
/*!
 * Compute the number of elements whose anisotropy is stored
 * @param[out] n - Computes the number of elements whose anisotropy is stored. Size = 1 int
 */
  int GetSize() const {
    double n;
    n = Aniso.size()/(D*(D+1));
    return (int)n;
  }

  /*!
 * Compute the geometric mean of the anisotropies without the factorial of p+1 so we ONLY compute (A_{1}*A_{2}*A_{3}...A_{d})^{1/d}
 * @param[out] prod - Computes the geometric mean of the anisotropy . Size = 1 int
 */
  double GetProd(const int i) const {
    vector<double> aniso_loc(D*(D+1), 0.0);
    GetComponents(i, aniso_loc);
    double prod = 1.0;
    for(int dd = 0; dd < D; ++dd)
    {   
      prod *= aniso_loc[dd];
    }

    return prod;
  }

};