
Solution sol_old;
Solution solPP;

/// Order upto which we calculate error estimates
vector<int> order_array;
/// The array that will store the new order of the elments after hp adaptation
vector<int> order_new_array;
vector<double> vecMa;




//clock_t starttime;
/// Times for start, solve and adaptation
double starttime;
double solve;
double adapt;

double get_time()
{
  timeval time;
  gettimeofday (&time, 0);
  return time.tv_sec + 1.e-6 * time.tv_usec;
}
Timer timer ("timer");

bool CheckForNan(double * array, int size) {
  for (int i = 0; i < size; ++i)
    if (isnan(array[i]))
      return true;
  return false;
}

template <typename T>
void LoadConstant(shared_ptr<PDE> apde, const string & name, T & val) {

  if (apde->ConstantUsed(name))
    val = apde->GetConstant(name);
}

template <>
void LoadConstant<bool>(shared_ptr<PDE> apde, const string & name, bool & val) {

  if (apde->ConstantUsed(name))
    val = apde->GetConstant(name) == 1;
}

template <>
void LoadConstant<string>(shared_ptr<PDE> apde, const string & name, string & val) {

  if (apde->StringConstantUsed(name))
    val = apde->GetStringConstant(name);
}

// ***************************************
// Parameters controlling anisotropic mesh generation

double sum_error;
double sum_abs_error;

bool testing;

bool visualize_adjoint = false;

int shock_capturing;
double eps_art_visc;
vector<double> deps;

vector<double> bdry_monitors;
bool compute_bdry_monitors = false;

vector<double> vol_monitors;
bool compute_vol_monitors = false;

vector<double> err_monitors;

double tecplot_x, tecplot_y, tecplot_z;

string outputname;

ofstream fcoeffs;

namespace output{

enum OutputTypes{
  // tecplot_surface,
  tecplot_volume,
  // paraview_surface,
  // paraview_volume,
  none
};

}

output::OutputTypes outputType;

