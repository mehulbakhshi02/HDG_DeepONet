/*! \mainpage Personal reference manual for commands
 *
 * \section Commands to search for hacks
 * - Hack for 2D - A hack that has been used such that this part of the code only works
 * for 2D problems. The section is concluded by "Hack for 2D ends"
 * - Hack for h only - A hack that has been used such that it only works for h only adaptation
 * and not hp. There is not exact place known where this hack ends
 * - Hack for scalar - A hack that has been used such that the code section only works for scalar
 * problems. No exact place is known where this hack ends
 * - Warning - A place where care has to be taken about how the algorithm is implemented. Perhpas
 * a not so obvious trick has been used. The subsequent comments should explain the trick used clearly
 * - Hack - A general hack where something else has been used so as to avoid rewriting code but is compatible 
 * with all possible cases
 *
 * \subsection step1 Step 1: Opening the box
 *
 * etc...
 */
#include <solve.hpp>
#include <fem.hpp>
#include <comp.hpp>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <stdio.h> 
#include <assert.h>
#include <vector>
#include <string>
#include <algorithm>
#include <typeinfo>
#include <limits>


using namespace std;
using namespace ngsolve;

#include "LinearAlgebra/petsc_wrapper.h"
//#include "dummyflux.h"
//#include "bamg_to_netgen.h"

#include "Scripts/bamg2netgen.h"
#include "Scripts/angener2netgen.h"
#include "Scripts/mmg2d2netgen.h"
#include "Scripts/mmg3d2netgen.h"
#include "Scripts/refine2netgen.h"
#include "Scripts/netgen2madlib2d.h"
#include "Scripts/netgen2madlib3d.h"
#include "Scripts/madlib3d2netgen.h"
#include "Scripts/orderinterpolate2d.h"

namespace Unifyingframework {

  double min_y;

#include "Models/models.h"  

#ifdef DG
  #include "DG/elementdata.h"
  #include "DG/helper.h"
#elif HDG
  #include "HDG/elementdata.h"
  #include "HDG/helper.h"
  #include "HDG/Adjoint/helper.h"
#endif

#include "helper.h"
#include "Solvers/helper.h"
#include "Anisotropy/helper.h"

#include "HDG/shock_residual.h"

template <int D, int COMP, class Model>
class UnifyingFramework : public NumProc {

  // Data structures including mesh data, quadrature info, basis functions, reconstructions, ...
  vector<ElementData<D, COMP>*> eldata;
  vector<FacetData<D, COMP>*> fadata;

  double residual[COMP];
  double residual_scale[COMP];

  Space fspace;
  GFunction gfunc;

  Space fspacedual;
  GFunction gfuncdual;

  Space fspacehp;

  shared_ptr<GridFunction> gf_excoe;
  shared_ptr<GridFunction> gf_err;
  shared_ptr<GridFunction> gf_const;

  shared_ptr<FESpace> fes_const;

  vector<double> points;
  vector<double> weights;

  // Preprocessing
  void GetElementInformation(const Space & space, vector<FacetData<D,COMP>*> & fadata, vector<ElementData<D,COMP>*> & eldata, shared_ptr<MeshAccess> ma, vector<int> & order_array, LocalHeap & lh);
  void AllocateTempMemory();
  void ComputeWallDistance();

  // Numerical convective and viscous fluxes
  template <typename SCAL>
  void NumConvFlux(Vec<COMP, SCAL> & state1, Vec<COMP, SCAL> & state2, SpatialParams<D> & sparam, Vec<COMP, SCAL> & cflux) const;

  template <typename SCAL>
  void NumViscFlux(Vec<COMP, SCAL> & state1, Mat<COMP, D, SCAL> & grad1, Vec<COMP, SCAL> & state2, Mat<COMP, D, SCAL> & grad2, SpatialParams<D> & sparam, Vec<COMP, SCAL> & vflux) const;

  double UpdateSolution(Solution & delta, Solution & sol);
  void CheckUpdate(double & alpha, Residual & res, Solution & delta, Solution & sol);
  void LimitUpdate(double & alpha, Solution & delta, Solution & sol);
  void ReconstructSolution(Solution & sol);
  void ReconstructSolution(Solution & sol, vector<FacetData<D, COMP> *> & fadata, vector<ElementData<D, COMP> *> & eldata);

  void LoadParameters(shared_ptr<PDE> apde);
  
  // Boundary conditions
  
  template <typename SCAL>
  void GetBoundaryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bstate) const;

  template <typename SCAL>
  void GetBoundaryGradient(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & bgrad) const;

  // Boundary conditions can be applied to the state(+gradient) or the flux directly. This distinguishing takes place within these functions.
  template <typename SCAL>
  void GetBoundaryConvFlux(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcflux) const;

  template <typename SCAL>
  void GetBoundaryViscFlux(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bvflux) const;

  // These functions have to be implemented by a given discretizations
  // (or at least the residual for explicit timestepping)
  void SolveLinearizedSystem(Solution & delta, Residual & res, int & its);
  void AssembleSystemMatrix();
  void AssembleResidual();

  // For adjoint computations (this should use one of the monitors).
  // The adjoint matrix is assumed to be the transposed system matrix.
  void SolveAdjointSystem(const Solution & sol, Solution & adj, vector<double> & error, int & its,  LocalHeap & lh);
  //  void AssembleAdjointRhs();
 
  // Time-stepping
  void Initial(Solution & sol) const;
  void CalcSpectralRadius(const Solution & sol);
  void Newton(Solution & sol);
  void SetPseudoCFL(const int i, const double resL2, const double resLInf) const;
  
  // Post-processing
  void TransferCoefficients(Solution & sol, GFunction & gfunc);
  void SaveTecplot(const string & filename, int max_order, const Solution & sol, const Space & sp, LocalHeap & lh) const;
  void SaveTecplotCellWise(int order, const string & var, const MeshMetric<D> & metric, const Space & sp, LocalHeap & lh) const;
  void SaveVtu(const string & filename, int max_order, const Solution & sol, const Space & sp, LocalHeap & lh) const;
  void SaveTecplotSurface(const string & filename, int max_order, const Solution & sol, const Space & sp, LocalHeap & lh) const;
  void SaveVtuSurface(string & filename, int max_order, Solution & sol, Space & sp, LocalHeap & lh);

  void ResetMonitors();
  void OutputMonitors(ostream & os = cout);
  void WriteLog();
  void OutputTimes();
  void OutputErrorMonitors(ostream & os = cout);
  void WriteErrorTimeLog();  
  // Adaptation
  void AnisotropicAdaptation(const Solution & sol, LocalHeap & lh);
  void PatchReconstruction(const Solution & sol, Solution & sol_rec,const int add_order, const int numlayers, LocalHeap & lh);
  void SaveMetric(string & metfilename, MeshMetric<D> & metric);
  void ComputeMeshMetric(MeshMetric<D> & metric);
  void ComputeSolutionAnisotropy(Solution & sol, vector< SolAniso<D> > &, const int add_order, LocalHeap & lh);
  void ComputeGradAnisotropy(Solution & sol, vector< SolAniso<D> > &, const int add_order, LocalHeap & lh);
  void ComputePrimalResidual(const Solution & sol, vector<double> &, LocalHeap & lh);
  void ComputeDualResidual(const Solution & sol, vector<double> &, LocalHeap & lh);
  void ComputeLocalResidual(Solution & sol, const int i, vector<double> &residual);
  void ComputeOptimumAnisotropy(const SolAniso<D> &aniso_sol_u, const SolAniso<D> &aniso_grad, const SolAniso<D> &aniso_dual, const SolAniso<D> &aniso_dual_grad, const vector<double> &res_primal, const vector<double> &res_dual, SolAniso<D> &aniso_sol, vector<double> &res_err);
  void ComputeOptimumAnisotropy(const SolAniso<D> aniso_sol, const vector<double> &total_weight, const int p, const double lambda, double sigma, double theta, vector<double> &aniso_loc);
  void OptimizeAnisotropy(const int i, const double lambda, const vector<double> &q1, const vector<double> &q2, const vector<double> &ap, const vector<double> &rho, const vector<double> &phi, const vector<double> &residuals, double &size_comp, double &sigma_opt, double &theta_opt, double &err_comp);
  void OptimizeAnisotropy(const double p, const double lambda, const vector<double> &q1, const vector<double> &q2, const vector<double> &ap, const vector<double> &rho, const vector<double> &phi, const vector<double> &total_weight, double &size_comp, double &sigma_opt, double &theta_opt, double &err_comp);
  double GFunc(const double q1, const double q2, const double rho, const double phi, const double sigma, const double theta);
  void ComputeHighOrderDerivative(const int i, const int order_der_loc, double* sensor, vector<double>&dw, LocalHeap & lh);
  void ComputeMaxDirectionalDerivatives(int order_der_loc, vector<double> & dw, vector<double> & aniso_loc);
  void ComputeMaxDirectionalDerivativesGrad(int order_der_loc, vector<double> & dw, vector<double> & aniso_loc);
  void ComputeAdjointBasedMetric(const Solution & adj, const int add_order, vector< SolAniso<D> > & aniso_sol_vec, SolAniso<D> &aniso_sol);
  void ComputeAdjointBasedMetric(const Solution & sol, const Solution & adj, const int add_order, vector< SolAniso<D> > & aniso_sol_vec, vector< SolAniso<D> > & aniso_dual_vec, SolAniso<D> &aniso_sol, const double coeff_nu, vector<double> &temp);
  void ComputeScaledAnisotropy(SolAniso<D> &aniso_sol, const vector<double> error, const int add_order);
  void ComputeScaledAnisotropy(SolAniso<D> &aniso_sol, vector<double> &error, const vector<double> adj_error, const int add_order);
  // void ComputeAdjointBasedMetric_sys(const Solution & adj, const int add_order, vector< SolAniso<D> > & aniso_sol_vec, SolAniso<D> &aniso_sol);
  void ComputeHPAnisotropy(const SolAniso<D> &aniso_sol_pm1, const SolAniso<D> &aniso_sol_p, const SolAniso<D> &aniso_sol_pp1, SolAniso<D> &aniso_sol, vector<double> &error);
  // Hack to regularize
  void RegularizeAnisotropy(SolAniso<D>& aniso_sol);
  // Hack to regularize ends
  void IntersectMetric(const MeshMetric<D> & metric, const double total_weight, const int p, vector<double> & ansio_loc);
  void IntersectTwoMetric(const vector<double> & metric_1, const vector<double> & metric_2, vector<double> & metric_int);
  // void IntersectTwoMetric2(const vector<double> & metric_1, const vector<double> & metric_2, vector<double> & metric_int);
  void ComputeSolutionSize(const vector<double> & error, vector<double> & metric_size);
  void ComputeSolutionSize(const SolAniso<D> & aniso_sol, vector<double> & metric_size);
  void ComputeSolutionSize(const SolAniso<D> & aniso_sol, const vector<double> & error, vector<double> & metric_size);
  // void ComputeSolutionSize(const vector<double> & error, const double tolerance, vector<double> & metric_size);
  // void AugmentMetric(Metric & metric, const Metric & metric_implied, vector<double> & error);
  void ReconstructFunction(const Solution & sol, Solution & sol_func, LocalHeap & lh);
  void AugmentMetric(const SolAniso<D> aniso_sol, const vector<double> & metric_size, LocalHeap & lh);
  void ComputeSolutionMetric(const SolAniso<D> aniso_sol, const vector<double> & metric_size, MeshMetric<D> & metric);
  void ComputeSolutionMetric(const SolAniso<D> aniso_sol, const vector<double> & metric_size, const int p, MeshMetric<D> & metric);
  void AnalyzeSolution(const Solution & sol, const Solution & adj, LocalHeap & lh);//const Solution & sol, const SolAniso<D> & aniso_sol, LocalHeap & lh
  // Input/output
  bool LoadSolution(const string & filename, Solution & sol) const;
  bool LoadSolutionBamg(const string & filename, Solution & sol, LocalHeap & lh) const;
  void SaveSolution(const string & filename, const Solution & sol, const Space & fspace, LocalHeap & lh) const;
  void SaveSolutionBamg(const string & filename, const Solution & sol, LocalHeap & lh) const;
  void SaveOrderBamg(const string & filename, const vector<int> & orders) const;
  void ReadOrder(const string & filename, vector<int> &eleorder);
  void WriteOrder(const string & filename, const vector<int> &eleorder);
  void ProlongateOrder(const Solution & sol_old, Solution & sol_new, const int delta_order);
  void ProlongateOrder(const Solution & sol_old, Solution & sol_new, const int delta_order, vector<FacetData<D, COMP> *> & fadata, vector<ElementData<D, COMP> *> & eldata);

  void Test(Solution & sol);

  // Special functions for DG
  #ifdef DG

    // For example assembling lifting operator
    #ifdef BR2
      void AssembleLifting();
    #endif

  
  // Special functions for HDG
  #elif HDG

    void AssembleLocalResidual(const int i, vector<double> & vecF, vector<double> & vecG);

    void AssembleLocalMatrix(const int i, vector<double> & matB, vector<double> & matC, vector<double> & matD, 
		                          vector<double> & matR, vector<double> & matS,
		                          vector<double> & matL, vector<double> & matM);

    void AssembleHybridSystemLocal(const int i, vector<double> & matQ, vector<double> & matW, vector<double> & matL, vector<double> & matM);

    void SolveLocalSystem(const int i, vector<double> & matQ, vector<double> & matW, Residual & res);

    // Adjoint Routines
    void AssembleLocalAdjointRhs(const int i, vector<double> & vecF, vector<double> & vecG);
    void AssembleHybridAdjointRhs(const int i, vector<double> & matR, vector<double> & matS, vector<double> & vecF, vector<double> & vecG);
    void SolveLocalAdjointSystem(const int i, vector<double> & matQ, vector<double> & matW);

    void AssembleBR2(const int i, bool derivative, int order_dec);
  //vector<double> & matD, vector<double> & matS, vector<double> & matM, bool derivative);
  
    // Testing routines
    double TestLocalSolves(Solution & sol, double eps);
    double TestJacobian(Solution & sol, double eps);
  #endif

public:
  UnifyingFramework ( shared_ptr<PDE>  apde, const Flags & flags);

  void Do(LocalHeap & lh);

  static shared_ptr<NumProc> Create (shared_ptr<PDE> pde, const Flags & flags)
  {
    return new UnifyingFramework(pde, flags);
  }

  virtual string GetClassName () const 
  {
      return string(name);
  }

};

#include "main_do.h"
#include "monitors.h"
#include "Solvers/newton.h"
  //#include "Solvers/setpseudocfl.h"
  
//#include "adaptation.h"

// DG ingredients
#ifdef DG

  #include "DG/element_access.h"

  #include "DG/solvelinearizedsystem.h"
  #include "DG/assemblesystemmatrix.h"
  #include "DG/assembleresidual.h"

// Convective numerical fluxes

  #ifdef LAXFRIEDRICH
    #include "DG/laxfriedrich.h"
  #endif

// Viscous numerical fluxes

  #ifdef BR2
    #include "DG/br2.h"
  #endif

// HDG ingredients
#elif HDG

  #include "HDG/hdg_init.h"

  #include "HDG/initial.h"

  #include "HDG/element_access.h"
  #include "HDG/allocatetempmemory.h"
  #include "HDG/computewalldistance.h"

  #include "HDG/loadparameters.h"
  #include "HDG/transfercoefficients.h"

  #include "HDG/solvelinearizedsystem.h"
  #include "HDG/Assembly/assemblelocalresidual.h"
  #include "HDG/Assembly/assemblelocalmatrix.h"
  #include "HDG/Assembly/assemblehybridsystemlocal.h"
  #include "HDG/Assembly/solvelocalsystem.h"

  #include "HDG/Adjoint/solveadjointsystem.h"
  #include "HDG/Adjoint/solvelocaladjointsystem.h"
  #include "HDG/Adjoint/assemblelocaladjointrhs.h"
  #include "HDG/Adjoint/assemblehybridadjointrhs.h"

  #include "HDG/updatesolution.h"
  #include "HDG/checkupdate.h"
  #include "HDG/limitupdate.h"
  #include "HDG/reconstructsolution.h"

  #include "HDG/prolongateorder.h"

  #include "HDG/loadsolution.h"
  #include "HDG/loadsolutionbamg.h"
  #include "HDG/savesolution.h"
  #include "HDG/savesolutionbamg.h"

  #include "HDG/writetecplotsurface.h"
  #include "HDG/writetecplot.h"
  #include "HDG/writetecplotcellwise.h"
  #include "HDG/writeVtu.h"
  #include "HDG/writeVtuSurface.h"

  #include "HDG/test.h"
  #include "HDG/testlocalsolves.h"
  #include "HDG/testjacobian.h"

// Convective numerical fluxes

  #ifdef LAXFRIEDRICH
    #include "HDG/NumericalFluxes/laxfriedrich.h"
  #elif ROE
    #include "HDG/NumericalFluxes/roe.h"
  #endif

// Viscous numerical fluxes

  #ifdef LOCALDG
    #include "HDG/NumericalFluxes/ldg.h"
  #elif BR2
    #include "HDG/NumericalFluxes/br2.h"
    #include "HDG/Assembly/assemblebr2.h"
  #endif

#endif


  
//anisotropic refinement

  #include "HDG/patchreconstruction.h"
  #include "Anisotropy/anisotropicadap.h"
  #include "Anisotropy/computemeshmetric.h"
  #include "Anisotropy/computesolutionanisotropy.h"
  #include "Anisotropy/computegradanisotropy.h"
  #include "Anisotropy/computeprimalresidual.h"
  #include "Anisotropy/computedualresidual.h"
  #include "Anisotropy/computelocalresidual.h"
  #include "Anisotropy/computeoptimumanisotropy.h"
  #include "Anisotropy/optimizeanisotropy.h"
  #include "Anisotropy/gfunc.h"
  #include "Anisotropy/computehighorderderivative.h"
  #include "Anisotropy/computemaxdirectionalderivatives.h"
  #include "Anisotropy/computemaxdirectionalderivativesgrad.h"
  #include "Anisotropy/regularizeanisotropy.h"
  #include "Anisotropy/computeadjointbasedmetric.h"
  #include "Anisotropy/computescaledanisotropy.h"
  // #include "Anisotropy/computeadjointbasedmetric_sys.h"
  #include "Anisotropy/computehpanisotropy.h"
  #include "Anisotropy/intersectmetric.h"
  #include "Anisotropy/intersecttwometric.h"
  // #include "Anisotropy/intersecttwometric_old.h"
  #include "Anisotropy/computesolutionsize.h"
  #include "Anisotropy/augmentmetric.h"
  #include "Anisotropy/computesolutionmetric.h"
  #include "Anisotropy/savemetric.h"
  #include "Anisotropy/readorder.h"
  #include "Anisotropy/writeorder.h"
  #include "Anisotropy/analyzesolution.h"
  #include "Anisotropy/reconstructfunction.h"
class Init
{ 
  public: 
    Init ();
};

Init::Init()
{
 // static RegisterNumProc<UnifyingFramework<2, 1, AdvectionDiffusion<2> >> advdiff2d("advectiondiffusion2d");
 // static RegisterNumProc<UnifyingFramework<2, 1, Poisson<2> >> poisson3d("poisson2d");
 // static RegisterNumProc<UnifyingFramework<3, 1, AdvectionDiffusion<3> >> advdiff3d("advectiondiffusion3d");
  // static RegisterNumProc<UnifyingFramework<2, 2, SimpleSystem<2> >> simplesystem("simplesystem2d");
  // static RegisterNumProc<UnifyingFramework<2, 4, CompressibleEuler<2> >> euler("euler2d");
  // static RegisterNumProc<UnifyingFramework<2, 4, CompressibleNavierStokes<2> >> navierstokes("navierstokes2d");
  // static RegisterNumProc<UnifyingFramework<3, 5, CompressibleNavierStokes<3> >> navierstokes("navierstokes3d");
  // static RegisterNumProc<UnifyingFramework<2, 5, CompressibleSpalartAllmaras<2> >> sa2d("sa2d");
   // GetNumProcs().AddNumProc ("euler2d", UnifyingFramework<2, 4, CompressibleEuler<2> >::Create);
  // static RegisterNumProc<UnifyingFramework<2, 3, ShallowWater<2> >> shallowwater("shallowwater2d");
  // static RegisterNumProc<UnifyingFramework<2, 1, Advection<2> >> advection2d("advection2d");
 // static RegisterNumProc<UnifyingFramework<3, 1, Poisson<3> >> poisson3d("poisson3d");
 // Testing codes
 static RegisterNumProc<UnifyingFramework<3, 1, TestAdvectionDiffusionSource<3> >> testadvdiff3d("Problem12");
 static RegisterNumProc<UnifyingFramework<2, 1, TestAdvectionDiffusionSource<2> >> testadvdiff2d("Problem6");
 static RegisterNumProc<UnifyingFramework<2, 1, TestDiffusionSource<2> >> testdiff2d("Problem3and4");
 static RegisterNumProc<UnifyingFramework<3, 1, TestDiffusionSource<3> >> testdiff3d("Problem9and10");
}
  
Init init;
	

}
