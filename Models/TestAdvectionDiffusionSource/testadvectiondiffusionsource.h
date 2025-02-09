#ifndef TestAdvectionDiffusionSource_H
#define TestAdvectionDiffusionSource_H

template <int D>
class TestAdvectionDiffusionSource : virtual public DummyModel<D, 1> {
 public:

  // Charaterization of the model
  static const int  Components = 1;
  static const bool Convection = true;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Function; //Function;

  static double epsilon;

/*!
 * Gives the advection speed \f$\beta\f$
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * direction. Size = D
 * @param[out] dir - Contains the direction \f$\beta(x,t)\f$. Size = D
 */
  static void GetDirection (SpatialParams<D> & sparam, Vec<D> & dir) {
    dir = 1.;
  }
/*!
 * The function returns the convective flux
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the diffusive flux \f$f_{c}(u,\nabla u)=\beta^{T}u\f$. Size = 1xD
 */

  template <typename SCAL>
  static void EvalConvFlux(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    Vec<D> dir;
    GetDirection(sparam, dir);
    res = state * Trans(dir);
  }
  /*!
 * <b>DO NOT KNOW WHAT THIS FUNCTION DOES<\b>
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * \c eig. Size = D
 * @param[in] fix - <b>DO NOT KNOW WHAT THIS VARIABLE</b>. Size = 1
 * @param[out] eig - <b>DO NOT KNOW WHAT THIS VARIABLE</b>. Size = 1
 * @param[in] eigvr - <b>DO NOT KNOW WHAT THIS VARIABLE</b>. Size = 1x1
 * @param[in] eigvl - <b>DO NOT KNOW WHAT THIS VARIABLE</b>. Size = 1x1
 */
  template <typename SCAL>
  static void EvalConvEigenSystem(Vec<1, SCAL> & state, SpatialParams<D> & sparam, double fix, Vec<1, SCAL> & eig, Mat<1, 1, SCAL> & eigvr, Mat<1, 1, SCAL> & eigvl) {
    Vec<D> dir;
    GetDirection(sparam, dir);
    eig = 0.;
    for (int dd = 0; dd < D; ++dd)
      eig(0) += dir(dd) * sparam.normal(dd);
    eigvr = 1.;
    eigvl = 1.;
  }
    /*!
 * <b>DO NOT KNOW WHAT THIS FUNCTION DOES</b>
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] rhs - <b>DO NOT KNOW WHAT THIS VARIABLE IS</b>. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate res. Size=D
 * @param[out] res - <b>DO NOT KNOW WHAT THIS FUNCTION RETURNS</b>. Size = 1xD
 */
  template <typename SCAL>
  static void ApplyK(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, Mat<1, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = epsilon * rhs;
  }  
    /*!
 * The function returns the difussion flux as a vector
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the diffusive flux \f$f_{c}(u,\nabla u)\f$. Size = 1xD
 */
  template <typename SCAL>  
  static void EvalDiffFlux(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = epsilon * grad;

  }
  /*!
 * The initial condition for the problem
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * initial condition. Size=D
 * @param[out] res - Contains the initial \f$u_{0}(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    // Scalar boundary layer
    double eps = epsilon;
    res = 1.0;
    for (int dd = 0; dd < D; ++dd)
    {
      double pos = sparam.pos(dd);
      res *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
    }
  }

  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & bcstate) {
    EvalInitial(sparam, bcstate);
  }

  template <typename SCAL>
  static void EvalSource(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

    double eps = epsilon;
    double temp = 0.0;
    for (int dd = 0; dd < D; ++dd)
    {
      double res_loc = 1.0; 
      for(int ll = 0; ll < D-1; ++ll)
      {
        int index = (dd+ll+1)%D;
        double pos = sparam.pos(index);
        res_loc *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
      }
      temp += res_loc;
    }
    res = temp;  
  }

/*!
 * The function returns the convective flux at the boundary
 * @param[in] bcnr - Contains the type of boundary. Size = 1
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] bcstate - Contains the boundary convective flux at \f$(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    Vec<1, SCAL> bcstate;
    Mat<1, D, SCAL> flux;
    EvalBdryState(bcnr, state, sparam, bcstate);
    EvalConvFlux(bcstate, sparam, flux);
    res = flux * sparam.normal;
  }

  /*!
 * The function returns the diffusive flux at the boundary
 * @param[in] bcnr - Contains the type of boundary. Size = 1
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] bcstate - Contains the boundary diffusive flux at \f$(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

    Vec<1, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);
    Mat<1, D, SCAL> flux;
    EvalDiffFlux(state, grad, sparam, flux);
    res = flux * sparam.normal;
  }

  static void LoadParameters(shared_ptr<PDE> apde) {
    if (apde->ConstantUsed("epsilon"))
      epsilon = apde->GetConstant("epsilon");

  }
  

};
template<int D> double TestAdvectionDiffusionSource<D>::epsilon;

#endif
