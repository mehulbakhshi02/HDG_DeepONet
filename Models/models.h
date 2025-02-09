#ifndef MODELS_H
#define MODELS_H

/*!
 * We define the class of spatial parameters
 */
template <int D>
class SpatialParams {
public:
  ///The current coordinate position. \f$(x_{1}, x_{2},\ldots,x_{d})\f$
  Vec<D> pos;
  ///The normal at the current position. \f$(n_{1},n_{2},\ldots,n_{d})\f$
  Vec<D> normal;
  ///The wall distance. Mostly used for turbulent cases.<b>DO NOT KNOW WHAT EXACTLY THIS IS</b>
  double walldistance;

  ///Boolean to check if the position is set
  bool pos_set;
  ///Boolean to check of the normal has been set
  bool normal_set;
  ///Boolean to check if the wall distance is set
  bool walldistance_set;
/*!
 * We initially set all the values to 0 and the booleans to \c False
 */
  SpatialParams() {
    pos = 0.;
    normal = 0.;
    walldistance = 0.;

    pos_set = false;
    normal_set = false;
    walldistance_set = false;
  }
/*!
 * The overloaded function SpatialParams to set the position \c pos to \c ppos and also set the
 * \c pos_set boolean to \c True. The others remain as 0 or \c False.
 * @param[in] ppos - Contains position. Size = D double
 */
  SpatialParams(const double * ppos) {

    for (int dd = 0; dd < D; ++dd)
      pos(dd) = ppos[dd];

    pos_set = true;
    normal_set = false;
    walldistance_set = false;
  }
/*!
 * The overloaded function SpatialParams to set the position \c pos to \c ppos. The \c walldistance
 * to \c wd. The booleans \c pos_set and \c walldistance_set is set to \c True.
 * The others remain as 0 or \c False.
 * @param[in] ppos - Contains position. Size = D double
 * @param[in] wd - Wall distance. Size = 1 double
 */
  SpatialParams(const double * ppos, double wd) {

    for (int dd = 0; dd < D; ++dd)
      pos(dd) = ppos[dd];

    walldistance = wd;

    pos_set = true;
    normal_set = false;
    walldistance_set = true;
  }

/*!
 * The overloaded function SpatialParams to set the position \c pos to \c ppos. The \c normal
 * to \c pn. The booleans \c pos_set and \c normal_set are set to \c True.
 * The others remain as 0 or \c False.
 * @param[in] ppos - Contains position. Size = D double
 * @param[in] pn - Contains the normals. Size = D double
 */
  SpatialParams(const double * ppos, const double * pn) {

    for (int dd = 0; dd < D; ++dd)
      pos(dd) = ppos[dd];

    for (int dd = 0; dd < D; ++dd)
      normal(dd) = pn[dd];

    pos_set = true;
    normal_set = true;
    walldistance_set = false;
  }
/*!
 * The overloaded function SpatialParams to set the position \c pos to \c ppos. The \c normal
 * to \c pn and the \c walldistance to \c wd. The booleans \c pos_set, \c normal_set and \c
 * \c walldistance_set are set to \c True. None are left as
 * 0 or \c False.
 * @param[in] ppos - Contains position. Size = D double
 * @param[in] pn - Contains the normals. Size = D double
 * @param[in] wd - Contains the wall distance. Size = 1 double
 */
  SpatialParams(const double * ppos, const double * pn, double wd) {

    for (int dd = 0; dd < D; ++dd)
      pos(dd) = ppos[dd];

    for (int dd = 0; dd < D; ++dd)
      normal(dd) = pn[dd];

    walldistance = wd;
    
    pos_set = true;
    normal_set = true;
    walldistance_set = true;
  }    

};

#include "dummymodel.h"

#include "Advection/advection.h"
#include "Poisson/poisson.h"
#include "Michigan1/michigan1.h"
#include "AdvectionDiffusion/advectiondiffusion.h"
#include "Todd/todd.h"

//#include "SimpleSystem/simplesystem.h"
//#include "ShallowWater/shallowwater.h"

#include "CompressibleEuler/euler.h"
#include "CompressibleNavierStokes/navierstokes.h"
#include "CompressibleKOmega/komega.h"
#include "CompressibleSpalartAllmaras/sa.h"

#include "TestAdvectionDiffusionSource/testadvectiondiffusionsource.h"
#include "TestDiffusionSource/testdiffusionsource.h"
#endif
