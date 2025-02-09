#include "MeshDataBaseInterface.h"
#include "LogSizeField.h"
#include "MAdDefines.h"
#include "AdaptInterface.h"
#include "MAdSolution.h"
#include "MAdMessage.h"

#include <iostream>
#include <iostream>
#include <sstream>
#include <stdio.h> 
#include <fstream>
using namespace MAd;
using namespace std;
void DefineMet(pVertex ver, MAdMetric& met) {
  
  // --- three distances and directions
  double lgt[3];
  double dir[3][3];
  
  
  double amp = 1.;
  
  int i, j;
  double xyz[3], nrm;
  

  V_coord(ver, xyz);
  lgt[0] = 0.1;
  lgt[1] = 0.1;
  lgt[2] = 0.001+0.198*fabs(xyz[2]-0.5);

  // dir[0][0] = amp*sin(MAdPI*xyz[1]);
  // dir[0][1] = 1.;
  // dir[0][2] = 0.;
  
  // dir[1][0] = 1.;
  // dir[1][1] = -amp*sin(MAdPI*xyz[1]);
  // dir[1][2] = 0;
  
  // dir[2][0] = 0.;
  // dir[2][1] = 0.;
  // dir[2][2] = 1.;

  dir[0][0] = 1.0;
  dir[0][1] = 0.;
  dir[0][2] = 0.;
  
  dir[1][0] = 0.;
  dir[1][1] = 1.;
  dir[1][2] = 0;
  
  dir[2][0] = 0.;
  dir[2][1] = 0.;
  dir[2][2] = 1.;
  
  for ( j = 0; j < 2; j++ ) {
    nrm = 0.;
    for ( i = 0; i < 2; i++ ) nrm += dir[j][i] * dir[j][i];
    nrm = sqrt(nrm);
    for ( i = 0; i < 2; i++ ) dir[j][i] /= nrm;
  }
  met = MAdMetric(1./(lgt[0]*lgt[0]), 1./(lgt[1]*lgt[1]), 1./(lgt[2]*lgt[2]), dir[0], dir[1], dir[2]);
}

double solVal(double xyz[3], int iSol) {
  switch ( iSol ) {
    case 0:
      return 2. + xyz[0] * xyz[0] + xyz[1] * xyz[1] + 0.1 * ( xyz[1] * xyz[1] * xyz[1] + 0.001 * xyz[0] * xyz[0] * xyz[0] );
    
    case 1:
      return 0.1 * sin(50. * xyz[0]) + atan( 0.1 / ( sin(5. * xyz[1]) - 2. * xyz[0] ) );
    
    default:
      MAdMsgSgl::instance().error(__LINE__, __FILE__, "Invalid function number");
  }
  return 0.;
}

void DefineSol(pMesh msh, Solution* sol, double*geo, int ord, int iSol) {
  Solution* outSol = NULL;
  if ( ord == 1 ) {
    outSol = dynamic_cast<SolAtVertices*> (sol);
    VIter iteVer = M_vertexIter(msh);
    while ( pVertex ver = VIter_next(iteVer) ) {
      double xyz[3], val;
      V_coord(ver, xyz);
      val = solVal(xyz, iSol);
      outSol->setValue((pEntity) ver, &val);
    }
    VIter_delete(iteVer);
  }
  else {
    outSol = dynamic_cast<SolAtElements*> (sol);
    int nbrNod = ( ord + 1 ) * ( ord + 2 ) / 2;
    FIter iteFac = M_faceIter(msh);
    while ( pFace fac = FIter_next(iteFac) ) {
      double val[nbrNod];
      pVertex facVer[nbrNod];
      for (int iVer = 0; iVer < 3; iVer++ ) {
	facVer[iVer] = F_vertex(fac,iVer);
      }
      for (int iVer = 0; iVer < nbrNod; iVer++ ) {
	double xyz[3];
	xyz[0] = geo[3*iVer] * facVer[0]->X + geo[3*iVer+1] * facVer[1]->X + geo[3*iVer+2] * facVer[2]->X;
	xyz[1] = geo[3*iVer] * facVer[0]->Y + geo[3*iVer+1] * facVer[1]->Y + geo[3*iVer+2] * facVer[2]->Y;
	xyz[2] = geo[3*iVer] * facVer[0]->Z + geo[3*iVer+1] * facVer[1]->Z + geo[3*iVer+2] * facVer[2]->Z;
	
	val[iVer] = solVal(xyz, iSol);
      }
      
      outSol->setValue(fac, val);
    }
  }
  
}

int main ( int argc, char* argv[] ) {
  
  // --- define mesh and geometric model
  pGModel mod = NULL;
  pMesh msh = NULL;
  
  GM_create(&mod);
  msh = M_new(mod);
  
  // --- load mesh (and model)
  M_load(msh,"adapted.msh");
  
  M_writeMesh(msh,"Ini.mesh");

  // --- allocate size field
  LogSField* logFld = new LogSField(msh);
  logFld->setMaxGrad(2.);
  logFld->setMaxAni(10.);
  
  
  // --- allocate mesh adapter
  CMeshAdapter* mshAdp = new CMeshAdapter(msh, logFld, true);
  
  mshAdp->setNbrCollapse(3);
  mshAdp->setNbrSplit(3);
  mshAdp->setNbrSwap(3);
  mshAdp->setMaxIterationsNumber(4);
  
  bool defineMet = true;
  bool defineSol = false;
  int order = 1;
  int iSol  = 0;
  
  // --- Allocate solution (if needed)
  Solution* sol = NULL;
  double* geo = NULL;
  if ( defineSol ) {
    pMeshDataId idSol = MD_lookupMeshDataId("Solution");
    if ( order == 1 ) sol = new SolAtVertices(idSol, msh, 1);
    else {
      int i, j, k;
      int nbrNod = ( order + 1 ) * ( order + 2 ) / 2;
      geo = new double[3*nbrNod];
    
      memset(geo, 0, sizeof(double)*3*nbrNod);
      
      geo[0] = 1.;
      geo[4] = 1.;
      geo[8] = 1.;
      
      int iVer = 3;
      for ( i = 0; i <= order-1; i++ ) {
	for ( j = 0; j <= order-i; j++ ) {
	  k = order-i-j;
	  if ( j == order  || k == order ) continue;
	  geo[3*iVer  ] = (double) i / (double) order;
	  geo[3*iVer+1] = (double) j / (double) order;
	  geo[3*iVer+2] = (double) k / (double) order;
	  iVer++;
	}
      }
      sol = new SolAtElements(idSol, msh, 1, order, geo);
    }
    logFld->setSolution(sol);
  }
  ifstream file("adj_metric.mtr", ios::in);
  int nnode, comp;
  file >> nnode >> comp;
  vector<vector<double> > MeshMetric;
  MeshMetric.resize(nnode,vector<double>(comp,0.0));
  for(int i = 0; i<nnode; i++)
  {
    double aa, bb, cc, dd, ee, ff;
    file >> aa >> bb >> cc >> dd >> ee >> ff;
    MeshMetric[i][0] = aa;
    MeshMetric[i][1] = bb;
    MeshMetric[i][2] = cc;
    MeshMetric[i][3] = dd;
    MeshMetric[i][4] = ee;
    MeshMetric[i][5] = ff;

  }


  // ----------------------- MAIN LOOP --------------------------
  int nbrIte = 1;
  for ( int iteAll = 0; iteAll < nbrIte; iteAll++ ) {
    printf("\n");
    printf("  -----------------------------\n");
    printf("     Main iteration %d / %d    \n", iteAll+1, nbrIte);
    printf("  -----------------------------\n");
    printf("\n");
    int count = 0;
    // --- compute or define metric
    if ( defineMet ) {
      VIter iteVer = M_vertexIter(msh);
      while ( pVertex ver = VIter_next(iteVer) ) {

  double aa = MeshMetric[count][0];
  double bb = MeshMetric[count][1];
  double cc = MeshMetric[count][2];
  double dd = MeshMetric[count][3];
  double ee = MeshMetric[count][4];
  double ff = MeshMetric[count][5];

  double vec[6] = {aa, bb, cc, dd, ee, ff};
  MAdMetric met(vec);
 //  MAdMetric met;
	// DefineMet(ver, met);
	AnisoMeshSize* siz = new AnisoMeshSize(met);
	logFld->setSize((pEntity) ver, siz);
  count++;
      }
      VIter_delete(iteVer);
    }
    else if ( defineSol ) {
      DefineSol(msh, sol, geo, order, iSol);
      logFld->computeAdaptationMet(2.* logFld->getMeshComplexity());
    }
    else MAdMsgSgl::instance().error(__LINE__, __FILE__, "Please give a way to define metric");
    
    logFld->smooth();
    
    // --- perform adaptation
    mshAdp->updateSizeField();
    
    mshAdp->run();
  }
  
  // --- write outputs
  M_writeMesh(msh, "adapted.mesh", 2, 3);
  M_writeMet(msh, logFld->getId(), "Final.sol", 2, 2);
  
  if ( defineSol && order == 1 ) {
    DefineSol(msh, sol, geo, order, iSol);
    M_writeMesh(msh, "FinalSol.mesh", 2, 2);
    M_writeSol(msh, sol, "FinalSol.sol", 2, 2);
  }
  
  // --- free memory
  if ( geo ) delete[] geo;
  if ( sol ) delete sol;
  delete logFld;
  delete mshAdp;
  GM_delete(mod);
  M_delete(msh);
  
  return 0;
}
