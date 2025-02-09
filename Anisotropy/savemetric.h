/** @file
 * @brief This function takes in a file name and a metric
 * to write out a metric at each node. The metric is computed at the nodes by using
 * volume avearged metric from the cells.
 * DO NOT KNOW WHAT IS HAPPENING IN THIS FUNCTION. Some strange conversion between the
 * cells to the vertex is happening.
 * @param[in] - fname - Contains the name of the file which will have the metric. Size = 1 string
 * @param[in] - metric - Contains the metric which is to be written out to the file. Size = nfxD*(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::SaveMetric(string & fname, MeshMetric<D> & metric) {
 
  fstream outnodemetric;
  // Mesh curving options
  int mesh_points = ma->GetNV();
  if(AnisotropyData::curved_mesh)
  {
      mesh_points = mesh_points - nf;
  }

  outnodemetric.open(fname.c_str(), ios::out);
  outnodemetric.precision(16);
  outnodemetric.setf(ios::scientific, ios::floatfield);
  if((AnisotropyData::mesh_generator == AnisotropyData::bamg || AnisotropyData::mesh_generator == AnisotropyData::angener || AnisotropyData::mesh_generator == AnisotropyData::madlib) && D==2)
    outnodemetric << mesh_points << " " << D*(D+1)/2 << endl;  
  if((AnisotropyData::mesh_generator == AnisotropyData::madlib) && D==3)
    outnodemetric << mesh_points << " " << D*(D+1)/2 << endl;  

  if((AnisotropyData::mesh_generator == AnisotropyData::mmg3d || AnisotropyData::mesh_generator == AnisotropyData::refine )&& D==3)
  {
    outnodemetric << "MeshVersionFormatted 2" << endl;
    outnodemetric << "Dimension 3" << endl;
    outnodemetric << "SolAtVertices" << endl;
    outnodemetric << mesh_points << endl;
    outnodemetric << "1 3"<<endl<<endl;
  }

  if(AnisotropyData::mesh_generator == AnisotropyData::mmg2d && D==2)
  {
    outnodemetric << "MeshVersionFormatted 2" << endl;
    outnodemetric << "Dimension 2" << endl;
    outnodemetric << "SolAtVertices" << endl;
    outnodemetric << mesh_points << endl;
    outnodemetric << "1 3"<<endl<<endl;
  }


  for (int j = 0; j < mesh_points; ++j) {
    Array<int> elns;
    ma->GetVertexElements(j, elns);

    vector<double> metric_avg(D*(D+1)/2,0.0);
    double vol = 0.0;

    for (int k = 0; k < elns.Size(); ++k) {
      int elnr = elns[k];
      ElementData<D,COMP> & ed = *eldata[elnr];

      double vol_loc = ed.vol;
      vector<double> metric_loc(D*(D+1)/2,0.0);

      metric.GetComponents(elnr, metric_loc);
      for(int ll = 0; ll < D*(D+1)/2; ++ll)
      {
        metric_avg[ll] += vol_loc * metric_loc[ll];
      }
      vol += vol_loc;
    }

    for(int ll = 0; ll < D*(D+1)/2; ++ll)
    {
      metric_avg[ll] = metric_avg[ll]/vol;
    }

    for(int ll = 0; ll < D*(D+1)/2; ++ll)
    {
      outnodemetric << setw(25) << metric_avg[ll]; 
    }
    outnodemetric << endl;
  }

  outnodemetric.close();
}
