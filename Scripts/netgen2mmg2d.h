using namespace std;

void Netgen2Mmg2D(string netgen_mesh, string mmg_mesh) 
{
  ifstream netgen_in;
  netgen_in.open(netgen_mesh.c_str(),ios::in);
  string line;

  vector<double> dummy(5,0.0);
  vector<vector<int> > triang;
  while(true)
  {
    getline(netgen_in,line);
    if (line == "surfaceelements") break;
  }
  int ntng, nvmax;
  netgen_in>>ntng;
  triang.resize(ntng,vector<int>(3,0.0));
  nvmax = 1;
  for(int i=0;i<ntng;i++)
  {
    netgen_in>>dummy[0]>>dummy[1]>>dummy[2]>>dummy[3]>>dummy[4]>>triang[i][0]>>triang[i][1]>>triang[i][2];
    nvmax=max(max(triang[i][0], triang[i][1]), max(triang[i][2],nvmax));
  }
  while(true){
    getline(netgen_in,line);
    if (line == "edgesegmentsgi2") break;
  }
  int medges, vertex_geom = 0;
  netgen_in >> medges;
  int epoints[medges][2], ednr1[medges], ednr2[medges], surfid[medges];
  double dummy_f;
  double dist1[medges], dist2[medges]; 
  int v_surfid[nvmax];
  for (int i = 0; i < nvmax; ++i) v_surfid[i] = 0;
  int i=0;
  while (i < medges){
    netgen_in >> surfid[i] >> dummy_f >> epoints[i][0] >> epoints[i][1] >> dummy_f >> dummy_f >> dummy_f >> dummy_f >> ednr1[i] >> dist1[i] >> ednr2[i] >> dist2[i];
    if (dist1[i] == 0 ) {
       v_surfid[epoints[i][0] - 1] = epoints[i][0]; 
       vertex_geom++;
    }
    else {
       v_surfid[epoints[i][0] - 1] = surfid[i];
    }
    i++;
  }

  while(true)
  {
    getline(netgen_in,line);
    if (line == "points") break;
  }
  int np;
  netgen_in>>np;
  vector<vector<double> > coord;
  coord.resize(np,vector<double>(3,0.0));
  for(int i=0;i<np;i++)
  {
    netgen_in>>coord[i][0]>>coord[i][1]>>coord[i][2];
  }

 //writtimg the MMG3D mesh
 ofstream mmg;
 mmg.open(mmg_mesh.c_str(),ios::out);

 mmg<<"MeshVersionFormatted 2"<<endl;
 mmg<<"Dimension 2"<<endl;
 mmg<<"Vertices"<<endl;
 mmg<<np<<endl;
  vector<int> np_bc_val(np, 0);
  int j=0;
  while (j < medges){
    np_bc_val[epoints[j][0] - 1] = surfid[j];
    np_bc_val[epoints[j][1] - 1] = surfid[j];
    j++;
  }

 for(int i=0;i<np;i++)
 {
   mmg<<coord[i][0]<<" "<<coord[i][1]<<" "<<np_bc_val[i]<<endl;
 }
mmg << endl << "Edges" << endl << medges << endl;
  j=0;
  while (j < medges){
    mmg << epoints[j][0] << " " << epoints[j][1] << " " << surfid[j] << endl;
    j++;
  }
 mmg<<"Triangles"<<endl;
 mmg<<ntng<<endl;
 for(int i=0;i<ntng;i++)
 {
   mmg<<triang[i][0]<<" "<<triang[i][1]<<" "<<triang[i][2]<<" "<<1<<endl;
 }
   mmg << "RequiredVertices " << np << endl;
  j=0 ;
  while(j < np){
    mmg << j+1 << endl;
    j++;
  }
   mmg<< "\nSubDomain 1\n2 1 1 0\n"<<endl;
 mmg<<"End"<<endl;
 mmg.close();
}
