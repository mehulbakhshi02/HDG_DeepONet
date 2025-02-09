using namespace std;

void Netgen2Refine(string netgen_mesh, string refine_mesh) 
{
  ifstream netgen_in;
  netgen_in.open(netgen_mesh.c_str(),ios::in);
  string line;

  vector<double> dummy(5,0.0);
  vector<vector<int> > triang;
  vector<int> bcnr;
  while(true)
  {
    getline(netgen_in,line);
    if (line == "surfaceelements") break;
    // cout<<count<<endl;
  }
  int ntng;
  netgen_in>>ntng;
  triang.resize(ntng,vector<int>(3,0.0));
  bcnr.resize(ntng);
  for(int i=0;i<ntng;i++)
  {
    netgen_in>>dummy[0]>>bcnr[i]>>dummy[2]>>dummy[3]>>dummy[4]>>triang[i][0]>>triang[i][1]>>triang[i][2];
  }

  while(true)
  {
    getline(netgen_in,line);
    if (line == "volumeelements") break;
    // cout<<count<<endl;
  }
  int ne;
  netgen_in>>ne;
  int dummy1,dummy2;
  vector<vector<int> > connect;
  connect.resize(ne,vector<int>(4,0.0));
  for(int i=0;i<ne;i++)
  {

    netgen_in>>dummy1>>dummy2>>connect[i][0]>>connect[i][1]>>connect[i][2]>>connect[i][3];
    // cout<<connect[i][0]<<"  "<<connect[i][1]<<" "<<connect[i][2]<<"  "<<connect[i][3]<<endl;

  }
  // while(true)
  // {
  //   getline(netgen_in,line);
  //   if (line == "edgesegmentsgi2") break;

  // }

  // int nedg;
  // netgen_in>>nedg;
  // // nedg=nedg/2;
  // vector<int> dummy3(10,0.0);
  // vector<vector<int> > edge;
  // edge.resize(nedg,vector<int>(2,0));
  // for(int i=0;i<nedg;i++)
  // {
  //   netgen_in>>dummy3[0]>>dummy3[1]>>edge[i][0]>>edge[i][1]>>dummy3[2]>>dummy3[3]>>dummy3[4]>>dummy3[5]>>dummy3[6]
  //   >>dummy3[7]>>dummy3[8]>>dummy3[9];
  //   // cout<<edge[i][0]<<"  "<<edge[i][1]<<endl;
  // }

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
    // cout<<setprecision(16)<<coord[i][0]<<"  "<<coord[i][1]<<"  "<<coord[i][2]<<endl;
  }

 //writtimg the MMG3D mesh
 ofstream mmg;
 mmg.open(mmg_mesh.c_str(),ios::out);

 mmg<<"MeshVersionFormatted 2"<<endl;
 mmg<<"Dimension"<<endl;
 mmg<<"3"<<endl;
 mmg<<"Vertices"<<endl;
 mmg<<np<<endl;

 for(int i=0;i<np;i++)
 {
   mmg<<coord[i][0]<<" "<<coord[i][1]<<" "<<coord[i][2]<<" "<<0<<endl;
 }

 mmg<<"Triangles"<<endl;
 mmg<<ntng<<endl;
 for(int i=0;i<ntng;i++)
 {
   mmg<<triang[i][0]<<" "<<triang[i][1]<<" "<<triang[i][2]<<" "<<bcnr[i]<<endl;
 }
 mmg<<" Tetrahedra"<<endl;
 mmg<<ne<<endl;
 for(int i=0;i<ne;i++)
 {
   mmg<<connect[i][0]<<" "<<connect[i][1]<<" "<<connect[i][2]<<" "<<connect[i][3]<<" "<<0<<endl;
 }

 mmg<<"End"<<endl;
 mmg.close();	
}