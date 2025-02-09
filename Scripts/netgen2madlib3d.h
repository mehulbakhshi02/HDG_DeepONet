using namespace std;

void Netgen2Madlib3D(string netgen_mesh, string mmg_mesh) 
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
  }
  int ne;
  netgen_in>>ne;
  int dummy1,dummy2;
  vector<vector<int> > connect;
  connect.resize(ne,vector<int>(4,0.0));
  for(int i=0;i<ne;i++)
  {

    netgen_in>>dummy1>>dummy2>>connect[i][0]>>connect[i][1]>>connect[i][2]>>connect[i][3];

  }
  while(true)
  {
    getline(netgen_in,line);
    if (line == "edgesegmentsgi2") break;

  }

  int nedg;
  netgen_in>>nedg;
  // nedg=nedg/2;
  vector<int> dummy3(10,0.0);
  vector<vector<int> > edge;
  edge.resize(nedg,vector<int>(2,0));
  for(int i=0;i<nedg;i++)
  {
    netgen_in>>dummy3[0]>>dummy3[1]>>edge[i][0]>>edge[i][1]>>dummy3[2]>>dummy3[3]>>dummy3[4]>>dummy3[5]>>dummy3[6]
    >>dummy3[7]>>dummy3[8]>>dummy3[9];
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

 mmg<<"$MeshFormat\n2.1 0 8\n$EndMeshFormat"<<endl;
 mmg<<"$Nodes"<<endl;
 mmg<<np<<endl;
  vector<int> np_bc_val(np, 0);
  int j=0;

 for(int i=0;i<np;i++)
 {
   mmg<<i+1<<" "<<coord[i][0]<<" "<<coord[i][1]<<" "<<coord[i][2]<<endl;
 }
  mmg<<"$EndNodes"<<endl;
  mmg<<"$Elements"<<endl;
  mmg<<nedg+ntng+ne<<endl;
  j=0;
  while (j < nedg){
    mmg <<j+1 <<" "<< 1 << " " << 3 << " "<< 0<< " " << 1 << " " << 0<< " " << edge[j][0] << " " << edge[j][1] << endl;
    j++;
  }
 for(int i=0;i<ntng;i++)
 {
   mmg<<nedg+1+i<<" "<<2<<" "<<3<<" "<<0<<" "<<bcnr[i]<<" "<<0<<" "<<triang[i][0]<<" "<<triang[i][1]<<" "<<triang[i][2]<<endl;
 }
 for(int i=0;i<ne;i++)
 {
   mmg<<nedg+ntng+1+i<<" "<<4<<" "<<3<<" "<<1<<" "<<1<<" "<<0<<" "<<connect[i][0]<<" "<<connect[i][1]<<" "<<connect[i][2]<<" "<<connect[i][3]<<endl;
 }
 mmg<<"$EndElements"<<endl;
 mmg.close();
}
