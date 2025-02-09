using namespace std;
#include "mesh_functions.h"
void OrderInterpolate (string in_mesh, string in_order, string out_mesh, string out_order) {

  int i;
  double dummy;
  fstream inmesh, outmesh;
  string line;
  //opening netgen mesh file
  inmesh.open(in_mesh.c_str(),ios::in);
  while(true){
    getline(inmesh,line);
    if (line == "surfaceelements") break;
  }

  int in_tris, in_nvmax;
  inmesh >> in_tris;
  int in_tripoints[in_tris][3];
  in_nvmax=1;
  i = 0;
  while (i < in_tris){
    inmesh >> dummy >> dummy >> dummy >> dummy >> dummy >> in_tripoints[i][0] >> in_tripoints[i][1] >> in_tripoints[i][2]; 
    in_nvmax=max(max(in_tripoints[i][0], in_tripoints[i][1]), max(in_tripoints[i][2],in_nvmax));
    i++;
  }

  int in_vol;
  while(true){
    getline(inmesh,line);
    if (line == "volumeelements") break;
  }
  inmesh >> in_vol;
  int in_volpoints[in_vol][4];
  i = 0;
  while (i < in_vol){
    inmesh >> dummy >> dummy >> in_volpoints[i][0] >> in_volpoints[i][1] >> in_volpoints[i][2] >> in_volpoints[i][3]; 
    i++;
  }
  // while(true){
  //   getline(inmesh,line);
  //   if (line == "edgesegmentsgi2") break;
  // }

  // int in_edges;
  // i=0;
  // while (i < in_edges){
  //   inmesh >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy;
  //   i++;
  // }

  while(true){
    getline(inmesh,line);
    if (line == "points") break;
  }
  int in_nv;
  inmesh >> in_nv;
  double in_points[in_nv][3];
  i=0;
  while(i < in_nv){
    inmesh >> in_points[i][0] >> in_points[i][1] >> in_points[i][2];
    i++;
  }
  inmesh.close();

  outmesh.open(out_mesh.c_str(),ios::in);
  while(true){
    getline(outmesh,line);
    if (line == "surfaceelements") break;
  }

  int out_tris, out_nvmax;
  outmesh >> out_tris;
  int out_tripoints[out_tris][3];
  out_nvmax=1;
  i = 0;
  while (i < out_tris){
    outmesh >> dummy >> dummy >> dummy >> dummy >> dummy >> out_tripoints[i][0] >> out_tripoints[i][1] >> out_tripoints[i][2]; 
    out_nvmax=max(max(out_tripoints[i][0], out_tripoints[i][1]), max(out_tripoints[i][2],out_nvmax));
    i++;
  }

  int out_out;
  while(true){
    getline(outmesh,line);
    if (line == "volumeelements") break;
  }
  outmesh >> out_out;
  int out_outpoints[out_out][4];
  i = 0;
  while (i < out_out){
    outmesh >> dummy >> dummy >> out_outpoints[i][0] >> out_outpoints[i][1] >> out_outpoints[i][2] >> out_outpoints[i][3]; 
    i++;
  }

  // while(true){
  //   getline(outmesh,line);
  //   if (line == "edgesegmentsgi2") break;
  // }
  // int out_edges;
  // i=0;
  // while (i < out_edges){
  //   outmesh >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy;
  //   i++;
  // }

  while(true){
    getline(outmesh,line);
    if (line == "points") break;
  }
  int out_nv;
  
  outmesh >> out_nv;


  double out_points[out_nv][3];
  i=0;
  while(i < out_nv){
    outmesh >> out_points[i][0] >> out_points[i][1] >> out_points[i][2];
    i++;
  }
  outmesh.close();

	fstream inorder, outorder;
	inorder.open(in_order.c_str(),ios::in);
	if (!inorder.is_open())
	{
		cout<<in_order<<" file not found."<<endl;
		exit(1);
	}
	int dummy1,comp,nnode, dim; 
	double dummy_real;
	inorder >> dim >> comp >> nnode >> dummy1;
	vector<double> in_nodeorder(nnode, 0.0); 
	if(in_nv != nnode)
	{
		cout << "Error : Number of nodes of the Mesh is NOT equal to that of Node-order file, OR put read_order to 0 in the PDE file" << endl;
		exit(1);
	}
	i=0;
	while (i < nnode)
	{
		inorder >> in_nodeorder[i];
		for(int k=0; k < comp-1; k++) inorder >> dummy_real;   
		i++;
	}
	inorder.close();

	// Do the interpolation here
	// for now let me just write a random order

	vector<double> out_nodeorder(out_nv, 2.0);
	// for(int j = 0; j < out_nv; j++)
	// 	out_nodeorder[j] = rand()%3 + 1;
  //  Looping over each of the nodes of the new mesh 
  vector<double> found_vec(out_nv, -1.0);
  for(int j = 0; j < out_nv; j++)
  {
    // Position of the new node
    vector<double> pos_loc(3, 0.0);
    for(int dd = 0; dd < 3; dd++)
    {   
      pos_loc[dd] = out_points[j][dd];
    }
    // Looping over each volume element from the old mesh
    for(int tr = 0; tr < in_vol; tr++)
    {
      if(found_vec[j] == -1.0 || found_vec[j] == 0.0)
      {
        vector<int> tet_pos(4, 0);
        vector<double> pos_tet(12, 0.0);
        vector<double> tet_order(4,0.0);
        for(int tet = 0; tet<4; tet++)
        {
          tet_pos[tet] = in_volpoints[tr][tet] - 1;
          int p = tet_pos[tet];
          for(int dd = 0; dd < 3; dd++)
          {
            pos_tet[tet*3 + dd] = in_points[p][dd];
          }
          tet_order[tet] = in_nodeorder[p];
        }
        // Check if the point j is in the tetrahedron tr
        double inside = is_inside(pos_tet, pos_loc, tet_order);
        found_vec[j] = inside;
      }
    }
    cout<<"Node found "<<j<<endl;
  }

  for(int j = 0; j < out_nv; j++)
  {
    if(found_vec[j] == -1.0)
    {
      cout<<"Something is wrong"<<endl;
      exit(1);
    }
    out_nodeorder[j] = max(found_vec[j], 1.0);
  }

	outorder.open(out_order.c_str(), ios::out);
    outorder << "3" << " " << "1" << " " << out_nv << " " << "2" << endl;
	outorder.precision(16);
	outorder.setf(ios::scientific, ios::floatfield);
	for (int j = 0; j < out_nv; j++)
	{
		double elemorder = out_nodeorder[j];
		outorder << elemorder << endl;
	}
	outorder.close();
}
