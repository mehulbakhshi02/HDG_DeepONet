/** @file
 * @brief This function will read the order and make it an
 * integer. The actually computed orders after interpolation
 * might be real numbers. Because of this issue
 * we need to put another check to set the order.
* @param[in] - filename - File name to read the order. Size = ne double
 * @param[out] - eleorder - The element order for the new mesh. Size = ne double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ReadOrder(const string & filename, vector<int> & eleorder) {
	ifstream file(filename.c_str(), ios::in);

	if (!file.is_open())
	{
		cout<<filename<<" file not found. Set read_order to 0 or provide the file"<<endl;
		exit(1);
	}

	// Mesh curving options
	int mesh_points = ma->GetNV();
	if(AnisotropyData::curved_mesh)
	{
	  mesh_points = mesh_points - nf;
	}
	// Order of the nodes
	vector<double> nodeorder(mesh_points, 0.0); 

	int dummy,comp,nnode; 
	double dummy_real;

	if(D==2)
	{
	    file >> dummy >> comp >> nnode >> dummy;
	}
	if((AnisotropyData::mesh_generator == AnisotropyData::refine || AnisotropyData::mesh_generator == AnisotropyData::mmg3d)&& D==3)
	{
		string dummy_text;
	    getline(file,dummy_text);// Take first few lines as they are just dummy text
	    getline(file,dummy_text);
	    getline(file,dummy_text);
	    getline(file,dummy_text);
	    getline(file,dummy_text);
		file >> nnode;
		// cout<<nnode<<" No of nodes "<<mesh_points<<endl;
	    getline(file,dummy_text);
	}	
	if(mesh_points != nnode) 
	{
		cout << "Error : Number of nodes of the Mesh is NOT equal to that of Node-order file, OR put read_order to 0 in the PDE file" << endl;
		exit(1);
	}
	int i=0;
	while (i < nnode)
	{
		file >> nodeorder[i];
		for(int k=0; k < comp-1; k++) file >> dummy_real;   
		i++;
	}
	file.close();
	// Setting the order from vertices to number of cells
	cout << string(32, '*') << endl;
	cout << "Reading new order" << endl;
	cout << string(32, '*') << endl;
	ne = ma->GetNE();

	for(int i = 0; i < ne; i++) 
	{
		Array<int> vnums;
	    vnums = ma->GetElVertices(ElementId(VOL,i));
		double eorder = 0.0;
		for(int j = 0; j < vnums.Size(); j++)
		{
			int n = vnums[j];
			eorder += (double)nodeorder[n];
		}
		double sz = vnums.Size();
		eorder = eorder/sz;
		double x,y;
		y = modf(eorder,&x);
		eleorder[i] = (y > 0.5) ? ceil(eorder) : floor(eorder);
		eleorder[i] = max(AnisotropyData::min_order_adap, eleorder[i]);
		eleorder[i] = min(AnisotropyData::max_order_adap, eleorder[i]);
		// cout<<"Element order is "<<eleorder[i]<<endl;
	}
}
