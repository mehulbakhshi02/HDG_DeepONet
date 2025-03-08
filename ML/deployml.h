extern"C" {
    void table_delaunay_wrapper_(int*, double*, int*, int*);
    void table_tet_wrapper_(int*, double*, int*, int*);
}

template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>::DeployML(std::vector<double> & u_coordinates, std::vector<double> & u_sol) {

  // Transformed adjoint coordinates picked up from trans_adj_coordinates.txt file
  
  std::ifstream file("ML/trans_adj_coordinates.txt");

    std::vector<double> trans_adj_coordinates;  // Vector to store separated double values
    std::string line;

    while (std::getline(file, line)) {  // Read each line
        std::stringstream ss(line);  // Convert line to stringstream
        std::string value;
        
        while (std::getline(ss, value, ',')) {  // Split by comma
            try {
                trans_adj_coordinates.push_back(std::stod(value));  // Convert to double and store
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number found: " << value << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Number out of range: " << value << std::endl;
            }
        }
    }

    cout << "trans_adj_coordinates_size: " << trans_adj_coordinates.size() << endl;

    file.close();

  // u solution normalization

  vector<double> norm_u_sol;
  int u_sol_size = u_sol.size()/ne;
  int k = 0;

  for (int pp = 0; pp < ne; ++pp){
      for (int ll=0; ll < u_sol_size; ++ll){
        norm_u_sol.push_back(u_sol[k]);
        k = k+1;
      }
    }

  // Initialize Python
  Py_Initialize();

  int PyCheckFlag = 0;
  PyCheckFlag = Py_IsInitialized();
  printf("PyCheckFlag: %d\n",PyCheckFlag);
  
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"./ML/\")");
  _import_array();
  std::cout << "Initializing numpy library" << std::endl;

  // Import the Python script (without .py extension)
  PyObject* pName = PyUnicode_DecodeFSDefault("deployment");
  PyObject* pModule = PyImport_Import(pName);
  Py_XDECREF(pName); // Clean up

  vector<vector<double>> ml_adjoint;  // 2D C++ vector to store Python list

  if (pModule != nullptr) {
      // Get the function from the module
      PyObject* pFunc = PyObject_GetAttrString(pModule, "process_data");

      if (PyCallable_Check(pFunc)) {
          // Convert C++ vectors to Python lists
          PyObject* pyList_trans_adj_coordinates = PyList_New(trans_adj_coordinates.size());
          PyObject* pyList_norm_u_sol = PyList_New(norm_u_sol.size());

          for (size_t i = 0; i < trans_adj_coordinates.size(); ++i) {
              PyList_SetItem(pyList_trans_adj_coordinates, i, PyFloat_FromDouble(trans_adj_coordinates[i]));
          }

          for (size_t i = 0; i < norm_u_sol.size(); ++i) {
              PyList_SetItem(pyList_norm_u_sol, i, PyFloat_FromDouble(norm_u_sol[i]));
          }

          // Create tuple of arguments
          PyObject* pArgs = PyTuple_Pack(2, pyList_trans_adj_coordinates, pyList_norm_u_sol);

          // Call the Python function
          PyObject* pValue = PyObject_CallObject(pFunc, pArgs);

          // Clean up
          Py_XDECREF(pArgs);
          Py_XDECREF(pyList_trans_adj_coordinates);
          Py_XDECREF(pyList_norm_u_sol);

            if (pValue != nullptr) {
                if (PyList_Check(pValue)) {
                    Py_ssize_t dim1_size = PyList_Size(pValue);
                    ml_adjoint.resize(dim1_size);

                    for (Py_ssize_t i = 0; i < dim1_size; ++i) {
                        PyObject* pySubList1 = PyList_GetItem(pValue, i);
                        if (!PyList_Check(pySubList1)) continue;

                        Py_ssize_t dim2_size = PyList_Size(pySubList1);
                        ml_adjoint[i].resize(dim2_size);

                        for (Py_ssize_t j = 0; j < dim2_size; ++j) {
                            PyObject* item = PyList_GetItem(pySubList1, j);
                                if (PyFloat_Check(item)) {
                                    ml_adjoint[i][j] = PyFloat_AsDouble(item);
                                }
                        }
                    }
                }

                Py_XDECREF(pValue); // Clean up return value
            } else {
                PyErr_Print();
            }
        } else {
            std::cerr << "Function not found or not callable!" << std::endl;
            PyErr_Print();
        }

        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
    } else {
        std::cerr << "Failed to load module!" << std::endl;
        PyErr_Print();
    }

  // Finalize Python
  Py_Finalize();

    // Write the adjoint solution to a file (Hard-coded)
    //cout << "ml_adjoint dimensions: " << ml_adjoint.size() << " x " << (ml_adjoint.empty() ? 0 : ml_adjoint[0].size()) << endl;

    stringstream oss(" ");
    oss << "ML-solution-adjoint-" << ne << "-" << max_order;
    Model::GetFilename(oss);
    string filename(oss.str());

    string fname_dat(filename);
    fname_dat.append(".dat");

    fstream output;
    output.open(fname_dat.c_str(),ios::out);

    output << "TITLE = DGD" << endl;
    output << "VARIABLES = \"X\" \"Y\" ";
    if (D == 3)
        output << "\"Z\" ";
    output << "\"W1\"";
    output << endl;

    // Calculate a triangulation of the unit triangle

    int nlt = 2 * (order+2); // adjusted for adjoint
    int npt = 0.5 * (nlt + 2) * (nlt + 1);

    vector<double> xllt(2*npt);

    int ll = 0;
    for (int j = 0; j <= nlt; j++) {
        double rl1 = 1.*j / (1. * nlt);
        for (int k = 0; k <= nlt; k++) {
        double rl2 = 1. * k / (1. * nlt);
        if ( (rl2 + rl1) < 1. + 1e-9) {	
            double rl3 = 1. - rl2 - rl1;      
            xllt[2*ll] = rl2;
            xllt[2*ll+1] = rl3;  
            ll++;
        }
        }
    }

    vector<int> nlct(9*npt);
    int ntr = 0;

    table_delaunay_wrapper_(&npt, &xllt[0], &ntr, &nlct[0]);

    int m1 = 0;
	int m2 = 1;
	int u_coordinate_size = u_coordinates.size()/ne;
    int adj_coordinate_size = trans_adj_coordinates.size();


    for(int i=0; i<ne; i++){
        char res_out[160];
        sprintf(res_out, "Zone T=\"INTERIOR\" N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", npt, ntr);
        output << res_out << endl;

		vector<double> x(u_coordinate_size/2);
		vector<double> y(u_coordinate_size/2);

		for(int j=0; j<u_coordinate_size/2; j++)
		{
			x[j] = u_coordinates[m1];
			y[j] = u_coordinates[m2];
			m1=m1+2;
			m2=m2+2;
		}
        
        double x1 = x[0];
		double y1 = y[0];
		double x2 = x[6];
		double y2 = y[6];
		double x3 = x[27];
		double y3 = y[27];

        double x21 = x2 - x1, x31 = x3 - x1;
        double y21 = y2 - y1, y31 = y3 - y1;

        int n1 = 0;
        int n2 = 1;

        // Writing .dat file

        for (int j=0; j<adj_coordinate_size/2; j++)
        {
            double xi = trans_adj_coordinates[n1];
            double eta = trans_adj_coordinates[n2];
            n1 = n1+2;
            n2 = n2+2;

            double adjx = x1 + x21 * xi + x31 * eta;
            double adjy = y1 + y21 * xi + y31 * eta;

            sprintf(res_out, "%10.10f %10.10f %10.10f", adjx, adjy, ml_adjoint[i][j]);
            output << res_out << endl;
        }

        // Writing the triangulation

        for (int j = 0; j < ntr; j++) 
        {
            sprintf(res_out, "%i %i %i", nlct[3*j], nlct[3*j+1], nlct[3*j+2]);
            output << res_out << endl;
        }
        
    }

    output.close();
    
}
