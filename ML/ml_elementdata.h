/*!
 * The class \c ElementData is decided by the dimensions \c D and number of
 * components \c COMP. It contains various properties related to the mesh element.
 */
template <int D, int COMP>
class ML_ElementData {
 public:

  // // Reconstructed solution
  // /// The solution \f$u\f$ reconstructed at integration points. Size = NIPxCOMP double
  vector<double> w;             // [NIP, COMP]
  // /// The gradient of the solution \f$\nabla u\f$ reconstructed at integration points. Size = NIPxDxCOMP double
  vector<double> q;             // [NIP, D, COMP]
 /// The solution coefficients for the given element. Size = ndof_wxCOMP double
  vector<double> vecW;
  
 /// The gradient coefficients for the given element. Size = ndof_wxCOMP double
  vector<double> vecQ;

  /// The size of each edge. Size = 1 double
  double h;
  /// The surface area of each face. Size = 1 double
  double surf;
  /// The volue of the element. Size = 1 double
  double vol;
  /// The coordinate of the center of the cell. Used for the computation of the \f$p+1^{st}\f$ derivatives. Size = D double
  double cntr[D];
  /// The number of integration points. Size = 1 int
  int nip;

  /// Aspect ratio only for 2D 
  double beta;
  /// Angle of rotation only for 2D
  double theta;
  // Connectivity
  /// The number of faces. Size = 1 int
  int nf;

  /// The index of the faces that form the element. Size = nf int
  vector<int> faces;
  /// total number of non-boundary faces for the element (initialized to 0)
  int nf_bc;
/// <b>DO NOT KNOW WHAT THIS VARIABLE IS</b>
  int elnr;
  /// This variable gives the number of dofs for \f$u\f$. Size = 1 int
  int ndof_w;

  /// The quadrature points \f$(x,y,z)\f$. Size = NIPxD double
  vector<double> qp;            // [NIP, D]


/*!
 * Initiates the variables of the class
 */
  ML_ElementData(int elnr, int elnf, int nip, int ndof_w) :
  elnr(elnr), nip(nip), ndof_w(ndof_w), nf(elnf), faces(elnf),nf_bc(0),
 vecW(ndof_w*COMP), vecQ(ndof_w*D*COMP), w(nip*COMP), q(nip*COMP*D), qp(nip*D)
  {; 
  }
// w(nip*COMP), q(nip*COMP*D), 

};



template <int D, int COMP>
class ML_FacetData {
 public:
  vector<double> w_jump;        // [NIP, COMP]
  ////@}
  /// The gradient of the solution \f$\nabla u\f$ reconstructed at the integration
  /// points. Both sides of the solution are reconstructed. Size = NIPxDxCOMP double
  ///@{
  vector<double> q_jump;        // [NIP, D, COMP]
  ///@}
  

  /// The size of each edge. Size = 1 double
  double h;
  /// The number of integration points. Size = 1 int
  int nip;

  // Topology
  /// This gives the element numbers which are on either side of the face. Size = 1 int
  ///@{
  int elnr1, elnr2;
  ///@}
  /// This gives the face center. Size = 1 int
  int facenr;

/*!
 * Initializes the members of the class faces
 */
 ML_FacetData(int facenr, int nip, int el1, int el2) :            
  nip(nip), elnr1(el1), elnr2(el2), facenr(facenr), 
  w_jump(nip*COMP), q_jump(nip*COMP*D)
  {; }

};

template <int D, int COMP>
class ML_OutputData {
 public:

  // /// The solution \f$u\f$ reconstructed at integration points. Size = NIPxCOMP double
  vector<double> w;             // [NIP, COMP]
 /// The solution coefficients for the given element. Size = ndof_wxCOMP double
  vector<double> vecW;

  int elnr;
  /// This variable gives the number of dofs for \f$u\f$. Size = 1 int
  int ndof_w;
  // Containts the number of integration points
  int nip;

  double error;
/*!
 * Initiates the variables of the class
 */
  ML_OutputData(int elnr, int nip, int ndof_w) :
  elnr(elnr), nip(nip), ndof_w(ndof_w),
 vecW(ndof_w*COMP), w(nip*COMP), error(0.0)
  {; 
  }
// w(nip*COMP), q(nip*COMP*D), 

};


template <int D, int COMP>
class ML_DeployData {
 public:

 // /// The solution \f$u\f$ reconstructed at integration points. Size = NIPxCOMP double
  vector<double> w;             // [NIP, COMP]
 
 // The solution coefficients for the given element. Size = ndof_wxCOMP double
  vector<double> vecW;

  int elnr;
  /// This variable gives the number of dofs for \f$u\f$. Size = 1 int
  int ndof_w;
  // Containts the number of integration points
  int nip;        

  // error predicted
  double error;
/*!
 * Initiates the variables of the class
 */
  ML_DeployData(int elnr, int nip, int ndof_w) :
  elnr(elnr), nip(nip), ndof_w(ndof_w), 
 vecW(ndof_w*COMP), w(nip*COMP), error(0.0)
  {; 
  }
// w(nip*COMP), q(nip*COMP*D), 

};