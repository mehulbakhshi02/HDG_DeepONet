/*!
 * The class \c ElementData is decided by the dimensions \c D and number of
 * components \c COMP. It contains various properties related to the mesh element.
 */
template <int D, int COMP>
class ElementData {
 public:

  // Reconstructed shape functions
  /// The lagrangian shape functions. \f$\phi_{i}(\mathbf{x})\f$ at each integration point. Size = NIPxndof double
  vector<double> phi;           // [NIP, NDOFW]
  /// Shape functions for gradient variable.\f$[\phi_{i}(\mathbf{x})]^D\f$ Size = NIPxDxndof double
  vector<double> dphi;          // [NIP, D, NDOFW]
  /// The mass matrix. So the product of basis functions integrated over the cell using quadrature. \f$M_{ij}=\int_{K}\phi_{i}\phi_{j}dx\f$
  vector<double> mass;
  /// The inverse of the mass matrix. \f$M^{-1}\f$
  vector<double> inv_mass;
  // Reconstructed solution
  /// The solution \f$u\f$ reconstructed at integration points. Size = NIPxCOMP double
  vector<double> w;             // [NIP, COMP]
  /// The gradient of the solution \f$\nabla u\f$ reconstructed at integration points. Size = NIPxDxCOMP double
  vector<double> q;             // [NIP, D, COMP]

  // Quadrature
  /// The quadrature points \f$(x,y,z)\f$. Size = NIPxD double
  vector<double> qp;            // [NIP, D]
  /// The quadrature weights \f$w\f$. Size = NIP double
  vector<double> qw;            // [NIP]
  /// The wall distance at each quadrature point. Used only in case of turbulence. Size = NIP double
  vector<double> wd;

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

  // Connectivity
  /// The number of faces. Size = 1 int
  int nf;

  /// The index of the faces that form the element. Size = nf int
  vector<int> faces;

  /// The nodes that form the element. They are ordered as \f$(x_{1}, y_{1}, x_{2}, y_{x}\ldots,x_{n},y_{n})\f$ where n is the number of nodes that form the element. This changes based on the type of element used.
  vector<double> nodes;

  /// This variable decides the type of element. Size = 1 int
  int type;
  /// <b>DO NOT KNOW WHAT THIS VARIABLE IS</b>
  int elnr;
  /// This variable gives the number of dofs for \f$u\f$. Size = 1 int
  int ndof_w;
  /// This variable gives the number of dofs for \f$\nabla u\f$. Size =  1 int
  int ndof_q;
  /// This variable gives the number of dofs for \f$\lambda\f$. Size = 1 int
  int ndof_lt;
  /// This gives the offset of the element in the global \c vecW. It points to the beginning of the location where coefficients related to this element. Computed appropriately for our data structure. Size = 1 int
  int offset_w;
  
  /// This gives the offset of the current element in the global \c vecQ. It points to the beginning of the location where coefficients related to this element. Size = 1 int
  int offset_q;

  ///This variable gives the polynomial order in the element. Size = 1 int
  int order;
  /// This boolean gives the state if the element is curved or not. Size = 1 bool
  bool is_curved;

/*!
 * Initiates the variables of the class
 */
  ElementData(int elnr, int nip, int elnf, int ndof_w, int ndof_q) :
  elnr(elnr), nip(nip), nf(elnf), ndof_w(ndof_w), ndof_q(ndof_q),
  faces(elnf),
  phi(nip*ndof_w), dphi(nip*ndof_w*D),
  mass(ndof_w*ndof_w), inv_mass(ndof_w*ndof_w),
  w(nip*COMP), q(nip*COMP*D),
  qp(nip*D), qw(nip), wd(nip)
    //    nodes(elnf*D), nodeorder(elnf), nodeindex(elnf)
  {; 
  }

};
/*!
 * The class \c FaceData is decided by the dimensions \c D and number of
 * components \c COMP. It contains various properties related to the element of the mesh skeleton.
 */
template <int D, int COMP>
class FacetData {
 public:
  // Reconstructed shape functions
  /// This variable contains the shape functions for \f$u\f$. Size = NIPxndof double
  /// @{
  vector<double> phi1, phi2;    // [NIP,NDOFW]
  ///@}
  /// This variable contains the shape functions for \f$\lambda\f$. Size = NIPxndof_skeleton double
  vector<double> mu;            // [NIP,NDOFL]
  /// The mass matrix. So the product of basis functions integrated over the faces using quadrature. \f$M_{ij}=\int_{e}\phi_{i}\phi_{j}dx\f$
  vector<double> mass;
  /// The inverse of the mass matrix. \f$M^{-1}\f$
  vector<double> inv_mass;

  // Reconstructed solution, numerical flux
  /// The solution \f$u\f$ reconstructed at integration points on two
  /// sides of the face. Size = NIPxCOMP double
  ///@{
  vector<double> w1, w2;        // [NIP, COMP]
  ////@}
  /// The gradient of the solution \f$\nabla u\f$ reconstructed at the integration
  /// points. Both sides of the solution are reconstructed. Size = NIPxDxCOMP double
  ///@{
  vector<double> q1, q2;        // [NIP, D, COMP]
  ///@}
  
  /// The hybrid variable, \f$\lambda\f$, is reconstructed at the integration points. Size = NIPxCOMP double
  vector<double> lambda;        // [NIP, COMP]
  /// The normals of the faces. One of them is the outer normal and the other the inner
  /// normal. 
  /// @{
  vector<double> n1, n2;
  ///@}
  // Quadrature
  /// The quadrature points \f$(x,y)\f$. Size = NIPxD double
  vector<double> qp;            // [NIP, D]
  /// The quadrature weights \f$w\f$. Size = NIP double
  vector<double> qw;            // [NIP]
  /// The nodes that form the face. They are ordered as \f$(x_{1}, y_{1}, x_{2}, y_{x}\ldots,x_{n},y_{n})\f$ where n is the number of nodes that form the face. This changes based on the type of element used.
  vector<double> nodes;
  /// The size of each edge. Size = 1 double
  double h;
  /// The number of integration points. Size = 1 int
  int nip;
  /// This variable decides the type of face. Size = 1 int
  int type;

  // Topology
  /// This gives the element numbers which are on either side of the face. Size = 1 int
  ///@{
  int elnr1, elnr2;
  ///@}
  /// This gives the face center. Size = 1 int
  int facenr;

  /// This contains the dofs for the solution, \f$u\f$, on either side of the face. Size = 1 int
  /// @{
  int ndof_w1, ndof_w2;
  ///@]
  /// This contains the dofs for the solution, \f$\nabla u\f$, on either side of the face. Size = 1 int
  ///@{
  int ndof_q1, ndof_q2;
  ///@]
  /// This contains the dofs for \f$\lambda\f$. Size = 1 int
  int ndof_l;
  /// This contains the offset for the solution \f$u\f$ on element on either side. Size = 1 int
  ///@{
  int offset_w1, offset_w2;
  ///@}
  /// This contains the offset for the solution \f$\nabla u\f$ on element on either side. Size = 1 int
  ///@{
  int offset_q1, offset_q2;
  ///@}
  /// This contains the offset for the solution \f$\lambda\f$ on element on either side. Size = 1 int
  int offset_l;
  ///This variable gives the polynomial order in the element on either side. Size = 1 int
  ///@{
  int order1, order2;
  ///@}
  ///This variable gives the polynomial order for \f$\lambda\f$. Size = 1 int
  int order_l;
  /// This gives the type of boundary. <b>DO NOT KNOW WHAT THIS IS</b>. Size = 1 int
  int bcnr;
  /// This boolean gives the state if the face is curved or not. Size = 1 bool
  bool is_curved;
/*!
 * Initializes the members of the class faces
 */
 FacetData(int facenr, int nip, int el1, int el2, int ndof_w1, int ndof_w2, 
            int ndof_q1, int ndof_q2, int ndof_l) :            
  nip(nip), elnr1(el1), elnr2(el2), facenr(facenr), ndof_w1(ndof_w1), ndof_w2(ndof_w2),
  ndof_q1(ndof_q1), ndof_q2(ndof_q2), ndof_l(ndof_l), 
  phi1(nip*ndof_w1), phi2(nip*ndof_w2), 
  mu(nip*ndof_l),
  mass(ndof_l*ndof_l), inv_mass(ndof_l*ndof_l),
  w1(nip*COMP), w2(nip*COMP), //q1(nip*COMP*D), q2(nip*COMP*D), 
  lambda(nip*COMP), 
  n1(nip*D), n2(nip*D), qp(nip*D), qw(nip),
  bcnr(-1)
  {; }

};
