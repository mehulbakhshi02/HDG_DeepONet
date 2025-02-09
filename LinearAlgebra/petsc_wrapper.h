

bool show_monitor = false;
int ksp_restart = 30;
int pc_factor_levels = 0;
double rel_tol = 1.e-4;
int max_steps = 1000;

// Initializes Petsc
void petsc_initialize(int system_size, int max_local_size, int * d_nnz, bool show_monitor, int ksp_restart,
		      int pc_factor_levels, bool set_initial_guess = false, double rel_tol = 1e-5, 
		      int max_steps = 1e5);



void petsc_set_rel_tolerance(double);

// Sets the matrix entries to 0.
void petsc_matrix_reset(void);
void petsc_rhs_reset(void);

// Inserts a submatrix (fifth argument) of type m \times n (first two arguments) into the big
// matrix beginning at m_0, n_0 (third and fourth argument).
void petsc_insert_submatrix(int, int, int, int, double*);

// Like insert_submatrix, but now it adds.
void petsc_add_submatrix(int, int, int, int, double*);

void petsc_add_vector(int, int, double*);
void petsc_add_adj_vector(int, int, double*);

// Sets the right hand side.
void petsc_set_rhs(double*);

void petsc_rhs_scale(double scaling);
void petsc_set_startvector(double* x_vec);

// Beginn Allocation
void petsc_begin_mat_alloc(void);
void petsc_begin_vec_alloc(void);

// End Allocation
void petsc_end_mat_alloc(void);
void petsc_end_vec_alloc(void);

void petsc_show_matrix(void);
void petsc_show_rhs(void);
void petsc_show_adj_rhs(void);
void petsc_write_matrix(char * filename);

void petsc_get_res_norm(double & res);

void petsc_matrix_transpose(void);

// Solves the system of equations
void petsc_solve_it(vector<double> & solution, int & its);
void petsc_solve_adjoint(vector<double> & solution, int & its, double & res);
void petsc_assemble_adj_rhs(vector<double> & adj, double & res);

// Read the right hand solution
void petsc_get_rhs(double * vec_rhs);
void petsc_get_adj_rhs(double * vec_rhs);

// Finalizes Petsc
void petsc_finalize(void);

// Just for testing purposes!
double petsc_test_solution(void);
