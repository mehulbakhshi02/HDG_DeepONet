#include <petscksp.h>
#include <petsc.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>

#include <cstring>
#include <cstdlib>

using namespace std;

#define MatOrderingType char*
#define MATORDERINGNATURAL     "natural"
#define MATORDERINGND          "nd"
#define MATORDERING1WD         "1wd"
#define MATORDERINGRCM         "rcm"
#define MATORDERINGQMD         "qmd"
#define MATORDERINGROWLENGTH   "rowlength"
#define MATORDERINGAMD         "amd"

Vec            x, rhs, adjrhs, adjrhs2, scale_left;
Mat            A;
KSP            ksp;
PC             pc;
int            gmres_max_it;
PetscErrorCode ierr;
PetscViewer    viewer;

// Constants
PetscScalar    zero_  = 0.0;
PetscScalar    m_one_ = -1.0;

int            size_sys;
int            size_blocks;
double         rel_tolerance;

bool solve_transpose = false;

double *vec_content;

bool first = true;

int * rows;
int * cols;
int * indices;

void petsc_initialize(int system_size, int max_local_size, int * d_nnz, bool show_monitor, int ksp_restart,
		      int pc_factor_levels, bool set_initial_guess = false, double rel_tol = 1e-5, 
		      int max_steps = 1e5) {

  size_sys        = system_size;

  gmres_max_it = max_steps;

  // ****************************************************
  // Set up the arguments. 
  if (first) {
    static char help[] = "This is my PetscWrapper!! \n\n";
    int argc_h = 3;
    char** args_h = new char*[argc_h];
    for(int n=0;n<argc_h;n++)
      args_h[n]=(char*) malloc(sizeof(char)*256);
    
    strcpy(args_h[0],"./prog");
    strcpy(args_h[1],"");
    strcpy(args_h[2],"");
    if (show_monitor) 
      strcpy(args_h[2],"-ksp_monitor");
      // strcpy(args_h[1],"-ksp_monitor_lg_residualnorm");
    //    strcpy(args_h[1],"-ksp_monitor_draw"); //-pc_hypre_boomeramg_relax_type_all SOR
    //    strcpy(args_h[2], "-log_summary");
    PetscInitialize(&argc_h,&args_h,(char *)0,help);
       // if (show_monitor) PetscOptionsSetValue("-ksp_monitor_lg_residualnorm",NULL);
    first = false;
  } else {
    /*    VecRestoreArray(x, &vec_content);
  
    VecDestroy(&x);
    VecDestroy(&scale_left);
    VecDestroy(&rhs);
    MatDestroy(&A);
    KSPDestroy(&ksp);
    
    delete[] rows;
    delete[] cols;*/
  }

  // ****************************************************
  // Create a solution vector
  VecCreate(PETSC_COMM_WORLD,&x);
  PetscObjectSetName((PetscObject) x, "Solution");
  VecSetSizes(x,PETSC_DECIDE,size_sys);
  VecSetFromOptions(x);

  // ****************************************************
  // Create right-hand side vector
  VecSet(x,0.0);
  VecDuplicate(x,&rhs);
  VecDuplicate(x, &scale_left);
  VecDuplicate(x, &adjrhs);
  VecDuplicate(x, &adjrhs2);
  
  rows    = new int[max_local_size];
  cols    = new int[max_local_size];
  indices = new int[size_sys];

  for (int i = 0; i < size_sys; i++)
    indices[i] = i;
  
  // ****************************************************
  // Set up matrices
  ierr = MatCreate(PETSC_COMM_WORLD,&A);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,size_sys,size_sys);
  ierr = MatSetType(A,MATAIJ);
  ierr = MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,d_nnz);
  ierr = MatSetFromOptions(A);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);

  // ****************************************************
  // Set KSP (KSPFMGRES is used because of square 
  // preconditioning. (See Petsc-Manual).
  KSPCreate(PETSC_COMM_WORLD,&ksp);
  KSPSetOperators(ksp,A,A);//,DIFFERENT_NONZERO_PATTERN); // uncomment for petsc version < 3.5
  KSPSetType(ksp,KSPGMRES);
  KSPSetPCSide(ksp,PC_RIGHT);
  KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization);
  KSPGMRESSetRestart(ksp,ksp_restart);
  KSPGetPC(ksp,&pc);

  // ****************************************************
  // Set preconditioner
  // ILU in the matrix explicit case.

  PCSetType(pc, PCILU);
  PCSetFromOptions(pc);
  PCFactorSetLevels(pc,pc_factor_levels);
  PCFactorSetMatOrderingType(pc, MATORDERINGRCM);
  rel_tolerance = rel_tol;
  KSPSetTolerances(ksp, rel_tolerance, PETSC_DEFAULT, PETSC_DEFAULT, max_steps);
  KSPSetFromOptions(ksp);
}

//void petsc_get_ordering() {

//}


void petsc_set_rel_tolerance(double tol) {
  KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);

}

void petsc_matrix_reset() {
  MatZeroEntries(A);
}

void petsc_rhs_reset() {
  VecZeroEntries(rhs);
  VecZeroEntries(adjrhs);
  VecZeroEntries(adjrhs2);
}

// Inserts a submatrix (fifth argument) of type m \times n (first two arguments) into the big
// matrix beginning at m_0, n_0 (third and fourth argument).
void petsc_insert_submatrix(int m, int n, int m_0, int n_0, double* submatrix) {

  for (int i = 0; i < m; i++)
    rows[i] = m_0 + i;
  
  for (int i = 0; i < n; i++)
    cols[i] = n_0 + i;

  MatSetValues(A,m,rows,n,cols,submatrix,INSERT_VALUES);
}

// Like insert_submatrix, but now it adds.
void petsc_add_submatrix(int m, int n, int m_0, int n_0, double* submatrix) {

  for (int i = 0; i < m; i++)
    rows[i] = m_0 + i;
  
  for (int i = 0; i < n; i++)
    cols[i] = n_0 + i;

  MatSetValues(A,m,rows,n,cols,submatrix,ADD_VALUES);

}

void petsc_add_vector(int m, int m_0, double* block) {

  for (int i = 0; i < m; i++)
    rows[i] = m_0 + i;

  VecSetValues(rhs, m, rows, block, ADD_VALUES);  
}

void petsc_add_adj_vector(int m, int m_0, double* block) {

  for (int i = 0; i < m; i++)
    rows[i] = m_0 + i;

  VecSetValues(adjrhs, m, rows, block, ADD_VALUES);  
}

// Sets the right hand side.
void petsc_set_rhs(double* rhs_vec) {

  VecSetValues(rhs, size_sys, indices, rhs_vec, INSERT_VALUES);
}

void petsc_rhs_scale(double scaling) {
  VecScale(rhs, scaling);
}

void petsc_set_startvector(double* x_vec) {

  VecSetValues(x, size_sys, indices, x_vec, INSERT_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

}


void petsc_show_matrix(void) {
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF , PETSC_VIEWER_ASCII_MATLAB);
  MatView(A, PETSC_VIEWER_STDOUT_SELF);
}

void petsc_show_rhs(void) {
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF , PETSC_VIEWER_ASCII_MATLAB);
  VecView(rhs, PETSC_VIEWER_STDOUT_SELF);
}


void petsc_show_adj_rhs(void) {
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF , PETSC_VIEWER_ASCII_MATLAB);
  VecView(adjrhs, PETSC_VIEWER_STDOUT_SELF);
}


void petsc_write_matrix(char * filename)
{
  PetscViewerFileSetName(viewer, filename);
  MatView(A, viewer);
}

void petsc_matrix_transpose(void) {
  MatTranspose(A,MAT_REUSE_MATRIX,&A);

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
}

// Beginn Allocation
void petsc_begin_mat_alloc(void) {
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
}

void petsc_begin_vec_alloc(void) {
  VecAssemblyBegin(rhs);
}

// End Allocation
void petsc_end_mat_alloc(void){
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
}

void petsc_end_vec_alloc(void) {
  VecAssemblyEnd(rhs);
}

void petsc_get_res_norm(double & res) {
  VecNorm(rhs, NORM_2, &res);
}

// Solves the system of equations
void petsc_solve_it(vector<double> & solution, int & its) {
  KSPSolve(ksp,rhs,x);
  its = 0;
  KSPGetIterationNumber(ksp, &its);
  VecGetArray(x, &vec_content);

  std::copy(vec_content, vec_content + size_sys, solution.begin());
}

void petsc_assemble_adj_rhs(vector<double> & adj, double & res) {
 
  VecSetValues(x, size_sys, indices, &adj[0], INSERT_VALUES);
  VecScale(x, -1.);
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
 
  VecAssemblyBegin(adjrhs);
  VecAssemblyEnd(adjrhs);  
 
  // J' - (N')^T * Psi
  MatMultTransposeAdd(A, x, adjrhs, adjrhs2);

  VecZeroEntries(x);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
  
  VecAssemblyBegin(adjrhs2);
  VecAssemblyEnd(adjrhs2);

  VecNorm(adjrhs2, NORM_2, &res);
}


void petsc_solve_adjoint(vector<double> & solution, int & its, double & res) {

  VecAssemblyBegin(adjrhs2);
  VecAssemblyEnd(adjrhs2);

  KSPSolveTranspose(ksp, adjrhs2, x);

  its = 0;
  KSPGetIterationNumber(ksp, &its);
  VecGetArray(x, &vec_content);

  for (int i = 0; i < size_sys; ++i)
    solution[i] += vec_content[i];
  
  res = 0.;
  KSPGetResidualNorm(ksp, &res);
}

void petsc_get_rhs(double * vec_rhs) {
  VecGetArray(rhs, &vec_rhs);
}

void petsc_get_adj_rhs(double * vec_rhs) {
  VecGetArray(adjrhs, &vec_rhs);
}

double petsc_test_solution() {

  PetscReal norm;

  MatMult(A, x, scale_left);

  VecNorm(scale_left, NORM_2, &norm);

  VecAXPY(scale_left, 1., rhs);
  VecNorm(scale_left, NORM_2, &norm);

  return norm;
}

// Finalizes Petsc
void petsc_finalize() {

  VecRestoreArray(x, &vec_content);
  VecDestroy(&x);
  VecDestroy(&scale_left);
  VecDestroy(&rhs);
  VecDestroy(&adjrhs);
  VecDestroy(&adjrhs2);
  MatDestroy(&A);
  KSPDestroy(&ksp);

  delete[] rows;
  delete[] cols;

   // PetscFinalize();
}





