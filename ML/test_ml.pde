################################ FLOW PARAMETERS ################################

define constant epsilon = 0.1
define constant alpha = 1.0
define constant beta = 1.0

################################ DISCRETIZATION PARAMETERS ######################

define constant order = 2
#define constant min_order = 0
#define constant max_order = 2
define constant stab_conv = 1
define constant stab_visc = 1

################################ SOLVER PARAMETERS ##############################

define constant pcfl_min = 1e20
define constant pcfl_max = 1e20
define constant pcfl_beta = 2
define constant newton_n = 100
define constant newton_res = 1.e-10
define constant show_residuals = 1

################################ ML PARAMETERS ##################################

define constant train = 1

# Addition of quadrature order (total order 2*max_order+quadadd)
define constant quadadd = 8

# Defining ML parameters
## hidden_layers = 5
## activation_func = swish
## number_neurons = 50
## epochs = 300
## batch_size = 4096
## validation_split = 0.1

################################ PETSC PARAMETERS ###############################

define constant show_monitor = 0
define constant ksp_restart = 120
define constant pc_factor_levels = 3
define constant rel_tol = 1.e-3
define constant max_steps = 1000

################################ MESH & GEOMETRY ################################

geometry = "netgen.in2d"
mesh = "netgen.vol"

################################ ADAPTATION #####################################

define constant adaptation = 1
define constant adjoint_based = 1
define constant hp = 0
define constant read_order = 0
define constant min_order_adap = 2
define constant max_order_adap = 2
define constant save_mesh_metric = 1
define constant save_adj_metric = 1
define constant read_bamg_order = 1
define constant save_bamg_order = 1
define constant read_bamg_solution = 0
define constant save_bamg_solution = 0
define constant write_cellwise_err = 0

define constant dof_control = 1
define constant dof_target = 2000
define constant comp_der = 0
define constant limit_metric = 0
define constant beta_max = 1.0

define constant norm = 1

define constant adjoint_write_error = 1
define constant adjoint_nit = 5

################################ ARAVIND BALAN'S PARAMETERS #####################

define constant rlim = 2
define constant clim = 2

define constant ref_err = 0.3
define constant r_max = 20
define constant c_max = 5

################################ ADAPTATION ALGORITHM ##########################

define constant excoe_ani_tol = 1e+40
# 0 - Mesh fraction, 1 - Analytic optimization, 2 - Numerical optimization
define constant opt_strategy = 1

################################ PROJECTION BASIS ###############################

# 1 - Dubiner basis(only for 2D), 2 - Monomial basis(both 2D and 3D)
define constant projection_basis = 2

################################ MESH GENERATOR #################################

# 1 - BAMG, 2 - Angener, 3 - MMG2d, 4 - Omega_h, 5 - MMG3d, 6 - Refine
define constant mesh_generator = 1
define constant sol_write_error = 1

# Mesh curving - 0 - Mesh not curved based on the metric, 1 - Mesh curving based on metric

################################ POSTPROCESSING ################################

define constant output = tecplot
define constant visualize_adjoint = 1

################################################################################

define constant heapsize = 1000000000
shared = "./main"

################################ NUMERICAL MODEL ###############################

#numproc advection2d advection2d
#numproc advection3d advection3d
#numproc poisson2d poisson2d
#numproc helmholtz2d helmholtz2d
#numproc poisson3d poisson3d
#numproc euler3d euler3d
#numproc euler2d euler2d

#numproc navierstokes2d navierstokes2d
#numproc navierstokes3d navierstokes3d
#numproc komega2d komega2d
#numproc sa2d sa2d
#numproc michigan12d michigan12d
numproc advectiondiffusion2d advectiondiffusion2d
#numproc advectiondiffusion3d advectiondiffusion3d
#numproc simplesystem2d simplesystem2d
#numproc advection2d advection2d
#numproc shallowwater2d shallowwater2d

################################################################################

numproc visualization vis -scalarfunction=w  -subdivision=2  -autoscale -nolineartexture
# -clipsolution=scalar -clipvec=[0,0,-1] -clipdist=-.004

