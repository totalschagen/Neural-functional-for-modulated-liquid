// tunable parameters
int lastDigits;
double packing_fraction;		// packing_fraction
double density;						// density
double Lperiod;					// period of the potential
double Amp;						// amplitude of the potential
int nperiods;					// eventually set the number of periods and from this fix Lperiod
double mu;

// system parameter
double T;						// temperature at which simulation is performed
double radius;					// particles radius
int d;							// to determine the size of the box Lx = Ly = d*sqrt(PI)/2
double Lx,Ly;					// box_size
int N;							// number of particles, actually occupied array indices
int N0;							// number of particles, array length, N>N0


// MC simulation parameters
double step_size, step_size_xshift;		// step_size for MC moves
int Neq, Nsim, Nupdate, Nprint;			// Number of equilibrations swaps, number of simulation swaps, update observables every Nupdate swaps, print every Nprint swaps

// Observable rho(x)
int Nbins;							// Number of bins in the density profile

// MC simulation counters
int Nacc_particle, Nmc_particle;		// Counters to calculate acceptance ratio for particles moves
int Nacc_insert, Nmc_insert;		// Counters to calculate acceptance ratio for particles inserts
int Nacc_remove, Nmc_remove;		// Counters to calculate acceptance ratio for particles removes
int Nacc_xshift, Nmc_xshift;			// Counters to calculate acceptance ratio for particles moves
int count_rhoxavg;
int count_rho_q;
int Nmsd;
double MSDx,MSDy,MSD;

// arrays initialization
double * rhoxavg;
double * c1;
double * x;			// x and y positions of the particles
double * y;
double * x00;			// x and y positions of the particles
double * y00;
int * ix;			// to take into account the number of times a particles cross boundaries
int * iy;
double * xx0;
double * yy0;
int * ix0;
int * iy0;


// cell list
int Ncell, Nmax;						// constants for cell list calculation
int * cell_list_index;
int **cell_list;

// functions

void run_simulation(const char *eta_print,const char *N_print,int file_count); 
void pbc(double &x, double &y, int &ix, int &iy);
void pbc_distance(double &x, double &y);
int lattice_packing();
void mc_particle_move();
void mc_xshift();
void mc_step();
void mc_swap();
void mc_insert();
void mc_remove();
void GC_mc_swap(); 

void update_rhoxavg ();
void update_rhoxavg_2D();

int determine_overlap(double xnew, double ynew, int i);
int determine_global_overlap_naive();
void create_cell_lists();
void update_cell_lists(int i, double x_old, double y_old, double x_new, double y_new);


void check_particle_0();
bool check_cell_lists();

void print_gr(int step);
void print_positions(int step);
void print_rhox_avg_c1(const char *cstr,int nperiod);

void print_N(const char *cstr,int step);
void print_eta(const char *cstr, double packing_fraction, double mu, double density);



// debugging
bool check0;

