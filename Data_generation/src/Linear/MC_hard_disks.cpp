// compile with: g++ MC_hard_disks.cpp nrutil.cpp -w -O2 -DsSEED_IN=23451 -DAmp_in=1.7 -DSLX=30 -DRHO=0.7 -DLLL=10 -o ./MC_run.exe


#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <csignal>
#include <omp.h>

#include "MC_hard_disks.h"
#include "nrutil.h"

#define PI 3.14159265358979323846264338

#ifndef sSEED_IN
#define sSEED_IN 12345
#endif

#ifndef Amp_in
//#define Amp_in 0.5
#endif

#ifndef LLL //is number
//#define LLL 30
#endif

#ifndef SLX // Lx
#define SLX 15
#endif
//period = 30/30 = 1.0

#ifndef RHO
#define RHO (0.6)
#endif

#ifndef MU
#define MU (14.00)
#endif

//#define TESTGC 

//#define BULK

long long rnumber;
int rnd;


int main() {

  bool CL_check;
  // system parameters (NOT TO BE CHANGED)
  T = 1.0;
  radius = 0.5;
  Lx = SLX;
  Ly = Lx;
 
  // initialize random number generator
  rnd = 5;
  rnumber = time(NULL);
  rnumber = rnumber + sSEED_IN;
  long long divisor = 1;
  for (int i = 0; i < rnd; ++i) {
    divisor *= 10;
  }
  lastDigits = sSEED_IN; // rnumber % divisor;
  // lastDigits = lastDigits + 12345;
  srand(lastDigits);
  printf("rand SEED: %d\n", lastDigits); 

//////////////////////////////////////////////////////////////////////
 //prepare file for packing fraction and particle number  
  
  std::stringstream sstr2;
  sstr2 << "Output/" << "packing_fraction"  << ".dat";
  const std::string tmp2 = sstr2.str();
  const char *cstr4 = tmp2.c_str();
  FILE *fp2 = fopen(cstr4, "w");
  fprintf(fp2,"eta ; mu ; rho \n");
  fclose(fp2);

  std::stringstream sstr;
  sstr << "Output/"<< "N"  << ".dat";
  const std::string tmp = sstr.str();
  const char *cstr = tmp.c_str();
  FILE *fp = fopen(cstr, "w");
  fprintf(fp,"step  N  rho  mu \n");
  fclose(fp);

  ////////////////////////////////////////////////////////////////////
  // tunable parameters
  
  for(int m=1;m<2;m++){
    run_simulation(cstr4,cstr,m); 
  }
}
void run_simulation(const char *eta_print,const char *N_print,int file_count) {
  clock_t begin = clock();
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);  
  timeinfo = localtime(&rawtime);
  printf("Strated at local time and date: %s", asctime(timeinfo));


  bool CL_check;
  density = 3*0.1;
  float r2 = 2*radius;
  int LLL_min = int(Lx / 2*r2);
  int LLL_max = int(Lx / 0.2*r2);
  int LLL = LLL_min + rand() % (LLL_max - LLL_min + 1);


  Lperiod = double(Lx * 1.0) / double(LLL *1.0); 
  nperiods = int((Lx + 0.000000001) / Lperiod);
  nperiods= 55;
  Lperiod = Lx / nperiods;
  Amp = 0.0;//initially for mixing

  double mu_min = -2.0;
  double mu_max = 8.0;
  mu = mu_min + (rand()/ (double)RAND_MAX) * (mu_max - mu_min);
  mu = 7.9;
  //if(j==18) check0=true;
  ////////////////////////////////////////////////////////////////////


  //print_seed();
  // srand (12345);

  // system parameters which follow from the tunable parameters
  N = int(Lx * Ly * density + 0.000000000001);
  N0 = 1000;
  packing_fraction = density * PI * radius * radius;

  // MC simulation parameters
  step_size = 0.01;
  step_size_xshift = 0.01;
  int N_mixing = 20000;   
  Neq = 1000000;
  Nsim = 10000001;
  Nprint = 5000;
  // int N_mixing = 200;   
  // Nprint = 100;
  // Neq = 100;
  // Nsim = 1000;
  Nupdate = 100;
  int t0 = 0;
  Nmsd = Nsim / Nprint;

  // Observable rho(x)
  Nbins = 256*LLL_max/10; // bins for 1 period
  printf("Number of bins: %d \n", Nbins);
  ////////////////////////////////////////////////////////////////////
  ////////////////// NO CHANGES IN THE FOLLOWING ! ///////////////////
  ////////////////////////////////////////xrandr --output None-1 --right-of eDP-1

  // arrays initialization
  rhoxavg = dvector(0, (Nbins )*(Nbins)-1);
  for (int i = 0; i <Nbins*(Nbins-1); i++) {
    rhoxavg[i] = 0.0;
  }

  x = dvector(0, N0 - 1);
  y = dvector(0, N0 - 1);
  ix = ivector(0, N0 - 1);
  iy = ivector(0, N0 - 1);
  //xx0 = dvector(0, N - 1);
  //yy0 = dvector(0, N - 1);
  //ix0 = ivector(0, N - 1);
  //iy0 = ivector(0, N - 1);


  // system initialization
  int check_initial_configuration = lattice_packing(); // initialize particles
    //~ print_positions(0);

  print_positions(0);

  // prepare output files

  ////////////////////////////////////////////////////////////////////
  /////////////// INITIALIZATION FINISHED ! //////////////////////////
  ////////////////////////////////////////////////////////////////////
  // perform simulation
  // mixing
	printf("|---------------------|\n");
	printf("|   mixing  started!  |\n");
	printf("|---------------------|\n");
	for(int im=1; im<=N_mixing; im++) {
		mc_swap();
		if(im%1000==0) {
      printf("mixing step\n: %d", im);
      //check_particle_0();
      print_N(N_print,im);
    }
  }
	printf("\n");
	//print_positions(1);
	printf("|---------------------|\n");
	printf("|   mixing finished!  |\n");
	printf("|---------------------|\n");
  float Amp_in = rand() / ((double)RAND_MAX)*10.0; 
  Amp = (Amp_in * 1.0);	
  Amp = 3;
  #ifdef BULK
  Amp = 0.0;
  #endif
  #ifdef TESTGC
  // test grand canonical steps
  int initial_N = N;
  printf("Initial particle number: %d \n", N);
  printf("chemical potential: %f \n", mu);
  //check_cell_lists();
  for (int i = 0; i < 1000; i++) {
    if(i % 100 == 0) printf("test-step: %d \n", i);
    //check_cell_lists();
    mc_insert();
    mc_remove(); 
  }

  printf("Initial particle number: %d \n", initial_N);
  printf("Final particle number: %d \n", N);
  printf("Removal acceptance ratio: %f \n",Nacc_remove / ((double)Nmc_remove));
  printf("Insertion acceptance ratio: %f \n",Nacc_insert / ((double)Nmc_insert));
  exit(0);
  #endif
	// perform simulation
	// Equilibration
  for(int i=1; i<Neq; ++i)
  {
    if(i%1000==0) {
      printf("equ step: \n%d", i);
      //check_particle_0();
      print_N(N_print,i);
    }
    if (i % Nupdate == 0) {
      if (i != 0) {
        if (Nacc_particle / ((double)Nmc_particle) < 0.45) {
          step_size *= 0.9;
          Nacc_particle = Nmc_particle = 0;
        } else if (Nacc_particle / ((double)Nmc_particle) > 0.55) {
          if (step_size * 1.1 <= Ly * 0.5 - 0.1)
            step_size *= 1.1;
          Nacc_particle = Nmc_particle = 0;
        }
        if (Nacc_xshift / ((double)Nmc_xshift) < 0.45) {
          step_size_xshift *= 0.9;
          Nacc_xshift = Nmc_xshift = 0;
        }
        if (Nacc_xshift / ((double)Nmc_xshift) > 0.55) {
          if (step_size_xshift * 1.1 < 0.5)
            step_size_xshift *= 1.1;
          Nacc_xshift = Nmc_xshift = 0;
        }
        Nacc_particle = Nmc_particle = 0;
        Nacc_xshift = Nmc_xshift = 0;
        if (i%100==0){
          //print_N(cstr,i);
        }
      }
    }
    GC_mc_swap();

  }
  printf("\n");
  printf("|---------------------|\n");
	printf("|   equili finished!  |\n");
	printf("|---------------------|\n");
  for (int i = 1; i <= Nsim; i++) {
    if (i % Nprint == 0) {
      printf(" %d steps done. \n", i);
    }
    if (i%100==0){
      update_rhoxavg_2D();
    }
    if (i%1000==0){
      printf("sim step: \n%d", i);

    }
    GC_mc_swap();

    if (i % Nprint == 0) {
      //print_positions(i);
      print_N(N_print,i);
      CL_check=check_cell_lists();
      if (CL_check == false) {
        printf("Cell list check failed! \n");
        for (int j = 0; j < Nmax; j++) {
          printf("Cell list: %d %d  \n", cell_list_index[0], cell_list[cell_list_index[0]][j]);
        }
      }
      packing_fraction = N /(Lx*Ly) * PI * radius * radius;
      //print_eta(eta_print, packing_fraction, mu, density);
      if (determine_global_overlap_naive() == 1) {
        printf("error overlap %d \n", i);
        break;
      }
    }
  }
  #ifndef BULK
  printf("Writing density profile to file:  \n");
  std::stringstream sstr2;
  std::string count = std::to_string(file_count);
  sstr2 << "Density_profiles/rho_MC_2D_seed" << count << ".dat";
  std::string tmp2 = sstr2.str();
  const char *cstr2 = tmp2.c_str();
  print_rhox_avg_c1(cstr2,nperiods);
  printf("%s\n", cstr2);
  printf("%s\n", count.c_str());
  #endif
  printf("\n");
  //print_positions(Nsim);
  printf("rand SEED: %d\n", lastDigits);
  printf("Packing fraction = %f \n", packing_fraction);
  printf("Period = %f \n", Lperiod);
  printf("Amplitude = %f \n", Amp);
  printf("Number of particles = %d \n", N);
  printf("Simulation Box = %f %f \n", Lx, Ly);
  printf("Number of periods = %d \n", nperiods);
  //printf("Wavenumber grid size = %d x %d \n", Mq, Mq);
  //printf("Wavenumber Delta = %f \n", Deltaq);
  printf("Check! Effective number density %f \n", N / Lx / Ly);
  clock_t end = clock();
  double time_spent = ((double)(end - begin)) / CLOCKS_PER_SEC;
  printf("%ld %ld, spent: %f\n", begin, end, time_spent);
  //
  time(&rawtime);
  printf(" Strated at local time and date: %s", asctime(timeinfo));
  timeinfo = localtime(&rawtime);
  printf("Finished at local time and date: %s", asctime(timeinfo));
 }



/********************************************************************/
int lattice_packing() {
  int nx = int(
      Lx /
      (2 * radius)); // number of particle that fit in the x direction on a line
  int ny = int(N / nx) + 1; // number of lines
  if (N % nx == 0)
    ny = ny - 1;
  double epsy = Ly / ny; // average distance between lines
  double epsy2 = 0.333 * (epsy - sqrt(3.) * radius);

  for (int index = 0; index < N; index++) {
    int j = index / nx;
    int i = index % nx;
    double xloc = 0.0;
    double yloc = 0.0;
    if (j % 2 == 0) {
      xloc = 2 * radius * (0.499 + i);
      yloc = 2 * radius * (j + 0.499) * epsy;
      if (i % 2 == 1)
        yloc += epsy2;
    } else {
      xloc = 2 * radius * (0.999 + i);
      yloc = 2 * radius * (j + 0.5) * epsy;
      if (i % 2 == 1)
        yloc += epsy2;
    }
    pbc_distance(xloc, yloc);
    x[index] = xloc;
    y[index] = yloc;
    ix[index] = 0;
    iy[index] = 0;
  }
  printf("Packing fraction = %f \n", packing_fraction);
  printf("Period = %f \n", Lperiod);
  printf("Amplitude = %f \n", Amp);
  printf("Number of particles = %d \n", N);
  printf("Simulation Box = %f %f \n", Lx, Ly);
  printf("Number of periods = %d \n", nperiods);
  printf("Chemical potential = %f \n", mu);
  //printf("Wavenumber grid size = %d x %d \n", Mq, Mq);
  //printf("Wavenumber Delta = %f \n", Deltaq);
  printf("Check! Effective number density %f \n", N / Lx / Ly);

  // cell list
  Ncell = int(Lx / (3 * radius)); // Should be a meaningful number
  Nmax = 200;
  cell_list_index = ivector(0, N0 - 1);
  cell_list = imatrix(0, (Ncell * Ncell) - 1, 0, Nmax - 1);
  create_cell_lists();
  printf("Number of particles prescibed: %d, Number of particles: %d \n", N0, N);
  //check_cell_lists();
  //check_particle_0();

  printf("Check! Effective packing fraction %f \n",
         N * PI * radius * radius / Lx / Ly);
  printf("---------------------------------\n");

  if (determine_global_overlap_naive() == 1) {
    printf("initial overlap %d \n", 1);
    return 1;
  } else {
    return 0;
  }
}

// consider periodic boundary conditions
void pbc(double &x, double &y, int &ix, int &iy) {
  if (x >= Lx / 2.0) {
    x -= Lx;
    ix += 1;
  }
  if (x < -Lx / 2.0) {
    x += Lx;
    ix -= 1;
  }
  if (y >= Ly / 2.0) {
    y -= Ly;
    iy += 1;
  }
  if (y < -Ly / 2.0) {
    y += Ly;
    iy -= 1;
  }
}

void pbc_distance(double &x, double &y) {
  if (x >= Lx / 2.0)
    x -= Lx;
  if (x < -Lx / 2.0)
    x += Lx;
  if (y >= Ly / 2.0)
    y -= Ly;
  if (y < -Ly / 2.0)
    y += Ly;
}

// initialize cell lists
void create_cell_lists() {
  for (int a = 0; a < Ncell; a++) {
    for (int b = 0; b < Ncell; b++) {
      for (int j = 0; j < Nmax; j++) {
        cell_list[a * Ncell + b][j] = -1;
      }
    }
  }

  // find cell lists for particles
  for (int i = 0; i < N; i++) {
    int xint = (int)(Ncell * (x[i] / Lx + 0.5));
    int yint = (int)(Ncell * (y[i] / Ly + 0.5));

    if (xint < 0)
      xint = 0;
    if (yint < 0)
      yint = 0;
    if (xint > (Ncell - 1))
      xint = Ncell - 1;
    if (yint > (Ncell - 1))
      yint = Ncell - 1;

    cell_list_index[i] = xint * Ncell + yint;

    bool particle_stored = false;
    for (int j = 0; j < Nmax; j++) {
      if (cell_list[xint * Ncell + yint][j] == -1) {
        cell_list[xint * Ncell + yint][j] = i;
        particle_stored = true;
        break;
      }
    }
    if (!particle_stored) {
      printf("ERROR: cell list storage exceeded!!! Nmax must be increased! \n");
      std::raise(SIGINT);
    }
  }
}

void update_cell_lists(int i, double x_old, double y_old, double x_new,
                       double y_new) {
  // update cell lists for particle i
  int cell_number_old = cell_list_index[i];  

  int xint_new = (int)(Ncell * (x_new / Lx + 0.5));
  int yint_new = (int)(Ncell * (y_new / Ly + 0.5));
  int cell_number_new = xint_new * Ncell + yint_new;

  if (cell_number_old != cell_number_new) {
    cell_list_index[i] = cell_number_new;
    int k = -1;
    for (int j = 0; j < Nmax; j++) {
      if (cell_list[cell_number_old][j] == i) {
        k = j;
        break;
      }
    }
    if (k == -1 || k >= Nmax) {
      create_cell_lists();
    } else {
      for (int j = k; j < Nmax - 1; j++) {
        cell_list[cell_number_old][j] = cell_list[cell_number_old][j + 1];
      }
      cell_list[cell_number_old][Nmax - 1] = -1;
      for (int j = 0; j < Nmax; j++) {
        if (cell_list[cell_number_new][j] == -1) {
          cell_list[cell_number_new][j] = i;
          break;
        }
      }
    }
  }
}

int determine_overlap(double xnew, double ynew, int i) {
  double sigma2 = 4 * radius * radius;
  // determine x,y,z cell index
  // int cell_list_indexi = cell_list_index[i]; // this is the old position
  int xint = (int)(Ncell * (xnew / Lx + 0.5));
  int yint = (int)(Ncell * (ynew / Ly + 0.5));
  if (xint < 0)
    xint = 0;
  if (yint < 0)
    yint = 0;
  if (xint > (Ncell - 1))
    xint = Ncell - 1;
  if (yint > (Ncell - 1))
    yint = Ncell - 1;
  int cell_list_indexi = xint * Ncell + yint;

  int cell_list_y = cell_list_indexi % Ncell;
  int cell_list_x = ((cell_list_indexi - cell_list_y) / Ncell);
  ////////////////////////////////////
  for (int a = cell_list_x - 1; a <= cell_list_x + 1; a++) {
    for (int b = cell_list_y - 1; b <= cell_list_y + 1; b++) {
      int aloc = a;
      int bloc = b;
      if (aloc < 0)
        aloc += Ncell;
      if (aloc >= Ncell)
        aloc -= Ncell;
      if (bloc < 0)
        bloc += Ncell;
      if (bloc >= Ncell)
        bloc -= Ncell;

      for (int j = 0; j < Nmax; j++) {
        int jloc = cell_list[aloc * Ncell + bloc][j];

        if (jloc != -1) {
          if (jloc != i) {
            // printf("%d\n",j);
            double dxloc = xnew - x[jloc];
            double dyloc = ynew - y[jloc];
            pbc_distance(dxloc, dyloc);
            double r2 = dxloc * dxloc + dyloc * dyloc;
            if (r2 < sigma2) {
              return 1;
            }
          }
        } else {
          break;
        }
      }
    }
  }
  return 0;
}

int determine_global_overlap_naive() {
  double sigma2 = 4 * radius * radius;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i != j) {
        double dxloc = x[i] - x[j];
        double dyloc = y[i] - y[j];
        pbc_distance(dxloc, dyloc);
        double r2 = dxloc * dxloc + dyloc * dyloc;
        if (r2 < sigma2 - 0.00000000001) {
          printf("pos %d %f %f\n", i, x[i], y[i]);
          printf("pos: %d %f %f\n", j, x[j], y[j]);
          printf("distance: %f\n", r2);
          return 1;
        }
      }
    }
  }
  return 0;
}

/********************************************************************/
void mc_particle_move() {
  // move a ramdom particle in a random direction
  int i = rand() % N;
  double dphi = (rand() / ((double)RAND_MAX)) * 2.0 * PI;
  double dx = step_size * cos(dphi);
  double dy = step_size * sin(dphi);
  double xnew = x[i] + dx;
  double ynew = y[i] + dy;
  int ixnew = ix[i];
  int iynew = iy[i];
  pbc(xnew, ynew, ixnew, iynew);
  // determine overlap
  int overlap = 0;
  overlap = determine_overlap(xnew, ynew, i);
  // if no overlap: move particle and update cell list
  if (overlap == 0) {
    // check potential in x direction
    double Vold = Amp * cos(2 * PI * x[N] / Lperiod);
    double Vnew = Amp * cos(2 * PI * xnew / Lperiod);
    double arg = -(Vnew - Vold) / T;
    double rr2 = rand() / ((double)RAND_MAX);
    if (exp(arg) > rr2) {
      double xold = x[i];
      double yold = y[i];
      x[i] = xnew;
      y[i] = ynew;
      ix[i] = ixnew;
      iy[i] = iynew;
      int CL_index = cell_list_index[i];
      update_cell_lists(i, xold, yold, xnew, ynew);
      Nacc_particle++;
      //~ printf("index: %d %d  \n",Nacc_particle, Nmc_particle);
      //~ printf("index: %d %f %f  \n",i,xold,yold);
      //~ printf("index: %d %f %f  \n",i,xnew,ynew);
      //~ printf("%f %f %f %f \n",Vnew, Vold, exp(arg), rr2);
      /*
      if(check0==true){
        if(i==0){
          printf("Particle 0 got moved from CL %d to CL %d \n", CL_index, cell_list_index[i]);
          printf("Particle 0 got moved from %f %f to %f %f \n", xold, yold, xnew, ynew);
        }
      }
      */
    }
  }
  Nmc_particle++;
}

void mc_xshift() {
  // move all particles in x direction
  int dir = -1;
  double rans = (rand() / ((double)RAND_MAX));
  if (rans > 0.5)
    dir = 1;
  double dx = dir * step_size_xshift;
  double Vold = 0.;
  double Vnew = 0.;
  for (int i = 0; i < N; i++) {
    double xnew = x[i] + dx;
    Vold += Amp * cos(2 * PI * x[i] / Lperiod);
    Vnew += Amp * cos(2 * PI * xnew / Lperiod);
  }
  double arg = -(Vnew - Vold) / T;
  double rr2 = rand() / ((double)RAND_MAX);
  if (exp(arg) > rr2) {
    for (int i = 0; i < N; i++) {
      double xnew = x[i] + dx;
      double ynew = y[i];
      int ixnew = ix[i];
      int iynew = iy[i];
      pbc(xnew, ynew, ixnew, iynew);
      x[i] = xnew;
      y[i] = ynew;
      ix[i] = ixnew;
      iy[i] = iynew;
    }
    //if(check0==true) printf("particle 0 was in cell %d \n", cell_list_index[0]);
    create_cell_lists();
    //if(check0==true) printf("particle 0 got shifted to cell %d \n", cell_list_index[0]);
    Nacc_xshift++;
  }
  Nmc_xshift++;
}

void mc_insert() {
  // insert a new particle
  /*
  if(check0== true){
    if(N==0){
      printf("No Particles in the system \n");
    }
  }
  */
  double xnew = rand() / ((double)RAND_MAX)*Lx;
  double ynew = rand() / ((double)RAND_MAX)*Ly;
  pbc_distance(xnew, ynew);
  int ixnew = 0;
  int iynew = 0;
  pbc(xnew, ynew, ixnew, iynew);
 // determine overlap
  int overlap = 0;
  overlap = determine_overlap(xnew, ynew, N+1);
  // if no overlap: add particle and update cell list
  if (overlap == 0) {
    // check potential in x direction
    double Vnew = Amp * cos(2 * PI * xnew / Lperiod);
    double arg = (mu-Vnew ) / T;
    double rr2 = rand() / ((double)RAND_MAX);
    if ((Lx*Ly)/(N+1)*exp(arg) > rr2) {    // if move is accepted, update N to accord for longer particle array
      x[N] = xnew;
      y[N] = ynew;
      ix[N] = ixnew;
      iy[N] = iynew;
      int xint = (int)(Ncell * (x[N] / Lx + 0.5));
      int yint = (int)(Ncell * (y[N] / Ly + 0.5));

      
      int c_index = xint * Ncell + yint;
      cell_list_index[N] = c_index;
      bool particle_stored = false;
      for (int j = 0; j < Nmax; j++) {
        /*
        if(check0==true){
          if (cell_list[c_index][j] == 0) {
            printf("ERROR: particle %d in manipulated cell list %d \n", 0, c_index);
          }
        }
        */
        if (cell_list[c_index][j] == -1) {
          cell_list[c_index][j] = N;
          particle_stored = true;
          break;
        }
      }
      if (!particle_stored) {
        printf("ERROR: cell list storage exceeded!!! Nmax must be increased! \n");
      }
/* 
      bool check = check_cell_lists();
      if (check == false) {
        printf("inserted particle %d %d  \n", N, cell_list_index[N] );
        printf("Ncell : %d xint:%d yint: %d  \n",Ncell, xint, yint );
        for(int j = 0; j < Nmax; j++) {
          printf("Inserted particle in Cell list: %d which holds %d  \n", c_index, cell_list[c_index][j]);
          if(cell_list[c_index][j] == -1) break;
        }
      }

 */      //check_particle_0();
      N++;
      Nacc_insert++;

      //~ printf("index: %d %d  \n",Nacc_particle, Nmc_particle);
      //~ printf("index: %d %f %f  \n",i,xold,yold);
      //~ printf("index: %d %f %f  \n",i,xnew,ynew);
      //~ printf("%f %f %f %f \n",Vnew, Vold, exp(arg), rr2);
    }
  } 
  Nmc_insert++;

}

void mc_remove() {
  // remove a particle
  // select a random particle
  int i = rand() % N;
  // check energy of removal
  double Vnew = Amp * cos(2 * PI * x[i] / Lperiod);
  double arg = (Vnew-mu) / T;
  double rr2 = rand() / ((double)RAND_MAX);
  if ( N/(Lx * Ly ) * exp(arg) > rr2) { // if move is accepted, update N to accord for shorter particle array
    // remove particle and update cell list
    int index;
    int old_index;
    index = cell_list_index[i];
    old_index = index;
    // update cell list of removed particle
    bool particle_stored = true;
    int particle_index_in_cell = -1;
    for (int j = 0; j < Nmax; j++) {
      //find position of removed particle in it's cell list
      if (cell_list[index][j] == i) {
        particle_index_in_cell = j;
      }
      if (cell_list[index][j+1] == -1) {
        // find last position in cell list
        // replace removed particle with last particle in cell list
        cell_list[index][particle_index_in_cell]=cell_list[index][j];
        cell_list[index][j] = -1;
        particle_stored = false;
        break;
      } 
    }
    if (particle_stored) {
      printf("ERROR: PARTICLE REMOVAL UNSUCCESSFUL!!!  \n");
      //printf("index: %d %d  \n", i, N);
      //for (int j = 0; j < Nmax; j++) {
       // printf("Cell list: %d %d  \n", index, cell_list[index][j]);
      //}
      //exit(1);
    }
    // replace particle with last particle in particle array
    x[i] = x[N - 1];
    y[i] = y[N - 1];
    ix[i] = ix[N - 1];
    iy[i] = iy[N - 1];
    // update list of cell list of particles
    cell_list_index[i] = cell_list_index[N - 1];
    index = cell_list_index[i];
    // update the last entry
    x[N - 1] = 0.0;
    y[N - 1] = 0.0;
    ix[N - 1] = 0;
    iy[N - 1] = 0;
    // update cell list of particle (former last one in particle list) replacing removed particle
    for (int j = 0; j < Nmax; j++) {
      if (cell_list[index][j] == N-1) {
        cell_list[index][j] = i;
        break;
      } 
    }

    cell_list_index[N - 1] = -1;

    N--;
    Nacc_remove++;
    /*
    if(check0==true){
        if(i==0){
          printf("Particle 0 got removed from CL %d and was replaced by particle %d \n", old_index, N+1);
        }
      }
        */
    //bool check = check_cell_lists();
    //if(!check){
    //  printf("removed particle %d cell list %d  \n", i, index );
    //  printf("replaced with particle formerly at %d cell list %d  \n", N, cell_list_index[i] );
    //}
    //check_particle_0();
  }
  Nmc_remove++;
  
}
void mc_step() {
  double rans = rand() / ((double)RAND_MAX);
  if (rans <= 1. / N) {
    mc_xshift();
  } else {
    mc_particle_move();
  }
}

void mc_swap() {
  for (int k = 0; k < N; k++) {
    double rans = rand() / ((double)RAND_MAX);
    if (rans <= 1. / N) {
      mc_xshift();
    } else {
      mc_particle_move();
    }
  }
}

void GC_mc_swap() {
  for (int k = 0; k < N; k++) {
    double rans = rand() / ((double)RAND_MAX);
    #ifndef BULK
    if (rans < 0.5){
      rans = rand() / ((double)RAND_MAX);
      if (rans <= 1. / N) {
        mc_xshift();
      } else {
        mc_particle_move();
      }
    }
    #endif
    #ifdef BULK
    if (rans < 0.5){
        mc_particle_move();
      }
    #endif
     else{
      rans = rand() / ((double)RAND_MAX);
      if (rans <= 0.5) {
        mc_insert();
        
      } else {
        mc_remove();
        
      }
    }
  }

} /********************************************************************/
// Diagnostic functions

bool check_cell_lists(){
  for(int i=0;i<N;i++){
    int index;
    bool check = false;
    index = cell_list_index[i];
    for(int j=0;j<Nmax;j++){
      if(cell_list[index][j]==i){
        check = true;
        return true;
      }
    }
    if(check==false){
        printf("ERROR: PARTICLE NOT FOUND IN CELL LIST!!!  \n");
        printf("Particle %d in Cell list: %d  \n", i, index);
        for(int k=0;k<Nmax;k++){
          printf("Cell list: %d has particle: %d  \n", index,cell_list[index][k] );
          if(cell_list[index][k]==i){
            printf("Found particle %d in cell list %d \n", i, index);
            break;
          }
          if(cell_list[index][k]==-1){
            break;
          }

        }
        for (int k =0;k<Ncell*Ncell-1;k++){
          for(int j=0;j<Nmax;j++){
            //printf("Cell list: %d has particle: %d  \n", k,cell_list[k][j] );
            if(cell_list[k][j]==i){
              printf("Found particle %d in cell list %d \n", i, k);
              break;
            }
            if(cell_list[k][j]==-1){
              break;
            } 
          }
        }
        exit(1);
        return false; 
    }
  }
  return true;
}

//Track particle 0
void check_particle_0(){
  int p_index = 0;
  if(cell_list_index[p_index] == -1){
    printf("Particle 0 removed by MC-move \n");
    printf("Current N is %d\n",N);
    return;
  }
  //printf("Particle 0 thinks it is in cell list %d \n", cell_list_index[p_index]);
  for(int i =0;i<Nmax;i++){
    if(cell_list[cell_list_index[p_index]][i]==p_index){
   //   printf("Particle 0 found in cell list %d \n", cell_list_index[p_index]);
      return;
    }
    if(cell_list[cell_list_index[p_index]][i]==-1){
      break;
    }
  }
  printf("ERROR: PARTICLE 0 NOT FOUND IN CELL LIST!!!  \n");
  for(int i =0;i<Ncell;i++){
    for(int j=0;j<Nmax;j++){
      printf("Cell list: %d has particle: %d  \n", i,cell_list[i][j] );
      if(cell_list[i][j]==p_index){
        printf("Found particle %d in cell list %d \n", p_index, i);
        break;
      }
      if(cell_list[i][j]==-1){
        break;
      } 
    }
  }
  exit(1);
}
/********************************************************************/
// 1D density profile, relative to periodic potential
void update_rhoxavg() {
  double binsize = Lperiod / Nbins;
  for (int i = 0; i < N; i++) {
    double xt = x[i];
    double yt = y[i];
    int j = (int)(xt / Lperiod);
    xt = xt - j * Lperiod;
    if (xt >= Lperiod / 2.0 - binsize / 2.0)
      xt -= Lperiod;
    if (xt < -Lperiod / 2.0 - binsize / 2.0)
      xt += Lperiod;
    xt += Lperiod / 2.0 + binsize / 2.0;

    j = (int)(yt / Lperiod);
    yt = yt - j * Lperiod;
    if (yt >= Lperiod / 2.0 - binsize / 2.0)
      yt -= Lperiod;
    if (yt < -Lperiod / 2.0 - binsize / 2.0)
      yt += Lperiod;
    yt += Lperiod / 2.0 + binsize / 2.0;
    
    int xindex = (int)(xt / binsize);
    int yindex = (int)(yt / binsize);
    rhoxavg[xindex*Nbins+yindex] += 1. / (binsize * binsize* nperiods);
  }
  count_rhoxavg += 1;
}

// new 2D density grid for training data
void update_rhoxavg_2D(){
  double binsize = Lx /Nbins;
  for (int i = 0; i < N; i++) {
    double xt = x[i];
    double yt = y[i];
    xt = fmod(fmod(xt, Lx) + Lx, Lx);
    yt = fmod(fmod(yt, Ly) + Ly, Ly);
    int ix = (int)(xt / binsize);
    int iy = (int)(yt / binsize);
    rhoxavg[ix*Nbins+iy] += 1.;
  }
  count_rhoxavg += 1;

}

void update_N_avg(){

}


/******************************* */
// Printing functions

/********************************************************************/
// calculate and print g(r)
void print_gr(int step) {

  // constants for histogram
  int Nhisto = 100;
  double drhisto = 0.05;

  // declare and initialize histogram
  double *histo = dvector(0, Nhisto - 1);
  for (int i = 0; i < Nhisto; i++) {
    histo[i] = 0.0;
  }

  for (int i = 0; i < N; i++) {
    double xi = x[i];
    double yi = y[i];

    for (int j = 0; j < N; j++) {
      if (j != i) {
        double dxloc = xi - x[j];
        double dyloc = yi - y[j];
        pbc_distance(dxloc, dyloc);

        double dr = sqrt(dxloc * dxloc + dyloc * dyloc);
        int drint = (int)(dr / drhisto + 0.5);

        // printf("dr %f drint %d\n",dr,drint);
        //  bin the particle distance into a histogram
        if (drint < Nhisto)
          histo[drint]++;
      }
    }
  }

  // write histogram data
  std::stringstream sstr;
  sstr << "gr_step" << step << ".dat";
  const std::string tmp = sstr.str();
  const char *cstr = tmp.c_str();

  FILE *out = fopen(cstr, "w");

  for (int i = 0; i < Nhisto; i++) {
    double A = PI * (pow(i * drhisto + drhisto, 2) - pow(i * drhisto, 2));
    fprintf(out, "%f %f\n", i * drhisto,
            histo[i] * Lx * Ly / ((double)N * N * A));
  }

  fclose(out);
}

// print particle positions
void print_positions(int step) {

  std::stringstream sstr;
  sstr << "positions_seed" << lastDigits << "_step" << step << ".dat";
  const std::string tmp = sstr.str();
  const char *cstr = tmp.c_str();

  FILE *out = fopen(cstr, "w");

  for (int i = 0; i < N; i++) {
    // fprintf(out,"%f %f  %d %d\n",x[i],y[i],ix[i],iy[i]);
    fprintf(out, "%f %f\n", x[i], y[i]); //,ix[i],iy[i]);
  }

  fclose(out);
}

// print density profile
void print_rhox_avg_c1(const char *cstr,int nperiod) {
  FILE *out = fopen(cstr, "w");
  fprintf(out, "x y rho muloc %d\n",nperiod);

  double binarea = (Lx / Nbins)*(Ly / Nbins);
  for (int i = 0; i < Nbins; i++) {
    for (int j = 0; j < Nbins; j++) {
      double xbin =Lx * (i + 0.5) / Nbins;
      double ybin =Ly * (j + 0.5) / Nbins;
      double muloc =  Amp * cos(2 * PI * xbin / Lperiod)-mu;
      fprintf(out, "%f %f %f %f\n", xbin, ybin, rhoxavg[i*Nbins+j] / (count_rhoxavg*binarea), muloc);
    }
  }
  fclose(out);
}


void print_N(const char *cstr,int step){


  FILE *out = fopen(cstr, "a");
  fprintf(out,"%d  %d  %f  %f\n", step, N,density,mu); 
  fclose(out);
}

void print_eta(const char *cstr, double packing_fraction, double mu, double density){
  FILE *fp2 = fopen(cstr, "a");
  fprintf(fp2,"%f ; %f ; %f \n",packing_fraction, mu, density);
  fclose(fp2);
}
