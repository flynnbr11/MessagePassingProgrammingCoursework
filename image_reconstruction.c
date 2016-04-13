#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pgmio.h"
#include "mpi.h"
#include <string.h>
#include "arralloc.h"

#define MAX_ITERS 15000 // if the program goes beyond this it terminates


float boundaryval(int i, int m); // for sawtooth boundary conds

void 	calculate_rank_limits(int M, int N, int rank, int size, int* i_low, int* i_high, int* 
i_length, int* j_low, int* j_high, int* j_length, int* local_num_iterations, int* dims, MPI_Comm grid_comm);

void init_old_matrix(int i_length, int j_length, int i_low,int M, float** old);

void halo_swaps(int up, int down, int right, int left, float** old, int i_length, int j_length, MPI_Comm comm, MPI_Datatype horiz_strip, MPI_Request* request );

void get_neighbours(int* up, int* down, int* left, int* right, MPI_Comm grid_comm);
void update_new_array(int i_length, int j_length, float**  new, float** old, float** edge);

void replace_old_array(int t, int test_freq, int i_length, int j_length, float* local_max, float* running_sum, float* delta, float** new, float** old);

void test_convergence(float* running_sum,float* global_running_sum, float* local_max, float* max_change, float* average_pixel, float* final_average_pixel, int* final_iter_count, float* final_delta, int M, int N, int rank, int t, MPI_Comm grid_comm);



int main(int argc, char *argv[]) {
	MPI_Init(NULL, NULL);
	double start_time, end_time, time_taken, start_iterative_loop_time,end_iterative_loop_time, time_in_loop;
	MPI_Comm comm, grid_comm;
	comm = MPI_COMM_WORLD;
	MPI_Request request[8]; // for use in the halo swapping sends, declare once here rather than inside function
	MPI_Status status; 

	int dims[2] = {0,0};
	int periods[2] = {1,0};
	int M,N, i, j, t,a,b, rank, size, iter; //loop variables and other constants
	int up, right, left, down; //find neighbours to halo swap with
	float max_change = 2.0; // Initialise to relatively high number compared with threshold
	double final_pixel_average=0;
	float delta, local_max, final_delta, average_pixel, final_average_pixel; //For testing pixel values and convergence
	int i_low, i_high, j_low, j_high, i_length, j_length, local_num_iterations, final_iter_count; //Individual processor limits
	float running_sum=0; //to keep track of pixel values to average at end
	float global_running_sum=0; //combining sum of pixel values on all processors

	char* image_to_read ="given_images/edgenew192x128.pgm";
	char* output_image ="output.pgm";
	float threshold = 0.2 ;
	int test_freq =10;
	/*
	* For testing, uncomment the next four lines and use any image, threshold and frequency
	* Can just uncomment first line to test different images
	*/
	
	/*
	if( argc < 2 ) {
			// printf("No threshold or image \n");
			exit;
	}	
	image_to_read = argv[1];
	output_image = argv[2];
	threshold = atof(argv[3]);
	test_freq = atof(argv[4]);
	//*/	
	
	/*
	*All ranks find size of input image and delare array of that size locally
	*/
	pgmsize(image_to_read, &M, &N);
	float master_image[M][N];
	
	/*
	* All ranks find their own id, and the total size of the communicator
	*/
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	if(rank==0) { // Set up timer
		start_time = MPI_Wtime();
	}

	/*
	* Create topology
	*/
	MPI_Dims_create(size,2,dims);
	MPI_Cart_create(comm, 2, dims, periods, 0, &grid_comm);
	
	/*
	*	Read in image on master only, and broadcast entire image to each array
	*/
	if(rank==0) { 
		pgmread(image_to_read, master_image, M, N);
	}
	MPI_Bcast(master_image, M*N, MPI_FLOAT, 0, comm);

	/*
	* Figure out section of master image to work on
	*/	
	calculate_rank_limits(M, N, rank, size, &i_low, &i_high, &i_length, &j_low, &j_high, &j_length, &local_num_iterations, dims, grid_comm);
	// printf(" rank %d : i_low=%d i_high = %d j_low=%d j_high=%d \n", rank, i_low, i_high, j_low, j_high); //To confirm topology distribution

	/*
	Dynamically allocate arrays so that we can compute them in functions
	*/
	float **old = arralloc(sizeof(float),2,i_length+2, j_length+2);
	float **new = arralloc(sizeof(float),2,i_length+2, j_length+2);
	float **edge = arralloc(sizeof(float),2,i_length+2, j_length+2);
	
	/*
	* Require static arrays to copy the above into,
	* since dynamically allocated arrays can't be passed to MPI_Send
	*/
	float final_local_array[i_length][j_length];
  float final_image[M][N];
  
	/*
	* Derived datatype to be used in halo swaps across horizontal borders
	*/
	MPI_Datatype horiz_strip;
	MPI_Type_vector(i_length, 1, j_length+2, MPI_FLOAT, &horiz_strip);
	MPI_Type_commit(&horiz_strip);
	
	/*
	* Copy section of the code for this processor to work on
	* to an array called edge, as per Jacobi equation
	*/
	for(i=1; i<i_length+1; i++) {
		for(j=1; j<j_length+1; j++) {
			edge[i][j] = master_image[i_low + i - 1][j_low + j - 1];
		}
	}

	/*
	* Set up "old" matrix, with boundary conditions
	*/
	init_old_matrix(i_length, j_length, i_low, M, old);

	/*
	* Find which processors are neighbours in the topology, to halo swap with
	*/
	get_neighbours(&up, &down, &left, &right, grid_comm);

	t=0;
	start_iterative_loop_time = MPI_Wtime();

	/*
	* Begin iterative loop
	* Terminate loop when threshold is reached or MAX_ITERS exceeded
	*/
	while( max_change > threshold && t < MAX_ITERS ) { 
	//for(a=1; a<=1501; a++) { //For comparison against serial version, use this instead of while loop
		local_max  = 0;
		running_sum = 0; //reset at the start to recalculate average
		
		/*
		*	Swap halos with neighbours, update the pixel values, and update the old arrays
		*/
		halo_swaps(up, down, right, left, old, i_length, j_length, comm, horiz_strip, request);
		update_new_array(i_length, j_length, new, old, edge);
		replace_old_array(t, test_freq, i_length, j_length, &local_max, &running_sum, &delta, new, old); //also replaces local_max, delta every tenth iteration

		/*
		* Test if the image has converged sufficiently
		*/	
		if(t % test_freq==0) {
			test_convergence(&running_sum, &global_running_sum, &local_max, &max_change, &average_pixel, &final_average_pixel, &final_iter_count, &final_delta, M, N, rank, t,  grid_comm);
		}
		
		t++; // so that we don't exceed max iterations
		
	}// end while loop
	end_iterative_loop_time = MPI_Wtime();

	/*
	* Can't send dynamically allocated arrays through MPI_Send
	* copy updated arrays into a final, statically allocated array
	*/
	for(i=0; i<i_length; i++) { 
			for(j=0; j<j_length; j++) { 
				final_local_array[i][j] = old[i+1][j+1];	
			}
		}
		
	/*
	* Put all small arrays onto master processor to combine 
	*/
	if(rank!=0) {
			MPI_Ssend(final_local_array, local_num_iterations, MPI_FLOAT, 0, 0, comm);
	}

	/*
	* Receive all images on the master
	* Place them in a statically allocated array
	* Print output
	*/
	if(rank==0) { 
	
		for(i=0; i<i_length; i++) { 
			for(j=0; j<j_length; j++) { 
				final_image[i+i_low][j+j_low] = final_local_array[i][j];
			}
		}	//fill in the part done by master before changing values of i_low etc
	
		/* 
		*	Receive from whoever gets here first 
		* Find which block of the image they worked on
		*	put the received message into that block
		*/  
		for(a=0; a<size-1; a++) {
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
			int sender = status.MPI_SOURCE;

			/*
			*	 Work out out how big the incoming message will be based on
			*  on which rank has issued the most recent send
			*/
			calculate_rank_limits(M, N, sender, size, &i_low, &i_high, &i_length, &j_low, &j_high, &j_length, &local_num_iterations, dims, grid_comm);
	
			float temp_buf[i_length][j_length]; // Local array the same size as incoming message to store the sub-image done by sender
			MPI_Recv(temp_buf, local_num_iterations, MPI_FLOAT, sender, 0, comm, &status); // synchronous receive. Put in temporary buf
			
			/*
			* Place received array into final, statically allocated array (on master)
			*/
			for(i=0; i<i_length; i++) { 
				for(j=0; j<j_length; j++) { 
					final_image[i+i_low][j+j_low] = temp_buf[i][j];
				}
			}	

		} //end for(a...)

		/*
		* Now the whole image is on final_image
		* we output the final array
		* and calculate timings
		*/
		for(i=0; i<M; i++) { 
			for(j=0; j<N; j++) { 
				final_pixel_average += final_image[i][j];
			}
		}
			
		final_pixel_average = ( final_pixel_average / (M*N) );
		end_time = MPI_Wtime();
		time_in_loop = end_iterative_loop_time - start_iterative_loop_time;
		time_taken = (double) end_time - start_time;
		
		pgmwrite(output_image, final_image, M, N);

		 printf("Time taken on %d processors for %d iterations  : %f secs \n Average pixel = %f \n Max change at final iteration =%f\n time in iterative loop =%lf sec \n", size, final_iter_count, time_taken, final_pixel_average, max_change, time_in_loop);
		 
		 /*
		 *Uncomment the next line for total results. More feasible for large runs 
		 */
		// printf("%d \t %lf \t %lf \t %lf \t %lf \t %d \t %s \t %d \t %f \n", size, final_pixel_average, max_change, time_taken, time_in_loop, final_iter_count, image_to_read, test_freq, threshold); 
	} // end if(rank==0)
	
	MPI_Finalize();
	return 0;
} // end main



void 	calculate_rank_limits(int M, int N, int rank, int size, int* i_low, int* i_high, int* 
i_length, int* j_low, int* j_high, int* j_length, int* local_num_iterations, int* dims, MPI_Comm grid_comm) {
	int coords[2];
	MPI_Cart_coords(grid_comm, rank, 2, coords);

	*i_low = (int) ceil( ( M/dims[0] ) * coords[0] );
	*i_high = (int) ceil( ( M/dims[0] ) * ( 1 + coords[0] ) );
	*j_low = (int) ceil( (N/dims[1]) * coords[1] );
	*j_high = (int) ceil( (N/dims[1]) * ( 1 + coords[1] ) );

	/*
	* Ensure processors along the top and right-hand side of the topology
	* take all iterations to the edges
	*/
	if(coords[0] == dims[0] - 1 ) {
	 *i_high = M ;
	}
	if(coords[1] == dims[1] - 1 ) { 
		*j_high = N;
	}
	
	*i_length = *i_high - *i_low;
	*j_length = *j_high - *j_low;
	*local_num_iterations = (*i_length *  *j_length);
}


void init_old_matrix(int i_length, int j_length, int i_low,int M, float** old) {
	int i,j;
	for(i=0; i<i_length+2; i++) {
		for(j=0; j<j_length+2; j++) {
			old[i][j] = 255.0;
		}
	}

	for (i=1; i < i_length+1; i++) {
		float val = boundaryval( (i + i_low), M);
		old[i][0]   = 255.0*val;
		old[i][j_length+1] = 255.0*(1.0-val);
	}

}

		
float boundaryval(int i, int m)
{
  float val;
  val = 2.0*((float)(i-1))/((float)(m-1));
  if (i >= m/2+1) val = 2.0-val;
  return val;
}


void get_neighbours(int* up, int* down, int* left, int* right, MPI_Comm grid_comm) {
	int u,d, l,r;
	MPI_Cart_shift(grid_comm,1,1, &d, &u);
	MPI_Cart_shift(grid_comm,0,1, &l, &r);
	*left= l;
	*right = r;
	*up = u;
	*down = d;
}


void halo_swaps(int up, int down, int right, int left, float** old, int i_length, int j_length, MPI_Comm comm, MPI_Datatype horiz_strip, MPI_Request* request ) {
	/*
	* Send and receive from all neighbours
	* Vertically, send messages of length j_length
	* Horizontally, send derived MPI_Vectors (horiz_strip) of length i_length
	*/

	MPI_Issend(&old[i_length][1], j_length, MPI_FLOAT, right, 0 , comm, &request[0]);
	MPI_Irecv(&old[0][1], j_length, MPI_FLOAT, left, 0, comm, &request[1]); 
	MPI_Issend(&old[1][1], j_length, MPI_FLOAT, left, 0 , comm, &request[2]);
	MPI_Irecv(&old[i_length+1][1], j_length, MPI_FLOAT, right, 0, comm, &request[3]); 
	MPI_Issend(&old[1][j_length], 1, horiz_strip, up, 0 , comm, &request[4]);
	MPI_Irecv(&old[1][0], 1, horiz_strip, down, 0, comm, &request[5]); 
	MPI_Issend(&old[1][1], 1, horiz_strip, down, 0 , comm, &request[6]);
	MPI_Irecv(&old[1][j_length+1], 1, horiz_strip, up, 0, comm, &request[7]); 
	MPI_Wait(&request[0], MPI_STATUS_IGNORE);
	MPI_Wait(&request[1], MPI_STATUS_IGNORE);
	MPI_Wait(&request[2], MPI_STATUS_IGNORE);
	MPI_Wait(&request[3], MPI_STATUS_IGNORE);
	MPI_Wait(&request[4], MPI_STATUS_IGNORE);
	MPI_Wait(&request[5], MPI_STATUS_IGNORE);
	MPI_Wait(&request[6], MPI_STATUS_IGNORE);
	MPI_Wait(&request[7], MPI_STATUS_IGNORE);
	
}


void update_new_array(int i_length, int j_length, float**  new, float** old, float** edge) {
	/*
	* Jacobi iteration
	*/
	int i,j;
	for(i=1; i<i_length+1; i++) {
					for(j=1; j<j_length+1; j++) {
							new[i][j] = ( ( old[i-1][j] + old[i+1][j] + old[i][j-1] + old[i][j+1] - edge[i][j] 		) / 4.0);
					}
				}
}

void replace_old_array(int t, int test_freq, int i_length, int j_length, float* local_max, float* running_sum, float* delta, float** new, float** old) { 
	int i,j;
	for(i=1; i<i_length+1; i++) {
		for(j=1; j<j_length+1; j++) {
			if(t % test_freq==0 ) { 
				*running_sum += new[i][j];
				*delta = fabs(new[i][j] - old[i][j]);
				if( *delta > *local_max  ) *local_max  = *delta;
			}
			old[i][j] = new[i][j];				
		}
	}
}	


void test_convergence(float* running_sum, float* global_running_sum, float* local_max, float* max_change, float* average_pixel, float* final_average_pixel, int* final_iter_count, float* final_delta, int M, int N, int rank, int t, MPI_Comm grid_comm) {
	float local_sum = *running_sum;
	float global_pixel_sum= *global_running_sum;
	float local_max_delta = *local_max; 
	float global_max_delta = *max_change;
	
	MPI_Reduce(&local_sum, &global_pixel_sum, 1, MPI_FLOAT, MPI_SUM,0, grid_comm);
	MPI_Allreduce(&local_max_delta , &global_max_delta, 1, MPI_FLOAT, MPI_MAX, grid_comm);
	*max_change = global_max_delta;
	*global_running_sum = global_pixel_sum;
	*final_iter_count = t;

	if(rank == 0 ) { 
		*average_pixel = ( *global_running_sum / (M*N) );
		 //printf("At iteration %d average pixel = %f and max_change = %f \n", t, *average_pixel, *max_change); //to track the average pixel throughout the processing
	}
}

