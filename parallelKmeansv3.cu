#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
//Problem Size 2^25
#define N 33554432
#define THREADS_PER_BLOCK 128

//This kernel is responsible for the following tasks
//Compute distance between each data point and the cluster centroids
//Based on distance assign the point to the closest cluster
//Count the number of points in each cluster 
__global__ void KMeansClustering(double *centroids,double *data,double *clstr1,double *clstr2,double *clstr3,int n,int *noOfCPoints)
{
//Input data read from the file is of the format as shown below
//x-coordinate,y-coordinate,clusterid
//So each thread is supposed to work on these 3 data 
int tid = (blockIdx.x*blockDim.x +threadIdx.x)*3;
 
if(tid<3*n){
  
   //Index to store the data into one of the clusters
   //Since the cluster arrays are one dimensional, data points are stored as follows in memory
   //[x1,y2,x2,y2,x3,y3....xn,yn]
   //So each thread is responsible for one point (x,y)
   int index = (blockIdx.x*blockDim.x +threadIdx.x)*2;
   //Shared memory array to store the number of points in the clusters
   //All the points in one thread block will update in this shared memory
   //Since the atomic operations on the shared memory are faster than the global memory
   //Similar to Histogram
   //Privatization concept is used
   __shared__ int s_cluster[3];
   //Initialsed the number of points in each cluster to 0
   s_cluster[0] = 0;
   s_cluster[1] = 0;
   s_cluster[2] = 0;
   //Syncronisations so that all the threads in the block
   //dont overwrite each others changes
   __syncthreads();
  
   //Storing the data point in the register
   //so as to avoid multiple global memory access
   double data_x = data[tid];
   double data_y = data[tid+1];
 
   //A cluster pointer array, to point to start
   //of each cluster
   double *cluster[3];
   
   //Calculating distance of the data point from each of the cluster
   double d_1 = pow(data_x-centroids[0],2)+pow(data_y-centroids[1],2);
   double d_2 = pow(data_x-centroids[2],2)+pow(data_y-centroids[3],2);
   double d_3 = pow(data_x-centroids[4],2)+pow(data_y-centroids[5],2);
   
   //Initialising the cluster pointer array
   cluster[0] = clstr1;
   cluster[1] = clstr2;
   cluster[2] = clstr3;

   //Using the ternary operator to find the closest cluster
   //Nested If else blocks result in a lot of control divergence
   int clusterIndex = d_1 > d_2 ? d_2 > d_3 ? 2 : 1  : d_1 < d_3 ? 0: 2 ;    
   
   //based on the clusterindex obtained
   //assign the point the corresponding cluster 
   for(int i=0;i<3;i++){
      if(i!=clusterIndex){ 
        double * clusterPtr = cluster[i];
        clusterPtr[index]   = 0.0;
        clusterPtr[index+1] = 0.0;
      }else{
        double * clusterPtr = cluster[clusterIndex];
        clusterPtr[index] = data_x;
        clusterPtr[index+1] = data_y;
      }
   }
   
   //Increment the counter in the shared memory
   //corresponding to the cluster to which point is assigned
   atomicAdd(&s_cluster[clusterIndex],1);
   //Synchronisation is required so that all the threads 
   //in the block have finished incrementing the number of points
   //in the correct cluster
   //before updating the global memory with the final value of each block
   __syncthreads();
   
   //Update the global memory with the count of points in a particular cluster
   //which is done by 3 threads in each block since we have 3 clusters
   //one warp in each block will be in control divergence due to this
    if(threadIdx.x < 3){
      atomicAdd(&noOfCPoints[threadIdx.x],s_cluster[threadIdx.x]);     
    }
   }
}
 

//This kernel is responsible for in place
//reduction of all the clusters to perform sum of all
//points in a particular cluster
__global__ void sumCluster(double *cluster1,double *cluster2,double *cluster3,int n){
//Each thread works on a 2 dimensional point (x,y)
int tid = (blockIdx.x*blockDim.x +threadIdx.x)*2;

//Shared memory to store the partial sum of each of clusters
//Shared memory is faster than the main memory hence time will be lesser
//There are 3 clusters so 3 array in shared memory
__shared__ double shared_data_1[THREADS_PER_BLOCK*2];
__shared__ double shared_data_2[THREADS_PER_BLOCK*2];
__shared__ double shared_data_3[THREADS_PER_BLOCK*2];
 
if(tid < n){ 
//Initiliase the shared memory with the data
//from the global memory
shared_data_1[2*threadIdx.x] = cluster1[tid];
shared_data_1[2*threadIdx.x+1] = cluster1[tid+1];
 
shared_data_2[2*threadIdx.x] = cluster2[tid];
shared_data_2[2*threadIdx.x+1] = cluster2[tid+1];
 
shared_data_3[2*threadIdx.x] = cluster3[tid];
shared_data_3[2*threadIdx.x+1] = cluster3[tid+1];
__syncthreads();
}
 
  //Making use of sequential addressing
  //which gives coalsced memory access, eliminates bank conflict
  //and reduces control divergence
  //Since each thread processes a point (x,y)
  //stride is intialised to blockDim instead of blockDim/2 
  int stride = blockDim.x; 
  while((stride >= 2) && (threadIdx.x < stride/2)){
   
    shared_data_1[2*threadIdx.x] += shared_data_1[2*threadIdx.x+stride];
    //addition for y
    shared_data_1[2*threadIdx.x+1]+=shared_data_1[2*threadIdx.x+stride+1];   
    
    //addition for x
    shared_data_2[2*threadIdx.x]+=shared_data_2[2*threadIdx.x+stride];
    //addition for y
    shared_data_2[2*threadIdx.x+1]+=shared_data_2[2*threadIdx.x+stride+1];
 
    //addition for x
    shared_data_3[2*threadIdx.x]+=shared_data_3[2*threadIdx.x+stride];
    //addition for y
    shared_data_3[2*threadIdx.x+1]+=shared_data_3[2*threadIdx.x+stride+1];
    //synchronisation to ensure all the threads have finished
    //before going to the next stride value
    __syncthreads();
    //Using right shift operator instead of divide
    //since it is faster
    stride =  stride>>1;
  }
 
//One thread from each block write the partial sum into
//the global memory
//This kernel is called again and again until one block
//is required to do the sum
if(threadIdx.x == 0){
  cluster1[blockIdx.x*2] = shared_data_1[threadIdx.x];
  cluster1[blockIdx.x*2+1] = shared_data_1[threadIdx.x+1];
 
  cluster2[blockIdx.x*2] = shared_data_2[threadIdx.x];
  cluster2[blockIdx.x*2+1] = shared_data_2[threadIdx.x+1];
 
  cluster3[blockIdx.x*2] = shared_data_3[threadIdx.x];
  cluster3[blockIdx.x*2+1] = shared_data_3[threadIdx.x+1];

}
}

//Function to check if any cuda errors have occured
void checkCudaError(cudaError_t error,int lineNo){
      if (error !=cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error),__FILE__,lineNo);
        exit(EXIT_FAILURE);
     }
 
}
 
int main(int argc, char *argv[]) {
    //Select the first gpu device
    cudaSetDevice(0);
    //Read data from the file
    FILE *inFile = fopen("33554432_CLUSTER_DATA.csv", "r");
    if(inFile == NULL){
        printf("Unable to read the data from the file");
        exit(1);
    }
    
    //Host memory allocation for data 
    double *host_data = (double *)malloc(sizeof(double)*N*3);
    //CUDA memory allocation for data
    double *dev_data;
    cudaError_t error = cudaMalloc(&dev_data,N*3*sizeof(double));
    checkCudaError(error,__LINE__-1);
    //Initialise the host data from the valies in the file
    for(int i =0;i<N;i++){
        fscanf(inFile, "%lf,%lf,%lf\n", &host_data[i*3],&host_data[i*3+1],&host_data[i*3+2]);
    }
   
   //Host memory allocation for the clusters
   double *host_cluster_1 = (double *)calloc(N*2,sizeof(double));
   double *host_cluster_2 = (double *)calloc(N*2,sizeof(double));
   double *host_cluster_3 = (double *)calloc(N*2,sizeof(double));
   
   //CUDA memory allocation for the clusters
   double *dev_c_1;
   double *dev_c_2;
   double *dev_c_3;
   error = cudaMalloc((void**)&dev_c_1,N*2*sizeof(double));
   checkCudaError(error,__LINE__-1);
   error = cudaMalloc((void**)&dev_c_2,N*2*sizeof(double));
   checkCudaError(error,__LINE__-1);
   error = cudaMalloc((void**)&dev_c_3,N*2*sizeof(double));
   checkCudaError(error,__LINE__-1);
   
   //memory allocation to store the cluster centroids
   double* host_centroids = (double*)malloc(6*sizeof(double));
   double* dev_centroids;
   error = cudaMalloc((void**)&dev_centroids,6*sizeof(double));
   checkCudaError(error,__LINE__-1);  
   //Randomly initialising K centroids for the clusters
   srand(29);
 
   int index1 = (rand() % N )*3;
  
   host_centroids[0] = host_data[index1];
   host_centroids[1] = host_data[index1+1];
   int index2 = (rand() % N)*3;
  
   host_centroids[2] = host_data[index2];
   host_centroids[3] = host_data[index2+1];
   int index3 = (rand() % N)*3;
   
   host_centroids[4] = host_data[index3];
   host_centroids[5] = host_data[index3+1];
 
   printf("Initial Centroid Estimate\n");
   for(int i=0;i<=4;i+=2){
          printf("centroid[%d][0] = %lf centroid[%d][1] = %lf\n",i,host_centroids[i],i,host_centroids[i+1]);
   }
   
   error = cudaMemcpy(dev_data,host_data,N*3*sizeof(double),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
   
   int *h_noOfCPoints = (int*)calloc(3,sizeof(int));
   int *c_noOfCPoints;
   error = cudaMalloc((void**)&c_noOfCPoints,3*sizeof(int));
   checkCudaError(error,__LINE__-1);
   error = cudaMemcpy(c_noOfCPoints,h_noOfCPoints,3*sizeof(int),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
 
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   
   double previous_centroids[6];
  //To count the number of itaerations required to converge 
  int noOfIterations = 0;
   while(1){
   noOfIterations++;
   //Keep a copy of centroids of previous iterations
   //So as to compare later
   for(int i=0;i<6;i++){
       previous_centroids[i] = host_centroids[i] ;
   }
   //transfer the centroids to gpu
   error = cudaMemcpy(dev_centroids,host_centroids,6*sizeof(double),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
   for(int i=0;i<3;i++){
       h_noOfCPoints[i] = 0 ;
   }
   //transfer the number of points in each cluster to gpu 
   error = cudaMemcpy(c_noOfCPoints,h_noOfCPoints,3*sizeof(int),cudaMemcpyHostToDevice);
   checkCudaError(error,__LINE__-1);
   
   //kernel computes distance and assign the data points to one of the clusters
   KMeansClustering<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_centroids,dev_data,dev_c_1,dev_c_2,dev_c_3,N,c_noOfCPoints);
   error = cudaGetLastError();
   checkCudaError(error,__LINE__-2);
   //get the number of points in each cluster to cpu
   error = cudaMemcpy(h_noOfCPoints,c_noOfCPoints,3*sizeof(int),cudaMemcpyDeviceToHost);
   checkCudaError(error,__LINE__-1);
   printf("\ncluster points %d %d %d\n",h_noOfCPoints[0],h_noOfCPoints[1],h_noOfCPoints[2]);  
   int blockSize = THREADS_PER_BLOCK;
   int temp = N;
   
   //repeatedly call the reduction kernel
   //so as to compute sum of all points in each cluster
   while(1){
   
      if(temp>blockSize){        
          sumCluster<<<temp/blockSize,blockSize>>>(dev_c_1,dev_c_2,dev_c_3,temp*2);
          error = cudaGetLastError();
          checkCudaError(error,__LINE__-2);
      }
      //If only 32 values are there only one block is enough of 32 threads required
      else if (temp >= 32){
          sumCluster<<<1,temp>>>(dev_c_1,dev_c_2,dev_c_3,temp*2);
          error = cudaGetLastError();
          //printf("%d,%d\n",temp,blockSize);
          checkCudaError(error,__LINE__-2);
          break;
      }
      else{//if less than 32 items to be added
           //add them serially
          error = cudaMemcpy(host_cluster_1,dev_c_1,temp*2*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(error,__LINE__-1);
          error = cudaMemcpy(host_cluster_2,dev_c_2,temp*2*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(error,__LINE__-1);
          error = cudaMemcpy(host_cluster_3,dev_c_3,temp*2*sizeof(double),cudaMemcpyDeviceToHost);
          checkCudaError(error,__LINE__-1);
          for(int i = 1 ; i < temp ; i++){
              host_cluster_1[0] += host_cluster_1[2*i];
              host_cluster_1[1] += host_cluster_1[2*i+1];
              host_cluster_2[0] += host_cluster_2[2*i];
              host_cluster_2[1] += host_cluster_2[2*i+1];
              host_cluster_3[0] += host_cluster_3[2*i];
              host_cluster_3[1] += host_cluster_3[2*i+1];       
          }
        break;    
      }
     if(temp > blockSize){
     temp = temp/blockSize;
     }      
   }
   //transfer the sum calculated to the host
   if(temp>=32){
       error = cudaMemcpy(host_cluster_1,dev_c_1,2*sizeof(double),cudaMemcpyDeviceToHost);
       checkCudaError(error,__LINE__-1);
       error = cudaMemcpy(host_cluster_2,dev_c_2,2*sizeof(double),cudaMemcpyDeviceToHost);
       checkCudaError(error,__LINE__-1);
       error = cudaMemcpy(host_cluster_3,dev_c_3,2*sizeof(double),cudaMemcpyDeviceToHost);
       checkCudaError(error,__LINE__-1);       
   }
     

   double sumXcluster1 = host_cluster_1[0];
   double sumYcluster1 = host_cluster_1[1];
 
   double sumXcluster2 = host_cluster_2[0];
   double sumYcluster2 = host_cluster_2[1];
 
   double sumXcluster3 = host_cluster_3[0];
   double sumYcluster3 = host_cluster_3[1];
    
   //compute the centroids by dividing the sum of points in each
   //cluster by the total no of points in each cluster
   host_centroids[0] = sumXcluster1/(double)h_noOfCPoints[0];
   host_centroids[1] = sumYcluster1/(double)h_noOfCPoints[0];
 
   host_centroids[2] = sumXcluster2/(double)h_noOfCPoints[1];
   host_centroids[3] = sumYcluster2/(double)h_noOfCPoints[1];
 
   host_centroids[4] = sumXcluster3/(double)h_noOfCPoints[2];
   host_centroids[5] = sumYcluster3/(double)h_noOfCPoints[2];
   
   for(int i=0;i<=4;i+=2){
      printf("centroid[%d][0] = %lf centroid[%d][1] = %lf\n",i,host_centroids[i],i,host_centroids[i+1]);
   }
   int count = 0;
   //if all the centroids are same as in previous iteration
   //algorithm has converged
   for(int i=0;i<6;i++){
      if(host_centroids[i] != previous_centroids[i]){
        break;
      }
      count++;
   }
   if(count == 6){
     break;
   }
 }
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   for(int i=0;i<=4;i+=2){
      printf("centroid[%d][0] = %lf centroid[%d][1] = %lf\n",i,host_centroids[i],i,host_centroids[i+1]);
   }
   
   //Total no computations for one iteration
   //16 * N for kernel 1
   //2 * N for kernel 2
   double throughput = (24  *sizeof(double)* 2.0 * noOfIterations) *N/(1000*milliseconds);
   printf("\nThroughput is %lf MFLOPS",throughput);
   printf("\nTime is %f ms\n",milliseconds);

  return 0; 
}
 
 


