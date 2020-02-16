#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
# define N 4194304

int main() {
    //Read the data from the file
    FILE *inFile = fopen("4194304_CLUSTER_DATA.csv", "r");
    if(inFile == NULL){
        exit(1);
    }
    int i,j;
    //Create a two dimensional array to store the data
    //Input data consists of 2 dimensional data points (x,y)
    double **data = (double **)malloc(sizeof(double*)*N);
    for(i=0;i<N;i++){
       data[i] = (double *)malloc(sizeof(double)*3);
    }
    //Read the data from the file and store it in the 2D array
    for(i=0;i<N;i++){
        fscanf(inFile, "%lf,%lf,%lf\n", &data[i][0],&data[i][1],&data[i][2]);
    }
    
   //Allocating memory to store the points in the clusters
   double **cluster_1 = (double **)malloc(sizeof(double*)*N);
   double **cluster_2 = (double **)malloc(sizeof(double*)*N);
   double **cluster_3 = (double **)malloc(sizeof(double*)*N);

   for(i=0;i<N;i++){
      cluster_1[i] = (double *)calloc(2,sizeof(double));
      cluster_2[i] = (double *)calloc(2,sizeof(double));
      cluster_3[i] = (double *)calloc(2,sizeof(double));
   }


   double centroid_1[2];
   double centroid_2[2];
   double centroid_3[2];

   //Randomly initialising K centroids for the clusters
   srand(41);
   int index1 = rand() % N;
   
   centroid_1[0] = data[index1][0];
   centroid_1[1] = data[index1][1];
   int index2 = rand() % N;
   
   centroid_2[0] = data[index2][0];
   centroid_2[1] = data[index2][1];
   int index3 = rand() % N;
   
   centroid_3[0] = data[index3][0];
   centroid_3[1] = data[index3][1];

   //Allocating memory to store the distance of points from each cluster
   double *distance_from_centroid_1 = (double *)calloc(N,sizeof(double));
   double *distance_from_centroid_2 = (double *)calloc(N,sizeof(double));
   double *distance_from_centroid_3 = (double *)calloc(N,sizeof(double));
 
   printf("initial\n");
   printf("%lf %lf \n",centroid_1[0],centroid_1[1]);
   printf("%lf %lf \n",centroid_2[0],centroid_2[1]);
   printf("%lf %lf \n",centroid_3[0],centroid_3[1]);

 
   clock_t start_time = clock();
   //Count of number of iterations required to converge
   int count = 0;
   while(1){
   count++;

   
   //Computing euclidean distance of the points from each of the centroids
   for(i=0;i<N;i++){
      distance_from_centroid_1[i] = sqrt(pow(centroid_1[0]-data[i][0],2)+pow(centroid_1[1]-data[i][1],2));
      distance_from_centroid_2[i] = sqrt(pow(centroid_2[0]-data[i][0],2)+pow(centroid_2[1]-data[i][1],2));
      distance_from_centroid_3[i] = sqrt(pow(centroid_3[0]-data[i][0],2)+pow(centroid_3[1]-data[i][1],2));
   }
   
   //Creating cluster of points based on the distance of point from each cluster
   //Assigning the point to the closest cluster based on distance from the centroid
   for(i=0;i<N;i++){
     if(distance_from_centroid_1[i]>distance_from_centroid_2[i]){
       if(distance_from_centroid_2[i]>distance_from_centroid_3[i]){
       // Point is closest to centroid 3
       cluster_3[i][0] = data[i][0];
       cluster_3[i][1] = data[i][1];

       cluster_1[i][0] = 0.0;
       cluster_1[i][1] = 0.0;

       cluster_2[i][0] = 0.0;
       cluster_2[i][1] = 0.0;
       }else{
       //Point is closest to centroid 2
       cluster_2[i][0] = data[i][0];
       cluster_2[i][1] = data[i][1];
       
       cluster_1[i][0] = 0.0;
       cluster_1[i][1] = 0.0;

       cluster_3[i][0] = 0.0;
       cluster_3[i][1] = 0.0;

       }
     }
    else if (distance_from_centroid_1[i] < distance_from_centroid_3[i]){
       //Point is closest to centroid 1 
       cluster_1[i][0] = data[i][0];
       cluster_1[i][1] = data[i][1];
       
       cluster_2[i][0] = 0.0;
       cluster_2[i][1] = 0.0;

       cluster_3[i][0] = 0.0;
       cluster_3[i][1] = 0.0;

     }else{
      //Point closest to centroid 3 
      cluster_3[i][0] = data[i][0];
      cluster_3[i][1] = data[i][1];
      
      cluster_1[i][0] = 0.0;
      cluster_1[i][1] = 0.0;

      cluster_2[i][0] = 0.0;
      cluster_2[i][1] = 0.0;
     }

   }
   
   // Computing the new centroids based on the clusters formed 
   int no_of_cluster_1_points = 0;
   int no_of_cluster_2_points = 0;
   int no_of_cluster_3_points = 0;
   
   double prev_centroid_1[2];
   double prev_centroid_2[2];
   double prev_centroid_3[2];

   prev_centroid_1[0] = centroid_1[0];
   prev_centroid_1[1] = centroid_1[1];

   prev_centroid_2[0] = centroid_2[0];
   prev_centroid_2[1] = centroid_2[1];

   prev_centroid_3[0] = centroid_3[0];
   prev_centroid_3[1] = centroid_3[1];
   
   centroid_1[0] = 0;
   centroid_1[1] = 0;
   
   centroid_2[0] = 0;
   centroid_2[1] = 0;

   centroid_3[0] = 0;
   centroid_3[1] = 0;
   
   //Taking the sum of all the points in each cluster and
   //counting the number of points on each cluster
   for(i=0;i<N;i++){
     if(cluster_1[i][0]!=0 && cluster_1[i][1]!=0){
       centroid_1[0]+=cluster_1[i][0];
       centroid_1[1]+=cluster_1[i][1];
       no_of_cluster_1_points++;
     }
     if(cluster_2[i][0]!=0 && cluster_2[i][1]!=0){
       centroid_2[0]+=cluster_2[i][0];
       centroid_2[1]+=cluster_2[i][1];
       no_of_cluster_2_points++;
     }
     if(cluster_3[i][0]!=0 && cluster_3[i][1]!=0){
       centroid_3[0]+=cluster_3[i][0];
       centroid_3[1]+=cluster_3[i][1];
       no_of_cluster_3_points++;
     }
   }
   printf("No of cluster 1 points: %d\n",no_of_cluster_1_points);
   printf("No of cluster 2 points: %d\n",no_of_cluster_2_points);
   printf("No of cluster 3 points: %d\n",no_of_cluster_3_points);
   
   //Computing the new centroids based on the clusters formed 
   centroid_1[0] = centroid_1[0]/no_of_cluster_1_points;
   centroid_1[1] = centroid_1[1]/no_of_cluster_1_points;

   centroid_2[0] = centroid_2[0]/no_of_cluster_2_points;
   centroid_2[1] = centroid_2[1]/no_of_cluster_2_points;

   centroid_3[0] = centroid_3[0]/no_of_cluster_3_points;
   centroid_3[1] = centroid_3[1]/no_of_cluster_3_points;
   
   printf("Centroid 1: %lf %lf\n",centroid_1[0],centroid_1[1]);
   printf("Centroid 2: %lf %lf\n",centroid_2[0],centroid_2[1]);
   printf("Centroid 3: %lf %lf\n",centroid_3[0],centroid_3[1]);

   //If centroids are same as the previous iteration
   //algorithm has converged
   if(prev_centroid_1[0] == centroid_1[0] &&
      prev_centroid_1[1] == centroid_1[1] &&
      prev_centroid_2[0] == centroid_2[0] &&
      prev_centroid_2[1] == centroid_2[1] &&
      prev_centroid_3[0] == centroid_3[0] &&
      prev_centroid_3[1] == centroid_3[1]){
      break;
      }
   }
   clock_t timeforexec = clock() - start_time;
   double totalTime = ((double)(timeforexec)/CLOCKS_PER_SEC);
   double throughput = (count*24*2.0*N*sizeof(double))/(totalTime*1000000);
   printf("Total Time: %lf seconds\n",totalTime);
   printf("Throughput: %lf MFLOPS\n",throughput);

   //Final centroid of the clusters, after the KMeans Clustering algorithm
   //Has converged
   printf("Centroid 1: %lf %lf\n",centroid_1[0],centroid_1[1]);
   printf("Centroid 2: %lf %lf\n",centroid_2[0],centroid_2[1]);
   printf("Centroid 3: %lf %lf\n",centroid_3[0],centroid_3[1]);
   
}
