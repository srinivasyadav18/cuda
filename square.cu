#include<stdio.h>
#include<unistd.h>
__global__ void square(float *d_out,float *d_in){
	int id= threadIdx.x+blockDim.x*blockIdx.x;
	float f = d_in[id];
	d_out[id]=f+f;
}
int main(){
	const int ARRAY_SIZE=4096;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	float *h_in=(float*)malloc(ARRAY_BYTES);
	float *h_out=(float*)malloc(ARRAY_BYTES);

	for (int i=0;i< ARRAY_SIZE ; i++)
	{
		h_in[i]=i ;
	}
	printf("Array on which we are performing :\n");
	for (int i=0;i< ARRAY_SIZE ; i++)
	{
		printf("%.2f", h_in[i]);
		if (i%4==3) printf("\n");
		else printf("\t");
	}
	
	float *d_in;
	float *d_out;

	cudaMalloc((void **) &d_in,ARRAY_BYTES);
	cudaMalloc((void **) &d_out,ARRAY_BYTES);

	cudaMemcpy (d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);
	
	square<<<8,512>>>(d_out,d_in);
	
	cudaMemcpy (h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost);
	
	printf("\nprinting out results\n");

	
	
	for(int i=0;i< ARRAY_SIZE ; i++)
	{
		printf("%.2f",h_out[i]);
        if (i%4==3) printf("\n");
        else printf("\t");

	}
	printf("\nClearing the memory\n");

	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;


}
