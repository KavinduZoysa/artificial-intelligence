#include<stdio.h>
#include<stdlib.h>

int main(){
	int randNo;
	randNo = 0;
	for (int i=0;i<15;i++){
		printf("%d\n",rand()%200);	
	}
	return 0;
}
