#include<stdio.h>
#include<stdlib.h>
int randomNo;


int main (){
	
	for(int i = 0;i<=20;i++){
		randomNo = rand()%100;
		printf("%d\n",randomNo);
	}
	
return 0;


}




