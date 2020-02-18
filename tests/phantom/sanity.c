#include <stdio.h>

int main() {

	int data[3];
	printf("Enter elements: ");
	
	for (int i = 0; i < 3; ++i)
		scanf("%d", data + i);
	printf("You entered: \n");

	for (int i = 0; i < 3; ++i)
		printf("%d\n", *data[0][i]);

	return 0; }
