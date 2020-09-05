#include <stdio.h>


void LinearEq (double a, double b);

int main()
    {
    double a = 0, b = 0;

    printf ("Linear Equation is a program which helps you to solve linear equation. (c) Kirill Acharya\n"
            "The format is the following: a*x + b = 0. Please enter the coefficients a and b.\n");


    printf ("Enter a: ");
    scanf ("%lf", &a);

    printf ("Enter b: ");
    scanf ("%lf", &b);

    printf ("\n");

    LinearEq (a, b);
    }

/*
1) 1 root      ==> a != 0
2) no roots    ==> a = 0 and b != 0
3) infty roots ==> a = 0 and b = 0


*/


void LinearEq (double a, double b)
    {
    if (a != 0)
        {
        double x = -b/a;
        printf ("Congrats! The value is: %lf", x);
        }

    if (a == 0 && b != 0)
        printf ("Oops... Sorry, there are no roots");

    if (a == 0 && b == 0)
        printf ("Wow, you're lucky! There are infinite number of roots!");

    }

