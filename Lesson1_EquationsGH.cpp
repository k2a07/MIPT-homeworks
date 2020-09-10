#include <stdio.h>
#include <math.h>

//-----------------------------------------------------------------------------

void LinearEq (double a, double b);
void SquareEq (double a, double b, double c);

void EqualityType ();

//-----------------------------------------------------------------------------

int main()
    {
    printf ("HELLO! KEQUATIONS IS A PROGRAM WHICH SOLVES LINEAR AND SQUARE EQUARTION  (c)Kirill Acharya\n\n");

    EqualityType();
    }

//-----------------------------------------------------------------------------

void LinearEq (double a, double b)
    {
    if (a != 0)
        {
        double x = -b/a;
        printf ("Congrats! The root is: %lf", x);
        }

    if (a == 0 && b != 0)
        printf ("Oops... Sorry, there are no roots");

    if (a == 0 && b == 0)
        printf ("Wow, you're lucky! There are infinite number of roots!");

    }

//-----------------------------------------------------------------------------

void SquareEq (double a, double b, double c)
    {
    if (a == 0) LinearEq (b, c);

    else
        {
        double Discriminant = b*b - 4*a*c;

        if (Discriminant > 0)
            {
            double x1 = 0, x2 = 0;

            x1 = (-b - sqrt(Discriminant))/(2*a);
            x2 = (-b + sqrt(Discriminant))/(2*a);

            printf ("Congrats! There are two roots: %lf, %lf", x1, x2);
            }

        if (Discriminant == 0)
            {
            double x = 0;

            x = (-b)/(2*a);

            printf ("There is one root: %lf", x);
            }

        if (Discriminant < 0)
            {
            printf ("Oops... Sorry, there are no roots");
            }
        }
    }


//-----------------------------------------------------------------------------

void EqualityType()
    {
    double a = 0, b = 0, c = 0, d = 0;

    printf ("Type <1> if you want to solve a linear equation\n"
            "Type <2> if you want to solve a square equation\n");

    int eq_type = 0;
    scanf ("%d", &eq_type);

    if (eq_type == 1)
        {
        printf ("The format is the following: a*x + b = 0. Please enter the coefficients a and b.\n");

        printf ("\na = ");
        scanf ("%lf", &a);

        printf ("b = ");
        scanf ("%lf", &b);

        LinearEq (a, b);
        }

    else if (eq_type == 2)
        {
        printf ("The format is the following: a*x^2 + b*x + c = 0. Please enter the coefficients a, b and c.\n");

        printf ("\na = ");
        scanf ("%lf", &a);

        printf ("b = ");
        scanf ("%lf", &b);

        printf ("c = ");
        scanf ("%lf", &c);

        SquareEq (a, b, c);
        }

    else
        {
        printf ("\nElon Musk would say: Just read the instructions! Try once again\n\n");

        EqualityType();
        }
    }
