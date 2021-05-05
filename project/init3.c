#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SRAND_VALUE 1989

#define dim 1000 // grid dimension excluding ghost cells
#define STARVE 5

//Function to init the grid

void printIt(int *grid)
{
    for (int i = 1; i <= dim; i++)
    {
        for (int j = 1; j <= dim; j++)
        {
            int id = i * (dim + 2) + j;
            //printf("%d ", grid[id]);
        }
        //printf("%c",'\n');
    }
    //printf("%c",'\n');
}

void makeGrid(int *grid)
{
    //seed random value, kept constant for validity checking
    srand(SRAND_VALUE);
    int rval;
    int guy;
    int tot = 0;
    //init each cell
    for (int i = 1; i <= dim; i++)
    {
        for (int j = 1; j <= dim; j++)
        {
            //kept at 20 to allow for differing probabilites, as opposed to uniform from %3.
            rval = rand() % 20;
            if (rval == 0)
            {
                tot++;
                //Initialize to fully fed. The exponent is for extensibility reasons, so that if we want to add a hunger counter to the prey we can do that without overlapping numbers, since powers of two will be unique, and we can then use powers of three for the prey hunger.
                guy = 1 << (STARVE);
            }
            else if (rval < 5)
            {
                //set to 0, AKA no predator or prey
                guy = 0;
            }
            else
            {
                //since prey dies off the fastest, we should initalize the most of them.
                tot++;
                guy = 1;
            }
            grid[i * (dim + 2) + j] = guy;
            //printf(" %d ", guy);
        }
        //printf("%c", '\n');
    }
}

void makeMove(int id, int dir, int *mGrid, int *grid)
{
    switch (dir)
    {

        //move is up
    case 1:
        grid[id - (dim + 2)] = grid[id];
        mGrid[id - (dim + 2)] = 1;
        grid[id] = 0;
        break;
        //move is Right
    case 2:
        grid[id + 1] = grid[id];
        mGrid[id + 1] = 1;
        grid[id] = 0;
        break;
        //Down
    case 3:
        grid[id + 2 + dim] = grid[id];
        mGrid[id + (dim + 2)] = 1;
        grid[id] = 0;
        break;
        //Left
    case 4:
        grid[id - 1] = grid[id];
        mGrid[id - 1] = 1;
        grid[id] = 0;
        break;

    default:

        mGrid[id] = 1;
        break;
    }
}
int randomlyDecideDirection(int *grid, int id, int *dirs)
{
    // int *dirs = (int *)malloc(4 * sizeof(int));

    int dSize = 4;
    int pos = 210;
    if (grid[id] == 0)
    {
        return (0);
    }
    //pos is the product of the first 4 primes (2*3*5*7).
    //let 2=up,3=east,5,south,and 7 west
    //we divide by the nth prime to signify that a certain direction can
    // NOT be moved into. The resulting number is guarenteed
    //to be unique for that series of directions.
    //Then, we use a case statement
    //to uniquely identify that combination of directions.
    //From there we pick one direction randomly of
    //our possibilities.

    //can't move down
    if (grid[id + (dim + 2)] || id >= ((dim + 1) * (dim + 1)))
    {
        pos /= 5;
    }
    //can't move up
    if (grid[id - (dim + 2)] || id < 2 * (dim + 2))
    {
        pos /= 2;
    }

    //can't move right
    if (grid[id + 1] || id % (dim + 2) == dim)
    {
        pos /= 3;
    }

    //can't move left
    if (grid[id - 1] || id % (dim + 2) == 1)
    {
        pos /= 7;
    }
    switch (pos)
    {
    case 210:
        dSize = 4;
        dirs[0] = 1;
        dirs[1] = 2;
        dirs[2] = 3;
        dirs[3] = 4;

        break;
    case 30:
        dirs[0] = 1;
        dirs[1] = 2;
        dirs[2] = 3;
        dSize = 3;
        break;
    case 42:
        dirs[0] = 1;
        dirs[1] = 2;
        dirs[2] = 4;
        dSize = 3;
        break;
    case 70:
        dirs[0] = 1;
        dirs[1] = 3;
        dirs[2] = 4;
        dSize = 3;
        break;
    case 105:
        dirs[0] = 3;
        dirs[1] = 2;
        dirs[2] = 4;
        dSize = 3;
        break;
    case 6:
        dirs[0] = 1;
        dirs[1] = 2;
        dSize = 2;
        break;
    case 10:
        dirs[0] = 1;
        dirs[1] = 3;
        dSize = 2;
        break;
    case 15:
        dirs[0] = 2;
        dirs[1] = 3;
        dSize = 2;
        break;
    case 14:
        dirs[0] = 1;
        dirs[1] = 4;
        dSize = 2;
        break;
    case 21:
        dirs[0] = 2;
        dirs[1] = 4;
        dSize = 2;
        break;
    case 35:
        dirs[0] = 3;
        dirs[1] = 4;
        dSize = 2;
        break;
    case 7:
        return (4);
    case 5:
        return (3);
    case 3:
        return (2);
    case 2:
        return (1);
    default:
        return (0);
    }
    int ret = (rand() % (dSize));

    //////printf("Direction! %d \n", dirs[ret]);
    return (dirs[ret]);
}

void ghost(int *grid)
{
    int i, j;

// ghost rows
#pragma acc kernels
    {
#pragma acc loop independent
        for (i = 1; i <= dim; i++)
        {
            // copy first row to bottom ghost row
            grid[(dim + 2) * (dim + 1) + i] = grid[(dim + 2) + i];

            // copy last row to top ghost row
            grid[i] = grid[(dim + 2) * dim + i];
        }

// ghost columns
#pragma acc loop independent
        for (i = 0; i <= dim + 1; i++)
        {
            // copy first column to right most ghost column
            grid[i * (dim + 2) + dim + 1] = grid[i * (dim + 2) + 1];

            // copy last column to left most ghost column
            grid[i * (dim + 2)] = grid[i * (dim + 2) + dim];
        }
    }
    return;
}

void gol(int *grid, int *newGrid)
{
    ghost(grid);
    //  printIt(grid);
    int Left, Right, Up, Down;

    int i, j;
    int checked;

// we iterate over the grid
#pragma acc kernels
    {
#pragma acc loop gang, vector(128)
        for (i = 1; i <= dim; i++)
        {
            for (j = 1; j <= dim; j++)
            {
                int id = i * (dim + 2) + j;
                ////printf(" %d! ", id);
                //initalize directions for ease of notation
                Left = id - 1;
                Right = id + 1;
                Up = id - (dim + 2);
                Down = id + (dim + 2);
                checked = 0;

                /*int numNeighbors =
                grid[id + (dim + 2)] + grid[id - (dim + 2)]    // lower + upper
                + grid[id + 1] + grid[id - 1]                  // right + left
                + grid[id + (dim + 3)] + grid[id - (dim + 3)]  // diagonal lower + upper right
                + grid[id - (dim + 1)] + grid[id + (dim + 1)]; // diagonal lower + upper left


            // the game rules
            if (grid[id] == 1 && numNeighbors < 2)
                newGrid[id] = 0;
            else if (grid[id] == 1 && (numNeighbors == 2 || numNeighbors == 3))
                newGrid[id] = 1;
            else if (grid[id] == 1 && numNeighbors > 3)
                newGrid[id] = 0;
            else if (grid[id] == 0 && numNeighbors == 3)
                newGrid[id] = 1;
            else
                newGrid[id] = grid[id]; */

                if (grid[id] == 0)
                {
                    newGrid[id] = 0;
                }
                else if (grid[id] == 1)
                {
                    // //
                    //Checks if the id is NOT on the far top of the grid
                    //and that there is a predator to the top

                    if (id > (2 * (dim + 2)) && (grid[Up]) > 1)
                    {
                        //printf("DEaten %d!", id);

                        newGrid[Up] = 1 << (STARVE + 1);
                        checked = 1;
                    }
                    //Checks if the id is NOT on the far bottom of the grid
                    //and that there is a predator to the botto,
                    if (id < ((dim + 1) * (dim + 1)) && (grid[Down]) > 1)
                    {

                        newGrid[Down] = 1 << (STARVE + 1);
                        //printf("UEaten %d! Here %d %c", id, Down, '\n');

                        checked = 1;
                    }
                    //Checks if the id is NOT on the far right of the grid
                    //and that there is a predator to the right
                    if ((id % (dim + 2) != dim && grid[Right] > 1))
                    {
                        //printf("REaten %d!", id);

                        newGrid[Right] = 1 << (STARVE + 1);
                        checked = 1;
                    }
                    //Checks if the id is NOT on the far LEFT of the grid
                    //and that there is a predator to the left
                    if (id % (dim + 2) != 1 && (grid[Left]) > 1)
                    {
                        //printf("LEaten %d!", id);

                        newGrid[Left] = 1 << (STARVE + 1);
                        checked = 1;
                    }
                    if (checked)
                    {
                        //   //printf("Eaten %d!", id);
                        newGrid[id] = 0;
                    }
                    else
                    {
                        newGrid[id] = 1;
                    }
                }
                //If it has hit 2, it's starved.
                else if (grid[id] == 2)
                {
                    newGrid[id] = 0;
                }
                //Otherwise, divide by two. Note we do this even when the entity ate in the same step, for simplicity's sake, which is why we add 1 to starve in the right shift
                else if (grid[id] > 2)
                {
                    //printf("%dbing%d", grid[id], newGrid[id]);
                    if (newGrid[id] > grid[id])
                    {

                        grid[id] = newGrid[id];
                    }
                    else
                    {
                        newGrid[id] = grid[id];
                    }

                    // ////printf("Cut! %d", id);
                }
                //   //printf(" %d ", newGrid[id]);
            }
            //
        }
//  printIt(newGrid);
// //printf("b4  copy %c", '\n');

// copy new grid over, as pointers cannot be switched on the device
#pragma acc loop independent
        for (i = 1; i <= dim; i++)
        {
            for (j = 1; j <= dim; j++)
            {
                int id = i * (dim + 2) + j;
                grid[id] = newGrid[id];
                if (newGrid[id] > 2)
                    grid[id] = grid[id] / 2;
            }
        }
        //   printIt(grid);
        // //printf("aft  copy %c", '\n');
    }
}

int main(int argc, char *argv[])
{
    int i, j;

    // number of game steps
    int itEnd = 5000;
    if (argc > 1)
    {
        itEnd = atoi(argv[1]);
    }

    // grid array with dimension dim + ghost columns and rows
    int arraySize = (dim + 2) * (dim + 2);
    size_t bytes = arraySize * sizeof(int);
    int *grid = (int *)malloc(bytes);
    int *mGrid = (int *)malloc(bytes);
    int *dirs = (int *)malloc(4 * sizeof(int));

    // allocate result grid
    int * /*restrict*/ newGrid = (int *)malloc(bytes);

    // assign initial population randomly
    int tot;
    makeGrid(grid);
    double st = omp_get_wtime(); // start timing
    ////printf("%c", '\n');

    int total = 0; // total number of alive cells
    int it;
    int dir;
    int Left, Right, Up, Down;
    //////printf("Hey! %d \n", itEnd);
    for (it = 0; it < itEnd; it++)
    {

        tot = 0;
        //        printIt(grid);

        gol(grid, newGrid);
        //
        // printIt(grid);

        for (i = 1; i <= dim; i++)
        {
            for (j = 1; j <= dim; j++)
            {
                int id = i * (dim + 2) + j;
                Left = id - 1;
                Right = id + 1;
                Up = id - (dim + 2);
                Down = id + (dim + 2);

                //if the entity is moved, don't move it again.
                if (mGrid[id])
                {
                    //  //printf("0");
                    continue;
                }

                dir = randomlyDecideDirection(grid, id, dirs);
                makeMove(id, dir, mGrid, grid);

                // //printf("%d", dir);
            }
            //printf("%c", '\n');
        }
        //printf("%c", '\n');
        //
        //   printIt(grid);

        // sum up alive cells
        for (i = 1; i <= dim; i++)
        {
            for (j = 1; j <= dim; j++)
            {
                mGrid[i * (dim + 2) + j] = 0;

                if (grid[i * (dim + 2) + j] != 0)
                {
                    tot++;
                }
                if (grid[i * (dim + 2) + j] > 2)
                {
                }
                if (grid[i * (dim + 2) + j] > 9)
                {
                    //printf(" %d ", grid[i * (dim + 2) + j]);
                }
                //  else
                //printf(" %d  ", grid[i * (dim + 2) + j]);
            }
            //printf("\n");
        }
        //printf("\n");

        //printf("Botal Alive: %d\n", tot);
    }
    double runtime = omp_get_wtime() - st;

    printf(" total time: %f s\n", runtime);
    free(dirs);
    free(grid);
    free(newGrid);
    free(mGrid);
    return 0;
}
