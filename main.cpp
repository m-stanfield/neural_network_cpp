#include <iostream>
#include "Node.h"
#include "Network.h"
#include "test_functions.h"
#include <fstream>
#include <string>
#include <vector>
#include <random>

using namespace std;


float equation(float point)
{

    return 0.1*point*point;
}



int main()
{



    test_and_update();
    test_or_update();
    test_xor_update();
    test_nor_update();
    test_nand_update();
  //  test_xor_small_update();

    return 0;
}
