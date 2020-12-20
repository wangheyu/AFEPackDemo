////////////////////////////////////////////////////////////////////////////////////////////
// main.cpp :
//

#include "simulation2D.h"

int main(int argc, char * argv[])
{
    if (argc != 2) {
	std::cout << "Usage: " << argv[0] 
		  << " meshfile" << std::endl;
	return 1;
    }
    Simulation2D the_app(argv[1]);
    the_app.run();
    return 0;
};

//
// end of file
////////////////////////////////////////////////////////////////////////////////////////////
