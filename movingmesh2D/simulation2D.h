/**
 * @file   simulation2D.h
 * @author Heyu Wang <scshw@cslin107.csunix.comp.leeds.ac.uk>
 * @date   Mon Mar 10 17:26:22 2014
 * 
 * @brief Building a framkwork for time dependent problems with moving
 * mesh method.
 * 
 * 
 */

#include <AFEPack/MovingMesh2D.h>
#include <AFEPack/HGeometry.h>
#include <AFEPack/Operator.h>
#include <AFEPack/BilinearOperator.h>

#include <AFEPack/AMGSolver.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>


/**
 * A template for double value function dependent on t and parameter a. 
 * 
 */
class ArgFunction : public Function<double>
{
private:
    double a;			/**< parameter. */
    double t;			/**< time. */
public:
    /** 
     * The function is depend on parameter a and time t.
     * 
     * @param _a parameter from outside. 
     * @param _t time t.
     * 
     * @return The function value not return here.
     */
    ArgFunction(const double& _a, const double& _t) :
	a(_a), t(_t)
	{};
    
    /** 
     * Destructor.
     * 
     * 
     * @return 
     */
    ~ArgFunction() {};
public:
    /** 
     * Function value return here.
     * 
     * 
     * @return function value.
     */
    virtual double value(const double *) const;
    
    /** 
     * Gradient vector of the function.
     * 
     * 
     * @return gradient vector.
     */
    virtual std::vector<double> gradient(const double *) const;
};


/**
 * A template of 2D simulation with time dependence.
 * 
 */
class Simulation2D : public MovingMesh2D
{
public:
    /**
     * The stiffMatrix constuctor.
     * 
     */
    class Matrix : public StiffMatrix<2, double>
    {
    private:
	double dt;		/**< time step. */
	double a;		/**< parameter from outside. */
    public:
	/** 
	 * Constructor of the Matrix class. Depend on the special
	 * finite element space, the time step and the parameter.
	 * 
	 * @param sp the finite element space.
	 * @param _dt time step.
	 * @param _a perameter.
	 * 
	 * @return same as other C++ constructor.
	 */
	Matrix(FEMSpace<double, 2>& sp, const double& _dt, const double& _a) :
	    dt(_dt), a(_a),
	    StiffMatrix<2,double>(sp) {};
	/** 
	 * A standard destructor.
	 * 
	 * 
	 * @return nothing specia.
	 */
	virtual ~Matrix() {};
    public:
	/** 
	 * This function really assemble the stiff matrix.
	 * 
	 * @param e0 one element in the space.
	 * @param e1 another element in the space.
	 * @param state some parameter I don't know the usage.
	 */
	virtual void getElementMatrix(const Element<double,2>& e0,
				      const Element<double,2>& e1,
				      const ActiveElementPairIterator<2>::State state);
    };

private:
    /// geometry template;
    TemplateGeometry<2> template_geometry; 
    /// coordinate transform;
    CoordTransform<2,2> coord_transform;
    /// template of the degree of freedom;
    TemplateDOF<2> template_dof;
    /// the administrator of the basis function;
    BasisFunctionAdmin<double,2,2> basis_function;
    /// element template;
    std::vector<TemplateElement<double,2,2> > template_element;
    /// finite element space;
    FEMSpace<double,2> fem_space;
    /// mesh file;
    std::string mesh_file;
    /// current time of the simulation;
    double t;
    /// current time step of the simulation;
    double dt;
    /// a parameter from outside;
    double a;
    /// current numerical solution;
    FEMFunction<double,2> u_h;
public:
    /** 
     * Constructor.
     * 
     * @param file mesh file name.
     */
    Simulation2D(const std::string& file);
    /** 
     * A standard destructor.
     * 
     */
    virtual ~ Simulation2D();
public:
    /** 
     * Main procedure.
     * 
     */
    void run();

    /** 
     * The simulation develop for one time step.
     * 
     */
    void stepForward();

    /** 
     * All the prepare things.
     * 
     */
    void initialize();

    /** 
     * Arrange the initial values of the simulation.
     * 
     */
    void initialValue();

    /** 
     * Set the boundary values here, but may useless.
     * 
     */
    void boundaryValue();

    /** 
     * Output the current numerical solution.
     * 
     */
    virtual void outputSolution();

    /**
     * Compute the monitor value on each element.
     * 
     */
    virtual void getMonitor();

    /**
     * Update the numerical solution from old mesh to new one.
     * 
     */
    virtual void updateSolution();
};

//
// end of file
//////////////////////////////////////////////////////////////////////////////

