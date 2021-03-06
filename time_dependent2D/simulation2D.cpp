/**
 * @file   simulation2D.cpp
 * @author Heyu Wang <scshw@cslin107.csunix.comp.leeds.ac.uk>
 * @date   Mon Mar 10 17:57:29 2014
 * 
 * @brief  Build a template for the normal simulation in 2D.
 * 
 * 
 */

#include "simulation2D.h"
#include <sstream>

double g_t;
double g_a;

double boundary_value(const double *);

double boundary_value(const double * p)
{
    return 1. / (1. + exp((p[0] + p[1] - g_t) / (2. * g_a)));
};


/** 
 * Setup the values of the initial function.
 * 
 * @param p input points in 2D space.
 * 
 * @return the initial function value of point p.
 */
double ArgFunction::value(const double * p) const
{
    return 1. / (1. + exp((p[0] + p[1] - t) / (2. * a)));
};

/** 
 * Setup the gradient vector of the initial function.
 * 
 * @param p input points in 2D space.
 * 
 * @return The gradient vector of the initial function in point p. All
 * zeros set here, right???
 */
std::vector<double> ArgFunction::gradient(const double * p) const
{
    std::vector<double> v(2);
    return v;
};

///////////////////////////////////////////////////////////////////////////////


/**
 * The next part really assemble the stiff matrix.
 * 
 */
void Simulation2D::Matrix::getElementMatrix(
    const Element<double,2>& element0,
    const Element<double,2>& element1,
    const ActiveElementPairIterator<2>::State state)
{
    /// usually the degree of freedom sizes of the two spaces (the
    /// test function's and the numerical solution's) are same.
    int n_element_dof0 = elementDof0().size();
    /// but what's will happened if two sizes are different?
    int n_element_dof1 = elementDof1().size();
    /// get the volume of the element (area in 2D).
    double volume = element0.templateElement().volume();
    /// get the quadraure info, with the specified algebric accuracy.
    const QuadratureInfo<2>& quad_info = element0.findQuadratureInfo(algebricAccuracy());
    /// in each quadraure point, the determinant of the jacobian
    /// matrix of the coordinates transform is a scale.
    std::vector<double> jacobian = element0.local_to_global_jacobian(quad_info.quadraturePoint());
    /// the number of the quadrature points in one element.
    int n_quadrature_point = quad_info.n_quadraturePoint();
    /// the real coordinates of the quadrature points of one element,
    /// and has already transfer to the global coordinates.
    std::vector<AFEPack::Point<2> > q_point = element0.local_to_global(quad_info.quadraturePoint());
    /// the values of all the basis function values on all the quadrature points on the real element.
    std::vector<std::vector<double> > basis_value = element0.basis_function_value(q_point);
    /// the gradient vectors of all the basis function on all the quadrature points on the real element.
    std::vector<std::vector<std::vector<double> > > basis_gradient = element0.basis_function_gradient(q_point);
    /// do the numerical integral by all quadrature points, get the
    /// stiff matrix coefficient of j row and k column, this loop
    /// complete the assemble of the stiff matrix.
    for (int l = 0; l < n_quadrature_point; l++) 
    {
	double Jxw = quad_info.weight(l) * jacobian[l] * volume;
	for (int j = 0; j < n_element_dof0; j++) 
	{
	    for (int k = 0; k < n_element_dof1; k++) 
	    {
		elementMatrix(j, k) += Jxw * ((1 / dt) * basis_value[j][l] * basis_value[k][l]
					   + a * innerProduct(basis_gradient[j][l], basis_gradient[k][l]));
	    }
	}
    }
};


/** 
 * Contructor, input the basic parameters, mesh file, parameter a and
 * the start time.
 * 
 * @param file mesh file name.
 */
Simulation2D::Simulation2D(const std::string& file) :
    mesh_file(file), a(0.005), t(0.25)
{
    g_a = a;
    g_t = t;
};


/** 
 * Standard decontructor.
 * 
 */
Simulation2D::~Simulation2D()
{};

/** 
 * All the prepare things here.
 * 
 */
void Simulation2D::initialize()
{
    /// get the mesh data.
    mesh.readData(mesh_file);
    /// 
    template_geometry.readData("triangle.tmp_geo");
    coord_transform.readData("triangle.crd_trs");
    template_dof.reinit(template_geometry);
    template_dof.readData("triangle.1.tmp_dof");
    basis_function.reinit(template_dof);
    basis_function.readData("triangle.1.bas_fun");
    template_element.resize(1);
    template_element[0].reinit(template_geometry,
			       template_dof,
			       coord_transform,
			       basis_function);
    fem_space.reinit(mesh, template_element);
    int n_element = mesh.n_geometry(2);
    fem_space.element().resize(n_element);
    for (int i = 0; i < n_element; i++)
	fem_space.element(i).reinit(fem_space, i, 0);
    fem_space.buildElement();
    fem_space.buildDof();
    fem_space.buildDofBoundaryMark();
    u_h.reinit(fem_space);
    initialValue();
    outputSolution();
};



void Simulation2D::run()
{
    initialize();

    do {
	stepForward();
	outputSolution();
	std::cout << "t  = " << t << std::endl;
    } while (t < 2.0);
};


void Simulation2D::initialValue()
{
    ArgFunction u(a, t);
    Operator::L2Project(u, u_h, Operator::LOCAL_LEAST_SQUARE, 3);
};


void Simulation2D::boundaryValue()
{
};

void Simulation2D::stepForward()
{
    int i, j, k, l;
    dt = 2.0e-3;
    FEMFunction<double,2> _u_h(u_h);
    Matrix matrix(fem_space, dt, a);
    matrix.algebricAccuracy() = 2;
    matrix.build();
    Vector<double> rhs(fem_space.n_dof());
    FEMSpace<double,2>::ElementIterator the_element = fem_space.beginElement();
    FEMSpace<double,2>::ElementIterator end_element = fem_space.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	double volume = the_element->templateElement().volume();
	const QuadratureInfo<2>& quad_info = the_element->findQuadratureInfo(2);
	std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
	int n_quadrature_point = quad_info.n_quadraturePoint();
	std::vector<AFEPack::Point<2> > q_point = the_element->local_to_global(quad_info.quadraturePoint());
	std::vector<std::vector<double> > basis_value = the_element->basis_function_value(q_point);
	std::vector<double> u_h_value = u_h.value(q_point, *the_element);
	std::vector<std::vector<double> > u_h_gradient = u_h.gradient(q_point, *the_element);
	int n_element_dof = the_element->n_dof();
	const std::vector<int>& element_dof = the_element->dof();
	for (l = 0; l < n_quadrature_point; l++) 
	{
	    double Jxw = quad_info.weight(l) * jacobian[l] * volume;
	    for (j = 0; j < n_element_dof; j++) 
	    {
		rhs(element_dof[j]) += Jxw * (u_h_value[l] * basis_value[j][l] / dt
					      - u_h_value[l] * (u_h_gradient[l][0] + u_h_gradient[l][1]) * basis_value[j][l]);
	    }
	}
    }

    BoundaryFunction<double,2> boundary(BoundaryConditionInfo::DIRICHLET, 1, &boundary_value);
    BoundaryConditionAdmin<double,2> boundary_admin(fem_space);
    boundary_admin.add(boundary);
    boundary_admin.apply(matrix, u_h, rhs);


    AMGSolver solver(matrix);
    solver.solve(u_h, rhs);
  
    t += dt;
    g_t = t;
};

void Simulation2D::outputSolution()
{
    std::stringstream result;
    result.setf(std::ios::fixed);
    result.precision(4);
    result << "u_h_" << int((t - 0.25) / dt) << ".dx";
    u_h.writeOpenDXData(result.str());

};



//
// end of file
///////////////////////////////////////////////////////////////////////////////

