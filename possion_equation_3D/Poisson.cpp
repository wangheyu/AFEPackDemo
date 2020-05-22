/**
 * @file   Poisson.cpp
 * @author gary <gary@ellie>
 * @date   Wed Jul 13 11:16:16 2016
 * 
 * @brief  
 * 
 * 
 */

#include <iostream>

#include <AFEPack/EasyMesh.h>
#include <AFEPack/Geometry.h>

#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>

#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>

#include <AFEPack/AMGSolver.h>

#define DIM 3
const double PI = 4.*atan(1.);

double u(const double * p)
{
  //  return sin(PI*p[0]) * sin(2*PI*p[1]);
  return sin(PI*p[0]) * sin(2*PI*p[1]) * sin(PI*p[2]);
};

double f(const double * p)
{
  //  return 5*PI*PI*u(p);
  return 6*PI*PI*u(p);
};


int main(int argc, char* argv[])
{
  Mesh<DIM> mesh;
  mesh.readData(argv[1]);

  TemplateGeometry<DIM>	tet_template_geometry;/// DIM means two dimensional problem
  tet_template_geometry.readData("tetrahedron.tmp_geo");
  CoordTransform<DIM,DIM>	tet_coord_transform;
  tet_coord_transform.readData("tetrahedron.crd_trs");
  TemplateDOF<DIM>	tet_template_dof(tet_template_geometry);
  tet_template_dof.readData("tetrahedron.1.tmp_dof");
  BasisFunctionAdmin<double,DIM,DIM> tet_basis_function(tet_template_dof);
  tet_basis_function.readData("tetrahedron.1.bas_fun");

  std::vector<TemplateElement<double,DIM,DIM> > template_element(1);
  template_element[0].reinit(tet_template_geometry,
			     tet_template_dof,
			     tet_coord_transform,
			     tet_basis_function);


  // TemplateGeometry<DIM>	triangle_template_geometry;/// DIM means two dimensional problem
  // triangle_template_geometry.readData("triangle.tmp_geo");
  // CoordTransform<DIM,DIM>	triangle_coord_transform;
  // triangle_coord_transform.readData("triangle.crd_trs");
  // TemplateDOF<DIM>	triangle_template_dof(triangle_template_geometry);
  // triangle_template_dof.readData("triangle.1.tmp_dof");
  // BasisFunctionAdmin<double,DIM,DIM> triangle_basis_function(triangle_template_dof);
  // triangle_basis_function.readData("triangle.1.bas_fun");

  // std::vector<TemplateElement<double,DIM,DIM> > template_element(1);
  // template_element[0].reinit(triangle_template_geometry,
  // 			     triangle_template_dof,
  // 			     triangle_coord_transform,
  // 			     triangle_basis_function);

  FEMSpace<double,DIM> fem_space(mesh, template_element);
	
  int n_element = mesh.n_geometry(DIM);
  fem_space.element().resize(n_element);
  for (int i = 0;i < n_element;i ++)
    fem_space.element(i).reinit(fem_space,i,0);/// number of template_element

  fem_space.buildElement();
  fem_space.buildDof();
  fem_space.buildDofBoundaryMark(); 

  /// Ax = b
  StiffMatrix<DIM,double> stiff_matrix(fem_space);
  stiff_matrix.algebricAccuracy() = 4;
  stiff_matrix.build();
  
  FEMFunction<double,DIM> solution(fem_space);
  Vector<double> right_hand_side;
  Operator::L2Discretize(&f, fem_space, right_hand_side, 4);

  BoundaryFunction<double,DIM> boundary(BoundaryConditionInfo::DIRICHLET, 1, &u);
  BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(stiff_matrix, solution, right_hand_side);

  AMGSolver solver(stiff_matrix);
  solver.solve(solution, right_hand_side, 1.0e-08, 200);	

  solution.writeOpenDXData("u.dx");
  double error = Functional::L2Error(solution, FunctionFunction<double>(&u), 3);
  std::cerr << "\nL2 error = " << error << std::endl;

  return 0;



}
/**
 * end of file
 * 
 */
