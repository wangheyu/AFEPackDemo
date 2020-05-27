/*
 * @file   possion_equation.cpp
 * @author Heyu Wang <scshw@cslin107.csunix.comp.leeds.ac.uk>
 * @date   Mon May 19 13:19:19 2014
 * 
 * @brief 求解 Navier-Stokes 方程稳定解的算例。配置是 2D 方腔流。
 * 
 * 
 */

#include <iostream>
#include <fstream>
#include <cmath>

#include <AFEPack/AMGSolver.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/EasyMesh.h>

#include <lac/sparse_matrix.h>
#include <lac/sparsity_pattern.h>
#include <lac/sparse_mic.h>
#include <lac/sparse_ilu.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/solver_cg.h>
#include <lac/solver_minres.h>
#include <lac/solver_gmres.h>
#include <lac/precondition.h>
#include <lac/sparse_ilu.h>

#include <lac/trilinos_sparse_matrix.h>
#include <lac/trilinos_block_sparse_matrix.h>
#include <lac/trilinos_vector.h>
#include <lac/trilinos_block_vector.h>
#include <lac/trilinos_precondition.h>

#define PI (4.0*atan(1.0))
/// vis 是动力学粘性系数。在单位正方形上 vis = 0.01 大约是 RE = 100.
#define vis 0.01

//#define N 2

#define DIM 2

double ux(const double *);
double uy(const double *);
double f(const double *);
double up(const double *);
double zfun(const double *);

class StokesPreconditioner
{
private:
    const SparseMatrix<double> *Ax; /**< 预处理矩阵各分块. */
    const SparseMatrix<double> *Ay;
    const SparseMatrix<double> *Q;
    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditionerX;
    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditionerY;
    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditionerQ;

public:
    StokesPreconditioner()
	{};

    ~StokesPreconditioner()
	{};

    /** 
     * 预处理子初始化.
     * 
     * @param _stiff_vx vx 空间的刚度矩阵. 
     * @param _stiff_vy vy 空间的刚度矩阵.
     * @param _mass_p_diag p 空间的质量矩阵的对角元. 
     */
    void initialize (const SparseMatrix<double> &_stiff_vx, 
		     const SparseMatrix<double> &_stiff_vy, 
		     const SparseMatrix<double> &_mass_p_diag) 
	{
	    Ax = &_stiff_vx;
	    Ay = &_stiff_vy;
	    Q = &_mass_p_diag;
	    Amg_preconditionerX = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>(new TrilinosWrappers::PreconditionAMG());
	    Amg_preconditionerX->initialize(*Ax);
	    Amg_preconditionerY = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>(new TrilinosWrappers::PreconditionAMG());
	    Amg_preconditionerY->initialize(*Ay);
	    Amg_preconditionerQ = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>(new TrilinosWrappers::PreconditionAMG());
	    Amg_preconditionerQ->initialize(*Q);
	};
    /** 
     * 实际估值 dst = M^{-1}src. 
     * 
     * @param dst 
     * @param src 
     */
    void vmult (Vector<double> &dst,
		const Vector<double> &src) const;
};

void StokesPreconditioner::vmult (Vector<double> &dst,
				  const Vector<double> &src) const
{
    int n_dof_v = Ax->n();
    int n_dof_p = Q->n();
    Vector<double> d0(n_dof_v);
    Vector<double> d1(n_dof_v);
    Vector<double> s0(n_dof_v);
    Vector<double> s1(n_dof_v);

    Vector<double> d2(n_dof_p);
    Vector<double> s2(n_dof_p);

    for (int i = 0; i < n_dof_v; ++i)
	s0(i) = src(i);
    for (int i = 0; i < n_dof_v; ++i)
	s1(i) = src(n_dof_v + i);
    for (int i = 0; i < n_dof_p; ++i)
	s2(i) = src(2 * n_dof_v + i);

    SolverControl solver_control (100, 1e-3, false, false);
    SolverCG<> solver (solver_control);

    SolverControl solver_controlQ (100, 1e-6, false, false);
    SolverCG<> solverQ (solver_controlQ);
    std::cout << "Stokes precondition applying..." << std::endl;
    solver.solve (*Ax, d0, s0, *Amg_preconditionerX);
    solver.solve (*Ay, d1, s1, *Amg_preconditionerY);
    solverQ.solve (*Q, d2, s2, *Amg_preconditionerQ);
    std::cout << "Stokes precondition completed." << std::endl;

    for (int i = 0; i < n_dof_v; ++i)
	dst(i) = d0(i);
    for (int i = 0; i < n_dof_v; ++i)
	dst(n_dof_v + i) = d1(i);
    for (int i = 0; i < n_dof_p; ++i)
	dst(2 * n_dof_v + i) = d2(i);
};

int main(int argc, char * argv[])
{
    /// 以下和 Stoke 问题一样，也采用了 Taylor-Hood 元（P2 - P1）。 这
    /// 里由于最终非线性迭代没有使用 AMG，因此开启了多核并行加速。这里
    /// 的参数应该和物理核数量一致。
    Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 4);
    HGeometryTree<DIM> h_tree;
    h_tree.readEasyMesh(argv[1]);
    IrregularMesh<DIM> *irregular_mesh_c; /**< 计算网格. 可构建高阶单元.*/
    IrregularMesh<DIM> *irregular_mesh_d; /**< 绘图网格. 构建分片线性.*/

    irregular_mesh_c = new IrregularMesh<DIM>;
    irregular_mesh_c->reinit(h_tree);
    irregular_mesh_c->semiregularize();
    irregular_mesh_c->regularize(false);
    RegularMesh<DIM> &mesh_c = irregular_mesh_c->regularMesh();

    /// 用于绘图的网格更加稠密, 但只有分片线性精度.
    irregular_mesh_d = new IrregularMesh<DIM>(*irregular_mesh_c);
    irregular_mesh_d->globalRefine(1);
    irregular_mesh_d->semiregularize();
    irregular_mesh_d->regularize(false);
    RegularMesh<DIM> &mesh_d = irregular_mesh_d->regularMesh();

    /// 输出两个网格. 用于观察调试.
    mesh_c.writeOpenDXData("D0.dx");
    mesh_d.writeOpenDXData("D1.dx");

    TemplateGeometry<DIM> triangle_template_geometry; /**< 几何模板: 三角形. */
    triangle_template_geometry.readData("triangle.tmp_geo");
    CoordTransform<DIM, DIM> triangle_coord_transform;
    triangle_coord_transform.readData("triangle.crd_trs");

    TemplateDOF<DIM> P1_dof(triangle_template_geometry); /**< P1元自由度. */
    P1_dof.readData("triangle.1.tmp_dof");
    BasisFunctionAdmin<double, DIM, DIM> P1_basis_function(P1_dof); /**< P1元基函数. */
    P1_basis_function.readData("triangle.1.bas_fun");

    TemplateDOF<DIM> P2_dof(triangle_template_geometry); /**< P2元自由度. */
    P2_dof.readData("triangle.2.tmp_dof");
    BasisFunctionAdmin<double, DIM, DIM> P2_basis_function(P2_dof); /**< P2元基函数. */
    P2_basis_function.readData("triangle.2.bas_fun");

    std::vector<TemplateElement<double, DIM, DIM> > P1_element(1);
    P1_element[0].reinit(triangle_template_geometry,
			 P1_dof,
			 triangle_coord_transform,
			 P1_basis_function);

    std::vector<TemplateElement<double, DIM, DIM> > P2_element(1);
    P2_element[0].reinit(triangle_template_geometry,
			 P2_dof,
			 triangle_coord_transform,
			 P2_basis_function);
    
    FEMSpace<double, DIM> fem_space_v(mesh_c, P2_element);
    int n_element = mesh_c.n_geometry(DIM);
    fem_space_v.element().resize(n_element);
    for (int i = 0; i < n_element; ++i)
	fem_space_v.element(i).reinit(fem_space_v, i, 0);
    fem_space_v.buildElement();
    fem_space_v.buildDof();
    fem_space_v.buildDofBoundaryMark();

    FEMSpace<double, DIM> fem_space_p(mesh_c, P1_element);
    fem_space_p.element().resize(n_element);
    for (int i = 0; i < n_element; ++i)
	fem_space_p.element(i).reinit(fem_space_p, i, 0);
    fem_space_p.buildElement();
    fem_space_p.buildDof();
    fem_space_p.buildDofBoundaryMark();

    FEMSpace<double, DIM> fem_space_output(mesh_d, P1_element);
	
    n_element = mesh_d.n_geometry(DIM);
    fem_space_output.element().resize(n_element);
    for (int i = 0; i < n_element; ++i)
    	fem_space_output.element(i).reinit(fem_space_output, i, 0);

    fem_space_output.buildElement();
    fem_space_output.buildDof();
    fem_space_output.buildDofBoundaryMark();
    /// 空间准备完毕.

    //
    // 统计非零元个数，用以建立稀疏矩阵结构.
    //
   
    int n_v = fem_space_v.n_dof();
    int n_p = fem_space_p.n_dof();
    int total_n_dof = 2 * n_v + n_p;
    std::vector<unsigned int> max_couple(total_n_dof);

    int n_dof_P1 = P1_element[0].n_dof();
    int n_dof_P2 = P2_element[0].n_dof();

    FEMSpace<double, DIM>::ElementIterator the_element = fem_space_v.beginElement();
    FEMSpace<double, DIM>::ElementIterator end_element = fem_space_v.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof = the_element->dof();
	for (int i = 0; i < n_dof_P2; ++i)
	{
	    max_couple[element_dof[i]] += (2 * n_dof_P2 + n_dof_P1);
	    max_couple[element_dof[i] + n_v] += (2 * n_dof_P2 + n_dof_P1);
	}
    }

    the_element = fem_space_p.beginElement();
    end_element = fem_space_p.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof = the_element->dof();
	for (int i = 0; i < n_dof_P1; ++i)
	    max_couple[element_dof[i] + 2 * n_v] += (2 * n_dof_P2 + n_dof_P1);
    }
    SparsityPattern sp_system_matrix(total_n_dof, max_couple);
  
    the_element = fem_space_v.beginElement();
    end_element = fem_space_v.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof_v = the_element->dof();
	Element<double, DIM>& element_p = fem_space_p.element(the_element->index());
	const std::vector<int>& element_dof_p = element_p.dof();
	for (int i = 0; i < n_dof_P2; ++i)
	{
	    for (int j = 0; j < n_dof_P2; ++j)
	    {
		sp_system_matrix.add(element_dof_v[i], element_dof_v[j]);
		sp_system_matrix.add(element_dof_v[i] + n_v, element_dof_v[j] + n_v);
	    }
	    for (int j = 0; j < n_dof_P1; ++j)
	    {
		sp_system_matrix.add(element_dof_v[i], element_dof_p[j] + 2 * n_v);
		sp_system_matrix.add(element_dof_v[i] + n_v, element_dof_p[j] + 2 * n_v);
	    }
	}
	for (int i = 0; i < n_dof_P1; ++i)
	    for (int j = 0; j < n_dof_P2; ++j)
	    {
		sp_system_matrix.add(element_dof_p[i] + 2 * n_v, element_dof_v[j]);
		sp_system_matrix.add(element_dof_p[i] + 2 * n_v, element_dof_v[j] + n_v);
	    }
    }

    sp_system_matrix.compress();
    SparseMatrix<double> system_matrix(sp_system_matrix);


    //
    // 稀疏矩阵模板创建完毕
    // 接下来我们要计算系数矩阵中各项的值
    //

    
    the_element = fem_space_v.beginElement();
    for (; the_element != end_element; ++the_element) 
    {
	double volume = the_element->templateElement().volume();
	const QuadratureInfo<DIM>& quad_info = the_element->findQuadratureInfo(4);
	std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
	int n_quadrature_point = quad_info.n_quadraturePoint();
	std::vector<AFEPack::Point<DIM> > q_point = the_element->local_to_global(quad_info.quadraturePoint());
	std::vector<std::vector<std::vector<double> > > basis_gradient_v = the_element->basis_function_gradient(q_point);
	const std::vector<int>& element_dof_v = the_element->dof();
	Element<double, DIM>& element_p = fem_space_p.element(the_element->index());
	const std::vector<int>& element_dof_p = element_p.dof();
	std::vector<std::vector<double> > basis_function_value_p = element_p.basis_function_value(q_point);
	/// NS 的矩阵比 Stokes 的矩阵要复杂，而且新增的对流项严重影响
	/// 了矩阵的性质。
	for (int l = 0; l < n_quadrature_point; ++l)
	{
	    double Jxw = quad_info.weight(l) * jacobian[l] * volume;
	    for (int i = 0; i < n_dof_P2; ++i)
	    {
		for (int j = 0; j < n_dof_P2; ++j)
		{
		    double cont = vis * Jxw * innerProduct(basis_gradient_v[i][l], basis_gradient_v[j][l]);
		    system_matrix.add(element_dof_v[i], element_dof_v[j], cont);
		    system_matrix.add(element_dof_v[i] + n_v, element_dof_v[j] + n_v, cont);
		}
		for (int j = 0; j < n_dof_P1; ++j)
		{
		    double cont = Jxw * basis_function_value_p[j][l] * basis_gradient_v[i][l][0];
		    system_matrix.add(element_dof_v[i], element_dof_p[j] + 2 * n_v, -cont);
		    cont = Jxw * basis_function_value_p[j][l] * basis_gradient_v[i][l][1];
		    system_matrix.add(element_dof_v[i] + n_v, element_dof_p[j] + 2 * n_v, -cont);
		}

	    }
	    for (int i = 0; i < n_dof_P1; ++i)
		for (int j = 0; j < n_dof_P2; ++j)
		{
		    double cont = Jxw * basis_function_value_p[i][l] * basis_gradient_v[j][l][0];
		    system_matrix.add(element_dof_p[i] + 2 * n_v, element_dof_v[j], -cont);
		    cont = Jxw * basis_function_value_p[i][l] * basis_gradient_v[j][l][1];
		    system_matrix.add(element_dof_p[i] + 2 * n_v, element_dof_v[j] + n_v, -cont);
		}
	}
    }

    //
    // 接下来处理边界条件，改变右端项，对块矩阵施加边界条件比较复杂。
    // 这里同样利用了矩阵的对称结构。尽管矩阵本身没有对称性，但非零元
    // 结构是对称的。
    //
    
    Vector<double> solution(total_n_dof);
    FEMFunction<double, DIM> vx(fem_space_v);
    FEMFunction<double, DIM> vy(fem_space_v);
    FEMFunction<double, DIM> p(fem_space_p);
    Vector<double> rhs(total_n_dof);
    Vector<double> rhs_vx(n_v);
    Vector<double> rhs_vy(n_v);
    Vector<double> rhs_p(n_p);

    Operator::L2Discretize(&zfun, fem_space_v, rhs_vx, 1);
    Operator::L2Discretize(&zfun, fem_space_v, rhs_vy, 1);
    Operator::L2Discretize(&zfun, fem_space_p, rhs_p, 1);
    for (int i = 0; i < n_v; i++)
	rhs(i) = rhs_vx(i);
    for (int i = n_v; i < 2 * n_v; i++)
	rhs(i) = rhs_vy(i - n_v);
    for (int i = 2 * n_v; i < total_n_dof; i++)
	rhs(i) = rhs_p(i - 2 * n_v);
    for (int i = 0; i < n_v; i++)
    {
    	FEMSpace<double, DIM>::dof_info_t dof = fem_space_v.dofInfo(i);
    	if (dof.boundary_mark == 1)
    	{
    	    SparseMatrix<double>::iterator row_iterator = system_matrix.begin(i);
    	    SparseMatrix<double>::iterator row_end = system_matrix.end(i);
    	    double diag = row_iterator->value();
    	    double bnd_value = zfun(dof.interp_point);
            rhs(i) = diag * bnd_value;
    	    for (++row_iterator; row_iterator != row_end; ++row_iterator)
            {
            	row_iterator->value() = 0.0;
    		int k = row_iterator->column();
                SparseMatrix<double>::iterator col_iterator = system_matrix.begin(k);   
                SparseMatrix<double>::iterator col_end = system_matrix.end(k);   
    	    	for (++col_iterator; col_iterator != col_end; ++col_iterator)
		    if (col_iterator->column() == i)
			break;
    		if (col_iterator == col_end)
    		{
		    std::cerr << "Error!" << std::endl;
		    exit(-1);
    		}
    		rhs(k) -= col_iterator->value() * bnd_value; 
    		col_iterator->value() = 0.0;	
            }
    	    row_iterator = system_matrix.begin(i + n_v);
    	    row_end = system_matrix.end(i + n_v);
    	    diag = row_iterator->value();
    	    bnd_value = zfun(dof.interp_point); 
            rhs(i + n_v) = diag * bnd_value;
    	    for (++row_iterator; row_iterator != row_end; ++row_iterator)
            {
            	row_iterator->value() = 0.0;
    	    	int k = row_iterator->column();
                SparseMatrix<double>::iterator col_iterator = system_matrix.begin(k);   
                SparseMatrix<double>::iterator col_end = system_matrix.end(k);   
    	    	for (++col_iterator; col_iterator != col_end; ++col_iterator)
		{
		    if (col_iterator->column() == i + n_v)
			break;
		}
    	    	if (col_iterator == col_end)
    	    	{
		    std::cerr << "Error!" << std::endl;
		    exit(-1);
    	    	}
    	    	rhs(k) -= col_iterator->value() * bnd_value; 
    	    	col_iterator->value() = 0.0;	
            }  
    	}	
    	if (dof.boundary_mark == 2)
    	{
    	    SparseMatrix<double>::iterator row_iterator = system_matrix.begin(i);
    	    SparseMatrix<double>::iterator row_end = system_matrix.end(i);
    	    double diag = row_iterator->value();
    	    double bnd_value = ux(dof.interp_point);
            rhs(i) = diag * bnd_value;
    	    for (++row_iterator; row_iterator != row_end; ++row_iterator)
            {
            	row_iterator->value() = 0.0;
    		int k = row_iterator->column();
                SparseMatrix<double>::iterator col_iterator = system_matrix.begin(k);   
                SparseMatrix<double>::iterator col_end = system_matrix.end(k);   
    	    	for (++col_iterator; col_iterator != col_end; ++col_iterator)
		    if (col_iterator->column() == i)
			break;
    		if (col_iterator == col_end)
    		{
		    std::cerr << "Error!" << std::endl;
		    exit(-1);
    		}
    		rhs(k) -= col_iterator->value() * bnd_value; 
    		col_iterator->value() = 0.0;	
            }  
    	    row_iterator = system_matrix.begin(i + n_v);
    	    row_end = system_matrix.end(i + n_v);
    	    diag = row_iterator->value();
    	    bnd_value = uy(dof.interp_point); 
            rhs(i + n_v) = diag * bnd_value;
    	    for (++row_iterator; row_iterator != row_end; ++row_iterator)
            {
            	row_iterator->value() = 0.0;
    		int k = row_iterator->column();
                SparseMatrix<double>::iterator col_iterator = system_matrix.begin(k);   
                SparseMatrix<double>::iterator col_end = system_matrix.end(k);   
    	    	for (++col_iterator; col_iterator != col_end; ++col_iterator)
		    if (col_iterator->column() == i + n_v)
			break;
    		if (col_iterator == col_end)
    		{
		    std::cerr << "Error!" << std::endl;
		    exit(-1);
    		}
    		rhs(k) -= col_iterator->value() * bnd_value; 
    		col_iterator->value() = 0.0;	
            }  
    	}	
    }		
	
    /// 到此为止Ax=b的问题已经建立完毕。

    std::vector<unsigned int> max_couple_vv(n_v);
    std::vector<unsigned int> max_couple_vp(n_v);
    
    for (int i = 0; i < n_v; i++)
    {
    	SparsityPattern::iterator row_start = sp_system_matrix.begin(i);
    	SparsityPattern::iterator row_end = sp_system_matrix.end(i);
    	for (; row_start != row_end; ++row_start)
    	{
	    int col = row_start->column();
    	    if (col < n_v)
		max_couple_vv[i]++;
	    else if (col >= 2 * n_v && col < total_n_dof)
		max_couple_vp[i]++;
     	}
    }
    SparsityPattern sp_vv(n_v, max_couple_vv);
    SparsityPattern sp_vp(n_v, n_p, max_couple_vp);
    for (int i = 0; i < n_v; i++)
    {
    	SparsityPattern::iterator row_start = sp_system_matrix.begin(i);
    	SparsityPattern::iterator row_end = sp_system_matrix.end(i);
	std::vector<unsigned int> vv_cols(max_couple_vv[i]);
	std::vector<unsigned int>::iterator vv_cols_iterator = vv_cols.begin();
	std::vector<unsigned int>::iterator vv_cols_start = vv_cols_iterator;
	std::vector<unsigned int> vp_cols(max_couple_vp[i]);
	std::vector<unsigned int>::iterator vp_cols_iterator = vp_cols.begin();
	std::vector<unsigned int>::iterator vp_cols_start = vp_cols_iterator;
    	for (; row_start != row_end; ++row_start)
    	{
	    int col = row_start->column();
    	    if (col < n_v)
	    {
		*vv_cols_iterator = col;
		++vv_cols_iterator;
	    }
	    else if (col >= 2 * n_v && col < total_n_dof)
	    {
		*vp_cols_iterator = col - 2 * n_v;
		++vp_cols_iterator;
	    }
	}
	sp_vv.add_entries(i, vv_cols_start, vv_cols_iterator);
	sp_vp.add_entries(i, vp_cols_start, vp_cols_iterator);
    }
    sp_vv.compress();
    sp_vp.compress();
    SparseMatrix<double> vxvx(sp_vv);
    SparseMatrix<double> vxp(sp_vp);
    SparseMatrix<double> vyvy(sp_vv);
    SparseMatrix<double> vyp(sp_vp);

    for (int i = 0; i < n_v; i++)
    {
    	SparseMatrix<double>::iterator row_start = system_matrix.begin(i);
    	SparseMatrix<double>::iterator row_end = system_matrix.end(i);
	std::vector<unsigned int> vv_cols(max_couple_vv[i]);
	std::vector<unsigned int>::iterator vv_cols_iterator = vv_cols.begin();
	std::vector<unsigned int>::iterator vv_cols_start = vv_cols_iterator;
	std::vector<double> vv_vals(max_couple_vv[i]);
	std::vector<double>::iterator vv_vals_iterator = vv_vals.begin();
	std::vector<double>::iterator vv_vals_start = vv_vals_iterator;
	std::vector<unsigned int> vp_cols(max_couple_vp[i]);
	std::vector<unsigned int>::iterator vp_cols_iterator = vp_cols.begin();
	std::vector<unsigned int>::iterator vp_cols_start = vp_cols_iterator;
	std::vector<double> vp_vals(max_couple_vp[i]);
	std::vector<double>::iterator vp_vals_iterator = vp_vals.begin();
	std::vector<double>::iterator vp_vals_start = vp_vals_iterator;
    	for (; row_start != row_end; ++row_start)
    	{
	    int col = row_start->column();
	    double val = row_start->value();
    	    if (col < n_v)
	    {
		*vv_cols_iterator = col;
		++vv_cols_iterator;
		*vv_vals_iterator = val;
		++vv_vals_iterator;
	    }
	    else if (col >= 2 * n_v && col < total_n_dof)
	    {
		*vp_cols_iterator = col - 2 * n_v;
		++vp_cols_iterator;
		*vp_vals_iterator = val;
		++vp_vals_iterator;

	    }
	}
	vxvx.add(i, vv_cols, vv_vals);
	vxp.add(i, vp_cols, vp_vals);
    }
    for (int i = n_v; i < 2 * n_v; i++)
    {
    	SparseMatrix<double>::iterator row_start = system_matrix.begin(i);
    	SparseMatrix<double>::iterator row_end = system_matrix.end(i);
	std::vector<unsigned int> vv_cols(max_couple_vv[i - n_v]);
	std::vector<unsigned int>::iterator vv_cols_iterator = vv_cols.begin();
	std::vector<unsigned int>::iterator vv_cols_start = vv_cols_iterator;
	std::vector<double> vv_vals(max_couple_vv[i - n_v]);
	std::vector<double>::iterator vv_vals_iterator = vv_vals.begin();
	std::vector<double>::iterator vv_vals_start = vv_vals_iterator;
	std::vector<unsigned int> vp_cols(max_couple_vp[i - n_v]);
	std::vector<unsigned int>::iterator vp_cols_iterator = vp_cols.begin();
	std::vector<unsigned int>::iterator vp_cols_start = vp_cols_iterator;
	std::vector<double> vp_vals(max_couple_vp[i - n_v]);
	std::vector<double>::iterator vp_vals_iterator = vp_vals.begin();
	std::vector<double>::iterator vp_vals_start = vp_vals_iterator;
    	for (; row_start != row_end; ++row_start)
    	{
	    int col = row_start->column();
	    double val = row_start->value();
    	    if (col >= n_v && col < 2 * n_v)
	    {
		*vv_cols_iterator = col - n_v;
		++vv_cols_iterator;
		*vv_vals_iterator = val;
		++vv_vals_iterator;
	    }
	    else if (col >= 2 * n_v && col < total_n_dof)
	    {
		*vp_cols_iterator = col - 2 * n_v;
		++vp_cols_iterator;
		*vp_vals_iterator = val;
		++vp_vals_iterator;
	    }
	}
	vyvy.add(i - n_v, vv_cols, vv_vals);
	vyp.add(i - n_v, vp_cols, vp_vals);
    }
    /*
      std::cout << "Vx = [" << std::endl;
      for (int i = 0; i < n_v; i++)
      {
      for (int j = 0; j < n_v; j++)
      std::cout << vxvx.el(i, j) << " ";
      std::cout << std::endl;
      }
      std::cout << "];";

      std::cout << "Px = [" << std::endl;
      for (int i = 0; i < n_v; i++)
      {
      for (int j = 0; j < n_p; j++)
      std::cout << vxp.el(i, j) << " ";
      std::cout << std::endl;
      }
      std::cout << "];";

      std::cout << "Vy = [" << std::endl;
      for (int i = 0; i < n_v; i++)
      {
      for (int j = 0; j < n_v; j++)
      std::cout << vxvx.el(i, j) << " ";
      std::cout << std::endl;
      }
      std::cout << "];";

      std::cout << "Py = [" << std::endl;
      for (int i = 0; i < n_v; i++)
      {
      for (int j = 0; j < n_p; j++)
      std::cout << vyp.el(i, j) << " ";
      std::cout << std::endl;
      }
      std::cout << "];";
    */
    std::vector<unsigned int> max_couple_pp(n_p);
    
    the_element = fem_space_p.beginElement();
    end_element = fem_space_p.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof = the_element->dof();
	for (int i = 0; i < n_dof_P1; ++i)
	    max_couple_pp[element_dof[i]] += n_dof_P1;
    }
    SparsityPattern sp_pp(n_p, max_couple_pp);
  
    the_element = fem_space_p.beginElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof_p = the_element->dof();
	for (int i = 0; i < n_dof_P1; ++i)
	    for (int j = 0; j < n_dof_P1; ++j)
		sp_pp.add(element_dof_p[i], element_dof_p[j]);
    }
    sp_pp.compress();
    SparseMatrix<double> mass_p(sp_pp);

    the_element = fem_space_p.beginElement();
    for (; the_element != end_element; ++the_element) 
    {
	double volume = the_element->templateElement().volume();
	const QuadratureInfo<DIM>& quad_info = the_element->findQuadratureInfo(2);
	std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
	int n_quadrature_point = quad_info.n_quadraturePoint();
	std::vector<AFEPack::Point<DIM> > q_point = the_element->local_to_global(quad_info.quadraturePoint());
	std::vector<std::vector<double> > basis_function_value_p = the_element->basis_function_value(q_point);
	const std::vector<int>& element_dof_p = the_element->dof();
	for (int l = 0; l < n_quadrature_point; ++l)
	{
	    double Jxw = quad_info.weight(l) * jacobian[l] * volume;
	    for (int i = 0; i < n_dof_P1; ++i)
	    {
		//for (int j = 0; j < n_dof_P1; ++j)
		//{
		double cont = Jxw * basis_function_value_p[i][l] * basis_function_value_p[i][l];
		mass_p.add(element_dof_p[i], element_dof_p[i], cont);
		//}
	    }
	}
    }


/// 这里首先求解一个 Stokes 问题的解，作为非线性迭代的初值。
    StokesPreconditioner preconditioner;
    preconditioner.initialize(vxvx, vyvy, mass_p);
    
    SolverControl solver_control (1000000, 1e-12, false);
//  求解器的选择, Stokes系统是对称不定的, 不能采用cg, 如果使用GMRES, 效率比较低.
//    SolverGMRES<Vector<double> > gmres(solver_control);
//    gmres.solve (system_matrix, solution, rhs, PreconditionIdentity());	

	
//  在处理NS系统中的对称不定问题时，我们需要用gmres
//  同时在这里，我们需要加入一步循环，即迭代过程	
	
//  更好的求解器是专门处理对称不定系统的MinRes, 效率相对GMRES好很多.     
    SolverMinRes<Vector<double> > minres(solver_control);
//  然而不做任何预处理仍然是低效的.    
//    minres.solve (system_matrix, solution, rhs, PreconditionIdentity());
//  在最基本的情况下, 至少也应该做一个MIC预处理. 由于系统矩阵不定且右下角有零块, 因此
//  必须对预处理阵的对角元加强以确保MIC能够完成. 对角强化参数是一个经验参数.     
    SparseMIC<double> mic;
    SparseMIC<double>::AdditionalData ad;
    ad.strengthen_diagonal = 0.5;
    mic.initialize(system_matrix, ad);
    std::cout << "Stokes equations solving..." << std::endl;
    minres.solve (system_matrix, solution, rhs, preconditioner);	
//    minres.solve (system_matrix, solution, rhs, mic);	
    std::cout << "Stokes equations solved..." << std::endl;
    for (int i = 0; i < n_v; i++)
	vx(i) = solution(i);
    for (int i = n_v; i < 2 * n_v; i++)
	vy(i - n_v) = solution(i);
    for (int i = 2 * n_v; i < total_n_dof; i++)
	p(i - 2 * n_v) = solution(i);

    
    // FEMFunction<double, DIM> stokes_solution_output(fem_space_output);
    // Operator::L2Interpolate(vx, stokes_solution_output);
    // stokes_solution_output.writeOpenDXData("ux.dx");
    // Operator::L2Interpolate(vy, stokes_solution_output);
    // stokes_solution_output.writeOpenDXData("uy.dx");
    // Operator::L2Interpolate(p, stokes_solution_output);
    // stokes_solution_output.writeOpenDXData("p.dx");
    // std::cout << "Stokes solution optputed." << std::endl;
    // getchar();
    
    // the error
    
    //double error1 = Functional::L2Error(vx, FunctionFunction<double>(&ux), 10);
    // std::cout << "L2 xError = " << error1 << std::endl;


    // - -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - //
    //
    // we now begin to solve the Navier_Stokes equation here, with the
    // u0 = soluation, espeacially v = 1;
    //
    //
    // 统计非零元个数，用以建立稀疏矩阵结构.
    //
    // - -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - //
    
    n_v = fem_space_v.n_dof();
    n_p = fem_space_p.n_dof();
    total_n_dof = 2 * n_v + n_p;
    std::vector<unsigned int> max_couples(total_n_dof);

    n_dof_P1 = P1_element[0].n_dof();
    n_dof_P2 = P2_element[0].n_dof();

    the_element = fem_space_v.beginElement();
    end_element = fem_space_v.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>&	element_dof = the_element->dof();
	for (int i = 0; i < n_dof_P2; ++i)
	{
	    max_couples[element_dof[i]] += (2 * n_dof_P2 + n_dof_P1);
	    max_couples[element_dof[i] + n_v] += (2 * n_dof_P2 + n_dof_P1);
	}
    }

    the_element = fem_space_p.beginElement();
    end_element = fem_space_p.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof = the_element->dof();
	for (int i = 0; i < n_dof_P1; ++i)
	    max_couples[element_dof[i] + 2 * n_v] += (2 * n_dof_P2 + n_dof_P1);
    }
    SparsityPattern sp_system_matrixs(total_n_dof, max_couples);
  
    the_element = fem_space_v.beginElement();
    end_element = fem_space_v.endElement();
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof_v = the_element->dof();
	Element<double, DIM>& element_p = fem_space_p.element(the_element->index());
	const std::vector<int>& element_dof_p = element_p.dof();
	for (int i = 0; i < n_dof_P2; ++i)
	{
	    for (int j = 0; j < n_dof_P2; ++j)
	    {
		sp_system_matrixs.add(element_dof_v[i], element_dof_v[j]);
		sp_system_matrixs.add(element_dof_v[i], element_dof_v[j] + n_v);
		sp_system_matrixs.add(element_dof_v[i] + n_v, element_dof_v[j] + n_v);
		sp_system_matrixs.add(element_dof_v[i] + n_v, element_dof_v[j]);
		// * Wxy and Wyx is non-zero. // 
	    }
	    for (int j = 0; j < n_dof_P1; ++j)
	    {
		sp_system_matrixs.add(element_dof_v[i], element_dof_p[j] + 2 * n_v);
		sp_system_matrixs.add(element_dof_v[i] + n_v, element_dof_p[j] + 2 * n_v);
	    }
	}
	for (int i = 0; i < n_dof_P1; ++i)
	    for (int j = 0; j < n_dof_P2; ++j)
	    {
		sp_system_matrixs.add(element_dof_p[i] + 2 * n_v, element_dof_v[j]);
		sp_system_matrixs.add(element_dof_p[i] + 2 * n_v, element_dof_v[j] + n_v);
	    }
    }

    sp_system_matrixs.compress();
    SparseMatrix<double> system_matrixs(sp_system_matrixs);
    system_matrixs.reinit(sp_system_matrixs);

    /// 构建非线性 Newton 迭代。
    int step = 0;
    double error = 1e-12;
    double res = 1;
    double h = 0.1;
    while (res > error)
    {


	system_matrixs.reinit(sp_system_matrixs);
	
	the_element = fem_space_v.beginElement();
	for (; the_element != end_element; ++the_element) 
	{
	    double volume = the_element->templateElement().volume();
	    const QuadratureInfo<DIM>& quad_info = the_element->findQuadratureInfo(6);
	    std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
	    int n_quadrature_point = quad_info.n_quadraturePoint();
	    std::vector<AFEPack::Point<DIM> > q_point = the_element->local_to_global(quad_info.quadraturePoint());
	    std::vector<std::vector<std::vector<double> > > basis_gradient_v = the_element->basis_function_gradient(q_point);
	    // +
	    std::vector<std::vector<double> > basis_function_value_v = the_element->basis_function_value(q_point);
	
	    const std::vector<int>& element_dof_v = the_element->dof();
	    Element<double, DIM>& element_p = fem_space_p.element(the_element->index());
	    const std::vector<int>& element_dof_p = element_p.dof();
	    std::vector<std::vector<double> > basis_function_value_p = element_p.basis_function_value(q_point);
	    for (int l = 0; l < n_quadrature_point; ++l)
	    {
		double Jxw = quad_info.weight(l) * jacobian[l] * volume;
		for (int i = 0; i < n_dof_P2; ++i)
		{
		    for (int j = 0; j < n_dof_P2; ++j)
		    {
			double cont = vis * Jxw * innerProduct(basis_gradient_v[i][l], basis_gradient_v[j][l]);
			system_matrixs.add(element_dof_v[i], element_dof_v[j], cont);
			system_matrixs.add(element_dof_v[i] + n_v, element_dof_v[j] + n_v, cont);

			//we add N
			double temp;
			temp = Jxw * ( vx.value(q_point, *the_element)[l] * basis_gradient_v[j][l][0] + vy.value(q_point, *the_element)[l] * basis_gradient_v[j][l][1]) * basis_function_value_v[i][l];
			system_matrixs.add(element_dof_v[i], element_dof_v[j], temp);
			temp = Jxw * ( vx.value(q_point, *the_element)[l] * basis_gradient_v[j][l][0] + vy.value(q_point, *the_element)[l] * basis_gradient_v[j][l][1]) * basis_function_value_v[i][l];
			system_matrixs.add(element_dof_v[i] + n_v, element_dof_v[j] + n_v, temp);

			//then we add W
			temp = Jxw * basis_function_value_v[i][l] * basis_function_value_v[j][l];
			std::vector<std::vector<double> > vx_gradient = vx.gradient(q_point, *the_element);
			std::vector<std::vector<double> > vy_gradient = vy.gradient(q_point, *the_element);
			system_matrixs.add(element_dof_v[i], element_dof_v[j], vx_gradient[l][0] * temp);
			system_matrixs.add(element_dof_v[i], element_dof_v[j] + n_v, vx_gradient[l][1] * temp);
			system_matrixs.add(element_dof_v[i] + n_v, element_dof_v[j], vy_gradient[l][0] * temp);
			system_matrixs.add(element_dof_v[i] + n_v, element_dof_v[j] + n_v, vy_gradient[l][1] * temp);
                    		    
		    }
		    for (int j = 0; j < n_dof_P1; ++j)
		    {
			double cont = Jxw * basis_function_value_p[j][l] * basis_gradient_v[i][l][0];
			system_matrixs.add(element_dof_v[i], element_dof_p[j] + 2 * n_v, -cont);
			cont = Jxw * basis_function_value_p[j][l] * basis_gradient_v[i][l][1];
			system_matrixs.add(element_dof_v[i] + n_v, element_dof_p[j] + 2 * n_v, -cont);
		    }

		}
		for (int i = 0; i < n_dof_P1; ++i)
		    for (int j = 0; j < n_dof_P2; ++j)
		    {
			double cont = Jxw * basis_function_value_p[i][l] * basis_gradient_v[j][l][0];
			system_matrixs.add(element_dof_p[i] + 2 * n_v, element_dof_v[j], -cont);
			cont = Jxw * basis_function_value_p[i][l] * basis_gradient_v[j][l][1];
			system_matrixs.add(element_dof_p[i] + 2 * n_v, element_dof_v[j] + n_v, -cont);
		    }
	    }
	}

	Vector<double> solutions(total_n_dof);
	FEMFunction<double, DIM> vxs(fem_space_v);
	FEMFunction<double, DIM> vys(fem_space_v);
	FEMFunction<double, DIM> ps(fem_space_p);
	Vector<double> rhss(total_n_dof);
	Vector<double> rhs_vxs(n_v);
	Vector<double> rhs_vys(n_v);
	Vector<double> rhs_ps(n_p);

//	Operator::L2Discretize(&zfun, fem_space_v, rhs_vxs, 1);
//	Operator::L2Discretize(&zfun, fem_space_v, rhs_vys, 1);
//	Operator::L2Discretize(&zfun, fem_space_p, rhs_ps, 1);

	//the J(A)dx = A(u_n)
	// here we get A(u_n) and deal the boundary 
	// system_matrixs.vmult(rhss, solution);
	//for (int i = 0; i < 2 * n_v + n_p; i++)
	//{
//	rhss(i) = -rhss(i);
	//   }

	the_element = fem_space_v.beginElement();
	end_element = fem_space_v.endElement();
	for (;the_element!=end_element;++the_element)
	{
	    double volume = the_element->templateElement().volume();
	    const QuadratureInfo<DIM>& quad_info = the_element->findQuadratureInfo(6);
	    std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
	    int n_quadrature_point = quad_info.n_quadraturePoint();
	    std::vector<AFEPack::Point<DIM> > q_point = the_element->local_to_global(quad_info.quadraturePoint());
	    const std::vector<int>& element_dof_v = the_element->dof();
	    Element<double, DIM>& element_p = fem_space_p.element(the_element->index());
	    const std::vector<int>& element_dof_p = element_p.dof();
	    std::vector<std::vector<double> > basis_function_value_v = the_element->basis_function_value(q_point);
	    std::vector<std::vector<double> > basis_function_value_p = element_p.basis_function_value(q_point);
	    std::vector<std::vector<std::vector<double> > > basis_gradient_v = the_element->basis_function_gradient(q_point);
	    std::vector<double> vx_value = vx.value(q_point, *the_element);
	    std::vector<double> vy_value = vy.value(q_point, *the_element);
	    std::vector<double> p_value = p.value(q_point, element_p);
	    std::vector<std::vector<double> > vx_gradient = vx.gradient(q_point, *the_element);
	    std::vector<std::vector<double> > vy_gradient = vy.gradient(q_point, *the_element);
	    // here we change the A(u_n) 
	    for (int l = 0; l < n_quadrature_point; ++l)
	    {
		double Jxw = quad_info.weight(l) * jacobian[l] * volume;
		for(int i = 0; i < n_dof_P2; i++)
		{
		    double cont = Jxw * ((-vx_value[l] * vx_gradient[l][0] - vy_value[l] * vx_gradient[l][1]) * basis_function_value_v[i][l] - vis * innerProduct(vx_gradient[l], basis_gradient_v[i][l]) + p_value[l] * basis_gradient_v[i][l][0]);
		    rhss(element_dof_v[i]) += cont;
		    cont = Jxw * ((-vx_value[l] * vy_gradient[l][0] - vy_value[l] * vy_gradient[l][1]) * basis_function_value_v[i][l] - vis * innerProduct(vy_gradient[l], basis_gradient_v[i][l]) + p_value[l] * basis_gradient_v[i][l][1]);
		    rhss(element_dof_v[i] + n_v) += cont;
		}
		for(int i = 0; i < n_dof_P1; i++)
		{
		    double cont = Jxw * basis_function_value_p[i][l] * (vx_gradient[l][0] + vy_gradient[l][1]);
		    rhss(element_dof_p[i] + 2 * n_v) += cont;
		}
	    }
	}
	
	/// Newton 迭代过程中也需要施加边界条件，因为边界条件实际上是
	/// 精确的，所以只要确保边界上增量为零即可。
	for (int i = 0; i < n_v; i++)
	{
	    FEMSpace<double, DIM>::dof_info_t dof = fem_space_v.dofInfo(i);
	    if (dof.boundary_mark == 1)
	    {
		SparseMatrix<double>::iterator row_iterator = system_matrixs.begin(i);
		SparseMatrix<double>::iterator row_end = system_matrixs.end(i);
		double diag = row_iterator->value();
		double bnd_value = zfun(dof.interp_point);
		rhss(i) = diag * bnd_value;
		for (++row_iterator; row_iterator != row_end; ++row_iterator)
		{
		    row_iterator->value() = 0.0;
		    int k = row_iterator->column();
		    SparseMatrix<double>::iterator col_iterator = system_matrixs.begin(k);   
		    SparseMatrix<double>::iterator col_end = system_matrixs.end(k);   
		    for (++col_iterator; col_iterator != col_end; ++col_iterator)
    			if (col_iterator->column() == i)
    			    break;
		    if (col_iterator == col_end)
		    {
    			std::cerr << "Error!" << std::endl;
    			exit(-1);
		    }
		    rhss(k) -= col_iterator->value() * bnd_value; 
		    col_iterator->value() = 0.0;	
		}
		row_iterator = system_matrixs.begin(i + n_v);
		row_end = system_matrixs.end(i + n_v);
		diag = row_iterator->value();
		bnd_value = zfun(dof.interp_point); 
		rhss(i + n_v) = diag * bnd_value;
		for (++row_iterator; row_iterator != row_end; ++row_iterator)
		{
		    row_iterator->value() = 0.0;
		    int k = row_iterator->column();
		    SparseMatrix<double>::iterator col_iterator = system_matrixs.begin(k);   
		    SparseMatrix<double>::iterator col_end = system_matrixs.end(k);   
		    for (++col_iterator; col_iterator != col_end; ++col_iterator)
		    {
    	    		if (col_iterator->column() == i + n_v)
    	    		    break;
		    }
		    if (col_iterator == col_end)
		    {
    	    		std::cerr << "Error!" << std::endl;
    	    		exit(-1);
		    }
		    rhs(k) -= col_iterator->value() * bnd_value; 
		    col_iterator->value() = 0.0;	
		}  
	    }
	    if (dof.boundary_mark == 2)
	    {
		SparseMatrix<double>::iterator row_iterator = system_matrixs.begin(i);
		SparseMatrix<double>::iterator row_end = system_matrixs.end(i);
		double diag = row_iterator->value();
		double bnd_value = zfun(dof.interp_point);
		rhss(i) = diag * bnd_value;
		for (++row_iterator; row_iterator != row_end; ++row_iterator)
		{
		    row_iterator->value() = 0.0;
		    int k = row_iterator->column();
		    SparseMatrix<double>::iterator col_iterator = system_matrixs.begin(k);   
		    SparseMatrix<double>::iterator col_end = system_matrixs.end(k);   
		    for (++col_iterator; col_iterator != col_end; ++col_iterator)
    			if (col_iterator->column() == i)
    			    break;
		    if (col_iterator == col_end)
		    {
    			std::cerr << "Error!" << std::endl;
    			exit(-1);
		    }
		    rhss(k) -= col_iterator->value() * bnd_value; 
		    col_iterator->value() = 0.0;	
		}  
		row_iterator = system_matrixs.begin(i + n_v);
		row_end = system_matrixs.end(i + n_v);
		diag = row_iterator->value();
		bnd_value = zfun(dof.interp_point); 
		rhss(i + n_v) = diag * bnd_value;
		for (++row_iterator; row_iterator != row_end; ++row_iterator)
		{
		    row_iterator->value() = 0.0;
		    int k = row_iterator->column();
		    SparseMatrix<double>::iterator col_iterator = system_matrixs.begin(k);   
		    SparseMatrix<double>::iterator col_end = system_matrixs.end(k);   
		    for (++col_iterator; col_iterator != col_end; ++col_iterator)
    			if (col_iterator->column() == i + n_v)
    			    break;
		    if (col_iterator == col_end)
		    {
    			std::cerr << "Error!" << std::endl;
    			exit(-1);
		    }
		    rhss(k) -= col_iterator->value() * bnd_value; 
		    col_iterator->value() = 0.0;	
		}  
	    }	
	}

	

	SparseILU<double> ilu;
	SparseILU<double>::AdditionalData ad;
	ad.strengthen_diagonal = 0.5;
	ilu.initialize(system_matrixs, ad);
	SolverControl solver_controlG (1000000, 1e-13, false);

    
	/// 这里调用的是带 ilu 预处理的 GMRES，因为写不出合适的 AMG 预
	/// 处理，所以严重影响了整体求解效率。
	SolverGMRES<Vector<double> > gmres(solver_controlG);
//    gmres.solve (system_matrixs, solutions, rhss, PreconditionIdentity());
	std::cout << "Newton iteration step begin..." << std::endl;
	gmres.solve (system_matrixs, solutions, rhss, ilu);
	std::cout << "Newton iteration step end." << std::endl;

    
	for (int i = 0; i < n_v; i++)
	    vx(i) += solutions(i);
	for (int i = n_v; i < 2 * n_v; i++)
	    vy(i - n_v) += solutions(i);
	for (int i = 2 * n_v; i < total_n_dof; i++)
	    p(i - 2 * n_v) += solutions(i);

	for (int i = 0; i < n_v; i++)
	    solution(i) = vx(i);
	for (int i = n_v; i < 2 * n_v; i++)
	    solution(i) = vy(i - n_v);
	for (int i = 2 * n_v; i < total_n_dof; i++)
	    solution(i) = p(i - 2 * n_v);

	res = solutions.l2_norm();
	std::cout << "nolinear res updated: " << res << std::endl;
	step++;
    }
    FEMFunction<double, DIM> solution_output(fem_space_output);
    Operator::L2Interpolate(vx, solution_output);
    solution_output.writeOpenDXData("ux.dx");
    Operator::L2Interpolate(vy, solution_output);
    solution_output.writeOpenDXData("uy.dx");
    Operator::L2Interpolate(p, solution_output);
    solution_output.writeOpenDXData("p.dx");
    std::cout << "total it steps:" << step << std::endl;
    return 0;
};

/// 边界条件：
double ux(const double * p)
{
    return 1;
};

double uy(const double * p)
{
    return 0;
};

double up(const double * p)
{
    return 0;
};

/// 右端项
double f(const double * p)
{
    return 0;
};

/// 边界条件
double zfun(const double * p)
{
    return 0;
};

