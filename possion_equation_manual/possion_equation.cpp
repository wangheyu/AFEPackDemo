/**
 * @file   possion_equation.cpp
 * @author Heyu Wang <scshw@cslin107.csunix.comp.leeds.ac.uk>
 * @date   Mon May 19 13:19:19 2014
 * 
 * @brief 一个例子, 如何脱离 AFEPack 的 BilinearOperator 结构, 自己构建
 * 一个刚度矩阵. 
 * 
 * 
 */

#include <iostream>
#include <fstream>
#include <limits>

#include <AFEPack/AMGSolver.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/EasyMesh.h>

/// 这里调用了 deal.II 提供的稀疏矩阵、向量、cg 求解器和 mic 等预处理
/// 器。
#include <lac/sparse_matrix.h>
#include <lac/sparsity_pattern.h>
#include <lac/sparse_ilu.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/solver_cg.h>
#include <lac/precondition.h>
#include <lac/sparse_mic.h>
#include <lac/sparse_decomposition.h>

#define PI (4.0*atan(1.0))

#define DIM 2

double u(const double *);
double f(const double *);

int main(int argc, char * argv[])
{
    /// HGeometryTree 结构可以对初始网格根据需要进行自适应的局部加密。
    /// 本程序只用于全局加密。
    HGeometryTree<DIM> h_tree;
    /// 网格树的根是从命令行参数读入的 easymesh 网格。
    h_tree.readEasyMesh(argv[1]);
    /// IrregularMesh 是依附于一棵网格树的一张链表，串联了全部真正采用
    /// 的网格和它们的依赖关系。
    IrregularMesh<DIM> *irregular_mesh0;
    IrregularMesh<DIM> *irregular_mesh1;
    
    irregular_mesh0 = new IrregularMesh<DIM>;
    /// 将一个 IrregularMesh 真正关联到我们的网格树。
    irregular_mesh0->reinit(h_tree);
    /// 判断一下网格层之间是否有跳跃，有的话用双生三角形解决。这一现象
    /// 只会出现在局部自适应加密，我们这里只用全局加密，不会出现。这里
    /// 只是走个流程。
    irregular_mesh0->semiregularize();
    /// 给具体的计算网格建立编号，false表示不全局对网格重新编号。
    irregular_mesh0->regularize(false);
    /// RegularMesh 是计算网格的一个读取接口。
    RegularMesh<DIM> &mesh0 = irregular_mesh0->regularMesh();

    irregular_mesh1 = new IrregularMesh<DIM>(*irregular_mesh0);
    /// 将网格全局加密 3 次，也就是 mesh1 的网格宽度将会是 mesh0 的
    /// 1/8。我们可以用这样的办法比较一个求解器在不同精度下的误差，或
    /// 者将 mesh0 上的高阶数值结果投影到 mesh1 上再输出。
    irregular_mesh1->globalRefine(3);
    irregular_mesh1->semiregularize();
    irregular_mesh1->regularize(false);
    RegularMesh<DIM> &mesh1 = irregular_mesh1->regularMesh();
    /// 输出网格观察一下。
    mesh0.writeOpenDXData("D0.dx");
    mesh1.writeOpenDXData("D1.dx");
    /// 以下和 possion_equation 无区别。除了现在用的是 3 次有限元。
    TemplateGeometry<DIM> triangle_template_geometry;
    triangle_template_geometry.readData("triangle.tmp_geo");
    CoordTransform<DIM, DIM> triangle_coord_transform;
    triangle_coord_transform.readData("triangle.crd_trs");
    TemplateDOF<DIM> triangle_template_dof(triangle_template_geometry);
    triangle_template_dof.readData("triangle.3.tmp_dof");
    BasisFunctionAdmin<double, DIM, DIM> triangle_basis_function(triangle_template_dof);
    triangle_basis_function.readData("triangle.3.bas_fun");

    std::vector<TemplateElement<double, DIM, DIM> > template_element(1);
    template_element[0].reinit(triangle_template_geometry,
			       triangle_template_dof,
			       triangle_coord_transform,
			       triangle_basis_function);


    FEMSpace<double, DIM> fem_space0(mesh0, template_element);
	
    int n_element = mesh0.n_geometry(DIM);
    fem_space0.element().resize(n_element);
    for (int i = 0; i < n_element; ++i)
	fem_space0.element(i).reinit(fem_space0, i, 0);

    fem_space0.buildElement();
    fem_space0.buildDof();
    fem_space0.buildDofBoundaryMark();

    /// 同样为 mesh1 建立了一个有限元空间，其实这里只需要 1 次精度就行
    /// 了，因为 mesh1 就是为了输出高阶数值结果，但这里懒得再定义模板
    /// 了。
    FEMSpace<double, DIM> fem_space1(mesh1, template_element);
	
    n_element = mesh1.n_geometry(DIM);
    fem_space1.element().resize(n_element);
    for (int i = 0; i < n_element; ++i)
    	fem_space1.element(i).reinit(fem_space1, i, 0);

    fem_space1.buildElement();
    fem_space1.buildDof();
    fem_space1.buildDofBoundaryMark();

    /// 准备统计系数矩阵的每一行有多少个非零元.
    int n_total_dof = fem_space0.n_dof();
    std::vector<unsigned int> n_non_zero_per_row(n_total_dof);

    /// 准备一个遍历全部单元的迭代器.
    FEMSpace<double, DIM>::ElementIterator the_element = fem_space0.beginElement();
    FEMSpace<double, DIM>::ElementIterator end_element = fem_space0.endElement();

    /// 第一次循环遍历全部单元, 只是为了统计每一行的非零元个数.
    for (; the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof = the_element->dof();
	int n_element_dof = the_element->n_dof();
	for (int i = 0; i < n_element_dof; ++i)
	    n_non_zero_per_row[element_dof[i]] += n_element_dof;
    }

    /// 根据每一行的非零元创建矩阵模板.
    SparsityPattern sp_stiff_matrix(fem_space0.n_dof(), n_non_zero_per_row);

    /// 第二次遍历, 指定每个非零元的坐标.
    for (the_element = fem_space0.beginElement(); 
	 the_element != end_element; ++the_element) 
    {
	const std::vector<int>& element_dof = the_element->dof();
	int n_element_dof = the_element->n_dof();
	for (int i = 0; i < n_element_dof; ++i)
	    for (int j = 0; j < n_element_dof; ++j)
		sp_stiff_matrix.add(element_dof[i], element_dof[j]);
    }

    /// 矩阵模板压缩. 创建矩阵.
    sp_stiff_matrix.compress();
    SparseMatrix<double> stiff_matrix(sp_stiff_matrix);

    Vector<double> rhs(fem_space0.n_dof());
    /// 第三次遍历, 给每个矩阵元素正确的赋值. 这一块和 BilinearOperator
    /// 中的 getElementMatrix 中做的事情一致.
    for (the_element = fem_space0.beginElement(); 
	 the_element != end_element; ++the_element) 
    {
	double volume = the_element->templateElement().volume();
	const QuadratureInfo<2>& quad_info = the_element->findQuadratureInfo(6);
	std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
	int n_quadrature_point = quad_info.n_quadraturePoint();
	std::vector<AFEPack::Point<2> > q_point = the_element->local_to_global(quad_info.quadraturePoint());
	std::vector<std::vector<std::vector<double> > > basis_gradient = the_element->basis_function_gradient(q_point);
        std::vector<std::vector<double> > basis_value = the_element->basis_function_value(q_point);
	const std::vector<int>& element_dof = the_element->dof();
	int n_element_dof = the_element->n_dof();
	for (int l = 0; l < n_quadrature_point; ++l)
	{
	    double Jxw = quad_info.weight(l) * jacobian[l] * volume;
	    for (int i = 0; i < n_element_dof; ++i)
            {
		for (int j = 0; j < n_element_dof; ++j)
		{
		    double cont = Jxw * innerProduct(basis_gradient[i][l], basis_gradient[j][l]);
		    stiff_matrix.add(element_dof[i], element_dof[j], cont);
		}
                double cont_rhs = Jxw * f(q_point[l]) * basis_value[i][l];
		rhs(element_dof[i]) += cont_rhs;
            }
	}
    }

    /// 接下去做的事情和之前一样.
    FEMFunction<double, DIM> solution0(fem_space0);

    /// 这就是手工施加边界条件的过程。遍历每一个自由度。
    for (int i = 0; i < fem_space0.n_dof(); i++)
    {
	/// 读取对应自由度的信息（自由度未必是一个网格端点）。
    	FEMSpace<double, DIM>::dof_info_t dof = fem_space0.dofInfo(i);
	/// 如果它落在 1 号边界。
    	if (dof.boundary_mark == 1)
    	{
    	    SparseMatrix<double>::iterator row_iterator = stiff_matrix.begin(i);
    	    SparseMatrix<double>::iterator row_end = stiff_matrix.end(i);
    	    double diag = row_iterator->value();
	    /// 计算边界条件在这一点的值。
    	    double bnd_value = u(dof.interp_point);
	    /// 对应右端项设为对角元乘以真解。
            rhs(i) = diag * bnd_value;
	    /// 遍历这一行除对角元以外的元素（对角元总是在i第一个，但
	    /// 这一点在 deal.II 8.1.0 以后有变化）。
    	    for ( ++row_iterator; row_iterator != row_end; ++row_iterator)
            {
		/// 非对角元全部设成 0。
            	row_iterator->value() = 0.0;
    		int k = row_iterator->column();
		/// 还要处理这一列，但我们不能在一个 CSR 格式的稀疏矩
		/// 阵中遍历一列，所以这里利用刚度矩阵的对称性，实际上
		/// 是遍历了对称的那一行。这一点在不对称或反对称块处理
		/// 时要格外小心。
                SparseMatrix<double>::iterator col_iterator = stiff_matrix.begin(k);   
                SparseMatrix<double>::iterator col_end = stiff_matrix.end(k);
		/// 这里实际搜索了第 k 行， 找到元素 (i, k) 的对称位置 (k, i)。
    	    	for (++col_iterator; col_iterator != col_end; ++col_iterator)
    			if (col_iterator->column() == i)
    			    break;
    		if (col_iterator == col_end)
    		{
    			std::cerr << "Error!" << std::endl;
    			exit(-1);
    		}
		/// 将第 k 行的右端项做消去。
    		rhs(k) -= col_iterator->value() * bnd_value;
		/// 消去后的矩阵对应系数为 0 。
    		col_iterator->value() = 0.0;	
            }  
            
    	}	
    }		


    /// 以下，具体的预处理可以用 mic，ilu 或 ssor。这里因为系数矩阵是
    /// 对称正定的，当用 cg 求解时，应该采用 mic。或者直接采用 AMG 求
    /// 解，不需要预处理。（AMG 本身就是最好的预处理。）各种方法均可尝
    /// 试比较。但是内置的 AMG 无法计算 1 次单元以上精度的问题。需要调
    /// 用 Trilinos 或其他库的 AMG 求解器，参见
    /// possion_equation_Trilinos 例子。
//    SparseILU<double> preconditioner(stiff_matrix);
    SparseMIC<double> mic;
    
    SparseMIC<double>::AdditionalData ad;
    mic.initialize(stiff_matrix, ad);

    PreconditionSSOR<SparseMatrix<double> > ssor;
    ssor.initialize (stiff_matrix, 1.2);

    double tol = std::numeric_limits<double>::epsilon() * fem_space0.n_dof();

    SolverControl solver_control (200000, tol);
    SolverCG<Vector<double> > cg (solver_control);
    /// PreconditionIdentity 表示不用预处理。
//    cg.solve (stiff_matrix,  solution0,  rhs, PreconditionIdentity());
    cg.solve (stiff_matrix,  solution0,  rhs, ssor);
//    cg.solve (stiff_matrix,  solution0,  rhs, mic);

    /// 测试结果是 ssor 能减少一半迭代次数，也就是说咩啥卵用...
    solution0.writeOpenDXData("u0.dx");


    FEMFunction<double, DIM> solution1(fem_space1);
	    
    /// 将 mesh0 上的 3 次元计算结果插值到 mesh1 上，体现其精度。
    Operator::L2Interpolate(solution0, solution1);
//    Operator::L2Project(solution0, solution1, 3, 3);
    solution1.writeOpenDXData("u1.dx");

    double error = Functional::L2Error(solution0, FunctionFunction<double>(&u), 5);
    std::cout << "L2 Error = " << error << std::endl;
    return 0;
};

double u(const double * p)
{
    return sin(PI * p[0]) * sin(2.0 * PI * p[1]);
};

double f(const double * p)
{
    return 5 * PI * PI * u(p);
};

