/**
 * @file   _possion_equation.cpp
 * @author Li ShiJie <lsj@lsj>
 * @date   Tue Jun  3 9:04:24 2020
 * 
 * @brief  利用AFEPack里的函数以及功能，划分结构化网格，使用双线性四节点基函数
 * 		　　，实现Possion方程编值问题计算．
 * 
 * 
 */
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <AFEPack/AMGSolver.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/EasyMesh.h>
#include <AFEPack/SparseMatrixTool.h>

#include <lac/sparse_matrix.h>
#include <lac/sparsity_pattern.h>
#include <lac/sparse_ilu.h>
#include <lac/vector.h>
#include <lac/full_matrix.h>
#include <lac/solver_cg.h>
#include <lac/sparse_mic.h>
#include <lac/sparse_decomposition.h>
#include <lac/full_matrix.h>

#define PI (4.0*atan(1.0))
double u(const double * p)
{
    return sin(PI * p[0]) * sin(PI * p[1]);
    //return exp( 2 * p[0] + p[1] );
};

double f(const double * p)
{
    return 2 * PI * PI * u(p);
    //return -5 * u( p );
}; 

/// good! 
typedef std::unordered_map<unsigned int, int> index_map;

int main(int argc, char* argv[])
{
    /// 这里基本上和 possion_equation 中配置一致。对比
    /// possion_equation_manual 看更清楚。
    TemplateGeometry<2> rectangle_template_geometry;
    rectangle_template_geometry.readData("rectangle.tmp_geo");
    CoordTransform<2, 2> rectangle_coord_transform;
    rectangle_coord_transform.readData("rectangle.crd_trs");
    TemplateDOF<2> rectangle_template_dof(rectangle_template_geometry);
    /// 一次元。
    rectangle_template_dof.readData("rectangle.1.tmp_dof");
    BasisFunctionAdmin<double, 2, 2> rectangle_basis_function(rectangle_template_dof);
    rectangle_basis_function.readData("rectangle.1.bas_fun");
    TemplateElement<double, 2, 2> template_element;
    template_element.reinit(rectangle_template_geometry,
			    rectangle_template_dof,
			    rectangle_coord_transform,
			    rectangle_basis_function);

    double volume = template_element.volume();
    /// 取了 4 次代数精度。
    const QuadratureInfo<2>& quad_info = template_element.findQuadratureInfo(4);   
    int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<AFEPack::Point<2> > q_point = quad_info.quadraturePoint();
    int n_element_dof = template_element.n_dof();
    int n_bas = rectangle_basis_function.size();

    /// 产生一个具体单元顶点的缓存。一个矩形的 4 个顶点。这里其实是这
    /// 四个顶点正好是 Q1 单元的 4 个单元内自由度。gv 表示全局的矩形坐
    /// 标，就是在物理计算区域内一个网格的顶点坐标；而 lv 表示局部的矩
    /// 形坐标，即参考单元的坐标。沿用了 AFEPack 的配置，是固定的 [-1,
    /// -1]-[-1, 1]-[1, 1]-[-1, 1]。
    double ** arr = (double **) new double* [4];
    for (int i = 0; i < 4; i++)
	arr[i] = (double *) new double [2];
    std::vector<AFEPack::Point<2> > gv(4);
    std::vector<AFEPack::Point<2> > lv(4);

    /// 观察一下模板单元中的自由度、基函数和基函数在具体积分点取值的情
    /// 况。
    arr[0][0] = -1.0;
    arr[0][1] = -1.0;
    arr[1][0] = 1.0;
    arr[1][1] = -1.0;
    arr[2][0] = 1.0;
    arr[2][1] = 1.0;
    arr[3][0] = -1.0;
    arr[3][1] = 1.0;
    /// 设置实际的计算矩形区域边界。
    double x0 = 0.0;	
    double y0 = 0.0;
    double x1 = 1.0;
    double y1 = 1.0;
    /// 设置剖分断数和节点总数。
    int n = 20;
    /// 这里 dim 相当与总自由度数。
    int dim = (n + 1) * (n + 1);

    Vector<double> rhs(dim);
    /// nozeroperow中每一个值表示对应行数非零元素个数
    std::vector<unsigned int> nozeroperow(dim);
    /// 每一行最多 9 个非零元。
    for(int i = 0;i <= dim ; i++)
	nozeroperow[i] = 9;

    /// 对应计算区域四个顶点的自由度所在行只有 4 个非零元。
    nozeroperow[0] = 4;
    nozeroperow[dim - 1] = 4;
    nozeroperow[n] = 4;
    nozeroperow[dim - 1 - n] = 4;
    /// 对应计算区域非顶点的边界自由度，只有 6 个非零元。
    for(int i = 1; i < n; i++)
    {
	nozeroperow[i] = 6;
	nozeroperow[dim - 1 - i] = 6;
	nozeroperow[i * (n + 1)] = 6;
	nozeroperow[(i + 1) * n + i] = 6;
    }
    /// 建立稀疏矩阵模板。
    SparsityPattern sp_stiff_matrix(dim, nozeroperow);
    /// 填充非零元素对应的行索引和列索引。
    for (int j = 0; j < n; j++)
	for (int i = 0; i < n; i++)
	{
	    /// 对每一个 (j, i) 的自由度，对应矩阵中产生的非零元是：
	    /// (j, i) -> j * (n + 1) + i;
	    /// (j, i + 1) -> j * (n + 1) + i + 1;
	    /// (j + 1, i + 1) -> (j + 1) * (n + 1) + i + 1;
	    /// (j + 1, i) -> (j + 1) * (n + 1) + i;
	    int idx00 = j * (n + 1) + i; 
	    int idx10 = j * (n + 1) + i + 1; 
	    int idx11 = (j + 1) * (n + 1) + i + 1; 
	    int idx01 = (j + 1) * (n + 1) + i;

	    /// 这里应该是为了尽可能使用 template_element 中的信息，所
	    /// 以采用了一个 index_map，不过这里本质上可以使用的信息只
	    /// 有 n_dof，也就是 4，所以似乎没有这个必要？index_map 应
	    /// 该是个不错的配置，这里不如直接用 4？
	    index_map index({{0, idx00}, {1, idx10}, {2, idx11}, {3, idx01}}, template_element.n_dof());
	    for (int i = 0;i < template_element.n_dof(); i++)
		for (int j = 0;j < template_element.n_dof(); j++)
		    sp_stiff_matrix.add(index[i], index[j]);
	}
    /// 稀疏矩阵模板生成。
    sp_stiff_matrix.compress();
    /// 系数矩阵初始化。
    SparseMatrix<double> stiff_mat(sp_stiff_matrix);
    /// 生成节点及节点坐标，单元，和刚度矩阵和右端项
    double h = (x1 - x0) / n;
    for (int j = 0; j < n; j++)
	for (int i = 0; i < n; i++)
	{
	    /// 这里第二次出现了同样的 map，所以应该将这个 map 写成一
	    /// 个全局的函数或数据结构，以方便调用。实际上，各自由度坐
	    /// 标也是。
	    double x00 = ((n - i) * x0 + i * x1) / n;
	    double y00 = ((n - j) * y0 + j * y1) / n;
	    int idx00 = j * (n + 1) + i; 
	    double x10 = ((n - i - 1) * x0 + (i + 1) * x1) / n;
	    double y10 = ((n - j ) * y0 + j * y1) / n;
	    int idx10 = j * (n + 1) + i + 1; 
	    double x11 = ((n - i - 1) * x0 + (i + 1) * x1) / n;
	    double y11 = ((n - j - 1) * y0 + (j + 1) * y1) / n;
	    int idx11 = (j + 1) * (n + 1) + i + 1; 
	    double x01 = ((n - i) * x0 + i * x1) / n;
	    double y01 = ((n - j - 1) * y0 + (j + 1) * y1) / n;
	    int idx01 = (j + 1) * (n + 1) + i; 
	    
	    /// 这里数据赋值继承了我的测试代码，有点生硬。这里 gv 应该
	    /// 做到现场生成，也就是上面的 x00 等等可以直接生成在 gv，
	    /// 而 lv 则应该是外部赋值的全局数据。
	    gv[0][0] = x00;
	    gv[0][1] = y00;
	    gv[1][0] = x10;
	    gv[1][1] = y10;
	    gv[2][0] = x11;
	    gv[2][1] = y11;
	    gv[3][0] = x01;
	    gv[3][1] = y01;
	    lv[0][0] = arr[0][0];
	    lv[0][1] = arr[0][1];
	    lv[1][0] = arr[1][0];
	    lv[1][1] = arr[1][1];
	    lv[2][0] = arr[2][0];
	    lv[2][1] = arr[2][1];
	    lv[3][0] = arr[3][0];
	    lv[3][1] = arr[3][1];
	    /// 现在尝试输出具体每个单元的积分点。
	    /// 合成整体刚度矩阵
	    /// 6----7----8
	    ///	|    |    |
	    ///	3----4----5     
	    /// |    |    |
	    ///	0----1----2
	    ///
	    /// element 1:0->1->4->3
	    /// element 2:1->2->5->4
	    /// element 3:3->4->7->6
	    /// element 4:4->5->8->7
	    /// 这里确实应该考虑一下单元、自由度之间的编号对应关系。
	    for (int l = 0; l < n_quadrature_point; l++)
	    {
		/// 得到积分点的全局坐标。
		auto point = rectangle_coord_transform.local_to_global(q_point, lv, gv);
		/// 积分点的权重、Jacobi变换系数。
		double Jxy = quad_info.weight(l) * rectangle_coord_transform.local_to_global_jacobian(q_point[l], lv, gv) * volume;
		index_map index({{0, idx00}, {1, idx10}, {2, idx11}, {3, idx01}}, template_element.n_dof());
		/// 这一段不错。
		for(int base1 = 0; base1 < template_element.n_dof(); base1++)
		{
		    for(int base2 = 0; base2 < template_element.n_dof(); base2++)
			stiff_mat.add(index[base1],
				      index[base2],
				      Jxy * innerProduct(rectangle_basis_function[base1].gradient(point[l], gv),
							 rectangle_basis_function[base2].gradient(point[l], gv)));
		    rhs(index[base1]) += Jxy * f(point[l]) * rectangle_basis_function[base1].value(point[l], gv);
		}
	    }
	    /// TO DO: 计算每个积分点上的基函数梯度值，数值积分，拼装局部刚度矩阵，累加至整体刚度矩阵。
	}
    ///处理所有边界条件
    for(unsigned int index = 0; index < dim; index++)
    {
	/// 这里用每一行的非零元个数判断是否是边界自由度，这是个有趣的
	/// 想法。
	if(nozeroperow[index] == 4 || nozeroperow[index] == 6)
	{
	    /// 首先要计算这个节点的实际坐标。这里具体如何做更好，可以
	    /// 在仔细斟酌。
	    int x_num = index % (n + 1);
	    int y_num = index / (n + 1);
	    double x = x_num * h;	
	    double y = y_num * h;
	    SparseMatrix<double>::iterator row_iterator = stiff_mat.begin(index);
    	    SparseMatrix<double>::iterator row_end = stiff_mat.end(index);
    	    double diag = row_iterator->value();
	    AFEPack::Point<2> bnd_point;
	    bnd_point[0]=x;
	    bnd_point[1]=y;
    	    double bnd_value = u(bnd_point);
	    /// good!
            rhs(index) = diag * bnd_value;
    	    for (++row_iterator; row_iterator != row_end; ++row_iterator)
            {
            	row_iterator->value() = 0.0;
		int k = row_iterator->column();
                SparseMatrix<double>::iterator col_iterator = stiff_mat.begin(k);   
                SparseMatrix<double>::iterator col_end = stiff_mat.end(k);
    	    	for (++col_iterator; col_iterator != col_end; ++col_iterator)
		    if (col_iterator->column() == index)
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
    ///　用代数多重网格(AMG)计算线性方程
    AMGSolver solver(stiff_mat);
    /// 这里设置线性求解器的收敛判定为机器 epsilon 乘以矩阵的阶数，也
    /// 就是自由度总数。这个参数基本上是理论可以达到的极限。
    Vector<double> solution(dim);
    double tol = std::numeric_limits<double>::epsilon() * dim;
    solver.solve(solution, rhs, tol, 10000);	
    std::ofstream fs;
    ///　输出到output.m用Matlab或Octave运行，得到计算结果。
    /// VERY GOOD!
    fs.open("output.m");
    fs << "x=0:1/" << n << ":1;" << std::endl;
    fs << "y=0:1/" << n << ":1;" << std::endl;
    fs << "[X,Y]=meshgrid(x,y);" << std::endl;
    fs << "U=[";
    int c = 0;
    for (int j = 0; j < n + 1; j++)
    {	
	for(int i = 0; i < n + 1; i++)
	    fs << solution[i + c + n * j] << " , ";
	fs << ";" << std::endl;
	c++;
    }
    fs << "]" << std::endl;
    fs << "surf(x,y,U);" << std::endl;

   
    /// 计算一下数值解的 l2 误差。(离散的)
    double error = 0;
    for(unsigned int index = 0; index < dim; index++)
    {
	int x_num = index % (n + 1);
	int y_num = index / (n + 1);
	double x = x_num * h;	
	double y = y_num * h;
	AFEPack::Point<2> pnt;
	pnt[0] = x;
	pnt[1] = y;

	double d = (u(pnt) - solution[index]);
	error += d*d;
    }
    error = std::sqrt(error);
    std::cerr << "\nL2 error = " << error << ", tol = " << tol << std::endl;

    return 0;
};
