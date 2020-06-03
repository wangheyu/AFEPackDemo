/**
 * @file   step2.cpp
 * @author Wang Heyu <hywang@sixears>
 * @date   Tue Jun  2 17:01:24 2020
 * 
 * @brief  尝试将 AFEPack 对接到我们在 step1 中产生的矩形区域的矩形网格上。
 * 
 * 
 */
#include <iostream>
#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>

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
    /// 取了 2 次代数精度。
    const QuadratureInfo<2>& quad_info = template_element.findQuadratureInfo(2);
//    std::vector<double> jacobian = the_element->local_to_global_jacobian(quad_info.quadraturePoint());
    int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<AFEPack::Point<2> > q_point = quad_info.quadraturePoint();
//    std::vector<std::vector<std::vector<double> > > basis_gradient = template_element.basisFunction_gradient(q_point);
//    std::vector<std::vector<double> > basis_value = template_element.basisFunction(q_point);
    int n_element_dof = template_element.n_dof();
    int n_bas = rectangle_basis_function.size();

    std::cout << "template element volume:" << volume << std::endl;
    std::cout << "no. of dofs in the element:" << n_element_dof << std::endl;
    std::cout << "no. of quadrature points:" << n_quadrature_point << std::endl;
    /// 产生一个具体单元顶点的缓存。
    double ** arr = (double **) new double* [4];
    for (int i = 0; i < 4; i++)
	arr[i] = (double *) new double [2];
    std::vector<AFEPack::Point<2> > gv(4);
    std::vector<AFEPack::Point<2> > lv(4);

    /// 观察一下模板单元中的自由度、基函数和基函数在具体积分点取值的情
    /// 况。
    for (int i = 0; i < n_element_dof; i++)
    {
	AFEPack::Point<2> pnt = q_point[i];
	/// 第 i 个积分点。
	std::cout << i << ": (" << pnt[0] << ", " << pnt[1] << ")" << std::endl;
	/// 一个模板单元内的基函数个数。
	std::cout << "no. of basis values:" << n_bas << std::endl;
	/// 该积分点在全部基函数上的取值。
	/// pnt[0] = 0.921132;
	/// pnt[1] = 0.921132;
	
	/// 这里需要给出具体网格信息。因为我们现在看模板单元，所以就给
	/// 模板单元信息。
	arr[0][0] = -1.0;
	arr[0][1] = -1.0;
	arr[1][0] = 1.0;
	arr[1][1] = -1.0;
	arr[2][0] = 1.0;
	arr[2][1] = 1.0;
	arr[3][0] = -1.0;
	arr[3][1] = 1.0;
	for (int j = 0; j < n_bas; j++)
	{
	    std::cout << "value of basis function " << j << ": " << rectangle_basis_function[j].value(pnt, (const double**)arr) << std::endl;
	}
    }
    double x0 = 0.0;	
    double y0 = 0.0;
    double x1 = 1.0;
    double y1 = 1.0;
    int n = 10;
    double h = (x1 - x0) / n;
    for (int j = 0; j < n; j++)
    	for (int i = 0; i < n; i++)
    	{
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
	    
    	    int ele_idx = j * n + i;

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
    	    std::cout << ele_idx << ": " << std::endl;
    	    std::cout << idx00 << ":(" << x00 << "," << y00 << ") -> "
    	    	      << idx10 << ":(" << x10 << "," << y10 << ") -> "
    	    	      << idx11 << ":(" << x11 << "," << y11 << ") -> "
    	    	      << idx01 << ":(" << x01 << "," << y01 << ")" << std::endl;
    	    /// 产生一个具体单元顶点的缓存。
    	    double ** tarr = (double **) new double* [4];
    	    for (int i = 0; i < 4; i++)
    		tarr[i] = (double *) new double [2];
    	    tarr[0][0] = x00;
    	    tarr[0][1] = y00;
    	    tarr[1][0] = x10;
    	    tarr[1][1] = y10;
    	    tarr[2][0] = x11;
    	    tarr[2][1] = y11;
    	    tarr[3][0] = x01;
    	    tarr[3][1] = y01;

    	    /// 现在尝试输出具体每个单元的积分点。
    	    for (int l = 0; l < n_quadrature_point; l++)
    	    {
    		std::cout << rectangle_coord_transform.local_to_global(q_point[l], lv, gv) << std::endl;
		AFEPack::Point<2> pnt = rectangle_coord_transform.local_to_global(q_point[l], lv, gv);
    		for (int k = 0; k < n_bas; k++)
    		    std::cout << rectangle_basis_function[k].value(pnt, (const double**)tarr) << std::endl;
    	    }
    	    /// TO DO: 计算每个积分点上的基函数梯度值，数值积分，拼装局部刚度矩阵，累加至整体刚度矩阵。
    	}
    /// 边界条件。
    /// 矩阵求解。
    return 0;
};
