/**
 * @file   possion_equation.cpp
 * @author Wang Heyu <hywang@sixears>
 * @date   Tue May 26 15:07:25 2020
 * 
 * @brief 从AFEPack的example直接复制过来，没有改动。这个程序可以作为有
 * 限元计算的起点。这里给一点基本解释。外部网格文件 D.d 和 E.d 分别提
 * 供了 h = 0.05 和 h = 0.01 的计算网格，用于观察最终截断误差的变化。
 * 
 * 
 */


#include <iostream>
#include <fstream>
#include <limits>

/// AMGSolver.h 提供李若老师写的一个代数多重网格求解器。在2D线性逼近的
/// 场合效率很高。但在3D效率下降，不能用于2次逼近及以上。
#include <AFEPack/AMGSolver.h>
/// Geometry.h 提供网格形状之类的几何计算功能。
#include <AFEPack/Geometry.h>
/// TemplateElement.h 提供有限元的模板单元。
#include <AFEPack/TemplateElement.h>
/// FEMSpace.h 将有限元数据在这里构建成实际的单元和空间。
#include <AFEPack/FEMSpace.h>
/// Operator.h 提供插值、L2逼近等基本运算工具。
#include <AFEPack/Operator.h>
/// Functional.h 提供函数、类函数、泛函数的定义和运算。比如边界函数等等。
#include <AFEPack/Functional.h>
/// EasyMesh.h 提供easymesh的网格处理。
#include <AFEPack/EasyMesh.h>

/// 给出一个PI常数。
#define PI (4.0*atan(1.0))

/** 
 * 真解，在测试时可用于边界条件插值。
 * 
 * 
 * @return 函数值。
 */
double u(const double *);

/** 
 * 源项函数。
 * 
 * 
 * @return 函数值。
 */
double f(const double *);


int main(int argc, char * argv[])
{
    /// 声明一个 easymesh 格式的网格。
    EasyMesh mesh;
    /// 网格数据从命令行参数读入。
    mesh.readData(argv[1]);
    /// 声明一个2维几何单元。
    TemplateGeometry<2>	triangle_template_geometry;
    /// 具体几何单元信息从外部读入，这里调用了三角形(triangle.tmp_geo)。
    /// 具体几何单元模板的写法参见 AFEPack 的文档。
    triangle_template_geometry.readData("triangle.tmp_geo");
    /// 声明一个从2维空间到2维空间的坐标变换。
    CoordTransform<2,2>	triangle_coord_transform;
    /// 具体的坐标变换公式从外部读入，这里调用了三角形上的坐标变换。具
    /// 体的坐标变换公式写法见 AFEPack 文档。
    triangle_coord_transform.readData("triangle.crd_trs");
    /// 声明一个有限元单元的自由度分配，依附于之前声明的有限元几何单元
    /// 模板，也就是三角形。
    TemplateDOF<2> triangle_template_dof(triangle_template_geometry);
    /// 这里进一步从外部文件读取了一次三角形单元的自由度信息，也就是经
    /// 典的P1单元的自由度分布方案。具体的单元自由度分布写法见 AFEPack
    /// 文档。
    triangle_template_dof.readData("triangle.1.tmp_dof");
    /// 声明一个2维空间实值基函数模板。并依附于之前声明的三角形几何单元自由度。
    BasisFunctionAdmin<double,2,2> triangle_basis_function(triangle_template_dof);
    /// 具体的基函数从外部文件读入。基函数信息的写法参见 AFEPack 文档。
    triangle_basis_function.readData("triangle.1.bas_fun");
    /// 声明参考单元列表。因为一个有限元空间可能有多个模板单元。这里事
    /// 实上是P1空间，因此只有一个参考单元。
    std::vector<TemplateElement<double,2,2> > template_element(1);
    /// 该参考单元由之前声明和定义的几何模板、自由度、坐标变换和基函数共同构成。
    template_element[0].reinit(triangle_template_geometry,
			       triangle_template_dof,
			       triangle_coord_transform,
			       triangle_basis_function);
    /// 声明有限元空间，依附了网格信息和参考单元信息。
    FEMSpace<double,2> fem_space(mesh, template_element);
    /// 对网格中的每一个几何单元（这里只有三角形），都一一对应了它们的
    /// 参考单元（这里只有一种参考元）。
    int n_element = mesh.n_geometry(2);
    fem_space.element().resize(n_element);
    for (int i = 0; i < n_element; i++)
	fem_space.element(i).reinit(fem_space, i, 0);
    /// 实际建立有限元空间。
    fem_space.buildElement();
    fem_space.buildDof();
    fem_space.buildDofBoundaryMark();
    /// 因为这个空间很标准，我们可以直接用 StiffMatrix 类生成2维刚度矩
    /// 阵，它就是我们要拼装的 Possion 方程的系数矩阵。
    StiffMatrix<2,double> stiff_matrix(fem_space);
    /// 内部拼装时需要做数值积分，这里指定了代数精度。事实上，对 P1 单
    /// 元，刚度矩阵理论上是两个 0 次多项式相乘后积分，代数精度只需 0
    /// 次。
    stiff_matrix.algebricAccuracy() = 0;
    /// 实际生成刚度矩阵。
    stiff_matrix.build();
    /// 生成一个2维，实值解向量。这里用了 FEMFuction 类，它除了向量的
    /// 功能，和对应的有限元空间和网格有索引关系，因此更便于输出计算结
    /// 果。
    FEMFunction<double,2> solution(fem_space);
    /// 右端项只需要标准的向量即可。
    Vector<double> right_hand_side;
    /// 这个算子的功能将函数 f 插值到右端项中，本质上也是一个积分拼装
    /// 过程，所以需要提供代数精度。右端项函数未必是多项式，所以理论上
    /// 应该提供足够高的精度。这里因为是 P1 有限元，数值解在最优条件下
    /// 至多只有二阶精度，因此代数精度理论上到 3 就已经足够。
    Operator::L2Discretize(&f, fem_space, right_hand_side, 2);
    /// 以下利用提供的内部工作施加了 Dirichlet 边界。这里的具体方式是
    /// 用消元过程将对应的自由度的矩阵系数的非对角元全部设成零，对应右
    /// 端项设成对角元乘以真解，并在其他各行右端项做相应的消除。目前的
    /// 工具有较大的局限，只能用于 Possion 方程，其他情况建议手工完成
    /// （参见 possion_equation_manual）。
    BoundaryFunction<double,2> boundary(BoundaryConditionInfo::DIRICHLET, 1, &u);
    BoundaryConditionAdmin<double,2> boundary_admin(fem_space);
    boundary_admin.add(boundary);
    boundary_admin.apply(stiff_matrix, solution, right_hand_side);
    /// 用内置的 AMG 求解器求解。
    AMGSolver solver(stiff_matrix);
    /// 这里设置线性求解器的收敛判定为机器 epsilon 乘以矩阵的阶数，也
    /// 就是自由度总数。这个参数基本上是理论可以达到的极限。
    double tol = std::numeric_limits<double>::epsilon() * fem_space.n_dof();
    solver.solve(solution, right_hand_side, tol, 1000);	
    /// 输出数值解的 OpenDX 格式图像。
    solution.writeOpenDXData("u.dx");
    /// 计算一下数值解的 L2 误差。
    double error = Functional::L2Error(solution, FunctionFunction<double>(&u), 3);
    std::cerr << "\nL2 error = " << error << ", tol = " << tol << std::endl;
    return 0;
};

double u(const double * p)
{
    return sin(PI*p[0]) * sin(2*PI*p[1]);
};

double f(const double * p)
{
    return 5*PI*PI*u(p);
};

