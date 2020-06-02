/**
 * @file   step1.cpp
 * @author Wang Heyu <hywang@sixears>
 * @date   Tue Jun  2 15:20:49 2020
 * 
 * @brief 演示规则网格原理。2D，单位正方形，矩形剖分。纯文本输出。
 * 
 * 
 */

#include <iostream>

int main(int argc ,char* argv[])
{
    /// 对于演示程序，2D区域只考虑单位正方形。
    double x0 = 0.0;	
    double y0 = 0.0;
    double x1 = 1.0;
    double y1 = 1.0;
    /// 一个方向上的划分数，这里只考虑均匀划分。网格点是 n + 1
    int n = 10;
    /// 定义网格宽度。
    double h = (x1 - x0) / n;
    /// 接下去考虑产生规则矩形网格。首先产生全部网格顶点。
    for (int j = 0; j < n; j++)
	for (int i = 0; i < n; i++)
	{
	    /// 输出一个矩形网格，从左下角开始，逆时针排列，记为00 ->
	    /// 10 -> 11 -> 01。产生具体分点时考虑了更稳定的计算。每一
	    /// 个顶点（一次元的自由度）应当拥有一个全局编号，这里采用
	    /// 从左至右，从下至上的次序。 这里还需要注意的是，一般我
	    /// 们编号矩阵时，喜欢用 i 行 j 列，但在编制网格时，喜欢用
	    /// (xi, yj)。这里正好会反一下。
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
	    
	    /// 网格的全局编号。
	    int ele_idx = j * n + i; 
	    std::cout << ele_idx << ": " << std::endl;
	    std::cout << idx00 << ":(" << x00 << "," << y00 << ") -> "
		      << idx10 << ":(" << x10 << "," << y10 << ") -> "
		      << idx11 << ":(" << x11 << "," << y11 << ") -> "
		      << idx01 << ":(" << x01 << "," << y01 << ")" << std::endl;
	}
    return 0;
};
