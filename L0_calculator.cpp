#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

int main() {
    double L2, Dw, Dp;

    cout << "====== L0 计算器 ======" << endl;
    cout << "公式: L0 = L2 * ( sqrt(Dw^2 / (Dw^2 - Dp^2)) - 1 )" << endl;
    cout << endl;

    cout << "请输入 L2: ";
    cin >> L2;
    cout << "请输入 Dw: ";
    cin >> Dw;
    cout << "请输入 Dp: ";
    cin >> Dp;

    // 校验：分母不能为零或负
    double diff = Dw * Dw - Dp * Dp;
    if (diff <= 0) {
        cout << endl;
        cout << "[错误] Dw^2 - Dp^2 = " << diff
             << " <= 0，根号下无意义！" << endl;
        cout << "要求: Dw > Dp (即 Dw^2 > Dp^2)" << endl;
        return 1;
    }

    double ratio = (Dw * Dw) / diff;
    double L0 = L2 * (sqrt(ratio) - 1);

    cout << fixed << setprecision(4);
    cout << endl;
    cout << "------ 计算结果 ------" << endl;
    cout << "Dw^2        = " << Dw * Dw << endl;
    cout << "Dp^2        = " << Dp * Dp << endl;
    cout << "Dw^2 - Dp^2 = " << diff << endl;
    cout << "Dw^2/(Dw^2-Dp^2) = " << ratio << endl;
    cout << "sqrt(...)   = " << sqrt(ratio) << endl;
    cout << "sqrt(...)-1 = " << sqrt(ratio) - 1 << endl;
    cout << "L0          = " << L0 << endl;

    return 0;
}
