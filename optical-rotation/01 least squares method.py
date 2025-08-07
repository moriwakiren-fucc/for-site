import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. フィッティングする非線形関数を定義
def nonlinear_function(x, a, b, c, d):
    """
    フィッティングに使用する非線形関数を定義します。
    """
    return a * np.sin(b * (x + c)) + d

# 2. 残差平方和を計算する関数を定義
def residual_sum_of_squares(params, x_data, y_data):
    """
    残差平方和（Residual Sum of Squares）を計算する関数。
    この関数をminimizeが最小化します。
    """
    # paramsから各パラメータa, b, c, dを取り出す
    a, b, c, d = params
    # 予測値を計算
    y_predicted = nonlinear_function(x_data, a, b, c, d)
    # 残差（実測値と予測値の差）の2乗の和を返す
    return np.sum((y_data - y_predicted) ** 2)

# 3. データ
x_deg = np.array([90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90])
y_data = np.array([1825, 1908, 1965, 1990, 1992, 1968, 1922, 1851, 1768, 1650, 1520, 1396, 1252, 1090, 950, 799, 644, 518, 410, 322.3, 260.4, 216.9, 206.5, 221.1, 269.3, 331.6, 427, 544, 676, 822, 981, 1146, 1297, 1455, 1585, 1691, 1821])
x_data = np.radians(x_deg)

# 4. パラメータの初期値を設定
# minimizeは初期値から最適な解を探索するため、適切な初期値は重要です。
initial_params = [900, -1.0, 1.0, 1000]

# 5. minimizeを実行して、残差平方和が最小になるパラメータを探す
# args=(x_data, y_data) で、残差平方和を計算する関数に追加の引数を渡す
result = minimize(residual_sum_of_squares, initial_params, args=(x_data, y_data), method='L-BFGS-B')

# 6. 最適化されたパラメータを取得
# result.x には最適化されたパラメータの配列が格納されています。
a_fit, b_fit, c_fit, d_fit = result.x

dydx = np.gradient(y_data,x_data)
minimum = min(np.degrees(x_data[1:][dydx[1:] * dydx[:-1] < 0]))
x_divided = np.linspace(minimum - 5, minimum + 5, 100)
y_divided = a_fit * np.sin(b_fit * (x_divided - c_fit) + d_fit)
min_index = np.argmin(y_divided)
x_min = x_divided[min_index]
y = a_fit * np.sin(b_fit * (x_data - c_fit) + d_fit)
squares_sum = np.sum((y - y_data)**2)

print(f"推定されたパラメータ: a={a_fit:.5f}, b={b_fit:.5f}, c={c_fit:.5f}, d={d_fit:.5f}")
print(f"極小値 = {x_min:.5f}")
print(f"残差平方和 = {squares_sum:.5f}")

# 7. 結果をプロット
y_fit = nonlinear_function(x_data, a_fit, b_fit, c_fit, d_fit)

plt.figure(figsize=(8, 6))
plt.scatter(x_deg, y_data, label='original data')
plt.plot(x_deg, y_fit, color='red', label='fitted curve')
plt.title('CURVE')
plt.xlabel('Degree')
plt.ylabel('Illuminance')
plt.legend()
plt.grid(True)
plt.show()
