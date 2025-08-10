import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. フィッティングする非線形関数を定義
def nonlinear_function(x1, a1, b1, c1, d1):
    """
    フィッティングに使用する非線形関数を定義します。
    """
    return a1 * np.sin(b1 * (x1 + c1)) + d1
def nonlinear_function(x2, a2, b2, c2, d2):
    """
    フィッティングに使用する非線形関数を定義します。
    """
    return a2 * np.sin(b2 * (x2 + c2)) + d2

# 2. 残差平方和を計算する関数を定義
def residual_sum_of_squares(params1, x1_data, y1_data):
    """
    残差平方和（Residual Sum of Squares）を計算する関数。
    この関数をminimizeが最小化します。
    """
    # params1から各パラメータa1, b1, c1, d1を取り出す
    a1, b1, c1, d1 = params1
    # 予測値を計算
    y1_predicted = nonlinear_function(x1_data, a1, b1, c1, d1)
    # 残差（実測値と予測値の差）の2乗の和を返す
    return np.sum((y1_data - y1_predicted) ** 2)
def residual_sum_of_squares(params2, x2_data, y2_data):
    """
    残差平方和（Residual Sum of Squares）を計算する関数。
    この関数をminimizeが最小化します。
    """
    # params2から各パラメータa2, b2, c2, d2を取り出す
    a2, b2, c2, d2 = params2
    # 予測値を計算
    y2_predicted = nonlinear_function(x2_data, a2, b2, c2, d2)
    # 残差（実測値と予測値の差）の2乗の和を返す
    return np.sum((y2_data - y2_predicted) ** 2)

# 3. データ
x1_deg = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90]
y1_data = [2063, 2114, 2131, 2129, 2091, 2035, 1944, 1833, 1703, 1574, 1423, 1251, 1087, 939, 771, 617, 494, 392, 311.7, 256.1, 227.7, 229.1, 263.1, 313.9, 398.5, 509, 649, 782, 920, 1088, 1247, 1409, 1569, 1737, 1867, 1984, 2066]
x2_deg = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90]
y2_data = [2063, 2114, 2131, 2129, 2091, 2035, 1944, 1833, 1703, 1574, 1423, 1251, 1087, 939, 771, 617, 494, 392, 311.7, 256.1, 227.7, 229.1, 263.1, 313.9, 398.5, 509, 649, 782, 920, 1088, 1247, 1409, 1569, 1737, 1867, 1984, 2066]
x1_data = np.radians(x1_deg)
x2_data = np.radians(x2_deg)

# 4. パラメータの初期値を設定
# minimizeは初期値から最適な解を探索するため、適切な初期値は重要です。
initial_params1 = [900, -2.0, 1.0, 1000]
initial_params2 = [900, -2.0, 1.0, 1000]

# 5. minimizeを実行して、残差平方和が最小になるパラメータを探す
# args=(x1_data, y1_data) で、残差平方和を計算する関数に追加の引数を渡す
result1 = minimize(residual_sum_of_squares, initial_params1, args=(x1_data, y1_data), method='L-BFGS-B')
result2 = minimize(residual_sum_of_squares, initial_params1, args=(x1_data, y1_data), method='L-BFGS-B')


# 6. 最適化されたパラメータを取得
# result.x1 には最適化されたパラメータの配列が格納されています。
a1_fit, b1_fit, c1_fit, d1_fit = result1.x1
a1_fit, b1_fit, c1_fit, d1_fit = result1.x1

dy1dx1 = np.gradient(y1_data,x1_data)
minimum = min(np.degrees(x1_data[1:][dy1dx1[1:] * dy1dx1[:-1] < 0]))
x1_divided = np.linspace(minimum - 5, minimum + 5, 100)
y1_divided = a1_fit * np.sin(b1_fit * (x1_divided - c1_fit) + d1_fit)
min_index1 = np.argmin(y1_divided)
x1_min = x1_divided[min_index1]
y1_min = y1_divided[min_index1]
y1 = a1_fit * np.sin(b1_fit * (x1_data + c1_fit) + d1_fit)
squares_sum = np.sum((y1 - y1_data)**2)
x1_values = list(range(-90, 91, 1))
y1_values = a1_fit * np.sin(b1_fit * (np.radians(x1_values) + c1_fit)) + d1_fit
y1_output = "\n".join(map(str, y1_values))

import pyperclip

# py1perclip.copy1()で文字列をクリップボードにコピー
pyperclip.copy1(y1_output)

print(f"数式 y1=a1･sinb1(x1+c1)+d1 における推定されたパラメーター: a1={a1_fit:.5f} b1={b1_fit:.5f} c1={c1_fit:.5f} d1={d1_fit:.5f}")
print(f"極小値(照度){y1_min:.5f}")
print(f"極小値をとる角度 = {x1_min:.5f}")
print(f"残差平方和 = {squares_sum:.5f}")

# 7. 結果をプロット
y1_fit = nonlinear_function(x1_data, a1_fit, b1_fit, c1_fit, d1_fit)

plt.figure(figsize=(8, 6))
plt.scatter(x1_deg, y1_data, label='original data')
plt.plot(x1_deg, y1_fit, color='red', label='fitted curve')
plt.title('CURVE')
plt.x1label('Degree')
plt.y1label('Illuminance')
plt.legend()
plt.grid(True)
plt.show()
