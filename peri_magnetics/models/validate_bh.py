import numpy as np
import pandas as pd

from peri_magnetics.models.bh_interpolation import BHInterpolator


# B-H 보간 모델 생성
bh = BHInterpolator()

# 원본 CSV 데이터 읽기
csv_path = bh.csv_path

df = pd.read_csv(csv_path, encoding="cp949")


# 25℃ 원본 데이터 재현 검증
H_25 = df["H 25℃ (A/m)"].to_numpy(dtype=float)
B_25 = df["B 25℃ (T)"].to_numpy(dtype=float)

B_25_calc = bh.get_B(H_25, 25)

error_25 = np.max(np.abs(B_25 - B_25_calc))


# 100℃ 원본 데이터 재현 검증
H_100 = df["H 100℃ (A/m)"].to_numpy(dtype=float)
B_100 = df["B 100℃ (T)"].to_numpy(dtype=float)

B_100_calc = bh.get_B(H_100, 100)

error_100 = np.max(np.abs(B_100 - B_100_calc))


# 결과 출력
print("===== B-H 원본 데이터 재현 검증 =====")
print(f"25℃ 최대 오차: {error_25:.12e} T")
print(f"100℃ 최대 오차: {error_100:.12e} T")

print("====================================")

if error_25 < 1e-10 and error_100 < 1e-10:
    print("검증 결과: PASS")
else:
    print("검증 결과: FAIL")
    # 온도 보간 검증
print("\n===== 온도 보간 검증 =====")

H_test = np.linspace(0, 80000, 300)

for temp in [50, 75]:

    B_25 = bh.get_B(H_test, 25)
    B_100 = bh.get_B(H_test, 100)

    weight = (temp - 25) / (100 - 25)

    B_expected = B_25 + weight * (B_100 - B_25)

    B_actual = bh.get_B(H_test, temp)

    error = np.max(np.abs(B_expected - B_actual))

    print(f"{temp}℃ 최대 오차: {error:.12e} T")

    if error < 1e-10:
        print(f"{temp}℃ 검증: PASS")
    else:
        print(f"{temp}℃ 검증: FAIL")
        # B-H 곡선 단조성 검증
print("\n===== B-H 곡선 단조성 검증 =====")

H_test = np.linspace(0, 80000, 300)

for temp in [25, 50, 75, 100]:

    B_values = bh.get_B(H_test, temp)

    is_monotonic = np.all(np.diff(B_values) >= 0)

    print(
        f"{temp}℃ 단조 증가 검증: "
        f"{'PASS' if is_monotonic else 'FAIL'}"
    )