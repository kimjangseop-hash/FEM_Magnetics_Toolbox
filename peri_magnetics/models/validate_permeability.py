import numpy as np
import pandas as pd

from peri_magnetics.models.bh_interpolation import BHInterpolator


bh = BHInterpolator()

df = pd.read_csv(bh.csv_path, encoding="cp949")

mu0 = 4 * np.pi * 1e-7

# 비교할 자계 강도
H_test = [100, 300, 500, 1000]


for temperature in [25, 100]:

    if temperature == 25:
        H_data = df["H 25℃ (A/m)"].to_numpy(dtype=float)
        B_data = df["B 25℃ (T)"].to_numpy(dtype=float)
    else:
        H_data = df["H 100℃ (A/m)"].to_numpy(dtype=float)
        B_data = df["B 100℃ (T)"].to_numpy(dtype=float)

    print(f"\n===== {temperature}℃ 미분투자율 비교 =====")
    print("H(A/m) | 7점 회귀 | 인접 구간 기울기")

    for h in H_test:

        # 현재 모델의 7점 회귀 결과
        slope_regression = bh.get_dBdH(
            h,
            temperature,
            window_points=7
        )

        # 가장 가까운 원본 데이터 위치
        index = np.searchsorted(H_data, h)

        index = max(1, min(index, len(H_data) - 1))

        # 인접한 두 원본 데이터의 기울기
        slope_segment = (
            (B_data[index] - B_data[index - 1])
            / (H_data[index] - H_data[index - 1])
        )

        mu_regression = slope_regression / mu0
        mu_segment = slope_segment / mu0

        print(
            f"{h:6.1f} | "
            f"{mu_regression:10.3f} | "
            f"{mu_segment:10.3f}"
        )
        # H=1000 A/m 주변 원본 데이터 확인
print("\n===== H=1000 주변 원본 데이터 =====")

for temperature in [25, 100]:

    if temperature == 25:
        H_data = df["H 25℃ (A/m)"].to_numpy(dtype=float)
        B_data = df["B 25℃ (T)"].to_numpy(dtype=float)
    else:
        H_data = df["H 100℃ (A/m)"].to_numpy(dtype=float)
        B_data = df["B 100℃ (T)"].to_numpy(dtype=float)

    index = np.searchsorted(H_data, 1000)

    print(f"\n{temperature}℃")

    for i in range(max(0, index - 3), min(len(H_data), index + 4)):
        print(f"H = {H_data[i]:10.3f}, B = {B_data[i]:.6f}")
        # 회귀 구간 크기별 미분투자율 비교
print("\n===== 회귀 구간 크기 비교 =====")

for temperature in [25, 100]:

    print(f"\n{temperature}℃")

    for h in [500, 1000]:

        print(f"\nH = {h} A/m")

        for window in [5, 7, 9, 11]:

            slope = bh.get_dBdH(
                h,
                temperature,
                window_points=window
            )

            mu_diff = slope / mu0

            print(
                f"{window}점 회귀: "
                f"{mu_diff:.4f}"
            )
            # 회귀 구간의 실제 H 범위 확인
print("\n===== 회귀 구간 H 범위 확인 =====")

for temperature in [25, 100]:

    if temperature == 25:
        H_data = df["H 25℃ (A/m)"].to_numpy(dtype=float)
    else:
        H_data = df["H 100℃ (A/m)"].to_numpy(dtype=float)

    print(f"\n{temperature}℃")

    for h in [500, 1000]:

        nearest = np.argsort(np.abs(H_data - h))

        print(f"\n기준 H = {h} A/m")

        for window in [5, 7, 9, 11]:

            indices = np.sort(nearest[:window])

            H_min = H_data[indices[0]]
            H_max = H_data[indices[-1]]

            print(
                f"{window}점: "
                f"{H_min:.1f} ~ {H_max:.1f} A/m "
                f"(범위 {H_max - H_min:.1f} A/m)"
            )
            # H 범위 기반 회귀 비교
print("\n===== H 범위 기반 회귀 비교 =====")

for temperature in [25, 100]:

    if temperature == 25:
        H_data = df["H 25℃ (A/m)"].to_numpy(dtype=float)
        B_data = df["B 25℃ (T)"].to_numpy(dtype=float)
    else:
        H_data = df["H 100℃ (A/m)"].to_numpy(dtype=float)
        B_data = df["B 100℃ (T)"].to_numpy(dtype=float)

    print(f"\n{temperature}℃")

    for h in [500, 1000]:

        print(f"\nH = {h} A/m")

        for delta in [25, 50, 100, 150]:

            mask = (
                (H_data >= h - delta)
                & (H_data <= h + delta)
            )

            x = H_data[mask] - h
            y = B_data[mask]

            if len(x) < 3:
                print(
                    f"±{delta} A/m: "
                    f"데이터 부족 ({len(x)}개)"
                )
                continue

            slope = np.polyfit(x, y, 1)[0]

            mu_diff = slope / mu0

            print(
                f"±{delta} A/m "
                f"({len(x)}개): "
                f"{mu_diff:.4f}"
            )