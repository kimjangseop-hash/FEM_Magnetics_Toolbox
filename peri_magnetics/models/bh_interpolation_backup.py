
from pathlib import Path

import numpy as np
import pandas as pd


class BHInterpolator:
    """제조사 B-H 데이터 기반 선형 보간 및 미분 모델."""

    def __init__(self, csv_path=None):

        if csv_path is None:
            csv_path = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "core_bh_data.csv"
            )

        self.csv_path = Path(csv_path)

        df = pd.read_csv(
            self.csv_path,
            encoding="cp949"
        )

        self.curves = {
            25.0: {
                "H": df["H 25℃ (A/m)"].to_numpy(dtype=float),
                "B": df["B 25℃ (T)"].to_numpy(dtype=float),
            },
            100.0: {
                "H": df["H 100℃ (A/m)"].to_numpy(dtype=float),
                "B": df["B 100℃ (T)"].to_numpy(dtype=float),
            },
        }

        self.temperatures = sorted(self.curves.keys())

        # B-H 데이터 유효성 검사
        for temp, curve in self.curves.items():

            H = curve["H"]
            B = curve["B"]

            if len(H) != len(B) or len(H) < 2:
                raise ValueError(
                    f"{temp}℃ 데이터 길이가 올바르지 않습니다."
                )

            if np.any(np.diff(H) <= 0):
                raise ValueError(
                    f"{temp}℃ H 데이터는 증가해야 합니다."
                )

            if np.any(np.diff(B) < 0):
                raise ValueError(
                    f"{temp}℃ B 데이터는 단조 증가해야 합니다."
                )

    def get_B(self, H, temperature):
        """
        자계 강도 H와 온도에 해당하는 자속 밀도 B 계산.

        H: A/m
        temperature: ℃
        return: T
        """

        temperature = float(temperature)
        H_input = np.asarray(H, dtype=float)

        if temperature < self.temperatures[0]:
            raise ValueError("지원 온도 범위를 벗어났습니다.")

        if temperature > self.temperatures[-1]:
            raise ValueError("지원 온도 범위를 벗어났습니다.")

        lower_temp = max(
            t for t in self.temperatures if t <= temperature
        )

        upper_temp = min(
            t for t in self.temperatures if t >= temperature
        )

        def interpolate_at_temperature(temp):

            curve = self.curves[temp]

            H_data = curve["H"]
            B_data = curve["B"]

            if (
                np.any(H_input < H_data[0])
                or np.any(H_input > H_data[-1])
            ):
                raise ValueError(
                    f"{temp}℃ H 데이터 범위를 벗어났습니다. "
                    f"허용 범위: {H_data[0]} ~ {H_data[-1]} A/m"
                )

            return np.interp(
                H_input,
                H_data,
                B_data
            )

        B_lower = interpolate_at_temperature(lower_temp)

        if lower_temp == upper_temp:

            result = B_lower

        else:

            B_upper = interpolate_at_temperature(upper_temp)

            weight = (
                (temperature - lower_temp)
                / (upper_temp - lower_temp)
            )

            result = B_lower + weight * (B_upper - B_lower)

        if np.ndim(H) == 0:
            return float(result)

        return result

    def get_dBdH(self, H, temperature, window_points=7):
        """
        주변 데이터의 국부 선형 회귀를 이용한 dB/dH 추정.

        H: A/m
        temperature: ℃
        window_points: 회귀에 사용할 주변 데이터 개수
        return: H/m
        """

        temperature = float(temperature)
        H_input = np.asarray(H, dtype=float)

        if (
            temperature < self.temperatures[0]
            or temperature > self.temperatures[-1]
        ):
            raise ValueError("지원 온도 범위를 벗어났습니다.")

        if window_points < 3 or window_points % 2 == 0:
            raise ValueError(
                "window_points는 3 이상의 홀수여야 합니다."
            )

        lower_temp = max(
            t for t in self.temperatures if t <= temperature
        )

        upper_temp = min(
            t for t in self.temperatures if t >= temperature
        )

        def get_local_slope(temp):

            curve = self.curves[temp]

            H_data = curve["H"]
            B_data = curve["B"]

            if (
                np.any(H_input < H_data[0])
                or np.any(H_input > H_data[-1])
            ):
                raise ValueError(
                    f"{temp}℃ H 데이터 범위를 벗어났습니다."
                )

            original_shape = H_input.shape
            H_values = np.atleast_1d(H_input).ravel()

            slopes = []

            for h in H_values:

                nearest = np.argsort(np.abs(H_data - h))
                indices = np.sort(nearest[:window_points])

                x = H_data[indices] - h
                y = B_data[indices]

                slope = np.polyfit(x, y, 1)[0]

                slopes.append(slope)

            result = np.array(slopes).reshape(original_shape)

            if H_input.ndim == 0:
                return float(result)

            return result

        slope_lower = get_local_slope(lower_temp)

        if lower_temp == upper_temp:
            return slope_lower

        slope_upper = get_local_slope(upper_temp)

        weight = (
            (temperature - lower_temp)
            / (upper_temp - lower_temp)
        )

        return slope_lower + weight * (slope_upper - slope_lower)
    def get_mu_diff(self, H, temperature, window_points=7):
        """
        상대 미분투자율 계산.

        H: A/m
        temperature: ℃
        window_points: 미분 계산에 사용할 데이터 개수
        return: 상대 미분투자율 (무차원)
        """

        mu0 = 4 * np.pi * 1e-7

        dBdH = self.get_dBdH(
            H,
            temperature,
            window_points=window_points
        )

        return dBdH / mu0
    def get_mu_secant(self, H, temperature):
        """
        상대 종투자율 계산.
        H: A/m
        temperature: ℃
        return: 상대 종투자율 (무차원)
        """

        mu0 = 4 * np.pi * 1e-7
        H_input = np.asarray(H, dtype=float)

        if np.any(H_input < 0):
            raise ValueError("H는 0 이상의 값이어야 합니다.")

        if np.any(H_input == 0):
            raise ValueError(
                "H=0에서는 종투자율을 계산할 수 없습니다."
            )

        B = self.get_B(H, temperature)
        result = B / (mu0 * H_input)

        if np.ndim(H) == 0:
            return float(result)

        return result   