
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


class BHInterpolator:
    """제조사 B-H 데이터 기반 PCHIP 보간 및 미분 모델."""

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

            if not np.all(np.isfinite(H)) or not np.all(np.isfinite(B)):
                raise ValueError(
                    f"{temp}℃ 데이터에 유효하지 않은 값이 있습니다."
                )

            if np.any(np.diff(H) <= 0):
                raise ValueError(
                    f"{temp}℃ H 데이터는 엄격하게 증가해야 합니다."
                )

            if np.any(np.diff(B) < 0):
                raise ValueError(
                    f"{temp}℃ B 데이터는 단조 증가해야 합니다."
                )

    def _get_temperature_pair(self, temperature):
        """온도에 해당하는 하한 및 상한 데이터 온도 반환."""

        temperature = float(temperature)

        if not np.isfinite(temperature):
            raise ValueError("온도는 유효한 숫자여야 합니다.")

        if (
            temperature < self.temperatures[0]
            or temperature > self.temperatures[-1]
        ):
            raise ValueError("지원 온도 범위를 벗어났습니다.")

        lower_temp = max(
            t for t in self.temperatures if t <= temperature
        )

        upper_temp = min(
            t for t in self.temperatures if t >= temperature
        )

        return temperature, lower_temp, upper_temp

    def _get_pchip(self, temperature, H_input):
        """해당 온도의 PCHIP 보간 객체 생성."""

        curve = self.curves[temperature]

        H_data = curve["H"]
        B_data = curve["B"]

        if (
            np.any(H_input < H_data[0])
            or np.any(H_input > H_data[-1])
        ):
            raise ValueError(
                f"{temperature}℃ H 데이터 범위를 벗어났습니다. "
                f"허용 범위: {H_data[0]} ~ {H_data[-1]} A/m"
            )

        return PchipInterpolator(
            H_data,
            B_data,
            extrapolate=False
        )

    def get_B(self, H, temperature):
        """
        PCHIP 보간을 이용한 자속밀도 B 계산.

        H: A/m
        temperature: ℃
        return: T
        """

        H_input = np.asarray(H, dtype=float)

        if not np.all(np.isfinite(H_input)):
            raise ValueError("H는 유효한 숫자여야 합니다.")

        temperature, lower_temp, upper_temp = (
            self._get_temperature_pair(temperature)
        )

        pchip_lower = self._get_pchip(lower_temp, H_input)
        B_lower = pchip_lower(H_input)

        if lower_temp == upper_temp:
            result = B_lower

        else:
            pchip_upper = self._get_pchip(upper_temp, H_input)
            B_upper = pchip_upper(H_input)

            weight = (
                (temperature - lower_temp)
                / (upper_temp - lower_temp)
            )

            result = B_lower + weight * (B_upper - B_lower)

        if np.ndim(H) == 0:
            return float(result)

        return result

    def get_dBdH(self, H, temperature, window_points=None):
        """
        PCHIP 보간 함수의 해석적 미분으로 dB/dH 계산.

        H: A/m
        temperature: ℃
        window_points: 기존 호출 방식과의 호환성을 위한 인자
        return: H/m
        """

        H_input = np.asarray(H, dtype=float)

        if not np.all(np.isfinite(H_input)):
            raise ValueError("H는 유효한 숫자여야 합니다.")

        temperature, lower_temp, upper_temp = (
            self._get_temperature_pair(temperature)
        )

        pchip_lower = self._get_pchip(lower_temp, H_input)

        slope_lower = pchip_lower.derivative()(H_input)

        if lower_temp == upper_temp:
            result = slope_lower

        else:
            pchip_upper = self._get_pchip(upper_temp, H_input)

            slope_upper = (
                pchip_upper.derivative()(H_input)
            )

            weight = (
                (temperature - lower_temp)
                / (upper_temp - lower_temp)
            )

            result = (
                slope_lower
                + weight * (slope_upper - slope_lower)
            )

        if np.ndim(H) == 0:
            return float(result)

        return result

    def get_mu_diff(self, H, temperature, window_points=None):
        """
        상대 미분투자율 계산.

        H: A/m
        temperature: ℃
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

        if not np.all(np.isfinite(H_input)):
            raise ValueError("H는 유효한 숫자여야 합니다.")

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
    

    
    def get_H(self, B, temperature):
        """
        B값으로부터 H를 수치적으로 역산한다.

        평탄 구간에서는 가장 큰 H값을 선택한다.
        """

        if not np.isfinite(B) or B < 0:
            raise ValueError("B는 0 이상의 유효한 숫자여야 합니다.")

        temperature, lower_temp, upper_temp = (
            self._get_temperature_pair(temperature)
        )

        # get_B에서 공통으로 사용할 수 있는 H 범위
        if lower_temp == upper_temp:
            H_min = self.curves[lower_temp]["H"][0]
            H_max = self.curves[lower_temp]["H"][-1]
        else:
            H_min = max(
                self.curves[lower_temp]["H"][0],
                self.curves[upper_temp]["H"][0]
            )
            H_max = min(
                self.curves[lower_temp]["H"][-1],
                self.curves[upper_temp]["H"][-1]
            )

        B_min = self.get_B(H_min, temperature)
        B_max = self.get_B(H_max, temperature)

        if B < B_min or B > B_max:
            raise ValueError(
                f"입력 B={B} T가 데이터 범위를 벗어났습니다. "
                f"유효 범위: {B_min} ~ {B_max} T"
            )

        # Monotonic bisection: find the rightmost H
        lower_H = H_min
        upper_H = H_max

        for _ in range(100):
            middle_H = (lower_H + upper_H) / 2.0

            B_calculated = self.get_B(
                middle_H,
                temperature
            )

            if B_calculated <= B:
                lower_H = middle_H
            else:
                upper_H = middle_H

        return float((lower_H + upper_H) / 2.0)

    def check_flat_regions(self, temperature=25):
        """
        B-H 데이터의 평탄 구간을 진단한다.
        """
        temperature, lower_temp, upper_temp = (
            self._get_temperature_pair(temperature)
        )

        def find_flat_regions(temp):
            B_data = self.curves[temp]["B"]
            H_data = self.curves[temp]["H"]

            delta_B = np.diff(B_data)
            flat_indices = np.where(delta_B == 0)[0]

            regions = []

            if len(flat_indices) == 0:
                return regions
            start = flat_indices[0]
            previous = flat_indices[0]

            for index in flat_indices[1:]:
                if index != previous + 1:
                    regions.append({
                        "B": float(B_data[start]),
                        "H_start": float(H_data[start]),
                        "H_end": float(H_data[previous + 1]),
                        "data_count": int(previous + 2 - start)
                    })
                    start = index

                previous = index

            regions.append({
                "B": float(B_data[start]),
                "H_start": float(H_data[start]),
                "H_end": float(H_data[previous + 1]),
                "data_count": previous + 2 - start
            })

            return regions
        return {
            lower_temp: find_flat_regions(lower_temp),
            **(
                {upper_temp: find_flat_regions(upper_temp)}
                if upper_temp != lower_temp else {}
            )
        }