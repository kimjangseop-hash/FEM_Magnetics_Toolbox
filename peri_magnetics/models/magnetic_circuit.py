import numpy as np

from peri_magnetics.models.bh_interpolation import BHInterpolator


class MagneticCircuit:
    """B-H 데이터 기반 비선형 자기회로 계산."""

    def __init__(self, bh_model=None):

        self.bh = (
            bh_model
            if bh_model is not None
            else BHInterpolator()
        )

        self.mu0 = 4 * np.pi * 1e-7

    def calculate_core_mmF(
        self,
        B,
        core_length,
        temperature=25
    ):
        """
        코어의 자기기전력 계산.

        B: 자속밀도 (T)
        core_length: 코어 평균 자로 길이 (m)
        temperature: 온도 (℃)

        return: 자기기전력 (A-turn)
        """

        if not np.isfinite(B) or B < 0:
            raise ValueError("B는 0 이상의 유한한 값이어야 합니다.")

        if not np.isfinite(core_length) or core_length <= 0:
            raise ValueError("코어 길이는 0보다 커야 합니다.")

        H = self.bh.get_H(B, temperature)

        return float(H * core_length)


    def calculate_gap_mmF(self, B, gap_length):
        """
        에어갭의 자기기전력 계산.

        B: 에어갭 자속밀도 (T)
        gap_length: 에어갭 길이 (m)

        return: 자기기전력 (A-turn)
        """

        if not np.isfinite(B) or B < 0:
            raise ValueError(
                "자속밀도 B는 0 이상의 유한한 값이어야 합니다."
            )

        if not np.isfinite(gap_length) or gap_length < 0:
            raise ValueError(
                "에어갭 길이는 0 이상의 유한한 값이어야 합니다."
            )

        H_gap = B / self.mu0

        return float(H_gap * gap_length)
    
    def calculate_total_mmF(
        self,
        B,
        core_length,
        gap_length,
        temperature=25
    ):
        """
        코어와 에어갭의 전체 자기기전력 계산.

        B: 자속밀도 (T)
        core_length: 코어 평균 자로 길이 (m)
        gap_length: 에어갭 길이 (m)
        temperature: 온도 (℃)

        return: 전체 자기기전력 (A-turn)
        """

        mmf_core = self.calculate_core_mmF(
            B,
            core_length,
            temperature
        )

        mmf_gap = self.calculate_gap_mmF(
            B,
            gap_length
        )

        return float(mmf_core + mmf_gap)
    
    def calculate_turns(self, total_mmf, current):
        """
        필요한 권선수 계산.

        total_mmf: 전체 자기기전력 (A-turn)
        current: 구동 전류 (A)

        return: 필요한 권선수
        """

        if not np.isfinite(total_mmf) or total_mmf < 0:
            raise ValueError(
                "전체 자기기전력은 0 이상의 유한한 값이어야 합니다."
            )

        if not np.isfinite(current) or current <= 0:
            raise ValueError(
                "전류는 0보다 큰 유한한 값이어야 합니다."
            )

        return float(total_mmf / current)

    def calculate_current(self, total_mmf, turns):
        """
        필요한 구동 전류 계산.

        total_mmf: 전체 자기기전력 (A-turn)
        turns: 권선수

        return: 구동 전류 (A)
        """

        if not np.isfinite(total_mmf) or total_mmf < 0:
            raise ValueError(
                "전체 자기기전력은 0 이상의 유한한 값이어야 합니다."
            )

        if not np.isfinite(turns) or turns <= 0:
            raise ValueError(
                "권선수는 0보다 큰 유한한 값이어야 합니다."
            )

        return float(total_mmf / turns)
    

    def solve_operating_point_with_areas(
        self,
        turns,
        current,
        core_length,
        gap_length,
        core_area,
        gap_area,
        temperature=25,
        tolerance=1e-6,
        max_iterations=100,
        use_fringing=False,
        core_width=None,
        core_depth=None
    ):
        """
        코어와 에어갭 단면적을 분리하여 비선형 자기회로 계산.

        use_fringing=True이면 에어갭 프린징 효과 적용.
        모든 길이: m
        모든 면적: m^2
        """

        if not np.isfinite(turns) or turns <= 0:
            raise ValueError("권선수는 0보다 커야 합니다.")

        if not np.isfinite(current) or current < 0:
            raise ValueError("전류는 0 이상의 유한한 값이어야 합니다.")

        if not np.isfinite(core_length) or core_length <= 0:
            raise ValueError("코어 길이는 0보다 커야 합니다.")

        if not np.isfinite(gap_length) or gap_length < 0:
            raise ValueError("에어갭 길이는 0 이상이어야 합니다.")

        if not np.isfinite(core_area) or core_area <= 0:
            raise ValueError("코어 단면적은 0보다 커야 합니다.")

        if not np.isfinite(gap_area) or gap_area <= 0:
            raise ValueError("에어갭 단면적은 0보다 커야 합니다.")

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("허용 오차는 0보다 커야 합니다.")

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("최대 반복 횟수는 양의 정수여야 합니다.")

        effective_gap_area = gap_area

        if use_fringing:
            if core_width is None or core_depth is None:
                raise ValueError(
                    "프린징 계산에는 core_width와 core_depth가 필요합니다."
                )

            effective_gap_area = (
                self.calculate_fringing_area_rectangular(
                    core_width,
                    core_depth,
                    gap_length
                )
            )

        target_mmf = turns * current

        H_limit = min(
            self.bh.curves[t]["H"][-1]
            for t in self.bh.temperatures
        )

        B_max = self.bh.get_B(
            H_limit,
            temperature
        )

        phi_low = 0.0
        phi_high = B_max * core_area

        B_gap_max = phi_high / effective_gap_area

        mmf_max = (
            self.calculate_core_mmF(
                B_max,
                core_length,
                temperature
            )
            + self.calculate_gap_mmF(
                B_gap_max,
                gap_length
            )
        )

        if target_mmf > mmf_max:
            raise ValueError(
                "B-H 데이터의 계산 가능 범위를 초과했습니다."
            )

        # B-H 데이터의 평탄 구간에서 MMF 불연속 검사
        flat_regions = self.bh.check_flat_regions(
            temperature=temperature
        )

        for regions in flat_regions.values():
            for region in regions:

                B_flat = float(region["B"])

                epsilon = max(
                    1e-10,
                    abs(B_flat) * 1e-8
                )

                B_before = B_flat - epsilon

                if B_before < 0:
                    continue

                B_gap_before = (
                    B_before * core_area / effective_gap_area
                )

                B_gap_flat = (
                    B_flat * core_area / effective_gap_area
                )

                mmf_before = (
                    self.calculate_core_mmF(
                        B_before,
                        core_length,
                        temperature
                    )
                    + self.calculate_gap_mmF(
                        B_gap_before,
                        gap_length
                    )
                )

                mmf_flat = (
                    self.calculate_core_mmF(
                        B_flat,
                        core_length,
                        temperature
                    )
                    + self.calculate_gap_mmF(
                        B_gap_flat,
                        gap_length
                    )
                )

                if mmf_before < target_mmf < mmf_flat:
                    raise ValueError(
                        f"B-H 평탄 구간의 MMF 불연속으로 "
                        f"계산할 수 없습니다. "
                        f"B={B_flat} T, "
                        f"MMF 범위={mmf_before:.6f}~"
                        f"{mmf_flat:.6f} A-turn, "
                        f"목표 MMF={target_mmf:.6f} A-turn"
                    )

        for iteration in range(1, max_iterations + 1):

            phi = (phi_low + phi_high) / 2

            B_core = phi / core_area
            B_gap = phi / effective_gap_area

            mmf_core = self.calculate_core_mmF(
                B_core,
                core_length,
                temperature
            )

            mmf_gap = self.calculate_gap_mmF(
                B_gap,
                gap_length
            )

            mmf_total = mmf_core + mmf_gap

            error = mmf_total - target_mmf

            if abs(error) <= tolerance:
                break

            if error > 0:
                phi_high = phi
            else:
                phi_low = phi

        else:
            raise RuntimeError(
                "자기회로 반복 계산이 수렴하지 않았습니다."
            )

        return {
            "flux": phi,
            "B": B_core,
            "B_core": B_core,
            "B_gap": B_gap,
            "H_core": mmf_core / core_length,
            "H_gap": B_gap / self.mu0,
            "mmf_core": mmf_core,
            "mmf_gap": mmf_gap,
            "mmf_total": mmf_total,
            "effective_gap_area": effective_gap_area,
            "iterations": iteration
        }

    def validate_operating_point(
        self,
        result,
        turns,
        current,
        core_area,
        gap_area,
        tolerance=1e-6
    ):
        """
        자기회로 계산 결과 자동 검증.
        """

        target_mmf = turns * current

        # 1. 코어와 에어갭 자속량 비교
        phi_core = result["B_core"] * core_area
        effective_gap_area = result.get(
            "effective_gap_area",
            gap_area
        )

        phi_gap = result["B_gap"] * effective_gap_area

        flux_error = abs(phi_core - phi_gap)

        # 2. 전체 MMF 오차
        mmf_error = abs(
            result["mmf_total"] - target_mmf
        )

        # 3. 계산 수렴 여부
        converged = mmf_error <= tolerance

        # 4. 자속량 일치 여부
        flux_valid = flux_error <= 1e-12

        # 5. B-H 데이터 범위 확인
        H_core = result["H_core"]

        H_limit = min(
            self.bh.curves[t]["H"][-1]
            for t in self.bh.temperatures
        )

        bh_range_valid = (
            0 <= H_core <= H_limit
        )

        return {
            "phi_core": phi_core,
            "phi_gap": phi_gap,
            "flux_error": flux_error,
            "flux_valid": flux_valid,
            "mmf_error": mmf_error,
            "converged": converged,
                        "bh_range_valid": bool(bh_range_valid),
            "overall_pass": bool(
                flux_valid
                and converged
                and bh_range_valid
            )
        }
    
    def calculate_fringing_area_rectangular(
        self,
        core_width,
        core_depth,
        gap_length
    ):
        """
        직사각형 코어의 에어갭 프린징 유효 면적 계산.

        모든 입력 단위: m
        return: m^2
        """
        values = [core_width, core_depth, gap_length]

        if not all(np.isfinite(value) for value in values):
            raise ValueError("모든 입력값은 유한한 숫자여야 합니다.")

        if core_width <= 0 or core_depth <= 0:
            raise ValueError("코어 폭과 깊이는 0보다 커야 합니다.")

        if gap_length < 0:
            raise ValueError("에어갭 길이는 0 이상이어야 합니다.")

        effective_area = (
            (core_width + gap_length)
            * (core_depth + gap_length)
        )

        return float(effective_area)

    
    def calculate_incremental_inductance(
        self,
        turns,
        current,
        core_length,
        gap_length,
        core_area,
        gap_area,
        temperature=25,
        delta_current=0.0001,
        use_fringing=False,
        core_width=None,
        core_depth=None,
        tolerance=1e-10,
        max_iterations=150
    ):
        """
        DC Bias 동작점 주변의 미분 인덕턴스 계산.

        delta_current: 전류 미분 간격 (A)
        return: 증분 인덕턴스 (H)
        """

        if current < 0:
            raise ValueError("전류는 0 이상이어야 합니다.")

        if delta_current <= 0:
            raise ValueError("전류 미분 간격은 0보다 커야 합니다.")

        def get_flux(test_current):

            result = self.solve_operating_point_with_areas(
                turns=turns,
                current=test_current,
                core_length=core_length,
                gap_length=gap_length,
                core_area=core_area,
                gap_area=gap_area,
                temperature=temperature,
                use_fringing=use_fringing,
                core_width=core_width,
                core_depth=core_depth,
                tolerance=tolerance,
                max_iterations=max_iterations
            )

            return result["flux"]

        if current >= delta_current:

            flux_plus = get_flux(current + delta_current)
            flux_minus = get_flux(current - delta_current)

            d_flux_d_current = (
                (flux_plus - flux_minus)
                / (2 * delta_current)
            )

        else:

            flux_base = get_flux(current)
            flux_plus = get_flux(current + delta_current)

            d_flux_d_current = (
                (flux_plus - flux_base)
                / delta_current
            )

        return float(turns * d_flux_d_current)

    
    def calculate_dc_bias_curve(
            self,
            turns,
            core_length,
            gap_length,
            core_area,
            gap_area,
            current_start=0.0,
            current_end=4.0,
            current_step=0.5,
            temperature=25,
            delta_current=0.0001,
            use_fringing=False,
            core_width=None,
            core_depth=None,
            tolerance=1e-10,
            max_iterations=150
    ):
        """
        DC Bias L-I Curve 데이터 생성.

        전류별 동작점, 자속량, 할선 및 미분 인덕턴스 계산.
        """

        if current_start < 0 or current_end < current_start:
            raise ValueError("전류 범위를 확인하세요.")

        if current_step <= 0:
            raise ValueError("전류 간격은 0보다 커야 합니다.")

        intervals = (current_end - current_start) / current_step

        if not np.isclose(intervals, round(intervals)):
            raise ValueError(
                "전류 범위는 전류 간격으로 나누어 떨어져야 합니다."
            )

        currents = np.linspace(
            current_start,
            current_end,
            int(round(intervals)) + 1
        )

        results = []

        for current in currents:

            operating_point = self.solve_operating_point_with_areas(
                turns=turns,
                current=float(current),
                core_length=core_length,
                gap_length=gap_length,
                core_area=core_area,
                gap_area=gap_area,
                temperature=temperature,
                use_fringing=use_fringing,
                core_width=core_width,
                core_depth=core_depth,
                tolerance=tolerance,
                max_iterations=max_iterations
            )

            flux = operating_point["flux"]

            if current > 0:
                L_secant = turns * flux / current
            else:
                L_secant = float("nan")

            L_diff = self.calculate_incremental_inductance(
                turns=turns,
                current=float(current),
                core_length=core_length,
                gap_length=gap_length,
                core_area=core_area,
                gap_area=gap_area,
                temperature=temperature,
                delta_current=delta_current,
                use_fringing=use_fringing,
                core_width=core_width,
                core_depth=core_depth,
                tolerance=tolerance,
                max_iterations=max_iterations
            )

            results.append({
                "current": float(current),
                "B_core": operating_point["B_core"],
                "B_gap": operating_point["B_gap"],
                "flux": flux,
                "L_secant": L_secant,
                "L_diff": L_diff
            })

        return results
