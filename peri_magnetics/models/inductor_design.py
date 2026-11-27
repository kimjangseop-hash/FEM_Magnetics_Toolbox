import math
from scipy.optimize import brentq

from peri_magnetics.models.magnetic_circuit import MagneticCircuit


class InductorDesign:

    def __init__(self, magnetic_circuit=None):
        self.mc = (
            magnetic_circuit
            if magnetic_circuit is not None
            else MagneticCircuit()
        )

    def calculate_inductance(
        self,
        turns,
        gap_length,
        core_length,
        core_area,
        gap_area,
        current=0.0,
        temperature=25,
        delta_current=0.0001,
        use_fringing=False,
        core_width=None,
        core_depth=None
    ):
        """
        턴수와 에어갭으로 인덕턴스를 계산한다.
        """

        # 입력값 검증
        values = {
            "turns": turns,
            "gap_length": gap_length,
            "core_length": core_length,
            "core_area": core_area,
            "gap_area": gap_area,
            "current": current,
            "temperature": temperature,
            "delta_current": delta_current
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}은 유효한 숫자여야 합니다.")

        if turns <= 0:
            raise ValueError("턴수는 0보다 커야 합니다.")

        if gap_length < 0:
            raise ValueError("에어갭은 0 이상이어야 합니다.")

        if core_length <= 0:
            raise ValueError("코어 길이는 0보다 커야 합니다.")

        if core_area <= 0 or gap_area <= 0:
            raise ValueError("코어 면적과 에어갭 면적은 0보다 커야 합니다.")

        if current < 0:
            raise ValueError("전류는 0 이상이어야 합니다.")

        if delta_current <= 0:
            raise ValueError("전류 변화량은 0보다 커야 합니다.")
        
        operating_point = self.mc.solve_operating_point_with_areas(
            turns=turns,
            current=current,
            core_length=core_length,
            gap_length=gap_length,
            core_area=core_area,
            gap_area=gap_area,
            temperature=temperature,
            use_fringing=use_fringing,
            core_width=core_width,
            core_depth=core_depth
        )

        L_diff = self.mc.calculate_incremental_inductance(
            turns=turns,
            current=current,
            core_length=core_length,
            gap_length=gap_length,
            core_area=core_area,
            gap_area=gap_area,
            temperature=temperature,
            delta_current=delta_current,
            use_fringing=use_fringing,
            core_width=core_width,
            core_depth=core_depth
        )

        flux = operating_point["flux"]

        L_secant = (
            turns * flux / current
            if current > 0
            else L_diff
        )

        return {
            "current": current,
            "flux": flux,
            "B_core": operating_point["B_core"],
            "B_gap": operating_point["B_gap"],
            "L_diff": L_diff,
            "L_secant": L_secant
        }
    
    def calculate_gap_length(
        self,
        target_inductance,
        turns,
        core_length,
        core_area,
        gap_area,
        current=0.0,
        temperature=25,
        delta_current=0.0001,
        use_fringing=False,
        core_width=None,
        core_depth=None,
        gap_min=0.00001,
        gap_max=0.01
    ):
        """
        목표 인덕턴스와 턴수로 필요한 에어갭을 역산한다.
        """

        if not math.isfinite(target_inductance) or target_inductance <= 0:
            raise ValueError("목표 인덕턴스는 0보다 큰 유효한 숫자여야 합니다.")

        if not math.isfinite(turns) or turns <= 0:
            raise ValueError("턴수는 0보다 커야 합니다.")

        def calculate_error(gap_length):
            result = self.calculate_inductance(
                turns=turns,
                gap_length=gap_length,
                core_length=core_length,
                core_area=core_area,
                gap_area=gap_area,
                current=current,
                temperature=temperature,
                delta_current=delta_current,
                use_fringing=use_fringing,
                core_width=core_width,
                core_depth=core_depth
            )
            return result["L_diff"] - target_inductance

        error_min = calculate_error(gap_min)
        error_max = calculate_error(gap_max)

        if error_min * error_max > 0:
            raise ValueError(
                "설정한 에어갭 범위에서 목표 인덕턴스를 찾을 수 없습니다."
            )

        gap_solution = brentq(
            calculate_error,
            gap_min,
            gap_max
        )

        # 역산된 에어갭으로 실제 인덕턴스 재계산
        final_result = self.calculate_inductance(
            turns=turns,
            gap_length=gap_solution,
            core_length=core_length,
            core_area=core_area,
            gap_area=gap_area,
            current=current,
            temperature=temperature,
            delta_current=delta_current,
            use_fringing=use_fringing,
            core_width=core_width,
            core_depth=core_depth
        )

        achieved_inductance = final_result["L_diff"]

        error_percent = (
            abs(achieved_inductance - target_inductance)
            / target_inductance * 100
        )

        return {
            "target_inductance": target_inductance,
            "turns": turns,
            "gap_length": gap_solution,
            "gap_length_mm": gap_solution * 1000,
            "achieved_inductance": achieved_inductance,
            "error_percent": error_percent
        }

    def calculate_turns(
        self,
        target_inductance,
        gap_length,
        core_length,
        core_area,
        gap_area,
        current=0.0,
        temperature=25,
        delta_current=0.0001,
        use_fringing=False,
        core_width=None,
        core_depth=None,
        turns_min=1,
        turns_max=500
    ):
        """
        목표 인덕턴스와 에어갭으로 필요한 턴수를 역산한다.
        """
        if not math.isfinite(target_inductance) or target_inductance <= 0:
            raise ValueError("목표 인덕턴스는 0보다 큰 유효한 숫자여야 합니다.")

        if not math.isfinite(gap_length) or gap_length < 0:
            raise ValueError("에어갭은 0 이상의 유효한 숫자여야 합니다.")

        if turns_min <= 0 or turns_max <= turns_min:
            raise ValueError("턴수 범위가 올바르지 않습니다.")

        def calculate_error(turns):
            result = self.calculate_inductance(
                turns=turns,
                gap_length=gap_length,
                core_length=core_length,
                core_area=core_area,
                gap_area=gap_area,
                current=current,
                temperature=temperature,
                delta_current=delta_current,
                use_fringing=use_fringing,
                core_width=core_width,
                core_depth=core_depth
            )

            return result["L_diff"] - target_inductance

        error_min = calculate_error(turns_min)
        error_max = calculate_error(turns_max)

        if error_min * error_max > 0:
            raise ValueError(
                "설정한 턴수 범위에서 목표 인덕턴스를 찾을 수 없습니다."
            )

        turns_solution = brentq(
            calculate_error,
            turns_min,
            turns_max
        )

        candidates = {
            max(turns_min, int(turns_solution)),
            min(turns_max, int(turns_solution) + 1)
        }

        best_turns = min(
            candidates,
            key=lambda n: abs(calculate_error(n))
        )

        # 최종 턴수로 실제 인덕턴스 재계산
        final_result = self.calculate_inductance(
            turns=best_turns,
            gap_length=gap_length,
            core_length=core_length,
            core_area=core_area,
            gap_area=gap_area,
            current=current,
            temperature=temperature,
            delta_current=delta_current,
            use_fringing=use_fringing,
            core_width=core_width,
            core_depth=core_depth
        )

        achieved_inductance = final_result["L_diff"]

        return {
            "target_inductance": target_inductance,
            "turns": best_turns,
            "gap_length": gap_length,
            "achieved_inductance": achieved_inductance,
            "error_percent": (
                abs(achieved_inductance - target_inductance)
                / target_inductance * 100
            )
        }