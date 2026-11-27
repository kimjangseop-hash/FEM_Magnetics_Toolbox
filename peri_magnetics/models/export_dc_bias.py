
import csv
from pathlib import Path


def export_dc_bias_csv(results, filename="dc_bias_results.csv"):
    """
    DC Bias 계산 결과를 CSV 파일로 저장.

    정상 계산과 오류 계산 결과를 모두 저장한다.
    """

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / filename
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "current",
        "B_core",
        "B_gap",
        "flux",
        "L_secant",
        "L_diff",
        "L_theory",
        "validation_error_percent",
        "validation_status",
        "status",
        "error"
    ]

    with open(
        output_path,
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path
