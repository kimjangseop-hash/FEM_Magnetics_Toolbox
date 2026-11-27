
import matplotlib.pyplot as plt

from peri_magnetics.models.bh_interpolation import BHInterpolator
from peri_magnetics.models.magnetic_circuit import MagneticCircuit


# 자기회로 모델 생성
bh = BHInterpolator()
mc = MagneticCircuit(bh)

# DC Bias 해석 온도 목록
temperatures = [25, 50, 75, 100]

# 온도별 DC Bias 해석
temperature_results = {}

for temp in temperatures:
    print(f"\n===== {temp}℃ DC Bias 해석 =====")

    dc_results = mc.calculate_dc_bias_curve(
        turns=100,
        core_length=0.02,
        gap_length=0.0005,
        core_area=25e-6,
        gap_area=25e-6,
        current_start=0.0,
        current_end=2.0,
        current_step=0.5,
        temperature=temp,
        delta_current=0.0001,
        tolerance=1e-6,
        max_iterations=150
    )

    temperature_results[temp] = dc_results

# 온도별 DC Bias 결과 추출
temperature_data = {}

for temp, results in temperature_results.items():
    temperature_data[temp] = [
        row for row in results
        if row["status"] == "success"
    ]

# 결과 데이터 추출
valid_results = [
    row for row in dc_results
    if row["status"] == "success"
]

currents = [row["current"] for row in valid_results]

L_diff = [
    row["L_diff"] * 1e6
    for row in valid_results
]

L_theory = [
    row["L_theory"] * 1e6
    for row in valid_results
]

B_core = [row["B_core"] for row in valid_results]
validation_error = [
    row["validation_error_percent"]
    for row in valid_results
]

# DC Bias L-I 곡선
plt.figure(figsize=(10, 6))

plt.plot(
    currents,
    L_diff,
    marker="o",
    label="Differential Inductance"
)
plt.plot(
    currents,
    L_theory,
    marker="s",
    linestyle="--",
    label="Theoretical Differential Inductance"
)
plt.xlabel("DC Bias Current (A)")
plt.ylabel("Differential Inductance (uH)")
plt.title("DC Bias L-I Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "peri_magnetics/results/dc_bias_LI.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# DC Bias B-I 곡선
plt.figure(figsize=(10, 6))

plt.plot(
    currents,
    B_core,
    marker="o",
    label="Core Flux Density"
)

plt.xlabel("DC Bias Current (A)")
plt.ylabel("Core Flux Density (T)")
plt.title("DC Bias B-I Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "peri_magnetics/results/dc_bias_BI.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# DC Bias Ldiff 검증 오차율 곡선
plt.figure(figsize=(10, 6))

plt.plot(
    currents,
    validation_error,
    marker="o",
    label="Ldiff Validation Error"
)

plt.axhline(
    y=1.0,
    linestyle="--",
    label="Tolerance (1.0%)"
)

plt.xlabel("DC Bias Current (A)")
plt.ylabel("Relative Error (%)")
plt.title("DC Bias Ldiff Validation Error")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "peri_magnetics/results/dc_bias_validation_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# 온도별 DC Bias L-I 비교 그래프
plt.figure(figsize=(10, 6))

for temp, results in temperature_data.items():
    currents_temp = [
        row["current"] for row in results
    ]

    L_diff_temp = [
        row["L_diff"] * 1e6 for row in results
    ]

    plt.plot(
        currents_temp,
        L_diff_temp,
        marker="o",
        label=f"{temp}℃"
    )

plt.xlabel("DC Bias Current (A)")
plt.ylabel("Differential Inductance (uH)")
plt.title("Temperature-dependent DC Bias L-I Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "peri_magnetics/results/temperature_dc_bias_LI.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# 온도별 DC Bias B-I 비교 그래프
plt.figure(figsize=(10, 6))

for temp, results in temperature_data.items():
    currents_temp = [
        row["current"] for row in results
    ]

    B_core_temp = [
        row["B_core"] for row in results
    ]

    plt.plot(
        currents_temp,
        B_core_temp,
        marker="o",
        label=f"{temp}℃"
    )

plt.xlabel("DC Bias Current (A)")
plt.ylabel("Core Flux Density (T)")
plt.title("Temperature-dependent DC Bias B-I Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "peri_magnetics/results/temperature_dc_bias_BI.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()