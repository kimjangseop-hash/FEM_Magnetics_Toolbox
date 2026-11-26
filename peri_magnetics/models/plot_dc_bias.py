
import matplotlib.pyplot as plt

from peri_magnetics.models.bh_interpolation import BHInterpolator
from peri_magnetics.models.magnetic_circuit import MagneticCircuit


# 자기회로 모델 생성
bh = BHInterpolator()
mc = MagneticCircuit(bh)


# DC Bias 해석 조건
dc_results = mc.calculate_dc_bias_curve(
    turns=100,
    core_length=0.02,
    gap_length=0.0005,
    core_area=25e-6,
    gap_area=25e-6,
    current_start=0.0,
    current_end=2.0,
    current_step=0.5,
    temperature=25,
    delta_current=0.0001,
    tolerance=1e-6,
    max_iterations=150
)


# 결과 데이터 추출
currents = [row["current"] for row in dc_results]
L_diff = [row["L_diff"] * 1e6 for row in dc_results]
B_core = [row["B_core"] for row in dc_results]


# DC Bias L-I 곡선
plt.figure(figsize=(10, 6))

plt.plot(
    currents,
    L_diff,
    marker="o",
    label="Differential Inductance"
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
