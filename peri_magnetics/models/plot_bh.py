
import numpy as np
import matplotlib.pyplot as plt

from peri_magnetics.models.bh_interpolation import BHInterpolator


bh = BHInterpolator()

H_values = np.linspace(1, 2000, 300)

temperatures = [25, 50, 75, 100]

# B-H 곡선 비교
plt.figure(figsize=(10, 6))

for temp in temperatures:
    B_values = bh.get_B(H_values, temp)

    plt.plot(
        H_values,
        B_values,
        label=f"{temp} C"
    )

plt.xlabel("Magnetic Field Strength H (A/m)")
plt.ylabel("Flux Density B (T)")
plt.title("Temperature Dependent B-H Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 상대 미분투자율 비교
plt.figure(figsize=(10, 6))

for temp in temperatures:
    mu_diff = bh.get_mu_diff(H_values, temp)

    plt.plot(
        H_values,
        mu_diff,
        label=f"{temp} C"
    )

plt.xlabel("Magnetic Field Strength H (A/m)")
plt.ylabel("Relative Differential Permeability")
plt.title("Temperature Dependent Differential Permeability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

