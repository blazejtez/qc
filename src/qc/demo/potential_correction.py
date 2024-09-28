import cupy as cp
import numpy as np
import matplotlib.pyplot as plt



def truncated_potential(x, V0):
    r = np.abs(x)
    V = np.where(r >= 1/V0, (1/r) + V0, 0)
    return V

def coulomb_potential(x):
    r = np.abs(x)
    # Potencjał Coulomba bez modyfikacji, ale z unikaniem nieskończoności w x = 0
    V = np.where(r != 0, -1/r, 0)
    return V

def coulomb_with_zero_at_0(x):
    r = np.abs(x)
    # Potencjał Coulomba z 0 wpisanym w punkcie x = 0
    V = np.where(r > 1e-10, -1/r, 0)  # 1e-10 to wartość bliska 0, aby uniknąć dzielenia przez 0
    return V
N=100
L=5
# Ustawienia zakresu x oraz wartości V0
x = np.linspace(-L, L, N)
V0_values = [N/L,N/(2*L),N]   # Różne wartości V0

# Tworzymy wykresy
plt.figure(figsize=(10, 8))

# Niezmodyfikowany potencjał Coulomba
V_coulomb = coulomb_potential(x)
# plt.plot(x, V_coulomb, label='Niezmodyfikowany potencjał Coulomba', linestyle='--', color='black')

# Potencjał z wpisanym 0 w punkcie 0
V_coulomb_zero_at_0 = coulomb_with_zero_at_0(x)
# plt.plot(x, V_coulomb_zero_at_0, label='Potencjał Coulomba z 0 w x = 0', linestyle='-.', color='blue')

# Zmodyfikowany potencjał dla różnych wartości V0
for V0 in V0_values:
    V_truncated = truncated_potential(x, V0)
    plt.plot(x, V_truncated, label=f'Zmodyfikowany potencjał V0 = {V0}')

# Ustawienia wykresu
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Porównanie różnych potencjałów')
plt.axhline(min(V0_values), color='black',linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

