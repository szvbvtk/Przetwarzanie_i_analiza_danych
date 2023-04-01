import numpy as np

# Zadanie 1
macierz = np.random.randint(0, 101, size=(10, 5))
slad = np.trace(macierz)
przekatna = np.diag(macierz)
print(f"Zadanie 1.\nSuma głównej przekątnej: {slad}\nWartości głównej przekątnej: {przekatna}")

# Zadanie 2

# Zadanie 3
C = np.random.randint(1, 101, size=(2, 5)).reshape((-1, 5))
D = np.random.randint(1, 101, size=(2, 5)).reshape((-1, 5))
print(f"\nZadanie 3.\n {C + D}")

# Zadanie 4
E = np.array([[1, 2, 3, 4, 5], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])
F = np.array([[1, 2, 3, 4], [3, 3, 3, 3], [1, 2, 2, 3], [1, 2, 0, 2], [0, 1, 2, 3]])
F = np.transpose(F)
G = E + F
print(f"\nZadanie 4.\nMacierz1 + macierz2 = \n{G}")

# Zadanie 5
print(f"\nZadanie 5.\n{E[:, 2]} * {F[:, 3]} = {E[:, 2] * F[:, 3]}")

# Zadanie 6
H = np.random.normal(size=(3, 3))
I = np.random.uniform(size=(3, 3))

print(f"\nZadanie 6.\n Rozkład normalny: średnia = {np.mean(H)}, suma = {np.sum(H)}, odchylenie standardowe = {np.std(H)}, wariancja = {np.var(H)}, minimum = {np.min(H)}, maksimum = {np.max(H)}"
      f"\nRozkład jednostajny: średnia = {np.mean(I)}, suma = {np.sum(I)}, odchylenie standardowe = {np.std(I)}, wariancja = {np.var(I)}, minimum = {np.min(I)}, maksimum = {np.max(I)}")

# Zadanie 7
A = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
print(f"\nZadanie 7.\n A * B = \n{A * B}\ndot(A, B) = \n{np.dot(A, B)}")
# Dodać komentarz

# Zadanie 8
macierz = np.arange(0, 30).reshape((6, 5))
print(f"\nZadanie 8.\nMacierz = \n{macierz}\n Strides = {macierz.strides}")

# Zadanie 9
A = np.ones((2, 3))
B = np.zeros((2, 3))
print(f"\nZadanie 9.\nvstack = \n{np.vstack((A, B))}\n stack = \n{np.stack((A, B))}")

# Zadanie 10
macierz = np.arange(0, 24).reshape((4, 6))
print(f"\n Zadanie 10.\n{np.lib.stride_tricks.as_strided(macierz)}")
