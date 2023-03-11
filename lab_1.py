import numpy as np

# Zadanie 1
macierz = np.random.randint(0, 101, size=(10, 5))
slad = np.trace(macierz)
przekatna = np.diag(macierz)
print(f"Zadanie 1.\nSuma głównej przekątnej: {slad}\nWartości głównej przekątnej: {przekatna}")

# Zadanie 2
macierz1 = np.random.normal(size=(5, 5))
macierz2 = np.random.normal(size=(5, 5))
macierz3 = macierz1 * macierz2
print(f"\nZadanie 2.\n{macierz3}")

# Zadanie 3
C = np.random.randint(1, 101, size=20).reshape((-1, 5))
D = np.random.randint(1, 101, size=20).reshape((-1, 5))
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
# * - mnożenie odpowiadających sobie elementów macierzy A[0, 0] z B[0, 0], A[0, 1] z B[0, 1] itd.
# dot - mnożenie macierzy - mnożenie wiersza jednej macierzy z kolumną drugiej A[0, :] z B[:, 0], A[1, :] z B[:, 1] itd.
# Warto wykorzystać funkcję dot do prawidłowego mnożenia dwóch macierzy

# Zadanie 8
macierz = np.arange(0, 30, dtype=np.int8).reshape((6, 5))
macierz_widok = np.lib.stride_tricks.as_strided(macierz[0:3, 0:5], shape=(3, 5), strides=macierz.strides)
print(f"\nZadanie 8.\nMacierz = \n{macierz}\n 3 kolumny z pięciu pierwszych wierszy = \n{macierz_widok}")

# Zadanie 9
A = np.ones((2, 3))
B = np.zeros((2, 3))
print(f"\nZadanie 9.\nvstack = \n{np.vstack((A, B))}\n stack = \n{np.stack((A, B))}")
# vstack łączy tablice w pionie
# stack łączy tablice wzdłuż nowo utworzonej osi

# Zadanie 10
macierz = np.arange(0, 24).reshape((4, 6))
blok1 = np.lib.stride_tricks.as_strided(macierz[0:2, 0:3], shape=(2, 3), strides=macierz.strides)
blok2 = np.lib.stride_tricks.as_strided(macierz[0:2, 3:6], shape=(2, 3), strides=macierz.strides)
blok3 = np.lib.stride_tricks.as_strided(macierz[2:4, 0:3], shape=(2, 3), strides=macierz.strides)
blok4 = np.lib.stride_tricks.as_strided(macierz[2:4, 3:6], shape=(2, 3), strides=macierz.strides)
print(f"\nZadanie 10.\nMaksymalne wartości:\nblok_1 = {np.max(blok1)},\tblok_2 = {np.max(blok2)},\tblok_3 = {np.max(blok3)},\tblok_4 = {np.max(blok4)}")
