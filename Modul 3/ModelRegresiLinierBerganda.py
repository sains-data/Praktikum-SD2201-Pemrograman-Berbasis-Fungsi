import numpy as np
import pandas as pd

def multiple_linear_regression(X, y):
    X = np.hstack((np.ones((len(X), 1)), X))
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def main():
    # Input dataset from user via terminal
    file_path = input("Masukkan path ke dataset CSV: ")
    
    try:
        # Membaca dataset dengan menentukan tipe data string agar semua nilai dapat diubah menjadi float
        dataset = pd.read_csv(file_path, dtype=str, delimiter=';')
    except FileNotFoundError:
        print("File tidak ditemukan.")
        return
    except Exception as e:
        print("Terjadi kesalahan:", e)
        return

    # Memastikan dataset memiliki setidaknya dua kolom
    if len(dataset.columns) < 2:
        print("Dataset harus memiliki setidaknya dua kolom.")
        return

    # Mengonversi dataset menjadi float
    try:
        dataset = dataset.astype(float)
    except ValueError as ve:
        print("Terjadi kesalahan dalam mengonversi dataset:", ve)
        return

    # Memisahkan variabel independen (X) dan dependen (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Menghitung koefisien regresi linier berganda
    try:
        coefficients = multiple_linear_regression(X, y)
        print("Koefisien regresi linier berganda:")
        for i, coef in enumerate(coefficients):
            print(f"beta_{i}: {coef}")
    except Exception as e:
        print("Terjadi kesalahan saat menghitung koefisien:", e)

if __name__ == "__main__":
    main()
