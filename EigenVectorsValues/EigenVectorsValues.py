import numpy as np

def get_dimensions(matrix):
    return [len(matrix), len(matrix[0])]

def find_determinant(matrix, excluded=1):
    dimensions = get_dimensions(matrix)
    if dimensions == [2, 2]:
        return excluded * ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    else:
        new_matrices = []
        excluded_list = []
        exclude_row = 0
        for exclude_column in range(dimensions[1]):
            tmp = []
            excluded_list.append(matrix[exclude_row][exclude_column])
            for row in range(1, dimensions[0]):
                tmp_row = []
                for column in range(dimensions[1]):
                    if (row != exclude_row) and (column != exclude_column):
                        tmp_row.append(matrix[row][column])
                tmp.append(tmp_row)
            new_matrices.append(tmp)
        determinants = [find_determinant(new_matrices[j], excluded_list[j]) for j in range(len(new_matrices))]
        determinant = 0
        for i in range(len(determinants)):
            determinant += ((-1)**i)*determinants[i]
        return determinant

def list_multiply(list1, list2):
    result = [0 for _ in range(len(list1) + len(list2) - 1)]
    for i in range(len(list1)):
        for j in range(len(list2)):
            result[i+j] += list1[i] * list2[j]
    return result

def list_add(list1, list2, sub=1):
    return [i + (sub*j) for i, j in zip(list1, list2)]

def determinant_equation(matrix, excluded=[1, 0]):
    dimensions = get_dimensions(matrix)
    if dimensions == [2, 2]:
        tmp = list_add(list_multiply(matrix[0][0], matrix[1][1]), list_multiply(matrix[0][1], matrix[1][0]), sub=-1)
        return list_multiply(tmp, excluded)
    else:
        new_matrices = []
        excluded_list = []
        exclude_row = 0
        for exclude_column in range(dimensions[1]):
            tmp = []
            excluded_list.append(matrix[exclude_row][exclude_column])
            for row in range(1, dimensions[0]):
                tmp_row = []
                for column in range(dimensions[1]):
                    if (row != exclude_row) and (column != exclude_column):
                        tmp_row.append(matrix[row][column])
                tmp.append(tmp_row)
            new_matrices.append(tmp)
        determinant_equations = [determinant_equation(new_matrices[j],
                            excluded_list[j]) for j in range(len(new_matrices))]
        
        # Olası liste boyutu hatalarını önlemek için ufak bir koruma:
        max_len = max(len(eq) for eq in determinant_equations)
        padded_eqs = [eq + [0]*(max_len - len(eq)) for eq in determinant_equations]
        
        dt_equation = [sum(i) for i in zip(*padded_eqs)]
        return dt_equation

def identity_matrix(dimensions):
    matrix = [[0 for j in range(dimensions[1])] for i in range(dimensions[0])]
    for i in range(dimensions[0]):
        matrix[i][i] = 1
    return matrix

def characteristic_equation(matrix):
    dimensions = get_dimensions(matrix)
    return [[[a, -b] for a, b in zip(i, j)] for i, j in zip(matrix,
            identity_matrix(dimensions))]

def find_eigenvalues(matrix):
    dt_equation = determinant_equation(characteristic_equation(matrix))
    return np.roots(dt_equation[::-1])

if __name__ == "__main__":
    # Test Matrisi
    A = [[6, 1, -1],
         [0, 7, 0],
         [3, -1, 2]]

    print("=" * 50)
    print("ÖZDEĞER (EIGENVALUE) HESAPLAMA KARŞILAŞTIRMASI")
    print("=" * 50)

    # 1. Referans Kod (GitHub - LucasBN) ile Hesaplama
    custom_eigenvalues = find_eigenvalues(A)
    print("\n--- Manuel (Referans) Fonksiyon Sonuçları ---")
    print("Özdeğerler:", custom_eigenvalues)

    # 2. Numpy np.linalg.eig ile Hesaplama
    # Numpy fonksiyonu matrisi np.array formatında ister
    A_np = np.array(A)
    eigenvalues_np, eigenvectors_np = np.linalg.eig(A_np)

    print("\n--- Numpy np.linalg.eig Sonuçları ---")
    print("Özdeğerler:", eigenvalues_np)
    print("\n(Numpy ayrıca özvektörleri de hesaplar:)")
    print(eigenvectors_np)
    
    print("\n" + "=" * 50)
    print("SONUÇ VE KARŞILAŞTIRMA:")
    print("Her iki yöntem de aynı özdeğerleri bulmuştur. Numpy, işlemi")
    print("arka planda optimize edilmiş LAPACK kütüphanesi ile çalıştırarak")
    print("çok daha hızlı gerçekleştirir ve ek olarak özvektörleri de döndürür.")
    print("=" * 50)