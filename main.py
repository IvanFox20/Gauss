import numpy as np



A = np.array(
    [[1.3, 0.4, 0.5],
     [0.4, 1.3, 0.3],
     [0.5, 0.3, 1.3]
     ], dtype=np.float32
)

B = np.array([[0.25,
              0.51,
              1.44]],
dtype=np.float32
)

def solve_gauss(matrix, n, m):
    rank = 0

    # Прямой ход
    for k in range(n):
        # Поиск первого ненулевого элемента в текущем столбце
        first_non_zero = -1
        for i in range(k, n):
            if not np.isclose(matrix[i][k], 0, atol=1e-4):
                first_non_zero = i
                break

        if first_non_zero != -1:
            # Обмен текущей строки с строкой, содержащей первый ненулевой элемент
            matrix[k], matrix[first_non_zero] = matrix[first_non_zero], matrix[k]

            # Нормализация строки
            div = matrix[k][k]
            matrix[k] = [elem / div for elem in matrix[k]]

            # Вычитание текущей строки из остальных строк
            for i in range(n):
                if i != k:
                    factor = matrix[i][k]
                    matrix[i] = [elem - factor * matrix[k][j] for j, elem in enumerate(matrix[i])]

            rank += 1
        print('\n')
        print(matrix)
        print('\n')
    return rank, matrix

def is_zero_vector(v):
    return all(x == 0 for x in v)

def print_solution(matrix, rank):
    if rank == len(matrix):
        print("Решение системы:")
        for i, row in enumerate(matrix):
            print(f"X{i + 1} = {row[-1]}")
    else:

      for i in range(matrix.shape[0]):
        copy_v = matrix[i][:-1]
        print(copy_v, matrix[i][matrix.shape[1] - 1])
        if is_zero_vector(copy_v) is True and matrix[i][matrix.shape[1] - 1]!=0:
          print("Решений нет")
          return

      print("Система имеет бесконечное количество решений.")
      print("Частное решение:")
      for i in range(len(matrix[0]) - 1):
          print(f"X{i + 1} = {(matrix[i][-1] if i < rank else 0)}")


def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]



def main():
    matrix = np.concatenate((A, B.T), axis=1)
    print(matrix)
    # Решение СЛАУ методом Гаусса
    rank, reduce_matrix = solve_gauss(np.array(matrix, dtype=np.float32), matrix.shape[0], matrix.shape[1])

    # Вывод решения
    print_solution(np.array(reduce_matrix), rank)


main()