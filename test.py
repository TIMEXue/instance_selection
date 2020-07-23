import numpy as np

if __name__ == '__main__':
    A = [[[1, 2, 3],
          [3, 4, 5],
          [1, 1, 1]],
         [[2, 3, 4],
          [4, 7, 6],
          [2, 2, 2]],
         [[3, 4, 5],
          [5, 6, 7],
          [3, 3, 3]]]

    order = np.random.permutation(3)
    A = np.asarray(A)
    A[2] = A[2, order]
    print(order)
    print(A)

    B = [[1, 2, 3],
         [4, 5, 6]]
    B = np.asarray(B)
    index = [False, True]
    index = np.asarray(index)
    print(index.shape)
    print(B[index])

    C = [1, 3, 2, 4, 4, 3]
    print(np.argsort(C))

    popo = np.load('population.npy')
    print(popo)