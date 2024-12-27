import numpy as np


class Vector(list):
    def __init__(self, data):
        self.data = data
        self.size = len(data)

    def __len__(self):
        return self.size

    def __add__(self, other):
        assert self.size == other.size, "Vectors must have the same size."
        return Vector([self.data[i] + other.data[i] for i in range(self.size)])

    def __sub__(self, other):
        assert self.size == other.size, "Vectors must have the same size."
        return Vector([self.data[i] - other.data[i] for i in range(self.size)])

    def __mul__(self, other):
        if isinstance(other, Vector):
            raise ValueError("Use 'v.T * w' for dot product (v^T * w).")
        elif isinstance(other, Matrix):  # Vector dyadic product with another Vector
            assert 1 == other.rows, "Vector * Matrix can only be dyadic product."
            return Matrix(
                [
                    [self.data[i] * other.data[0][j] for j in range(other.cols)]
                    for i in range(len(self.data))
                ]
            )
        else:  # Scalar multiplication
            return Vector([x * other for x in self.data])

    def __rmul__(self, other):
        return self * other

    def __getitem__(self, key):
        return self.data[key]

    def dot(self, other):
        assert self.size == other.size, "Vectors must have the same size."
        return sum([self.data[i] * other.data[i] for i in range(self.size)])

    @property
    def norm2(self):
        return self.dot(self) ** 0.5

    @property
    def T(self):
        # Return the vector as a "row vector" (1-row matrix)
        return Matrix([[x for x in self.data]])

    @property
    def scalarize(self):
        if self.size == 1:
            return self.data[0]
        else:
            return self

    @property
    def skew(self):
        return Matrix(
            [
                [0, -self.data[2], self.data[1]],
                [self.data[2], 0, -self.data[0]],
                [-self.data[1], self.data[0], 0],
            ]
        )

    def __repr__(self):
        return f"Vector({self.data})"


class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.__inv = None

    def __add__(self, other):
        assert (
            self.rows == other.rows and self.cols == other.cols
        ), "Matrix dimensions must match."
        return Matrix(
            [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __sub__(self, other):
        assert (
            self.rows == other.rows and self.cols == other.cols
        ), "Matrix dimensions must match."
        return Matrix(
            [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if isinstance(other, Vector):  # Matrix * Vector
            assert self.cols == other.size, "Incompatible dimensions."
            vector = Vector(
                [
                    sum(self.data[i][j] * other.data[j] for j in range(self.cols))
                    for i in range(self.rows)
                ]
            )
            return vector.scalarize
        elif isinstance(other, Matrix):  # Matrix * Matrix
            assert self.cols == other.rows, "Incompatible dimensions."
            return Matrix(
                [
                    [
                        sum(
                            self.data[i][k] * other.data[k][j] for k in range(self.cols)
                        )
                        for j in range(other.cols)
                    ]
                    for i in range(self.rows)
                ]
            )
        else:  # Scalar multiplication
            return Matrix(
                [
                    [other * self.data[i][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ]
            )

    @property
    def inverse(self):
        """Calculate the inverse using numpy."""
        if self.__inv is not None:
            return self.__inv

        if self.rows != self.cols:
            raise ValueError("Matrix must be square to calculate inverse.")

        np_matrix = np.array(self.data)
        try:
            np_inv = np.linalg.inv(np_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular and cannot be inverted.")

        self.__inv = Matrix(np_inv.tolist())

        return self.__inv

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self.data[row][col]
        return self.data[key]

    @property
    def transpose(self):
        return Matrix(
            [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        )

    @property
    def T(self):
        return self.transpose

    @property
    def inv(self):
        return self.inverse

    def col(self, j):
        return Vector([self.data[i][j] for i in range(self.rows)])

    def __repr__(self):
        return f"Matrix({self.data})"

    @staticmethod
    def Identity(dim):
        return Matrix([[1 if i == j else 0 for i in range(dim)] for j in range(dim)])

    @staticmethod
    def I(dim):
        return Matrix.Identity(dim)
