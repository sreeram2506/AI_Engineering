from vectors import Vector
import numpy as np
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")


import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")


# ex-pb:2 Create a 2D scaling matrix that doubles the x-coordinate and triples the y-coordinate, then apply it to the vector [1, 1]

import numpy as np

scaling_matrix = np.array([
    [2, 0],
    [0, 3]
])

vector = np.array([1, 1])

result = scaling_matrix @ vector

print(result)


#ex-pb: 3 Given 5 random word-like vectors (dimension 50), find the two most similar using cosine similarity

vectors = np.random.rand(5,50)

highest_similarity = -1
most_similar_pair = None

for i in range(5):
    for j in range(i+1, 5):

        dot_product = np.dot(vectors[i], vectors[j])

        magnitude_a = np.linalg.norm(vectors[i])
        magnitude_b = np.linalg.norm(vectors[j])

        similarity = dot_product / (magnitude_a * magnitude_b)

        print(f"Vector {i} vs Vector {j}: {similarity:.4f}")

        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_pair = (i, j)

            
assert most_similar_pair is not None

print("\nMost similar pair:")
print(f"Vector {most_similar_pair[0]} and Vector {most_similar_pair[1]}")
print(f"Cosine similarity: {highest_similarity:.4f}")











