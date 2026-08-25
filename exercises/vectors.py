import math 
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a+b for a,b in zip(self.components, other.components)])
    
    def __sub__(self, other):
        return Vector([a-b for a,b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a*b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x **2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x/mag for x in self.components])
    
    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        dot_product = self.dot(other)
        magnitude_product = self.magnitude()

        if magnitude_product == 0:
            raise ValueError("Angle is undefined for a zero vector")

        cos_theta = dot_product / magnitude_product
        cos_theta = max(-1, min(1, cos_theta))
        angle_radians = math.acos(cos_theta)

        return math.degrees(angle_radians)



    def __repr__(self):
        return f"Vector({self.components})"

a = Vector([1,0])
b = Vector([0,1])

print(f" a + b = {a + b}")

print(f"a .b = {a.dot(b)}")

print(f"|a| = {a.magnitude():.4f}")

print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
print(a.angle_between(b))


