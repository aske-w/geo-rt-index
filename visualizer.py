import numpy as np
import matplotlib.pyplot as plt
# this is mostly the work of ChatGPT

LO = 0
HI = 1

def mk_query(selectivity: float, lo = 0, hi = 1):
    # raise NotImplementedError()
    assert(0 < selectivity and selectivity <= 1)
    area = selectivity
    # Generate random dimensions
    width = np.random.uniform(1, area)  # Limit the width to be less than or equal to the area
    height = area / width
    
    # Check if the dimensions produce the desired area
    if abs(width * height - area) < 1e-9:
        # return width, height
        x1 = np.random.uniform(lo, hi - width)
        y1 = np.random.uniform(lo, hi - height)
        x2 = x1 + width
        y2 = y1 + height
        return (x1, y1), (x2, y2) # minx, miny, maxx maxy semantics



# Create a plot
fig, ax = plt.subplots()

# Generate random width and length

for s in [0.01, 0.02, 0.05, 0.10, 0.20]:
    for _ in range(1024):
    # Generate random coordinates
        corner1, corner2 = mk_query(0.1, LO, HI)
        print(corner1, corner2)
        width = corner2[0] - corner1[0]
        length = corner2[1] - corner1[1]

        # Plot the rectangle
        rectangle = plt.Rectangle(corner1, width, length, fill=True, edgecolor=None)
        ax.add_patch(rectangle)
# ax.scatter(*corner1)
# ax.scatter(*corner2)
# Set limits and labels
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bounding Box Visualization')

# Show the plot
# plt.grid(True)
plt.show()
