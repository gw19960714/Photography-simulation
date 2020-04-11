import numpy as np
import photography as phgy
import shape
import matplotlib.pyplot as plt

[fig, ax3d] = phgy.create_fig3d(fig_name="Default")
xyz_abc_rgb = shape.water_wave()
ax3d.scatter(xyz_abc_rgb[:, 0], xyz_abc_rgb[:, 1], xyz_abc_rgb[:, 2],
             marker='o', c=xyz_abc_rgb[:, 6:10].tolist(), s=1.0)
plt.show()

# a=np.array([[1, 2, 3],
#             [2, 3, 4],
#             [3, 4, 5],
#             [3, 4, 5]])
# b=np.array([[1], [2], [3], [4]])
# c=np.array([1,2,3])
# # print(np.sum((a-b)*(a-b),axis=1))
# # print(a*c)
# # print(b/b)
# # print(np.sum(a*a, axis=0, keepdims=True))
# print(np.hstack((a, a, b)))
# # print(np.sum(a*a, axis=1, keepdims=True)/b)
