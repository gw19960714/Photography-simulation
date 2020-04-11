import numpy as np

def water_wave():
    """Irradiance projected to unit area surface.
    Returns:
        xyz_abc_rgb: List of disk cloud, each point is list of [x, y, z, a, b, c, r, g, b],
                     xyz is location of point, unit mm;
                     abc is surf unit vector;
                     rgb is 0.0~1.0 valued color. default cloud list is:
                     [[0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                      [-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2]...].
    """
    x = np.arange(-100, 100, 10)
    y = np.arange(-100, 100, 10)
    X, Y = np.meshgrid(x, y)
    X= np.transpose(np.array([X.flatten().tolist()]))
    Y= np.transpose(np.array([Y.flatten().tolist()]))
    R = np.sqrt(X**2 + Y**2)
    Z = np.cos(R/10)*20
    XYZ = np.hstack((X, Y, Z))
    ABC = np.zeros_like(XYZ) + np.array([0.0, 0.0, 1.0])
    RGB = np.ones_like(XYZ) * np.array([0.2, 1.0, 0.2])
    return np.hstack((XYZ, ABC, RGB))