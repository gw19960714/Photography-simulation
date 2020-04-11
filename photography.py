import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def set_axes3d_equal(ax3d):
    """Make axes of 3D plot have equal scale.
      Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    code from https://stackoverflow.com/a/31364297
    Args:
        ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
    """
    x_limits = ax3d.get_xlim3d()
    y_limits = ax3d.get_ylim3d()
    z_limits = ax3d.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax3d.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax3d.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax3d.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def rgb_to_RGB(r, g, b):
    """Convert r, g, b value(0.0~1.0) to hex color code."""
    Rgb = 0
    Rgb += (int(r) << 16)
    Rgb += (int(g) << 8)
    Rgb += int(b)
    return Rgb


def create_fig3d(fig_name="Default"):
    """Create a figure with 3d axes.
    Returns:
        fig: Figure in matplotlib.pyplot.
        ax3d: Axes3D in mpl_toolkits.mplot3d.
    """
    fig = plt.figure(fig_name)
    ax3d = Axes3D(fig)
    # ax3d.w_xaxis.set_pane_color((1.0, 1.0, 0.0, 1.0))
    # ax3d.w_xaxis.set_pane_color((1.0, 1.0, 0.0, 1.0))
    # ax3d.w_xaxis.set_pane_color((1.0, 1.0, 0.0, 1.0))
    return [fig, ax3d]


def show_axes(ax3d, equal=True):
    """Show axes in equal axis.
    Args:
        ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
    """
    if equal is True:
        set_axes3d_equal(ax3d)
    plt.show()


class Camera:
    """Virtual basic camera class, with no practical function.
    Attributes:
        data: Initialize dictionary of basic camera, only has item of name.
        instance_set: All instance set of cameras.
        instance_count: All instance number of cameras.
    """
    instance_set = {}
    instance_count = 0

    def __init__(self, name='Basic Camera'):
        """Initializes basic camera and assign a name."""
        self.data={}
        self.data["name"] = name
        Camera.instance_set[Camera.instance_count] = self
        Camera.instance_count += 1


class Pinhole_Camera(Camera):
    """Implementation of pinhole perspective camera."""
    def __init__(self, name='Basic Camera', focus=50.0,
                 width=1024.0, height=768.0,
                 dx=5.5, dy=5.5, u0=512.0, v0=384.0,
                 location=None,
                 axis=None,
                 inner=None):
        """Initializes camera parameters using JSON format strings.
           Args:
               name: Camera name.
               focus: Camera focus, unit mm default 50.0.
               width,height: Image size, unit pixel.
               dx, dy: Sensor size unit um.
               u0, v0: Center of image, unit pixel.
               location: Location vector of camera center, unit mm default [0.0, 0.0, 1000.0].
               axis: Assign y-axis as None and solve it by cross product of z-axis X x-axis, such as
                    [[1.0, 0.0, 0.0],[None, None, None],[0.0, 0.0, -1.0]].
               inner: If None inner=[[focus/dx, 0, u0],[0, focus/dy, v0],[0, 0, 1]].
        """
        # Initialize Virtual basic camera
        Camera.__init__(self, name)
        # Initialize parameters
        self.data["name"] = name
        self.data["focus"] = focus
        self.data["width"] = width
        self.data["height"] = height
        self.data["dx"] = dx
        self.data["dy"] = dy
        self.data["sensor_size"] = [self.data["dx"] * self.data["width"] / 1000,
                                    self.data["dy"] * self.data["height"] / 1000]
        self.data["u0"] = u0
        self.data["v0"] = v0
        if inner is None:
            self.data["inner"] =np.array([[self.data["focus"] / self.data["dx"], 0, self.data["u0"]],
                                          [0, self.data["focus"] / self.data["dy"], self.data["v0"]],
                                          [0, 0, 1]], dtype=float)
        else:
            self.data["inner"] = np.array(self.data["inner"][0:3][0:3], dtype=float)
        if location is None:
            self.data["location"] = np.array([0.0, 0.0, 1000.0], dtype=float)
        else:
            self.data["location"] = np.array(self.data["location"][0:3], dtype=float)
        if axis is None:
            self.data["axis"] = np.array([[1.0, 0.0, 0.0],
                                          [0.0, -1.0, 0.0],
                                          [0.0, 0.0, -1.0]], dtype=float)
        else:
            self.data["axis"] = np.array(self.data["axis"][0:3][0:3], dtype=float)
            if np.isnan(self.data["axis"][1][1]):
                self.data["axis"][1] = np.cross(self.data["axis"][2], self.data["axis"][0])
        R_ = np.transpose(np.array(self.data["axis"], dtype=float))
        R = np.linalg.inv(R_)
        d = -np.dot(R, np.transpose(np.array([self.data["location"]], dtype=float)))
        Rd = np.hstack((R, d))
        Rd = np.vstack((Rd, np.array([0, 0, 0, 1], dtype=float)))
        self.data["outer"] = Rd

    def plot(self, ax3d, obj_dist=1000):
        """Draw camera in 3d axes.
        Args:
            obj_dist: Object distance of camera.
            ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
        """
        # draw axis_z
        cam_o = np.array(self.data["location"])
        axis_x = np.array(self.data["axis"][0])
        axis_y = np.array(self.data["axis"][1])
        axis_z = np.array(self.data["axis"][2])
        sns_o = cam_o - axis_z * self.data["focus"]
        obj_o = cam_o + axis_z * obj_dist
        ax3d.plot([sns_o[0], obj_o[0]], [sns_o[1], obj_o[1]], [sns_o[2], obj_o[2]], color='blue', linewidth=0.5)
        # draw sensor plane
        sns_w = self.data["sensor_size"][0] / 2
        sns_h = self.data["sensor_size"][1] / 2
        sns_1 = sns_o + axis_x * sns_w + axis_y * sns_h
        sns_2 = sns_o + axis_x * sns_w - axis_y * sns_h
        sns_3 = sns_o - axis_x * sns_w + axis_y * sns_h
        sns_4 = sns_o - axis_x * sns_w - axis_y * sns_h
        ax3d.plot([sns_1[0], sns_2[0]], [sns_1[1], sns_2[1]], [sns_1[2], sns_2[2]], color='blue', linewidth=0.5)
        ax3d.plot([sns_2[0], sns_4[0]], [sns_2[1], sns_4[1]], [sns_2[2], sns_4[2]], color='blue', linewidth=0.5)
        ax3d.plot([sns_4[0], sns_3[0]], [sns_4[1], sns_3[1]], [sns_4[2], sns_3[2]], color='blue', linewidth=0.5)
        ax3d.plot([sns_3[0], sns_1[0]], [sns_3[1], sns_1[1]], [sns_3[2], sns_1[2]], color='blue', linewidth=0.5)
        # draw object plane
        obj_w = (obj_dist / self.data["focus"]) * self.data["sensor_size"][0] / 2
        obj_h = (obj_dist / self.data["focus"]) * self.data["sensor_size"][1] / 2
        obj_1 = obj_o + axis_x * obj_w + axis_y * obj_h
        obj_2 = obj_o + axis_x * obj_w - axis_y * obj_h
        obj_3 = obj_o - axis_x * obj_w + axis_y * obj_h
        obj_4 = obj_o - axis_x * obj_w - axis_y * obj_h
        ax3d.plot([obj_1[0], obj_2[0]], [obj_1[1], obj_2[1]], [obj_1[2], obj_2[2]], color='blue', linewidth=0.5)
        ax3d.plot([obj_2[0], obj_4[0]], [obj_2[1], obj_4[1]], [obj_2[2], obj_4[2]], color='blue', linewidth=0.5)
        ax3d.plot([obj_4[0], obj_3[0]], [obj_4[1], obj_3[1]], [obj_4[2], obj_3[2]], color='blue', linewidth=0.5)
        ax3d.plot([obj_3[0], obj_1[0]], [obj_3[1], obj_1[1]], [obj_3[2], obj_1[2]], color='blue', linewidth=0.5)
        # # draw view cone
        # ax.plot([sns_3[0], obj_2[0]], [sns_3[1], obj_2[1]], [sns_3[2], obj_2[2]], color='blue', linewidth=0.5)
        # ax.plot([sns_1[0], obj_4[0]], [sns_1[1], obj_4[1]], [sns_1[2], obj_4[2]], color='blue', linewidth=0.5)
        # ax.plot([sns_2[0], obj_3[0]], [sns_2[1], obj_3[1]], [sns_2[2], obj_3[2]], color='blue', linewidth=0.5)
        # ax.plot([sns_4[0], obj_1[0]], [sns_4[1], obj_1[1]], [sns_4[2], obj_1[2]], color='blue', linewidth=0.5)
        ax3d.plot([cam_o[0], obj_2[0]], [cam_o[1], obj_2[1]], [cam_o[2], obj_2[2]], color='blue', linewidth=0.5)
        ax3d.plot([cam_o[0], obj_4[0]], [cam_o[1], obj_4[1]], [cam_o[2], obj_4[2]], color='blue', linewidth=0.5)
        ax3d.plot([cam_o[0], obj_3[0]], [cam_o[1], obj_3[1]], [cam_o[2], obj_3[2]], color='blue', linewidth=0.5)
        ax3d.plot([cam_o[0], obj_1[0]], [cam_o[1], obj_1[1]], [cam_o[2], obj_1[2]], color='blue', linewidth=0.5)

    def moveto(self, location, axis):
        """Move camera into new location and direction.
        Args:
            location: New location of camera, e. g. [0.0, 0.0, 1000.0].
            axis: Axis vector of x-y-z,  e. g. [[1.0, 0.0, 0.0],[NaN, NaN, NaN],[0.0, 0.0, -1.0]],
                assign y-axis as NaN and solve it by cross product of z-axis X x-axis.
        """
        self.data["location"] = np.array(location[0:3], dtype=float)
        self.data["axis"] = np.array([axis[0],
                                      np.cross(axis[2], axis[0]).tolist(),
                                      axis[2]], dtype=float)
        R_ = np.transpose(np.array(self.data["axis"], dtype=float))
        R = np.linalg.inv(R_)
        d = -np.dot(R, np.transpose(np.array([self.data["location"]], dtype=float)))
        Rd = np.hstack((R, d))
        Rd = np.vstack((Rd, np.array([0, 0, 0, 1], dtype=float)))
        self.data["outer"] = Rd


class Light:
    """Virtual basic light class, with no practical function.
    Attributes:
        data:  Initialize dictionary of light, such as name, luminance, color and so on.
        instance_set: All instance set of lights.
        instance_count: All instance number of lights.
    """
    instance_set = {}
    instance_count = 0

    def __init__(self, name='Default Light', value=1000.0, color=None):
        """Initializes basic light parameters using JSON format strings.
           Args:
               name: Light name.
               value: Light value, unit differs within light types.
               color: Color vector, unit % default [1.0, 0.5, 0.5].
        """
        self.data={}
        self.data["name"] = name
        self.data["value"] = float(value)
        if color is None:
            color =  np.array([1.0, 0.5, 0.5], dtype=float)
        else:
            color = np.array(color[0:3], dtype=float)
        self.data["color"] = color.tolist()
        # Y=0.30R+0.59G+0.11B
        color_ratio = color*np.array([0.3, 0.59, 0.11], dtype=float)
        self.data["color_value"] = (float(value)*color_ratio/np.sum(color_ratio)).tolist()
        Light.instance_set[Light.instance_count] = self
        Light.instance_count += 1


class Point_Light(Light):
    """Implementation of point light illumination.
    Attributes:
        data: Initialize dictionary of light, add location item.
    """
    def __init__(self, name='Point Light', intensity=1000.0, color=None, location=None):
        """Initializes point light parameters using JSON format strings.
           Args:
               name: Light name.
               intensity: Light luminance intensity, unit lm/sr default 1000.0.
               color: Color vector, unit % default [1.0, 0.5, 0.5].
               location: Location vector, unit mm default [0.0, 0.0, 1500.0].
        """
        Light.__init__(self, name, intensity, color)
        if location is None:
            self.data["location"] = np.array([0.0, 0.0, 1500.0], dtype=float)
        else:
            self.data["location"] = np.array(location[0:3], dtype=float)

    def plot(self, ax3d, size=100):
        """Draw point light in 3d axes.
        Args:
            size: Point light size in 3d axes.
            ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
        """
        light_o = np.array(self.data["location"])
        ax3d.scatter(self.data["location"][0], self.data["location"][1], self.data["location"][2],
                     marker='o', c=[self.data["color"]], s=size)

    def moveto(self, location):
        """Move point light into new location.
        Args:
            location: New location of point light, e. g. [0.0, 0.0, 1000.0].
        """
        self.data["location"] = np.array(location[0:3], dtype=float)

    def project(self, location, direction):
        """Irradiance projected to unit area surface.
        Args:
            location: Location of ray projected, e. g. np.array([[0.0, 0.0, 1000.0],
                                                                 [0.0, 0.0, 1000.0]]).
            direction: Direction of received surface, default np.array([[0.0, 0.0, 1.0],
                                                                        [0.0, 0.0, 1.0]]).
        Returns:
            irradiance: Luminous flux received by unit area of specific direction from this light, unit lm/m2.
            [irradiance, irradiance_r, irradiance_g, irradiance_b] = [YRGB]
        """
        ray_vec = self.data["location"] - location #(n,3)
        ray_len2 = np.sum(ray_vec*ray_vec, axis=1, keepdims=True) #(n,1)
        sr_perarea = 1000000/ray_len2 #(n,1)
        ray_cos = np.sum(ray_vec*direction, axis=1, keepdims=True)/np.sqrt(ray_len2) #(n,1)/(n,1)=(n,1)
        cos_sr_perarea = ray_cos * sr_perarea #(n,1)*(n,1)=(n,1)
        irradiance = self.data["value"]*cos_sr_perarea
        irradiance_r = self.data["color_value"][0]*cos_sr_perarea
        irradiance_g = self.data["color_value"][1]*cos_sr_perarea
        irradiance_b = self.data["color_value"][2]*cos_sr_perarea
        return irradiance, irradiance_r, irradiance_g, irradiance_b


class Paral_Light(Light):
    """Implementation of parallel light illumination.
    Attributes:
        data: Initialize dictionary of light, add direction item.
    """
    def __init__(self, name='Parallel Light', illuminance=1000.0, color=None, direction=None):
        """Initializes basic light parameters using JSON format strings.
           Args:
               name: Light name.
               illuminance: Light illuminance, unit lm/m2 default 1000.0.
               color: Color vector, unit % default [1.0, 0.5, 0.5].
               direction: Location unit vector, default [0.0, 0.0, -1.0].
        """
        Light.__init__(self, name, illuminance, color)
        if direction is None:
            self.data["direction"] = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            self.data["direction"] = np.array(direction[0:3], dtype=float)

    def plot(self, ax3d, size=200, start=None):
        """Draw parallel light in 3d axes.
        Args:
            size: Parallel light arrow size in 3d axes.
            start: Parallel light arrow start in 3d axes.
            ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
        """
        if start is None:
            start = [0.0, 0.0, 2000.0]
        light_s = np.array(start[0:3], dtype=float)
        # light_e = light_s + size*np.array(self.data["direction"], dtype=float)
        # ax3d.plot([light_e[0], light_s[0]], [light_e[1], light_s[1]], [light_e[2], light_s[2]],
        #           color='blue', linewidth=0.5)
        ax3d.quiver(light_s[0], light_s[1], light_s[2],
                    self.data["direction"][0], self.data["direction"][1], self.data["direction"][2],
                    color=[self.data["color"]], length=size, normalize=True)

    def moveto(self, direction):
        """Move parallel light into new direction.
        Args:
            direction: New direction vector of parallel light,  e. g. [0.0, 0.0, -1.0].
        """
        self.data["direction"] = np.array(direction[0:3], dtype=float)

    def project(self, location, direction=None):
        """Irradiance projected to unit area surface.
        Args:
            location: Location of ray projected, e. g. np.array([[0.0, 0.0, 1000.0],
                                                                 [0.0, 0.0, 1000.0]]).
            direction: Direction of received surface, default np.array([[0.0, 0.0, 1.0],
                                                                        [0.0, 0.0, 1.0]]).
        Returns:
            irradiance: Luminous flux received by unit area of specific direction from this light, unit lm/m2.
        """
        ray_cos = -np.sum(self.data["direction"]*direction, axis=1, keepdims=True) #(n,1)/(n,1)=(n,1)
        irradiance = self.data["value"]*ray_cos #(n,1)
        irradiance_r = self.data["color_value"][0]*ray_cos #(n,1)
        irradiance_g = self.data["color_value"][1]*ray_cos #(n,1)
        irradiance_b = self.data["color_value"][2]*ray_cos #(n,1)
        return irradiance, irradiance_r, irradiance_g, irradiance_b


class Object:
    """Virtual basic object class, with no practical function.
    Attributes:
        data:  Initialize dictionary of object, just as name.
        instance_set: All instance set of lights.
        instance_count: All instance number of lights.
    """
    instance_set = {}
    instance_count = 0

    def __init__(self, name='Basic Object'):
        """Initializes basic object and assign a name."""
        self.data={}
        self.data["name"] = name
        Camera.instance_set[Camera.instance_count] = self
        Camera.instance_count += 1


class Cloud(Object):
    """Disk cloud object in xyz_abc_rgb format.
    Attributes:
        data: Initialize dictionary of basic camera, only has item of name.
        instance_set: All instance set of cameras.
        instance_count: All instance number of cameras.
    """
    instance_set = {}
    instance_count = 0

    def __init__(self, name='Cloud', xyz_abc_rgb=None, size=None, center=None):
        """Initializes cloud object and assign a name.
           Args:
               name: Cloud name.
               xyz_abc_rgb: List of disk cloud, each point is list of [x, y, z, a, b, c, r, g, b],
                            xyz is location of point, unit mm;
                            abc is surf unit vector;
                            rgb is 0.0~1.0 valued color. default cloud list is:
                            [[0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                             [-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2]...].
               size: Disk diameter size, unit mm.
               center: Center of cloud default is geometry center vector.
        """
        Object.__init__(self, name)
        if size is None:
            self.data["size"] = 1.0
            self.data["area"] = 0.25*self.data["size"]*self.data["size"]*np.pi
        else:
            self.data["size"] = float(size)
            self.data["area"] = 0.25*self.data["size"]*self.data["size"]*np.pi
        if xyz_abc_rgb is None:
            self.data["len"] = 9
            self.data["xyz_abc_rgb"] = np.array([[-100.0, -100.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [-100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [-100.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [0.0, -100.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [100.0, -100.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2],
                                                 [100.0, 100.0, 0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.2]], dtype=float)
        else:
            self.data["xyz_abc_rgb"] = np.array(xyz_abc_rgb, dtype=float)[:, 0:9]
            self.data["len"] = len(self.data["xyz_abc_rgb"])
        self.data["irr"] = np.zeros((self.data["len"],4), dtype=float) #[[Y, R, G, B]]
        self.data["lum"] = np.zeros((self.data["len"],4), dtype=float) #[[Y, R, G, B]]
        if center is None:
            self.data["center"] = np.mean(self.data["xyz_abc_rgb"][:, 0:3], axis=0)
        else:
            self.data["center"] = np.array(center[0:3], dtype=float)

    def plot(self, ax3d):
        """Draw cloud object in 3d axes.
        Args:
            ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
        """
        ax3d.scatter(self.data["xyz_abc_rgb"][:, 0], self.data["xyz_abc_rgb"][:, 1], self.data["xyz_abc_rgb"][:, 2],
                     marker='o', c=self.data["xyz_abc_rgb"][:, 6:10].tolist(), s=self.data["size"])

    def moveto(self, location):
        """Move cloud object to new place.
        Args:
            location: New location vector of cloud center,  e. g. [0.0, 0.0, 10.0].
        """
        moveing = np.array(location[0:3], dtype=float) - self.data["center"]
        self.data["xyz_abc_rgb"][:,0:3] = self.data["xyz_abc_rgb"][:,0:3] + moveing

    def clear(self):
        """Clear iluminance receive by disk cloud object."""
        self.data["iluminance"] = np.zeros((self.data["len"],4), dtype=float) #[[Y, R, G, B]]
        self.data["luminous"] = np.zeros((self.data["len"],4), dtype=float) #[[Y, R, G, B]]

    def receive(self, light, shade=False):
        """Irradiance and luminous received by disk cloud object.
        Args:
            light: Light in use.
        Returns:
            lum: Luminous received by disk cloud object, unit lm.
            irr: Irradiance received by disk cloud object, unit lm/m2.
        """
        irr, irr_r, irr_g, irr_b = light.project(self.data["xyz_abc_rgb"][:, 0:3],
                                                 self.data["xyz_abc_rgb"][:, 3:6])
        self.data["irr"] = self.data["irr"] + np.hstack((irr, irr_r, irr_g, irr_b))
        self.data["lum"] = self.data["lum"] * self.data["area"]


# class Mesh(Object):
#     """Mesh object in X Y Z Nx Ny Nz format.
#     Attributes:
#         data: Initialize dictionary of basic camera, only has item of name.
#         instance_set: All instance set of cameras.
#         instance_count: All instance number of cameras.
#     """
#     instance_set = {}
#     instance_count = 0
#
#     def __init__(self, name='Mesh', X, Y, Z, Nx, Ny, Nz):
#         """Initializes mesh object and assign a name.
#            Args:
#                name: Mesh name.
#                X, Y, Z, Nx, Ny, Nz: 2D mesh array.
#                size: Disk diameter size, unit mm.
#                center: Center of cloud default is geometry center vector.
#         """
#         Object.__init__(self, name)
#         self.data["X"] = np.array(X, dtype=float)
#         self.data["Y"] = np.array(Y, dtype=float)
#         self.data["Z"] = np.array(Z, dtype=float)
#         self.data["Nx"] = np.array(Nx, dtype=float)
#         self.data["Ny"] = np.array(Ny, dtype=float)
#         self.data["Nz"] = np.array(Nz, dtype=float)
#         self.data["irr"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["lum"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["irr_r"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["lum_r"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["irr_g"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["lum_g"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["irr_b"] = np.zeros_like(self.data["X"], dtype=float)
#         self.data["lum_b"] = np.zeros_like(self.data["X"], dtype=float)
#
#         if size is None:
#             self.data["size"] = 1.0
#             self.data["area"] = 0.25*self.data["size"]*self.data["size"]*np.pi
#         else:
#             self.data["size"] = float(size)
#             self.data["area"] = 0.25*self.data["size"]*self.data["size"]*np.pi
#         self.data["center"] = np.array([np.mean(X), np.mean(Y), np.mean(Z)], dtype=float)


class Scene:
    """Scene of illumination.
    Attributes:
        obj_list: List of  dictionary of scene.
        light_list: All instance set of cameras.
    """

    def __init__(self, name='Scene'):
        """Initializes basic Scene and assign a name."""
        self.obj_list = {}
        self.obj_count = 0
        self.light_list = {}
        self.light_count = 0

    def add_object(self, object):
        """Add object in scene."""
        self.obj_list[self.obj_count] = object
        self.obj_count += 1

    def add_light(self, light):
        """Add light in scene."""
        self.light_list[self.light_count] = light
        self.light_count += 1

    def plot(self, ax3d):
        """Draw scene object and light in 3d axes.
        Args:
            ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").
        """
        for o in range(0, self.obj_count):
            self.obj_list[o].plot(ax3d)
        for l in range(0, self.light_count):
            self.light_list[l].plot(ax3d)

    def render(self, shade=False):
        """Render object in Scene.

            Render view by reverse tracking the incoming lights into camera
            which were reflected on object surface from light source.

        Args:
            ax3d: Axes3D in mpl_toolkits.mplot3d, call [fig, ax3d]=create_fig3d("fig_name").

        """
        print("Scene")
        for o in range(0, self.obj_count):
            self.obj_list[o].clear()
        for l in range(0, self.light_count):
            for o in range(0, self.obj_count):
                self.obj_list[o].receive(self.light_list[l])
        print("OK")
