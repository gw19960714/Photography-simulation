import photography as phgy
import shape

xyz_abc_rgb = shape.water_wave()
[fig, ax3d]=phgy.create_fig3d(fig_name="Default")
Scene = phgy.Scene()
O = phgy.Cloud(name='Cloud', xyz_abc_rgb=xyz_abc_rgb)
Scene.add_object(O)
L0 = phgy.Point_Light()
Scene.add_light(L0)
L1 = phgy.Paral_Light()
Scene.add_light(L1)
Scene.plot(ax3d)
Scene.render()
A = phgy.Pinhole_Camera()
A.plot(ax3d, 1000)
phgy.show_axes(ax3d, True)
