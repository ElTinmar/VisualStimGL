import sys
from vispy import gloo, app,  use 
from vispy.geometry import create_cylinder
from vispy.util.transforms import perspective, translate, rotate
import numpy as np
from PyQt5.QtCore import QPoint, Qt
import time

# we need full gl context for instanced rendering (this requires PyOpenGL: pip install PyOpenGL PyOpenGL_accelerate)
use(gl='gl+')

VERT_SHADER_CYLINDER ="""
attribute vec3 a_position;
attribute vec4 a_color;
attribute vec3 a_normal;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
varying vec4 v_color;
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    v_color = a_color;
}
"""

VERT_SHADER_FLOOR ="""
attribute vec3 a_position;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
void main()
{
    gl_Position =  u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;
FRAG_SHADER_CYLINDER = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""

FRAG_SHADER_FLOOR = """
void main()
{
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""

# TODO use instance rendering to display multiple cylinders
# TODO use keyboard + mouse to control the view matrix

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1280,720), fullscreen=False, keys='interactive')

        mesh_data = create_cylinder(rows=10, cols = 36)
        self.phi, self.theta = 60, 20
        self.step_t = 0.1
        self.step_t_fast = 0.2
        self.step_r = 0.1
        self.cam_x = 0
        self.cam_y = 0
        self.cam_z = -5
        self.cam_phi = 0
        self.cam_theta = 0
        self.z_near = 0.1
        self.z_far = 100
        self.fovy = 90
        self.last_mouse_pos = None
        self.view = translate((self.cam_x, self.cam_y, self.cam_z))
        self.cylinder_model = rotate(self.theta, (0, 0, 1)).dot(rotate(self.phi, (0, 1, 0)))
        self.floor_model = translate((0,0,0))

        # FLOOR
        self.floor_program = gloo.Program(VERT_SHADER_FLOOR, FRAG_SHADER_FLOOR)
        self.floor_program["a_position"] = [(-1,-1,-1),(-1,-1,1),(1,-1,-1),(1,-1,1)]
        self.floor_program["u_view"] = self.view
        self.floor_program["u_model"] = self.floor_model

        # CYLINDER
        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        positions = mesh_data.get_vertices()
        print(positions)
        normals =  mesh_data.get_vertex_normals()
        col = np.array([1.0, 0.0, 0.0, 1.0])
        colors =  np.tile(col, (mesh_data.n_vertices,1))
        vtype = [('a_position', np.float32, 3),
             ('a_normal', np.float32, 3),
             ('a_color', np.float32, 4)]
        vertex = np.zeros(mesh_data.n_vertices, dtype=vtype)
        vertex['a_position'] = positions
        vertex['a_normal'] = normals
        vertex['a_color'] = colors
        indices = mesh_data.get_faces()
        vbo = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(indices)
        self.cylinder_program.bind(vbo)
        self.cylinder_program["u_model"] = self.cylinder_model
        self.cylinder_program["u_view"] = self.view

        # projection 
        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        projection = perspective(self.fovy, width / float(height), self.z_near, self.z_far)
        self.cylinder_program['u_projection'] = projection
        self.floor_program['u_projection'] = projection

        # hide cursor
        self.native.setCursor(Qt.BlankCursor)

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_mouse_move(self,event):

        w, h = self.physical_size
        x, y = event.pos

        if self.last_mouse_pos is not None:
            x0, y0 = self.last_mouse_pos
        else:
            self.last_mouse_pos = (w//2, h//2)
            self.native.cursor().setPos(self.native.mapToGlobal(QPoint(w//2,h//2)))
            return

        dx = x-x0
        dy = y-y0
        self.cam_theta += self.step_r*dx
        self.cam_phi = max(min(self.cam_phi + self.step_r*dy, 90), -90)

        self.view = translate((self.cam_x, self.cam_y, self.cam_z)).dot(rotate(self.cam_phi, (1, 0, 0))).dot(rotate(self.cam_theta, (0, 1, 0)))
        self.cylinder_program['u_view'] = self.view      
        self.floor_program['u_view'] = self.view

        self.native.cursor().setPos(self.native.mapToGlobal(QPoint(w//2,h//2))) 

        print(f'Theta: {self.cam_theta}, phi: {self.cam_phi}, X: {self.cam_x}, Y: {self.cam_y}, Z: {self.cam_z}')

    def on_key_press(self, event):

        tx = 0
        ty = 0
        tz = 0

        if event.text == 'w':
            tz = self.step_t
        elif event.text == 'a':
            tx = self.step_t
        elif event.text == 's':
            tz = -self.step_t
        elif event.text == 'd':
            tx = -self.step_t
        if event.text == 'W':
            tz = self.step_t_fast
        elif event.text == 'A':
            tx = self.step_t_fast
        elif event.text == 'S':
            tz = -self.step_t_fast
        elif event.text == 'D':
            tx = -self.step_t_fast

        # move in the direction of the view 
        tx,ty,tz,_ = rotate(self.cam_phi, (1, 0, 0)).dot(rotate(self.cam_theta, (0, 1, 0))) @ np.array((tx,ty,tz,1.0))

        self.cam_x += tx
        self.cam_y += ty
        self.cam_z += tz

        self.view = translate((self.cam_x, self.cam_y, self.cam_z)).dot(rotate(self.cam_phi, (1, 0, 0))).dot(rotate(self.cam_theta, (0, 1, 0)))
        self.cylinder_program['u_view'] = self.view
        self.floor_program['u_view'] = self.view

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        self.cylinder_program.draw('triangles', self.indices)
        self.floor_program.draw('triangle_strip')

    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        self.cylinder_model = rotate(self.theta, (0, 1, 0)).dot(rotate(self.phi, (0, 0, 1)))
        self.cylinder_program['u_model'] = self.cylinder_model
        self.update()

if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
