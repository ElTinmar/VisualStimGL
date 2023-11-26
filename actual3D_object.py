import sys
from vispy import gloo, app
from vispy.geometry import create_cylinder
from vispy.util.transforms import perspective, translate, rotate
import numpy as np
import time
from functools import partial
from PyQt5.QtCore import QPoint

VERT_SHADER ="""
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

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;
FRAG_SHADER = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""

# TODO use instance rendering to display multiple cylinders
# TODO use keyboard+mouse to control the view matrix

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1280,720), fullscreen=False, keys='interactive')

        mesh_data = create_cylinder(rows=10, cols = 36)
        self.phi, self.theta = 60, 20
        self.step_t = 0.1
        self.step_r = 1.0
        self.cam_x = 0
        self.cam_y = 0
        self.cam_z = -5
        self.cam_phi = 0
        self.cam_theta = 0
        self.z_near = 0.1
        self.z_far = 100
        self.fovy = 90
        self.view = translate((self.cam_x, self.cam_y, self.cam_z))
        self.model = rotate(self.theta, (0, 0, 1)).dot(rotate(self.phi, (0, 1, 0)))

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        

        positions = mesh_data.get_vertices()
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
        self.program.bind(vbo)
        self.program["u_model"] = self.model
        self.program["u_view"] = self.view

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        projection = perspective(self.fovy, width / float(height), self.z_near, self.z_far)
        self.program['u_projection'] = projection

        gloo.set_clear_color('white')
        gloo.set_state('opaque')

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_mouse_move(self,event):

        # find angle between current view rotation and (x,y,inf)
        w, h = self.physical_size
        x, y = event.pos

        #self.cam_theta = self.step_r * np.rad2deg(np.arctan2((x-w/2)/(w/2), self.z_near))
        #self.cam_phi = self.step_r * np.rad2deg(np.arctan2((y-h/2)/(h/2), self.z_near))
        aspect = w / float(h)
        fovx = self.fovy * aspect
        self.cam_theta = self.step_r * fovx/2 * (x-w/2)/(w/2)
        self.cam_phi = self.step_r * self.fovy/2 * (y-h/2)/(h/2)

        self.view = translate((self.cam_x, 0, self.cam_z)).dot(rotate(self.cam_phi, (1, 0, 0))).dot(rotate(self.cam_theta, (0, 1, 0)))
        self.program['u_view'] = self.view      

        # self.native.cursor().setPos(self.native.mapToGlobal(QPoint(w/2,h/2))) # lots of jitter, probably emits mouse event imediately 


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

        # move in the direction of the view 
        tx,ty,tz,_ = rotate(self.cam_phi, (1, 0, 0)).dot(rotate(self.cam_theta, (0, 1, 0))) @ np.array((tx,ty,tz,1.0))

        self.cam_x += tx
        self.cam_y += ty
        self.cam_z += tz

        self.view = translate((self.cam_x, 0, self.cam_z)).dot(rotate(self.cam_phi, (1, 0, 0))).dot(rotate(self.cam_theta, (0, 1, 0)))
        self.program['u_view'] = self.view

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear()
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program.draw('triangles', self.indices)

    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        self.model = rotate(self.theta, (0, 1, 0)).dot(rotate(self.phi, (0, 0, 1)))
        self.program['u_model'] = self.model
        
        self.update()

if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
