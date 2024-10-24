import sys
from vispy import gloo, app,  use 
from vispy.geometry import create_cylinder
from vispy.util.transforms import perspective, translate, rotate
import numpy as np
from PyQt5.QtCore import QPoint, Qt 
from typing import Tuple

# we need full gl context for instanced rendering (this requires PyOpenGL: pip install PyOpenGL PyOpenGL_accelerate)
use(gl='gl+')

VERT_SHADER_CYLINDER ="""
// uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// per-vertex attributes
attribute vec3 a_position;
attribute vec3 a_fish;
attribute float a_cylinder_radius;
attribute vec4 a_color;

// per-instance attributes
attribute vec3 instance_shift;

// varying
varying vec4 v_color;
varying float v_depth;

// TODO check what happens when fish_pos = vertex_pos
vec3 cylinder_proj(vec3 fish_pos, vec3 vertex_pos, float cylinder_radius) { 
    // freely-swimming: fish is not at (0,0,0) but moves in world coordinates
    // we compute the intersection of the line between fish coords and vertex coords 
    // with a cylinder of radius <cylinder_radius>.
    // With the hypothesis that the fish always stays inside the cylinder,
    // there will always be at least 1 solution.

    // vertex coords
    float x_v = vertex_pos.x;
    float y_v = vertex_pos.y;
    float z_v = vertex_pos.z;

    // fish coords
    float x_f = fish_pos.x;
    float y_f = fish_pos.y;
    float z_f = fish_pos.z;

    // cylinder radius
    float r = cylinder_radius;

    // helpful variables
    float x_ = x_f - x_v;
    float y_ = y_f - y_v;
    float z_ = z_f - z_v;
    float xz_ = (x_f*z_v - x_v*z_f);
    float xy_ = (x_f*y_v - x_v*y_f);
    float d = x_*x_ + z_*z_;
    float s = sqrt(r*r*d - xz_*xz_);

    // projection to cylinder (two solutions)
    float x0 = 1/d * (x_v*(z_f*z_ + s) - x_f*(z_v*z_ + s));
    float x1 = 1/d * (x_v*(z_f*z_ - s) - x_f*(z_v*z_ - s));
    float y0 = 1/d * (x_*xy_ + y_v * (z_f*z_ + s) - y_f * (z_v*z_ + s));
    float y1 = 1/d * (x_*xy_ + y_v * (z_f*z_ - s) - y_f * (z_v*z_ - s));
    float z0 = 1/d * (x_*xz_ - z_*s);
    float z1 = 1/d * (x_*xz_ + z_*s);

    // find correct solution: fish->vertex and fish->cylinder vectors should be in same direction
    vec3 sol0 = vec3(x0, y0, z0);
    vec3 sol1 = vec3(x1, y1, z1);
    vec3 sol;
    float dir = dot(sol0-fish_pos, vertex_pos-fish_pos);
    if (dir > 0.0f) {
        sol = sol0;
    }
    else {
        sol = sol1;
    }
    return sol;
} 

void main()
{
    vec4 vertex_coords = u_model * vec4(a_position+instance_shift,1.0);
    vec3 proj = cylinder_proj(a_fish, vertex_coords.xyz, a_cylinder_radius);

    // view and projection
    gl_Position = u_projection * u_view * vec4(proj,1.0);
    v_color = a_color;

    // send depth info
    vec4 position = u_projection * u_view * vertex_coords;
    v_depth = position.z/position.w;
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
varying float v_depth;
void main()
{
    gl_FragColor = v_color;
    gl_FragDepth = v_depth; // this disables early depth testing and comes at a perf cost
}
"""

FRAG_SHADER_FLOOR = """
void main()
{
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""

class Slave(app.Canvas):
    '''
    Side view, what needs to be projected (need to calibrate)
    '''

    def __init__(
            self,
            window_size: Tuple[int, int] = (1920, 1080), 
            window_position: Tuple[int, int] = (0,0), 
            fullscreen: bool = True,
            tx: float = 0,
            ty: float = -0.1,
            tz: float = -20,
            yaw: float = 0,
            pitch: float = 0,
            roll: float = 0,
        ):

        app.Canvas.__init__(
            self, 
            size=window_size, 
            position=window_position, 
            fullscreen=fullscreen, 
            decorate=False,
            keys='interactive'
        )

        mesh_data = create_cylinder(rows=10, cols = 36)
        
        # floor
        self.floor_program = gloo.Program(VERT_SHADER_FLOOR, FRAG_SHADER_FLOOR)
        self.floor_program["a_position"] = [(-100,0,-100),(-100,0,100),(100,0,-100),(100,0,100)]

        # cylinder
        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        positions = mesh_data.get_vertices()
        positions = np.hstack((positions, np.ones((mesh_data.n_vertices,1))))
        positions = positions.dot(rotate(90, (1,0,0)))
        positions = positions[:,:-1]
        col = np.array([1.0, 0.0, 0.0, 1.0])
        colors =  np.tile(col, (mesh_data.n_vertices,1))
        colors[positions[:,0]<0] = np.array([0.0, 0.0, 1.0, 1.0])

        vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4)]
        vertex = np.zeros(mesh_data.n_vertices, dtype=vtype)
        vertex['a_position'] = positions
        vertex['a_color'] = colors
        indices = mesh_data.get_faces()
        vbo = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(indices)
        self.cylinder_program.bind(vbo)
        self.cylinder_program['a_fish'] = [0,0,5]
        self.cylinder_program['a_cylinder_radius'] = 10

        # instances
        instance_shift = gloo.VertexBuffer([(-2,1,-2),(-2,1,2),(2,1,2),(2,1,-2)], divisor=1)
        self.cylinder_program['instance_shift'] = instance_shift

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        projection = perspective(60, width / float(height), 1, 1000)

        u_view = translate((tx,ty,tz)).dot(rotate(yaw, (0,1,0))).dot(rotate(roll, (0,0,1))).dot(rotate(pitch, (1,0,0)))

        self.floor_program["u_view"] = u_view
        self.floor_program["u_model"] = translate((0,0,0))
        self.floor_program['u_projection'] = projection

        self.cylinder_program["u_view"] = u_view
        self.cylinder_program["u_model"] = translate((0,0,0))
        self.cylinder_program['u_projection'] = projection
        
        # required for object in the Z axis to hide each other
        gloo.set_state(depth_test=True) 

        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.floor_program.draw('triangle_strip')
        self.cylinder_program.draw('triangles', self.indices)

    def set_state(self, x, y, z):
        self.cylinder_program['a_fish'] = [x, y, z]
        self.update()


class Master(app.Canvas):
    def __init__(self, slaves):
        app.Canvas.__init__(self, size=(800,600), position=(0,0), fullscreen=False, keys='interactive')

        self.slaves = slaves

        # mesh
        mesh_data = create_cylinder(rows=10, cols = 36)

        # rotation and translation gain
        self.step_t = 0.1
        self.step_t_fast = 0.2
        self.step_r = 0.1

        # camera location and rotation
        self.cam_x = 0
        self.cam_y = 0.1
        self.cam_z = 5
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        # NOTE: Euler angles suffer from gimbal lock, the order of euler rotation matters. It's ok here cause I never roll.
        # You can get rid of gimbal lock with quaternions

        # perspective frustum
        self.z_near = 0.1
        self.z_far = 1000
        self.fovy = 90

        # store last mouse position
        self.last_mouse_pos = None

        # floor
        self.floor_program = gloo.Program(VERT_SHADER_FLOOR, FRAG_SHADER_FLOOR)
        self.floor_program["a_position"] = [(-100,0,-100),(-100,0,100),(100,0,-100),(100,0,100)]

        # cylinder
        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        positions = mesh_data.get_vertices()
        positions = np.hstack((positions, np.ones((mesh_data.n_vertices,1))))
        positions = positions.dot(rotate(90, (1,0,0)))
        positions = positions[:,:-1]
        col = np.array([1.0, 0.0, 0.0, 1.0])
        colors =  np.tile(col, (mesh_data.n_vertices,1))
        colors[positions[:,0]<0] = np.array([0.0, 0.0, 1.0, 1.0])

        vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4)]
        vertex = np.zeros(mesh_data.n_vertices, dtype=vtype)
        vertex['a_position'] = positions
        vertex['a_color'] = colors
        indices = mesh_data.get_faces()
        vbo = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(indices)
        self.cylinder_program.bind(vbo)
        self.cylinder_program['a_fish'] = [self.cam_x, self.cam_y, self.cam_z]
        self.cylinder_program['a_cylinder_radius'] = 10

        # instances
        instance_shift = gloo.VertexBuffer([(-2,1,-2),(-2,1,2),(2,1,2),(2,1,-2)], divisor=1)
        self.cylinder_program['instance_shift'] = instance_shift

        # model, view, projection 
        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z)).dot(rotate(self.cam_yaw, (0, 1, 0))).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0)))
        self.cylinder_model = translate((0,0,0))
        self.floor_model = translate((0,0,0))

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        projection = perspective(self.fovy, width / float(height), self.z_near, self.z_far)

        self.floor_program["u_view"] = self.view
        self.floor_program["u_model"] = self.floor_model
        self.floor_program['u_projection'] = projection

        self.cylinder_program["u_view"] = self.view
        self.cylinder_program["u_model"] = self.cylinder_model
        self.cylinder_program['u_projection'] = projection
        
        # required for object in the Z axis to hide each other
        gloo.set_state(depth_test=True) 

        # hide cursor
        self.native.setCursor(Qt.BlankCursor)

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
        self.cam_yaw += self.step_r*dx
        #self.cam_pitch = max(min(self.cam_pitch + self.step_r*dy, 90), -90)

        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z)).dot(rotate(self.cam_yaw, (0, 1, 0))).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0)))
        self.cylinder_program['u_view'] = self.view      
        self.floor_program['u_view'] = self.view

        self.native.cursor().setPos(self.native.mapToGlobal(QPoint(w//2,h//2))) 

        #print(f'Yaw: {self.cam_yaw}, Pitch: {self.cam_pitch}, Roll: {self.cam_roll}, X: {self.cam_x}, Y: {self.cam_y}, Z: {self.cam_z}')

    def on_key_press(self, event):

        tx = 0
        ty = 0
        tz = 0
    
        if len(event.modifiers) > 0 and event.modifiers[0] == 'Shift':
            step = self.step_t_fast
        else:
            step = self.step_t

        if event.key == 'W':
            tz = -1
        elif event.key == 'A':
            tx = -1
        elif event.key == 'S':
            tz = 1
        elif event.key == 'D':
            tx = 1
        elif event.key == 'Up':
            ty = 1
        elif event.key == 'Down':
            ty = -1

        # move in the direction of the view
        t_vec = np.array((tx,ty,tz))
        norm = np.linalg.norm(t_vec)
        if norm>0:
            t_vec = step * t_vec/norm
            tx,ty,tz,_ = rotate(self.cam_yaw, (0, 1, 0)).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0))) @ np.hstack((t_vec,1.0))

            self.cam_x += tx
            self.cam_y += ty
            self.cam_z += tz

            self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z)).dot(rotate(self.cam_yaw, (0, 1, 0))).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0)))
            self.cylinder_program['u_view'] = self.view
            self.cylinder_program['a_fish'] = [self.cam_x, self.cam_y, self.cam_z]
            self.floor_program['u_view'] = self.view
            #print(f'Yaw: {self.cam_yaw}, Pitch: {self.cam_pitch}, Roll: {self.cam_roll}, X: {self.cam_x}, Y: {self.cam_y}, Z: {self.cam_z}')
            
            for slave in self.slaves:
                slave.set_state(self.cam_x, self.cam_y, self.cam_z)

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.floor_program.draw('triangle_strip')
        self.cylinder_program.draw('triangles', self.indices)
        self.update()

    def on_close(self, event):
        for slave in self.slaves:
            slave.close()

if __name__ == '__main__':

    proj0 = Slave(
        window_size = (800,600),
        window_position = (1080,0),
        fullscreen = False,
        tx = 0,
        ty = -0.1,
        tz = -20,
        yaw = 0,
        pitch = 0,
        roll = 0
    )
    proj1 = Slave(
        window_size = (800,600),
        window_position = (1680,0),
        fullscreen = False,
        tx = 20,
        ty = -0.1,
        tz = 0,
        yaw = 90,
        pitch = 0,
        roll = 0
    )
    proj2 = Slave(
        window_size = (800,600),
        window_position = (2280,0),
        fullscreen = False,
        tx = 0,
        ty = -0.1,
        tz = 20,
        yaw = 180,
        pitch = 0,
        roll = 0
    )
    proj3 = Slave(
        window_size = (1280,800),
        window_position = (2880,0),
        fullscreen = False,
        tx = -20,
        ty = -0.1,
        tz = 0,
        yaw = 270,
        pitch = 0,
        roll = 0
    )

    master = Master([proj0, proj1, proj2, proj3])

    if sys.flags.interactive != 1:
        app.run()
    
