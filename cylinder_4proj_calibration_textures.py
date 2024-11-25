import sys
from vispy import gloo, app,  use 
from vispy.geometry import create_cylinder
from vispy.util.transforms import perspective, translate, rotate, frustum
import numpy as np
from PyQt5.QtCore import QPoint, Qt 
from typing import Tuple

# TODO for the calibration, the object is the walls of the cylinder. 
# if fish moves inside, nothing changes because object is on the walls.
# if fish outside, it breaks, because fish can't be outside.
# Remove fish and navigation alltogether for calibration ?    

# TODO can I match perspective frustum with projector offset ? 

# TODO identify calibration parameters
#   - radius
#   - position of each proj
#   - angles of each proj
#   - fovy  

# TODO let the master control calibration parameters ?

def cylinder_texcoords(rows, cols):
    texcoords = np.empty((rows+1, cols, 2), dtype=np.float32)
    texcoords[..., 0] = np.linspace(0, 1, num=rows+1, endpoint=True).reshape(rows+1,1)
    texcoords[..., 1] = np.linspace(0, 1, num=cols, endpoint=True)
    texcoords = texcoords.reshape((rows+1)*cols, 2)
    return texcoords

def checkerboard(height=256, width=256, grid_num=8, aspect_ratio=1):
    grid_size = height // grid_num
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = ((xv // grid_size) + (aspect_ratio*yv // grid_size)) % 2
    return 255*out.astype(np.uint8)

def vertical_lines(height=1024, width=1024, line_num=4, thickness=3, offset=0):
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = ((yv+offset) % (width//line_num)) < thickness
    return 255*out.astype(np.uint8)

def unit_grid(height=1024, width=1024, radius=1.0, length=1.0, thickness=3, gridsize_mm=10, offset_x=0, offset_y=0):
    aspect_ratio = (2*np.pi*radius)/length
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = ((yv+offset_y) % (gridsize_mm*width/(2*np.pi*radius)) < thickness) | ((xv+offset_x) % (gridsize_mm*height/length) < thickness*aspect_ratio) 
    return 255*out.astype(np.uint8)
 
use(gl='gl+')

VERT_SHADER_CYLINDER ="""
// uniforms
uniform sampler2D texture;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// per-vertex attributes
attribute vec3 a_position;
attribute vec2 texcoord;
attribute vec3 a_fish;
attribute float a_cylinder_radius;
attribute vec4 a_color;

// varying
varying vec4 v_color;
varying float v_depth;
varying vec2 v_texcoord;


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
    //dir > 0.0f ? sol = sol0 : sol = sol1;
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
    vec4 vertex_coords = u_model * vec4(a_position,1.0);
    vec3 proj = cylinder_proj(a_fish, vertex_coords.xyz, a_cylinder_radius);

    // view and projection
    gl_Position = u_projection * u_view * vec4(proj,1.0);
    v_color = a_color;

    // send depth info
    vec4 position = u_projection * u_view * vertex_coords;
    v_depth = position.z/position.w;
    v_texcoord = texcoord;
}
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;
FRAG_SHADER_CYLINDER = """
uniform sampler2D texture_vertical_bars;
uniform sampler2D texture_grid;
varying vec2 v_texcoord;

uniform vec2 u_resolution;
uniform float u_blend_width;

varying vec4 v_color;
varying float v_depth;

float edge_blending(vec2 pos, float width) 
{
    return smoothstep(0.0, width, pos.x) * smoothstep(0.0, width, 1.0 - pos.x);
}

void main()
{
    vec4 grid_color = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 bar_color = vec4(0.0, 1.0, 1.0, 1.0);
    gl_FragColor = (
        bar_color * texture2D(texture_vertical_bars, v_texcoord) 
        + grid_color * texture2D(texture_grid, v_texcoord)
    ) * edge_blending(gl_FragCoord.xy/u_resolution, u_blend_width);
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
            ty: float = 0,
            tz: float = 0,
            yaw: float = 0,
            pitch: float = 0,
            roll: float = 0,
            radius_mm: float = 100,
            height_mm: float = 50,
            fovy: float = 60,
            blend_width: float = 0.2
        ):

        app.Canvas.__init__(
            self, 
            size = window_size, 
            position = window_position, 
            fullscreen = fullscreen, 
            decorate = False,
            keys = 'interactive'
        )

        mesh_data = create_cylinder(
            rows = int(height_mm//10), 
            cols = int(2*np.pi*radius_mm//10), 
            radius= (radius_mm,radius_mm),
            length = height_mm 
        )
        texcoord = cylinder_texcoords(
            rows = int(height_mm//10), 
            cols = int(2*np.pi*radius_mm//10)
        )

        # cylinder
        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        positions = mesh_data.get_vertices()
        positions = np.hstack((positions, np.ones((mesh_data.n_vertices,1))))
        positions = positions.dot(rotate(-90, (1,0,0)))
        positions = positions[:,:-1]
        col = np.array([1.0, 1.0, 0.0, 1.0])
        colors =  np.tile(col, (mesh_data.n_vertices,1))

        vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4),
             ('texcoord', np.float32, 2)]
        vertex = np.zeros(mesh_data.n_vertices, dtype=vtype)
        vertex['a_position'] = positions
        vertex['a_color'] = colors
        vertex['texcoord'] = texcoord
        
        indices = mesh_data.get_faces()
        vbo = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(indices)
        self.cylinder_program.bind(vbo)
        self.cylinder_program['a_fish'] = [0,0,0]
        self.cylinder_program['a_cylinder_radius'] = radius_mm
        self.cylinder_program['u_blend_width'] = blend_width
        self.cylinder_program['texture_vertical_bars'] = vertical_lines()
        self.cylinder_program['texture_grid'] = unit_grid(radius=radius_mm, length=height_mm, offset_x=0, offset_y=10)

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.cylinder_program['u_resolution'] = [width, height]

        # oblique frustum to match projector offset
        aspect_ratio = width / float(height)
        znear = 1
        zfar = 10_000
        top = np.tan(np.deg2rad(fovy)) * znear
        bottom = 0
        right = top/2 * aspect_ratio
        left = -right
        projection = frustum(left, right, bottom, top, znear, zfar)

        u_view = translate((tx,ty,tz)).dot(rotate(yaw, (0,1,0))).dot(rotate(roll, (0,0,1))).dot(rotate(pitch, (1,0,0)))

        self.cylinder_program["u_view"] = u_view
        self.cylinder_program["u_model"] = translate((0,0,0))
        self.cylinder_program['u_projection'] = projection
        
        # required for object in the Z axis to hide each other
        gloo.set_state(depth_test=True) 

        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.cylinder_program.draw('triangles', self.indices)

    def set_state(self, x, y, z):
        #self.cylinder_program['a_fish'] = [x, y, z]
        print(x,y)
        self.cylinder_program['texture_grid'] = unit_grid(radius=radius_mm, length=height_mm, offset_x=x, offset_y=y) 
        self.update()


class Master(app.Canvas):
    def __init__(
            self, 
            slaves,
            radius_mm: float = 100,
            height_mm: float = 50, 
            blend_width: float = 0.2
        ):

        app.Canvas.__init__(
            self, 
            size = (800,600), 
            position = (0,0), 
            fullscreen = False, 
            keys = 'interactive'
        )

        self.slaves = slaves

        # mesh
        mesh_data = create_cylinder(
            rows = int(height_mm//10), 
            cols = int(2*np.pi*radius_mm//10), 
            radius = (radius_mm,radius_mm), 
            length = height_mm
        )
        texcoord = cylinder_texcoords(
            rows = int(height_mm//10), 
            cols = int(2*np.pi*radius_mm//10)
        )

        # rotation and translation gain
        self.step_t = 1
        self.step_t_fast = 2
        self.step_r = 0.1

        # camera location and rotation
        self.cam_x = 0
        self.cam_y = 0
        self.cam_z = 0
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        # NOTE: Euler angles suffer from gimbal lock, the order of euler rotation matters. It's ok here cause I never roll.
        # You can get rid of gimbal lock with quaternions

        # perspective frustum
        self.z_near = 1
        self.z_far = 10_000
        self.fovy = 90

        # store last mouse position
        self.last_mouse_pos = None

        # cylinder
        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        positions = mesh_data.get_vertices()
        positions = np.hstack((positions, np.ones((mesh_data.n_vertices,1))))
        positions = positions.dot(rotate(-90, (1,0,0)))
        positions = positions[:,:-1]
        col = np.array([1.0, 1.0, 0.0, 1.0])
        colors =  np.tile(col, (mesh_data.n_vertices,1))
        colors[positions[:,0]<0] = np.array([0.0, 1.0, 0.0, 1.0])

        vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4),
             ('texcoord', np.float32, 2)]
        vertex = np.zeros(mesh_data.n_vertices, dtype=vtype)
        vertex['a_position'] = positions
        vertex['a_color'] = colors
        vertex['texcoord'] = texcoord
        indices = mesh_data.get_faces()
        vbo = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(indices)
        self.cylinder_program.bind(vbo)
        self.cylinder_program['a_fish'] = [self.cam_x, self.cam_y, self.cam_z]
        self.cylinder_program['a_cylinder_radius'] = radius_mm
        self.cylinder_program['u_blend_width'] = blend_width
        self.cylinder_program['texture_vertical_bars'] = vertical_lines()
        self.cylinder_program['texture_grid'] = unit_grid(radius=radius_mm, length=height_mm)

        # model, view, projection 
        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z)).dot(rotate(self.cam_yaw, (0, 1, 0))).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0)))
        self.cylinder_model = translate((0,0,0))

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.cylinder_program['u_resolution'] = [width, height]

        projection = perspective(self.fovy, width / float(height), self.z_near, self.z_far)

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
            #print(f'Yaw: {self.cam_yaw}, Pitch: {self.cam_pitch}, Roll: {self.cam_roll}, X: {self.cam_x}, Y: {self.cam_y}, Z: {self.cam_z}')
            
            for slave in self.slaves:
                slave.set_state(self.cam_x, self.cam_y, self.cam_z)

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.cylinder_program.draw('triangles', self.indices)
        self.update()

    def on_close(self, event):
        for slave in self.slaves:
            slave.close()

if __name__ == '__main__':

    radius_mm = 33.7
    height_mm = 100
    fovy = 35.5
    blend_width = 0.4
    proj_distance_mm = 170

    proj0 = Slave(
        window_size = (800,600),
        window_position = (1920,0),
        fullscreen = False,
        tx = 0,
        ty = 0,
        tz = proj_distance_mm,
        yaw = 180,
        pitch = 0,
        roll = 0,
        radius_mm = radius_mm,
        height_mm = height_mm,
        fovy = fovy,
        blend_width = blend_width
    )
    proj1 = Slave(
        window_size = (800,600),
        window_position = (2720,0),
        fullscreen = False,
        tx = proj_distance_mm,
        ty = 0,
        tz = 0,
        yaw = 90,
        pitch = 0,
        roll = 0,
        radius_mm = radius_mm,
        height_mm = height_mm,
        fovy = fovy,
        blend_width = blend_width
    )
    proj2 = Slave(
        window_size = (800,600),
        window_position = (3520,0),
        fullscreen = False,
        tx = 0,
        ty = 0,
        tz = -proj_distance_mm,
        yaw = 0,
        pitch = 0,
        roll = 0,
        radius_mm = radius_mm,
        height_mm = height_mm,
        fovy = fovy,
        blend_width = blend_width
    )
    proj3 = Slave(
        window_size = (800,600),
        window_position = (4320,0),
        fullscreen = False,
        tx = -proj_distance_mm,
        ty = 0,
        tz = 0,
        yaw = 270,
        pitch = 0,
        roll = 0,
        radius_mm = radius_mm,
        height_mm = height_mm,
        fovy = fovy,
        blend_width = blend_width
    )

    master = Master(
        slaves = [proj0, proj1, proj2, proj3],
        radius_mm = radius_mm,
        height_mm = height_mm,
        blend_width = blend_width
    )

    if sys.flags.interactive != 1:
        app.run()
    
