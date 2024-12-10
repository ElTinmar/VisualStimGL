import sys
from vispy import gloo, app,  use 
from vispy.geometry import create_box
from vispy.util.transforms import perspective, translate, rotate, frustum, ortho
from vispy.io import imread, read_mesh
import numpy as np
from PyQt5.QtCore import QPoint, Qt 
from typing import Tuple

# TODO for the calibration, the object is the walls of the cylinder. 
# if fish moves inside, nothing changes because object is on the walls.
# if fish outside, it breaks, because fish can't be outside.
# Remove fish and navigation alltogether for calibration ?    

def lookAt(eye, target, up=[0, 1, 0]):
    """Computes matrix to put eye looking at target point."""

    eye = np.asarray(eye).astype(np.float32)
    target = np.asarray(target).astype(np.float32)
    up = np.asarray(up).astype(np.float32)

    vforward = eye - target
    vforward /= np.linalg.norm(vforward)

    # if up and vforward vectors are collinear, choose a fallback vector
    up_normalized = up / np.linalg.norm(up)
    if abs(np.dot(up_normalized, vforward)) > 0.9999:
        up = np.zeros(3, dtype=np.float32)
        up[np.argmin(abs(vforward))] = 1

    vright = np.cross(up, vforward)
    vright /= np.linalg.norm(vright)
    vup = np.cross(vforward, vright)

    view = np.r_[vright, -np.dot(vright, eye),
                 vup, -np.dot(vup, eye),
                 vforward, -np.dot(vforward, eye),
                 [0, 0, 0, 1]].reshape(4, 4, order='F')

    return view


## UV mapping
def cylinder_texcoords(rows, cols):
    texcoords = np.empty((rows+1, cols, 2), dtype=np.float32)
    texcoords[..., 0] = np.linspace(0, 1, num=rows+1, endpoint=True).reshape(rows+1,1)
    texcoords[..., 1] = np.linspace(0, 1, num=cols, endpoint=True)
    texcoords = texcoords.reshape((rows+1)*cols, 2)
    return texcoords

## Texture generation
def checkerboard(height=256, width=256, grid_num=8, aspect_ratio=1):
    grid_size = height // grid_num
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = ((xv // grid_size) + (aspect_ratio*yv // grid_size)) % 2
    return 255*out.astype(np.uint8)

def vertical_lines(height=1024, width=1024, line_num=4, thickness=3, offset=0):
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = ((yv+offset) % (width//line_num)) < thickness
    return 255*out.astype(np.uint8)

def unit_grid(height=1024, width=1024, radius=1.0, length=1.0, thickness_mm=0.5, gridsize_mm=10, offset_x=0, offset_y=0):
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = ((yv+offset_y) % (gridsize_mm*width/(2*np.pi*radius)) < (thickness_mm * width/(2*np.pi*radius))) | ((xv+offset_x) % (gridsize_mm*height/length) < (thickness_mm * height/length)) 
    return 255*out.astype(np.uint8)
 
def two_colors(height=1024, width=1024):
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = np.dstack(((yv < width//2), (yv >= width//2), (yv<0)))
    return 255*out.astype(np.uint8)

def black(height=1024, width=1024):
    xv, yv = np.meshgrid(range(width), range(height), indexing='xy')
    out = np.dstack(((yv < 0), (yv < 0), (yv>=0)))
    return 255*out.astype(np.uint8)

use(gl='gl+')

VERT_SHADER_CYLINDER = """
#version 150
  
// uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_lightspace;

uniform vec3 u_fish;
uniform float u_cylinder_radius;
uniform float u_master;

// per-vertex attributes
attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;

// varying
varying float v_depth;
varying vec2 v_texcoord;
varying vec3 v_normal_world;
varying vec3 v_view_position;
varying vec4 v_world_position;
varying vec4 v_lightspace_position;

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
    //dir >= 0.0f ? sol = sol0 : sol = sol1; 
    if (dir > 0.0f) {sol = sol0;}
    else {sol = sol1;}

    return sol;
} 

void main()
{
    vec4 vertex_world = u_model * vec4(a_position, 1.0);
    vec3 screen_world = cylinder_proj(u_fish, vertex_world.xyz, u_cylinder_radius);
    vec3 normal_world = transpose(inverse(mat3(u_model))) * a_normal;
    vec4 screen_clip = u_projection * u_view * vec4(screen_world, 1.0);

    vec3 viewpoint_world = vec3(inverse(u_view)[3]);
    float magnitude = length(vertex_world.xyz - u_fish)/length(screen_world - u_fish);
    vec3 direction = normalize(screen_world-viewpoint_world);
    float orientation = sign(dot(direction.xz, screen_world.xz));

    vec3 offset_world = viewpoint_world;
    if (u_master == 1) {offset_world += orientation * direction * magnitude;}
    else {offset_world -= orientation * direction * magnitude;}
    vec4 offset_clip = u_projection * u_view * vec4(offset_world, 1.0);

    v_depth = offset_clip.z/offset_clip.w;
    v_texcoord = a_texcoord;
    v_normal_world = normal_world;
    v_view_position = viewpoint_world;
    v_world_position = vertex_world;
    v_lightspace_position = u_lightspace * vertex_world;
    gl_Position = screen_clip;
}
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;
FRAG_SHADER_CYLINDER = """
#version 150

uniform sampler2D u_texture;
uniform sampler2D u_shadow_map_texture;
uniform vec2 u_resolution;
uniform vec3 u_fish;
uniform vec3 u_light_position;
uniform float u_master;

varying vec3 v_normal_world;
varying vec2 v_texcoord;
varying float v_depth;
varying vec3 v_view_position;
varying vec4 v_world_position;
varying vec4 v_lightspace_position;

float get_shadow(vec4 lightspace_position,  vec3 norm, vec3 light_direction)
{
    float bias = mix(0.005, 0.0, dot(norm, light_direction));    

    vec3 position_ndc = lightspace_position.xyz / lightspace_position.w;
    position_ndc = position_ndc * 0.5 + 0.5;
    
    float closest_depth = texture2D(u_shadow_map_texture, position_ndc.xy).r; 
    float current_depth = position_ndc.z;
    float shadow = 0.0;

    if ( position_ndc.z > 1.0 || 
        position_ndc.x < 0.0 || position_ndc.x > 1.0 ||
        position_ndc.y < 0.0 || position_ndc.y > 1.0
    ) {
        return shadow;
    }

    vec2 texelSize = 1.0 / textureSize(u_shadow_map_texture, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture2D(u_shadow_map_texture, position_ndc.xy + vec2(x, y) * texelSize).r; 
            shadow += (current_depth - bias) > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;

    return shadow;
}

vec4 Blinn_Phong(vec3 object_color, vec3 normal, vec3 fragment_position, vec3 view_position, vec3 light_position, vec4 lightspace_position) {

    float enable_shadows = 1.0;

    vec3 ambient_color = vec3(1.0, 1.0, 1.0);
    vec3 diffuse_color = vec3(1.0, 1.0, 1.0);
    vec3 specular_color = vec3(1.0, 1.0, 1.0);

    // ambient
    float light_ambient = 0.1;
    vec3 ambient = light_ambient * ambient_color;

    // diffuse
    vec3 norm = normalize(normal); 
    vec3 light_direction = normalize(light_position - fragment_position);  
    float lambertian = max(dot(norm, light_direction), 0.0); 
    vec3 diffuse = lambertian * diffuse_color;

    // specular
    float light_specular = 0.5;
    float light_shininess = 32;

    vec3 view_direction = normalize(view_position - fragment_position);
    vec3 half_vector = normalize(view_direction + light_direction);
    float specular_angle = max(dot(half_vector, normal), 0.0);
    float spec = pow(specular_angle, light_shininess);
    vec3 specular = light_specular * spec * specular_color;  

    // Blinn_Phong shading
    vec3 result;
    if (enable_shadows == 1.0) {
        float shadow = get_shadow(lightspace_position, norm, light_direction);
        result = (ambient + (1.0 - shadow) * (diffuse + specular)) * object_color;
    }
    else {
        result = (ambient + diffuse + specular) * object_color;
    }
    return vec4(result, 1.0);
}

vec4 edge_blending(vec3 object_color, vec2 pos, float start, float stop) 
{
    vec3 result = smoothstep(start, stop, pos.x) * smoothstep(start, stop, 1.0 - pos.x) * object_color;
    return vec4(result, 1.0);
}

void main()
{
    float gamma = 2.2;

    // texture
    vec4 object_color = texture2D(u_texture, v_texcoord);

    // lighting
    vec4 phong_shading = Blinn_Phong(vec3(object_color), v_normal_world, vec3(v_world_position), u_fish, u_light_position, v_lightspace_position);

    // gamma correction    
    vec4 gamma_corrected = phong_shading;
    gamma_corrected.rgb = pow(gamma_corrected.rgb, vec3(1.0/gamma));
    
    // blend projector edges together
    vec4 final = gamma_corrected;
    if (u_master == 0) {
        final = edge_blending(vec3(gamma_corrected), gl_FragCoord.xy/u_resolution, 0.125, 0.35);
    };
    
    // output
    vec3 position_ndc = v_lightspace_position.xyz / v_lightspace_position.w;
    position_ndc = position_ndc * 0.5 + 0.5;
    float closest_depth = texture2D(u_shadow_map_texture, position_ndc.xy).r; 
    gl_FragColor = vec4(vec3(closest_depth), 1.0);

    gl_FragColor = final;
    gl_FragDepth = v_depth;
}
"""

VERTEX_SHADER_SHADOW="""
uniform mat4 u_model;
uniform mat4 u_lightspace;

attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;

void main()
{
    gl_Position = u_lightspace * u_model * vec4(a_position, 1.0);
}
"""

FRAGMENT_SHADER_SHADOW="""
void main()
{
    float depth = gl_FragCoord.z;
    gl_FragColor = vec4(depth,0.0,0.0,1.0);
}
"""

GROUND_OFFSET = 50
SHELL_MODEL = rotate(90,(1,0,0)).dot(rotate(180,(0,0,1))).dot(translate((0,GROUND_OFFSET+0.6,0)))
GROUND_MODEL = translate((0,GROUND_OFFSET,0))
SHADOWMAP_RES = 2048

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
        ):

        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.radius_mm = radius_mm
        self.height_mm = height_mm
        self.fovy = fovy
        
        app.Canvas.__init__(
            self, 
            size = window_size, 
            position = window_position, 
            fullscreen = fullscreen, 
            decorate = False,
            keys = 'interactive'
        )

        self.set_context()
        self.create_view()
        self.create_projection()
        self.create_scene()

        self.light_theta = 0
        self.light_theta_step = 0.01
        self.t = 0
        self.t_step = 1/30
        self.timer = app.Timer(1/30, connect=self.on_timer, start=True)

        self.show()

    def set_context(self):
        self.width, self.height = self.physical_size
        gloo.set_viewport(0, 0, self.width, self.height)
        gloo.set_state(depth_test=True)  # required for object in the Z axis to hide each other

    def create_view(self):
        self.view = translate((self.tx,self.ty,self.tz)).dot(rotate(self.yaw, (0,1,0))).dot(rotate(self.roll, (0,0,1))).dot(rotate(self.pitch, (1,0,0)))

    def create_projection(self):
        aspect_ratio = self.width / float(self.height)
        znear = 0.001
        zfar = 10_000
        top = np.tan(np.deg2rad(self.fovy)) * znear
        bottom = 0
        right = top/2 * aspect_ratio
        left = -right
        self.projection = frustum(left, right, bottom, top, znear, zfar)

    def create_scene(self):

        light_position =  [5,5,5]
        light_projection = ortho(-10,10,-10,10,0.01,20)
        light_view = lookAt(light_position, [0,0,0], [0,1,0])
        lightspace = light_view.dot(light_projection)

        # set up shadow map buffer
        self.shadow_map_texture = gloo.Texture2D(
            data = ((SHADOWMAP_RES, SHADOWMAP_RES, 3)), 
            format = 'rgb',
            interpolation = 'nearest',
            wrapping = 'repeat',
            internalformat = 'rgb32f'
        )

        # attach texture as depth buffer
        self.fbo = gloo.FrameBuffer(color = self.shadow_map_texture)

        ## ground ----------------------------------------------------------------------------
        # load texture
        texture = np.flipud(imread('sand.jpeg'))

        vertices, faces, _ = create_box(width=10, height=10, depth=1, height_segments=100, width_segments=100, depth_segments=10)
        vtype = [
            ('a_position', np.float32, 3),
            ('a_texcoord', np.float32, 2),
            ('a_normal', np.float32, 3)
        ]
        vertex = np.zeros(vertices.shape[0], dtype=vtype)
        vertex['a_position'] = vertices['position']
        vertex['a_texcoord']  = vertices['texcoord']
        vertex['a_normal'] = vertices['normal']
        vbo_ground = gloo.VertexBuffer(vertex)
        self.ground_indices = gloo.IndexBuffer(faces)

        self.shadowmap_ground = gloo.Program(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
        self.shadowmap_ground.bind(vbo_ground)
        self.shadowmap_ground['u_model'] = GROUND_MODEL
        self.shadowmap_ground['u_lightspace'] = lightspace
        
        self.ground_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        self.ground_program.bind(vbo_ground)
        self.ground_program['u_master'] = 0
        self.ground_program['u_fish'] = [0,0,0]
        self.ground_program['u_cylinder_radius'] = radius_mm
        self.ground_program['u_texture'] = texture
        self.ground_program['u_resolution'] = [self.width, self.height]
        self.ground_program['u_view'] = self.view
        self.ground_program['u_model'] = GROUND_MODEL
        self.ground_program['u_projection'] = self.projection
        self.ground_program['u_lightspace'] = lightspace
        self.ground_program['u_light_position'] = light_position
        self.ground_program['u_shadow_map_texture'] = self.shadow_map_texture

        ## shell -----------------------------------------------------------------------------
        # load texture
        texture = np.flipud(imread('quartz.jpg'))

        # load mesh
        vertices, faces, normals, texcoords = read_mesh('shell_simplified.obj')
        vtype = [
            ('a_position', np.float32, 3),
            ('a_texcoord', np.float32, 2),
            ('a_normal', np.float32, 3)
        ]
        vertex = np.zeros(vertices.shape[0], dtype=vtype)
        vertex['a_position'] = vertices
        vertex['a_texcoord']  = texcoords
        vertex['a_normal'] = normals
        vbo_shell = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(faces)

        self.shadowmap_program = gloo.Program(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
        self.shadowmap_program.bind(vbo_shell)
        self.shadowmap_program['u_model'] = SHELL_MODEL
        self.shadowmap_program['u_lightspace'] = lightspace

        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        self.cylinder_program.bind(vbo_shell)
        self.cylinder_program['u_master'] = 0
        self.cylinder_program['u_fish'] = [0,0,0]
        self.cylinder_program['u_cylinder_radius'] = radius_mm
        self.cylinder_program['u_texture'] = texture
        self.cylinder_program['u_resolution'] = [self.width, self.height]
        self.cylinder_program['u_view'] = self.view
        self.cylinder_program['u_model'] = SHELL_MODEL
        self.cylinder_program['u_projection'] = self.projection
        self.cylinder_program['u_lightspace'] = lightspace
        self.cylinder_program['u_light_position'] = light_position
        self.cylinder_program['u_shadow_map_texture'] = self.shadow_map_texture

    def on_draw(self, event):
        # draw to the fbo 
        with self.fbo: 
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, SHADOWMAP_RES, SHADOWMAP_RES)
            gloo.set_cull_face('front')
            self.shadowmap_ground.draw('triangles', self.ground_indices)
            self.shadowmap_program.draw('triangles', self.indices)
            
        # draw to screen
        gloo.clear(color=True, depth=True)
        gloo.set_viewport(0, 0, self.width, self.height)
        gloo.set_cull_face('back')
        self.ground_program.draw('triangles', self.ground_indices)
        self.cylinder_program.draw('triangles', self.indices)

    def set_state(self, x, y, z):
        self.ground_program['u_fish'] = [x, y, z]
        self.cylinder_program['u_fish'] = [x, y, z]
        self.update()

    def on_timer(self, event):
        self.t += self.t_step
        self.light_theta += self.light_theta_step
        light_position =  [5*np.cos(self.light_theta),np.sin(self.t)+GROUND_OFFSET+2,5*np.sin(self.light_theta)]

        light_projection = ortho(-1,1,-1,1,0.1,20)
        light_view = lookAt(light_position, [0,GROUND_OFFSET+0.6,0], [0,1,0])
        lightspace = light_view.dot(light_projection)
        self.shadowmap_ground['u_lightspace'] = lightspace
        self.shadowmap_program['u_lightspace'] = lightspace
        self.ground_program['u_lightspace'] = lightspace
        self.ground_program['u_light_position'] = light_position
        self.cylinder_program['u_lightspace'] = lightspace
        self.cylinder_program['u_light_position'] = light_position
        self.update()

class Master(app.Canvas):
    def __init__(
            self, 
            slaves,
            radius_mm: float = 100,
            height_mm: float = 50, 
        ):

        app.Canvas.__init__(
            self, 
            size = (1280,800), 
            position = (0,0), 
            fullscreen = False, 
            keys = 'interactive'
        )

        self.slaves = slaves

        # rotation and translation gain
        self.step_t = 0.25
        self.step_t_fast = 1
        self.step_r = 0.1

        # camera location and rotation
        self.cam_x = 0
        self.cam_y = GROUND_OFFSET+0.8
        self.cam_z = 0
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        # NOTE: Euler angles suffer from gimbal lock, the order of euler rotation matters. It's ok here cause I never roll.
        # You can get rid of gimbal lock with quaternions

        # perspective frustum
        self.z_near = 0.00001
        self.z_far = 1000
        self.fovy = 90

        # store last mouse position
        self.last_mouse_pos = None

        self.set_context()
        self.create_view()
        self.create_projection()
        self.create_scene()

        # hide cursor
        self.native.setCursor(Qt.BlankCursor)

        self.light_theta = 0
        self.light_theta_step = 0.01
        self.t = 0
        self.t_step = 1/30
        self.timer = app.Timer(1/30, connect=self.on_timer, start=True)

        for slave in self.slaves:
            slave.set_state(self.cam_x, self.cam_y, self.cam_z)

        self.show()

    def set_context(self):
        self.width, self.height = self.physical_size
        gloo.set_viewport(0, 0, self.width, self.height)
        gloo.set_state(depth_test=True)  # required for object in the Z axis to hide each other

    def create_view(self):
        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z)).dot(rotate(self.cam_yaw, (0, 1, 0))).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0)))

    def create_projection(self):
        self.projection = perspective(self.fovy, self.width / float(self.height), self.z_near, self.z_far)

    def create_scene(self):

        light_position =  [5,5,5]
        light_projection = ortho(-10,10,-10,10,0.01,20)
        light_view = lookAt(light_position, [0,0,0], [0,0,1])
        lightspace = light_view.dot(light_projection)

        # set up shadow map buffer
        self.shadow_map_texture = gloo.Texture2D(
            data = ((SHADOWMAP_RES, SHADOWMAP_RES, 3)), 
            format = 'rgb',
            interpolation = 'nearest',
            wrapping = 'repeat',
            internalformat = 'rgb32f'
        )
        # attach texture as depth buffer
        self.fbo = gloo.FrameBuffer(color = self.shadow_map_texture)

        ## ground ----------------------------------------------------------------------------

        # load texture
        texture = np.flipud(imread('sand.jpeg'))

        vertices, faces, _ = create_box(width=10, height=10, depth=1, height_segments=100, width_segments=100, depth_segments=10)
        vtype = [
            ('a_position', np.float32, 3),
            ('a_texcoord', np.float32, 2),
            ('a_normal', np.float32, 3)
        ]
        vertex = np.zeros(vertices.shape[0], dtype=vtype)
        vertex['a_position'] = vertices['position']
        vertex['a_texcoord']  = vertices['texcoord']
        vertex['a_normal'] = vertices['normal']
        vbo_ground = gloo.VertexBuffer(vertex)
        self.ground_indices = gloo.IndexBuffer(faces)

        self.shadowmap_ground = gloo.Program(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
        self.shadowmap_ground.bind(vbo_ground)
        self.shadowmap_ground['u_model'] = GROUND_MODEL
        self.shadowmap_ground['u_lightspace'] = lightspace
        
        self.ground_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        self.ground_program.bind(vbo_ground)
        self.ground_program['u_master'] = 1
        self.ground_program['u_fish'] = [self.cam_x,self.cam_y,self.cam_z]
        self.ground_program['u_cylinder_radius'] = radius_mm
        self.ground_program['u_texture'] = texture
        self.ground_program['u_resolution'] = [self.width, self.height]
        self.ground_program['u_view'] = self.view
        self.ground_program['u_model'] = GROUND_MODEL
        self.ground_program['u_projection'] = self.projection
        self.ground_program['u_lightspace'] = lightspace
        self.ground_program['u_light_position'] = light_position
        self.ground_program['u_shadow_map_texture'] = self.shadow_map_texture

        ## shell -----------------------------------------------------------------------------
        
        # load texture
        texture = np.flipud(imread('quartz.jpg'))

        # load mesh
        vertices, faces, normals, texcoords = read_mesh('shell_simplified.obj')
        vtype = [
            ('a_position', np.float32, 3),
            ('a_texcoord', np.float32, 2),
            ('a_normal', np.float32, 3)
        ]
        vertex = np.zeros(vertices.shape[0], dtype=vtype)
        vertex['a_position'] = vertices
        vertex['a_texcoord']  = texcoords
        vertex['a_normal'] = normals
        vbo_shell = gloo.VertexBuffer(vertex)
        self.indices = gloo.IndexBuffer(faces)

        self.shadowmap_program = gloo.Program(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
        self.shadowmap_program.bind(vbo_shell)
        self.shadowmap_program['u_model'] = SHELL_MODEL
        self.shadowmap_program['u_lightspace'] = lightspace

        self.cylinder_program = gloo.Program(VERT_SHADER_CYLINDER, FRAG_SHADER_CYLINDER)
        self.cylinder_program.bind(vbo_shell)
        self.cylinder_program['u_master'] = 1
        self.cylinder_program['u_fish'] = [self.cam_x,self.cam_y,self.cam_z]
        self.cylinder_program['u_cylinder_radius'] = radius_mm
        self.cylinder_program['u_texture'] = texture
        self.cylinder_program['u_resolution'] = [self.width, self.height]
        self.cylinder_program['u_view'] = self.view
        self.cylinder_program['u_model'] = SHELL_MODEL
        self.cylinder_program['u_projection'] = self.projection
        self.cylinder_program['u_lightspace'] = lightspace
        self.cylinder_program['u_light_position'] = light_position
        self.cylinder_program['u_shadow_map_texture'] = self.shadow_map_texture

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
        self.cam_pitch = max(min(self.cam_pitch + self.step_r*dy, 90), -90)

        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z)).dot(rotate(self.cam_yaw, (0, 1, 0))).dot(rotate(self.cam_roll, (0, 0, 1))).dot(rotate(self.cam_pitch, (1, 0, 0)))
        self.ground_program['u_view'] = self.view
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
            self.ground_program['u_view'] = self.view
            self.ground_program['u_fish'] = [self.cam_x, self.cam_y, self.cam_z]
            self.cylinder_program['u_view'] = self.view
            self.cylinder_program['u_fish'] = [self.cam_x, self.cam_y, self.cam_z]
            #print(f'Yaw: {self.cam_yaw}, Pitch: {self.cam_pitch}, Roll: {self.cam_roll}, X: {self.cam_x}, Y: {self.cam_y}, Z: {self.cam_z}')
            
            for slave in self.slaves:
                slave.set_state(self.cam_x, self.cam_y, self.cam_z)

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_timer(self, event):

        self.t += self.t_step
        self.light_theta += self.light_theta_step

        light_position =  [5*np.cos(self.light_theta),np.sin(self.t)+GROUND_OFFSET+2,5*np.sin(self.light_theta)]
        light_projection = ortho(-1,1,-1,1,1,7)
        light_view = lookAt(light_position, [0,GROUND_OFFSET+0.6,0], [0,1,0])
        lightspace = light_view.dot(light_projection)

        self.shadowmap_ground['u_lightspace'] = lightspace
        self.shadowmap_program['u_lightspace'] = lightspace
        self.ground_program['u_lightspace'] = lightspace
        self.ground_program['u_light_position'] = light_position
        self.cylinder_program['u_lightspace'] = lightspace
        self.cylinder_program['u_light_position'] = light_position
        
    def on_draw(self, event):
        # draw to the fbo 
        with self.fbo: 
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, SHADOWMAP_RES, SHADOWMAP_RES)
            gloo.set_cull_face('front')
            self.shadowmap_ground.draw('triangles', self.ground_indices)
            self.shadowmap_program.draw('triangles', self.indices)
            
        # draw to screen
        gloo.clear(color=True, depth=True)
        gloo.set_viewport(0, 0, self.width, self.height)
        gloo.set_cull_face('back')
        self.ground_program.draw('triangles', self.ground_indices)
        self.cylinder_program.draw('triangles', self.indices)
        self.update()

    def on_close(self, event):
        for slave in self.slaves:
            slave.close()

if __name__ == '__main__':

    radius_mm = 33.7
    height_mm = 100
    fovy = 25.6
    proj_distance_mm = 210

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
    )

    master = Master(
        slaves = [proj0, proj1, proj2, proj3],
        radius_mm = radius_mm,
        height_mm = height_mm,
    )

    if sys.flags.interactive != 1:
        app.run()
    
