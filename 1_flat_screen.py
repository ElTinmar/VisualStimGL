import sys
from vispy import gloo, app,  use 
from vispy.geometry import create_box
from vispy.util.transforms import perspective, translate, rotate, frustum, ortho
from vispy.io import imread, read_mesh
import numpy as np
from PyQt5.QtCore import QPoint, Qt 
from typing import Tuple

def lookAt(eye, target, up=[0, 1, 0]):
    """Computes matrix to put eye looking at target point."""

    eye = np.asarray(eye).astype(np.float32)
    target = np.asarray(target).astype(np.float32)
    up = np.asarray(up).astype(np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    side = np.cross(forward, up)
    side /= np.linalg.norm(side)

    up = np.cross(side, forward)

    M = np.eye(4)
    M[0,:3] = side
    M[1,:3] = up
    M[2,:3] = -forward
    M[0,3] = -np.dot(side, eye)
    M[1,3] = -np.dot(up, eye)
    M[2,3] = np.dot(forward, eye)
    
    return M

use(gl='gl+')

VERT_SHADER = """
#version 140
  
// uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_lightspace;
uniform vec3 u_fish;
uniform vec3 u_screen_bottomleft;
uniform vec3 u_screen_normal;

// per-vertex attributes
attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;
attribute vec3 a_instance_shift;

// varying
varying float v_depth;
varying vec2 v_texcoord;
varying vec3 v_normal_world;
varying vec4 v_world_position;
varying vec4 v_lightspace_position;

vec3 plane_proj(vec3 fish_pos, vec3 vertex_pos, vec3 screen_bottomleft, vec3 screen_normal) { 

    // vertex coords
    float x_v = vertex_pos.x;
    float y_v = vertex_pos.y;
    float z_v = vertex_pos.z;

    // fish coords
    float x_f = fish_pos.x;
    float y_f = fish_pos.y;
    float z_f = fish_pos.z;

    // plane
    float a = screen_normal.x;
    float b = screen_normal.y;
    float c = screen_normal.z;
    float d = -dot(screen_normal, screen_bottomleft);

    float denominator = dot(screen_normal, fish_pos-vertex_pos);
    // TODO handle degenerate case?

    float x = (b*(x_v*y_f - x_f*y_v) + c*(x_v*z_f - x_f*z_v) + d*(x_v-x_f)) / denominator;
    float y = (a*(y_v*x_f - y_f*x_v) + c*(y_v*z_f - y_f*z_v) + d*(y_v-y_f)) / denominator;
    float z = (a*(z_v*x_f - z_f*x_v) + b*(z_v*y_f - z_f*y_v) + d*(z_v-z_f)) / denominator;

    vec3 sol = vec3(x, y, z);
    return sol;
}

void main()
{
    vec4 vertex_world = u_model * vec4(a_position, 1.0);
    vertex_world.xyz = vertex_world.xyz + a_instance_shift;  
    vec3 screen_world = plane_proj(u_fish, vertex_world.xyz, u_screen_bottomleft, u_screen_normal);
    vec3 normal_world = transpose(inverse(mat3(u_model))) * a_normal;
    vec4 screen_clip = u_projection * u_view * vec4(screen_world, 1.0);

    float magnitude = length(vertex_world.xyz - u_fish);
    vec3 direction = normalize(screen_world - u_fish);
    float orientation = sign(dot(vertex_world.xyz - u_fish, screen_world - u_fish));

    vec3 offset_world = u_fish;
    offset_world += orientation*direction * magnitude;
    vec4 offset_clip = u_projection * u_view * vec4(offset_world, 1.0);

    v_depth = offset_clip.z/offset_clip.w;
    v_texcoord = a_texcoord;
    v_normal_world = normal_world;
    v_world_position = vertex_world;
    v_lightspace_position = u_lightspace * vertex_world;
    gl_Position = screen_clip;
}
"""

FRAG_SHADER = """
#version 140

uniform sampler2D u_texture;
uniform sampler2D u_shadow_map_texture;
uniform vec2 u_resolution;
uniform vec3 u_light_position;
uniform vec3 u_fish;

varying vec3 v_normal_world;
varying vec2 v_texcoord;
varying float v_depth;
varying vec4 v_world_position;
varying vec4 v_lightspace_position;

float get_shadow(vec4 lightspace_position,  vec3 norm, vec3 light_direction)
{
    float bias = mix(0.05, 0.0, dot(norm, light_direction));    

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
    
    // output
    vec3 position_ndc = v_lightspace_position.xyz / v_lightspace_position.w;
    position_ndc = position_ndc * 0.5 + 0.5;
    float closest_depth = texture2D(u_shadow_map_texture, position_ndc.xy).r; 
    gl_FragColor = vec4(vec3(closest_depth), 1.0);

    gl_FragColor = gamma_corrected;
    gl_FragDepth = v_depth;
}
"""

VERTEX_SHADER_SHADOW="""
#version 140

uniform mat4 u_model;
uniform mat4 u_lightspace;

attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;
attribute vec3 a_instance_shift;

void main()
{
    vec4 world_pos = u_model * vec4(a_position, 1.0);
    world_pos.xyz = world_pos.xyz + a_instance_shift;
    gl_Position = u_lightspace * world_pos;
}
"""

FRAGMENT_SHADER_SHADOW="""
#version 140

void main()
{
    float depth = gl_FragCoord.z;
    gl_FragColor = vec4(depth,0.0,0.0,1.0);
}
"""

SHELL_MODEL = rotate(90,(1,0,0)).dot(rotate(180,(0,0,1))).dot(translate((0,0.6,0)))
GROUND_MODEL = translate((0,0,0))
SHADOWMAP_RES = 2048

class Master(app.Canvas):
    def __init__(
            self,
            screen_width_cm: float,
            screen_height_cm: float,
            screen_bottomleft: Tuple,
            screen_normal: Tuple
        ):

        app.Canvas.__init__(
            self, 
            size = (1280,800), 
            position = (0,0), 
            fullscreen = False, 
            keys = 'interactive'
        )

        self.screen_width_cm = screen_width_cm 
        self.screen_height_cm = screen_height_cm
        self.screen_bottomleft = screen_bottomleft
        self.screen_normal = screen_normal
        self.screen_bottomleft_x, self.screen_bottomleft_y, self.screen_bottomleft_z = screen_bottomleft

        # rotation and translation gain
        self.step_t = 0.5
        self.step_t_fast = 2
        self.step_r = 0.1

        # camera location and rotation
        self.cam_x = 0
        self.cam_y = 1
        self.cam_z = 20

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

        self.show()

    def set_context(self):
        self.width, self.height = self.physical_size
        gloo.set_viewport(0, 0, self.width, self.height)
        gloo.set_state(depth_test=True)  # required for object in the Z axis to hide each other

    def create_view(self):
        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z))

    def create_projection(self):
        left = self.screen_bottomleft_x-self.cam_x
        bottom = self.screen_bottomleft_y-self.cam_y
        depth = self.screen_bottomleft_z-self.cam_z
        right = left + self.screen_width_cm
        top = bottom + self.screen_height_cm
        znear = 0.1
        zfar = 1000
        scale = znear/abs(depth)
        
        self.projection = frustum(scale*left, scale*right, scale*bottom, scale*top, znear, zfar)

    def create_scene(self):

        light_position = [5,5,5]
        light_projection = ortho(-50,50,-50,50,1,30)
        light_view = lookAt(light_position, [0,0,0], [0,1,0])
        lightspace = light_projection.T @ light_view    
        lightspace = lightspace.T

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

        vertices, faces, _ = create_box(width=30, height=30, depth=1, height_segments=100, width_segments=100, depth_segments=10)
        vtype = [
            ('a_position', np.float32, 3),
            ('a_texcoord', np.float32, 2),
            ('a_normal', np.float32, 3)
        ]
        vertex = np.zeros(vertices.shape[0], dtype=vtype)
        vertex['a_position'] = vertices['position']
        vertex['a_texcoord'] = vertices['texcoord']*2
        vertex['a_normal'] = vertices['normal']
        vbo_ground = gloo.VertexBuffer(vertex)
        self.ground_indices = gloo.IndexBuffer(faces)

        self.shadowmap_ground = gloo.Program(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
        self.shadowmap_ground.bind(vbo_ground)
        self.shadowmap_ground['u_model'] = GROUND_MODEL
        self.shadowmap_ground['u_lightspace'] = lightspace
        self.shadowmap_ground['a_instance_shift'] = [0,0,0]
        
        self.ground_program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.ground_program.bind(vbo_ground)
        self.ground_program['u_fish'] = [self.cam_x,self.cam_y,self.cam_z]
        self.ground_program['u_texture'] = gloo.Texture2D(texture, wrapping='repeat')
        self.ground_program['a_instance_shift'] = [0,0,0]
        self.ground_program['u_resolution'] = [self.width, self.height]
        self.ground_program['u_view'] = self.view
        self.ground_program['u_model'] = GROUND_MODEL
        self.ground_program['u_projection'] = self.projection
        self.ground_program['u_lightspace'] = lightspace
        self.ground_program['u_light_position'] = light_position
        self.ground_program['u_shadow_map_texture'] = self.shadow_map_texture
        self.ground_program['u_screen_normal'] = self.screen_normal
        self.ground_program['u_screen_bottomleft'] = self.screen_bottomleft

        ## shell -----------------------------------------------------------------------------

        # load texture
        texture = np.flipud(imread('quartz.jpg'))

        # load mesh
        vertices, faces, normals, texcoords = read_mesh('shell_simplified.obj')
        vertices = 10*vertices
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
        instance_shift = gloo.VertexBuffer([(10,0,-2),(0,0,-10),(0,0,0),(-5,5,-1)], divisor=1)

        self.shadowmap_program = gloo.Program(VERTEX_SHADER_SHADOW, FRAGMENT_SHADER_SHADOW)
        self.shadowmap_program.bind(vbo_shell)
        self.shadowmap_program['u_model'] = SHELL_MODEL
        self.shadowmap_program['u_lightspace'] = lightspace
        self.shadowmap_program['a_instance_shift'] = instance_shift

        self.main_program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.main_program.bind(vbo_shell)
        self.main_program['u_texture'] = texture
        self.main_program['u_fish'] = [self.cam_x,self.cam_y,self.cam_z]
        self.main_program['a_instance_shift'] = instance_shift
        self.main_program['u_resolution'] = [self.width, self.height]
        self.main_program['u_view'] = self.view
        self.main_program['u_model'] = SHELL_MODEL
        self.main_program['u_projection'] = self.projection
        self.main_program['u_lightspace'] = lightspace
        self.main_program['u_light_position'] = light_position
        self.main_program['u_shadow_map_texture'] = self.shadow_map_texture
        self.main_program['u_screen_normal'] = self.screen_normal
        self.main_program['u_screen_bottomleft'] = self.screen_bottomleft

    def on_key_press(self, event):

        tx = 0
        ty = 0
        tz = 0
    
        if len(event.modifiers) > 0 and event.modifiers[0] == 'Shift':
            step = self.step_t_fast
        else:
            step = self.step_t

        if event.key == 'W':
            tz = -step
        elif event.key == 'A':
            tx = -step
        elif event.key == 'S':
            tz = step
        elif event.key == 'D':
            tx = step
        elif event.key == 'Up':
            ty = step
        elif event.key == 'Down':
            ty = -step

        self.cam_x += tx
        self.cam_y += ty
        self.cam_z += tz

        self.view = translate((-self.cam_x, -self.cam_y, -self.cam_z))
        self.create_projection()
        
        self.ground_program['u_view'] = self.view
        self.ground_program['u_fish'] = [self.cam_x, self.cam_y, self.cam_z]
        self.ground_program['u_projection'] = self.projection

        self.main_program['u_view'] = self.view
        self.main_program['u_fish'] = [self.cam_x, self.cam_y, self.cam_z]
        self.main_program['u_projection'] = self.projection

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_timer(self, event):

        self.t += self.t_step
        self.light_theta += self.light_theta_step

        light_position =  [5*np.cos(self.light_theta), 20, 5*np.sin(self.light_theta)]
        light_projection = ortho(-100,100,-100,100,10,25)
        light_view = lookAt(light_position, [0,0,0], [0,1,0])
        lightspace = light_projection.T @ light_view
        lightspace = lightspace.T

        self.shadowmap_ground['u_lightspace'] = lightspace
        self.shadowmap_program['u_lightspace'] = lightspace
        self.ground_program['u_lightspace'] = lightspace
        self.ground_program['u_light_position'] = light_position
        self.main_program['u_lightspace'] = lightspace
        self.main_program['u_light_position'] = light_position
        
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
        self.main_program.draw('triangles', self.indices)
        self.update()

if __name__ == '__main__':

    screen_width_cm = 27
    screen_height_cm = 17 
    screen_bottomleft = (-screen_width_cm/2, -screen_height_cm/2, 0)
    screen_normal = (0,0,1)

    master = Master(
        screen_width_cm,
        screen_height_cm,
        screen_bottomleft,
        screen_normal
    )

    if sys.flags.interactive != 1:
        app.run()
    
