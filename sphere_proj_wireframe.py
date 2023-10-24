import sys
from vispy import app, gloo

# TODO maybe I should use regular coordinates and rotate view

VERT_SHADER = """
attribute vec2 a_position;
attribute vec2 a_resolution;
attribute float a_time;
varying vec2 v_resolution;
varying float v_time;

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_resolution = a_resolution;
    v_time = a_time;
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = """
#version 150

varying vec2 v_resolution;
varying float v_time;

vec2 rotate(vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, s, -s, c);
	return m * v;
}

vec2 map_plane_to_sphere_surface(vec2 cartesian_coord, float r) {
    float x = cartesian_coord.x;
    float y = cartesian_coord.y;
    float z = sqrt(pow(r,2) - pow(x,2) - pow(y,2));
    float theta = sign(z) * acos( x / sqrt(pow(x,2)+pow(z,2)) );
    float phi = acos( y / sqrt(pow(x,2) + pow(y,2) + pow(z,2)) );
    return(vec2(theta, phi));
}

float lines(vec2 pos, float freq, float thickness) {
    // TODO maybe use smoothedge
    float value = float(mod(pos.x, freq)>0) * float(mod(pos.x, freq)<thickness) 
        + float(mod(pos.y, freq)>0) * float(mod(pos.y, freq)<thickness);

    return(value);
}

float gaussian_2D(vec2 x, vec2 mu, mat2 sigma) {
    float tau = 2*3.14159;
    float norm = sqrt( pow(tau,2) * determinant(sigma) );
    float gaussian = exp(-1.0/2.0 * dot((x-mu) * inverse(sigma), (x-mu)) );
    return(gaussian/norm);
}

float halo_lines(vec2 pos, float freq, float thickness) {
    int ksize = 15;
    vec2 mu = vec2(0.0, 0.0);
    mat2 sigma = mat2(6.0, 0.0, 0.0, 6.0);
    float pixel_value = lines(pos, freq, thickness);
    float strength = 1.0;

    for(int i=-ksize;i<=ksize;++i) {
        for(int j=-ksize;j<=ksize;++j) {
            vec2 e = vec2(pos.x+i, pos.y+j);
            pixel_value += strength * gaussian_2D(e, pos, sigma) * lines(e, freq, thickness);
        }
    }
    
    return(pixel_value);
}

vec3 hsv2rgb(vec3 hsv_color) {
    // from wikipedia https://en.wikipedia.org/wiki/HSL_and_HSV
    float h = hsv_color.x; float s = hsv_color.y; float v = hsv_color.z;

    float r = v - v*s*max(0, min(min(mod(5 + h/60,6), 4-mod(5 + h/60,6)), 1.0));
    float g = v - v*s*max(0, min(min(mod(3 + h/60,6), 4-mod(3 + h/60,6)), 1.0));
    float b = v - v*s*max(0, min(min(mod(1 + h/60,6), 4-mod(1 + h/60,6)), 1.0));
    return(vec3(r,g,b));
}

void main()
{
    float tau = 2*3.14159;
    float deg2rad = tau/360.0;
    float bar_freq = 0.33;
    float bar_thickness = 0.02;
    float expansion_freq = 2.0;
    float bounce_period = 2.5;
    float damping = 1.5;
    float hue_rot_speed = 0.25 * 360.0;
    float initial_radius = 2.0/3.0 * min(v_resolution.x, v_resolution.y)/2.0;
    float expansion = 1.0/3.0 * min(v_resolution.x, v_resolution.y)/2.0;
    
    float phase = deg2rad * 90.0 * v_time;
    float radius = initial_radius + expansion * exp( -damping * mod(v_time, bounce_period) ) * sin( expansion_freq*tau*v_time );
    vec2 center = v_resolution/2;
    vec2 cartesian_coord = rotate(gl_FragCoord.xy-center, 0.5);
    vec2 spherical_coord = map_plane_to_sphere_surface(cartesian_coord, radius);
    float value = halo_lines(spherical_coord + vec2(phase, 0.0), bar_freq, bar_thickness);
     
    vec3 col_hsv =  vec3(mod(hue_rot_speed*v_time,360),1.0,1.0);
    vec3 col_rgb = hsv2rgb(col_hsv);
    gl_FragColor = vec4(value*col_rgb,1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        sz = (912,1140)

        app.Canvas.__init__(self, size=sz, decorate=True, position=(2560,0), keys='interactive')

        self.t = 0

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_resolution'] = sz
        self.program['a_time'] = 0
        self.program['a_position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]
 
        self.timer = app.Timer(1/120, self.on_timer)
        self.timer.start()
        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.program['a_resolution'] = (width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        self.t += 1/120 # maybe give the actual time ?
        self.program['a_time'] = self.t
        self.update()
    
def fps(canvas: Canvas, fps: float):
        canvas.title = f'FPS: {fps}'

if __name__ == '__main__':
    canvas = Canvas()
    #canvas.measure_fps(callback=partial(fps, canvas))
    canvas.measure_fps()
    if sys.flags.interactive != 1:
        app.run()
    
