import sys
from vispy import app, gloo
import numpy as np
from functools import partial

# TODO maybe I should use regular coordinates and rotate view

VERT_SHADER = """
attribute vec2 a_position;
attribute float phase;
attribute float radius;
varying float v_phase;
varying float v_radius;

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_phase = phase;
    v_radius = radius;
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = """
varying float v_phase;
varying float v_radius;

vec2 rotate(vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, s, -s, c);
	return m * v;
}

float lines(float x, float y, float freq, float thickness) {
    
    float value = float(mod(x, freq)>0) * float(mod(x, freq)<thickness) 
        + float(mod(y, freq)>0) * float(mod(y, freq)<thickness);

    return(value);
}

vec2 map(vec2 cartesian_coord, float r) {
// map cartesian to spherical coords
    float x = cartesian_coord.x;
    float y = cartesian_coord.y;
    float z = sqrt(pow(r,2) - pow(x,2) - pow(y,2));
    float theta = sign(z) * acos( x / sqrt(pow(x,2)+pow(z,2)) );
    float phi = acos( y / sqrt(pow(x,2) + pow(y,2) + pow(z,2)) );
    return(vec2(theta, phi));
}

vec4 blue_halo(float x, float y, float freq, float thickness) {
    float middle = lines(x, y, freq, thickness);
    float blue = middle;
    
    float d = 1.0;
    int ksize = 15;
    float sigma = 6;
    for(int i=-ksize;i<=ksize;++i) {
        for(int j=-ksize;j<=ksize;++j) {
            blue = blue + 2/(2*3.14159*pow(sigma,2)) * exp(-0.5*(pow(i*d/sigma,2) + pow(j*d/sigma,2)))*lines(x+i*d, y+j*d, freq, thickness);
        }
    }
    
    return(vec4(middle,middle,blue,1.0));
}

void main()
{
    float freq = 0.33;
    vec2 center = vec2(512.0, 512.0);
    float radius = v_radius;
    vec2 coords = rotate(gl_FragCoord.xy-center, 0.5);
    vec2 spherical_coord = map(coords, radius);
    gl_FragColor = blue_halo(spherical_coord.x + v_phase, spherical_coord.y, freq, 0.02);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        self.t = 0
        self.phase = 0
        self.radius = 200

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['phase'] = 0
        self.program['radius'] = 200
        self.program['a_position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]
 

        self.timer = app.Timer('auto',self.on_timer)
        self.timer.start()

        self.show()


    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        self.t += 1/60
        self.phase += np.deg2rad(90) * 1/60 
        self.program['phase'] = self.phase
        self.radius = 300 + 300 * np.exp(-1.5*self.t)*np.sin(2*np.pi*self.t)
        self.program['radius'] = self.radius
        self.update()
    
def fps(canvas: Canvas, fps: float):
        canvas.title = f'FPS: {fps}'


if __name__ == '__main__':
    canvas = Canvas()
    canvas.measure_fps(callback=partial(fps, canvas))
    if sys.flags.interactive != 1:
        app.run()
    
