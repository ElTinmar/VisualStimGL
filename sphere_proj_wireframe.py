import sys
from vispy import app, gloo
import numpy as np
from functools import partial

# TODO maybe I should use regular coordinates and rotate view

VERT_SHADER = """
attribute vec2 a_position;
attribute float phase;
varying float v_phase;
void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_phase = phase;
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = """
varying float v_phase;

float lines(float x, float y, float freq, float thickness) {
    
    float value = float(mod(x, freq)>0) * float(mod(x, freq)<thickness) 
        + float(mod(y, freq)>0) * float(mod(y, freq)<thickness);

    return(value);
}

vec2 map(vec2 cartesian_coord, float r, vec2 center) {
// map cartesian to spherical coords
    float x = cartesian_coord.x - center.x;
    float y = cartesian_coord.y - center.y;
    float z = sqrt(pow(r,2) - pow(x,2) - pow(y,2));
    float theta = sign(z) * acos( x / sqrt(pow(x,2)+pow(z,2)) );
    float phi = acos( y / sqrt(pow(x,2) + pow(y,2) + pow(z,2)) );
    return(vec2(theta, phi));
}

void main()
{
    float freq = 0.33;
    vec2 center = vec2(512.0, 512.0);
    float radius = 512.0;
    vec2 spherical_coord = map(gl_FragCoord.xy, radius, center);
    float value = lines(spherical_coord.x + v_phase, spherical_coord.y, freq, 0.02);
    gl_FragColor = vec4(value, value, value, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        self.phase = 0

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['phase'] = 0
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
        self.phase += np.deg2rad(90) * 1/60 
        self.program['phase'] = self.phase
        self.update()
    
def fps(canvas: Canvas, fps: float):
        canvas.title = f'FPS: {fps}'


if __name__ == '__main__':
    canvas = Canvas()
    canvas.measure_fps(callback=partial(fps, canvas))
    if sys.flags.interactive != 1:
        app.run()
    
