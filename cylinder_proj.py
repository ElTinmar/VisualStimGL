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

float checkerboard(float theta, float h, float freq) {
// create checkerboard texture on the surface of the cylinder
    float value = mod( floor(theta / freq) + floor(h / freq) , 2);
    return(value);
}

vec2 map(vec2 cartesian_coord, float r, vec2 center) {
// map cartesian to cylindrical coords
    float x = cartesian_coord.x - center.x;
    float y = cartesian_coord.y - center.y;
    float theta = acos(x/r);
    float h = cartesian_coord.y/r;
    return(vec2(theta, h));
}

void main()
{
    float freq = 2*3.14159*0.01;
    vec2 center = vec2(640.0, 400.0);
    float radius = 400.0;
    vec2 cylindrical_coord = map(gl_FragCoord.xy, radius, center);
    float value = checkerboard(cylindrical_coord.s + v_phase, cylindrical_coord.t, freq);
    gl_FragColor = vec4(value, value, value, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(912,1140), decorate=False, position=(2560,0), keys='interactive')

        self.phase = 0

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['phase'] = 0
        self.program['a_position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]
 
        self.timer = app.Timer(1/120, self.on_timer)
        self.timer.start()

        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        self.phase +=  np.deg2rad(30) * 1/120 
        self.program['phase'] = self.phase
        self.update()
    
def fps(canvas: Canvas, fps: float):
    canvas.title = f'FPS: {fps}'

if __name__ == '__main__':
    canvas = Canvas()
    #canvas.measure_fps(callback=partial(fps, canvas))
    canvas.measure_fps()
    if sys.flags.interactive != 1:
        app.run()
    
