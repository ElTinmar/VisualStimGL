import sys

from vispy import gloo
from vispy import app
import math

FREQ = 0.01

VERT_SHADER = """
attribute vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = """
vec2 rotate(vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, s, -s, c);
	return m * v;
}

void main()
{
    vec2 center = vec2(512.0, 512.0);
    float lambda = 0.01;
    float theta = 0.1;
    float psi = 0.0;
    float sigma = 50.0;
    float gamma = 1.0;
    float tau = 2*3.14159;
    vec2 rot_coord = rotate(gl_FragCoord.xy-center, theta);
    float value = exp(-( pow(rot_coord.x,2)+pow(gamma,2)*pow(rot_coord.y,2) ) / ( 2*pow(sigma,2) ))*cos(tau*rot_coord.x/lambda + psi);
    gl_FragColor = vec4(value, value, value, 1.0);
    
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        ps = self.pixel_scale
        self.phase = 0

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Set uniforms and attributes
        self.program['position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]
 
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('triangle_strip')
 
    def on_timer(self, event):
        self.phase += 0.1
        #self.program['norm'] = [math.cos(self.alpha), math.sin(self.alpha)]
        self.update()
    
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
