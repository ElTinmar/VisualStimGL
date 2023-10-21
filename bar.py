import sys

from vispy import gloo
from vispy import app
import math

FREQ = 0.01

VERT_SHADER = """
attribute vec2 position;
attribute vec2 norm;
attribute vec2 origin; 
varying vec2 v_norm;
varying vec2 v_origin;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_norm = norm;
    v_origin = origin;
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = """
varying vec2 v_norm;
varying vec2 v_origin;
void main()
{
    float width = 20;
    bvec2 pix_in_bar = bvec2(dot(gl_FragCoord.xy-v_origin, v_norm)>0, dot(gl_FragCoord.xy-v_origin, v_norm)<width);
    if (all(pix_in_bar)) {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } 
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        ps = self.pixel_scale
        self.alpha = 0

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Set uniforms and attributes
        self.program['norm'] = [1.0, 0.0]
        self.program['origin'] = [512, 512]
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
        self.alpha += 0.1
        self.program['norm'] = [math.cos(self.alpha), math.sin(self.alpha)]
        self.update()
    
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
