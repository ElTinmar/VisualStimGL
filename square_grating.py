import sys

from vispy import gloo
from vispy import app

FREQ = 0.01

VERT_SHADER = """
attribute vec2 position;
attribute float phase;
varying float v_phase;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_phase = phase;
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = f"""
varying float v_phase;
void main()
{{
    const float tau = 2.0*3.14159;
    const float freq = {FREQ};
    bool value = sin(freq*tau*gl_FragCoord.x + v_phase) > 0;
    gl_FragColor = vec4(value, value, value, 1.0);
}} 
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        ps = self.pixel_scale
        self.phase = 0

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Set uniforms and attributes
        self.program['phase'] = 0
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
        self.phase += 20 * 1/60 
        self.program['phase'] = self.phase
        self.update()
    
    
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
