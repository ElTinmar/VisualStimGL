import sys

from vispy import gloo
from vispy import app
import numpy as np

VERT_SHADER = """
attribute vec2 a_position;
uniform float u_size;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = u_size;
}
"""

FRAG_SHADER = """
void main() {
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')

        ps = self.pixel_scale
        

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.pos = np.array([[0,0]], dtype=np.float32)
        self.program['a_position'] = self.pos
        self.program['u_size'] = 20.*ps
 
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('points')

    def on_timer(self, event):
        pass
        
    def on_key_press(self, event):
        h_inc = np.array([[0.01, 0.0]], dtype=np.float32)
        v_inc = np.array([[0.0, 0.01]], dtype=np.float32)
        
        if event.text == 'a':
            self.pos -= h_inc 
        if event.text == 'd':
            self.pos += h_inc 
        if event.text == 'w':
            self.pos += v_inc 
        if event.text == 's':
            self.pos -= v_inc 
            
        self.program['a_position'] = self.pos
        
        self.update()
            
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
