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
        data = np.array([[0,0]], dtype=np.float32)
        self.program['a_position'] = data
        self.program['u_size'] = 20.*ps
 
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.text = 'tick'        
        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('points')

    def on_timer(self, event):
        print(self.text)
        
    def on_key_press(self, event):
        if event.text == ' ':
            self.text = 'tock' 
            
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
