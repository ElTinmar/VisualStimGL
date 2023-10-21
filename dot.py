import sys

from vispy import gloo
from vispy import app
import numpy as np
from multiprocessing import Process, Queue
import time

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
    def __init__(self, queue):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        self.queue = queue
        ps = self.pixel_scale
        

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.pos = np.array([[0,0]], dtype=np.float32)
        self.program['a_position'] = self.pos
        self.program['u_size'] = 20.*ps

        gloo.set_viewport(0, 0, *self.physical_size)
 
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
        self.pos = self.queue.get()
        self.program['a_position'] = self.pos
        self.update()
        
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

def monitor(queue):
    while True:
        print(queue.qsize())    
        time.sleep(0.1)
            
def producer(queue):
    pos = np.array([[0.0,0.0]], dtype=np.float32)
    alpha_inc = 0.1
    alpha = 0
    while True:
        alpha = alpha + alpha_inc
        pos[0,0] = 0.5*np.cos(alpha)
        pos[0,1] = 0.5*np.sin(alpha)
        queue.put(pos)
        time.sleep(0.017)
    
    
if __name__ == '__main__':
    q = Queue()
    canvas = Canvas(q)
    prod = Process(target=producer, args=(q,))
    mon = Process(target=monitor, args=(q,))
    prod.start()
    mon.start()
    if sys.flags.interactive != 1:
        app.run()
    prod.terminate()
    mon.terminate()
