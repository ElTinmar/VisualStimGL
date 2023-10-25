import sys
from vispy import app, gloo
import numpy as np
from vispy.util.transforms import rotate


VERT_SHADER = """
attribute vec3 a_position;
attribute vec2 a_resolution;
attribute float a_time;
varying vec2 v_resolution;
varying float v_time;

uniform float u_size;
uniform mat4 u_model;
uniform mat4 u_view;

void main()
{
    gl_Position = u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = u_size;
    v_resolution = a_resolution;
    v_time = a_time;
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = """
varying vec2 v_resolution;
varying float v_time;

void main()
{
    gl_FragColor = vec4(1.0,1.0,1.0, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, fullscreen=True, keys='interactive')

        ps = self.pixel_scale
        self.t = 0
        self.phi = 0
        self.rho = 0.0
        self.sigma = 0.0
        self.beta = 0.0
        self.theta = 0
        self.num_particles = 50000
        self.coords = 60*np.random.rand(3, self.num_particles).astype(np.float32)-30

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_resolution'] = self.physical_size
        self.program['a_time'] = 0
        self.program['a_position'] = self.coords.T
        self.program['u_size'] = 1.*ps
        self.program['u_view'] =  np.eye(4, dtype=np.float32) 
        self.program['u_model'] = np.eye(4, dtype=np.float32)
 
        self.timer = app.Timer(1/60, self.on_timer)
        self.timer.start()
        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.program['a_resolution'] = (width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('points')

    def on_timer(self, event):

        dt = 1/60
        self.t += dt # maybe give the actual time ?
        self.phi += 25*dt 
        self.program['a_time'] = self.t

        self.rho = 22+10*np.sin(0.5*self.t)
        self.sigma = 10+3*np.sin(0.1*self.t)
        self.beta = 2+1*np.sin(0.05*self.t)
        sigma = self.sigma
        beta = self.beta
        rho = self.rho
        
        x = self.coords[0,:]
        y = self.coords[1,:]
        z = self.coords[2,:]

        dx = sigma*(y - x)
        dy = x*(rho - z) - y
        dz = x*y - beta*z

        self.coords = self.coords + np.vstack((dx*dt,dy*dt,dz*dt))
        self.program['a_position'] = self.coords.T / np.array((60,60,60), dtype=np.float32)
        self.program['u_model'] = rotate(self.phi,(0,1,0))

        self.update()
    
def fps(canvas: Canvas, fps: float):
        canvas.title = f'FPS: {fps}'

if __name__ == '__main__':
    canvas = Canvas()
    #canvas.measure_fps(callback=partial(fps, canvas))
    canvas.measure_fps()
    if sys.flags.interactive != 1:
        app.run()
    
