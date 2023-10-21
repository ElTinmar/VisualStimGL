import sys
from vispy import app, gloo
from vispy.gloo import IndexBuffer
from vispy.geometry import create_cylinder
from vispy.util.transforms import rotate
import numpy as np

FREQ = 100

VERT_SHADER = """
attribute vec3 a_position;
uniform   mat4 u_model;
void main()
{
    gl_Position = u_model * vec4(a_position, 1.0);
} 
"""

# Fragment Shaders have the following built-in input variables. 
# in vec4 gl_FragCoord;
# in bool gl_FrontFacing;
# in vec2 gl_PointCoord;

FRAG_SHADER = f"""
void main()
{{
    const float freq = {FREQ};
    float value = mod( floor(gl_FragCoord.x / freq) + floor(gl_FragCoord.y / freq) , 2);
    gl_FragColor = vec4(value, value, value, 1.0);
}} 
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        ps = self.pixel_scale
        self.phase = 0

        mesh_data = create_cylinder(100,100,radius=(0.5, 0.5),length=1.0)
        V = mesh_data.get_vertices()
        I = mesh_data.get_faces().ravel()
        self.indices = IndexBuffer(I)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = V
        self.program['u_model'] = rotate(90, (0,1,0), dtype = np.float32)

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangles', self.indices)

    def on_timer(self, event):
        self.update()
    
    
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
