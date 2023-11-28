import sys
from vispy import gloo, app
from vispy.geometry import create_plane
import numpy as np

FREQ = 0.1

VERT_SHADER = """
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

uniform mat4 u_cylinder;

varying vec2 v_texcoord;

void main()
{
    vec4 pos = vec4(position, 1.0);
    gl_Position =  pos * u_cylinder * pos;
    v_texcoord = texcoord;
} 
"""

FRAG_SHADER = f"""
varying vec2 v_texcoord;

vec2 rotate(vec2 v, float a) {{
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, s, -s, c);
	return m * v;
}}

void main()
{{
    const float tau = 2.0*3.14159;
    float deg2rad = tau/360.0;
    float theta = deg2rad*30;
    const float freq = {FREQ};

    vec2 rot_coord = rotate(v_texcoord.xy, theta);
    float value = mod(floor(rot_coord.x / freq) + floor(rot_coord.y / freq) , 2);
    gl_FragColor = vec4(value, value, value, 1.0);
}} 
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(1024,1024), keys='interactive')

        self.phase = 0
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        V,I,_ = create_plane(2,2,100,100)
        vbo = gloo.VertexBuffer(V)
        self.indices = gloo.IndexBuffer(I)
        self.program.bind(vbo)
        self.program['u_cylinder'] = np.array([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
 
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw('triangles', self.indices)

    def on_timer(self, event):
        self.update()
    
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
    
