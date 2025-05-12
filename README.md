# Fun with GL Shaders

This repository is a playground to create visual stimuli using opengl.

## KEY STEPS

- define geometry on which you want to project.
- project 3D meshes on the geometry (plane/cylinder/sphere) by intersecting
the geometry with vertex-observer lines.
- map depth properly using observer-screen and observer-vertex distance.
- view the scene (mesh projected on geometry) from the point of view of the video projector (or screen, it's basically the same) using a frustum with the correct size.

# Resources

 OpenGL 4.5 Reference Pages:  https://registry.khronos.org/OpenGL-Refpages/gl4/index.php
 
# TODO

- The behavior of sqrt for negative numbers is unspecified.
Some implementation may use abs. Make sure you define 
the intended behavior in your shader

- send (window size / time / mouse) as uniforms to the fragment shader


- normalize the coordinates in the fragment shader at the beginning?
