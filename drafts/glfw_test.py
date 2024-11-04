import glfw
from OpenGL.GL import *


def main():
    # Initialize the GLFW library
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "Hello World", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return

    glfw.make_context_current(window)

    # Main loop
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # Render here (OpenGL calls)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
