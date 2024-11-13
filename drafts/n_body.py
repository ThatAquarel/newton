import time

import glfw
import numpy as np

import imgui
from imgui.integrations.glfw import GlfwRenderer

from OpenGL.GL import *
from OpenGL.GLU import *

angle_x, angle_y = 45.0, 45.0
pan_x, pan_y = 0.0, 0.0
last_x, last_y = 0.0, 0.0
dragging = False
panning = False
zoom_level = 1.0
imgui_impl = None
viewport_left = 0
viewport_right = 0


def mouse_button_callback(window, button, action, mods):
    global dragging, panning, imgui_impl

    if imgui_impl != None and imgui.get_io().want_capture_mouse:
        return

    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            dragging = True
            panning = False
        elif action == glfw.RELEASE:
            dragging = False
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        if action == glfw.PRESS:
            dragging = True
            panning = True
        elif action == glfw.RELEASE:
            dragging = False
            panning = False


def cursor_pos_callback(window, xpos, ypos):
    global last_x, last_y, angle_x, angle_y, pan_x, pan_y, dragging, panning, imgui_impl, zoom_level

    if imgui_impl != None and imgui.get_io().want_capture_mouse:
        return

    if dragging:
        dx = xpos - last_x
        dy = ypos - last_y
        if panning:
            pan_x += dx * 0.001 * zoom_level
            pan_y -= dy * 0.001 * zoom_level
        else:
            angle_x += dy * 0.1
            angle_y += dx * 0.1
    last_x, last_y = xpos, ypos


def scroll_callback(window, xoffset, yoffset):
    global zoom_level, imgui_impl

    if imgui_impl != None and imgui.get_io().want_capture_mouse:
        return

    zoom_factor = 0.1
    if yoffset > 0:
        zoom_level /= 1 + zoom_factor  # Zoom in
    elif yoffset < 0:
        zoom_level *= 1 + zoom_factor  # Zoom out


def resize_callback(window, width, height):
    glViewport(0, 0, width, height)

    global viewport_left, viewport_right
    aspect_ratio = width / height if height > 0 else 1.0
    viewport_left = -aspect_ratio
    viewport_right = aspect_ratio


T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


_axes_y = np.mgrid[0:2, 0:1:11j, 0:1].T.reshape((-1, 3)) - [0.5, 0.5, 0.0]
_axes_x = _axes_y[:, [1, 0, 2]]


def draw_axes():
    glLineWidth(1.0)
    glBegin(GL_LINES)

    glColor3f(1.0, 1.0, 1.0)
    for point in _axes_x:
        glVertex3f(*(point) @ T)
    for point in _axes_y:
        glVertex3f(*point @ T)
    glEnd()

    glLineWidth(2.0)
    glBegin(GL_LINES)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(*[-0.5, 0.0, 0.0] @ T)
    glVertex3f(*[0.5, 0.0, 0.0] @ T)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(*[0.0, -0.5, 0.0] @ T)
    glVertex3f(*[0.0, 0.5, 0.0] @ T)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(*[0.0, 0.0, -0.5] @ T)
    glVertex3f(*[0.0, 0.0, 0.5] @ T)

    glEnd()


aruco_point_color = np.array(
    [[0.92, 0.26, 0.21], [0.20, 0.66, 0.33], [0.26, 0.52, 0.96], [0.98, 0.74, 0.02]]
)


def draw_points(points):
    glPointSize(4.0)
    glBegin(GL_LINE_STRIP)
    glColor3f(1.0, 1.0, 1.0)

    for i, point in enumerate(points):
        # glColor3f(*aruco_point_color[i % aruco_point_color.shape[0]])
        glVertex3f(*point @ T)

    glEnd()


def init():
    window_size = (800, 800)

    global last_x, last_y

    if not glfw.init():
        raise Exception("GLFW could not be initialized.")

    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(*window_size, "position", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window could not be created.")

    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_framebuffer_size_callback(window, resize_callback)

    last_x, last_y = glfw.get_cursor_pos(window)

    resize_callback(window, *window_size)

    return window


def init_imgui(window):
    global imgui_impl
    imgui.create_context()
    imgui_impl = GlfwRenderer(window, attach_callbacks=False)
    return imgui_impl


def window_should_close(window):
    return glfw.window_should_close(window)


def terminate():
    glfw.terminate()


def update():
    global viewport_left, viewport_right

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.86, 0.87, 0.87, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(
        viewport_left * zoom_level,
        viewport_right * zoom_level,
        -zoom_level,
        zoom_level,
        -32,
        32,
    )

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(pan_x, pan_y, 0.0)
    glRotatef(angle_x, 1.0, 0.0, 0.0)
    glRotatef(angle_y, 0.0, 1.0, 0.0)

    draw_axes()


# def main(n_body=2, G=6.6743e-11):
def main(n_body=2, G=6.6743e-2):
    window = init()
    imgui_impl = init_imgui(window)

    # m = np.array([300, 100], dtype=np.float32)
    m = np.array([3,4, 2], dtype=np.float32)
    a = np.zeros((n_body, 3), dtype=np.float32)
    v = np.array(
        [
            [0, 0.25, 0.1],
            [0, -0.25, 0.1],
        ],
        dtype=np.float32,
    )
    s = np.array([[1, 0, 0], [0, 1, 0],[0,0,0]], dtype=np.float32)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)

    start = time.time()
    dt = 0

    while not window_should_close(window):
        update()

        for body in range(n_body):
            others = [n for n in range(n_body) if n != body]

            m_a = m[body]
            F_a = np.zeros(3, dtype=np.float32)
            for j in others:

                m_b = m[j]
                s_a, s_b = s[body], s[j]

                ds = s_b - s_a
                d = np.linalg.norm(ds)
                Fg = G * m_a * m_b / d**2

                F_a += ds / d * Fg

            a[body] = F_a / m_a
            v[body] = a[body] * dt + v[body]
            s[body] = v[body] * dt + s[body]

            glPointSize(m[body] * 10)
            glBegin(GL_POINTS)
            glColor3f(1, 1, 1)
            glVertex3f(*s[body] @ T)
            glEnd()

        glfw.swap_buffers(window)
        glfw.poll_events()

        current = time.time()
        dt = current - start
        start = current

    terminate()


if __name__ == "__main__":
    main()
