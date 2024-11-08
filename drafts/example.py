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


def rotation_matrix(rx, ry, rz):
    y_t = np.array(
        [
            [np.cos(rz), np.sin(rz), 0, 0],
            [-np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    p_t = np.array(
        [
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1],
        ]
    )
    r_t = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rx), np.sin(rx), 0],
            [0, -np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1],
        ]
    )

    # return y_t @ p_t @ r_t
    return r_t @ p_t @ y_t


def main():
    window = init()
    imgui_impl = init_imgui(window)

    n_samples = 1024

    a = np.array(
        [
            [1.0, 0.0, -9.7],
        ]
        * n_samples,
        dtype=np.float32,
    )
    a_g = np.zeros((n_samples, 3), dtype=np.float32)
    v = np.zeros((n_samples, 3), dtype=np.float32)
    v[-1] = [0.0, -1.0, 0.0]
    s = np.zeros((n_samples, 3), dtype=np.float32)

    w = np.array(
        [
            [0.0, 0.0, 1.0],
        ]
        * n_samples,
        dtype=np.float32,
    )
    r = np.zeros((n_samples, 3), dtype=np.float32)

    dt = np.array(
        [
            0.01,
        ]
        * n_samples,
        dtype=np.float32,
    )

    n = 0

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)

    while not window_should_close(window):
        r[n] = dt[n] * w[n] + r[n - 1]

        mat = rotation_matrix(*r[n])[0:3, 0:3]

        a_g[n] = a[n] @ mat + [0.0, 0.0, 9.8]

        v[n] = dt[n] * a_g[n] + v[n - 1]
        s[n] = dt[n] * v[n] + s[n - 1]

        n = n + 1
        if n >= n_samples:
            v[:] = 0
            s[:] = 0
            r[:] = 0
            n = 0
            v[-1] = [0.0, -1.0, 0.0]

        update()
        draw_points(s[:n])

        glfw.swap_buffers(window)
        glfw.poll_events()

    terminate()


if __name__ == "__main__":
    main()
