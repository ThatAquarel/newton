import atexit
import os
import struct
import time

import glfw
import numpy as np
import serial

import imgui
from imgui.integrations.glfw import GlfwRenderer

from multiprocessing import Process, Event, shared_memory

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
    global last_x, last_y, angle_x, angle_y, pan_x, pan_y, dragging, panning, imgui_impl

    if imgui_impl != None and imgui.get_io().want_capture_mouse:
        return

    if dragging:
        dx = xpos - last_x
        dy = ypos - last_y
        if panning:
            pan_x += dx * 0.001
            pan_y -= dy * 0.001
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
# T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])


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

    draw_arrow([-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0, 5.0)
    draw_arrow([0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], 2.0, 5.0)
    draw_arrow([0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 2.0, 5.0)


T_2D = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def draw_axes_2D():
    glLineWidth(1.0)

    draw_arrow(
        [0.0, 0.0, 0.0] @ T_2D, [3.0, 0.0, 0.0] @ T_2D, [0.0, 0.0, 0.0], 2.0, 5.0
    )
    draw_arrow(
        [0.0, -0.5, 0.0] @ T_2D, [0.0, 0.5, 0.0] @ T_2D, [0.0, 0.0, 0.0], 2.0, 5.0
    )


aruco_point_color = np.array(
    [[0.92, 0.26, 0.21], [0.20, 0.66, 0.33], [0.26, 0.52, 0.96], [0.98, 0.74, 0.02]]
)

RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def draw_points(points):
    glPointSize(4.0)
    glBegin(GL_POINTS)

    for i, point in enumerate(points):
        glColor3f(*aruco_point_color[i % aruco_point_color.shape[0]])
        glVertex3f(*point @ T)

    glEnd()


def draw_line_strip(points, color):
    points = points @ T

    glBegin(GL_LINE_STRIP)
    glColor3f(*color)

    for point in points:
        glVertex3f(*point)

    glEnd()


def draw_arrow(start, end, color, line_width, point_size):
    glLineWidth(line_width)
    glBegin(GL_LINES)
    glColor3f(*color)
    glVertex3f(*start @ T)
    glVertex3f(*end @ T)
    glEnd()

    glPointSize(point_size)
    glBegin(GL_POINTS)
    glColor3f(*color)
    glVertex3f(*end @ T)
    glEnd()


def init():
    window_size = (800, 800)

    global last_x, last_y

    if not glfw.init():
        raise Exception("GLFW could not be initialized.")

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
        -10,
        10,
    )

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(pan_x, pan_y, 0.0)
    glRotatef(angle_x, 1.0, 0.0, 0.0)
    glRotatef(angle_y, 0.0, 1.0, 0.0)

    draw_axes()


def update_2D():
    zoom_2D = 2.0

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(viewport_left * zoom_2D, viewport_right * zoom_2D, -zoom_2D, zoom_2D, -1, 1)
    glTranslatef(viewport_left * zoom_2D + 0.1, -1.5, 0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def plot_line(x, y, color):
    glBegin(GL_LINE_STRIP)
    glColor3f(*color)
    for i, x0 in enumerate(x):
        glVertex3f(*[x0, y[i], 0] @ T_2D @ T)
    glEnd()


def draw_sensor(S_T):
    glBegin(GL_LINES)
    glColor3f(*RED)
    glVertex4f(*ORIGIN @ S_T @ T)
    glVertex4f(*[0.05, 0.00, 0.00] @ S_T @ T)
    glColor3f(*GREEN)
    glVertex4f(*ORIGIN @ S_T @ T)
    glVertex4f(*[0.00, 0.04, 0.00] @ S_T @ T)
    glColor3f(*BLUE)
    glVertex4f(*ORIGIN @ S_T @ T)
    glVertex4f(*[0.00, 0.00, 0.03] @ S_T @ T)
    glEnd()


def shm_create(_shared, name):
    shm = shared_memory.SharedMemory(name=name, create=True, size=_shared.nbytes)
    shared = np.ndarray(_shared.shape, dtype=_shared.dtype, buffer=shm.buf)
    shared[:] = _shared[:]

    return shm, shared


def shm_recv(_shared, name):
    existing_shm = shared_memory.SharedMemory(name=name)
    shared = np.ndarray(_shared.shape, dtype=_shared.dtype, buffer=existing_shm.buf)
    return existing_shm, shared


def shift_back(new, array):
    array[1:] = array[:-1]
    array[0] = new


DATA_T = np.eye(7) * np.array(
    [
        1e-6,
        # *[
        #     1 / 256 * 9.81,
        # ]
        # * 3,
        *[
            1,
        ]
        * 3,
        *[
            1 / 14.375 * np.pi / 180,
        ]
        * 3,
    ]
)

ACCEL_CAL = np.array(
    [
        [3.74565944e-02, 5.47801144e-04, -1.29205117e-03, -1.41535595e-04],
        [-2.52692280e-06, 3.82816643e-06, 1.56933766e-05, 1.47016881e-05],
        [-2.23799652e-04, 3.69483978e-02, -2.08327765e-04, 2.28314457e-04],
        [-3.95537063e-06, 3.85893009e-06, 1.59897209e-05, 1.42687695e-05],
        [3.99417069e-04, -4.47370170e-04, 3.94730233e-02, 7.51391635e-04],
        [-2.24582277e-06, 3.50554274e-06, 1.46161465e-05, 1.57830345e-05],
        [-5.05255184e-06, 8.02234081e-06, 2.97957140e-05, 6.37418452e-07],
    ]
)


def accel_cal(a):
    a = np.repeat(a, 2)
    a[1::2] **= 2
    a = [*a, 1]
    return (a @ ACCEL_CAL)[:-1]


def rotation_matrix(rx, ry, rz):
    z_t = np.array(
        [
            [np.cos(rz), np.sin(rz), 0],
            [-np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ]
    )
    y_t = np.array(
        [
            [np.cos(ry), 0, -np.sin(ry)],
            [0, 1, 0],
            [np.sin(ry), 0, np.cos(ry)],
        ]
    )
    x_t = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), np.sin(rx)],
            [0, -np.sin(rx), np.cos(rx)],
        ]
    )

    return z_t @ y_t @ x_t


def serial_process(
    port,
    stop_event,
    cal_event,
    context_shm,
    realtime_shm,
):
    ser = serial.Serial(port=port, baudrate=115200, dsrdtr=os.name != "nt")

    print(f"connected serial port: {port}")

    _context_shm, context = shm_recv(*context_shm)
    _realtime_shm, realtime = shm_recv(*realtime_shm)

    dt_i = 0
    sample_dt_buffer_size = 128
    sample_dt_buffer = np.zeros(sample_dt_buffer_size, dtype=np.float32)
    prev_time = time.time()
    sample_total = 0

    r = np.zeros(3, np.float32)
    v = np.zeros(3, np.float32)
    s = np.zeros(3, np.float32)
    a_cal = np.zeros(3, np.float32)
    w_cal = np.zeros(3, np.float32)

    a_i = 0
    a_cache = np.zeros((512, 3), np.float32)

    while not stop_event.is_set():
        ser.read_until(b"\x7E")
        buf = ser.read(16)
        if ser.read(1) != b"\x7D":
            break

        data = struct.unpack("<L6h", buf)

        data = data @ DATA_T

        if cal_event.is_set() or a_i != 0:
            a_cache[a_i] = data[1:4]
            a_i += 1

            if a_i == len(a_cache):
                a_i = 0
                print(np.mean(a_cache, axis=0))

        dt, a, w = data[0], data[1:4], data[4:7]
        a = accel_cal(a)

        if cal_event.is_set():
            print(a)
            r[:], v[:], s[:] = 0, 0, 0
            a_cal[:] = -a
            w_cal[:] = -w
            cal_event.clear()

        r = dt * (w + w_cal) + r
        m = rotation_matrix(*r)

        # a = a @ m + a_cal
        a = a @ m + a_cal
        # a[np.abs(a) < 0.025] = 0
        v = dt * a + v
        # v[np.abs(v) < 0.001] = 0
        s = dt * v + s

        realtime[:] = s

        sample_total += 1

        current_time = time.time()
        sample_dt_buffer[dt_i] = current_time - prev_time
        prev_time = current_time
        dt_i = (dt_i + 1) % sample_dt_buffer_size

        if dt_i == 0:
            context[1:] = sample_total, 1 / np.mean(sample_dt_buffer)
        context[0]

    ser.close()
    print("cleanup serial")

    stop_event.set()


WHITE = (1.0, 1.0, 1.0)
ORIGIN = np.array([0.0, 0.0, 0.0])


def main(
    port, buffer_size=1024, context_shm_name="context", realtime_shm_name="realtime"
):
    stop_event = Event()
    cal_event = Event()

    _context = np.zeros(3, dtype=np.float32)
    context_shm, context = shm_create(_context, context_shm_name)

    _realtime = np.zeros(3, dtype=np.float32)
    realtime_shm, realtime = shm_create(_realtime, realtime_shm_name)

    def close_shm():
        context_shm.close()
        context_shm.unlink()

        realtime_shm.close()
        realtime_shm.unlink()

        print("shm closed")

    atexit.register(close_shm)

    serial_manager = Process(
        target=serial_process,
        args=(
            port,
            stop_event,
            cal_event,
            (_context, context_shm.name),
            (_realtime, realtime_shm.name),
        ),
        daemon=True,
    )

    serial_manager.start()

    window = init()
    imgui_impl = init_imgui(window)

    while not stop_event.is_set() and not window_should_close(window):
        update()

        imgui.new_frame()
        imgui.begin("Sampling")

        imgui.text(f"device: {port}")
        n, ts, sr = context
        n = int(n)

        glPointSize(20.0)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 1)
        glVertex3f(*realtime @ T)
        glEnd()

        # glLineWidth(10.0)
        # glBegin(GL_LINES)

        # glColor3f(1.0, 0.0, 0.0)
        # glVertex3f(0.0, 0.0, 0.0)
        # glVertex3f(*[1, 0, 0] @ rotation_matrix(*realtime) @ T)

        # glColor3f(0.0, 1.0, 0.0)
        # glVertex3f(0.0, 0.0, 0.0)
        # glVertex3f(*[0, 1, 0] @ rotation_matrix(*realtime) @ T)

        # glColor3f(0.0, 0.0, 1.0)
        # glVertex3f(0.0, 0.0, 0.0)
        # glVertex3f(*[0, 0, 1] @ rotation_matrix(*realtime) @ T)
        # glEnd()

        imgui.text(f"total samples: {ts:.0f}")
        imgui.text(f"sample rate: {sr:.2f} Hz")

        imgui.separator()

        if imgui.button("calibrate"):
            cal_event.set()

        imgui.end()
        imgui.render()
        imgui_impl.process_inputs()
        imgui_impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)
        glfw.poll_events()

    stop_event.set()

    terminate()


if __name__ == "__main__":
    if os.name == "nt":
        main("COM11")
    else:
        main("/dev/ttyACM0")
