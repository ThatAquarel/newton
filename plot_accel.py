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


aruco_point_color = np.array(
    [[0.92, 0.26, 0.21], [0.20, 0.66, 0.33], [0.26, 0.52, 0.96], [0.98, 0.74, 0.02]]
)


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


DATA_T = np.eye(7) * np.array([
    1e-6,
    *[1 / -256 * 9.81,] * 3,
    *[1 / 14.375 * np.pi / 180,] * 3
])

def serial_process(
    port,
    stop_event,
    save_event,
    sr_shm,
    c_shm,
    acc_shm,
    vel_shm,
    dis_shm,
):
    ser = serial.Serial(port=port, baudrate=115200, dsrdtr=os.name != "nt")

    print(f"connected serial port: {port}")

    _sr_shm, sample_rate = shm_recv(*sr_shm)
    _c_shm, cal_offset = shm_recv(*c_shm)
    _acc_shm, acc = shm_recv(*acc_shm)
    _vel_shm, vel = shm_recv(*vel_shm)
    _dis_shm, dis = shm_recv(*dis_shm)

    dt_i = 0
    sample_dt_buffer_size = 128
    sample_dt_buffer = np.zeros(sample_dt_buffer_size, dtype=np.float32)
    prev_time = time.time()

    sample_total = 0

    start_saving = False

    while not stop_event.is_set():
        ser.read_until(b"\x7E")
        buf = ser.read(16)
        if ser.read(1) != b"\x7D":
            break

        data = struct.unpack("<L6h", buf)

        dt = acc[0, 0]
        dvdt = (acc[0, 1:7] + acc[1, 1:7]) * dt / 2
        dsdt = (vel[0] + vel[1]) * dt / 2

        shift_back(data @ DATA_T, acc)
        acc[0, 4:7] -= cal_offset
        shift_back(dvdt + vel[0], vel)
        shift_back(dsdt + dis[0], dis)

        sample_total += 1

        current_time = time.time()
        sample_dt_buffer[dt_i] = current_time - prev_time
        prev_time = current_time
        dt_i = (dt_i + 1) % sample_dt_buffer_size

        if dt_i == 0:
            sample_rate[:] = sample_total, 1 / np.mean(sample_dt_buffer)

        buffer_pos = sample_total % acc.shape[0]
        if save_event.is_set() and buffer_pos == 0:
            start_saving = True
            save_event.clear()
            print("start saving")
        if start_saving and buffer_pos == (acc.shape[0] - 1):
            start_saving = False
            print("stop saving")
            np.save(filename := f"{round(time.time())}_acc_export.npy", acc)
            print(f"saved as: {filename}")

    ser.close()
    print("cleanup serial")

    stop_event.set()


WHITE = (1.0, 1.0, 1.0)
ORIGIN = np.array([0.0, 0.0, 0.0])


def main(
    port,
    buffer_size=1024,
    sr_shm_name="sample_rate",
    c_shm_name="cal",
    acc_shm_name="acc",
    vel_shm_name="vel",
    dis_shm_name="dis",
):
    stop_event = Event()
    save_event = Event()

    _sample_rate = np.array([0, 0], dtype=np.float32)
    sr_shm, sample_rate = shm_create(_sample_rate, sr_shm_name)

    _cal_offset = np.zeros(3, dtype=np.float32)
    cal_shm, cal_offset = shm_create(_cal_offset, c_shm_name)

    _acc = np.zeros((buffer_size, 7), dtype=np.float32)
    acc_shm, acc = shm_create(_acc, acc_shm_name)

    _vel = np.zeros((buffer_size, 6), dtype=np.float32)
    vel_shm, vel = shm_create(_vel, vel_shm_name)

    _dis = np.zeros((buffer_size, 6), dtype=np.float32)
    dis_shm, dis = shm_create(_dis, dis_shm_name)

    def close_shm():
        sr_shm.close()
        sr_shm.unlink()

        acc_shm.close()
        acc_shm.unlink()

        cal_shm.close()
        cal_shm.unlink()

        vel_shm.close()
        vel_shm.unlink()

        dis_shm.close()
        dis_shm.unlink()

        print("shm closed")

    atexit.register(close_shm)

    serial_manager = Process(
        target=serial_process,
        args=(
            port,
            stop_event,
            save_event,
            (_sample_rate, sr_shm.name),
            (_cal_offset, cal_shm.name),
            (_acc, acc_shm.name),
            (_vel, vel_shm.name),
            (_dis, dis_shm.name),
        ),
        daemon=True,
    )

    serial_manager.start()

    window = init()
    imgui_impl = init_imgui(window)

    show_acc_lin, show_acc_rot = False, True

    while not stop_event.is_set() and not window_should_close(window):
        update()

        imgui.new_frame()
        imgui.begin("Sampling")

        imgui.text(f"device: {port}")
        ts, sr = sample_rate
        imgui.text(f"total samples: {ts:.0f}")
        imgui.text(f"sample rate: {sr:.2f} Hz")

        imgui.separator()

        _, show_acc_lin = imgui.checkbox("linear accel (m/s^2)", show_acc_lin)
        if show_acc_lin:
            points = acc[:, 1:4]
            draw_line_strip(points, WHITE)
            draw_arrow(ORIGIN, points[0], WHITE, 5, 10)

        _, show_acc_rot = imgui.checkbox("angular accel (rad/s^2)", show_acc_rot)
        if show_acc_rot:
            # points = acc[:, 4:7] - cal_offset
            points = acc[:, 4:7]
            draw_line_strip(points, WHITE)
            draw_arrow(ORIGIN, points[0], WHITE, 5, 10)

        imgui.separator()

        if cal := imgui.button("calibrate gyro"):
            cal_offset[:] = np.mean(acc[:, 4:7], axis=0)

        imgui.text(f"gyro offsets:")
        imgui.text(f"x: {cal_offset[0]:.4f} rad/s^2")
        imgui.text(f"y: {cal_offset[1]:.4f} rad/s^2")
        imgui.text(f"z: {cal_offset[2]:.4f} rad/s^2")

        imgui.separator()

        if imgui.button("reset integration") or cal:
            vel[:] = 0.0
            dis[:] = 0.0

        imgui.separator()

        imgui.text(f"current buffer: {ts % buffer_size:.0f}/{buffer_size}")
        imgui.text(f"buffer time: {buffer_size / sr :.2f} s")
        if imgui.button("save next buffer"):
            save_event.set()
            print("set save next buffer")

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
        main("COM10")
    else:
        main("/dev/ttyACM0")
