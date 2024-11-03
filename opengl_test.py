import os
import struct
import time

import glfw
import numpy as np
import serial

import imgui
from imgui.integrations.glfw import GlfwRenderer

import dearpygui.dearpygui as dpg

from queue import Empty
from multiprocessing import Process, Queue, Event

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
    glBegin(GL_POINTS)

    for i, point in enumerate(points):
        glColor3f(*aruco_point_color[i % aruco_point_color.shape[0]])
        glVertex3f(*point @ T)

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
        -1,
        1,
    )

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(pan_x, pan_y, 0.0)
    glRotatef(angle_x, 1.0, 0.0, 0.0)
    glRotatef(angle_y, 0.0, 1.0, 0.0)

    draw_axes()


def serial_process(port, stop_event, serial_data):
    ser = serial.Serial(port=port, baudrate=115200, dsrdtr=os.name != "nt")

    while not stop_event.is_set():
        ser.read_until(b"\x7E")
        buf = ser.read(16)
        if ser.read(1) != b"\x7D":
            break

        data = struct.unpack("<L6h", buf)
        serial_data.put((data[0], data[1:]))

    ser.close()
    stop_event.set()


def main(port, buffer_size=2048, sample_dt_buffer_size=128):
    serial_data = Queue()
    stop_event = Event()

    serial_manager = Process(
        target=serial_process,
        args=(port, stop_event, serial_data),
        daemon=True,
    )

    serial_manager.start()

    window = init()
    imgui_impl = init_imgui(window)

    sample_total = 0
    n, prev_n = 0, buffer_size - 1
    sample_buffer = np.empty((buffer_size, 6), dtype=np.float64)
    dt_buffer = np.empty(buffer_size, dtype=np.float64)

    new_data = False

    dt_i = 0
    sample_dt_buffer = np.zeros(sample_dt_buffer_size, dtype=np.float32)
    prev_time = time.time()

    while not stop_event.is_set() and not window_should_close(window):
        try:
            data_raw = serial_data.get_nowait()
            new_data = True
        except Empty:
            pass

        if new_data:
            current_time = time.time()
            sample_dt_buffer[dt_i] = current_time - prev_time
            prev_time = current_time
            dt_i = (dt_i + 1) % sample_dt_buffer_size

            dt, data = data_raw
            dt_buffer[n] = dt
            sample_buffer[n] = data

            n = (n + 1) % buffer_size
            prev_n = (prev_n + 1) % buffer_size
            sample_total += 1

            new_data = False

        update()

        # points = np.array(
        #     [
        #         [0, 0, 1],
        #         [0, 1, 0],
        #         [1, 0, 0],
        #         [0, 0, 0],
        #     ],
        #     dtype=np.float32,
        # )

        points = sample_buffer[:, 0:3].astype(np.float32) / 256

        imgui.new_frame()
        imgui.begin("Sampling")

        imgui.text(f"device: {port}")
        imgui.text(f"total samples: {sample_total}")
        imgui.text(f"sample rate {1/np.mean(sample_dt_buffer):.2f} Hz")

        imgui.spacing()

        imgui.text(f"current n: {n}")
        imgui.text(f"prev n: {prev_n}")

        imgui.spacing()

        if imgui.button("Clear buffer"):
            n, prev_n = 0, buffer_size - 1
            sample_total = 0

        imgui.end()
        imgui.render()
        imgui_impl.process_inputs()
        imgui_impl.render(imgui.get_draw_data())

        draw_points(points)

        glfw.swap_buffers(window)
        glfw.poll_events()

    stop_event.set()
    terminate()


if __name__ == "__main__":
    main("COM10")
