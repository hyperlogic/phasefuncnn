#
# Copyright (c) 2025 Anthony J. Thibault
# This software is licensed under the MIT License. See LICENSE for more details.
#

from time import perf_counter

import numpy as np
import platform
import pygame
import pygfx as gfx
import wgpu
from wgpu.gui.auto import WgpuCanvas

import math_util as mu
from flycam import FlyCamInterface, FlyCam

KEY_REPEAT_DELAY = 1 / 2
KEY_REPEAT_RATE = 1 / 20


class RenderBuddy:
    joystick: pygame.joystick.JoystickType | None
    scene: gfx.Scene
    camera: gfx.PerspectiveCamera
    canvas: wgpu.gui.WgpuCanvasBase
    renderer: gfx.renderers.WgpuRenderer
    flycam: FlyCamInterface
    last_tick_time: float
    left_stick: np.ndarray
    right_stick: np.ndarray
    mouse_look_stick: np.ndarray
    roll_amount: float
    up_amount: float
    pointer_down: bool
    pointer_drag_start: np.ndarray
    dpad_right_down: bool
    dpad_right_time: float
    dpad_left_down: bool
    dpad_left_time: float

    def __init__(self):
        pygame.init()
        # joystick doesn't work on linux.
        if pygame.joystick.get_count() > 0 and platform.system() != "Linux":
            joystick = pygame.joystick.Joystick(0)  # Use the first joystick
            joystick.init()
            print(f"Joystick initialized: {joystick.get_name()}")
            print(f"  num_axes = {joystick.get_numaxes()}")
            print(f"  num_buttons = {joystick.get_numbuttons()}")
            print(f"  num_hats = {joystick.get_numhats()}")
        else:
            joystick = None
            print("No joystick detected.")
        self.joystick = joystick

        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight(intensity=1))
        self.scene.add(gfx.DirectionalLight())
        self.scene.add(gfx.helpers.GridHelper(size=100))
        self.camera = gfx.PerspectiveCamera(70, 1)
        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)

        MOVE_SPEED = 22.5
        ROT_SPEED = 1.15
        self.flycam = FlyCam(
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0.1, 10, 50], dtype=np.float32),
            np.array([0, 0, 0, 1], dtype=np.float32),
            MOVE_SPEED,
            ROT_SPEED,
        )

        self.renderer.add_event_handler(lambda event: self.on_key_down(event), "key_down")
        self.renderer.add_event_handler(lambda event: self.on_key_up(event), "key_up")
        self.renderer.add_event_handler(lambda event: self.on_close(event), "close")
        self.renderer.add_event_handler(lambda event: self.on_pointer_up(event), "pointer_up")
        self.renderer.add_event_handler(lambda event: self.on_pointer_down(event), "pointer_down")
        self.renderer.add_event_handler(lambda event: self.on_pointer_move(event), "pointer_move")
        self.renderer.add_event_handler(lambda event: self.on_before_render(event), "before_render")

        self.canvas.request_draw(lambda: self._animate())

        self.last_tick_time = perf_counter()
        self.left_stick = np.array([0, 0], dtype=np.float32)
        self.right_stick = np.array([0, 0], dtype=np.float32)
        self.mouse_look_stick = np.array([0, 0], dtype=np.float32)
        self.roll_amount = 0
        self.up_amount = 0
        self.pointer_down = False
        self.pointer_drag_start = np.array([0, 0], dtype=np.float32)
        self.dpad_right_down = False
        self.dpad_right_time = 0
        self.dpad_left_down = False
        self.dpad_left_time = 0

    def _animate(self):
        # request draw
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

    def on_key_down(self, event: gfx.objects.KeyboardEvent):
        if event.key == "Escape":
            self.canvas.close()
        elif event.key == "a":
            self.left_stick[0] -= 1
        elif event.key == "d":
            self.left_stick[0] += 1
        elif event.key == "w":
            self.left_stick[1] += 1
        elif event.key == "s":
            self.left_stick[1] -= 1
        elif event.key == "ArrowLeft":
            self.right_stick[0] -= 1
        elif event.key == "ArrowRight":
            self.right_stick[0] += 1
        elif event.key == "ArrowUp":
            self.right_stick[1] += 1
        elif event.key == "ArrowDown":
            self.right_stick[1] -= 1
        elif event.key == "r":
            self.up_amount += 1
        elif event.key == "f":
            self.up_amount -= 1
        elif event.key == "q":
            self.roll_amount -= 1
        elif event.key == "e":
            self.roll_amount += 1

    def on_key_up(self, event: gfx.objects.KeyboardEvent):
        if event.key == "a":
            self.left_stick[0] += 1
        elif event.key == "d":
            self.left_stick[0] -= 1
        if event.key == "w":
            self.left_stick[1] -= 1
        elif event.key == "s":
            self.left_stick[1] += 1
        elif event.key == "ArrowLeft":
            self.right_stick[0] += 1
        elif event.key == "ArrowRight":
            self.right_stick[0] -= 1
        elif event.key == "ArrowUp":
            self.right_stick[1] -= 1
        elif event.key == "ArrowDown":
            self.right_stick[1] += 1
        elif event.key == "r":
            self.up_amount -= 1
        elif event.key == "f":
            self.up_amount += 1
        elif event.key == "q":
            self.roll_amount += 1
        elif event.key == "e":
            self.roll_amount -= 1

    def on_pointer_up(self, event: gfx.objects.PointerEvent):
        if event.button == 2 and self.pointer_down:
            self.pointer_down = False

    def on_pointer_move(self, event: gfx.objects.PointerEvent):
        if self.pointer_down:
            mouse_pos = np.array([event.x, event.y], dtype=np.float32)
            mouse_rel = mouse_pos - self.pointer_drag_start
            self.pointer_drag_start = mouse_pos
            MOUSE_SENSITIVITY = 0.0025
            self.mouse_look_stick[0] = self.mouse_look_stick[0] + MOUSE_SENSITIVITY * mouse_rel[0]
            self.mouse_look_stick[1] = self.mouse_look_stick[1] - MOUSE_SENSITIVITY * mouse_rel[1]

    def on_pointer_down(self, event: gfx.objects.PointerEvent):
        if event.button == 2:
            self.pointer_down = True
            mouse_pos = np.array([event.x, event.y], dtype=np.float32)
            self.pointer_drag_start = mouse_pos

    def on_before_render(self, event):
        start = perf_counter()

        _ = event
        now = perf_counter()
        dt = now - self.last_tick_time
        self.last_tick_time = now

        self.dpad_left_time -= dt
        self.dpad_right_time -= dt

        joystick_left_stick = np.array([0, 0], dtype=np.float32)
        joystick_right_stick = np.array([0, 0], dtype=np.float32)
        joystick_roll_amount = 0
        joystick_up_amount = 0

        if self.joystick:
            joystick_left_stick[0] = mu.deadspot(self.joystick.get_axis(0))
            joystick_left_stick[1] = mu.deadspot(-self.joystick.get_axis(1))
            joystick_right_stick[0] = mu.deadspot(self.joystick.get_axis(2))
            joystick_right_stick[1] = mu.deadspot(-self.joystick.get_axis(3))

            if self.joystick.get_button(4):  # Left Bumper
                joystick_roll_amount += 1
            if self.joystick.get_button(5):  # Right Bumper
                joystick_roll_amount -= 1

            if self.joystick.get_numhats() > 0:
                dpad_state = self.joystick.get_hat(0)
                # d-pad up/down
                joystick_up_amount += dpad_state[1]

                # d-pad left/right
                if dpad_state[0] == -1 and (not self.dpad_left_down or self.dpad_left_time < 0):
                    self.on_dpad_left()
                    self.dpad_left_time = KEY_REPEAT_RATE if self.dpad_left_down else KEY_REPEAT_DELAY
                    self.dpad_left_down = True

                elif dpad_state[0] == 1 and (not self.dpad_right_down or self.dpad_right_time < 0):
                    self.on_dpad_right()
                    self.dpad_right_time = KEY_REPEAT_RATE if self.dpad_right_down else KEY_REPEAT_DELAY
                    self.dpad_right_down = True
                elif dpad_state[0] == 0:
                    self.dpad_left_down = False
                    self.dpad_right_down = False

        mouse_stick = self.mouse_look_stick / dt
        self.mouse_look_stick = np.array([0, 0], dtype=np.float32)

        left_stick = np.clip(self.left_stick + joystick_left_stick, -1, 1)
        right_stick = np.clip(self.right_stick + mouse_stick + joystick_right_stick, -1, 1)
        roll_amount = np.clip(self.roll_amount + joystick_roll_amount, -1, 1)
        up_amount = np.clip(self.up_amount + joystick_up_amount, -1, 1)

        self.flycam.process(dt, left_stick, right_stick, roll_amount, up_amount)
        self.camera.set_state({"position": self.flycam.pos, "rotation": self.flycam.rot})

        # animate stuff here
        animate_start = perf_counter()
        self.on_animate(dt)
        animate_end = perf_counter()

        end = perf_counter()
        # print(f"total {int((end - start) * 1000)} ms, animate = {int((animate_end - animate_start) * 1000)} ms")

    def on_close(self, event: gfx.Event):
        _ = event
        pass

    def on_dpad_left(self):
        pass

    def on_dpad_right(self):
        pass

    def on_animate(self, dt: float):
        _ = dt
        pass
