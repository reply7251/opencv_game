import cv2
import numpy as np

default_screen_size = (1920, 1080)
radio = 1.0

screen_size = int(default_screen_size[0] * radio), int(default_screen_size[1] * radio)

class Component:
    def __init__(self) -> None:
        self._enabled = True

    def inside(self, x: int, y: int) -> bool:
        pass

    def draw(self):
        pass
    
    def update(self):
        pass
    
    def children(self) -> list['Component']:
        return []

    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        return False

    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter()
    def set_enabled(self, enabled):
        self._enabled = enabled
        self.update()

class Button(Component):
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x, self.y, self.width, self.height = x, y, width, height
    
    def inside(self, x: int, y: int) -> bool:
        return self.x <= x <= self.x+self.width and self.y < y <= self.y + self.height

    def draw(self):
        pass

    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        cancel = False
        if self.enabled and self.inside(x, y):
            cancel = True
        return cancel

class Board(Component):
    def __init__(self, name: str) -> None:
        self.name = name
        cv2.namedWindow(name)
        self.buf = np.zeros(screen_size, np.uint8)
        self._children: list[Component] = []
        cv2.setMouseCallback(name, self.on_mouse)
    
    @property
    def children(self) -> list[Component]:
        return self.children

    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        is_cancelled = False
        for child in self.children:
            if child.on_mouse(x, y, flags):
                is_cancelled = True
        pass