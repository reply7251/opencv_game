import cv2
import numpy as np

default_screen_size = (1080, 1920)
radio = 1.0

screen_size = int(default_screen_size[0] * radio), int(default_screen_size[1] * radio)

class Component:
    def __init__(self, parent: 'Component' = None) -> None:
        self._enabled = True
        self.parent = parent
        self._children: list[Component] = []
        if parent != None:
            parent.children.append(self)

    def inside(self, x: int, y: int) -> bool:
        pass

    def draw(self, buffer):
        pass
    
    def update(self):
        for child in self.children:
            child.update()

    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        return False
    
    @property
    def children(self) -> list['Component']:
        return self._children

    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def set_enabled(self, enabled):
        self._enabled = enabled
        self.update()
    
    def reset(self):
        pass
"""
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
"""

class Vector:
    def __init__(self, x, y) -> None:
        self.vec = np.array((x,y), float)
    
    @property
    def x(self):
        return self.vec[0]
    
    @property
    def y(self):
        return self.vec[1]

def is_collide(type1, pos1, type2, pos2):
    if type1 == Ball:
        pos1
    pass

class Collidable:
    def __init__(self, *args) -> None:
        self._vec = Vector(0,0)

    @property
    def vector(self):
        return self._vec
    
    @vector.setter
    def set_vec(self, vec: Vector):
        self._vec = vec

    def get_position(self):
        pass

    def will_collide(self, other: 'Collidable') -> bool:
        pass

    def get_collision(self, other: 'Collidable') -> list[tuple[Vector]]:
        pass

    def ray(self) -> list['Collidable']:
        pass

    def after(self, time) -> np.ndarray:
        pass

class Ball(Component, Collidable):
    ref_count = 0
    def __init__(self, parent: 'Component' = None) -> None:
        super().__init__(parent)
        Collidable.__init__(self)
        self.id = Ball.ref_count
        Ball.ref_count += 1
        self.set_pos(0, 0)
        self.cache_after = [None * 40]
    
    def draw(self, buffer):
        cv2.circle(buffer, (int(self.x), int(self.y)), 30, (255,0,0),-1)
    
    def set_pos(self, x, y):
        self.x = x
        self.y = y
    
    def ray(self) -> list['Collidable']:
        collidables = [collidable for collidable in [self.parent, *self.parent.children] if isinstance(collidable, Collidable)]
        result = []
        for i in range(40):
            i_pos = self.after(i)
            for collidable in collidables:
                o_pos = collidable.after(i)
    
    def reset(self):
        self.cache_after = [None * 40]
    
    def after(self, time) -> np.ndarray:
        if self.cache_after[time]:
            return self.cache_after[time] 
        offset = self.vector.vec * time
        self.cache_after[time] = np.array((self.x, self.y), float) + offset
        return self.cache_after[time]
            

    def update(self):
        collidables = [collidable for collidable in [self.parent, *self.parent.children] if isinstance(collidable, Collidable)]
        collidables.sort(key=lambda x: x.get_distance(self) or 1e9)
        for collidable in collidables:
            if self.will_collide(collidable):
                (i_pos, i_vec), (o_pos, o_vec) = self.get_collision(collidable)

            else:
                vec = self.vector
                self.x += vec.x
                self.y += vec.y
                l = np.linalg.norm(self.vector.vec)
                if l:
                    l = max(l - 0.1, 0) / l #摩擦
                    self.vector.vec *= l
                else:
                    self.vector.vec *= 0
    

class Pole(Component, Collidable):
    pass

class Board(Component, Collidable):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        self.buf: np.ndarray = np.zeros((*screen_size,3), np.uint8)
        cv2.setMouseCallback(name, self.on_mouse)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.running = True
        self.mouse_pos = (0,0)

    def on_mouse(self, x: int, y: int, flags: int, *param) -> bool:
        for child in self.children:
            if child.on_mouse(x, y, flags):
                break
        self.mouse_pos = x, y
        pass
    
    def draw(self, buffer):
        self.buf *= 0
        cv2.putText(self.buf, str(self.mouse_pos), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        for child in self.children:
            child.draw(buffer)

    def mainloop(self):
        while self.running:
            self.reset()
            self.update()
            self.draw(self.buf)
            cv2.imshow(self.name, self.buf)
            if cv2.waitKey(25) & 0xff == 0x1b:
                self.running = False
        pass



if __name__ == "__main__":
    board = Board("test")
    ball = Ball(board)
    ball.set_pos(100,200)
    board.mainloop()