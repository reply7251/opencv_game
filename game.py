import enum
import cv2
import numpy as np
import pymunk as pm
import math
import random
from typing import Callable
#127 254

debug = False

default_screen_size = (1080, 1920)
unit = 220
boarder_size = (unit * 4, unit * 8)
radio = 1.0
boarder_thickness = 30
mid_pocket_size = 0.22 * unit # 中袋0.5 底袋0.45
corner_pocket_size = unit * 0.3 / math.sqrt(2)
boarder4_length = unit*4 - mid_pocket_size - corner_pocket_size
boarder2_length = unit*4 - corner_pocket_size * 2

ball_size = int(boarder_size[1] / 45)

screen_size = int(default_screen_size[0] * radio), int(default_screen_size[1] * radio)

sqrt3 = math.sqrt(3)

offset_y = (default_screen_size[0] - boarder_size[0]) // 2
offset_x = (default_screen_size[1] - boarder_size[1]) // 2
offset_ball_x = offset_x + boarder_size[1] * 0.6
offset_ball_y = offset_y + boarder_size[0] / 2

def clamp(x, low, high):
    return max(min(x, high), low)

class CollisionType:
    CUE_BALL = 0
    POLE = 1
    BALL = 2
    HOLE = 3
    OTHERS = 100

class State:
    IDLE = 1
    WAIT = 2
    PLACE = 4
    WAIT_MOUSE = 8

img_folder = "./imgs/"
img_balls = cv2.cvtColor(cv2.imread(img_folder + "ball.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA) 
ball_height, ball_width = 56, 54375
img_balls_splitted = [img_balls[:, x//1000:(x+ball_width)//1000 , : ] for x in range(0, img_balls.shape[1]*1000, ball_width)]
img_cue = cv2.cvtColor(cv2.imread(img_folder + "cue2.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA)[:,3:]
img_board = cv2.rotate(cv2.cvtColor(cv2.imread(img_folder + "background.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2BGRA), cv2.ROTATE_180) 

class Images:
    def ball(index : int) -> np.ndarray:
        return img_balls_splitted[index]
    
    def cue() -> np.ndarray:
        return img_cue
    
    def board() -> np.ndarray:
        return img_board
    
    def rotate(img: np.ndarray, angle, center = None) -> np.ndarray:
        h, w = img.shape[:2]
        if center == None:
            center = w//2, h//2
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        return cv2.warpAffine(img, M, (nW, nH))
    
    def draw2(img: np.ndarray, buffer: np.ndarray, x, y, mask: np.ndarray = None):
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        h, w = buffer.shape[:2]
        mask = mask[max(0, -y):min(img.shape[0],h-y), max(0, -x):min(img.shape[1],w-x)]
        img = img[max(0, -y):min(img.shape[0],h-y), max(0, -x):min(img.shape[1],w-x)]
        y = max(y, 0)
        x = max(x, 0)
        cv2.copyTo(img, mask, buffer[y:y+img.shape[0],x:x+img.shape[1]])

    def draw(img, buffer, pos, mask = None):
        Images.draw2(img, buffer, *pos, mask)
        pass
    
    def text(text, buffer, pos, color = (0,0,0), size = 1):
        cv2.putText(buffer, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

class Math:
    def rotate(x0, y0, x1, y1, rad):
        x2 = ((x1 - x0) * math.cos(rad)) - ((y1 - y0) * math.sin(rad)) + x0
        y2 = ((x1 - x0) * math.sin(rad)) + ((y1 - y0) * math.cos(rad)) + y0
        return x2, y2

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

class Button(Component):
    def __init__(self, parent: Component, x: int, y: int, width: int, height: int, label: str = "label", callback: Callable[[int,int,int], None] = None) -> None:
        super().__init__(parent)
        self.x, self.y, self.width, self.height = x, y, width, height
        self.label = label
        self.pressed = False
        self.mouse_on = False
        if callback == None:
            callback = lambda x,y,flags: None
        self.callback = callback
        self.font_size = 1.5
    
    def inside(self, x: int, y: int) -> bool:
        return self.x <= x <= self.right() and self.y <= y <= self.bottom()
    
    def bottom(self):
        return self.y + self.height
    
    def right(self):
        return self.x + self.width
    
    def center(self):
        return (self.x + self.right()) // 2, (self.y + self.bottom()) // 2
    
    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        self.mouse_on = self.inside(x, y)
        if self.mouse_on and self.pressed and flags == 0:
            self.callback(x, y, flags)
            return True
        self.pressed = bool(flags)
        return super().on_mouse(x, y, flags)

    def draw(self, buffer):
        Images.text(self.label, buffer, (self.x, self.center()[1]+int(self.font_size*10)), (255,0,0) if self.mouse_on else (0, 0, 0))

class Logger(Component):
    def __init__(self, parent: 'Component', x: int, y: int, line_limit = 10, last_time = 1000) -> None:
        super().__init__(parent)
        self.x, self.y = x, y
        self.last_time, self.time_left = last_time, last_time
        self.line_limit = line_limit
        self.lines = []
        self.time = 0

    
    def update(self):
        self.time += 1
        while self.lines and self.lines[0][1] < self.time:
            self.lines.pop(0)
        return super().update()
    
    def log(self, text: str):
        self.lines.append((text, self.time + self.last_time))
        if len(self.lines) > self.line_limit:
            self.lines.pop(0)
            self.time_left = self.last_time

    def draw(self, buffer):
        offset = 20
        for line, _ in self.lines:
            Images.text(line, buffer, (self.x, self.y + offset))
            offset += 30

class Ball(Component):
    def __init__(self, ball_id, pos: tuple[float, float] = (0, 0), parent: 'GameBoard' = None) -> None:
        super().__init__(parent)

        self.body = pm.Body()
        self.circle = pm.Circle(self.body, ball_size)
        self.circle.density = 1
        self.circle.elasticity = 1
        self.init_pos = pos
        self.body.position = pos
        parent.space.add(self.body, self.circle)
        self.circle.collision_type = CollisionType.BALL
        self.ball_id = ball_id
    
    def draw(self, buffer):
        img = Images.ball(self.ball_id).copy()
        buf = cv2.resize(img, (ball_size * 2, ball_size * 2))
        img, alpha = buf[:,:,0:3], buf[:,:,3]
        draw_pos = (self.pos-(ball_size,ball_size)).int_tuple
        Images.draw(img, buffer, draw_pos, alpha)
    
    @property
    def pos(self) -> pm.Vec2d:
        return self.body.position
    
    @pos.setter
    def set_pos(self, x, y):
        self.body.position = x, y

    def update(self):
        vec:pm.Vec2d = self.body.velocity
        if vec.length:
            self.body.velocity *= (vec.length - 0.5) / vec.length

    def reset(self):
        self.body.position = self.init_pos
        self.body.velocity = 0, 0
    
    def __repr__(self) -> str:
        return self.name()

    def name(self):
        return f"ball{self.ball_id+1}"
        
class CueBall(Ball):
    def __init__(self, ball_id, pos: tuple[float, float] = (0, 0), parent: 'GameBoard' = None) -> None:
        super().__init__(ball_id, pos, parent)
        self.circle.collision_type = CollisionType.CUE_BALL
        self.mouse = 0,0
    
    def reset(self):
        self.parent.state |= State.PLACE
        super().reset()
    
    def update(self):
        if self.parent.state & State.WAIT_MOUSE:
            if self.body.position.get_distance(self.mouse) > ball_size * 3:
                self.parent.state &= ~State.WAIT_MOUSE
        else:
            super().update()

    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        self.mouse = x, y
        if self.parent.state & State.PLACE:
            self.body.position = clamp(x, offset_x + ball_size * 2, offset_x + unit * 8 - ball_size * 2), clamp(y, offset_y + ball_size * 2, offset_y + unit * 4 - ball_size * 2)
            self.body.velocity = 0, 0
            if flags != 1 or (self.parent.state & State.IDLE) == 0:
                return
            for ball in self.parent.balls[1:]:
                if ball.body.position.get_distance((x, y)) < ball_size * 2.1:
                    return
            self.parent.state &= ~State.PLACE
    
    def name(self):
        return "cue_ball"


class Pole(Component):
    def __init__(self, pos: tuple[float, float] = (0, 0), parent: 'GameBoard' = None) -> None:
        super().__init__(parent)

        self.body = pm.Body(body_type=pm.Body.KINEMATIC)
        self.shape = pm.Poly(self.body, [(0,0), (0.005*unit, 0.025*unit), (0.001 * unit, 0.05*unit)])
        self.body.position = pos
        self.shape.elasticity = 0.5
        self.mouse = (0,0)
        parent.space.add(self.body, self.shape)
        self.shape.collision_type = CollisionType.POLE
        self.shoot = False

    def draw(self, buffer):
        if self.parent.state & State.IDLE == 0 or self.parent.state & (State.PLACE | State.WAIT_MOUSE):
            return
        img = Images.cue()
        pos: pm.Vec2d = self.body.position - (img.shape[0]/2, img.shape[0]/2)
        angle = math.degrees(-self.body.angle)
        
        rad = -self.body.angle#math.radians(math.radians(angle))
        point = pm.Vec2d(535, 0)
        point = pm.Vec2d(point.x * math.cos(rad) - point.y * math.sin(rad), point.x * math.sin(rad) + point.y * math.cos(rad))
        if -90 > angle > -180: #右上
            pass #pm.Vec2d(point.x * math.cos(rad) - point.y * math.sin(rad), point.x * math.sin(rad) + point.y * math.cos(rad))
        elif angle < -180: #右下
            point = pm.Vec2d(point.x, -point.y)#point = pm.Vec2d(point.x * math.cos(rad) - point.y * math.sin(rad), -(point.x * math.sin(rad) + point.y * math.cos(rad)))
        elif 0<angle < 90: #左下
            point = pm.Vec2d(-point.x, -point.y)#point = pm.Vec2d(-(point.x * math.cos(rad) - point.y * math.sin(rad)), -(point.x * math.sin(rad) + point.y * math.cos(rad)))
        else: #左上
            point = pm.Vec2d(-point.x, point.y)#point = pm.Vec2d(-(point.x * math.cos(rad) - point.y * math.sin(rad)), (point.x * math.sin(rad) + point.y * math.cos(rad)))

        img = Images.rotate(img, angle)
        img, alpha = img[:,:,0:3], img[:,:,3]
        Images.draw(img, buffer, (pos+point).int_tuple, alpha)

    
    def on_mouse(self, x: int, y: int, flags: int) -> bool:
        self.mouse = x, y

    def update(self):
        velocity = (pm.Vec2d(*self.mouse) - self.body.position) * 10
        cue_ball: Ball = self.parent.get_cue_ball()
        if self.shoot:
            return
        self.body.velocity = velocity
        self.body.angle = -math.atan2(*(cue_ball.pos - self.body.position)) + math.radians(90)
        

class Boarder(Component):
    def __init__(self, position, rotate, length, parent: 'GameBoard' = None) -> None:
        super().__init__(parent)
        transform = pm.Transform.identity().translated(*position).rotated(math.pi * rotate)
        self.shape = pm.Poly(parent.space.static_body, [(0,0),(length,0),(length-boarder_thickness, boarder_thickness),(0+boarder_thickness, boarder_thickness)], transform)
        self.shape.elasticity = 1
        parent.space.add(self.shape)
        self.rotation = rotate
        self.shape.collision_type = CollisionType.OTHERS
    
    def draw(self, buffer):
        if debug:
            cv2.polylines(buffer, [np.array([i.int_tuple for i in self.shape.get_vertices()], np.int32)], True, (0,255,0), 5)

class Boarder4(Boarder):
    def __init__(self, position, rotate, parent: 'GameBoard' = None) -> None:
        super().__init__(position, rotate, boarder4_length, parent)

class Boarder2(Boarder):
    def __init__(self, position, rotate, parent: 'GameBoard' = None) -> None:
        super().__init__(position, rotate, boarder2_length, parent)

class OuterBoarder(Component):
    def __init__(self, parent: 'GameBoard' = None) -> None:
        super().__init__(parent)

        self.body = pm.Body(body_type=pm.Body.STATIC)
        pos = [(0,0), (screen_size[1], 0), (screen_size[1], screen_size[0]), (0, screen_size[0])]
        self.shapes = [pm.Segment(self.body, pos[i], pos[(i+1)%4], 1) for i in range(4)]
            
        #self.shape = pm.Poly(self.body, )
        parent.space.add(self.body, *self.shapes)
    
    def draw(self, buffer):
        if debug:
            for shape in self.shapes:
                cv2.line(buffer, shape._get_a().int_tuple, shape._get_b().int_tuple, (0,255,0), 5)

class Hole(Component):
    def __init__(self, position, parent: 'GameBoard' = None) -> None:
        super().__init__(parent)
        self.body = pm.Body(body_type=pm.Body.STATIC)

        self.circle = pm.Circle(self.body, corner_pocket_size / 1.5)
        self.body.position = position
        parent.space.add(self.body, self.circle)

        self.circle.collision_type = CollisionType.HOLE
    
    def draw(self, buffer):
        if debug:
            cv2.circle(buffer, (self.body.position.int_tuple), int(self.circle.radius), (0,0,255),-1)


class GameBoard(Component):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        self.buf: np.ndarray = np.zeros((*default_screen_size,3), np.uint8)
        cv2.setMouseCallback(name, self.on_mouse)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.running = True
        self.mouse = None

        self.space = pm.Space()
        self.space.gravity = (0,0)
        self.dt = 25

        self.state = State.IDLE
        self.balls: list[Ball] = []
        self.moving_ball = None

        self.scale = 1.25
        self.pos = pm.Vec2d(-23,-2)

        self.build()

        self.reset_btn = Button(self, 170,10,200,50,'reset', lambda x, y, flags:self.reset())
        self.logger = Logger(self, 500, 10)
    
    def get_ball_from_shape(self, shape) -> Ball:
        for ball in self.balls:
            if ball.circle == shape:
                return ball

    def build(self):
        Boarder4((offset_x + boarder4_length + corner_pocket_size + mid_pocket_size * 2, offset_y - boarder_thickness), 0, self) #右上
        Boarder4((offset_x + corner_pocket_size, offset_y - boarder_thickness), 0, self) #左上
        Boarder2((offset_x - boarder_thickness, (boarder2_length + corner_pocket_size + offset_y)), 1.5, self) #左
        Boarder4((offset_x + corner_pocket_size + boarder4_length, boarder2_length + corner_pocket_size * 2 + offset_y+boarder_thickness),1,self) #左下
        Boarder4((offset_x + boarder4_length * 2 + corner_pocket_size + mid_pocket_size * 2, boarder2_length + corner_pocket_size * 2 + offset_y+boarder_thickness),1,self) #右下
        Boarder2((offset_x + unit*8 + boarder_thickness, corner_pocket_size + offset_y),0.5,self) #右
        Hole((offset_x + boarder4_length + corner_pocket_size + mid_pocket_size, offset_y - ball_size * 1.5),self) #上
        Hole((offset_x - ball_size, offset_y - ball_size),self) #左上
        Hole((offset_x - ball_size, offset_y + boarder2_length + corner_pocket_size * 2 + ball_size),self) #左下
        Hole((offset_x + boarder4_length + corner_pocket_size + mid_pocket_size, offset_y + boarder2_length + corner_pocket_size * 2 + ball_size * 1.5),self) #下
        Hole((offset_x + unit*8 + ball_size, offset_y + boarder2_length + corner_pocket_size * 2 + ball_size),self) #右下
        Hole((offset_x + unit*8 + ball_size, offset_y - ball_size),self) #右上
        self.cue_ball = CueBall(15, (offset_ball_x - unit * 3.5, offset_ball_y),self)
        self.balls.append(self.cue_ball)
        
        i = 0
        for rows in range(5):
            for cols in range(rows+1):
                #pos = offset_ball_x + rows * sqrt3 * ball_size * 2.1 / 2, offset_ball_y + (cols - (rows)/2) * ball_size * 2.1
                pos = i * ball_size * 2.1, 2000
                self.balls.append(Ball(i , pos,self))
                i+=1

        self.pole = Pole((screen_size[1] / 2, screen_size[0] / 2), self)
        OuterBoarder(self)

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            if self.state & State.PLACE:
                return False
            self.logger.log(f"{self.get_ball_from_shape(arb.shapes[0])} hits {self.get_ball_from_shape(arb.shapes[1])}")
            return True
            
        handler = self.space.add_collision_handler(CollisionType.BALL, CollisionType.CUE_BALL)
        handler.pre_solve = pre_solve

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            self.logger.log(f"{self.get_ball_from_shape(arb.shapes[0])} hits {self.get_ball_from_shape(arb.shapes[1])}")
            return True

        handler = self.space.add_collision_handler(CollisionType.BALL, CollisionType.BALL)
        handler.pre_solve = pre_solve

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            return False
            
        handler = self.space.add_collision_handler(CollisionType.BALL, CollisionType.POLE)
        handler.pre_solve = pre_solve

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            if arb.contact_point_set.points[0].distance < -ball_size * 0.5:
                shape = arb.shapes[0]
                self.get_ball_from_shape(shape).reset()
            return False
            
        handler = self.space.add_collision_handler(CollisionType.BALL, CollisionType.HOLE)
        handler.pre_solve = pre_solve
        handler = self.space.add_collision_handler(CollisionType.CUE_BALL, CollisionType.HOLE)
        handler.pre_solve = pre_solve

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            if self.state & State.IDLE and (self.state & (State.PLACE | State.WAIT_MOUSE)) == 0:
                self.state &= ~State.IDLE
                self.state |= State.WAIT | State.WAIT_MOUSE
                return True
            return False
            
        handler = self.space.add_collision_handler(CollisionType.CUE_BALL, CollisionType.POLE)
        handler.pre_solve = pre_solve
        
        self.reset()

    def reset(self):
        self.balls[0].body.position = offset_ball_x - unit * 3 + corner_pocket_size, offset_ball_y
        self.balls[0].body.velocity = 0, 0
        indices = [*range(0,15)]
        indices.remove(7)
        
        small = random.randint(0,6)
        great = random.randint(8,14)
        indices.remove(small)
        indices.remove(great)
        random.shuffle(indices)
        indices.insert(4, 7)
        indices.insert(10,small)
        indices.append(great)
        i = 0
        for rows in range(5):
            for cols in range(rows+1):
                pos = offset_ball_x + rows * sqrt3 * ball_size * 2.1 / 2, offset_ball_y + (cols - (rows)/2) * ball_size * 2.1
                ball = self.balls[indices[i]+1]
                ball.body.position = pos
                ball.body.velocity *= 0
                i+=1

    def get_cue_ball(self) -> Ball:
        return self.cue_ball

    def on_mouse(self, event, x: int, y: int, flags: int, *param) -> bool:
        self.mouse = x, y, flags, event
        for child in self.children:
            if child.on_mouse(x, y, flags):
                break
        pass
    
    def draw(self, buffer):
        self.buf *= 0
        img = Images.board()
        img = cv2.resize(img, (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)))
        img, alpha = img[:,:,:3], img[:,:,3]
        Images.draw(img, self.buf, self.pos.int_tuple, alpha)
        if debug:
            Images.text(f"{self.mouse} scale: {self.scale} pos: {self.pos}", buffer, (10, 50), (255,0,0))
        for child in self.children:
            child.draw(buffer)

    def mainloop(self):
        while self.running:
            for _ in range(self.dt):
                self.update()
                self.space.step(0.001)
            self.draw(self.buf)
            buf = cv2.cvtColor(self.buf, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, buf)
            key = cv2.waitKeyEx(self.dt)
            if key & 0xff == 0x1b:
                self.running = False
            elif key & 0xff == ord('r'):
                self.reset()
            elif key & 0xff == ord(',') and debug:
                self.scale += 0.01
            elif key & 0xff == ord('.') and debug:
                self.scale -= 0.01
            elif key % 0x10000 == 0 and debug:
                key = key // 0x10000 
                if key == 0x25:
                    self.pos += (-1, 0)
                elif key == 0x26:
                    self.pos += (0, -1)
                elif key == 0x27:
                    self.pos += (1, 0)
                elif key == 0x28:
                    self.pos += (0, 1)
                else:
                    print("ex:",hex(key))
            elif key != -1 and debug:
                print(key, chr(key & 0xff))
        pass

    def update(self):
        super().update()
        for ball in self.balls:
            if 0 < ball.body.velocity.length < 1:
                ball.body.velocity *= 0
            if ball.body.velocity.length > 0:
                self.moving_ball = ball
                break
        else:
            self.state &= ~State.WAIT
            self.state |= State.IDLE

if __name__ == "__main__":
    board = GameBoard("test")
    board.mainloop()