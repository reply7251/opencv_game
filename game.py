import enum
import cv2
import numpy as np
import pymunk as pm
import math

#127 254

default_screen_size = (1080, 1920)
unit = 220
boarder_size = (unit * 4, unit * 8)
radio = 1.0
boarder_thickness = 30
mid_pocket_size = 0.25 * unit # 中袋0.5 底袋0.45
corner_pocket_size = unit * 0.45 / math.sqrt(2)
boarder4_length = unit*4 - mid_pocket_size - corner_pocket_size
boarder2_length = unit*4 - corner_pocket_size * 2

ball_size = int(boarder_size[1] / 45)

screen_size = int(default_screen_size[0] * radio), int(default_screen_size[1] * radio)

class CollisionType:
    CUE_BALL = 0
    POLE = 1
    BALL = 2
    OTHERS = 100

class State:
    IDLE = 0
    WAIT = 1

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

class Ball(Component):
    def __init__(self, ball_id, pos: tuple[float, float] = (0, 0), parent: 'GameBoard' = None) -> None:
        super().__init__(parent)

        self.body = pm.Body()
        self.circle = pm.Circle(self.body, ball_size)
        self.circle.density = 1
        self.circle.elasticity = 1
        self.body.position = pos
        parent.space.add(self.body, self.circle)

        if ball_id == 15:
            self.circle.collision_type = CollisionType.CUE_BALL
        else:
            self.circle.collision_type = CollisionType.BALL
        self.ball_id = ball_id
    
    def draw(self, buffer):
        color = [(255,0,0), (255,255,255)]
        if self.ball_id == 15:
            color.reverse()
        cv2.circle(buffer, (self.pos.int_tuple), int(self.circle.radius), color[0],-1)
        cv2.putText(buffer, str(self.ball_id), (self.pos - (20,0)).int_tuple, cv2.FONT_HERSHEY_PLAIN, 2, color[1])
    
    @property
    def pos(self) -> pm.Vec2d:
        return self.body.position
    
    @pos.setter
    def set_pos(self, x, y):
        self.body.position = x, y

    def update(self):
        vec:pm.Vec2d = self.body.velocity
        if vec.length:
            self.body.velocity *= (vec.length - 0.3) / vec.length

        pass

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
        if self.parent.state != State.IDLE:
            return
        verts = []
        for v in [(0,0), (0.005*unit, 0.025*unit), (0.001 * unit, 0.05*unit), (-800, 0.075*unit), (-800, -0.075*unit)]:
            v = pm.Vec2d(*v)
            x = v.rotated(self.body.angle)[0] + self.body.position[0]
            y = v.rotated(self.body.angle)[1] + self.body.position[1]
            verts.append((int(0 if math.isnan(x) else x), int(0 if math.isnan(y) else y)))
        cv2.polylines(buffer, [np.array(verts)], True, (0,255,0), 5)
        #cv2.putText(buffer, str(self.body.angle), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    
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
        self.shape = pm.Poly(parent.space.static_body, [(0,0),(length,0),(length, boarder_thickness),(0, boarder_thickness)], transform)
        self.shape.elasticity = 1
        parent.space.add(self.shape)
        self.rotation = rotate
    
    def draw(self, buffer):
        cv2.polylines(buffer, [np.array([i.int_tuple for i in self.shape.get_vertices()], np.int32)], True, (0,255,0), 5)

class Boarder4(Boarder):
    def __init__(self, position, rotate, parent: 'GameBoard' = None) -> None:
        super().__init__(position, rotate, boarder4_length, parent)
        self.shape.collision_type = CollisionType.OTHERS

class Boarder2(Boarder):
    def __init__(self, position, rotate, parent: 'GameBoard' = None) -> None:
        super().__init__(position, rotate, boarder2_length, parent)
        self.shape.collision_type = CollisionType.OTHERS

class OuterBoarder(Component):
    def __init__(self, parent: 'GameBoard' = None) -> None:
        super().__init__(parent)

        self.body = pm.Body(body_type=pm.Body.STATIC)
        pos = [(0,0), (screen_size[1], 0), (screen_size[1], screen_size[0]), (0, screen_size[0])]
        self.shapes = [pm.Segment(self.body, pos[i], pos[(i+1)%4], 1) for i in range(4)]
            
        #self.shape = pm.Poly(self.body, )
        parent.space.add(self.body, *self.shapes)
    
    def draw(self, buffer):
        for shape in self.shapes:
            cv2.line(buffer, shape._get_a().int_tuple, shape._get_b().int_tuple, (0,255,0), 5)

class GameBoard(Component):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        self.buf: np.ndarray = np.zeros((*default_screen_size,3), np.uint8)
        cv2.setMouseCallback(name, self.on_mouse)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.running = True
        self.mouse_pos = (0,0)

        self.space = pm.Space()
        self.space.gravity = (0,0)
        self.dt = 25

        self.state = State.IDLE
        self.balls: list[Ball] = []
        self.moving_ball = None

        self.build()

    def build(self):
        offset_y = (default_screen_size[0] - boarder_size[0]) // 2
        offset_x = (default_screen_size[1] - boarder_size[1]) // 2
        Boarder4((offset_x + boarder4_length + corner_pocket_size + mid_pocket_size * 2, offset_y - boarder_thickness), 0, self) #右上
        Boarder4((offset_x + corner_pocket_size, offset_y - boarder_thickness), 0, self) #左上
        Boarder2((offset_x, corner_pocket_size + offset_y), 0.5, self) #左
        Boarder4((offset_x + corner_pocket_size, boarder2_length + corner_pocket_size * 2 + offset_y),0,self) #左下
        Boarder4((offset_x + boarder4_length + corner_pocket_size + mid_pocket_size * 2, boarder2_length + corner_pocket_size * 2 + offset_y),0,self) #右下
        Boarder2((offset_x + unit*8 + boarder_thickness, corner_pocket_size + offset_y),0.5,self) #右
        offset_ball_x = offset_x + boarder_size[1] * 0.6
        offset_ball_y = offset_y + boarder_size[0] / 2
        self.cue_ball = Ball(15, (offset_ball_x - unit * 3.5, offset_ball_y),self)
        self.balls.append(self.cue_ball)
        
        i = 0
        sqrt3 = math.sqrt(3)
        for rows in range(5):
            for cols in range(rows+1):
                pos = offset_ball_x + rows * sqrt3 * ball_size * 2.1 / 2, offset_ball_y + (cols - (rows)/2) * ball_size * 2.1
                self.balls.append(Ball(i , pos,self))
                i+=1

        Pole((screen_size[1] / 2, screen_size[0] / 2), self)
        OuterBoarder(self)

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            return False
            
        handler = self.space.add_collision_handler(CollisionType.BALL, CollisionType.POLE)
        handler.pre_solve = pre_solve

        def pre_solve(arb: pm.Arbiter, space: pm.Space, data):
            if self.state == State.IDLE:
                self.state = State.WAIT
                return True
            return False
            
        handler = self.space.add_collision_handler(CollisionType.CUE_BALL, CollisionType.POLE)
        handler.pre_solve = pre_solve

    def get_cue_ball(self) -> Ball:
        return self.cue_ball

    def on_mouse(self, event, x: int, y: int, flags: int, *param) -> bool:
        self.mouse_pos = x, y
        for child in self.children:
            if child.on_mouse(x, y, flags):
                break
        pass
    
    def draw(self, buffer):
        self.buf *= 0
        cv2.putText(self.buf, str(self.mouse_pos), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        if self.moving_ball:
            cv2.putText(self.buf, str(self.moving_ball.body.velocity), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
            pass
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
            if cv2.waitKey(self.dt) & 0xff == 0x1b:
                self.running = False
        pass

    def update(self):
        super().update()
        for ball in self.balls:
            if 0 < ball.body.velocity.length < 0.2:
                ball.body.velocity *= 0
            if ball.body.velocity.length > 0:
                self.moving_ball = ball
                break
        else:
            self.state = State.IDLE



if __name__ == "__main__":
    board = GameBoard("test")
    board.mainloop()