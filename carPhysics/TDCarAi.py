from Box2D import *
import math


class TDTyreAi(object):

    def __init__(self, car, max_drive_force=150,
                 dimensions=(0.5, 1.25), density=1.0,
                 position=(0, 0)):

        self.car = car
        world = car.body.world

        self.current_traction = 1

        self.max_drive_force = max_drive_force

        self.ground_areas = []

        # Create body
        friction = 0.3
        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(box=dimensions, density=density, friction=friction)
        self.body.userData = {'obj': self}

        car_collision_filter = b2Filter(
            groupIndex=-1,
            categoryBits=0x003,
            maskBits=0xFFFF
        )
        self.body.fixtures[0].filterData = car_collision_filter

    def get_lateral_velocity(self):
        current_right_normal = self.body.GetWorldVector(b2Vec2(1, 0))
        return b2Dot(current_right_normal, self.body.linearVelocity) * current_right_normal

    def get_forward_velocity(self):
        current_forward_normal = self.body.GetWorldVector(b2Vec2(0, 1))
        return b2Dot(current_forward_normal, self.body.linearVelocity) * current_forward_normal

    def update_friction(self):
        impulse = self.body.mass * -self.get_lateral_velocity()
        self.body.ApplyLinearImpulse(self.current_traction * impulse, self.body.worldCenter, True)
        self.body.ApplyAngularImpulse(0.1 * self.current_traction * self.body.inertia * -self.body.angularVelocity, True)

        current_forward_normal = self.get_forward_velocity()
        current_forward_speed = current_forward_normal.Normalize()
        drag_force_magnitude = -2 * current_forward_speed
        self.body.ApplyForce(drag_force_magnitude * current_forward_normal, self.body.worldCenter, True)

    def update_drive(self, desired_speed):
        # find current speed in forward direction
        current_forward_normal = self.body.GetWorldVector(b2Vec2(0, 1))
        current_speed = b2Dot(self.get_forward_velocity(), current_forward_normal)

        # apply necessary force
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(self.current_traction * force * current_forward_normal, self.body.worldCenter, True)


class TDCarAi(object):

    vertices = [(2, -3),
                (2, 3),
                (1.5, 5.5),
                (-1.5, 5.5),
                (-2, 3),
                (-2, -3),
                ]

    tyre_anchors = [(-2.0, -2),
                    (2.0, -2),
                    (-2.0, 4),
                    (2.0, 4),
                    ]

    def __init__(self, world, vertices=None, tyre_anchors=None,
                 max_forward_speed=100.0, max_backward_speed=-20,
                 density=0.1, position=(0, 0), **tyre_kws):
        if vertices is None:
            vertices = TDCarAi.vertices

        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(vertices=vertices, density=density)
        self.body.userData = {'obj': self}

        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed

        # Don't let cars collide with each other
        car_collision_filter = b2Filter(
            groupIndex=-1,
            categoryBits=0x003,
            maskBits=0xFFFF
        )
        self.body.fixtures[0].filterData = car_collision_filter
        self.tyres = [TDTyreAi(self, **tyre_kws) for i in range(4)]

        if tyre_anchors is None:
            anchors = TDCarAi.tyre_anchors

        joints = self.joints = []
        for tyre, anchor in zip(self.tyres, anchors):
            j = world.CreateRevoluteJoint(bodyA=self.body,
                                          bodyB=tyre.body,
                                          localAnchorA=anchor,
                                          localAnchorB=(0, 0),
                                          enableMotor=False,
                                          maxMotorTorque=1000,
                                          enableLimit=True,
                                          lowerAngle=0,
                                          upperAngle=0
                                          )
            tyre.body.position = self.body.worldCenter + anchor
            joints.append(j)

        # Track gating system
        self.lastGated = -1

    def update(self, desired_speed, desired_angle, hz):

        for tyre in self.tyres:
            tyre.update_friction()

        # Set speed
        if desired_speed > self.max_forward_speed:
            desired_speed = self.max_forward_speed
        elif desired_speed < self.max_backward_speed:
            desired_speed = self.max_backward_speed

        for tyre in self.tyres:
            tyre.update_drive(desired_speed)

        # Set angle - from lock to lock in 0.5 sec
        lock_angle = math.radians(40.)
        turn_speed_per_sec = math.radians(160.)
        turn_per_timestep = turn_speed_per_sec / hz

        if desired_angle > lock_angle:
            desired_angle = lock_angle
        elif desired_angle < -lock_angle:
            desired_angle = -lock_angle

        front_left_joint, front_right_joint = self.joints[2:4]
        angle_now = front_left_joint.angle
        angle_to_turn = desired_angle - angle_now

        if angle_to_turn < -turn_per_timestep:
            angle_to_turn = -turn_per_timestep
        elif angle_to_turn > turn_per_timestep:
            angle_to_turn = turn_per_timestep

        new_angle = angle_now + angle_to_turn

        # Rotate the tyres by locking the limits:
        front_left_joint.SetLimits(new_angle, new_angle)
        front_right_joint.SetLimits(new_angle, new_angle)
