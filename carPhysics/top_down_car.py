
import numpy as np

from framework import (Framework, Keys, main)
from Box2D import *
import math


class TDGroundArea(object):
    """
    An area on the ground that the car can run over
    """

    def __init__(self, friction_modifier, gateIndex):
        self.friction_modifier = friction_modifier
        self.gateIndex = gateIndex


class TDTyre(object):

    def __init__(self, car, max_forward_speed=100.0,
                 max_backward_speed=-20, max_drive_force=150,
                 turn_torque=15, max_lateral_impulse=3,
                 dimensions=(0.5, 1.25), density=1.0,
                 position=(0, 0)):

        world = car.body.world

        self.current_traction = 1
        self.turn_torque = turn_torque
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.max_drive_force = max_drive_force
        self.max_lateral_impulse = max_lateral_impulse

        self.ground_areas = []

        # Create body
        friction = 0.3
        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(box=dimensions, density=density, friction=friction)
        self.body.userData = {'obj': self}

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

    def update_drive(self, keys):
        if 'up' in keys:
            desired_speed = self.max_forward_speed
        elif 'down' in keys:
            desired_speed = self.max_backward_speed
        else:
            return

        # find current speed in forward direction
        current_forward_normal = self.body.GetWorldVector(b2Vec2(0, 1))
        current_speed = b2Dot(self.get_forward_velocity(), current_forward_normal)

        # apply necessary force
        force = 0
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(self.current_traction * force * current_forward_normal, self.body.worldCenter, True)

    def update_turn(self, keys):
        # NOTE: not actually used
        if 'left' in keys:
            desired_torque = self.turn_torque
        elif 'right' in keys:
            desired_torque = self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)

    def add_ground_area(self, ud):
        if ud not in self.ground_areas:
            self.ground_areas.append(ud)
            self.update_traction()

    def remove_ground_area(self, ud):
        if ud in self.ground_areas:
            self.ground_areas.remove(ud)
            self.update_traction()

    def update_traction(self):
        if not self.ground_areas:
            self.current_traction = 1
        else:
            self.current_traction = 0
            mods = [ga.friction_modifier for ga in self.ground_areas]

            max_mod = max(mods)
            if max_mod > self.current_traction:
                self.current_traction = max_mod


class TDCar(object):

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
                 density=0.1, position=(0,0), **tyre_kws):
        if vertices is None:
            vertices = TDCar.vertices

        self.body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(vertices=vertices, density=density)
        self.body.userData = {'obj': self}

        self.tyres = [TDTyre(self, **tyre_kws) for i in range(4)]

        if tyre_anchors is None:
            anchors = TDCar.tyre_anchors

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

    def update(self, keys, hz):
        for tyre in self.tyres:
            tyre.update_friction()

        for tyre in self.tyres:
            tyre.update_drive(keys)

        # steering?
        lock_angle = math.radians(40.)
        # from lock to lock in 0.5 sec
        turn_speed_per_sec = math.radians(160.)
        turn_per_timestep = turn_speed_per_sec / hz
        desired_angle = 0.0

        if 'left' in keys:
            desired_angle = lock_angle
        elif 'right' in keys:
            desired_angle = -lock_angle

        front_left_joint, front_right_joint = self.joints[2:4]
        angle_now = front_left_joint.angle
        angle_to_turn = desired_angle - angle_now

        # TODO fix b2Clamp for non-b2Vec2 types
        if angle_to_turn < -turn_per_timestep:
            angle_to_turn = -turn_per_timestep
        elif angle_to_turn > turn_per_timestep:
            angle_to_turn = turn_per_timestep

        new_angle = angle_now + angle_to_turn
        # Rotate the tyres by locking the limits:
        front_left_joint.SetLimits(new_angle, new_angle)
        front_right_joint.SetLimits(new_angle, new_angle)


class TopDownCarFramework(Framework):
    name = "Top Down Car"
    description = "test"

    def __init__(self):
        super(TopDownCarFramework, self).__init__()

        # Top-down: no gravity
        self.world.gravity = (0, 0)

        self.key_map = {Keys.K_w: 'up',
                        Keys.K_s: 'down',
                        Keys.K_a: 'left',
                        Keys.K_d: 'right'
                        }

        self.pressed_keys = set()

        self.car = TDCar(self.world)

        # Add track sections
        for trackSectionIndex in range(10):
            width = 5
            halfHeight = 2

            centreX = 0
            centreY = (halfHeight + halfHeight*2)*trackSectionIndex
            rotation = np.radians(0)

            collisionObject = TDGroundArea(1, trackSectionIndex)

            segment = self.world.CreateStaticBody(userData={"obj": collisionObject})
            fixture = segment.CreatePolygonFixture(
                box=(width, halfHeight, (centreX , centreY), rotation)
            )
            fixture.sensor = True
            print(dir(fixture.shape))


    def Keyboard(self, key):
        key_map = self.key_map
        if key in key_map:
            self.pressed_keys.add(key_map[key])
        else:
            super(TopDownCarFramework, self).Keyboard(key)

    def KeyboardUp(self, key):
        key_map = self.key_map
        if key in key_map:
            self.pressed_keys.remove(key_map[key])
        else:
            super(TopDownCarFramework, self).KeyboardUp(key)

    def handle_contact(self, contact, began):

        #print(contact)

        # A contact happened -- see if a wheel hit a
        # ground area
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a, body_b = fixture_a.body, fixture_b.body
        ud_a, ud_b = body_a.userData, body_b.userData
        if not ud_a or not ud_b:
            return

        tyre = None
        ground_area = None
        for ud in (ud_a, ud_b):
            obj = ud['obj']
            if isinstance(obj, TDTyre):
                tyre = obj
            elif isinstance(obj, TDGroundArea):
                ground_area = obj

        if ground_area is not None and tyre is not None:
            if began:
                tyre.add_ground_area(ground_area)
            else:
                tyre.remove_ground_area(ground_area)

                # Update gate, if required
                if self.car.lastGated < ground_area.gateIndex:
                    self.car.lastGated = ground_area.gateIndex

    def BeginContact(self, contact):
        self.handle_contact(contact, True)

    def EndContact(self, contact):
        self.handle_contact(contact, False)

    def Step(self, settings):
        self.car.update(self.pressed_keys, settings.hz)

        super(TopDownCarFramework, self).Step(settings)

        tractions = [tyre.current_traction for tyre in self.car.tyres]
        self.Print('Current tractions: %s' % tractions)

        #
        self.Print("Car Gate %d" % self.car.lastGated)


if __name__ == '__main__':
    main(TopDownCarFramework)
