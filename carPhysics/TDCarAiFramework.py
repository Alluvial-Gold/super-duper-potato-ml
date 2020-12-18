
import numpy as np

from framework import (Framework, Keys, main)
from Box2D import *
import math
from TDCarAi import TDCarAi


class RayCastClosestCallback(b2RayCastCallback):
    # TODO move somewhere else later?
    """This callback finds the closest hit. Copied from
     https://github.com/openai/box2d-py/blob/master/examples/raycast.py"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        '''

        # Filter using car_collision_filter values - TODO neaten this
        if hasattr(fixture, 'filterData'):
            if fixture.filterData.categoryBits == 0x003:
                return -1

        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)

        return fraction


class TDCarAiFramework(Framework):
    name = "Top Down Car (AI)"
    description = "Top down car with AI"

    def __init__(self):
        super(TDCarAiFramework, self).__init__()

        # Top-down: no gravity
        self.world.gravity = (0, 0)

        num_cars = 1
        self.cars = [TDCarAi(self.world, position=(10*i, 0)) for i in range(num_cars)]

        self.wall_body = self.world.CreateStaticBody(position=(0, 0))
        self.set_up_walls()

        # TODO add reward gates

    def create_wall_segment(self, points):
        for p1, p2 in zip(points, points[1:]):
            edge = b2EdgeShape(vertices=[p1, p2])
            self.wall_body.CreateFixture(b2FixtureDef(shape=edge))

    def set_up_walls(self):
        # TODO add tracks from file
        outsideWallPoints = ((-50, 50), (-50, -50), (50, -50), (50, 50), (-50, 50))
        self.create_wall_segment(outsideWallPoints)

    def handle_contact(self, contact, began):
        # TODO add reward gate mechanics

        # A contact happened -- see if a wheel hit a
        # ground area
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a, body_b = fixture_a.body, fixture_b.body
        ud_a, ud_b = body_a.userData, body_b.userData
        if not ud_a or not ud_b:
            return

        '''tyre = None
        ground_area = None
        for ud in (ud_a, ud_b):
            obj = ud['obj']
            if isinstance(obj, TDTyreAi):
                tyre = obj
            elif isinstance(obj, TDGroundArea):
                ground_area = obj

        if ground_area is not None and tyre is not None:
            if began:
                tyre.add_ground_area(ground_area)
            else:
                tyre.remove_ground_area(ground_area)


                # Update gate, if required
                if tyre.car.lastGated == (ground_area.gateIndex - 1):
                    tyre.car.lastGated = ground_area.gateIndex'''

    def BeginContact(self, contact):
        self.handle_contact(contact, True)

    def EndContact(self, contact):
        self.handle_contact(contact, False)

    def DrawRaycastHit(self, point1, cb_point, cb_normal):
        cb_point = self.renderer.to_screen(cb_point)
        head = b2Vec2(cb_point) + 0.5 * cb_normal

        p1_color = b2Color(0.4, 0.9, 0.4)
        s1_color = b2Color(0.8, 0.8, 0.8)
        s2_color = b2Color(0.9, 0.9, 0.4)

        cb_normal = self.renderer.to_screen(cb_normal)
        self.renderer.DrawPoint(cb_point, 5.0, p1_color)
        self.renderer.DrawSegment(point1, cb_point, s1_color)
        self.renderer.DrawSegment(cb_point, head, s2_color)

    def DoRaycast(self, car, angle, length=50):
        callback = RayCastClosestCallback()

        wc = car.body.worldCenter
        car_normal = car.body.GetWorldVector(b2Vec2(0, 1))
        body_angle = np.arctan2(car_normal(1), car_normal(0))
        total_angle = body_angle + np.deg2rad(angle)

        point1 = wc
        change = b2Vec2(np.cos(total_angle), np.sin(total_angle)) * length
        point2 = wc + change

        self.world.RayCast(callback, point1, point2)

        if callback.hit:
            distance = np.linalg.norm(point1 - callback.point)
            point1 = self.renderer.to_screen(point1)
            self.DrawRaycastHit(point1, callback.point, callback.normal)
        else:
            distance = length
            point1 = self.renderer.to_screen(point1)
            point2 = self.renderer.to_screen(point2)
            self.renderer.DrawSegment(point1, point2, b2Color(0.9, 0.5, 0.5))

        return distance


    def Step(self, settings):
        # TODO put the controller section in here
        desired_speed = 10
        desired_angle = math.radians(40)

        for car in self.cars:
            car.update(desired_speed, desired_angle, settings.hz)
            for angle in range(-90, 91, 30):
                self.DoRaycast(car, angle)

        super(TDCarAiFramework, self).Step(settings)

        for carNumber, car in enumerate(self.cars):
            tractions = [tyre.current_traction for tyre in car.tyres]
            self.Print('Car %d: Current tractions: %s' % (carNumber, tractions))

        #
        self.Print("Car Gate %d" % self.cars[0].lastGated)


if __name__ == '__main__':
    main(TDCarAiFramework)
