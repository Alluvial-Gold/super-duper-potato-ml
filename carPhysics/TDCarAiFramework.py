
import numpy as np

from framework import (Framework, Keys, main)
from Box2D import *
import math
from TDCarAi import TDCarAi

class TDCarAiFramework(Framework):
    name = "Top Down Car (AI)"
    description = "Top down car with AI"

    def __init__(self):
        super(TDCarAiFramework, self).__init__()

        # Top-down: no gravity
        self.world.gravity = (0, 0)

        num_cars = 2
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

    def Step(self, settings):
        # TODO put the controller section in here
        desired_speed = 30
        desired_angle = math.radians(40)

        for car in self.cars:
            car.update(desired_speed, desired_angle, settings.hz)

        super(TDCarAiFramework, self).Step(settings)

        for carNumber, car in enumerate(self.cars):
            tractions = [tyre.current_traction for tyre in car.tyres]
            self.Print('Car %d: Current tractions: %s' % (carNumber, tractions))

        #
        self.Print("Car Gate %d" % self.cars[0].lastGated)


if __name__ == '__main__':
    main(TDCarAiFramework)
