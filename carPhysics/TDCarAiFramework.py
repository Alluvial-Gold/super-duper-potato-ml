import math

import numpy as np
from Box2D import *

from framework import (Framework, Keys, main)
from TDCarAi import TDCarAi
import mapReader

class Controller():
    """
    Somewhere to store as-yet undetermined state for the AI
    """    
    def __init__(self):
        self.car_done_count = 0
        self.speed = 0
        self.angle = 0
        
        self.k = 1/(2000*np.random.rand(1)[0])
        print(self.k)

class TDGate(object):
    """
    An area on the ground that the car can run over
    """

    def __init__(self, gateIndex):
        self.gateIndex = gateIndex

class TDCarAiFramework(Framework):
    name = "Top Down Car (AI)"
    description = "Top down car with AI"

    def __init__(self):
        super(TDCarAiFramework, self).__init__()

        # Top-down: no gravity
        self.world.gravity = (0, 0)

        # Create map
        self.wall_body = self.world.CreateStaticBody(position=(0, 0))
        start_coordinate = self.create_map()

        num_cars = 100
        raycast_angles = (-90, -45, -30, -20, -10, 0, 10, 20, 30, 45, 90)
        raycast_angles = (-45, 45)
        self.cars = [TDCarAi(self.world, position=start_coordinate,
                             raycast_angles=raycast_angles) for i in range(num_cars)]

        self.controllers = [Controller() for i in range(num_cars)]

    def create_wall_segment(self, points):
        for p1, p2 in zip(points, points[1:]):
            edge = b2EdgeShape(vertices=[p1, p2])
            self.wall_body.CreateFixture(b2FixtureDef(shape=edge))

    def create_map(self):
        filename = "test.svg"
        all_wall_points, all_gate_data, start_coordinate = mapReader.read_svg_map(filename)

        # Create walls
        for wall_section in all_wall_points:
            # Convert the wall section into a bunch of tuples
            wall_points = []
            for row_idx in range(wall_section.shape[0]):
                wall_points.append(tuple(wall_section[row_idx, :]))

            self.create_wall_segment(wall_points)

        # Create gates
        for gate in all_gate_data:
            collisionObject = TDGate(gate["number"])

            center_x = gate["x"] + gate["width"]/2
            center_y = gate["y"] + gate["height"]/2
            rotation = 0

            car_collision_filter = b2Filter(
                groupIndex=1,
                categoryBits=0x003,
                maskBits=0xFFFF
            )

            segment = self.world.CreateStaticBody(userData={"obj": collisionObject})
            fixture = segment.CreatePolygonFixture(
                box=(gate["width"]/2, gate["height"]/2, (center_x , center_y), rotation)
            )
            fixture.sensor = True
            fixture.filterData = car_collision_filter

        # Return start point
        return start_coordinate

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

        car = None
        gate = None
        for ud in (ud_a, ud_b):
            obj = ud['obj']
            if isinstance(obj, TDCarAi):
                car = obj
            elif isinstance(obj, TDGate):
                gate = obj

        if gate is not None and car is not None:
            if began:
                pass
            else:
                # Update gate, if required
                if car.lastGated == (gate.gateIndex - 1):
                    car.lastGated = gate.gateIndex

    def BeginContact(self, contact):
        self.handle_contact(contact, True)

    def EndContact(self, contact):
        self.handle_contact(contact, False)

    def DrawCarRaycast(self, car, angle, distance):
        wc = car.body.worldCenter
        car_normal = car.body.GetWorldVector(b2Vec2(0, 1))
        body_angle = np.arctan2(car_normal(1), car_normal(0))
        total_angle = body_angle + np.deg2rad(angle)

        point1 = wc
        change = b2Vec2(np.cos(total_angle), np.sin(total_angle)) * distance
        point2 = wc + change

        point1 = self.renderer.to_screen(point1)
        point2 = self.renderer.to_screen(point2)

        if distance < car.max_raycast_dist:
            self.renderer.DrawPoint(point2, 5.0, b2Color(0.4, 0.9, 0.4))
            self.renderer.DrawSegment(point1, point2, b2Color(0.8, 0.8, 0.8))
        else:
            self.renderer.DrawSegment(point1, point2, b2Color(0.9, 0.5, 0.5))

    def Step(self, settings):

        observations = self.cars[0].get_observations()
        self.Print("Speed: " + f'{observations[0]:.2f}' + " Angle: " + f'{observations[1]:.2f}')

        # Draw raycast
        for car in self.cars[:2]:
            for idx in range(len(car.raycast_angles)):
                self.DrawCarRaycast(car, car.raycast_angles[idx], car.raycast_distances[idx])



        # TODO put the controller section in here

        """
        The AI controller is designed to be roughly similar to an OpenAI Gym
        This, in the simple case, takes the form:

        ```
        while True:
            world.simulate()
            action = AiMagic(observations)
            observation, reward, done, info = env.step(action)

            if done:
                break
        ```

        As this is being called from inside the Step, it's all a bit inside-out.
        """
        for car_index, (car, controller) in enumerate(zip(self.cars, self.controllers)):
            observations = car.get_observations()
            reward = car.lastGated

            # TODO: Integrate `Done` calculation better
            # Do `Done` based on if the car has been still for a while
            if( abs(observations[0]) < 4):
                controller.car_done_count += 1
            else:
                controller.car_done_count = 0
            max_done = 100
            if controller.car_done_count > max_done:
                done = True
                controller.car_done_count = max_done  # cap it for numerical and presentation reasons.
            else:
                done = False


            
            # The `action` consists of a speed and a direction
            #   The calculation of this should probably be more in the controller
            # Just have a constant speed
            controller.speed = 50

            # Adjust angle based on the observations
            raycast_distances = observations[2:]
            port_ray = raycast_distances[-1]
            starboard_ray = raycast_distances[0]
            difference = port_ray - starboard_ray
            controller.angle += np.deg2rad( controller.k * difference )
            #controller.angle = 0
            # Reset controller angle if they are about equal
            if abs(difference) < 3:
                controller.angle = 0

            #self.Print(f"Starboard:{starboard_ray:.2f}, Port:{port_ray:.2f}")
            #self.Print("Hence condition:%s for direction %.2f" % (port_ray < starboard_ray, np.rad2deg(controller.angle)))
            #self.Print(f"Car {car_index}: Done Count:{controller.car_done_count}: Done:{done}")

            # Update car
            if not done:
                car.update(controller.speed, controller.angle, self.world, settings.hz)
            else:
                car.active = False




        super(TDCarAiFramework, self).Step(settings)

        for carNumber, car in enumerate(self.cars):
            tractions = [tyre.current_traction for tyre in car.tyres]
            #self.Print('Car %d: Current tractions: %s' % (carNumber, tractions))

        #
        #self.Print("Car Gate %d" % self.cars[0].lastGated)


if __name__ == '__main__':
    main(TDCarAiFramework)
