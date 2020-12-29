import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Box2D import *
import gym
import torch

from framework import (Framework, Keys, main)
from TDCarAi import TDCarAi
import mapReader
import TDCarAiEnv
import pytorch_ai

class Controller():
    """
    Somewhere to store as-yet undetermined state for the AI
    """    
    def __init__(self):
        self.car_done_count = 0
        self.speed = 0
        self.angle = 0

        self.done = False
        
        # for MLP neural network
        self.obs = []
        self.rewards = []
        self.acts = []



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
        self.start_coordinate = self.create_map()

        num_cars = 100
        self.raycast_angles = (-90, -45, -30, -20, -10, 0, 10, 20, 30, 45, 90)
        #self.raycast_angles = (-45, 0, 45)
        self.cars = [TDCarAi(self.world, position=self.start_coordinate,
                             raycast_angles=self.raycast_angles) for i in range(num_cars)]

        self.controllers = [Controller() for i in range(num_cars)]

        # OpenAI Gym Stuff
        # See https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
        self.env = TDCarAiEnv.TDCarEnv()
        # Get size of observation and action space
        obs_dim = self.env.observation_space.shape[0]
        #acts_dim = self.env.action_space.shape[0]
        acts_dim = self.env.action_space.n

        print(f"Observation Space Dim:{obs_dim}")
        print(f"Action Space Dim:{acts_dim}")

        self.epoch = 0

        # Do this many simulations before updating parameters
        self.batch_size = 400
        self.batch_counter = 0

        hidden_sizes= [32]
        lr = 1e-2

        # Make core of the policy network
        logits_net = pytorch_ai.mlp(sizes=[obs_dim] + hidden_sizes + [acts_dim])
        self.logits_net = logits_net
        print(logits_net)
        print(dir(logits_net))
        print(list(logits_net.parameters()))

        # Make function to compute action distribution
        def get_policy(obs):
            logits = logits_net(obs)
            return torch.distributions.categorical.Categorical(logits=logits)

        # Make action selection function (outputs actions sampled from policy)
        def get_action(obs):
            return get_policy(obs).sample().item()
        self.get_action = get_action

        # Make loss function whose gradient, for the right data, is policy gradient
        def compute_loss(obs, act, weights):
            logp = get_policy(obs).log_prob(act)
            return -(logp*weights).mean()
        self.compute_loss = compute_loss

        # Make optimizer
        self.optimizer = torch.optim.Adam(logits_net.parameters(), lr=lr)

        # Prepare storage values
        self.batch_obs = []
        self.batch_acts = []
        self.batch_weights = []

        # Figure control
        self.plot = True
        self.plot_update_counter = 25
        self.plot_update_counter_reset = self.plot_update_counter
        if self.plot:
            figsize = (3,3)
            fig_x = 10
            fig_y_offset = 200
            fig_size_y = figsize[1] * 100

            # Make figure to show reward over time
            fig_index = 0
            self.fig_reward, self.ax_reward = plt.subplots(figsize=figsize)
            self.line_reward, = self.ax_reward.plot([], [])  # Line is the first return
            self.ax_reward.set_xlabel("Timestep")
            self.ax_reward.set_ylabel("Reward")
            import pygame_framework
            self.canvases.append(pygame_framework.FigureCanvas(self.fig_reward, (fig_x, fig_y_offset + fig_size_y*fig_index)))

            # Make figure to show observations
            fig_index = 1
            self.fig_obs, self.ax_obs = plt.subplots(figsize=figsize)
            self.obs_lines = []
            for obs_index in range(obs_dim):
                self.obs_lines.append(self.ax_obs.plot([], [], label=str(obs_index))[0])
            self.ax_obs.legend()
            self.ax_obs.set_title("Observations")
            self.ax_obs.set_xlabel("Timestep")
            self.canvases.append(pygame_framework.FigureCanvas(self.fig_obs, (fig_x, fig_y_offset + fig_size_y*fig_index)))


    def create_wall_segment(self, points):
        for p1, p2 in zip(points, points[1:]):
            edge = b2EdgeShape(vertices=[p1, p2])
            self.wall_body.CreateFixture(b2FixtureDef(shape=edge))

    def create_map(self):
        filename = "test.svg"
        self.num_gates = 18
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
                    car.last_gated_time = 0

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
            controller.obs.append(observations)

            reward = car.lastGated - car.last_gated_time/(car.life_time+1)  # Remove divide by zero - and it's close enough
            #reward = -(self.num_gates - car.lastGated)
            if len(controller.rewards) > 0:
                reward += controller.rewards[-1]
            controller.rewards.append(reward)
            #self.Print(f"Reward:{reward}")

            # TODO: Integrate `Done` calculation better
            controller.done = False
            # Do `Done` based on if the car has been still for a while
            if( abs(observations[0]) < 4):
                controller.car_done_count += 1
            else:
                controller.car_done_count -= 1

            # IF it has been too long since we last hit a gate, call it done
            if car.last_gated_time > 90:
                controller.car_done_count += 10
            
            # If we have looped around, call it done
            if car.lastGated == self.num_gates:
                controller.done = True

            max_done = 30
            if controller.car_done_count > max_done:
                controller.done = True
                controller.car_done_count = max_done  # cap it for numerical and presentation reasons.

                reward -= 1e6



            # Act
            act = self.get_action(torch.as_tensor(observations, dtype=torch.float32))
            controller.acts.append(act)

            # Convert action into required
            controller.speed = 40
            controller.angle = np.deg2rad( 10 * (1 if act else -1 ) )

            # Update car
            if not controller.done:
                car.update(controller.speed, controller.angle, self.world, settings.hz)
            else:
                # Turn off
                car.active = False

                # Store the observations, as one of the batch
                self.batch_obs += controller.obs
                self.batch_acts += controller.acts
                self.batch_weights += [sum(controller.rewards)]*len(controller.rewards)  # The weight for each logprob(a|s) is R(tau). TODO: Understand this

                # Delete car
                self.world.DestroyBody(car.body)
                for tyre in car.tyres:
                    self.world.DestroyBody(tyre.body)
                    del(tyre)
                del(car)

                # Reset
                # Make new car
                self.cars[car_index] = TDCarAi(self.world, position=self.start_coordinate,
                                               raycast_angles=self.raycast_angles)
                # Reset controller
                controller.speed = 0
                controller.angle = 0
                controller.car_done_count = 0
                controller.obs = []
                controller.rewards = []
                controller.acts = []

                self.batch_counter += 1

                # Take a single policy gradient update step
                if self.batch_counter == self.batch_size:
                    self.batch_counter = 0

                    self.optimizer.zero_grad()
                    batch_loss = self.compute_loss(obs=torch.as_tensor(self.batch_obs, dtype=torch.float32),
                                                   act=torch.as_tensor(self.batch_acts, dtype=torch.float32),
                                                   weights=torch.as_tensor(self.batch_weights, dtype=torch.float32))
                    batch_loss.backward()
                    self.optimizer.step()

                    # Clear batch parameters
                    self.batch_obs = []
                    self.batch_acts = []
                    self.batch_weights = []

                    self.epoch += 1

        self.Print(f"Epoch: {self.epoch}, Batch: {self.batch_counter}")

        car_plot_index = 0
        if self.plot:
            # Don't want to update the image each time, because slow
            self.plot_update_counter -= 1

            # Update observations plot
            if not self.controllers[car_plot_index].done:
                for obs_idx, observation in enumerate(self.controllers[car_plot_index].obs[-1]):
                    line = self.obs_lines[obs_idx]
                    # Append current data
                    data_x, data_y = line.get_data()
                    data_x = np.append(data_x, len(data_x))
                    data_y = np.append(data_y, observation)
                    line.set_data(data_x, data_y)
                # Update graph view limits
                self.ax_obs.relim()
                self.ax_obs.autoscale_view()
                if self.plot_update_counter == 0:
                    self.canvases[1].requires_refresh = True

                # Plot the reward plot
                line = self.line_reward
                # Append current data
                data_x, data_y = line.get_data()
                data_x = np.append(data_x, len(data_x))
                data_y = np.append(data_y, self.controllers[car_plot_index].rewards[-1])
                line.set_data(data_x, data_y)
                # Update graph view limits
                self.ax_reward.relim()
                self.ax_reward.autoscale_view()
                if self.plot_update_counter == 0:
                    self.canvases[0].requires_refresh = True

            else:
                # Car has reset -> clear observations plot data
                for obs_idx in range(self.env.observation_space.shape[0]):
                    line = self.obs_lines[obs_idx]
                    line.set_data([], [])

                # Reset reward plot
                self.line_reward.set_data([], [])

            if self.plot_update_counter == 0:
                self.plot_update_counter = self.plot_update_counter_reset



        observations = self.cars[car_plot_index].get_observations()
        self.Print("Speed: " + f'{observations[0]:.2f}' + " Angle: " + f'{observations[1]:.2f}')

        # Draw raycast
        car = self.cars[car_plot_index]
        for idx in range(len(car.raycast_angles)):
            self.DrawCarRaycast(car, car.raycast_angles[idx], car.raycast_distances[idx])

        super(TDCarAiFramework, self).Step(settings)

        for carNumber, car in enumerate(self.cars):
            tractions = [tyre.current_traction for tyre in car.tyres]
            #self.Print('Car %d: Current tractions: %s' % (carNumber, tractions))

        #
        #self.Print("Car Gate %d" % self.cars[0].lastGated)


if __name__ == '__main__':
    main(TDCarAiFramework)
