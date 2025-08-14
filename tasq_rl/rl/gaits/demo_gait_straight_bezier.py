from rl.utils.bezier_gait_planner import BezierGaitPlanner
import numpy as np

class DemoGaitStraight:
    def __init__(self):
        self.num_channels = 12
        self.gait_length = 1000 #steps
        self.dt = 1/24 #s
        self.planner = BezierGaitPlanner(self.num_channels)
    # Define gait curve for channel 0
    def sine(self,channels,start_amplitude, amplitude, start_step, length, period_start, period, num_points):
        if period_start != 0:
            start_amplitude += -amplitude*np.sin(period_start)
        for channel in channels:
            for i in range(num_points):
                self.planner.add_point(channel, start_step*self.dt + (i+1)*length*self.dt/num_points, start_amplitude + amplitude*np.sin(period_start+(i+1)*period/num_points))

    def single_point(self,channels, step, pressure):
        for channel in channels:
            self.planner.add_point(channel, step*self.dt, pressure)

    def flat_line(self, channels, start_step, length, pressure):
        for step in range(int(start_step)+1,int(start_step+length)):
            for channel in channels:
                self.planner.add_point(channel, step*self.dt, pressure)

    def gait(self):
        #start
        self.single_point([0,1,2,3,4,5,6,7,8,9,10,11], 0, 10)
        #lift leg 1
        self.flat_line([3,4,5,6,7,8,9,10,11], 0, 60, 10)
        self.single_point([1,2], 24, 390)
        self.single_point([1], 48, 10)
        self.single_point([0,2], 72, 10)
        self.single_point([7,8], 72, 390)
        self.single_point([8], 96, 10)
        self.single_point([6], 120, 390)


        """
        #start
        self.single_point([0,1,2,3,4,5,6,7,8,9,10,11], 0, 10)
        #raise body
        self.sine([0,1,2,3,4,5,6,7,8,9,10,11], 10, 190/2, 0, 24, -np.pi/2, np.pi,20)
        #tilt body for next step
        self.sine([0,1,2,6,7,8], 200, -190/2, 24, 24, -np.pi/2, np.pi,20)
        self.sine([3,4,5,9,10,11], 200, 190/2, 24, 24, -np.pi/2, np.pi,20)
        
        self.flat_line([3,4,5,9,10,11], 48, 48, 390)
        self.flat_line([6,7,8], 48, 48, 10)
        #lift leg 1
        self.flat_line([1,2], 48, 10, 10)
        self.flat_line([2], 58, 8, 10)
        #move leg 1
        self.sine([0], 10, 380/2, 50, 24, -np.pi/2, 2*np.pi,100)
        self.flat_line([0], 74, 22, 10)
        self.sine([1], 10, 380/2, 58, 24, -np.pi/2, 2*np.pi,100)
        self.flat_line([1], 82, 14, 10)
        self.sine([2], 10, 380/2, 66, 12, -np.pi/2, np.pi,100)
        self.flat_line([2], 78, 18, 390)
        #place leg 1 & tilt body for next step
        self.sine([6,7,8], 10, 380/2, 96, 24, -np.pi/2, np.pi,20)
        self.sine([2], 390, -380/2, 96, 24, -np.pi/2, np.pi,100)
        self.flat_line([3,4,5], 96, 24, 390)
        self.flat_line([0,1], 96, 94, 10)
        self.flat_line([2], 120, 70, 10)
        self.sine([9,10,11], 390, -380/2, 96, 24, -np.pi/2, np.pi,100)
        #lift leg 3
        self.sine([6,7,8], 390, -380/2, 120, 24, -np.pi/2, np.pi,20)
        self.sine([9,10,11], 10, 380/2, 120, 24, -np.pi/2, np.pi,20)
        self.flat_line([9,10,11], 144, 46, 390)
        #move leg 3      
        self.sine([8], 10, 380/2, 144, 24, -np.pi/2, 2*np.pi,100)
        self.flat_line([8], 168, 22, 10)
        self.sine([7], 10, 380/2, 152, 12, -np.pi/2, np.pi,100)
        self.flat_line([7], 164, 26, 390)
        self.sine([6], 10, 380/2, 160, 12, -np.pi/2, np.pi,100)
        self.flat_line([6], 172, 18, 390)
        #place leg 3
        self.sine([0,1,2], 10, 380/2, 190, 24, -np.pi/2, np.pi,20)
        self.sine([6,7], 390, -380/2, 190, 24, -np.pi/2, np.pi,100)
        self.flat_line([3,4,5], 190, 24, 390)
        self.sine([9,10,11], 390, -380/2, 190, 24, -np.pi/2, np.pi,100)
        """
    def demo_pressures(self, step):
        gait_cycle = step // self.gait_length
        rel_step = step % self.gait_length
        flat_d_pressures = self.planner.get_gradients_at_time(rel_step, self.dt)
        d_pressures = flat_d_pressures.reshape(4, 3)
        return d_pressures
    
    def demo_suction(self, step):
        gait_cycle = step // self.gait_length
        rel_step = step % self.gait_length
        suction_action = np.array([False,False,False,False])
        
        if rel_step < 5/self.dt:
            suction_action[0] = False
            suction_action[1] = True
            suction_action[2] = True
            suction_action[3] = True
        
        elif rel_step < 10/self.dt:
            suction_action[0] = True
            suction_action[1] = True
            suction_action[2] = False
            suction_action[3] = True

        elif rel_step < 15/self.dt:
            suction_action[0] = True
            suction_action[1] = False
            suction_action[2] = True
            suction_action[3] = True

        elif rel_step < 16/self.dt:
            suction_action[0] = True
            suction_action[1] = True
            suction_action[2] = True
            suction_action[3] = False

        else:
            suction_action[0] = True
            suction_action[1] = True
            suction_action[2] = True
            suction_action[3] = True
    
        return suction_action

    def plot(self):
        # Plot the full curves
        self.planner.plot(duration=12.0, dt=0.005)
