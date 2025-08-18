from rl.utils.bezier_gait_planner import BezierGaitPlanner
import numpy as np

class DemoGaitStraight:
    def __init__(self):
        self.num_channels = 16
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
        #suctions:
        self.flat_line([12],0,295,0)
        self.sine([13,14,15], 0, 50, 0, 5, -np.pi/2, np.pi, 20)
        self.flat_line([13,14,15], 5, 290, 100)

        self.sine([12], 0, 50, 295, 10, -np.pi/2, np.pi, 20)
        self.sine([13], 100, -50, 295, 10, -np.pi/2, np.pi, 20)
        self.flat_line([14,15],295,10,100)
        self.flat_line([13],305,290,0)
        self.flat_line([12,14,15], 305, 290, 100)

        self.sine([13], 0, 50, 595, 10, -np.pi/2, np.pi, 20)
        self.sine([14], 100, -50, 595, 10, -np.pi/2, np.pi, 20)
        self.flat_line([12,15],595,10,100)
        self.flat_line([14],605,290,0)
        self.flat_line([12,13,15], 605, 290, 100)

        self.sine([14], 0, 50, 895, 10, -np.pi/2, np.pi, 20)
        self.sine([15], 100, -50, 895, 10, -np.pi/2, np.pi, 20)
        self.flat_line([12,13],895,10,100)
        self.flat_line([15],905,290,0)
        self.flat_line([12,13,14], 905, 290, 100)        

        self.sine([15], 0, 50, 1195, 10, -np.pi/2, np.pi, 20)
        self.sine([12], 100, -50,1195, 10, -np.pi/2, np.pi, 20)
        self.flat_line([13,14],1195,10,100)
        self.flat_line([12],1205,195,0)
        self.flat_line([13,14,15], 1205, 195, 100)   
        """
        #start
        self.single_point([0,1,2,3,4,5,6,7,8,9,10,11], 0, 10)

        #lift leg 1
        self.flat_line([0,1,2,6,7,8], 0, 100, 10)
        self.sine([3,4,5,9,10,11], 10, 195, 0, 100, -np.pi/2, np.pi, 20)
        """

    def demo_pressures(self, step, mode=0):
        gait_cycle = step // self.gait_length
        rel_step = step % self.gait_length
        flat_d_pressures = self.planner.get_gradients_at_time(rel_step, self.dt)
        d_pressures = flat_d_pressures.reshape(4, 3)
        return d_pressures
    
    def demo_suction(self, step, mode=0):
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
