from rl.utils.gait_planning.bezier_gait_planner import BezierGaitPlanner
import numpy as np

class DemoGaitStraight:
    def __init__(self):
        self.num_channels = 16
        self.speed_increase = 2  # Speed increase factor
        self.gait_length = 1400/self.speed_increase #steps
        self.dt = 1/120 #s
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

    def flat_line(self, channels, start_step, length, pressure, no_points=50):
        for step in range(no_points):
            for channel in channels:
                self.planner.add_point(channel, start_step*self.dt + (length/no_points)*(step+1)*self.dt, pressure)

    def gait(self):
        #"""
        #suctions:
        self.flat_line([12],0,295,0)
        #self.sine([13,14,15], 0, 50, 0, 5, -np.pi/2, np.pi, 20)
        self.flat_line([13,14,15], 0, 295, 100)

        self.sine([12], 0, 50, 295, 10, -np.pi/2, np.pi, 40)
        self.sine([13], 100, -50, 295, 10, -np.pi/2, np.pi, 40)
        self.flat_line([14,15],295,10,100)
        self.flat_line([13],305,290,0)
        self.flat_line([12,14,15], 305, 290, 100)

        self.sine([13], 0, 50, 595, 10, -np.pi/2, np.pi, 40)
        self.sine([15], 100, -50, 595, 10, -np.pi/2, np.pi, 40)
        self.flat_line([12,14],595,10,100)
        self.flat_line([15],605,290,0)
        self.flat_line([12,13,14], 605, 290, 100)

        self.sine([15], 0, 50, 895, 10, -np.pi/2, np.pi, 40)
        self.sine([14], 100, -50, 895, 10, -np.pi/2, np.pi, 40)
        self.flat_line([12,13],895,10,100)
        self.flat_line([14],905,290,0)
        self.flat_line([12,13,15], 905, 290, 100)        

        self.sine([14], 0, 50, 1195, 10, -np.pi/2, np.pi, 40)
        self.sine([12], 100, -50,1195, 10, -np.pi/2, np.pi, 40)
        self.flat_line([13,15],1195,10,100)
        self.flat_line([12],1205,195,0)
        self.flat_line([13,14,15], 1205, 195, 100)
        #"""
        #pressures:   
        
        #t=0 [10,10,10,  10,10,10,  10,10,10,  10,10,10]
        self.single_point([0,1,2,3,4,5,6,7,8,9,10,11], 0, 10)
        self.flat_line([0,1,2,6,7,8], 0, 100, 10)
        self.sine([3,4,5,9,10,11], 10, 195, 0, 100, -np.pi/2, np.pi, 20)

        #t=100 [10,10,10,  10,10,10,  10,10,10,  10,10,10]
        self.flat_line([3,4,5,9,10,11], 100, 100, 400)
        self.flat_line([0,6,7,8], 100, 100, 10)
        self.sine([1], 10, 40, 100, 100, -np.pi/2, np.pi, 20)
        self.sine([2], 10, 150, 100, 100, -np.pi/2, np.pi, 20)

        #t=200 [10,90,310,  400,400,400,  10,10,10,  400,400,400]
        self.sine([3,4,5,9,10,11], 400, -192.5, 200, 100, -np.pi/2, np.pi, 20)
        self.sine([6,7,8], 10, 192.5, 200, 100, -np.pi/2, np.pi, 20)
        self.sine([0], 10, 192.5, 200, 100, -np.pi/2, np.pi, 20)
        self.sine([1], 90, 30, 200, 100, -np.pi/2, np.pi, 20)
        self.sine([2], 310, -150, 200, 100, -np.pi/2, np.pi, 20)

        #t=300 [395,150,10,  15,15,15,  395,395,395,  15,15,15]
        self.flat_line([3,4,5,9,10,11], 300, 100, 15)
        self.flat_line([0,6,7,8], 300, 100, 395)
        self.sine([1], 150, 245/2, 300, 100, -np.pi/2, np.pi, 20)
        self.sine([2], 10, 385/2, 300, 100, -np.pi/2, np.pi, 20)

        #t=400 [395,395,395,  15,15,15,  395,395,395,  15,15,15]
        self.flat_line([3,9,10,11], 400, 100, 15)
        self.flat_line([0,1,2,6,7,8], 400, 100, 395)
        self.sine([4], 10, 305/2, 400, 100, -np.pi/2, np.pi, 20)
        self.sine([5], 10, 85/2, 400, 100, -np.pi/2, np.pi, 20)

        #t=500 [395,395,395,  15,315,95,  395,395,395,  15,15,15]
        self.sine([0,1,2,6,7,8], 395, -375/2, 500, 100, -np.pi/2, np.pi, 20)
        self.sine([9,10,11], 15, 375/2, 500, 100, -np.pi/2, np.pi, 20)
        self.sine([3], 15, 375/2, 500, 100, -np.pi/2, np.pi, 20)
        self.sine([4], 315, -295/2, 500, 100, -np.pi/2, np.pi, 20)
        self.sine([5], 95, 60/2, 500, 100, -np.pi/2, np.pi, 20)

        #t=600 [20,20,20  390,20,155,  20,20,20,  390,390,390]
        self.sine([0,1,2,6,7,8], 20, 370/2, 600, 100, -np.pi/2, np.pi, 20)
        self.sine([9,10,11], 390, -370/2, 600, 100, -np.pi/2, np.pi, 20)
        self.sine([3], 390, -370/2, 600, 100, -np.pi/2, np.pi, 20)
        self.flat_line([4], 600, 100, 20)
        self.sine([5], 155, -135/2, 600, 100, -np.pi/2, np.pi, 20)

        #t=700 [390,390,390  20,20,20,  390,390,390,  20,20,20]
        self.flat_line([3,4,5,10], 700, 100, 20)
        self.flat_line([0,1,2,6,7,8], 700, 100, 390)
        self.sine([9], 20, 300/2, 700, 100, -np.pi/2, np.pi, 20)
        self.sine([11], 20, 215/2, 700, 100, -np.pi/2, np.pi, 20)

        #t=800 [390,390,390  20,20,20,  390,390,390,  320,20,235]
        self.sine([0,1,2,6,7,8], 390, -370/2, 800, 100, -np.pi/2, np.pi, 20)
        self.sine([3,4,5], 20, 370/2, 800, 100, -np.pi/2, np.pi, 20)
        self.sine([9], 320, -300/2, 800, 100, -np.pi/2, np.pi, 20)
        self.sine([10], 20, 370/2, 800, 100, -np.pi/2, np.pi, 20)
        self.sine([11], 235, -130/2, 800, 100, -np.pi/2, np.pi, 20)

        #t=900 [20,20,20  390,390,390,  20,20,20,  20,390,105]
        self.sine([0,1,2,6,7,8], 20, -8/2, 900, 100, -np.pi/2, np.pi, 20)
        self.sine([3,4,5], 390, 8/2, 900, 100, -np.pi/2, np.pi, 20)
        self.sine([9], 20, 378/2, 900, 100, -np.pi/2, np.pi, 20)
        self.sine([10], 390, 8/2, 900, 100, -np.pi/2, np.pi, 20)
        self.sine([11], 105, 293/2, 900, 100, -np.pi/2, np.pi, 20)

        #t=1000 [12,12,12  398,398,398,  12,12,12,  398,398,398]
        self.flat_line([3,4,5,9,10,11], 1000, 100, 398)
        self.flat_line([0,1,2,8], 1000, 100, 12)
        self.sine([6], 12, 300/2, 1000, 100, -np.pi/2, np.pi, 20)
        self.sine([7], 12, 220/2, 1000, 100, -np.pi/2, np.pi, 20)

        #t=1100 [12,12,12  398,398,398,  312,232,12,  398,398,398]
        self.sine([0,1,2,8], 12, 386/2, 1100, 100, -np.pi/2, np.pi, 20)
        self.sine([3,4,5,9,10,11], 398, -386/2, 1100, 100, -np.pi/2, np.pi, 20)
        self.sine([6], 312, -300/2, 1100, 100, -np.pi/2, np.pi, 20)
        self.sine([7], 232, -120/2, 1100, 100, -np.pi/2, np.pi, 20)

        #t=1200 [398,398,398,  12,12,12,  12,112,398,  12,12,12]
        self.sine([0,1,2,8], 398, -384/2, 1200, 100, -np.pi/2, np.pi, 20)
        self.sine([3,4,5,9,10,11], 12, 378/2, 1200, 100, -np.pi/2, np.pi, 20)
        self.sine([6], 12, 2/2, 1200, 100, -np.pi/2, np.pi, 20)
        self.sine([7], 112, -98/2, 1200, 100, -np.pi/2, np.pi, 20)

        #t=1300 [14,14,14,  390,390,390,  14,14,14, 390,390,390]
        self.sine([0,1,2,6,7,8], 14, -4/2, 1300, 100, -np.pi/2, np.pi, 20)
        self.sine([3,4,5,9,10,11], 390, -380/2, 1300, 100, -np.pi/2, np.pi, 20)

        #t=1400/t=0 [10,10,10,  10,10,10,  10,10,10,  10,10,10]
        #"""
    
    def demo_pressures(self, step, mode=0):
        gait_cycle = step // self.gait_length
        rel_step = step % self.gait_length
        flat_d_pressures = self.planner.get_gradients_at_time(rel_step, self.speed_increase*self.dt)[0:12]
        d_pressures = flat_d_pressures.reshape(4, 3)
        return d_pressures

    def demo_suction(self, step, mode=0):
        gait_cycle = step // self.gait_length
        rel_step = step % self.gait_length
        suction_states = self.planner.evaluate(rel_step * self.speed_increase*self.dt)[12:16]
        return suction_states.astype(np.float32)

    def plot(self):
        # Plot the full curves
        self.planner.plot(duration=12.0, dt=0.005)
