
import cProfile
import timeit
from a_star2 import Map, AStar
import rospkg
from geometry_msgs.msg import PoseStamped

class Test:
    def __init__(self) -> None:
        rospack = rospkg.RosPack()
        self.pkgpath = rospack.get_path("lab_4_pkg")
        self.start = PoseStamped()
        self.start.pose.position.x = 0
        self.start.pose.position.y = 0
        
        self.frontier = PoseStamped()
        self.frontier.pose.position.x = 1
        self.frontier.pose.position.y = 1
        
    def run(self):
        
        self.map = Map(f'{self.pkgpath}/maps/map2')
        raw_path, dist = AStar(self.map, self.start, self.frontier).run()
        # print(raw_path, dist)
        # self.map.display(raw_path)




if __name__ == "__main__":
    t = Test()
    profiler = cProfile.Profile()
    profiler.run('t.run()')
    profiler.print_stats()
    profiler.dump_stats('astar.prof')
