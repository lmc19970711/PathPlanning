
"""
Grid based Dijkstra planning
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math

show_animation = True

# 函数定义 首先Dijkstra类
class Dijkstra:

    def __init__(self, ox, oy, resolution, robot_radius):
        """
        Initialize map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None  #障碍物地图, 遍历时会检测是否是障碍物的地图, 如果是则跳过该点, 代价为无穷大

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    # 定义搜索区域节点(Node)的类
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node
        # 每个Node都包含坐标x和y,代价cost和其父节点(上一个遍历的点)的序号. 在遍历时需要计算每个坐标的index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    # 规划函数 输入起始点和目标点坐标,最终输出的结果是路径包含的点的集合rx和ry
    # 首先根据起始点和目标点的输入坐标来定义Node类, 同时初始化open_set 和 closed_set 将起始点存放在open_set中
    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        # 在while循环中从起始点开始遍历所有点寻找最优路径, 直到找到目标点后循环终止，
        # 在遍历时会优先选择open_set待遍历点中存储cost最小的点作为curret进行尝试
        while 1:
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # 首先判断current是否为目标点, 如果是则遍历结束, break掉这个循环
            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            # 否则的话，将该current点从open_set中删除，并存储在closed_set中
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # 并使用前面提到的运动学模型motion来遍历该点周边的点
            # 在遍历时需要依次判断这个待遍历的点是否已经被遍历了(是否已经在closed_set中)
            # 以及验证该点是否是正确的点
            # 如果该点还未遍历,并且是在地图中的合理点
            # 那么继续判定该点是否在open_set中
            # 如果不在 会存储到open_set中作为候选遍历点
            # 如果在 则判断该点的cost是否比当前点小
            # 如果小的话则替换该点作为当前点
            # 并重复上述遍历过程直到到达goal

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        # 最后 通过将目标点和closed_set传入calc_final_path函数来产生最后的路径并结束while循环
        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    '''
    将地图(假设是方形)均匀的网格化，每一个网格代表一个点，同时每一个点有一个序列号index变量来记录点的位置
    同时，需要定义地图的边界min_x, min_y 和 max_x, max_y 用来计算遍历的点的具体位置
    minp : minx或者min_y
    index : 需要计算的节点的序列号
    resolution : grid_size 网格的尺寸
    index * resolution : 得到x或者y方向的长度，这个长度加在边界点min_x 或者min_y上即可得到准确的坐标
    '''
    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    # 验证节点的verify_node函数:判断节点的坐标在[min_x, max_x]和[min_y, max_y]的范围内,同时验证该节点不在障碍层
    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    # 计算障碍层地图的函数calc_obstacle_map 其中障碍物的点是在main函数中定义的
    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost , 每个数组中的三个元素分别表示向x,y方向前进的距离以及前进的代价
        motion = [[1, 0, 1], # 右
                  [0, 1, 1], # 上
                  [-1, 0, 1], # 左
                  [0, -1, 1], # 下
                  [-1, -1, math.sqrt(2)], # 左下
                  [-1, 1, math.sqrt(2)], # 左上
                  [1, -1, math.sqrt(2)], # 右下
                  [1, 1, math.sqrt(2)]] # 右上

        return motion
    # 即在贪婪遍历时, 会遍历周边8个节点的cost以选出最小的cost

# main函数是定义起点和目标点 设置障碍物的位置 调用类以及里面的函数进行规划运算，并动态展示出来的运算结果
def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m] 起始位置x坐标
    sy = -5.0  # [m] 起始位置y坐标
    gx = 50.0  # [m] 目标点x坐标
    gy = 50.0  # [m] 目标点y坐标
    grid_size = 2.0  # [m] 网格的尺寸
    robot_radius = 1.0  # [m] 机器人尺寸

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()