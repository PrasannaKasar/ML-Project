# modules/path_plan.py
import numpy as np
import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y)
        self.parent = parent
        self.g = 0  # Cost from start
        self.h = 0  # Heuristic to goal
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

class AStarPlanner:
    def __init__(self, obstacle_threshold=0.2):
        """
        Simple A* planner using depth map as obstacle map
        :param obstacle_threshold: Depth value above which we consider obstacle
        """
        self.obstacle_threshold = obstacle_threshold

    def plan(self, depth_map, start, goal):
        """
        Plan path from start to goal avoiding obstacles
        :param depth_map: normalized depth map (0-1)
        :param start: (x, y)
        :param goal: (x, y)
        :return: list of (x, y) waypoints or empty if no path
        """
        start_node = Node(start)
        goal_node = Node(goal)

        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()

        max_y, max_x = depth_map.shape

        while open_list:
            current = heapq.heappop(open_list)
            if current.position == goal_node.position:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            closed_set.add(current.position)

            # 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current.position[0] + dx, current.position[1] + dy
                    if 0 <= nx < max_x and 0 <= ny < max_y:
                        if depth_map[ny, nx] < self.obstacle_threshold:
                            neighbor_pos = (nx, ny)
                            if neighbor_pos in closed_set:
                                continue
                            neighbor = Node(neighbor_pos, current)
                            neighbor.g = current.g + 1
                            neighbor.h = (nx - goal_node.position[0])**2 + (ny - goal_node.position[1])**2
                            neighbor.f = neighbor.g + neighbor.h
                            heapq.heappush(open_list, neighbor)
        # No path found
        return []

    def draw_path(self, frame, path):
        """
        Optional visualization of path on frame
        """
        import cv2
        for i in range(1, len(path)):
            cv2.line(frame, path[i-1], path[i], (0, 0, 255), 2)
        return frame
