import sys
import math
import random
import time
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QSlider, QLabel, QGroupBox, QGridLayout, 
                               QCheckBox, QFrame)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF, QFont

# --- Configuration ---
GRID_W, GRID_H = 20, 15
CELL_SIZE = 35
MAX_MCTS_DEPTH = 6
EXPANSION_THRESHOLD = 2 # Progressive Widening: Visit count needed to expand a node

# Actions: (dx, dy)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]

# Terrain Types
T_NORMAL, T_OBSTACLE, T_MUD, T_BOOST = 0, 1, 2, 3

# Modern Dark Theme QSS
QSS = """
QWidget { background-color: #1a1a1a; color: #e0e0e0; font-size: 10pt; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 12px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #00ccff; }
QPushButton { background-color: #2a2a2a; border: 1px solid #444; border-radius: 4px; padding: 6px; }
QPushButton:hover { background-color: #3a3a3a; border: 1px solid #00ccff; }
QPushButton:pressed { background-color: #00ccff; color: #000; }
QSlider::groove:horizontal { border: 1px solid #444; height: 4px; background: #2a2a2a; }
QSlider::handle:horizontal { background: #00ccff; width: 12px; margin: -5px 0; border-radius: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; border-radius: 3px; }
QCheckBox::indicator:checked { background: #00ccff; border: 1px solid #00ccff; }
"""

class Environment:
    def __init__(self):
        self.start = (1, 1)
        self.goal = (GRID_W - 2, GRID_H // 2)
        self.terrain = {}
        self.generate_terrain()
        
        self.storm_pos = [GRID_W // 2, GRID_H // 4]
        self.storm_vel = [1, 1]
        self.storm_trail = []
        
        self.wind_strength = 1.0
        self.time = 0.0
        self.steps = 0
        self.best_steps = float('inf')
        self.reset()

    def generate_terrain(self):
        self.terrain.clear()
        for _ in range(15):
            x, y = random.randint(2, GRID_W-3), random.randint(1, GRID_H-2)
            if (x, y) != self.start and (x, y) != self.goal:
                self.terrain[(x, y)] = T_OBSTACLE
        for _ in range(8):
            x, y = random.randint(2, GRID_W-3), random.randint(1, GRID_H-2)
            if (x, y) not in self.terrain:
                self.terrain[(x, y)] = T_MUD
        for _ in range(6):
            x, y = random.randint(2, GRID_W-3), random.randint(1, GRID_H-2)
            if (x, y) not in self.terrain:
                self.terrain[(x, y)] = T_BOOST

    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        self.done = False
        self.storm_pos = [GRID_W // 2, GRID_H // 4]
        self.storm_vel = [1, 1]
        self.storm_trail.clear()

    def set_start(self, x, y):
        if self.terrain.get((x, y), T_NORMAL) != T_OBSTACLE and (x, y) != self.goal:
            self.start = (x, y)
            self.reset()

    def set_goal(self, x, y):
        if self.terrain.get((x, y), T_NORMAL) != T_OBSTACLE and (x, y) != tuple(self.agent_pos):
            self.goal = (x, y)
            self.done = False

    def toggle_obstacle(self, x, y):
        if (x, y) == self.start or (x, y) == self.goal: return
        if self.terrain.get((x, y)) == T_OBSTACLE:
            del self.terrain[(x, y)]
        else:
            self.terrain[(x, y)] = T_OBSTACLE
        self.done = False

    def update_time(self, dt):
        self.time += dt

    def get_wind(self, x, y):
        wx = math.sin(y * 0.4 + self.time * 0.5) * 0.6 * self.wind_strength
        wy = math.cos(x * 0.4 + self.time * 0.7) * 0.6 * self.wind_strength
        return wx, wy

    def get_terrain_modifier(self, x, y):
        t = self.terrain.get((x, y), T_NORMAL)
        if t == T_MUD: return 0.5
        if t == T_BOOST: return 1.5
        return 1.0

    def is_valid(self, x, y):
        return 0 <= x < GRID_W and 0 <= y < GRID_H and self.terrain.get((x, y)) != T_OBSTACLE

    def get_full_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self.storm_pos[0], self.storm_pos[1], self.storm_vel[0], self.storm_vel[1])

    def simulate_step(self, state, action_idx):
        ax, ay, sx, sy, svx, svy = state
        dx, dy = ACTIONS[action_idx]
        wx, wy = self.get_wind(ax, ay)
        mod = self.get_terrain_modifier(ax, ay)
        
        new_ax = max(0, min(GRID_W - 1, round(ax + (dx * mod) + wx)))
        new_ay = max(0, min(GRID_H - 1, round(ay + (dy * mod) + wy)))
        
        if self.terrain.get((new_ax, new_ay)) == T_OBSTACLE:
            new_ax, new_ay = ax, ay
            
        new_sx = sx + svx
        new_sy = sy + svy
        new_svx, new_svy = svx, svy
        if new_sx <= 0 or new_sx >= GRID_W - 1: new_svx = -svx
        if new_sy <= 0 or new_sy >= GRID_H - 1: new_svy = -svy
        new_sx = max(0, min(GRID_W - 1, new_sx))
        new_sy = max(0, min(GRID_H - 1, new_sy))
        
        done = (new_ax == self.goal[0] and new_ay == self.goal[1])
        reward = 10.0 if done else -0.1
        
        if self.terrain.get((new_ax, new_ay)) == T_MUD: reward -= 0.5
        if self.terrain.get((new_ax, new_ay)) == T_BOOST: reward += 0.2
            
        dist_to_storm = math.hypot(new_ax - new_sx, new_ay - new_sy)
        if dist_to_storm < 1.5: reward -= 5.0
            
        return (new_ax, new_ay, new_sx, new_sy, new_svx, new_svy), reward, done

    def step(self, action_idx):
        state = self.get_full_state()
        next_state, reward, done = self.simulate_step(state, action_idx)
        
        self.agent_pos = [next_state[0], next_state[1]]
        self.storm_pos = [next_state[2], next_state[3]]
        self.storm_vel = [next_state[4], next_state[5]]
        self.storm_trail.append(tuple(self.storm_pos))
        if len(self.storm_trail) > 5: self.storm_trail.pop(0)
        
        self.steps += 1
        self.done = done
        if done and self.steps < self.best_steps:
            self.best_steps = self.steps
        return next_state, reward, done

    def mock_predictor(self, state):
        ax, ay, sx, sy, _, _ = state
        gx, gy = self.goal
        
        policy = []
        for dx, dy in ACTIONS:
            tx, ty = ax + dx, ay + dy
            if not self.is_valid(tx, ty):
                policy.append(1e-6)
            else:
                dist_goal = math.sqrt((gx - tx)**2 + (gy - ty)**2)
                dist_storm = math.sqrt((sx - tx)**2 + (sy - ty)**2)
                storm_penalty = 2.0 if dist_storm < 2.0 else 0.0
                policy.append(math.exp(-(dist_goal * 0.5 - dist_storm * 0.1 + storm_penalty)))
            
        total = sum(policy)
        policy = [p / total for p in policy]
        
        noise = [random.random() for _ in range(len(policy))]
        total_noise = sum(noise)
        noise = [n / total_noise for n in noise]
        policy = [0.75 * p + 0.25 * n for p, n in zip(policy, noise)]
        
        dist = math.sqrt((gx - ax)**2 + (gy - ay)**2)
        dist_storm = math.sqrt((sx - ax)**2 + (sy - ay)**2)
        value = -dist * 0.2
        if dist_storm < 2.0: value -= 2.0
        if self.terrain.get((ax, ay)) == T_MUD: value -= 1.0
            
        return policy, value

class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.depth = depth
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_expanded = False
        
    def best_child(self, c_puct):
        best_score = -float('inf')
        best = None
        for child in self.children:
            q = child.value / (child.visits + 1e-6)
            u = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        return best

class PUCTAgent:
    def __init__(self, env):
        self.env = env
        self.c_puct = 1.5
        self.simulations = 100
        self.last_root = None
        self.last_search_time = 0.0
        self.max_depth_reached = 0
        self.chosen_q_value = 0.0
        self.visited_nodes = {} # For Visit Heatmap
        
    def expand_node(self, node):
        if node.is_expanded or node.depth >= MAX_MCTS_DEPTH:
            return
        node.is_expanded = True
        policy, _ = self.env.mock_predictor(node.state)
        
        for i, p in enumerate(policy):
            if p < 1e-5: continue
            next_state, _, _ = self.env.simulate_step(node.state, i)
            node.children.append(Node(next_state, node, i, p, node.depth + 1))

    def run_mcts(self, root_state):
        start_time = time.perf_counter()
        root = Node(root_state)
        self.expand_node(root)
        self.max_depth_reached = 0
        self.visited_nodes.clear()
        
        for _ in range(self.simulations):
            node = root
            
            # Selection & Progressive Widening
            while True:
                if not node.is_expanded:
                    if node.visits >= EXPANSION_THRESHOLD and node.depth < MAX_MCTS_DEPTH:
                        self.expand_node(node)
                    else:
                        break # Evaluate leaf
                
                if node.is_expanded and node.children:
                    node = node.best_child(self.c_puct)
                else:
                    break
                    
            self.max_depth_reached = max(self.max_depth_reached, node.depth)
            
            # Record visit for heatmap (only x, y of agent)
            state_key = (node.state[0], node.state[1])
            self.visited_nodes[state_key] = self.visited_nodes.get(state_key, 0) + 1
            
            _, value = self.env.mock_predictor(node.state)
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
                
        self.last_root = root
        self.last_search_time = (time.perf_counter() - start_time) * 1000
        
        best_visits = -1
        best_child = None
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child
                
        self.chosen_q_value = best_child.value / (best_child.visits + 1e-6) if best_child else 0.0
        return best_child.action if best_child else 0

def catmull_rom_spline(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2 * p1.x()) + (-p0.x() + p2.x()) * t + 
               (2 * p0.x() - 5 * p1.x() + 4 * p2.x() - p3.x()) * t2 + 
               (-p0.x() + 3 * p1.x() - 3 * p2.x() + p3.x()) * t3)
    y = 0.5 * ((2 * p1.y()) + (-p0.y() + p2.y()) * t + 
               (2 * p0.y() - 5 * p1.y() + 4 * p2.y() - p3.y()) * t2 + 
               (-p0.y() + 3 * p1.y() - 3 * p2.y() + p3.y()) * t3)
    return QPointF(x, y)

class SimulationCanvas(QWidget):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.setFixedSize(GRID_W * CELL_SIZE, GRID_H * CELL_SIZE)
        
        self.current_pixel_pos = self.grid_to_pixel(*env.agent_pos)
        self.target_pixel_pos = self.current_pixel_pos
        self.is_animating = False
        
        self.path_history = [tuple(env.agent_pos)]
        self.show_heatmap = False
        self.show_visit_map = False
        
        # Wind Particles
        self.particles = []
        for _ in range(250):
            self.particles.append([random.uniform(0, GRID_W), random.uniform(0, GRID_H)])
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_state)
        self.timer.start(16) 
        
        self.is_running = True
        self.do_step = False

    def grid_to_pixel(self, gx, gy):
        return QPointF(gx * CELL_SIZE + CELL_SIZE / 2, gy * CELL_SIZE + CELL_SIZE / 2)

    def update_state(self):
        self.env.update_time(0.016)

        # Update Particles
        for p in self.particles:
            wx, wy = self.env.get_wind(p[0], p[1])
            p[0] += wx * 0.15
            p[1] += wy * 0.15
            if p[0] < 0 or p[0] >= GRID_W or p[1] < 0 or p[1] >= GRID_H:
                p[0] = random.uniform(0, GRID_W)
                p[1] = random.uniform(0, GRID_H)

        if self.is_animating:
            dx = self.target_pixel_pos.x() - self.current_pixel_pos.x()
            dy = self.target_pixel_pos.y() - self.current_pixel_pos.y()
            dist = math.hypot(dx, dy)
            if dist < 2.0:
                self.current_pixel_pos = self.target_pixel_pos
                self.is_animating = False
                if self.env.done:
                    self.is_running = False
            else:
                self.current_pixel_pos = QPointF(
                    self.current_pixel_pos.x() + dx * 0.15,
                    self.current_pixel_pos.y() + dy * 0.15
                )
        else:
            if (self.is_running or self.do_step) and not self.env.done:
                self.do_step = False
                state = self.env.get_full_state()
                action = self.agent.run_mcts(state)
                next_state, reward, done = self.env.step(action)
                
                self.path_history.append((next_state[0], next_state[1]))
                self.target_pixel_pos = self.grid_to_pixel(*self.env.agent_pos)
                self.is_animating = True

        self.update()

    def draw_tree(self, painter, node, depth=0):
        if depth > 3: return
        if not node.children: return
        
        max_visits = max([c.visits for c in node.children] + [1])
        for child in node.children:
            if child.visits > 1:
                start = self.grid_to_pixel(node.state[0], node.state[1])
                end = self.grid_to_pixel(child.state[0], child.state[1])
                
                alpha = int(150 * (child.visits / max_visits) * (1 - depth * 0.2))
                width = 1 + 2 * (child.visits / max_visits)
                
                painter.setPen(QPen(QColor(0, 255, 255, alpha), width))
                painter.drawLine(start, end)
                self.draw_tree(painter, child, depth + 1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(15, 15, 15)) # Darker bg
        
        # Draw Value Heatmap
        if self.show_heatmap:
            for x in range(GRID_W):
                for y in range(GRID_H):
                    if self.env.terrain.get((x, y)) != T_OBSTACLE:
                        state = (x, y, self.env.storm_pos[0], self.env.storm_pos[1], 0, 0)
                        _, val = self.env.mock_predictor(state)
                        max_dist = math.sqrt(GRID_W**2 + GRID_H**2)
                        norm = 1.0 - (abs(val) / (max_dist * 0.2))
                        r = int(255 * (1 - norm))
                        g = int(255 * norm)
                        painter.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, QColor(r, g, 0, 40))

        # Draw Visit Heatmap
        if self.show_visit_map and self.agent.visited_nodes:
            max_visits = max(self.agent.visited_nodes.values())
            for (x, y), visits in self.agent.visited_nodes.items():
                alpha = int(150 * (visits / max_visits))
                painter.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE, QColor(0, 100, 255, alpha))

        # Draw Grid & Terrain
        for x in range(GRID_W):
            for y in range(GRID_H):
                rect = QRectF(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                t = self.env.terrain.get((x, y), T_NORMAL)
                if t == T_OBSTACLE:
                    painter.fillRect(rect, QColor(80, 30, 30))
                elif t == T_MUD:
                    painter.fillRect(rect, QColor(60, 40, 15))
                elif t == T_BOOST:
                    painter.fillRect(rect, QColor(60, 60, 15))

        painter.setPen(QPen(QColor(35, 35, 35), 1))
        for x in range(0, GRID_W * CELL_SIZE, CELL_SIZE):
            painter.drawLine(x, 0, x, GRID_H * CELL_SIZE)
        for y in range(0, GRID_H * CELL_SIZE, CELL_SIZE):
            painter.drawLine(0, y, GRID_W * CELL_SIZE, y)

        # Draw Wind Particles
        painter.setPen(QPen(QColor(100, 150, 200, 120), 2))
        for p in self.particles:
            px = p[0] * CELL_SIZE
            py = p[1] * CELL_SIZE
            wx, wy = self.env.get_wind(p[0], p[1])
            painter.drawLine(QPointF(px, py), QPointF(px + wx * 5, py + wy * 5))

        # Draw Spline Path
        if len(self.path_history) > 1:
            points = [self.grid_to_pixel(x, y) for x, y in self.path_history]
            spline_points = []
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    p_prev = points[i-1] if i > 0 else points[i]
                    p_curr = points[i]
                    p_next = points[i+1]
                    p_after = points[i+2] if i < len(points) - 2 else points[i+1]
                    for t in range(0, 11, 1):
                        spline_points.append(catmull_rom_spline(p_prev, p_curr, p_next, p_after, t / 10.0))
                        
            for i in range(len(spline_points) - 1):
                ratio = i / len(spline_points)
                r = 255
                g = int(50 + 105 * ratio)
                b = 0
                painter.setPen(QPen(QColor(r, g, b, 150), 3))
                painter.drawLine(spline_points[i], spline_points[i+1])

        # Draw MCTS Tree
        if self.agent.last_root and not self.is_animating and not self.env.done:
            self.draw_tree(painter, self.agent.last_root)

        # Draw Storm Trail
        for i, pos in enumerate(self.env.storm_trail):
            alpha = int(100 * (i + 1) / len(self.env.storm_trail))
            painter.setBrush(QBrush(QColor(255, 0, 0, alpha)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.grid_to_pixel(*pos), CELL_SIZE * 0.4, CELL_SIZE * 0.4)

        # Draw Storm
        sx, sy = self.env.storm_pos
        storm_center = self.grid_to_pixel(sx, sy)
        painter.setBrush(QBrush(QColor(255, 0, 0, 150)))
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawEllipse(storm_center, CELL_SIZE * 0.7, CELL_SIZE * 0.7)
        painter.setPen(QColor(255, 200, 200, 200))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(storm_center.x() - 20, storm_center.y() - 20, 40, 40), Qt.AlignCenter, "⚡")

        # Draw Goal
        gx, gy = self.env.goal
        painter.setBrush(QBrush(QColor(0, 200, 100)))
        painter.drawRect(gx * CELL_SIZE + 5, gy * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)

        # Draw Start
        sx, sy = self.env.start
        painter.setBrush(QBrush(QColor(200, 200, 0, 100)))
        painter.drawRect(sx * CELL_SIZE + 5, sy * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)

        # Draw NN Policy Priors
        if self.agent.last_root and not self.is_animating and not self.env.done:
            policy, _ = self.env.mock_predictor(self.env.get_full_state())
            agent_pos = self.grid_to_pixel(*self.env.agent_pos)
            for i, p in enumerate(policy):
                dx, dy = ACTIONS[i]
                length = p * CELL_SIZE * 1.5
                end_x = agent_pos.x() + dx * length
                end_y = agent_pos.y() + dy * length
                painter.setPen(QPen(QColor(255, 255, 0, 150), 2))
                painter.drawLine(agent_pos, QPointF(end_x, end_y))

        # Draw Agent
        painter.setBrush(QBrush(QColor(0, 150, 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(self.current_pixel_pos, 12, 12)

    def mousePressEvent(self, event):
        pos = event.position()
        gx, gy = int(pos.x() // CELL_SIZE), int(pos.y() // CELL_SIZE)
        
        if event.button() == Qt.LeftButton:
            self.env.set_goal(gx, gy)
            self.path_history = [tuple(self.env.agent_pos)]
            self.is_running = True
        elif event.button() == Qt.RightButton:
            self.env.toggle_obstacle(gx, gy)
            self.path_history = [tuple(self.env.agent_pos)]
            self.is_running = True
        elif event.button() == Qt.MiddleButton:
            self.env.set_start(gx, gy)
            self.path_history = [tuple(self.env.agent_pos)]
            self.current_pixel_pos = self.grid_to_pixel(*self.env.agent_pos)
            self.target_pixel_pos = self.current_pixel_pos
            self.is_running = True
        self.update()

class ControlPanel(QWidget):
    def __init__(self, env, agent, canvas):
        super().__init__()
        self.env = env
        self.agent = agent
        self.canvas = canvas
        
        layout = QVBoxLayout(self)
        
        # Status Group
        status_group = QGroupBox("AI Telemetry")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Status: Running")
        self.status_label.setStyleSheet("color: #00ccff; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        self.metrics_label = QLabel("MCTS Time: 0.00 ms\nNodes/Sec: 0\nMax Depth: 0\nQ-Value: 0.0\nSteps: 0 (Best: N/A)")
        status_layout.addWidget(self.metrics_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Controls Group
        ctrl_group = QGroupBox("Simulation")
        ctrl_layout = QGridLayout()
        
        self.play_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.play_btn, 0, 0)
        
        step_btn = QPushButton("Step")
        step_btn.clicked.connect(self.step_sim)
        ctrl_layout.addWidget(step_btn, 0, 1)
        
        reset_btn = QPushButton("Reset Pos")
        reset_btn.clicked.connect(self.reset_sim)
        ctrl_layout.addWidget(reset_btn, 1, 0)
        
        maze_btn = QPushButton("Gen Procedural Env")
        maze_btn.clicked.connect(self.gen_terrain)
        ctrl_layout.addWidget(maze_btn, 1, 1)
        
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # AI Brain Group
        ai_group = QGroupBox("AI Brain (PUCT)")
        ai_layout = QVBoxLayout()
        
        self.val_heatmap_chk = QCheckBox("Show Value Heatmap")
        self.val_heatmap_chk.stateChanged.connect(self.toggle_val_heatmap)
        ai_layout.addWidget(self.val_heatmap_chk)
        
        self.visit_heatmap_chk = QCheckBox("Show MCTS Visit Heatmap")
        self.visit_heatmap_chk.stateChanged.connect(self.toggle_visit_heatmap)
        ai_layout.addWidget(self.visit_heatmap_chk)
        
        self.sim_label = QLabel("MCTS Simulations: 100")
        ai_layout.addWidget(self.sim_label)
        self.sim_slider = QSlider(Qt.Horizontal)
        self.sim_slider.setRange(10, 500)
        self.sim_slider.setValue(100)
        self.sim_slider.valueChanged.connect(self.update_sims)
        ai_layout.addWidget(self.sim_slider)
        
        self.c_puct_label = QLabel("C_PUCT (Exploration): 1.5")
        ai_layout.addWidget(self.c_puct_label)
        self.c_puct_slider = QSlider(Qt.Horizontal)
        self.c_puct_slider.setRange(0, 40)
        self.c_puct_slider.setValue(15)
        self.c_puct_slider.valueChanged.connect(self.update_c_puct)
        ai_layout.addWidget(self.c_puct_slider)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        # Environment Group
        env_group = QGroupBox("Environment Physics")
        env_layout = QVBoxLayout()
        
        self.speed_label = QLabel("Sim Speed: 60 FPS")
        env_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 60)
        self.speed_slider.setValue(60)
        self.speed_slider.valueChanged.connect(self.update_speed)
        env_layout.addWidget(self.speed_slider)
        
        env_layout.addWidget(QLabel("Wind Strength: 1.0"))
        self.wind_slider = QSlider(Qt.Horizontal)
        self.wind_slider.setRange(0, 30)
        self.wind_slider.setValue(10)
        self.wind_slider.valueChanged.connect(self.update_wind)
        env_layout.addWidget(self.wind_slider)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        help_text = QLabel("Interactions:\nLeft-Click: Move Goal\nRight-Click: Toggle Obstacle\nMiddle-Click: Move Start\nLegend:\nBrown=Mud, Yellow=Boost, Red=Storm")
        help_text.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(help_text)
        
        layout.addStretch()
        
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start(100)

    def refresh_ui(self):
        if self.env.done:
            self.status_label.setText("Status: Goal Reached!")
            self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        elif self.canvas.is_running:
            self.status_label.setText("Status: Navigating...")
            self.status_label.setStyleSheet("color: #00ccff; font-weight: bold;")
        else:
            self.status_label.setText("Status: Paused")
            self.status_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
            
        nodes_per_sec = self.agent.simulations / (self.agent.last_search_time / 1000) if self.agent.last_search_time > 0 else 0
        best_str = str(self.env.best_steps) if self.env.best_steps != float('inf') else "N/A"
            
        self.metrics_label.setText(
            f"MCTS Time: {self.agent.last_search_time:.2f} ms\n"
            f"Nodes/Sec: {int(nodes_per_sec)}\n"
            f"Max Depth: {self.agent.max_depth_reached}\n"
            f"Q-Value: {self.agent.chosen_q_value:.2f}\n"
            f"Steps: {self.env.steps} (Best: {best_str})"
        )

    def toggle_play(self):
        self.canvas.is_running = not self.canvas.is_running
        self.play_btn.setText("Pause" if self.canvas.is_running else "Play")

    def step_sim(self):
        self.canvas.is_running = False
        self.canvas.do_step = True
        self.play_btn.setText("Play")

    def reset_sim(self):
        self.env.reset()
        self.canvas.path_history = [tuple(self.env.agent_pos)]
        self.canvas.current_pixel_pos = self.canvas.grid_to_pixel(*self.env.agent_pos)
        self.canvas.target_pixel_pos = self.canvas.current_pixel_pos
        self.canvas.is_animating = False
        self.canvas.is_running = True
        self.play_btn.setText("Pause")

    def gen_terrain(self):
        self.env.generate_terrain()
        self.reset_sim()

    def toggle_val_heatmap(self, state):
        self.canvas.show_heatmap = (state == Qt.Checked.value)
        self.canvas.update()

    def toggle_visit_heatmap(self, state):
        self.canvas.show_visit_map = (state == Qt.Checked.value)
        self.canvas.update()

    def update_speed(self, val):
        interval = int(1000 / val)
        self.canvas.timer.setInterval(interval)
        self.speed_label.setText(f"Sim Speed: {val} FPS")

    def update_wind(self, val):
        self.env.wind_strength = val / 10.0
        self.sender().parent().children()[1].setText(f"Wind Strength: {val / 10.0}")

    def update_sims(self, val):
        self.agent.simulations = val
        self.sim_label.setText(f"MCTS Simulations: {val}")

    def update_c_puct(self, val):
        self.agent.c_puct = val / 10.0
        self.c_puct_label.setText(f"C_PUCT (Exploration): {val / 10.0}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlphaZero PUCT Sandbox - Fluid Dynamics & Deep Search")
        self.setStyleSheet(QSS)
        
        self.env = Environment()
        self.agent = PUCTAgent(self.env)
        self.canvas = SimulationCanvas(self.env, self.agent)
        
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.canvas)
        
        self.panel = ControlPanel(self.env, self.agent, self.canvas)
        self.panel.setFixedWidth(300)
        main_layout.addWidget(self.panel)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())