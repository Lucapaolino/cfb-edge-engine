import json
import math
import random
import subprocess
import sys

from pathlib import Path

from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
    QThread
)

from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient
)

from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget
)


# ============================================================
# SORTABLE TABLE ITEM
#
# QTableWidget normally sorts visible text. This subclass stores
# a separate machine-sort value in Qt.UserRole + 1 so numeric
# columns sort numerically and Strength sorts in the intended
# order: HIGH > MEDIUM > LOW > N/A.
# ============================================================

class SortableTableWidgetItem(QTableWidgetItem):

    def __lt__(
        self,
        other
    ):

        left = self.data(
            Qt.UserRole + 1
        )

        right = other.data(
            Qt.UserRole + 1
        )

        if (
            left is not None
            and
            right is not None
        ):

            try:

                return left < right

            except TypeError:

                return str(
                    left
                ) < str(
                    right
                )

        return super().__lt__(
            other
        )


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


CONTROLLER = (
    PROJECT_ROOT
    / "src"
    / "run_game_day_v1.py"
)


BOARD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "live_model_market_board_2026_v1.json"
)


# ============================================================
# PIPELINE THREAD
#
# The GUI now has ONE pipeline dependency: run_game_day_v1.py.
#
# SYSTEM TEST:
#   python src/run_game_day_v1.py --dry-run
#   - syntax/preflight only
#   - NO CFBD / The Odds API / OddsPapi calls
#
# RUN GAME DAY:
#   python src/run_game_day_v1.py
#   - controller owns every live-data, model, market, board,
#     and prospective-ledger stage
#
# The controller emits:
#   PIPELINE_STAGE|<stage name>|<0-100 progress>
# so the GUI can show real pipeline progress without duplicating
# pipeline logic or making its own sportsbook API calls.
# ============================================================

class PipelineThread(QThread):

    output_received = Signal(str)

    stage_changed = Signal(
        str,
        int
    )

    completed = Signal(bool)


    def __init__(
        self,
        dry_run=False
    ):

        super().__init__()

        self.dry_run = dry_run


    def detect_stage(
        self,
        line
    ):

        if not line.startswith(
            "PIPELINE_STAGE|"
        ):
            return

        pieces = line.split(
            "|",
            2
        )

        if len(pieces) != 3:
            return

        stage = pieces[1].strip()

        try:
            progress = int(
                pieces[2].strip()
            )
        except ValueError:
            return

        progress = max(
            0,
            min(
                100,
                progress
            )
        )

        self.stage_changed.emit(
            stage,
            progress
        )


    def run(self):

        command = [
            sys.executable,
            str(
                CONTROLLER
            )
        ]

        if self.dry_run:
            command.append(
                "--dry-run"
            )

        try:

            if not CONTROLLER.exists():

                raise RuntimeError(
                    "Canonical controller not found:\n"
                    f"{CONTROLLER}"
                )

            self.stage_changed.emit(
                "STARTING SYSTEM TEST"
                if self.dry_run
                else "STARTING GAME DAY",
                1
            )

            process = subprocess.Popen(

                command,

                cwd=PROJECT_ROOT,

                stdout=subprocess.PIPE,

                stderr=subprocess.STDOUT,

                text=True,

                bufsize=1
            )

            if process.stdout is None:

                raise RuntimeError(
                    "Could not read pipeline output."
                )

            for line in process.stdout:

                clean_line = line.rstrip()

                self.output_received.emit(
                    clean_line
                )

                self.detect_stage(
                    clean_line
                )

            return_code = process.wait()

            if return_code != 0:

                raise RuntimeError(
                    "Canonical game-day controller failed "
                    f"with exit code {return_code}."
                )

            self.stage_changed.emit(
                "SYSTEM TEST COMPLETE"
                if self.dry_run
                else "ANALYSIS COMPLETE",
                100
            )

            self.completed.emit(
                True
            )

        except Exception as error:

            self.output_received.emit(
                f"ERROR: {error}"
            )

            self.stage_changed.emit(
                "SYSTEM ERROR",
                0
            )

            self.completed.emit(
                False
            )


# ============================================================
# NEURAL GRAPH VISUALIZATION
#
# A pseudo-3D network rendered entirely with PySide6.
#
# The graph is intentionally NOT an atom/orbit visualization.
# Nodes exist in 3D space, drift with independent linear
# velocities, remain connected by a sparse graph, and project
# into 2D with perspective.
#
# Interaction:
#   - Idle: subtle drift + random signal pulses.
#   - Select a game: a branch activates, pulses travel through
#     it, and one terminal node extends outward before settling
#     back into the network.
#   - Pipeline activity: movement and signal traffic increase.
# ============================================================

class AICoreWidget(QWidget):

    def __init__(self):

        super().__init__()

        self.setMinimumSize(
            320,
            260
        )

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.activity = 0.20
        self.target_activity = 0.20
        self.progress = 0
        self.state = "SYSTEM READY"

        self.frame_counter = 0
        self.camera_yaw = -0.24
        self.camera_pitch = 0.13
        self.camera_phase = 0.0

        self.nodes = []
        self.edges = []
        self.adjacency = {}
        self.pulses = []

        self.active_nodes = {}
        self.extension_node = None
        self.extension_direction = (
            0.0,
            0.0,
            0.0
        )
        self.extension_amount = 0.0
        self.extension_target = 0.0

        self.selection_generation = 0
        self.selected_game = ""

        self.random = random.Random(
            42
        )

        self.build_network()

        self.timer = QTimer(
            self
        )

        self.timer.timeout.connect(
            self.animate
        )

        # ~30 FPS is smooth while staying light enough for a
        # normal MacBook GUI workload.
        self.timer.start(
            33
        )


    # ========================================================
    # NETWORK MODEL
    # ========================================================

    def build_network(self):

        self.nodes = []

        # Dense central cloud plus a smaller set of outer nodes.
        # This shape is closer to a neural / graph visualization
        # than a spherical atom.
        node_count = 74

        for index in range(node_count):

            if index < 54:

                x = self.random.gauss(
                    0.0,
                    0.48
                )

                y = self.random.gauss(
                    0.0,
                    0.42
                )

                z = self.random.gauss(
                    0.0,
                    0.52
                )

            else:

                angle = self.random.uniform(
                    0.0,
                    math.tau
                )

                radius = self.random.uniform(
                    0.88,
                    1.30
                )

                x = math.cos(
                    angle
                ) * radius

                y = math.sin(
                    angle
                ) * radius * 0.72

                z = self.random.uniform(
                    -0.85,
                    0.85
                )

            speed = self.random.uniform(
                0.0010,
                0.0031
            )

            direction = self.random.uniform(
                0.0,
                math.tau
            )

            vz = self.random.uniform(
                -speed,
                speed
            )

            node = {
                "x": x,
                "y": y,
                "z": z,
                "home_x": x,
                "home_y": y,
                "home_z": z,
                "vx": math.cos(direction) * speed,
                "vy": math.sin(direction) * speed,
                "vz": vz,
                "size": self.random.uniform(
                    2.0,
                    4.6
                ),
                "phase": self.random.uniform(
                    0.0,
                    math.tau
                ),
                "energy": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "offset_z": 0.0
            }

            self.nodes.append(
                node
            )

        self.build_edges()


    def node_distance_squared(
        self,
        left,
        right
    ):

        dx = (
            left["x"]
            -
            right["x"]
        )

        dy = (
            left["y"]
            -
            right["y"]
        )

        dz = (
            left["z"]
            -
            right["z"]
        )

        return (
            dx * dx
            +
            dy * dy
            +
            dz * dz
        )


    def build_edges(self):

        edge_set = set()

        # Each node links to its nearest few neighbors.
        for index, node in enumerate(
            self.nodes
        ):

            distances = []

            for other_index, other in enumerate(
                self.nodes
            ):

                if index == other_index:
                    continue

                distances.append(
                    (
                        self.node_distance_squared(
                            node,
                            other
                        ),
                        other_index
                    )
                )

            distances.sort(
                key=lambda item: item[0]
            )

            neighbor_count = (
                3
                if index < 54
                else 2
            )

            for _, other_index in distances[
                :neighbor_count
            ]:

                edge = tuple(
                    sorted(
                        (
                            index,
                            other_index
                        )
                    )
                )

                edge_set.add(
                    edge
                )

        # Add a few longer-range connections so pulses can jump
        # between sub-clusters without making the graph dense.
        for _ in range(18):

            left = self.random.randrange(
                len(
                    self.nodes
                )
            )

            right = self.random.randrange(
                len(
                    self.nodes
                )
            )

            if left == right:
                continue

            edge_set.add(
                tuple(
                    sorted(
                        (
                            left,
                            right
                        )
                    )
                )
            )

        self.edges = sorted(
            edge_set
        )

        self.adjacency = {
            index: []
            for index in range(
                len(
                    self.nodes
                )
            )
        }

        for left, right in self.edges:

            self.adjacency[
                left
            ].append(
                right
            )

            self.adjacency[
                right
            ].append(
                left
            )


    # ========================================================
    # STATE / INTERACTION
    # ========================================================

    def set_stage(
        self,
        state,
        progress
    ):

        self.state = str(
            state
        )

        self.progress = int(
            progress
        )

        if progress <= 5:
            self.target_activity = 0.22

        elif progress < 30:
            self.target_activity = 0.44

        elif progress < 60:
            self.target_activity = 0.68

        elif progress < 90:
            self.target_activity = 0.88

        elif progress < 100:
            self.target_activity = 1.0

        else:
            self.target_activity = 0.48

        # Stage changes send a small wave through the network.
        self.trigger_random_pulse(
            count=3
        )

        self.update()


    def set_signal_activity(
        self,
        strength,
        warning_text=""
    ):

        strength = str(
            strength
            or
            ""
        ).upper()

        warning_text = str(
            warning_text
            or
            ""
        ).upper()

        targets = {
            "HIGH": 0.82,
            "MEDIUM": 0.64,
            "LOW": 0.42,
            "N/A": 0.26
        }

        target = targets.get(
            strength,
            0.32
        )

        if "MODEL_MARKET_OUTLIER" in warning_text:
            target = max(
                target,
                0.58
            )

        self.target_activity = target

        self.update()


    def activate_game(
        self,
        game_key,
        strength="N/A",
        warning_text=""
    ):

        """
        React to a selected game.

        The visual reaction is deterministic for the selected
        game during this session but does not claim to represent
        literal model neurons or hidden reasoning.
        """

        self.selected_game = str(
            game_key
            or
            ""
        )

        self.set_signal_activity(
            strength,
            warning_text
        )

        self.selection_generation += 1

        if not self.nodes:
            return

        # Pick a focal point from the central graph, then walk a
        # connected branch. This makes the response look like a
        # signal actually propagating rather than a random flash.
        focal = self.random.randrange(
            min(
                54,
                len(
                    self.nodes
                )
            )
        )

        branch = [
            focal
        ]

        previous = None
        current = focal

        branch_length = self.random.randint(
            5,
            8
        )

        for _ in range(branch_length - 1):

            neighbors = [
                node_index
                for node_index in self.adjacency.get(
                    current,
                    []
                )
                if node_index != previous
            ]

            if not neighbors:
                break

            next_node = self.random.choice(
                neighbors
            )

            branch.append(
                next_node
            )

            previous = current
            current = next_node

        # Light the branch and create sequential edge pulses.
        for order, node_index in enumerate(
            branch
        ):

            self.active_nodes[
                node_index
            ] = max(
                self.active_nodes.get(
                    node_index,
                    0.0
                ),
                1.0
                -
                order * 0.06
            )

        for order in range(
            len(branch) - 1
        ):

            self.pulses.append(
                {
                    "from": branch[order],
                    "to": branch[order + 1],
                    "t": -0.16 * order,
                    "speed": 0.030 + self.activity * 0.018,
                    "life": 1.0
                }
            )

        # Extend the branch's terminal node outward from the
        # network center. It eases back naturally over time.
        terminal = branch[-1]

        terminal_node = self.nodes[
            terminal
        ]

        length = math.sqrt(
            terminal_node["x"] ** 2
            +
            terminal_node["y"] ** 2
            +
            terminal_node["z"] ** 2
        )

        if length < 0.05:

            direction = (
                1.0,
                0.0,
                0.0
            )

        else:

            direction = (
                terminal_node["x"] / length,
                terminal_node["y"] / length,
                terminal_node["z"] / length
            )

        self.extension_node = terminal
        self.extension_direction = direction
        self.extension_target = self.random.uniform(
            0.42,
            0.68
        )

        # Add two additional short pulses elsewhere to make the
        # rest of the graph feel alive during the selection.
        self.trigger_random_pulse(
            count=2
        )

        self.update()


    def trigger_random_pulse(
        self,
        count=1
    ):

        if not self.edges:
            return

        for _ in range(
            count
        ):

            left, right = self.random.choice(
                self.edges
            )

            if self.random.random() < 0.5:
                left, right = right, left

            self.pulses.append(
                {
                    "from": left,
                    "to": right,
                    "t": 0.0,
                    "speed": self.random.uniform(
                        0.014,
                        0.024
                    ) + self.activity * 0.009,
                    "life": 1.0
                }
            )

            self.active_nodes[
                left
            ] = max(
                self.active_nodes.get(
                    left,
                    0.0
                ),
                0.55
            )


    # Clicking directly on the visualization also causes a
    # harmless visual burst. Game selection remains the main
    # interaction source.
    def mousePressEvent(
        self,
        event
    ):

        self.activate_game(
            self.selected_game
            or
            "MANUAL VISUAL PULSE",
            "MEDIUM",
            ""
        )

        super().mousePressEvent(
            event
        )


    # ========================================================
    # ANIMATION
    # ========================================================

    def animate(self):

        self.frame_counter += 1

        self.activity += (
            self.target_activity
            -
            self.activity
        ) * 0.045

        self.camera_phase += (
            0.008
            +
            self.activity * 0.004
        )

        # Slow camera drift gives actual depth without making the
        # whole object look like a rotating atom.
        self.camera_yaw += (
            0.00035
            +
            self.activity * 0.00025
        )

        self.camera_pitch = (
            0.11
            +
            math.sin(
                self.camera_phase * 0.33
            ) * 0.055
        )

        motion_scale = (
            0.72
            +
            self.activity * 1.25
        )

        # ----------------------------------------------------
        # Linear node motion + soft bounds / home attraction.
        # ----------------------------------------------------

        for node in self.nodes:

            node["x"] += node["vx"] * motion_scale
            node["y"] += node["vy"] * motion_scale
            node["z"] += node["vz"] * motion_scale

            # Weak home attraction prevents the cloud from
            # wandering away while still allowing clear motion.
            node["vx"] += (
                node["home_x"]
                -
                node["x"]
            ) * 0.000020

            node["vy"] += (
                node["home_y"]
                -
                node["y"]
            ) * 0.000020

            node["vz"] += (
                node["home_z"]
                -
                node["z"]
            ) * 0.000020

            # Bounce softly around the 3D volume.
            for axis in (
                "x",
                "y",
                "z"
            ):

                velocity_key = (
                    "v"
                    +
                    axis
                )

                limit = (
                    1.45
                    if axis != "y"
                    else 1.18
                )

                if node[axis] > limit:
                    node[axis] = limit
                    node[velocity_key] *= -1.0

                elif node[axis] < -limit:
                    node[axis] = -limit
                    node[velocity_key] *= -1.0

            node["energy"] *= 0.955

        # ----------------------------------------------------
        # Selected-branch extension.
        # ----------------------------------------------------

        self.extension_amount += (
            self.extension_target
            -
            self.extension_amount
        ) * 0.12

        self.extension_target *= 0.965

        if self.extension_target < 0.008:
            self.extension_target = 0.0

        for index, node in enumerate(
            self.nodes
        ):

            if (
                self.extension_node is not None
                and
                index == self.extension_node
            ):

                target_offset = (
                    self.extension_direction[0] * self.extension_amount,
                    self.extension_direction[1] * self.extension_amount,
                    self.extension_direction[2] * self.extension_amount
                )

            else:

                target_offset = (
                    0.0,
                    0.0,
                    0.0
                )

            node["offset_x"] += (
                target_offset[0]
                -
                node["offset_x"]
            ) * 0.16

            node["offset_y"] += (
                target_offset[1]
                -
                node["offset_y"]
            ) * 0.16

            node["offset_z"] += (
                target_offset[2]
                -
                node["offset_z"]
            ) * 0.16

        # ----------------------------------------------------
        # Active node energy decay.
        # ----------------------------------------------------

        expired_active = []

        for node_index in list(
            self.active_nodes.keys()
        ):

            self.active_nodes[
                node_index
            ] *= 0.955

            self.nodes[
                node_index
            ]["energy"] = max(
                self.nodes[
                    node_index
                ]["energy"],
                self.active_nodes[
                    node_index
                ]
            )

            if self.active_nodes[
                node_index
            ] < 0.03:

                expired_active.append(
                    node_index
                )

        for node_index in expired_active:

            self.active_nodes.pop(
                node_index,
                None
            )

        # ----------------------------------------------------
        # Traveling edge pulses.
        # ----------------------------------------------------

        live_pulses = []

        for pulse in self.pulses:

            pulse["t"] += pulse["speed"]

            if pulse["t"] <= 1.18:

                if pulse["t"] >= 0.0:

                    destination = pulse["to"]

                    if pulse["t"] > 0.82:

                        self.nodes[
                            destination
                        ]["energy"] = max(
                            self.nodes[
                                destination
                            ]["energy"],
                            0.72
                        )

                live_pulses.append(
                    pulse
                )

        self.pulses = live_pulses

        # Random background traffic. More activity means more
        # signal movement, but idle mode remains subtle.
        random_chance = (
            0.010
            +
            self.activity * 0.028
        )

        if self.random.random() < random_chance:
            self.trigger_random_pulse(
                count=1
            )

        self.update()


    # ========================================================
    # 3D PROJECTION
    # ========================================================

    def rotated_position(
        self,
        node
    ):

        x = (
            node["x"]
            +
            node["offset_x"]
        )

        y = (
            node["y"]
            +
            node["offset_y"]
        )

        z = (
            node["z"]
            +
            node["offset_z"]
        )

        cos_yaw = math.cos(
            self.camera_yaw
        )

        sin_yaw = math.sin(
            self.camera_yaw
        )

        x1 = (
            x * cos_yaw
            -
            z * sin_yaw
        )

        z1 = (
            x * sin_yaw
            +
            z * cos_yaw
        )

        cos_pitch = math.cos(
            self.camera_pitch
        )

        sin_pitch = math.sin(
            self.camera_pitch
        )

        y2 = (
            y * cos_pitch
            -
            z1 * sin_pitch
        )

        z2 = (
            y * sin_pitch
            +
            z1 * cos_pitch
        )

        return (
            x1,
            y2,
            z2
        )


    def project_node(
        self,
        node,
        width,
        height
    ):

        x, y, z = self.rotated_position(
            node
        )

        # Perspective camera. Positive z comes slightly closer.
        camera_distance = 4.0

        denominator = max(
            1.8,
            camera_distance
            -
            z
        )

        perspective = (
            camera_distance
            /
            denominator
        )

        scale = min(
            width,
            height
        ) * 0.295

        center_x = width * 0.50
        center_y = height * 0.46

        screen_x = (
            center_x
            +
            x * scale * perspective
        )

        screen_y = (
            center_y
            +
            y * scale * perspective
        )

        depth_normalized = max(
            0.0,
            min(
                1.0,
                (
                    z
                    +
                    1.8
                )
                /
                3.6
            )
        )

        return {
            "point": QPointF(
                screen_x,
                screen_y
            ),
            "depth": z,
            "depth_norm": depth_normalized,
            "perspective": perspective
        }


    # ========================================================
    # DRAW
    # ========================================================

    def paintEvent(
        self,
        event
    ):

        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.Antialiasing
        )

        width = self.width()
        height = self.height()

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        background = QLinearGradient(
            0,
            0,
            0,
            height
        )

        background.setColorAt(
            0.0,
            QColor(
                7,
                10,
                14
            )
        )

        background.setColorAt(
            1.0,
            QColor(
                2,
                3,
                5
            )
        )

        painter.fillRect(
            self.rect(),
            background
        )

        # Soft central amber atmosphere — not a central orb.
        ambient = QRadialGradient(
            QPointF(
                width * 0.50,
                height * 0.45
            ),
            min(
                width,
                height
            ) * 0.52
        )

        ambient.setColorAt(
            0.0,
            QColor(
                255,
                115,
                20,
                int(
                    24
                    +
                    self.activity * 38
                )
            )
        )

        ambient.setColorAt(
            0.48,
            QColor(
                255,
                95,
                8,
                int(
                    7
                    +
                    self.activity * 12
                )
            )
        )

        ambient.setColorAt(
            1.0,
            QColor(
                0,
                0,
                0,
                0
            )
        )

        painter.setPen(
            Qt.NoPen
        )

        painter.setBrush(
            ambient
        )

        painter.drawRect(
            self.rect()
        )

        if not self.nodes:
            return

        projected = [
            self.project_node(
                node,
                width,
                height
            )
            for node in self.nodes
        ]

        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        for left, right in self.edges:

            left_projection = projected[
                left
            ]

            right_projection = projected[
                right
            ]

            average_depth = (
                left_projection["depth_norm"]
                +
                right_projection["depth_norm"]
            ) / 2.0

            active_edge = (
                left in self.active_nodes
                or
                right in self.active_nodes
            )

            if active_edge:

                alpha = int(
                    58
                    +
                    self.activity * 80
                )

                width_pen = 1.15

            else:

                alpha = int(
                    14
                    +
                    average_depth * 28
                    +
                    self.activity * 12
                )

                width_pen = 0.65

            painter.setPen(
                QPen(
                    QColor(
                        255,
                        125,
                        28,
                        max(
                            8,
                            min(
                                180,
                                alpha
                            )
                        )
                    ),
                    width_pen
                )
            )

            painter.drawLine(
                left_projection["point"],
                right_projection["point"]
            )

        # ----------------------------------------------------
        # Traveling pulses
        # ----------------------------------------------------

        for pulse in self.pulses:

            t = pulse["t"]

            if t < 0.0 or t > 1.0:
                continue

            left = projected[
                pulse["from"]
            ]["point"]

            right = projected[
                pulse["to"]
            ]["point"]

            x = (
                left.x()
                +
                (
                    right.x()
                    -
                    left.x()
                ) * t
            )

            y = (
                left.y()
                +
                (
                    right.y()
                    -
                    left.y()
                ) * t
            )

            point = QPointF(
                x,
                y
            )

            pulse_radius = (
                2.5
                +
                self.activity * 3.5
            )

            pulse_glow = QRadialGradient(
                point,
                pulse_radius * 4.2
            )

            pulse_glow.setColorAt(
                0.0,
                QColor(
                    255,
                    245,
                    208,
                    255
                )
            )

            pulse_glow.setColorAt(
                0.26,
                QColor(
                    255,
                    176,
                    70,
                    230
                )
            )

            pulse_glow.setColorAt(
                1.0,
                QColor(
                    255,
                    90,
                    0,
                    0
                )
            )

            painter.setPen(
                Qt.NoPen
            )

            painter.setBrush(
                pulse_glow
            )

            painter.drawEllipse(
                point,
                pulse_radius * 4.2,
                pulse_radius * 4.2
            )

        # ----------------------------------------------------
        # Nodes, back to front
        # ----------------------------------------------------

        draw_order = sorted(
            range(
                len(
                    self.nodes
                )
            ),
            key=lambda index: projected[index]["depth"]
        )

        for node_index in draw_order:

            node = self.nodes[
                node_index
            ]

            projection = projected[
                node_index
            ]

            point = projection[
                "point"
            ]

            depth_factor = (
                0.72
                +
                projection["depth_norm"] * 0.68
            )

            energy = max(
                node["energy"],
                self.active_nodes.get(
                    node_index,
                    0.0
                )
            )

            flicker = (
                0.88
                +
                0.12
                *
                math.sin(
                    self.camera_phase * 2.2
                    +
                    node["phase"]
                )
            )

            node_radius = (
                node["size"]
                *
                depth_factor
                *
                (
                    0.88
                    +
                    energy * 0.72
                )
                *
                flicker
            )

            if energy > 0.04:

                glow_radius = (
                    node_radius
                    *
                    (
                        4.0
                        +
                        energy * 2.0
                    )
                )

                glow = QRadialGradient(
                    point,
                    glow_radius
                )

                glow.setColorAt(
                    0.0,
                    QColor(
                        255,
                        231,
                        175,
                        int(
                            180
                            +
                            energy * 70
                        )
                    )
                )

                glow.setColorAt(
                    0.25,
                    QColor(
                        255,
                        145,
                        38,
                        int(
                            120
                            +
                            energy * 80
                        )
                    )
                )

                glow.setColorAt(
                    1.0,
                    QColor(
                        255,
                        80,
                        0,
                        0
                    )
                )

                painter.setPen(
                    Qt.NoPen
                )

                painter.setBrush(
                    glow
                )

                painter.drawEllipse(
                    point,
                    glow_radius,
                    glow_radius
                )

            # Inactive nodes stay dark amber instead of green so
            # the visualization remains in the product palette.
            base_alpha = int(
                120
                +
                projection["depth_norm"] * 90
            )

            if energy > 0.04:

                node_color = QColor(
                    255,
                    int(
                        156
                        +
                        energy * 70
                    ),
                    55,
                    245
                )

            else:

                node_color = QColor(
                    181,
                    92,
                    28,
                    base_alpha
                )

            painter.setPen(
                QPen(
                    QColor(
                        255,
                        183,
                        92,
                        min(
                            210,
                            base_alpha + 24
                        )
                    ),
                    0.75
                )
            )

            painter.setBrush(
                node_color
            )

            painter.drawEllipse(
                point,
                node_radius,
                node_radius
            )

        # ----------------------------------------------------
        # Visual labels
        # ----------------------------------------------------

        painter.setPen(
            QColor(
                141,
                148,
                158,
                220
            )
        )

        label_font = QFont()
        label_font.setPointSize(
            8
        )
        label_font.setBold(
            True
        )
        painter.setFont(
            label_font
        )

        painter.drawText(
            QRectF(
                15,
                12,
                width - 30,
                22
            ),
            Qt.AlignLeft,
            "NEURAL SIGNAL GRAPH"
        )

        painter.setPen(
            QColor(
                255,
                159,
                62,
                235
            )
        )

        state_font = QFont()
        state_font.setPointSize(
            11
        )
        state_font.setBold(
            True
        )
        painter.setFont(
            state_font
        )

        painter.drawText(
            QRectF(
                0,
                height - 48,
                width,
                24
            ),
            Qt.AlignCenter,
            self.state
        )

        painter.setPen(
            QColor(
                104,
                111,
                121,
                225
            )
        )

        status_font = QFont()
        status_font.setPointSize(
            8
        )
        painter.setFont(
            status_font
        )

        selected_text = (
            self.selected_game
            if self.selected_game
            else "AWAITING GAME SELECTION"
        )

        if len(
            selected_text
        ) > 46:

            selected_text = (
                selected_text[:43]
                +
                "..."
            )

        painter.drawText(
            QRectF(
                12,
                height - 25,
                width - 24,
                18
            ),
            Qt.AlignCenter,
            selected_text
        )

# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.pipeline_thread = None

        self.board_records = []

        self.board_signals = []

        self.signal_by_key = {}

        self.board_expanded = False


        self.setWindowTitle(
            "CFB Edge Engine"
        )


        self.resize(
            1540,
            1050
        )


        self.setMinimumSize(
            1120,
            760
        )


        self.build_interface()

        self.load_board()


    # ========================================================
    # INTERFACE
    # ========================================================

    def build_interface(self):

        central = QWidget()


        self.setCentralWidget(
            central
        )


        main_layout = QVBoxLayout(
            central
        )


        main_layout.setContentsMargins(
            16,
            14,
            16,
            16
        )


        main_layout.setSpacing(
            10
        )


        # ====================================================
        # HEADER
        # ====================================================

        title = QLabel(
            "CFB EDGE ENGINE"
        )


        title.setAlignment(
            Qt.AlignCenter
        )


        title.setObjectName(
            "title"
        )


        subtitle = QLabel(

            "HARD ROCK EXECUTION  •  "
            "NEURAL ANALYSIS  •  "
            "PROSPECTIVE VALIDATION"
        )


        subtitle.setAlignment(
            Qt.AlignCenter
        )


        subtitle.setObjectName(
            "subtitle"
        )


        main_layout.addWidget(
            title
        )


        main_layout.addWidget(
            subtitle
        )


        # ====================================================
        # VERTICAL SPLITTER
        # ====================================================

        self.main_splitter = QSplitter(
            Qt.Vertical
        )


        self.main_splitter.setChildrenCollapsible(
            False
        )


        self.main_splitter.setHandleWidth(
            8
        )


        # ====================================================
        # TOP DASHBOARD
        # ====================================================

        self.top_container = QWidget()


        top_layout = QHBoxLayout(
            self.top_container
        )


        top_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )


        top_layout.setSpacing(
            14
        )


        # ----------------------------------------------------
        # AI CORE
        # ----------------------------------------------------

        core_panel = QFrame()


        core_panel.setObjectName(
            "corePanel"
        )


        core_layout = QVBoxLayout(
            core_panel
        )


        core_layout.setContentsMargins(
            8,
            8,
            8,
            8
        )


        self.core = AICoreWidget()


        core_layout.addWidget(
            self.core
        )


        # ----------------------------------------------------
        # CONTROL PANEL
        # ----------------------------------------------------

        control_panel = QFrame()


        control_panel.setObjectName(
            "controlPanel"
        )


        control_panel.setFixedWidth(
            400
        )


        control_layout = QVBoxLayout(
            control_panel
        )


        control_layout.setContentsMargins(
            18,
            16,
            18,
            16
        )


        control_layout.setSpacing(
            9
        )


        control_title = QLabel(
            "SYSTEM CONTROL"
        )


        control_title.setObjectName(
            "sectionTitle"
        )


        self.status_label = QLabel(
            "SYSTEM READY"
        )


        self.status_label.setObjectName(
            "status"
        )


        self.progress_bar = QProgressBar()


        self.progress_bar.setRange(
            0,
            100
        )


        self.progress_bar.setValue(
            0
        )


        self.progress_bar.setTextVisible(
            False
        )


        self.dry_button = QPushButton(
            "SYSTEM TEST"
        )


        self.dry_button.clicked.connect(
            self.run_dry_test
        )


        self.run_button = QPushButton(
            "RUN GAME DAY"
        )


        self.run_button.setObjectName(
            "runButton"
        )


        self.run_button.clicked.connect(
            self.run_game_day
        )


        board_title = QLabel(
            "CURRENT BOARD"
        )


        board_title.setObjectName(
            "sectionTitle"
        )


        self.summary_label = QLabel(
            "Loading live board..."
        )


        self.summary_label.setObjectName(
            "summary"
        )


        self.summary_label.setWordWrap(
            True
        )


        log_title = QLabel(
            "LIVE SYSTEM LOG"
        )


        log_title.setObjectName(
            "sectionTitle"
        )


        self.log_box = QTextEdit()


        self.log_box.setReadOnly(
            True
        )


        self.log_box.setMinimumHeight(
            70
        )


        self.log_box.setPlaceholderText(
            "Pipeline activity will appear here..."
        )


        mode_label = QLabel(
            "2026 PROSPECTIVE VALIDATION ACTIVE"
        )


        mode_label.setObjectName(
            "researchMode"
        )


        warning = QLabel(

            "Hard Rock execution: OddsPapi. "
            "Reference market: The Odds API. "
            "Strength is prospectively unvalidated "
            "and is not a cover probability."
        )


        warning.setWordWrap(
            True
        )


        warning.setObjectName(
            "warning"
        )


        control_layout.addWidget(
            control_title
        )


        control_layout.addWidget(
            self.status_label
        )


        control_layout.addWidget(
            self.progress_bar
        )


        control_layout.addWidget(
            self.dry_button
        )


        control_layout.addWidget(
            self.run_button
        )


        control_layout.addSpacing(
            4
        )


        control_layout.addWidget(
            board_title
        )


        control_layout.addWidget(
            self.summary_label
        )


        control_layout.addSpacing(
            4
        )


        control_layout.addWidget(
            log_title
        )


        control_layout.addWidget(
            self.log_box,
            1
        )


        control_layout.addWidget(
            mode_label
        )


        control_layout.addWidget(
            warning
        )


        top_layout.addWidget(
            core_panel,
            1
        )


        top_layout.addWidget(
            control_panel
        )


        # ====================================================
        # SIGNAL BOARD
        # ====================================================

        self.board_panel = QFrame()


        self.board_panel.setObjectName(
            "boardPanel"
        )


        board_layout = QVBoxLayout(
            self.board_panel
        )


        board_layout.setContentsMargins(
            14,
            12,
            14,
            14
        )


        board_layout.setSpacing(
            9
        )


        # ----------------------------------------------------
        # BOARD HEADER
        # ----------------------------------------------------

        board_header = QHBoxLayout()


        board_label = QLabel(
            "LIVE HARD ROCK SIGNAL BOARD"
        )


        board_label.setObjectName(
            "sectionTitle"
        )


        self.board_status = QLabel(
            "PROSPECTIVE VALIDATION"
        )


        self.board_status.setObjectName(
            "boardStatus"
        )


        self.expand_board_button = QPushButton(
            "EXPAND BOARD"
        )


        self.expand_board_button.setObjectName(
            "expandButton"
        )


        self.expand_board_button.clicked.connect(
            self.toggle_board_expansion
        )


        board_header.addWidget(
            board_label
        )


        board_header.addStretch()


        board_header.addWidget(
            self.board_status
        )


        board_header.addSpacing(
            10
        )


        board_header.addWidget(
            self.expand_board_button
        )


        board_layout.addLayout(
            board_header
        )


        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        search_layout = QHBoxLayout()


        search_label = QLabel(
            "SEARCH"
        )


        search_label.setObjectName(
            "searchLabel"
        )


        self.search_box = QLineEdit()


        self.search_box.setPlaceholderText(

            "Search team, model line, Hard Rock line, "
            "strength, warning..."
        )


        self.search_box.setClearButtonEnabled(
            True
        )


        self.search_box.textChanged.connect(
            self.filter_board
        )


        self.result_count_label = QLabel(
            ""
        )


        self.result_count_label.setObjectName(
            "resultCount"
        )


        search_layout.addWidget(
            search_label
        )


        search_layout.addWidget(
            self.search_box,
            1
        )


        search_layout.addWidget(
            self.result_count_label
        )


        board_layout.addLayout(
            search_layout
        )


        # ----------------------------------------------------
        # BOARD CONTENT SPLITTER
        # ----------------------------------------------------

        self.board_content_splitter = QSplitter(
            Qt.Horizontal
        )


        self.board_content_splitter.setChildrenCollapsible(
            False
        )


        self.board_content_splitter.setHandleWidth(
            7
        )


        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        self.table = QTableWidget()


        self.table.setColumnCount(
            6
        )


        self.table.setHorizontalHeaderLabels(
            [
                "GAME",
                "MODEL SPREAD",
                "HARD ROCK",
                "MODEL EDGE",
                "WIN %",
                "STRENGTH"
            ]
        )


        table_header = (
            self.table
            .horizontalHeader()
        )


        table_header.setSectionResizeMode(
            QHeaderView.ResizeToContents
        )


        table_header.setSectionResizeMode(
            0,
            QHeaderView.Stretch
        )


        table_header.setSectionResizeMode(
            1,
            QHeaderView.Stretch
        )


        table_header.setSectionResizeMode(
            2,
            QHeaderView.Stretch
        )


        table_header.setSectionResizeMode(
            3,
            QHeaderView.Stretch
        )


        self.table.setSelectionBehavior(
            QTableWidget.SelectRows
        )


        self.table.setSelectionMode(
            QTableWidget.SingleSelection
        )


        self.table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )


        self.table.verticalHeader().setVisible(
            False
        )


        self.table.setAlternatingRowColors(
            True
        )


        self.table.setSortingEnabled(
            True
        )


        self.table.verticalHeader().setDefaultSectionSize(
            40
        )


        self.table.itemSelectionChanged.connect(
            self.update_detail_from_selection
        )


        self.board_content_splitter.addWidget(
            self.table
        )


        # ----------------------------------------------------
        # SELECTED GAME CARD
        # ----------------------------------------------------

        self.detail_scroll = QScrollArea()


        self.detail_scroll.setObjectName(
            "detailScroll"
        )


        self.detail_scroll.setWidgetResizable(
            True
        )


        self.detail_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )


        self.detail_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )


        self.detail_scroll.setFrameShape(
            QFrame.NoFrame
        )


        self.detail_scroll.setMinimumWidth(
            360
        )


        self.detail_panel = QFrame()


        self.detail_panel.setObjectName(
            "detailPanel"
        )


        # The selected-game card contains more information than
        # can always fit vertically when the main dashboard is
        # visible. Giving the content a real minimum height lets
        # the QScrollArea scroll instead of clipping the bottom.
        self.detail_panel.setMinimumHeight(
            620
        )


        detail_layout = QVBoxLayout(
            self.detail_panel
        )


        detail_layout.setContentsMargins(
            18,
            16,
            18,
            16
        )


        detail_layout.setSpacing(
            7
        )


        selected_title = QLabel(
            "SELECTED GAME"
        )


        selected_title.setObjectName(
            "sectionTitle"
        )


        self.detail_game = QLabel(
            "Select a game"
        )


        self.detail_game.setObjectName(
            "detailGame"
        )


        self.detail_game.setWordWrap(
            True
        )


        detail_layout.addWidget(
            selected_title
        )


        detail_layout.addWidget(
            self.detail_game
        )


        detail_layout.addSpacing(
            4
        )


        (
            self.detail_model_label,
            self.detail_model
        ) = self.add_metric_block(

            detail_layout,

            "MODEL SPREAD"
        )


        (
            self.detail_hardrock_label,
            self.detail_hardrock
        ) = self.add_metric_block(

            detail_layout,

            "HARD ROCK"
        )


        (
            self.detail_edge_label,
            self.detail_edge
        ) = self.add_metric_block(

            detail_layout,

            "MODEL EDGE"
        )


        (
            self.detail_probability_label,
            self.detail_probability
        ) = self.add_metric_block(

            detail_layout,

            "WIN PROBABILITY"
        )


        (
            self.detail_strength_label,
            self.detail_strength
        ) = self.add_metric_block(

            detail_layout,

            "STRENGTH"
        )


        self.detail_warning_title = QLabel(
            "WARNING"
        )


        self.detail_warning_title.setObjectName(
            "metricLabel"
        )


        self.detail_warning = QLabel(
            ""
        )


        self.detail_warning.setObjectName(
            "detailWarning"
        )


        self.detail_warning.setWordWrap(
            True
        )


        self.detail_warning_title.hide()

        self.detail_warning.hide()


        detail_layout.addWidget(
            self.detail_warning_title
        )


        detail_layout.addWidget(
            self.detail_warning
        )


        detail_layout.addSpacing(
            5
        )


        diagnostics_title = QLabel(
            "MARKET + MODEL DIAGNOSTICS"
        )


        diagnostics_title.setObjectName(
            "sectionTitle"
        )


        self.detail_diagnostics = QLabel(
            ""
        )


        self.detail_diagnostics.setObjectName(
            "detailDiagnostics"
        )


        self.detail_diagnostics.setWordWrap(
            True
        )


        detail_layout.addWidget(
            diagnostics_title
        )


        detail_layout.addWidget(
            self.detail_diagnostics
        )


        detail_layout.addStretch()


        self.detail_scroll.setWidget(
            self.detail_panel
        )


        self.board_content_splitter.addWidget(
            self.detail_scroll
        )


        self.board_content_splitter.setStretchFactor(
            0,
            4
        )


        self.board_content_splitter.setStretchFactor(
            1,
            2
        )


        board_layout.addWidget(
            self.board_content_splitter,
            1
        )


        # ====================================================
        # MAIN SPLITTER CONTENT
        # ====================================================

        self.main_splitter.addWidget(
            self.top_container
        )


        self.main_splitter.addWidget(
            self.board_panel
        )


        self.main_splitter.setStretchFactor(
            0,
            2
        )


        self.main_splitter.setStretchFactor(
            1,
            3
        )


        main_layout.addWidget(
            self.main_splitter,
            1
        )


        self.apply_style()


        QTimer.singleShot(
            100,
            self.set_default_splitter_sizes
        )


        QTimer.singleShot(
            150,
            self.set_default_board_content_sizes
        )


    # ========================================================
    # METRIC BLOCK
    # ========================================================

    def add_metric_block(
        self,
        layout,
        title
    ):

        label = QLabel(
            title
        )


        label.setObjectName(
            "metricLabel"
        )


        value = QLabel(
            "—"
        )


        value.setObjectName(
            "metricValue"
        )


        value.setWordWrap(
            True
        )


        layout.addWidget(
            label
        )


        layout.addWidget(
            value
        )


        return (
            label,
            value
        )


    # ========================================================
    # DEFAULT SPLITTER SIZES
    # ========================================================

    def set_default_splitter_sizes(self):

        total = self.main_splitter.height()


        if total <= 0:

            return


        top_size = int(
            total * 0.40
        )


        self.main_splitter.setSizes(
            [
                top_size,
                total - top_size
            ]
        )


    def set_default_board_content_sizes(self):

        total = (
            self.board_content_splitter.width()
        )


        if total <= 0:

            return


        detail_size = min(
            430,
            int(
                total * 0.34
            )
        )


        table_size = max(
            600,
            total - detail_size
        )


        self.board_content_splitter.setSizes(
            [
                table_size,
                detail_size
            ]
        )


    # ========================================================
    # EXPAND / RESTORE BOARD
    # ========================================================

    def toggle_board_expansion(self):

        if not self.board_expanded:

            self.top_container.hide()


            self.expand_board_button.setText(
                "RESTORE DASHBOARD"
            )


            self.board_expanded = True


            self.main_splitter.setSizes(
                [
                    0,
                    1000
                ]
            )


        else:

            self.top_container.show()


            self.expand_board_button.setText(
                "EXPAND BOARD"
            )


            self.board_expanded = False


            QTimer.singleShot(
                50,
                self.set_default_splitter_sizes
            )


    # ========================================================
    # STYLE
    # ========================================================

    def apply_style(self):

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #040609;
            }

            QWidget {
                color: #e8e9eb;
                font-family: Arial;
            }

            #title {
                font-size: 27px;
                font-weight: 700;
                letter-spacing: 5px;
                color: #f4f4f4;
            }

            #subtitle {
                font-size: 10px;
                letter-spacing: 2px;
                color: #a76a31;
                padding-bottom: 3px;
            }

            #corePanel {
                background-color: #06080c;
                border: 1px solid #171b21;
                border-radius: 14px;
            }

            #controlPanel,
            #boardPanel,
            #detailPanel {
                background-color: #0b0e13;
                border: 1px solid #242931;
                border-radius: 14px;
            }

            #detailPanel {
                background-color: #080b10;
            }

            #detailScroll {
                background-color: transparent;
                border: none;
            }

            #detailScroll > QWidget > QWidget {
                background-color: transparent;
            }

            QScrollBar:vertical {
                background-color: #080b10;
                width: 9px;
                margin: 3px 1px 3px 1px;
                border: none;
            }

            QScrollBar::handle:vertical {
                background-color: #4d2d17;
                min-height: 34px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #8b4b20;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
                border: none;
                height: 0px;
            }

            #sectionTitle {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 2px;
                color: #9ba1aa;
            }

            #status {
                font-size: 18px;
                font-weight: 700;
                color: #ff9d3d;
                padding-top: 3px;
                padding-bottom: 3px;
            }

            #summary {
                color: #d3d5d8;
                font-size: 11px;
            }

            #boardStatus {
                font-size: 10px;
                font-weight: 700;
                color: #e28a39;
                letter-spacing: 1px;
            }

            #searchLabel {
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
                color: #a0a6ae;
            }

            #resultCount {
                min-width: 95px;
                color: #dd8a3b;
                font-size: 10px;
                font-weight: 700;
            }

            #detailGame {
                font-size: 17px;
                font-weight: 700;
                color: #f1f2f3;
                padding-top: 2px;
                padding-bottom: 5px;
            }

            #metricLabel {
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1px;
                color: #777f89;
                margin-top: 2px;
            }

            #metricValue {
                font-size: 16px;
                font-weight: 700;
                color: #f1f1f1;
                padding-bottom: 3px;
            }

            #detailWarning {
                font-size: 11px;
                font-weight: 700;
                color: #e58b3b;
                padding: 7px;
                background-color: #17100b;
                border: 1px solid #5b3518;
                border-radius: 6px;
            }

            #detailDiagnostics {
                color: #aeb4bc;
                font-family: Menlo;
                font-size: 9px;
                line-height: 1.25;
            }

            QLineEdit {
                min-height: 36px;
                background-color: #06080b;
                border: 1px solid #303741;
                border-radius: 7px;
                padding-left: 11px;
                padding-right: 11px;
                color: #eeeeee;
                font-size: 13px;
            }

            QLineEdit:focus {
                border: 1px solid #d87524;
            }

            QPushButton {
                min-height: 36px;
                background-color: #171b22;
                border: 1px solid #343a43;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #222832;
                border: 1px solid #59616d;
            }

            QPushButton:disabled {
                background-color: #101217;
                color: #515761;
                border: 1px solid #20242a;
            }

            #runButton {
                min-height: 46px;
                background-color: #b9500f;
                border: 1px solid #ee8b39;
                font-size: 13px;
                letter-spacing: 1px;
            }

            #runButton:hover {
                background-color: #d76317;
            }

            #expandButton {
                min-height: 30px;
                max-height: 30px;
                padding-left: 14px;
                padding-right: 14px;
                color: #ff9d3d;
                background-color: #161a20;
                border: 1px solid #68401f;
                font-size: 10px;
            }

            #expandButton:hover {
                background-color: #2a1b10;
                border: 1px solid #c87328;
            }

            QProgressBar {
                min-height: 7px;
                max-height: 7px;
                background-color: #171b20;
                border: none;
                border-radius: 3px;
            }

            QProgressBar::chunk {
                background-color: #e47a26;
                border-radius: 3px;
            }

            QTextEdit {
                background-color: #06080b;
                border: 1px solid #252a31;
                border-radius: 7px;
                padding: 6px;
                color: #b4b9c0;
                font-family: Menlo;
                font-size: 8px;
            }

            QTableWidget {
                background-color: #07090d;
                alternate-background-color: #0b0e13;
                border: 1px solid #222831;
                border-radius: 8px;
                gridline-color: #191e25;
                font-size: 12px;
                selection-background-color: #40230f;
                selection-color: #ffffff;
            }

            QTableWidget::item {
                padding-left: 7px;
                padding-right: 7px;
            }

            QHeaderView::section {
                background-color: #10141a;
                color: #9da4ad;
                border: none;
                border-right: 1px solid #252a31;
                border-bottom: 1px solid #252a31;
                padding: 10px;
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1px;
            }

            QSplitter::handle {
                background-color: #11151a;
            }

            QSplitter::handle:hover {
                background-color: #7d461d;
            }

            #researchMode {
                color: #e39a4d;
                font-weight: 700;
                font-size: 10px;
                letter-spacing: 1px;
            }

            #warning {
                color: #757c85;
                font-size: 8px;
            }
            """
        )


    # ========================================================
    # BOARD HELPERS
    # ========================================================

    def signal_key(
        self,
        signal
    ):

        game_id = signal.get(
            "game_id"
        )


        if game_id not in (
            None,
            ""
        ):

            return str(
                game_id
            )


        return (

            f"{signal.get('away_team', '')}"
            f"|"
            f"{signal.get('home_team', '')}"
            f"|"
            f"{signal.get('start_date', '')}"
        )


    def format_probability(
        self,
        value
    ):

        try:

            probability = float(
                value
            )

        except Exception:

            return "N/A"


        if math.isnan(
            probability
        ):

            return "N/A"


        if probability <= 1.0:

            probability *= 100.0


        return (
            f"{probability:.1f}%"
        )


    def format_optional_number(
        self,
        value,
        decimals=1,
        suffix=""
    ):

        try:

            number = float(
                value
            )

        except Exception:

            return "N/A"


        if math.isnan(
            number
        ):

            return "N/A"


        return (
            f"{number:.{decimals}f}"
            f"{suffix}"
        )


    def clean_warning_text(
        self,
        warning
    ):

        warning = str(
            warning
            or
            ""
        ).strip()


        if warning in (
            "",
            "NONE",
            "nan"
        ):

            return ""


        pieces = [

            piece
            .strip()
            .replace(
                "_",
                " "
            )

            for piece
            in warning.split(
                "|"
            )

            if piece.strip()
        ]


        return (
            "  •  ".join(
                pieces
            )
        )


    # ========================================================
    # LOAD BOARD
    # ========================================================

    def load_board(self):

        if not BOARD_FILE.exists():

            self.summary_label.setText(

                "No live model/market board found.\n"
                "Run the V3 board builder first."
            )


            self.table.setRowCount(
                0
            )


            self.board_status.setText(
                "BOARD UNAVAILABLE"
            )


            return


        try:

            with open(
                BOARD_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                board = json.load(
                    file
                )


        except Exception as error:

            self.summary_label.setText(

                "Could not load live board:\n"
                f"{error}"
            )


            self.board_status.setText(
                "BOARD ERROR"
            )


            return


        if isinstance(
            board,
            dict
        ):

            if isinstance(
                board.get(
                    "records"
                ),
                list
            ):

                records = board[
                    "records"
                ]


            elif isinstance(
                board.get(
                    "signals"
                ),
                list
            ):

                records = board[
                    "signals"
                ]


            else:

                records = []


        elif isinstance(
            board,
            list
        ):

            records = board


        else:

            records = []


        self.board_records = records


        self.board_signals = [

            record

            for record in records

            if safe_int_local(
                record.get(
                    "game_is_actionable"
                )
            )
            ==
            1
        ]


        self.signal_by_key = {

            self.signal_key(
                signal
            ):
                signal

            for signal in self.board_signals
        }


        high = sum(

            1

            for signal in self.board_signals

            if str(
                signal.get(
                    "strength_label"
                )
            ).upper()
            ==
            "HIGH"
        )


        medium = sum(

            1

            for signal in self.board_signals

            if str(
                signal.get(
                    "strength_label"
                )
            ).upper()
            ==
            "MEDIUM"
        )


        low = sum(

            1

            for signal in self.board_signals

            if str(
                signal.get(
                    "strength_label"
                )
            ).upper()
            ==
            "LOW"
        )


        na_count = sum(

            1

            for signal in self.board_signals

            if str(
                signal.get(
                    "strength_label"
                )
            ).upper()
            ==
            "N/A"
        )


        executable = sum(

            1

            for signal in self.board_signals

            if safe_int_local(
                signal.get(
                    "hardrock_available"
                )
            )
            ==
            1
        )


        outliers = sum(

            1

            for signal in self.board_signals

            if "MODEL_MARKET_OUTLIER"
            in
            str(
                signal.get(
                    "signal_warning",
                    ""
                )
            )
        )


        self.summary_label.setText(

            f"{len(self.board_signals)} PREGAME GAMES\n"
            f"{executable} HARD ROCK HEADLINES\n"
            f"{high} HIGH  •  "
            f"{medium} MEDIUM  •  "
            f"{low} LOW  •  "
            f"{na_count} N/A\n"
            f"{outliers} MODEL/MARKET OUTLIERS"
        )


        self.board_status.setText(
            "LIVE • PROSPECTIVE VALIDATION"
        )


        self.populate_table(
            self.board_signals
        )


    # ========================================================
    # SORT VALUE HELPER
    # ========================================================

    def numeric_sort_value(
        self,
        value
    ):

        try:

            if value is None:

                return -1000000000.0


            number = float(
                value
            )


            if math.isnan(
                number
            ):

                return -1000000000.0


            return number


        except Exception:

            return -1000000000.0


    # ========================================================
    # TABLE
    # ========================================================

    def populate_table(
        self,
        signals
    ):

        self.table.blockSignals(
            True
        )


        self.table.setSortingEnabled(
            False
        )


        self.table.setRowCount(
            len(
                signals
            )
        )


        for row_index, signal in enumerate(
            signals
        ):

            game = (

                f"{signal.get('away_team', '')} "
                f"@ "
                f"{signal.get('home_team', '')}"
            )


            model_spread = signal.get(
                "model_spread_display",
                "N/A"
            )


            hardrock = signal.get(
                "hardrock_display",
                "Unavailable"
            )


            model_edge = signal.get(
                "model_edge_display",
                "N/A"
            )


            win_probability = self.format_probability(

                signal.get(
                    "model_favorite_win_probability"
                )
            )


            strength = str(

                signal.get(
                    "strength_label",
                    "N/A"
                )
            ).upper()


            values = [

                game,

                model_spread,

                hardrock,

                model_edge,

                win_probability,

                strength
            ]


            # Separate machine-sort values from displayed text.
            # In particular, Strength must sort numerically as:
            # HIGH > MEDIUM > LOW > N/A.
            strength_rank = {
                "N/A": 0,
                "LOW": 1,
                "MEDIUM": 2,
                "HIGH": 3
            }.get(
                strength,
                0
            )


            model_spread_sort = self.numeric_sort_value(
                signal.get(
                    "model_spread"
                )
            )


            hardrock_spread_sort = self.numeric_sort_value(
                signal.get(
                    "hardrock_spread"
                )
            )


            model_edge_sort = self.numeric_sort_value(
                signal.get(
                    "model_edge_points"
                )
            )


            win_probability_sort = self.numeric_sort_value(
                signal.get(
                    "model_favorite_win_probability"
                )
            )


            sort_values = [
                game.lower(),
                model_spread_sort,
                hardrock_spread_sort,
                model_edge_sort,
                win_probability_sort,
                strength_rank
            ]


            key = self.signal_key(
                signal
            )


            warning = str(

                signal.get(
                    "signal_warning",
                    ""
                )
            )


            for column_index, value in enumerate(
                values
            ):

                item = SortableTableWidgetItem(
                    str(
                        value
                    )
                )


                item.setData(
                    Qt.UserRole,
                    key
                )


                item.setData(
                    Qt.UserRole + 1,
                    sort_values[
                        column_index
                    ]
                )


                if column_index in (
                    4,
                    5
                ):

                    item.setTextAlignment(
                        Qt.AlignCenter
                    )


                if column_index == 5:

                    if strength == "HIGH":

                        item.setForeground(
                            QColor(
                                255,
                                167,
                                66
                            )
                        )


                    elif strength == "MEDIUM":

                        item.setForeground(
                            QColor(
                                220,
                                170,
                                112
                            )
                        )


                    elif strength == "LOW":

                        item.setForeground(
                            QColor(
                                176,
                                181,
                                188
                            )
                        )


                    else:

                        item.setForeground(
                            QColor(
                                103,
                                110,
                                120
                            )
                        )


                if (
                    column_index == 0
                    and
                    warning not in (
                        "",
                        "NONE"
                    )
                ):

                    item.setForeground(
                        QColor(
                            230,
                            144,
                            67
                        )
                    )


                self.table.setItem(

                    row_index,

                    column_index,

                    item
                )


        self.table.setSortingEnabled(
            True
        )


        self.table.blockSignals(
            False
        )


        self.result_count_label.setText(

            f"{len(signals)} SHOWN"
        )


        if len(
            signals
        ) > 0:

            self.table.setCurrentCell(
                0,
                0
            )


            self.update_detail_from_selection()


        else:

            self.clear_detail()


    # ========================================================
    # SELECTED GAME DETAIL
    # ========================================================

    def update_detail_from_selection(self):

        selected_items = self.table.selectedItems()


        if not selected_items:

            self.clear_detail()

            return


        row = selected_items[
            0
        ].row()


        first_item = self.table.item(
            row,
            0
        )


        if first_item is None:

            self.clear_detail()

            return


        key = first_item.data(
            Qt.UserRole
        )


        signal = self.signal_by_key.get(
            str(
                key
            )
        )


        if signal is None:

            self.clear_detail()

            return


        self.set_detail_signal(
            signal
        )


    def clear_detail(self):

        self.detail_game.setText(
            "Select a game"
        )


        self.detail_model.setText(
            "—"
        )


        self.detail_hardrock.setText(
            "—"
        )


        self.detail_edge.setText(
            "—"
        )


        self.detail_probability.setText(
            "—"
        )


        self.detail_strength.setText(
            "—"
        )


        self.detail_warning_title.hide()

        self.detail_warning.hide()


        self.detail_diagnostics.setText(
            ""
        )


    def set_detail_signal(
        self,
        signal
    ):

        game = (

            f"{signal.get('away_team', '')} "
            f"@ "
            f"{signal.get('home_team', '')}"
        )


        model_spread = str(

            signal.get(
                "model_spread_display",
                "N/A"
            )
        )


        hardrock = str(

            signal.get(
                "hardrock_display",
                "Unavailable"
            )
        )


        model_edge = str(

            signal.get(
                "model_edge_display",
                "N/A"
            )
        )


        win_probability = self.format_probability(

            signal.get(
                "model_favorite_win_probability"
            )
        )


        strength = str(

            signal.get(
                "strength_label",
                "N/A"
            )
        ).upper()


        warning_raw = str(

            signal.get(
                "signal_warning",
                ""
            )
        )


        warning_clean = self.clean_warning_text(
            warning_raw
        )


        self.detail_game.setText(
            game
        )


        self.detail_model.setText(
            model_spread
        )


        self.detail_hardrock.setText(
            hardrock
        )


        self.detail_edge.setText(
            model_edge
        )


        self.detail_probability.setText(
            win_probability
        )


        self.detail_strength.setText(
            strength
        )


        if strength == "HIGH":

            self.detail_strength.setStyleSheet(
                "color: #ffa742;"
            )


        elif strength == "MEDIUM":

            self.detail_strength.setStyleSheet(
                "color: #dca970;"
            )


        elif strength == "LOW":

            self.detail_strength.setStyleSheet(
                "color: #b4b9c0;"
            )


        else:

            self.detail_strength.setStyleSheet(
                "color: #6c737d;"
            )


        if warning_clean:

            self.detail_warning.setText(
                warning_clean
            )


            self.detail_warning_title.show()

            self.detail_warning.show()


        else:

            self.detail_warning_title.hide()

            self.detail_warning.hide()


        reference_display = str(

            signal.get(
                "reference_market_display",
                "Unavailable"
            )
        )


        reference_books = safe_int_local(

            signal.get(
                "reference_market_books"
            )
        )


        hr_vs_consensus = str(

            signal.get(
                "hardrock_vs_consensus_display",
                "N/A"
            )
        )


        model_market_gap = self.format_optional_number(

            signal.get(
                "model_vs_consensus_abs_points"
            ),

            decimals=1,

            suffix=" pts"
        )


        input_status = str(

            signal.get(
                "nn_input_status",
                "N/A"
            )
        )


        margin_sd = self.format_optional_number(

            signal.get(
                "nn_margin_std"
            ),

            decimals=2
        )


        probability_sd = self.format_optional_number(

            signal.get(
                "nn_win_probability_std"
            ),

            decimals=3
        )


        heads_agree = (

            "YES"

            if safe_int_local(
                signal.get(
                    "nn_heads_agree"
                )
            )
            ==
            1

            else "NO"
        )


        strength_score = self.format_optional_number(

            signal.get(
                "strength_score"
            ),

            decimals=1,

            suffix="/100"
        )


        hardrock_quality = str(

            signal.get(
                "hardrock_spread_quality_status",
                "N/A"
            )
        )


        diagnostics = (

            f"Reference: {reference_display} "
            f"| {reference_books} books\n"

            f"HR vs consensus: "
            f"{hr_vs_consensus}\n"

            f"Model vs consensus gap: "
            f"{model_market_gap}\n"

            f"Hard Rock quality: "
            f"{hardrock_quality}\n"

            f"Input: "
            f"{input_status}\n"

            f"Ensemble margin SD: "
            f"{margin_sd}\n"

            f"Win probability SD: "
            f"{probability_sd}\n"

            f"Heads agree: "
            f"{heads_agree}\n"

            f"Strength score: "
            f"{strength_score}"
        )


        self.detail_diagnostics.setText(
            diagnostics
        )


        # Every newly selected game opens at the top of the card.
        # The user can scroll down for diagnostics when needed.
        self.detail_scroll.verticalScrollBar().setValue(
            0
        )


        self.core.activate_game(

            game,

            strength,

            warning_raw
        )


    # ========================================================
    # SEARCH / FILTER
    # ========================================================

    def filter_board(
        self,
        search_text
    ):

        query = (
            search_text
            .strip()
            .lower()
        )


        if not query:

            self.populate_table(
                self.board_signals
            )

            return


        filtered = []


        for signal in self.board_signals:

            searchable_values = [

                signal.get(
                    "away_team",
                    ""
                ),

                signal.get(
                    "home_team",
                    ""
                ),

                signal.get(
                    "model_favorite",
                    ""
                ),

                signal.get(
                    "model_spread_display",
                    ""
                ),

                signal.get(
                    "hardrock_display",
                    ""
                ),

                signal.get(
                    "model_edge_display",
                    ""
                ),

                signal.get(
                    "strength_label",
                    ""
                ),

                signal.get(
                    "signal_warning",
                    ""
                ),

                signal.get(
                    "reference_market_display",
                    ""
                ),

                signal.get(
                    "nn_input_status",
                    ""
                )
            ]


            searchable_text = (

                " "
                .join(

                    str(
                        value
                    )

                    for value
                    in searchable_values
                )
                .lower()
            )


            if query in searchable_text:

                filtered.append(
                    signal
                )


        self.populate_table(
            filtered
        )


    # ========================================================
    # PIPELINE RUNNERS
    # ========================================================

    def run_dry_test(self):

        self.start_pipeline(
            dry_run=True
        )


    def run_game_day(self):

        self.start_pipeline(
            dry_run=False
        )


    def start_pipeline(
        self,
        dry_run
    ):

        if (
            self.pipeline_thread
            is not None

            and

            self.pipeline_thread.isRunning()
        ):

            return


        self.log_box.clear()


        self.run_button.setEnabled(
            False
        )


        self.dry_button.setEnabled(
            False
        )


        self.status_label.setText(
            "INITIALIZING..."
        )


        self.progress_bar.setValue(
            2
        )


        self.core.set_stage(
            "INITIALIZING",
            2
        )


        self.pipeline_thread = PipelineThread(
            dry_run=dry_run
        )


        self.pipeline_thread.output_received.connect(
            self.append_log
        )


        self.pipeline_thread.stage_changed.connect(
            self.change_stage
        )


        self.pipeline_thread.completed.connect(
            self.pipeline_finished
        )


        self.pipeline_thread.start()


    # ========================================================
    # PIPELINE EVENTS
    # ========================================================

    def append_log(
        self,
        line
    ):

        if line.strip():

            self.log_box.append(
                line
            )


            scrollbar = (
                self.log_box
                .verticalScrollBar()
            )


            scrollbar.setValue(
                scrollbar.maximum()
            )


    def change_stage(
        self,
        stage,
        progress
    ):

        self.status_label.setText(
            stage
        )


        self.progress_bar.setValue(
            progress
        )


        self.core.set_stage(
            stage,
            progress
        )


    def pipeline_finished(
        self,
        success
    ):

        self.run_button.setEnabled(
            True
        )


        self.dry_button.setEnabled(
            True
        )


        if success:

            self.status_label.setText(
                "SYSTEM COMPLETE"
            )


            self.progress_bar.setValue(
                100
            )


            self.core.set_stage(
                "SYSTEM COMPLETE",
                100
            )


            self.load_board()


            self.filter_board(
                self.search_box.text()
            )


        else:

            self.status_label.setText(
                "SYSTEM ERROR"
            )


            self.progress_bar.setValue(
                0
            )


            self.core.set_stage(
                "SYSTEM ERROR",
                0
            )


# ============================================================
# LOCAL SAFE INTEGER
# ============================================================

def safe_int_local(
    value,
    default=0
):

    try:

        if value is None:

            return default


        return int(
            float(
                value
            )
        )


    except Exception:

        return default


# ============================================================
# APPLICATION
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )


    app.setApplicationName(
        "CFB Edge Engine"
    )


    window = MainWindow()


    window.show()


    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()
