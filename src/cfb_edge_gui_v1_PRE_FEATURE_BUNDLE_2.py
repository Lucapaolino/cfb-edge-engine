import json
import math
import random
import sqlite3
import subprocess
import sys

from datetime import datetime, timezone
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
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
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


HISTORY_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "prediction_history.sqlite"
)


WAGER_DB = (
    PROJECT_ROOT
    / "data"
    / "prospective"
    / "wager_ledger.sqlite"
)


WHAT_CHANGED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "what_changed_2026_v1.json"
)


AUDIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_audit_2026_v1.json"
)


HEALTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data_health_2026_v1.json"
)


WHAT_CHANGED_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "what_changed_v1.py"
)


AUDIT_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "prediction_audit_v1.py"
)


HEALTH_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "data_health_v1.py"
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


# ============================================================
# PREDICTION HISTORY WINDOW
#
# Read-only viewer for the immutable prospective prediction
# database. This screen never calls an API and never mutates
# prediction evidence. It shows scored historical predictions,
# the captured Hard Rock line, the actual result, model winner and
# spread results, plus the outright result of the Hard Rock favorite.
# ============================================================

class PredictionHistoryWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.all_scored_rows = []
        self.display_rows = []
        self.row_by_id = {}

        self.setWindowTitle("CFB Edge Engine — Prediction History")
        self.resize(1500, 900)
        self.setMinimumSize(1050, 680)

        self.build_interface()
        self.reload_data()

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel("MODEL HISTORY — PROSPECTIVE RECORD")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "IMMUTABLE PRE-KICKOFF SNAPSHOTS  •  ACTUAL RESULTS  •  LOCAL DATA ONLY"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("controlPanel")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(14, 10, 14, 10)
        controls_layout.setSpacing(10)

        week_label = QLabel("WEEK")
        week_label.setObjectName("searchLabel")
        self.week_combo = QComboBox()
        self.week_combo.setMinimumWidth(150)
        self.week_combo.currentIndexChanged.connect(self.refresh_view)

        policy_label = QLabel("SNAPSHOT")
        policy_label.setObjectName("searchLabel")
        self.policy_combo = QComboBox()
        self.policy_combo.addItem("LATEST", "latest")
        self.policy_combo.addItem("FIRST", "first")
        self.policy_combo.addItem("ALL", "all")
        self.policy_combo.setMinimumWidth(130)
        self.policy_combo.currentIndexChanged.connect(self.refresh_view)

        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search previous games or teams...")
        self.history_search.setClearButtonEnabled(True)
        self.history_search.textChanged.connect(self.refresh_view)

        self.refresh_button = QPushButton("REFRESH HISTORY")
        self.refresh_button.clicked.connect(self.reload_data)

        controls_layout.addWidget(week_label)
        controls_layout.addWidget(self.week_combo)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(policy_label)
        controls_layout.addWidget(self.policy_combo)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self.history_search, 1)
        controls_layout.addWidget(self.refresh_button)

        layout.addWidget(controls)

        self.history_summary = QLabel("Loading prediction history...")
        self.history_summary.setObjectName("summary")
        self.history_summary.setWordWrap(True)
        layout.addWidget(self.history_summary)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels(
            [
                "GAME",
                "MODEL SPREAD",
                "HARD ROCK",
                "FINAL",
                "MODEL WINNER",
                "SPREAD RESULT",
                "HR FAVORITE RESULT",
                "STRENGTH",
            ]
        )
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SingleSelection)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSortingEnabled(True)
        self.history_table.verticalHeader().setDefaultSectionSize(40)
        self.history_table.itemSelectionChanged.connect(self.update_history_detail)
        splitter.addWidget(self.history_table)

        self.history_detail_scroll = QScrollArea()
        self.history_detail_scroll.setWidgetResizable(True)
        self.history_detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_detail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.history_detail_scroll.setFrameShape(QFrame.NoFrame)
        self.history_detail_scroll.setMinimumWidth(380)

        detail_panel = QFrame()
        detail_panel.setObjectName("detailPanel")
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(18, 16, 18, 16)
        detail_layout.setSpacing(7)

        detail_title = QLabel("HISTORICAL GAME")
        detail_title.setObjectName("sectionTitle")
        detail_layout.addWidget(detail_title)

        self.history_game = QLabel("Select a completed game")
        self.history_game.setObjectName("detailGame")
        self.history_game.setWordWrap(True)
        detail_layout.addWidget(self.history_game)
        detail_layout.addSpacing(4)

        self.history_prediction = self._history_metric(detail_layout, "PREDICTION MADE")
        self.history_model = self._history_metric(detail_layout, "MODEL SPREAD")
        self.history_hr = self._history_metric(detail_layout, "HARD ROCK AT CAPTURE")
        self.history_hr_favorite = self._history_metric(detail_layout, "HARD ROCK FAVORITE")
        self.history_edge = self._history_metric(detail_layout, "MODEL EDGE")
        self.history_prob = self._history_metric(detail_layout, "WIN PROBABILITY")
        self.history_strength = self._history_metric(detail_layout, "STRENGTH")

        result_title = QLabel("ACTUAL RESULT")
        result_title.setObjectName("sectionTitle")
        detail_layout.addSpacing(7)
        detail_layout.addWidget(result_title)

        self.history_final = QLabel("—")
        self.history_final.setObjectName("metricValue")
        self.history_final.setWordWrap(True)
        detail_layout.addWidget(self.history_final)

        self.history_results = QLabel("")
        self.history_results.setObjectName("detailDiagnostics")
        self.history_results.setWordWrap(True)
        detail_layout.addWidget(self.history_results)

        diagnostics_title = QLabel("CAPTURED DIAGNOSTICS")
        diagnostics_title.setObjectName("sectionTitle")
        detail_layout.addSpacing(7)
        detail_layout.addWidget(diagnostics_title)

        self.history_diagnostics = QLabel("")
        self.history_diagnostics.setObjectName("detailDiagnostics")
        self.history_diagnostics.setWordWrap(True)
        detail_layout.addWidget(self.history_diagnostics)

        note = QLabel(
            "SPREAD RESULT is diagnostic only: it grades the model side against the captured Hard Rock line. "
            "HR FAVORITE RESULT grades only whether Hard Rock's favorite won outright. Neither is an audited "
            "record of wagers actually placed."
        )
        note.setObjectName("warning")
        note.setWordWrap(True)
        detail_layout.addWidget(note)
        detail_layout.addStretch()

        self.history_detail_scroll.setWidget(detail_panel)
        splitter.addWidget(self.history_detail_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1000, 440])

        layout.addWidget(splitter, 1)

    def _history_metric(self, layout, title):
        label = QLabel(title)
        label.setObjectName("metricLabel")
        value = QLabel("—")
        value.setObjectName("metricValue")
        value.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(value)
        return value

    @staticmethod
    def _safe_float(value):
        try:
            if value is None or value == "":
                return None
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return None
            return number
        except Exception:
            return None

    @staticmethod
    def _safe_int(value):
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except Exception:
            return None

    @staticmethod
    def _number(value, decimals=1):
        number = PredictionHistoryWindow._safe_float(value)
        if number is None:
            return "N/A"
        return f"{number:.{decimals}f}"

    @staticmethod
    def _points_text(value):
        number = PredictionHistoryWindow._safe_float(value)
        if number is None:
            return "N/A"
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.1f}"

    @staticmethod
    def _probability_text(value):
        number = PredictionHistoryWindow._safe_float(value)
        if number is None:
            return "N/A"
        if number <= 1.0:
            number *= 100.0
        return f"{number:.1f}%"

    @staticmethod
    def _clean_warning(value):
        text = str(value or "NONE").strip()
        if text.upper() in ("", "NONE", "NAN"):
            return "NONE"
        return "  •  ".join(
            part.strip().replace("_", " ")
            for part in text.split("|")
            if part.strip()
        )

    @staticmethod
    def _team_equals(left, right):
        return str(left or "").strip().casefold() == str(right or "").strip().casefold()

    def _hardrock_favorite_info(self, row):
        """Return (favorite_team, outright_result) from the captured Hard Rock spread.

        hardrock_spread_for_model_favorite is stored from the perspective of the
        model favorite. A negative number means Hard Rock also favored the model
        favorite; a positive number means Hard Rock favored the opponent. A true
        pick'em or missing/unusable headline has no Hard Rock favorite.
        """
        spread_for_model_favorite = self._safe_float(
            row.get("hardrock_spread_for_model_favorite")
        )
        if spread_for_model_favorite is None or math.isclose(
            spread_for_model_favorite,
            0.0,
            abs_tol=1e-9,
        ):
            return None, None

        model_favorite = str(row.get("model_favorite") or "").strip()
        home_team = str(row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or "").strip()

        if self._team_equals(model_favorite, home_team):
            opponent = away_team
        elif self._team_equals(model_favorite, away_team):
            opponent = home_team
        else:
            return None, None

        hardrock_favorite = (
            model_favorite
            if spread_for_model_favorite < 0
            else opponent
        )

        actual_winner = str(row.get("actual_winner") or "").strip()
        if not actual_winner:
            return hardrock_favorite, None

        result = (
            "W"
            if self._team_equals(hardrock_favorite, actual_winner)
            else "L"
        )
        return hardrock_favorite, result

    @staticmethod
    def _row_identifier(row):
        return str(row.get("snapshot_id") or "")

    def _connect_read_only(self):
        # Open read-only so the history viewer itself can never mutate evidence.
        return sqlite3.connect(
            f"file:{HISTORY_DB}?mode=ro",
            uri=True,
        )

    def reload_data(self):
        self.all_scored_rows = []
        self.row_by_id = {}

        if not HISTORY_DB.exists():
            self.history_summary.setText(
                "Prediction-history database not found. Run GAME DAY successfully to begin collecting prospective records."
            )
            self.history_table.setRowCount(0)
            return

        try:
            connection = self._connect_read_only()
            connection.row_factory = sqlite3.Row
            query = """
                SELECT
                    p.*,
                    s.actual_home_margin,
                    s.predicted_home_margin,
                    s.margin_error,
                    s.absolute_margin_error,
                    s.squared_margin_error,
                    s.winner_correct,
                    s.brier_score,
                    s.hardrock_implied_home_margin,
                    s.hardrock_absolute_margin_error,
                    s.model_side_ats_result,
                    s.model_side_flat_units,
                    r.away_points,
                    r.home_points,
                    r.winner AS actual_winner
                FROM prediction_snapshots p
                JOIN prediction_scores s ON s.snapshot_id = p.snapshot_id
                JOIN game_results r ON r.game_key = p.game_key
                ORDER BY p.kickoff_utc, p.captured_utc, p.snapshot_id
            """
            self.all_scored_rows = [dict(row) for row in connection.execute(query).fetchall()]
            connection.close()
        except Exception as error:
            self.history_summary.setText(f"Could not read prediction history: {error}")
            self.history_table.setRowCount(0)
            return

        weeks = sorted(
            {
                self._safe_int(row.get("week"))
                for row in self.all_scored_rows
                if self._safe_int(row.get("week")) is not None
            }
        )

        old_week = self.week_combo.currentData()
        self.week_combo.blockSignals(True)
        self.week_combo.clear()
        self.week_combo.addItem("ALL WEEKS", None)
        for week in weeks:
            self.week_combo.addItem(f"WEEK {week}", week)

        if old_week in weeks:
            index = self.week_combo.findData(old_week)
            self.week_combo.setCurrentIndex(index)
        elif weeks:
            # Default to the most recently completed/scored week.
            index = self.week_combo.findData(max(weeks))
            self.week_combo.setCurrentIndex(index)
        else:
            self.week_combo.setCurrentIndex(0)
        self.week_combo.blockSignals(False)

        self.refresh_view()

    def _policy_rows(self):
        policy = self.policy_combo.currentData() or "latest"
        rows = list(self.all_scored_rows)
        if policy == "all":
            return rows

        grouped = {}
        for row in rows:
            grouped.setdefault(str(row.get("game_key") or ""), []).append(row)

        selected = []
        for game_rows in grouped.values():
            ordered = sorted(
                game_rows,
                key=lambda row: (
                    str(row.get("captured_utc") or ""),
                    str(row.get("snapshot_id") or ""),
                ),
            )
            selected.append(ordered[0] if policy == "first" else ordered[-1])

        return sorted(
            selected,
            key=lambda row: (
                str(row.get("kickoff_utc") or ""),
                str(row.get("game_key") or ""),
            ),
        )

    def refresh_view(self, *args):
        rows = self._policy_rows()
        selected_week = self.week_combo.currentData()
        if selected_week is not None:
            rows = [row for row in rows if self._safe_int(row.get("week")) == int(selected_week)]

        query = self.history_search.text().strip().lower()
        if query:
            rows = [
                row
                for row in rows
                if query
                in " ".join(
                    str(row.get(key) or "")
                    for key in (
                        "away_team",
                        "home_team",
                        "model_favorite",
                        "model_spread_display",
                        "hardrock_display",
                        "strength_label",
                        "signal_warning",
                    )
                ).lower()
            ]

        self.display_rows = rows
        self.row_by_id = {
            self._row_identifier(row): row
            for row in rows
            if self._row_identifier(row)
        }
        self.populate_history_table(rows)
        self.update_history_summary(rows)

    def update_history_summary(self, rows):
        winner_values = [
            self._safe_int(row.get("winner_correct"))
            for row in rows
            if self._safe_int(row.get("winner_correct")) is not None
        ]
        winner_wins = sum(1 for value in winner_values if value == 1)
        winner_losses = sum(1 for value in winner_values if value == 0)
        accuracy = winner_wins / len(winner_values) if winner_values else None

        abs_errors = [
            self._safe_float(row.get("absolute_margin_error"))
            for row in rows
            if self._safe_float(row.get("absolute_margin_error")) is not None
        ]
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else None

        briers = [
            self._safe_float(row.get("brier_score"))
            for row in rows
            if self._safe_float(row.get("brier_score")) is not None
        ]
        brier = sum(briers) / len(briers) if briers else None

        spread_results = [
            str(row.get("model_side_ats_result") or "").upper()
            for row in rows
        ]
        spread_w = sum(1 for value in spread_results if value == "W")
        spread_l = sum(1 for value in spread_results if value == "L")
        spread_p = sum(1 for value in spread_results if value == "P")

        hardrock_results = [
            self._hardrock_favorite_info(row)[1]
            for row in rows
        ]
        hardrock_w = sum(1 for value in hardrock_results if value == "W")
        hardrock_l = sum(1 for value in hardrock_results if value == "L")

        units_values = [
            self._safe_float(row.get("model_side_flat_units"))
            for row in rows
            if self._safe_float(row.get("model_side_flat_units")) is not None
        ]
        units = sum(units_values) if units_values else None

        policy = str(self.policy_combo.currentText() or "LATEST")
        week = self.week_combo.currentText() or "ALL WEEKS"
        unit_text = "N/A" if units is None else f"{units:+.2f}u"
        acc_text = "N/A" if accuracy is None else f"{accuracy * 100:.1f}%"
        mae_text = "N/A" if mae is None else f"{mae:.2f} pts"
        brier_text = "N/A" if brier is None else f"{brier:.3f}"
        hardrock_record = (
            "N/A"
            if hardrock_w + hardrock_l == 0
            else f"{hardrock_w}-{hardrock_l}"
        )

        self.history_summary.setText(
            f"{week}  •  {policy} SNAPSHOT POLICY  •  {len(rows)} SCORED RECORDS\n"
            f"MODEL WINNER {winner_wins}-{winner_losses}  •  ACC {acc_text}  •  "
            f"HR FAVORITE {hardrock_record}  •  SPREAD RESULT {spread_w}-{spread_l}-{spread_p}\n"
            f"SPREAD MAE {mae_text}  •  BRIER {brier_text}  •  FLAT DIAGNOSTIC {unit_text}"
        )

    def populate_history_table(self, rows):
        table = self.history_table
        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            game = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
            model_spread = str(row.get("model_spread_display") or "N/A")
            hardrock = str(row.get("hardrock_display") or "Unavailable")

            away_points = self._safe_float(row.get("away_points"))
            home_points = self._safe_float(row.get("home_points"))
            if away_points is None or home_points is None:
                final = "N/A"
            else:
                final = (
                    f"{row.get('away_team', '')} {self._points_text(away_points)} — "
                    f"{row.get('home_team', '')} {self._points_text(home_points)}"
                )

            winner_correct = self._safe_int(row.get("winner_correct"))
            if winner_correct == 1:
                winner_display = "✓ CORRECT"
                winner_rank = 2
            elif winner_correct == 0:
                winner_display = "✕ WRONG"
                winner_rank = 1
            else:
                winner_display = "N/A"
                winner_rank = 0

            spread_result = str(row.get("model_side_ats_result") or "N/A").upper()
            spread_display = {"W": "✓ WIN", "L": "✕ LOSS", "P": "PUSH"}.get(
                spread_result,
                "N/A",
            )
            spread_rank = {"W": 3, "P": 2, "L": 1}.get(spread_result, 0)

            hardrock_favorite, hardrock_result = self._hardrock_favorite_info(row)
            hardrock_result_display = {
                "W": "✓ WIN",
                "L": "✕ LOSS",
            }.get(hardrock_result, "N/A")
            hardrock_result_rank = {"W": 2, "L": 1}.get(hardrock_result, 0)

            strength = str(row.get("strength_label") or "N/A").upper()
            strength_rank = {"N/A": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(strength, 0)

            model_margin = self._safe_float(row.get("model_home_margin"))
            values = [
                game,
                model_spread,
                hardrock,
                final,
                winner_display,
                spread_display,
                hardrock_result_display,
                strength,
            ]
            sort_values = [
                game.lower(),
                model_margin if model_margin is not None else -9999.0,
                self._safe_float(row.get("hardrock_spread_for_model_favorite")) or -9999.0,
                str(row.get("kickoff_utc") or ""),
                winner_rank,
                spread_rank,
                hardrock_result_rank,
                strength_rank,
            ]
            snapshot_id = self._row_identifier(row)

            for column_index, value in enumerate(values):
                item = SortableTableWidgetItem(str(value))
                item.setData(Qt.UserRole, snapshot_id)
                item.setData(Qt.UserRole + 1, sort_values[column_index])

                if column_index in (4, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignCenter)

                if column_index == 4:
                    if winner_correct == 1:
                        item.setForeground(QColor(120, 210, 160))
                    elif winner_correct == 0:
                        item.setForeground(QColor(235, 110, 100))

                if column_index == 6:
                    if spread_result == "W":
                        item.setForeground(QColor(120, 210, 160))
                    elif spread_result == "L":
                        item.setForeground(QColor(235, 110, 100))
                    elif spread_result == "P":
                        item.setForeground(QColor(210, 180, 110))

                if column_index == 6:
                    if hardrock_result == "W":
                        item.setForeground(QColor(120, 210, 160))
                    elif hardrock_result == "L":
                        item.setForeground(QColor(235, 110, 100))
                    elif hardrock_favorite is None:
                        item.setForeground(QColor(103, 110, 120))

                if column_index == 7:
                    if strength == "HIGH":
                        item.setForeground(QColor(255, 167, 66))
                    elif strength == "MEDIUM":
                        item.setForeground(QColor(220, 170, 112))
                    elif strength == "LOW":
                        item.setForeground(QColor(176, 181, 188))
                    else:
                        item.setForeground(QColor(103, 110, 120))

                table.setItem(row_index, column_index, item)

        table.setSortingEnabled(True)
        table.blockSignals(False)

        if rows:
            table.setCurrentCell(0, 0)
            self.update_history_detail()
        else:
            self.clear_history_detail()

    def update_history_detail(self):
        selected = self.history_table.selectedItems()
        if not selected:
            self.clear_history_detail()
            return

        row_index = selected[0].row()
        item = self.history_table.item(row_index, 0)
        if item is None:
            self.clear_history_detail()
            return

        snapshot_id = str(item.data(Qt.UserRole) or "")
        row = self.row_by_id.get(snapshot_id)
        if row is None:
            self.clear_history_detail()
            return

        game = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
        self.history_game.setText(game)
        self.history_prediction.setText(str(row.get("captured_utc") or "N/A"))
        self.history_model.setText(str(row.get("model_spread_display") or "N/A"))
        self.history_hr.setText(str(row.get("hardrock_display") or "Unavailable"))
        hardrock_favorite, hardrock_result = self._hardrock_favorite_info(row)
        self.history_hr_favorite.setText(hardrock_favorite or "N/A")

        edge = self._safe_float(row.get("model_edge_points"))
        favorite = str(row.get("model_favorite") or "Model side")
        if edge is None:
            edge_text = "N/A"
        elif edge > 0:
            edge_text = f"{favorite} +{edge:.1f} pts"
        else:
            edge_text = f"No positive edge ({edge:.1f} pts)"
        self.history_edge.setText(edge_text)

        self.history_prob.setText(self._probability_text(row.get("model_favorite_win_probability")))
        strength = str(row.get("strength_label") or "N/A").upper()
        self.history_strength.setText(strength)
        if strength == "HIGH":
            self.history_strength.setStyleSheet("color: #ffa742;")
        elif strength == "MEDIUM":
            self.history_strength.setStyleSheet("color: #dca970;")
        elif strength == "LOW":
            self.history_strength.setStyleSheet("color: #b4b9c0;")
        else:
            self.history_strength.setStyleSheet("color: #6c737d;")

        away_points = self._safe_float(row.get("away_points"))
        home_points = self._safe_float(row.get("home_points"))
        if away_points is not None and home_points is not None:
            self.history_final.setText(
                f"{row.get('away_team', '')} {self._points_text(away_points)}  —  "
                f"{row.get('home_team', '')} {self._points_text(home_points)}"
            )
        else:
            self.history_final.setText("N/A")

        winner_correct = self._safe_int(row.get("winner_correct"))
        winner_text = "CORRECT ✓" if winner_correct == 1 else "WRONG ✕" if winner_correct == 0 else "N/A"
        spread_result = str(row.get("model_side_ats_result") or "N/A").upper()
        spread_text = {"W": "WIN ✓", "L": "LOSS ✕", "P": "PUSH"}.get(
            spread_result,
            "N/A",
        )
        hardrock_result_text = {
            "W": "WIN ✓",
            "L": "LOSS ✕",
        }.get(hardrock_result, "N/A")
        abs_error = self._safe_float(row.get("absolute_margin_error"))
        margin_text = "N/A" if abs_error is None else f"{abs_error:.2f} pts"
        brier = self._safe_float(row.get("brier_score"))
        brier_text = "N/A" if brier is None else f"{brier:.3f}"
        units = self._safe_float(row.get("model_side_flat_units"))
        units_text = "N/A" if units is None else f"{units:+.2f}u"

        self.history_results.setText(
            f"Actual winner: {row.get('actual_winner') or 'N/A'}\n"
            f"Model winner result: {winner_text}\n"
            f"Model spread result: {spread_text}\n"
            f"Hard Rock favorite: {hardrock_favorite or 'N/A'}\n"
            f"Hard Rock favorite result: {hardrock_result_text}\n"
            f"Model margin error: {margin_text}\n"
            f"Brier score: {brier_text}\n"
            f"Flat spread diagnostic: {units_text}"
        )

        reference = str(row.get("reference_market_display") or "Unavailable")
        books = self._safe_int(row.get("reference_market_books"))
        books_text = "N/A" if books is None else str(books)
        warning = self._clean_warning(row.get("signal_warning"))
        margin_sd = self._number(row.get("nn_margin_std"), 2)
        win_sd = self._number(row.get("nn_win_probability_std"), 3)
        coverage = self._safe_int(row.get("nn_feature_coverage"))
        coverage_text = "N/A" if coverage is None else f"{coverage}/54"
        heads = self._safe_int(row.get("nn_heads_agree"))
        heads_text = "YES" if heads == 1 else "NO" if heads == 0 else "N/A"
        score = self._safe_float(row.get("strength_score"))
        score_text = "N/A" if score is None else f"{score:.1f}/100"

        self.history_diagnostics.setText(
            f"Kickoff: {row.get('kickoff_utc') or 'N/A'}\n"
            f"Reference: {reference} | {books_text} books\n"
            f"Input: {row.get('nn_input_status') or 'N/A'}\n"
            f"Feature coverage: {coverage_text}\n"
            f"Ensemble margin SD: {margin_sd}\n"
            f"Win probability SD: {win_sd}\n"
            f"Heads agree: {heads_text}\n"
            f"Strength score: {score_text}\n"
            f"Warning: {warning}"
        )

        self.history_detail_scroll.verticalScrollBar().setValue(0)

    def clear_history_detail(self):
        self.history_game.setText("Select a completed game")
        for label in (
            self.history_prediction,
            self.history_model,
            self.history_hr,
            self.history_hr_favorite,
            self.history_edge,
            self.history_prob,
            self.history_strength,
            self.history_final,
        ):
            label.setText("—")
        self.history_results.setText("")
        self.history_diagnostics.setText("")




# ============================================================
# ACTUAL WAGER ENTRY
#
# Records tickets the user actually places. This is deliberately
# separate from Prediction History's hypothetical spread diagnostic.
# ============================================================

class LogWagerDialog(QDialog):

    def __init__(self, signal, parent=None):
        super().__init__(parent)

        self.signal = dict(signal or {})
        self.setWindowTitle("CFB Edge Engine — Log Actual Wager")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("LOG ACTUAL HARD ROCK WAGER")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        away = str(self.signal.get("away_team") or "").strip()
        home = str(self.signal.get("home_team") or "").strip()
        model_favorite = str(self.signal.get("model_favorite") or "").strip()

        game = QLabel(f"{away} @ {home}")
        game.setObjectName("detailGame")
        game.setAlignment(Qt.AlignCenter)
        game.setWordWrap(True)
        layout.addWidget(game)

        context = QLabel(
            f"MODEL: {self.signal.get('model_spread_display') or 'N/A'}\n"
            f"HARD ROCK: {self.signal.get('hardrock_display') or 'Unavailable'}\n"
            f"STRENGTH: {str(self.signal.get('strength_label') or 'N/A').upper()}"
        )
        context.setObjectName("detailDiagnostics")
        context.setAlignment(Qt.AlignCenter)
        context.setWordWrap(True)
        layout.addWidget(context)

        instruction = QLabel(
            "Enter the ticket exactly as it appears in Hard Rock. "
            "The wager is timestamped when you save it and is linked to the "
            "latest immutable prediction snapshot that existed before that time."
        )
        instruction.setObjectName("warning")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        team_row = QHBoxLayout()
        team_label = QLabel("BET TEAM")
        team_label.setObjectName("metricLabel")
        self.team_combo = QComboBox()
        if away:
            self.team_combo.addItem(away)
        if home and home != away:
            self.team_combo.addItem(home)
        if model_favorite:
            index = self.team_combo.findText(model_favorite, Qt.MatchFixedString)
            if index >= 0:
                self.team_combo.setCurrentIndex(index)
        team_row.addWidget(team_label)
        team_row.addWidget(self.team_combo, 1)
        layout.addLayout(team_row)

        line_row = QHBoxLayout()
        line_label = QLabel("SPREAD")
        line_label.setObjectName("metricLabel")
        self.line_spin = QDoubleSpinBox()
        self.line_spin.setDecimals(1)
        self.line_spin.setSingleStep(0.5)
        self.line_spin.setRange(-99.5, 99.5)
        self.line_spin.setValue(0.0)
        line_row.addWidget(line_label)
        line_row.addWidget(self.line_spin, 1)
        layout.addLayout(line_row)

        odds_row = QHBoxLayout()
        odds_label = QLabel("AMERICAN ODDS")
        odds_label.setObjectName("metricLabel")
        self.odds_edit = QLineEdit()
        self.odds_edit.setPlaceholderText("Example: -110 or +105")
        odds_row.addWidget(odds_label)
        odds_row.addWidget(self.odds_edit, 1)
        layout.addLayout(odds_row)

        stake_row = QHBoxLayout()
        stake_label = QLabel("STAKE ($)")
        stake_label.setObjectName("metricLabel")
        self.stake_spin = QDoubleSpinBox()
        self.stake_spin.setDecimals(2)
        self.stake_spin.setSingleStep(5.0)
        self.stake_spin.setRange(0.01, 1000000.0)
        self.stake_spin.setValue(10.0)
        stake_row.addWidget(stake_label)
        stake_row.addWidget(self.stake_spin, 1)
        layout.addLayout(stake_row)

        notes_row = QHBoxLayout()
        notes_label = QLabel("NOTES")
        notes_label.setObjectName("metricLabel")
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional")
        notes_row.addWidget(notes_label)
        notes_row.addWidget(self.notes_edit, 1)
        layout.addLayout(notes_row)

        self.price_note = QLabel("")
        self.price_note.setObjectName("subtitle")
        self.price_note.setWordWrap(True)
        layout.addWidget(self.price_note)

        button_row = QHBoxLayout()
        button_row.addStretch()

        cancel_button = QPushButton("CANCEL")
        cancel_button.clicked.connect(self.reject)

        save_button = QPushButton("SAVE ACTUAL WAGER")
        save_button.setObjectName("runButton")
        save_button.clicked.connect(self._validate_and_accept)

        button_row.addWidget(cancel_button)
        button_row.addWidget(save_button)
        layout.addLayout(button_row)

        self.team_combo.currentTextChanged.connect(self._apply_market_defaults)
        self._apply_market_defaults()

    @staticmethod
    def _same_team(left, right):
        return str(left or "").strip().casefold() == str(right or "").strip().casefold()

    def _apply_market_defaults(self, *args):
        selected = self.team_combo.currentText().strip()
        model_favorite = str(self.signal.get("model_favorite") or "").strip()

        try:
            stored_line = float(self.signal.get("hardrock_spread_for_model_favorite"))
        except (TypeError, ValueError):
            stored_line = None

        try:
            stored_price = int(float(self.signal.get("hardrock_price_for_model_favorite")))
        except (TypeError, ValueError):
            stored_price = None

        if stored_line is not None:
            if self._same_team(selected, model_favorite):
                self.line_spin.setValue(stored_line)
            else:
                self.line_spin.setValue(-stored_line)

        if self._same_team(selected, model_favorite) and stored_price is not None:
            self.odds_edit.setText(f"{stored_price:+d}" if stored_price > 0 else str(stored_price))
            self.price_note.setText(
                "Line and juice were pre-filled from the captured Hard Rock offer "
                "for the model side. Confirm the live ticket before saving."
            )
        else:
            self.odds_edit.clear()
            self.price_note.setText(
                "The opposite-side juice is not stored in the model-side snapshot. "
                "Enter the exact Hard Rock odds shown on your ticket."
            )

    def _validate_and_accept(self):
        odds_text = self.odds_edit.text().strip().replace("−", "-")
        try:
            odds = int(odds_text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid odds",
                "Enter American odds such as -110 or +105.",
            )
            return

        if -100 < odds < 100:
            QMessageBox.warning(
                self,
                "Invalid odds",
                "American odds must be <= -100 or >= +100.",
            )
            return

        if not self.team_combo.currentText().strip():
            QMessageBox.warning(
                self,
                "Missing team",
                "Choose the team on the actual ticket.",
            )
            return

        self.accept()

    def wager_values(self):
        odds = int(self.odds_edit.text().strip().replace("−", "-"))
        return {
            "wager_team": self.team_combo.currentText().strip(),
            "wager_line": float(self.line_spin.value()),
            "american_odds": odds,
            "stake": float(self.stake_spin.value()),
            "notes": self.notes_edit.text().strip(),
        }


# ============================================================
# ACTUAL WAGER LEDGER WINDOW
# ============================================================

class WagerLedgerWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.all_rows = []
        self.row_by_id = {}

        self.setWindowTitle("CFB Edge Engine — Actual Wagers")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)

        self.build_interface()
        self.reload_data(grade=True)

    @staticmethod
    def _safe_float(value):
        try:
            if value is None or value == "":
                return None
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return None
            return number
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _money(value):
        number = WagerLedgerWindow._safe_float(value)
        if number is None:
            return "N/A"
        sign = "+" if number > 0 else ""
        return f"{sign}${number:,.2f}"

    @staticmethod
    def _local_time(value):
        if not value:
            return "N/A"
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local = dt.astimezone()
            return local.strftime("%a %b %d • %-I:%M %p %Z")
        except Exception:
            return str(value)

    @staticmethod
    def _ticket_text(row):
        team = str(row.get("wager_team") or "N/A")
        line = WagerLedgerWindow._safe_float(row.get("wager_line"))
        odds = row.get("american_odds")
        line_text = "N/A" if line is None else f"{line:+.1f}"
        odds_text = "N/A" if odds in (None, "") else str(int(float(odds)))
        return f"{team} {line_text} ({odds_text})"

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel("ACTUAL WAGER LEDGER")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "REAL TICKETS ONLY  •  HARD ROCK EXECUTION  •  SEPARATE FROM MODEL DIAGNOSTICS"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("controlPanel")
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(14, 10, 14, 10)
        control_layout.setSpacing(10)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search wagers, teams, Strength, result...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self.refresh_view)

        self.grade_button = QPushButton("GRADE PENDING")
        self.grade_button.clicked.connect(self.grade_pending)

        self.refresh_button = QPushButton("REFRESH")
        self.refresh_button.clicked.connect(lambda: self.reload_data(grade=False))

        control_layout.addWidget(self.search_box, 1)
        control_layout.addWidget(self.grade_button)
        control_layout.addWidget(self.refresh_button)
        layout.addWidget(controls)

        self.summary_label = QLabel("Loading actual wagers...")
        self.summary_label.setObjectName("summary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)

        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(
            [
                "PLACED",
                "GAME",
                "WAGER",
                "ODDS",
                "STAKE",
                "STATUS",
                "P/L",
                "STRENGTH",
                "CLV",
            ]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.itemSelectionChanged.connect(self.update_detail)
        splitter.addWidget(self.table)

        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        detail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        detail_scroll.setFrameShape(QFrame.NoFrame)
        detail_scroll.setMinimumWidth(390)

        self.detail_panel = QFrame()
        self.detail_panel.setObjectName("detailPanel")

        detail_layout = QVBoxLayout(self.detail_panel)
        detail_layout.setContentsMargins(18, 16, 18, 16)
        detail_layout.setSpacing(7)

        detail_title = QLabel("ACTUAL TICKET")
        detail_title.setObjectName("sectionTitle")
        detail_layout.addWidget(detail_title)

        self.detail_game = QLabel("Select a wager")
        self.detail_game.setObjectName("detailGame")
        self.detail_game.setWordWrap(True)
        detail_layout.addWidget(self.detail_game)

        self.detail_ticket = self._metric(detail_layout, "WAGER")
        self.detail_stake = self._metric(detail_layout, "STAKE")
        self.detail_result = self._metric(detail_layout, "RESULT")
        self.detail_profit = self._metric(detail_layout, "PROFIT / LOSS")
        self.detail_clv = self._metric(detail_layout, "CLV")

        model_title = QLabel("MODEL AT BET TIME")
        model_title.setObjectName("sectionTitle")
        detail_layout.addSpacing(5)
        detail_layout.addWidget(model_title)

        self.detail_model = self._metric(detail_layout, "MODEL SPREAD")
        self.detail_hr = self._metric(detail_layout, "HARD ROCK AT CAPTURE")
        self.detail_edge = self._metric(detail_layout, "MODEL EDGE")
        self.detail_strength = self._metric(detail_layout, "STRENGTH")

        self.detail_meta = QLabel("")
        self.detail_meta.setObjectName("detailDiagnostics")
        self.detail_meta.setWordWrap(True)
        detail_layout.addSpacing(5)
        detail_layout.addWidget(self.detail_meta)
        detail_layout.addStretch()

        detail_scroll.setWidget(self.detail_panel)
        splitter.addWidget(detail_scroll)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

    @staticmethod
    def _metric(layout, title):
        label = QLabel(title)
        label.setObjectName("metricLabel")
        value = QLabel("—")
        value.setObjectName("metricValue")
        value.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(value)
        return value

    def grade_pending(self):
        try:
            from wager_ledger_v1 import score_pending_wagers
            stats = score_pending_wagers()
            self.reload_data(grade=False)
            QMessageBox.information(
                self,
                "Wager grading complete",
                (
                    f"Pending checked: {stats.get('pending', 0)}\n"
                    f"Newly graded: {stats.get('graded', 0)}\n"
                    f"Still awaiting result: {stats.get('awaiting_result', 0)}"
                ),
            )
        except Exception as error:
            QMessageBox.warning(
                self,
                "Could not grade wagers",
                str(error),
            )

    def reload_data(self, grade=False):
        if grade:
            try:
                from wager_ledger_v1 import score_pending_wagers
                score_pending_wagers()
            except Exception:
                pass

        self.all_rows = []
        self.row_by_id = {}

        if not WAGER_DB.exists():
            self.summary_label.setText(
                "Wager Ledger is not initialized yet. Log an actual wager from the main board."
            )
            self.table.setRowCount(0)
            self.clear_detail()
            return

        try:
            connection = sqlite3.connect(
                f"file:{WAGER_DB}?mode=ro",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            self.all_rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM wagers ORDER BY placed_utc DESC, wager_id DESC"
                ).fetchall()
            ]
            connection.close()
        except Exception as error:
            self.summary_label.setText(f"Could not read Wager Ledger: {error}")
            self.table.setRowCount(0)
            self.clear_detail()
            return

        self.row_by_id = {
            str(row.get("wager_id") or ""): row
            for row in self.all_rows
            if str(row.get("wager_id") or "")
        }

        self.update_summary(self.all_rows)
        self.refresh_view()

    def update_summary(self, rows):
        settled = [
            row
            for row in rows
            if str(row.get("status") or "").upper() in {"WON", "LOST", "PUSH"}
        ]
        won = sum(str(row.get("status") or "").upper() == "WON" for row in settled)
        lost = sum(str(row.get("status") or "").upper() == "LOST" for row in settled)
        push = sum(str(row.get("status") or "").upper() == "PUSH" for row in settled)
        pending = sum(str(row.get("status") or "").upper() == "PENDING" for row in rows)

        risked = sum(float(row.get("stake") or 0.0) for row in settled)
        profit = sum(float(row.get("profit_loss") or 0.0) for row in settled)
        roi = (profit / risked) if risked > 0 else None

        running = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for row in sorted(
            settled,
            key=lambda item: (
                str(item.get("placed_utc") or ""),
                str(item.get("wager_id") or ""),
            ),
        ):
            running += float(row.get("profit_loss") or 0.0)
            peak = max(peak, running)
            max_drawdown = max(max_drawdown, peak - running)

        high = [
            row
            for row in settled
            if str(row.get("strength_label") or "").upper() == "HIGH"
        ]
        medium = [
            row
            for row in settled
            if str(row.get("strength_label") or "").upper() == "MEDIUM"
        ]
        low = [
            row
            for row in settled
            if str(row.get("strength_label") or "").upper() == "LOW"
        ]

        def record_text(group):
            w = sum(str(row.get("status") or "").upper() == "WON" for row in group)
            l = sum(str(row.get("status") or "").upper() == "LOST" for row in group)
            p = sum(str(row.get("status") or "").upper() == "PUSH" for row in group)
            return f"{w}-{l}-{p}"

        roi_text = "N/A" if roi is None else f"{roi * 100:+.1f}%"

        self.summary_label.setText(
            f"ACTUAL RECORD {won}-{lost}-{push}  •  PENDING {pending}  •  "
            f"RISKED ${risked:,.2f}  •  P/L {self._money(profit)}  •  "
            f"ROI {roi_text}  •  MAX DRAWDOWN ${max_drawdown:,.2f}\n"
            f"HIGH {record_text(high)}  •  MEDIUM {record_text(medium)}  •  "
            f"LOW {record_text(low)}"
        )

    def refresh_view(self, *args):
        query = self.search_box.text().strip().lower()
        rows = list(self.all_rows)

        if query:
            rows = [
                row
                for row in rows
                if query in " ".join(
                    str(row.get(key) or "")
                    for key in (
                        "away_team",
                        "home_team",
                        "wager_team",
                        "status",
                        "strength_label",
                        "result",
                    )
                ).lower()
            ]

        self.populate_table(rows)

    def populate_table(self, rows):
        table = self.table
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            wager_id = str(row.get("wager_id") or "")
            game = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
            ticket = self._ticket_text(row)
            odds = row.get("american_odds")
            odds_text = "N/A" if odds in (None, "") else str(int(float(odds)))
            stake = self._safe_float(row.get("stake"))
            stake_text = "N/A" if stake is None else f"${stake:,.2f}"
            status = str(row.get("status") or "PENDING").upper()
            profit = self._safe_float(row.get("profit_loss"))
            profit_text = "—" if profit is None else self._money(profit)
            strength = str(row.get("strength_label") or "N/A").upper()
            clv = self._safe_float(row.get("clv_points"))
            clv_text = (
                f"{clv:+.1f} pts"
                if clv is not None
                else str(row.get("clv_status") or "PENDING")
            )

            values = [
                self._local_time(row.get("placed_utc")),
                game,
                ticket,
                odds_text,
                stake_text,
                status,
                profit_text,
                strength,
                clv_text,
            ]

            sort_values = [
                str(row.get("placed_utc") or ""),
                game.lower(),
                str(row.get("wager_team") or "").lower(),
                int(float(odds)) if odds not in (None, "") else 0,
                stake if stake is not None else -1.0,
                {"PENDING": 0, "LOST": 1, "PUSH": 2, "WON": 3}.get(status, 0),
                profit if profit is not None else -999999.0,
                {"N/A": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}.get(strength, 0),
                clv if clv is not None else -9999.0,
            ]

            for column, (value, sort_value) in enumerate(zip(values, sort_values)):
                item = SortableTableWidgetItem(str(value))
                item.setData(Qt.UserRole, wager_id)
                item.setData(Qt.UserRole + 1, sort_value)
                table.setItem(row_index, column, item)

        table.setSortingEnabled(True)

        if rows:
            table.selectRow(0)
        else:
            self.clear_detail()

    def update_detail(self):
        selected = self.table.selectedItems()
        if not selected:
            self.clear_detail()
            return

        first = self.table.item(selected[0].row(), 0)
        if first is None:
            self.clear_detail()
            return

        wager_id = str(first.data(Qt.UserRole) or "")
        row = self.row_by_id.get(wager_id)
        if row is None:
            self.clear_detail()
            return

        game = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
        self.detail_game.setText(game)
        self.detail_ticket.setText(self._ticket_text(row))

        stake = self._safe_float(row.get("stake"))
        potential = self._safe_float(row.get("potential_profit"))
        self.detail_stake.setText(
            (
                f"${stake:,.2f} risked • ${potential:,.2f} potential profit"
                if stake is not None and potential is not None
                else "N/A"
            )
        )

        status = str(row.get("status") or "PENDING").upper()
        final_away = self._safe_float(row.get("final_away_points"))
        final_home = self._safe_float(row.get("final_home_points"))
        if final_away is not None and final_home is not None:
            result_text = (
                f"{status} • Final {row.get('away_team')} "
                f"{final_away:g}-{final_home:g} {row.get('home_team')}"
            )
        else:
            result_text = status
        self.detail_result.setText(result_text)

        profit = self._safe_float(row.get("profit_loss"))
        self.detail_profit.setText("—" if profit is None else self._money(profit))

        clv = self._safe_float(row.get("clv_points"))
        if clv is None:
            self.detail_clv.setText(str(row.get("clv_status") or "PENDING"))
        else:
            self.detail_clv.setText(f"{clv:+.1f} pts")

        self.detail_model.setText(str(row.get("model_spread_display") or "N/A"))
        self.detail_hr.setText(str(row.get("hardrock_display") or "Unavailable"))

        edge = self._safe_float(row.get("model_edge_points"))
        self.detail_edge.setText("N/A" if edge is None else f"{edge:+.1f} pts")
        self.detail_strength.setText(str(row.get("strength_label") or "N/A").upper())

        self.detail_meta.setText(
            f"Placed: {self._local_time(row.get('placed_utc'))}\n"
            f"Kickoff: {self._local_time(row.get('kickoff_utc'))}\n"
            f"Prediction capture: {self._local_time(row.get('snapshot_captured_utc'))}\n"
            f"Snapshot link: {row.get('snapshot_link_status') or 'N/A'}\n"
            f"Snapshot ID: {row.get('linked_snapshot_id') or 'N/A'}\n"
            f"Game ID: {row.get('game_id') or 'N/A'}\n"
            f"Reference at bet time: {row.get('reference_market_display') or 'Unavailable'}\n"
            f"Warning: {row.get('signal_warning') or 'NONE'}\n"
            f"Notes: {row.get('notes') or '—'}"
        )

    def clear_detail(self):
        self.detail_game.setText("Select a wager")
        for label in (
            self.detail_ticket,
            self.detail_stake,
            self.detail_result,
            self.detail_profit,
            self.detail_clv,
            self.detail_model,
            self.detail_hr,
            self.detail_edge,
            self.detail_strength,
        ):
            label.setText("—")
        self.detail_meta.setText("")



class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.pipeline_thread = None

        self.history_window = None

        self.wager_window = None

        self.selected_signal = None

        self.change_by_game_key = {}

        self.change_by_game_id = {}

        self.audit_by_game_key = {}

        self.audit_by_game_id = {}

        self.health_report = {}

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

        self.refresh_support_artifacts(
            silent=True
        )


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
        # SYSTEM HEALTH STRIP
        # ====================================================

        self.health_panel = QFrame()
        self.health_panel.setObjectName(
            "controlPanel"
        )

        health_layout = QHBoxLayout(
            self.health_panel
        )
        health_layout.setContentsMargins(
            12,
            7,
            12,
            7
        )
        health_layout.setSpacing(
            8
        )

        self.health_overall = QLabel(
            "SYSTEM HEALTH • CHECKING"
        )
        self.health_overall.setObjectName(
            "sectionTitle"
        )
        health_layout.addWidget(
            self.health_overall
        )

        self.health_labels = {}

        for component_name in (
            "BOARD",
            "MODEL",
            "FEATURES",
            "HARD ROCK",
            "MARKET",
            "HISTORY",
            "WAGERS"
        ):
            chip = QLabel(
                f"{component_name} —"
            )
            chip.setAlignment(
                Qt.AlignCenter
            )
            chip.setMinimumWidth(
                92
            )
            chip.setStyleSheet(
                "padding: 4px 7px; "
                "border: 1px solid #3a3f46; "
                "border-radius: 4px; "
                "color: #8e949c;"
            )
            self.health_labels[
                component_name
            ] = chip
            health_layout.addWidget(
                chip
            )

        health_layout.addStretch()

        main_layout.addWidget(
            self.health_panel
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


        self.history_button = QPushButton(
            "PREDICTION HISTORY"
        )


        self.history_button.setObjectName(
            "historyButton"
        )


        self.history_button.clicked.connect(
            self.show_prediction_history
        )


        self.wagers_button = QPushButton(
            "ACTUAL WAGERS"
        )

        self.wagers_button.setObjectName(
            "historyButton"
        )

        self.wagers_button.clicked.connect(
            self.show_wager_ledger
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


        control_layout.addWidget(
            self.history_button
        )


        control_layout.addWidget(
            self.wagers_button
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
            7
        )


        self.table.setHorizontalHeaderLabels(
            [
                "GAME",
                "KICKOFF",
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
            QHeaderView.ResizeToContents
        )


        table_header.setSectionResizeMode(
            2,
            QHeaderView.Stretch
        )


        table_header.setSectionResizeMode(
            3,
            QHeaderView.Stretch
        )


        table_header.setSectionResizeMode(
            4,
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
            900
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


        self.detail_audit_badge = QLabel(
            "AUDIT • NO SNAPSHOT SELECTED"
        )
        self.detail_audit_badge.setObjectName(
            "researchMode"
        )
        self.detail_audit_badge.setWordWrap(
            True
        )
        detail_layout.addWidget(
            self.detail_audit_badge
        )


        detail_layout.addSpacing(
            4
        )


        (
            self.detail_kickoff_label,
            self.detail_kickoff
        ) = self.add_metric_block(

            detail_layout,

            "KICKOFF"
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


        self.log_wager_button = QPushButton(
            "LOG ACTUAL WAGER"
        )
        self.log_wager_button.setObjectName(
            "runButton"
        )
        self.log_wager_button.setEnabled(
            False
        )
        self.log_wager_button.clicked.connect(
            self.log_selected_wager
        )
        detail_layout.addWidget(
            self.log_wager_button
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


        changes_title = QLabel(
            "WHAT CHANGED SINCE LAST RUN"
        )
        changes_title.setObjectName(
            "sectionTitle"
        )

        self.detail_changes = QLabel(
            "No earlier snapshot loaded."
        )
        self.detail_changes.setObjectName(
            "detailDiagnostics"
        )
        self.detail_changes.setWordWrap(
            True
        )

        detail_layout.addWidget(
            changes_title
        )
        detail_layout.addWidget(
            self.detail_changes
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


    def kickoff_local_datetime(
        self,
        signal
    ):

        raw_value = None

        for field in (
            "kickoff_utc",
            "start_date",
            "start_date_utc",
            "commence_time_utc",
            "start_time"
        ):

            candidate = signal.get(
                field
            )

            if candidate not in (
                None,
                ""
            ):

                raw_value = str(
                    candidate
                ).strip()

                break

        if not raw_value:

            return None

        normalized = raw_value

        if normalized.endswith(
            "Z"
        ):

            normalized = (
                normalized[:-1]
                + "+00:00"
            )

        try:

            parsed = datetime.fromisoformat(
                normalized
            )

        except ValueError:

            parsed = None

            for pattern in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S"
            ):

                try:

                    parsed = datetime.strptime(
                        raw_value,
                        pattern
                    ).replace(
                        tzinfo=timezone.utc
                    )

                    break

                except ValueError:

                    continue

        if parsed is None:

            return None

        if parsed.tzinfo is None:

            # Live-board timestamps are UTC when the source omits
            # an explicit offset. Treat them as UTC, then convert
            # to the Mac's local timezone for display.
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        try:

            return parsed.astimezone()

        except Exception:

            return parsed


    def format_kickoff(
        self,
        signal
    ):

        kickoff = self.kickoff_local_datetime(
            signal
        )

        if kickoff is None:

            return "N/A"

        date_text = (
            kickoff.strftime(
                "%a %b"
            )
            + f" {kickoff.day}"
        )

        time_text = kickoff.strftime(
            "%I:%M %p"
        ).lstrip(
            "0"
        )

        zone_text = (
            kickoff.tzname()
            or ""
        )

        if zone_text:

            return (
                f"{date_text} • "
                f"{time_text} {zone_text}"
            )

        return (
            f"{date_text} • "
            f"{time_text}"
        )


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


            kickoff = self.format_kickoff(
                signal
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

                kickoff,

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


            kickoff_datetime = self.kickoff_local_datetime(
                signal
            )


            kickoff_sort = (
                kickoff_datetime.timestamp()
                if kickoff_datetime is not None
                else -1000000000.0
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
                kickoff_sort,
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
                    1,
                    5,
                    6
                ):

                    item.setTextAlignment(
                        Qt.AlignCenter
                    )


                if column_index == 6:

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

        self.selected_signal = None

        self.detail_game.setText(
            "Select a game"
        )


        self.detail_kickoff.setText(
            "—"
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


        self.log_wager_button.setEnabled(
            False
        )

        self.detail_audit_badge.setText(
            "AUDIT • NO SNAPSHOT SELECTED"
        )

        self.detail_changes.setText(
            "No earlier snapshot loaded."
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

        self.selected_signal = signal

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


        self.detail_kickoff.setText(
            self.format_kickoff(
                signal
            )
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


        self.log_wager_button.setEnabled(
            True
        )

        self.detail_audit_badge.setText(
            self.audit_badge_text(
                signal
            )
        )

        self.detail_changes.setText(
            self.what_changed_text(
                signal
            )
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
    # SUPPORT ARTIFACTS / HEALTH / AUDIT / WHAT CHANGED
    # ========================================================

    @staticmethod
    def _load_json_file(path):
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @staticmethod
    def _support_game_key(signal):
        game_id = signal.get("game_id")
        if game_id not in (None, ""):
            return f"cfbd:{str(game_id).strip()}"

        season = safe_int_local(signal.get("season"), default=None)
        week = safe_int_local(signal.get("week"), default=None)
        away = str(signal.get("away_team") or "").strip()
        home = str(signal.get("home_team") or "").strip()

        if season is not None and week is not None and away and home:
            return f"fallback:{season}:{week}:{away}:{home}"

        return ""

    def refresh_support_artifacts(self, silent=True):
        # These are LOCAL, READ-ONLY analytical builders. They make no sportsbook
        # or CFBD calls. The canonical live pipeline dependency remains
        # run_game_day_v1.py.
        for script in (
            AUDIT_SCRIPT,
            WHAT_CHANGED_SCRIPT,
            HEALTH_SCRIPT,
        ):
            if not script.exists():
                continue

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0 and not silent:
                    self.log_box.append(
                        f"SUPPORT ARTIFACT WARNING: {script.name} exited "
                        f"{result.returncode}"
                    )

            except Exception as error:
                if not silent:
                    self.log_box.append(
                        f"SUPPORT ARTIFACT WARNING: {script.name}: {error}"
                    )

        self.load_support_artifacts()

    def refresh_health_only(self):
        if HEALTH_SCRIPT.exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(HEALTH_SCRIPT),
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception:
                pass

        self.load_support_artifacts()

    def load_support_artifacts(self):
        changes_payload = self._load_json_file(
            WHAT_CHANGED_FILE
        )
        self.change_by_game_key = {}
        self.change_by_game_id = {}

        if isinstance(changes_payload, dict):
            comparisons = changes_payload.get("comparisons")
            if isinstance(comparisons, list):
                for row in comparisons:
                    if not isinstance(row, dict):
                        continue

                    game_key = str(row.get("game_key") or "").strip()
                    game_id = str(row.get("game_id") or "").strip()

                    if game_key:
                        self.change_by_game_key[game_key] = row
                    if game_id:
                        self.change_by_game_id[game_id] = row

        audit_payload = self._load_json_file(
            AUDIT_FILE
        )
        self.audit_by_game_key = {}
        self.audit_by_game_id = {}

        if isinstance(audit_payload, dict):
            audit_records = audit_payload.get("records")
            if isinstance(audit_records, list):
                ordered_records = sorted(
                    [
                        row
                        for row in audit_records
                        if isinstance(row, dict)
                    ],
                    key=lambda row: (
                        str(row.get("captured_utc") or ""),
                        str(row.get("snapshot_id") or ""),
                    ),
                )

                for row in ordered_records:
                    game_key = str(row.get("game_key") or "").strip()
                    game_id = str(row.get("game_id") or "").strip()

                    if game_key:
                        self.audit_by_game_key[game_key] = row
                    if game_id:
                        self.audit_by_game_id[game_id] = row

        health_payload = self._load_json_file(
            HEALTH_FILE
        )
        self.health_report = (
            health_payload
            if isinstance(health_payload, dict)
            else {}
        )

        self.update_health_strip()

        # Refresh currently selected game's support text without disturbing
        # the table selection.
        if self.selected_signal is not None:
            self.detail_audit_badge.setText(
                self.audit_badge_text(
                    self.selected_signal
                )
            )
            self.detail_changes.setText(
                self.what_changed_text(
                    self.selected_signal
                )
            )

    def update_health_strip(self):
        report = self.health_report
        overall = str(
            report.get("overall_status") or "UNKNOWN"
        ).upper()

        self.health_overall.setText(
            f"SYSTEM HEALTH • {overall}"
        )

        components = report.get("components")
        if not isinstance(components, dict):
            components = {}

        key_map = {
            "BOARD": "BOARD",
            "MODEL": "MODEL",
            "FEATURES": "FEATURES",
            "HARD ROCK": "HARD_ROCK",
            "MARKET": "MARKET",
            "HISTORY": "HISTORY",
            "WAGERS": "WAGERS",
        }

        for display_name, component_key in key_map.items():
            component = components.get(component_key)
            if not isinstance(component, dict):
                status = "N/A"
            else:
                status = str(component.get("status") or "N/A").upper()

            if status == "PASS":
                symbol = "✓"
                style = (
                    "padding: 4px 7px; border: 1px solid #675336; "
                    "border-radius: 4px; color: #ffa742;"
                )
            elif status == "PARTIAL":
                symbol = "◐"
                style = (
                    "padding: 4px 7px; border: 1px solid #5a5145; "
                    "border-radius: 4px; color: #dca970;"
                )
            elif status == "WARNING":
                symbol = "!"
                style = (
                    "padding: 4px 7px; border: 1px solid #704f3c; "
                    "border-radius: 4px; color: #d98a59;"
                )
            elif status == "FAIL":
                symbol = "✕"
                style = (
                    "padding: 4px 7px; border: 1px solid #723e3e; "
                    "border-radius: 4px; color: #db7777;"
                )
            else:
                symbol = "—"
                style = (
                    "padding: 4px 7px; border: 1px solid #3a3f46; "
                    "border-radius: 4px; color: #8e949c;"
                )

            chip = self.health_labels.get(display_name)
            if chip is not None:
                chip.setText(
                    f"{display_name} {symbol}"
                )
                chip.setToolTip(
                    f"{display_name}: {status}"
                )
                chip.setStyleSheet(style)

    def _audit_row_for_signal(self, signal):
        game_id = str(signal.get("game_id") or "").strip()
        if game_id and game_id in self.audit_by_game_id:
            return self.audit_by_game_id[game_id]

        game_key = self._support_game_key(signal)
        if game_key and game_key in self.audit_by_game_key:
            return self.audit_by_game_key[game_key]

        return None

    def audit_badge_text(self, signal):
        row = self._audit_row_for_signal(signal)

        if row is None:
            return "AUDIT • NO LOGGED SNAPSHOT FOR CURRENT GAME"

        status = str(
            row.get("verification_status") or "UNKNOWN"
        ).upper()

        if status == "VERIFIED_PREKICKOFF_WITH_BATCH":
            return "✓ VERIFIED PRE-KICKOFF SNAPSHOT • BATCH CORROBORATED"

        if status == "VERIFIED_PREKICKOFF":
            return "✓ VERIFIED PRE-KICKOFF SNAPSHOT"

        if status == "PREKICKOFF_VALID_UNPROTECTED_DB":
            return "AUDIT • PRE-KICKOFF VALID • DB PROTECTION NOT VERIFIED"

        flags = row.get("audit_flags")
        if isinstance(flags, list) and flags:
            return "AUDIT FAILED • " + " | ".join(str(flag) for flag in flags)

        return f"AUDIT • {status}"

    def _change_row_for_signal(self, signal):
        game_id = str(signal.get("game_id") or "").strip()
        if game_id and game_id in self.change_by_game_id:
            return self.change_by_game_id[game_id]

        game_key = self._support_game_key(signal)
        if game_key and game_key in self.change_by_game_key:
            return self.change_by_game_key[game_key]

        return None

    @staticmethod
    def _change_value(value, decimals=1, suffix=""):
        try:
            if value is None or value == "":
                return "N/A"
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return "N/A"
            return f"{number:+.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    def what_changed_text(self, signal):
        row = self._change_row_for_signal(signal)

        if row is None:
            return (
                "No earlier immutable snapshot is available for this game yet. "
                "The next successful GAME DAY run will give us a direct comparison."
            )

        model = row.get("model") or {}
        hardrock = row.get("hardrock") or {}
        edge = row.get("edge") or {}
        strength = row.get("strength") or {}

        previous_model = str(model.get("previous_spread_display") or "N/A")
        current_model = str(model.get("current_spread_display") or "N/A")

        previous_hr = str(hardrock.get("previous_display") or "Unavailable")
        current_hr = str(hardrock.get("current_display") or "Unavailable")

        previous_edge = self._change_value(
            edge.get("previous_points"),
            decimals=1,
            suffix=" pts",
        )
        current_edge = self._change_value(
            edge.get("current_points"),
            decimals=1,
            suffix=" pts",
        )

        previous_probability = model.get(
            "previous_probability_for_current_favorite"
        )
        current_probability = model.get(
            "current_probability_for_current_favorite"
        )

        try:
            previous_probability_text = f"{float(previous_probability):.1f}%"
        except (TypeError, ValueError):
            previous_probability_text = "N/A"

        try:
            current_probability_text = f"{float(current_probability):.1f}%"
        except (TypeError, ValueError):
            current_probability_text = "N/A"

        previous_strength = str(
            strength.get("previous_label") or "N/A"
        ).upper()
        current_strength = str(
            strength.get("current_label") or "N/A"
        ).upper()

        hours = row.get("hours_between_snapshots")
        try:
            hours_text = f"{float(hours):.1f}h between captures"
        except (TypeError, ValueError):
            hours_text = "Capture interval N/A"

        flags = row.get("flags")
        flag_text = ""
        if isinstance(flags, list) and flags:
            flag_text = "\nFlags: " + " • ".join(
                str(flag).replace("_", " ")
                for flag in flags
            )

        if not edge.get("directly_comparable", True):
            edge_line = (
                f"Model edge: {previous_edge} → {current_edge} "
                "(side changed; delta not directly comparable)"
            )
        else:
            delta = self._change_value(
                edge.get("delta_points"),
                decimals=1,
                suffix=" pts",
            )
            edge_line = (
                f"Model edge: {previous_edge} → {current_edge} "
                f"(Δ {delta})"
            )

        return (
            f"Model spread: {previous_model} → {current_model}\n"
            f"Hard Rock: {previous_hr} → {current_hr}\n"
            f"{edge_line}\n"
            f"Win probability: {previous_probability_text} → "
            f"{current_probability_text}\n"
            f"Strength: {previous_strength} → {current_strength}\n"
            f"{hours_text}"
            f"{flag_text}"
        )

    # ========================================================
    # ACTUAL WAGERS
    # ========================================================

    def log_selected_wager(self):
        signal = self.selected_signal

        if signal is None:
            QMessageBox.information(
                self,
                "No game selected",
                "Select a game before logging an actual wager.",
            )
            return

        dialog = LogWagerDialog(
            signal,
            self,
        )

        # Reuse the command-center styling.
        dialog.setStyleSheet(
            self.styleSheet()
        )

        if dialog.exec() != QDialog.Accepted:
            return

        values = dialog.wager_values()

        try:
            from wager_ledger_v1 import log_spread_wager

            row = log_spread_wager(
                signal,
                wager_team=values["wager_team"],
                wager_line=values["wager_line"],
                american_odds=values["american_odds"],
                stake=values["stake"],
                notes=values["notes"],
            )

        except Exception as error:
            QMessageBox.warning(
                self,
                "Wager was not logged",
                str(error),
            )
            return

        QMessageBox.information(
            self,
            "Actual wager logged",
            (
                f"{row.get('wager_team')} "
                f"{float(row.get('wager_line')):+.1f} "
                f"({int(row.get('american_odds'))})\n"
                f"Stake: ${float(row.get('stake')):,.2f}\n"
                f"Snapshot link: {row.get('snapshot_link_status')}"
            ),
        )

        if self.wager_window is not None:
            self.wager_window.reload_data(
                grade=False
            )

        self.refresh_health_only()

    def show_wager_ledger(self):
        if self.wager_window is None:
            self.wager_window = WagerLedgerWindow(
                self
            )
            self.wager_window.setStyleSheet(
                self.styleSheet()
            )
        else:
            self.wager_window.reload_data(
                grade=True
            )

        self.wager_window.show()
        self.wager_window.raise_()
        self.wager_window.activateWindow()

    def grade_wagers_safely(self):
        try:
            from wager_ledger_v1 import score_pending_wagers
            return score_pending_wagers()
        except Exception:
            return None



    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    def show_prediction_history(self):

        if self.history_window is None:

            self.history_window = PredictionHistoryWindow(
                self
            )

            # Reuse the exact command-center styling from the main GUI.
            self.history_window.setStyleSheet(
                self.styleSheet()
            )

        else:

            self.history_window.reload_data()

        self.history_window.show()
        self.history_window.raise_()
        self.history_window.activateWindow()


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


            if self.history_window is not None:

                self.history_window.reload_data()


            self.grade_wagers_safely()


            if self.wager_window is not None:

                self.wager_window.reload_data(
                    grade=False
                )


            self.refresh_support_artifacts(
                silent=False
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
