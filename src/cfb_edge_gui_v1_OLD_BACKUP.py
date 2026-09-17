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
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget
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


HARDROCK_COLLECTOR = (
    PROJECT_ROOT
    / "src"
    / "snapshot_hardrock_oddspapi_2026_v1.py"
)


LIVE_BOARD_BUILDER = (
    PROJECT_ROOT
    / "src"
    / "build_live_model_market_board_v1.py"
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
# SYSTEM TEST:
#   - controller dry run
#   - rebuild board from existing files
#   - no sportsbook API calls
#
# RUN GAME DAY:
#   - existing game-day controller
#   - fresh Hard Rock OddsPapi snapshot
#   - rebuild V3 live model/market board
#
# NOTE:
# The current controller remains responsible for the
# reference-market workflow. The GUI adds the Hard Rock
# execution refresh and the V3 board assembly after it.
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


    def emit_output(
        self,
        text
    ):

        self.output_received.emit(
            str(
                text
            )
        )


    def run_command(
        self,
        command,
        stage,
        progress
    ):

        self.stage_changed.emit(
            stage,
            progress
        )

        self.emit_output(
            ""
        )

        self.emit_output(
            "=" * 76
        )

        self.emit_output(
            stage
        )

        self.emit_output(
            "=" * 76
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


        return_code = process.wait()


        if return_code != 0:

            raise RuntimeError(

                f"{stage} failed "
                f"with exit code {return_code}."
            )


    def run(self):

        try:

            if self.dry_run:

                self.run_command(

                    [
                        sys.executable,
                        str(
                            CONTROLLER
                        ),
                        "--dry-run"
                    ],

                    "SYSTEM TEST",

                    35
                )


                if LIVE_BOARD_BUILDER.exists():

                    self.run_command(

                        [
                            sys.executable,
                            str(
                                LIVE_BOARD_BUILDER
                            )
                        ],

                        "VERIFYING LIVE BOARD",

                        80
                    )


                self.stage_changed.emit(
                    "SYSTEM TEST COMPLETE",
                    100
                )

                self.completed.emit(
                    True
                )

                return


            self.run_command(

                [
                    sys.executable,
                    str(
                        CONTROLLER
                    )
                ],

                "REFRESHING REFERENCE MARKET",

                25
            )


            if not HARDROCK_COLLECTOR.exists():

                raise RuntimeError(

                    "Hard Rock collector not found:\n"
                    f"{HARDROCK_COLLECTOR}"
                )


            self.run_command(

                [
                    sys.executable,
                    str(
                        HARDROCK_COLLECTOR
                    )
                ],

                "REFRESHING HARD ROCK",

                65
            )


            if not LIVE_BOARD_BUILDER.exists():

                raise RuntimeError(

                    "Live board builder not found:\n"
                    f"{LIVE_BOARD_BUILDER}"
                )


            self.run_command(

                [
                    sys.executable,
                    str(
                        LIVE_BOARD_BUILDER
                    )
                ],

                "ASSEMBLING LIVE BOARD",

                88
            )


            self.stage_changed.emit(
                "ANALYSIS COMPLETE",
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
# AI CORE VISUALIZATION
# ============================================================

class AICoreWidget(QWidget):

    def __init__(self):

        super().__init__()


        # Reduced minimum height.
        #
        # This was one of the things forcing the board
        # to become tiny in the previous version.
        self.setMinimumSize(
            320,
            260
        )


        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )


        self.rotation = 0.0

        self.pulse = 0.0

        self.activity = 0.18

        self.target_activity = 0.18

        self.progress = 0

        self.state = "SYSTEM READY"


        self.nodes = []


        random.seed(
            42
        )


        for _ in range(52):

            angle = random.uniform(
                0,
                math.tau
            )


            radius = random.uniform(
                0.10,
                0.95
            )


            layer = random.uniform(
                0.35,
                1.0
            )


            size = random.uniform(
                1.8,
                4.8
            )


            self.nodes.append(
                (
                    angle,
                    radius,
                    layer,
                    size
                )
            )


        self.timer = QTimer(
            self
        )


        self.timer.timeout.connect(
            self.animate
        )


        self.timer.start(
            33
        )


    # ========================================================
    # STATE
    # ========================================================

    def set_stage(
        self,
        state,
        progress
    ):

        self.state = state

        self.progress = progress


        if progress <= 5:

            self.target_activity = 0.20


        elif progress < 30:

            self.target_activity = 0.42


        elif progress < 60:

            self.target_activity = 0.68


        elif progress < 90:

            self.target_activity = 0.92


        elif progress < 100:

            self.target_activity = 1.0


        else:

            self.target_activity = 0.48


        self.update()


    def animate(self):

        self.rotation += (
            0.005
            +
            self.activity
            *
            0.015
        )


        self.pulse += (
            0.035
            +
            self.activity
            *
            0.065
        )


        self.activity += (
            self.target_activity
            -
            self.activity
        ) * 0.05


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
            "HIGH": 0.88,
            "MEDIUM": 0.68,
            "LOW": 0.44,
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


        center = QPointF(
            width / 2,
            height / 2 - 5
        )


        scale = min(
            width,
            height
        )


        core_radius = (
            scale
            *
            0.20
        )


        # ====================================================
        # BACKGROUND
        # ====================================================

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
                9,
                13
            )
        )


        background.setColorAt(
            1.0,
            QColor(
                1,
                2,
                4
            )
        )


        painter.fillRect(
            self.rect(),
            background
        )


        # ====================================================
        # AMBIENT GLOW
        # ====================================================

        ambient = QRadialGradient(
            center,
            scale * 0.46
        )


        ambient.setColorAt(
            0.0,
            QColor(
                255,
                132,
                25,
                int(
                    80
                    *
                    self.activity
                )
            )
        )


        ambient.setColorAt(
            0.4,
            QColor(
                255,
                82,
                0,
                int(
                    28
                    *
                    self.activity
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


        painter.drawEllipse(
            center,
            scale * 0.46,
            scale * 0.46
        )


        # ====================================================
        # ORBITAL RINGS
        # ====================================================

        for index in range(8):

            radius_x = (
                core_radius
                *
                (
                    1.12
                    +
                    index
                    *
                    0.17
                )
            )


            radius_y = (
                radius_x
                *
                (
                    0.26
                    +
                    0.07
                    *
                    math.sin(
                        self.rotation
                        +
                        index
                    )
                )
            )


            alpha = int(
                42
                +
                110
                *
                self.activity
            )


            painter.setPen(
                QPen(
                    QColor(
                        255,
                        134,
                        38,
                        alpha
                    ),
                    1.0
                )
            )


            painter.setBrush(
                Qt.NoBrush
            )


            painter.save()


            painter.translate(
                center
            )


            angle = (
                math.sin(
                    self.rotation
                    +
                    index
                    *
                    0.72
                )
                *
                58
            )


            painter.rotate(
                angle
            )


            painter.drawEllipse(
                QRectF(
                    -radius_x,
                    -radius_y,
                    radius_x * 2,
                    radius_y * 2
                )
            )


            painter.restore()


        # ====================================================
        # NODE POSITIONS
        # ====================================================

        node_positions = []


        for (
            angle,
            radius,
            layer,
            size
        ) in self.nodes:

            animated_angle = (
                angle
                +
                self.rotation
                *
                (
                    0.30
                    +
                    layer
                )
            )


            wobble = (
                1.0
                +
                0.045
                *
                math.sin(
                    self.pulse
                    +
                    angle
                    *
                    4
                )
            )


            x = (
                center.x()
                +
                math.cos(
                    animated_angle
                )
                *
                core_radius
                *
                radius
                *
                wobble
            )


            y = (
                center.y()
                +
                math.sin(
                    animated_angle
                )
                *
                core_radius
                *
                radius
                *
                0.80
                *
                wobble
            )


            node_positions.append(
                (
                    QPointF(
                        x,
                        y
                    ),
                    layer,
                    size
                )
            )


        # ====================================================
        # CONNECTIONS
        # ====================================================

        for index in range(
            len(
                node_positions
            )
        ):

            point_a = (
                node_positions[
                    index
                ][0]
            )


            for offset in (
                3,
                7,
                12
            ):

                target_index = (
                    index
                    +
                    offset
                ) % len(
                    node_positions
                )


                point_b = (
                    node_positions[
                        target_index
                    ][0]
                )


                alpha = int(
                    17
                    +
                    self.activity
                    *
                    55
                )


                painter.setPen(
                    QPen(
                        QColor(
                            255,
                            119,
                            18,
                            alpha
                        ),
                        0.65
                    )
                )


                painter.drawLine(
                    point_a,
                    point_b
                )


        # ====================================================
        # NODES
        # ====================================================

        for (
            point,
            layer,
            size
        ) in node_positions:

            flicker = (
                0.65
                +
                0.35
                *
                math.sin(
                    self.pulse
                    *
                    layer
                    *
                    2
                    +
                    point.x()
                    *
                    0.018
                )
            )


            node_size = (
                size
                *
                (
                    1.0
                    +
                    self.activity
                    *
                    0.75
                    *
                    flicker
                )
            )


            glow = QRadialGradient(
                point,
                node_size * 4
            )


            glow.setColorAt(
                0.0,
                QColor(
                    255,
                    226,
                    160,
                    240
                )
            )


            glow.setColorAt(
                0.28,
                QColor(
                    255,
                    135,
                    25,
                    170
                )
            )


            glow.setColorAt(
                1.0,
                QColor(
                    255,
                    70,
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
                node_size * 4,
                node_size * 4
            )


            painter.setBrush(
                QColor(
                    255,
                    214,
                    135,
                    235
                )
            )


            painter.drawEllipse(
                point,
                node_size,
                node_size
            )


        # ====================================================
        # CENTRAL CORE
        # ====================================================

        pulse_scale = (
            1.0
            +
            math.sin(
                self.pulse
            )
            *
            0.07
            *
            self.activity
        )


        core_glow = QRadialGradient(
            center,
            core_radius * 0.82
        )


        core_glow.setColorAt(
            0.0,
            QColor(
                255,
                244,
                209,
                255
            )
        )


        core_glow.setColorAt(
            0.12,
            QColor(
                255,
                183,
                70,
                245
            )
        )


        core_glow.setColorAt(
            0.46,
            QColor(
                255,
                102,
                10,
                160
            )
        )


        core_glow.setColorAt(
            1.0,
            QColor(
                255,
                60,
                0,
                0
            )
        )


        painter.setBrush(
            core_glow
        )


        painter.setPen(
            Qt.NoPen
        )


        painter.drawEllipse(
            center,
            core_radius
            *
            0.68
            *
            pulse_scale,
            core_radius
            *
            0.68
            *
            pulse_scale
        )


        # ====================================================
        # STATUS
        # ====================================================

        painter.setPen(
            QColor(
                255,
                177,
                78,
                235
            )
        )


        state_font = QFont()

        state_font.setPointSize(
            12
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
                height - 55,
                width,
                28
            ),
            Qt.AlignCenter,
            self.state
        )


        painter.setPen(
            QColor(
                130,
                136,
                145
            )
        )


        progress_font = QFont()

        progress_font.setPointSize(
            9
        )


        painter.setFont(
            progress_font
        )


        painter.drawText(
            QRectF(
                0,
                height - 30,
                width,
                22
            ),
            Qt.AlignCenter,
            f"{self.progress}%"
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

        self.detail_panel = QFrame()


        self.detail_panel.setObjectName(
            "detailPanel"
        )


        self.detail_panel.setMinimumWidth(
            340
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


        self.board_content_splitter.addWidget(
            self.detail_panel
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

                item = QTableWidgetItem(
                    str(
                        value
                    )
                )


                item.setData(
                    Qt.UserRole,
                    key
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


        self.core.set_signal_activity(

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
