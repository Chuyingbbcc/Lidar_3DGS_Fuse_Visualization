#include "MainWindow.h"
#include "OpenGLWidget.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QFont>
#include <QElapsedTimer>
#include <QPalette>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QTimer>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

int main(int argc, char* argv[])
{
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    const bool enable_experimental_gpu_reorder =
        qEnvironmentVariableIntValue("LIDAR_3DGS_GPU_REORDER") == 1;
    format.setVersion(enable_experimental_gpu_reorder ? 4 : 3,
                      enable_experimental_gpu_reorder ? 3 : 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    // Gaussian footprints are analytically antialiased in the fragment
    // shader. Multisampling this pass adds cost and softens the resolved image.
    format.setSamples(0);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication application(argc, argv);
    application.setApplicationName("chabot-gpt");
    application.setOrganizationName("Lidar3DGS");
    application.setStyle(QStyleFactory::create("Fusion"));
    application.setFont(QFont("Noto Sans", 10));

    QPalette palette;
    palette.setColor(QPalette::Window, QColor(34, 36, 40));
    palette.setColor(QPalette::WindowText, QColor(232, 234, 237));
    palette.setColor(QPalette::Base, QColor(24, 26, 29));
    palette.setColor(QPalette::AlternateBase, QColor(42, 45, 50));
    palette.setColor(QPalette::ToolTipBase, QColor(46, 49, 54));
    palette.setColor(QPalette::ToolTipText, QColor(245, 246, 247));
    palette.setColor(QPalette::Text, QColor(226, 229, 232));
    palette.setColor(QPalette::Button, QColor(52, 55, 61));
    palette.setColor(QPalette::ButtonText, QColor(238, 240, 242));
    palette.setColor(QPalette::BrightText, QColor(255, 110, 110));
    palette.setColor(QPalette::Highlight, QColor(54, 160, 154));
    palette.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
    palette.setColor(QPalette::Link, QColor(91, 205, 198));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor(116, 120, 126));
    palette.setColor(
        QPalette::Disabled, QPalette::ButtonText, QColor(116, 120, 126));
    application.setPalette(palette);

    QCommandLineParser command_line;
    command_line.setApplicationDescription("chabot-gpt");
    command_line.addHelpOption();
    QCommandLineOption demo_option(
        "demo", "Launch the preloaded, presentation-focused demo UI.");
    command_line.addOption(demo_option);
    QCommandLineOption benchmark_load_option(
        "benchmark-load",
        "Load the demo and exit after its first rendered frame.");
    command_line.addOption(benchmark_load_option);
    QCommandLineOption benchmark_render_option(
        "benchmark-render",
        "Load the demo, benchmark continuous camera rotation, and exit.");
    command_line.addOption(benchmark_render_option);
    QCommandLineOption lidar_option(
        "lidar", "Load a colored LiDAR PLY at startup.", "path");
    command_line.addOption(lidar_option);
    command_line.process(application);

    application.setStyleSheet(R"QSS(
        QWidget {
            color: #e8eaed;
            font-family: "Noto Sans", "DejaVu Sans", sans-serif;
            font-size: 10pt;
        }
        QMainWindow, QDialog { background: #222428; }
        QFrame[frameShape="6"] {
            background: #2a2d32;
            border: 1px solid #444850;
            border-radius: 8px;
        }
        QLabel { background: transparent; }
        QPushButton, QLineEdit, QPlainTextEdit,
        QComboBox, QSpinBox, QDoubleSpinBox {
            min-height: 25px;
            padding: 3px 8px;
            color: #eef0f2;
            background: #383c42;
            border: 1px solid #555a63;
            border-radius: 5px;
        }
        QPushButton:hover, QLineEdit:hover, QPlainTextEdit:hover, QComboBox:hover,
        QSpinBox:hover, QDoubleSpinBox:hover {
            background: #444950;
            border-color: #6d747e;
        }
        QPushButton:pressed {
            background: #2d817c;
            border-color: #48aaa4;
        }
        QPlainTextEdit#navigationPrompt {
            padding: 10px 12px;
            color: #f1f3f4;
            background: #303338;
            border: 1px solid #565b63;
            border-radius: 12px;
            selection-background-color: #369f99;
        }
        QPlainTextEdit#navigationPrompt:focus {
            background: #34383d;
            border: 1px solid #5fc0ba;
        }
        QPushButton#navigateAction {
            min-height: 30px;
            font-weight: 600;
            background: #368f89;
            border-color: #55b5ae;
        }
        QPushButton#navigateAction:hover { background: #40a39c; }
        QPushButton:disabled {
            color: #74787e;
            background: #303237;
            border-color: #3d4046;
        }
        QComboBox::drop-down {
            width: 24px;
            border: 0;
        }
        QComboBox QAbstractItemView {
            color: #eef0f2;
            background: #303339;
            border: 1px solid #555a63;
            selection-background-color: #369f99;
        }
        QTabWidget::pane {
            background: #2a2d32;
            border: 1px solid #484c54;
            border-radius: 5px;
            top: -1px;
        }
        QTabBar::tab {
            padding: 7px 9px;
            color: #aeb3ba;
            background: #292c31;
            border: 1px solid #42464d;
            border-bottom: 0;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        QTabBar::tab:selected {
            color: #ffffff;
            background: #383c42;
            border-top: 2px solid #48aaa4;
        }
        QTabBar::tab:hover:!selected { background: #31343a; }
        QCheckBox { spacing: 7px; }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #686e77;
            border-radius: 3px;
            background: #25272b;
        }
        QCheckBox::indicator:checked {
            background: #369f99;
            border-color: #5fc0ba;
        }
        QToolTip {
            color: #f5f6f7;
            background: #36393f;
            border: 1px solid #60656e;
            padding: 4px;
        }
    )QSS");

    QString lidar_path = command_line.value(lidar_option);
    if (lidar_path.isEmpty() && !command_line.isSet(demo_option))
        lidar_path = QStringLiteral(PROJECT_ROOT_DIR) +
            "/data/kitti/merged_lidar_rgb.ply";
    const bool benchmark_load = command_line.isSet(benchmark_load_option);
    const bool benchmark_render = command_line.isSet(benchmark_render_option);
    const bool demo_mode =
        command_line.isSet(demo_option) || benchmark_load || benchmark_render;
    MainWindow window(demo_mode, nullptr, lidar_path);
    window.show();
    if (benchmark_load || benchmark_render) {
        auto timer = std::make_shared<QElapsedTimer>();
        auto loading_complete = std::make_shared<bool>(false);
        auto frame_times = std::make_shared<std::vector<double>>();
        auto frame_timer = std::make_shared<QElapsedTimer>();
        auto warmup_frames = std::make_shared<int>(0);
        OpenGLWidget* viewer = window.findChild<OpenGLWidget*>();
        if (!viewer) return 1;
        QObject::connect(viewer, &QOpenGLWidget::frameSwapped, &application,
            [timer, frame_timer, frame_times, warmup_frames, loading_complete,
             benchmark_render, viewer, &application]() {
                if (!*loading_complete) return;
                if (!benchmark_render) {
                    qInfo("BENCHMARK demo_load_to_first_frame_ms=%lld",
                          static_cast<long long>(timer->elapsed()));
                    application.quit();
                    return;
                }
                if (!frame_timer->isValid()) {
                    qInfo("BENCHMARK demo_load_to_first_frame_ms=%lld",
                          static_cast<long long>(timer->elapsed()));
                    frame_timer->start();
                } else {
                    const double milliseconds = frame_timer->nsecsElapsed() / 1.0e6;
                    frame_timer->restart();
                    if (*warmup_frames >= 10) frame_times->push_back(milliseconds);
                    else ++*warmup_frames;
                }
                if (frame_times->size() >= 120) {
                    std::vector<double> sorted = *frame_times;
                    std::sort(sorted.begin(), sorted.end());
                    const double mean = std::accumulate(
                        sorted.begin(), sorted.end(), 0.0) / sorted.size();
                    const std::size_t p95_index = static_cast<std::size_t>(
                        std::ceil(0.95 * sorted.size())) - 1;
                    qInfo("BENCHMARK rotation_frames=%zu average_fps=%.3f mean_frame_ms=%.3f p95_frame_ms=%.3f",
                          sorted.size(), 1000.0 / mean, mean, sorted[p95_index]);
                    application.quit();
                    return;
                }
                QMatrix4x4 transform;
                const float angle = static_cast<float>(
                    *warmup_frames + frame_times->size());
                transform.rotate(angle, 0.0f, 1.0f, 0.0f);
                viewer->setInteractionTransform(transform, angle, 0.0f);
            });
        QTimer::singleShot(0, &window,
            [&window, viewer, timer, loading_complete]() {
                timer->start();
                window.loadDemoAssets();
                *loading_complete = true;
                viewer->update();
            });
    } else if (demo_mode) {
        QTimer::singleShot(0, &window, &MainWindow::loadDemoAssets);
    }
    return application.exec();
}
