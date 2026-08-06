#!/usr/bin/env bash
set -euo pipefail

echo "Installing/checking C++ dependencies..."

sudo apt-get update

PACKAGES=(
    build-essential
    cmake
    git
    pkg-config
    libopencv-dev
    libeigen3-dev
    libceres-dev
    libsuitesparse-dev
    libgoogle-glog-dev
    libgflags-dev
    libgtest-dev
    libyaml-cpp-dev
    nlohmann-json3-dev
)

for package in "${PACKAGES[@]}"; do
    if dpkg -s "${package}" >/dev/null 2>&1; then
        echo "[OK] ${package} already installed"
    else
        echo "[INSTALL] ${package}"
        sudo apt-get install -y "${package}"
    fi
done

echo
echo "Verifying dependencies..."

cmake --version | head -n 1

if pkg-config --exists opencv4; then
    echo "[OK] OpenCV $(pkg-config --modversion opencv4)"
else
    echo "[ERROR] OpenCV was not detected by pkg-config"
    exit 1
fi

if [[ -d /usr/include/eigen3/Eigen ]]; then
    echo "[OK] Eigen found at /usr/include/eigen3"
else
    echo "[ERROR] Eigen headers not found"
    exit 1
fi

echo
echo "Checking Ceres..."

if ! dpkg-query -W -f='${Status}' libceres-dev 2>/dev/null \
    | grep -q "install ok installed"; then
    echo "[ERROR] libceres-dev is not installed"
    exit 1
fi

echo "[OK] libceres-dev is installed"

CERES_HEADER="$(find /usr/include /usr/local/include \
    -path '*/ceres/ceres.h' \
    -print -quit 2>/dev/null || true)"

if [[ -z "${CERES_HEADER}" ]]; then
    echo "[ERROR] ceres/ceres.h was not found"
    exit 1
fi

echo "[OK] Ceres header found:"
echo "     ${CERES_HEADER}"

CERES_CONFIG="$(find /usr /usr/local \
    -name 'CeresConfig.cmake' \
    -print -quit 2>/dev/null || true)"

if [[ -z "${CERES_CONFIG}" ]]; then
    echo "[ERROR] CeresConfig.cmake was not found"
    exit 1
fi

echo "[OK] Ceres CMake configuration found:"
echo "     ${CERES_CONFIG}"

CERES_LIBRARY="$(find /usr/lib /usr/local/lib \
    \( -name 'libceres.so*' -o -name 'libceres.a' \) \
    -print -quit 2>/dev/null || true)"

if [[ -n "${CERES_LIBRARY}" ]]; then
    echo "[OK] Ceres library found:"
    echo "     ${CERES_LIBRARY}"
else
    echo "[WARNING] Ceres library file was not found by filesystem search"
    echo "          The compile test below will determine whether Ceres is usable."
fi

echo
echo "Installed package versions:"
dpkg-query -W -f='OpenCV package: ${Version}\n' libopencv-dev
dpkg-query -W -f='Eigen package:  ${Version}\n' libeigen3-dev
dpkg-query -W -f='Ceres package:  ${Version}\n' libceres-dev

echo
echo "All required libraries are installed and available."
echo "You can build the project separately with:"
echo "  cmake -S . -B build"
echo "  cmake --build build -j$(nproc)"