#!/usr/bin/env bash
set -euo pipefail

# Installer for NVIDIA drivers, CUDA Toolkit 12.6, and cuDNN 9 (matching JCuda/JCudnn 12.6.0).
# Supports Ubuntu (20.04 / 22.04 / 24.04) and Debian on x86_64 and aarch64.

CUDA_VERSION_DASH="12-6"
CUDA_VERSION_DOT="12.6"
CUDNN_PACKAGE="cudnn9-cuda-12"
DRIVER_PACKAGE="nvidia-open"
INSTALL_DRIVER="auto" # auto | yes | no
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Installs NVIDIA CUDA Toolkit ${CUDA_VERSION_DOT} and cuDNN 9 (required by JCuda/JCudnn 12.6.0),
and optionally installs the NVIDIA GPU driver.

Options:
  --install-driver         Force installation of the NVIDIA GPU driver (${DRIVER_PACKAGE})
  --skip-driver            Skip NVIDIA GPU driver installation (toolkit + cuDNN only)
  --driver-package <pkg>   Driver package to install (default: ${DRIVER_PACKAGE}; e.g. cuda-drivers)
  --cuda-version <X-Y>     CUDA toolkit version with dash (default: ${CUDA_VERSION_DASH})
  --cudnn-package <pkg>    cuDNN package name (default: ${CUDNN_PACKAGE})
  --dry-run                Print commands without executing them
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-driver)
      INSTALL_DRIVER="yes"
      shift
      ;;
    --skip-driver)
      INSTALL_DRIVER="no"
      shift
      ;;
    --driver-package)
      DRIVER_PACKAGE="$2"
      shift 2
      ;;
    --cuda-version)
      CUDA_VERSION_DASH="$2"
      CUDA_VERSION_DOT="${CUDA_VERSION_DASH//-/.}"
      shift 2
      ;;
    --cudnn-package)
      CUDNN_PACKAGE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    echo "+ $*"
    "$@"
  fi
}

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  SUDO="sudo"
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Error: /etc/os-release not found. This script supports Ubuntu/Debian Linux." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

UNAME_ARCH="$(uname -m)"
case "$UNAME_ARCH" in
  x86_64)  REPO_ARCH="x86_64" ;;
  aarch64) REPO_ARCH="sbsa" ;;
  *)
    echo "Error: Unsupported architecture: ${UNAME_ARCH}" >&2
    exit 1
    ;;
esac

case "${ID:-}" in
  ubuntu)
    DISTRO="ubuntu${VERSION_ID//./}"
    ;;
  debian)
    DISTRO="debian${VERSION_ID%%.*}"
    ;;
  *)
    echo "Error: Unsupported distribution '${ID:-unknown}'. Supported: Ubuntu, Debian." >&2
    exit 1
    ;;
esac

echo "Detected OS: ${PRETTY_NAME:-$DISTRO} (${DISTRO}/${REPO_ARCH})"

# Check existing NVIDIA driver
DRIVER_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DRIVER_PRESENT=1
  echo "Existing NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "No active NVIDIA driver detected via nvidia-smi."
fi

SHOULD_INSTALL_DRIVER=0
if [[ "$INSTALL_DRIVER" == "yes" ]]; then
  SHOULD_INSTALL_DRIVER=1
elif [[ "$INSTALL_DRIVER" == "auto" && "$DRIVER_PRESENT" -eq 0 ]]; then
  SHOULD_INSTALL_DRIVER=1
fi

# Step 1: Configure official NVIDIA CUDA apt repository keyring
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
KEYRING_DEB="${TMP_DIR}/cuda-keyring_1.1-1_all.deb"

echo "Configuring NVIDIA CUDA apt repository (${DISTRO}/${REPO_ARCH})..."
if command -v curl >/dev/null 2>&1; then
  run curl -fsSL -o "$KEYRING_DEB" "$KEYRING_URL"
elif command -v wget >/dev/null 2>&1; then
  run wget -q -O "$KEYRING_DEB" "$KEYRING_URL"
else
  echo "Error: Neither curl nor wget is installed." >&2
  exit 1
fi

run $SUDO dpkg -i "$KEYRING_DEB"
run $SUDO apt-get update

# Step 2: Install NVIDIA driver if needed/requested
if [[ "$SHOULD_INSTALL_DRIVER" -eq 1 ]]; then
  echo "Installing NVIDIA driver package: ${DRIVER_PACKAGE}..."
  run $SUDO apt-get install -y "$DRIVER_PACKAGE"
else
  echo "Keeping existing NVIDIA driver (pass --install-driver to override)."
fi

# Step 3: Install CUDA Toolkit and cuDNN 9
TOOLKIT_PKG="cuda-toolkit-${CUDA_VERSION_DASH}"
echo "Installing ${TOOLKIT_PKG} and ${CUDNN_PACKAGE}..."
run $SUDO apt-get install -y "$TOOLKIT_PKG" "$CUDNN_PACKAGE"

# Step 4: Configure dynamic linker and shell environment
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION_DOT}"
LD_CONF_FILE="/etc/ld.so.conf.d/cuda-${CUDA_VERSION_DASH}.conf"
PROFILE_FILE="/etc/profile.d/cuda-${CUDA_VERSION_DASH}.sh"

echo "Configuring ldconfig (${LD_CONF_FILE}) and PATH (${PROFILE_FILE})..."
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] write ${CUDA_HOME}/lib64 to ${LD_CONF_FILE}"
  echo "[dry-run] $SUDO ldconfig"
  echo "[dry-run] write CUDA_HOME/PATH exports to ${PROFILE_FILE}"
else
  echo "${CUDA_HOME}/lib64" | $SUDO tee "$LD_CONF_FILE" >/dev/null
  $SUDO ldconfig
  $SUDO tee "$PROFILE_FILE" >/dev/null <<PROFILE
export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
PROFILE
fi

# Step 5: Verify installation
echo ""
echo "=== Verification ==="
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Skipping verification."
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  fi
  if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    "${CUDA_HOME}/bin/nvcc" --version
  fi
  echo "Checking shared libraries required by JCuda/JCudnn ${CUDA_VERSION_DOT}:"
  for lib in libcuda.so.1 libcublas.so.12 libnvrtc.so.12 libcudnn.so.9; do
    if ldconfig -p | grep -q "$lib"; then
      echo "  [OK] $lib"
    else
      echo "  [MISSING] $lib" >&2
    fi
  done
fi

if [[ "$SHOULD_INSTALL_DRIVER" -eq 1 ]]; then
  echo ""
  echo "NOTE: A new NVIDIA driver was installed. Reboot your system before running GPU workloads."
fi
