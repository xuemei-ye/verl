#!/usr/bin/env bash
# Install the pinned E2B SDK from the shared offline wheelhouse and verify the
# real E2B create -> execute -> close path. Run once in every training container.
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3}
VERL_ROOT=${VERL_ROOT:-/root/work/yexuemei/verl}
E2B_WHEEL_DIR=${E2B_WHEEL_DIR:-/root/work/wheels/e2b-2.50.0-cp311-aarch64}
E2B_SDK_VERSION=${E2B_SDK_VERSION:-2.50.0}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-${VERL_ROOT}/examples/e2b_agent/tool_config.yaml}

E2B_DOMAIN=${E2B_DOMAIN:-dev-e2b.xiaobei.top}
E2B_API_URL=${E2B_API_URL:-https://dev-e2b.xiaobei.top}
E2B_PROXY=${E2B_PROXY:-http://10.202.9.50:3928}
E2B_API_KEY=${E2B_API_KEY:-e2b_4291fc7c3dabdebbd3c0d12151d4e3762a55}


fail() {
    echo "[e2b-setup] ERROR: $*" >&2
    exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python interpreter not found: ${PYTHON_BIN}"
[[ -d "${VERL_ROOT}/verl" ]] || fail "verl source tree not found: ${VERL_ROOT}"
[[ -d "${E2B_WHEEL_DIR}" ]] || fail "E2B wheel directory not found: ${E2B_WHEEL_DIR}"
[[ -f "${TOOL_CONFIG_PATH}" ]] || fail "tool config not found: ${TOOL_CONFIG_PATH}"
[[ -n "${E2B_API_KEY:-}" ]] || fail "E2B_API_KEY must be injected through the Jiutian secret environment"

# A caller may source this script after assigning E2B_API_KEY as a regular
# shell variable. Export it explicitly so the Python preflight can read it.
export E2B_API_KEY

echo "[e2b-setup] python=${PYTHON_BIN}"
echo "[e2b-setup] verl_root=${VERL_ROOT}"
echo "[e2b-setup] wheel_dir=${E2B_WHEEL_DIR}"
echo "[e2b-setup] api_url=${E2B_API_URL}"
echo "[e2b-setup] domain=${E2B_DOMAIN}"
echo "[e2b-setup] proxy=${E2B_PROXY}"

"${PYTHON_BIN}" -m pip install \
    --no-index \
    --find-links="${E2B_WHEEL_DIR}" \
    "e2b==${E2B_SDK_VERSION}"

installed_version=$("${PYTHON_BIN}" -c 'import importlib.metadata; print(importlib.metadata.version("e2b"))')
[[ "${installed_version}" == "${E2B_SDK_VERSION}" ]] \
    || fail "installed E2B version ${installed_version}, expected ${E2B_SDK_VERSION}"

export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export E2B_DOMAIN
export E2B_API_URL
export E2B_PROXY
unset E2B_ACCESS_TOKEN

cd "${VERL_ROOT}"

echo "[e2b-setup] E2B SDK ${installed_version} installed; starting preflight"
"${PYTHON_BIN}" -m verl.tools.e2b_preflight \
    --tool-config "${TOOL_CONFIG_PATH}"

echo "[e2b-setup] READY: offline install and E2B preflight succeeded"
