#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package_path="${repo_root}/swift/Package.swift"

exec "${repo_root}/scripts/with_apple_toolchain.sh" open -a Xcode "${package_path}"
