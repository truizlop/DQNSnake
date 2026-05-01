#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

# Remove known Conda path segments that can shadow Apple toolchain binaries (e.g. ld).
clean_path=":${PATH}:"
for segment in "/opt/anaconda3/bin" "/opt/anaconda3/condabin"; do
  clean_path="${clean_path//:${segment}:/":"}"
done
clean_path="${clean_path#:}"
clean_path="${clean_path%:}"

# Ensure Apple developer toolchain and system binaries resolve first.
if developer_dir="$(xcode-select -p 2>/dev/null)"; then
  if [[ -d "${developer_dir}/Toolchains/XcodeDefault.xctoolchain/usr/bin" ]]; then
    clean_path="${developer_dir}/Toolchains/XcodeDefault.xctoolchain/usr/bin:/usr/bin:/bin:${clean_path}"
  else
    clean_path="/usr/bin:/bin:${clean_path}"
  fi
else
  clean_path="/usr/bin:/bin:${clean_path}"
fi

export PATH="${clean_path}"
exec "$@"
