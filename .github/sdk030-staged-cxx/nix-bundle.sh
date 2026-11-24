#!/bin/sh
set -eu
mkdir -p /root/.config/nix /build
cat > /root/.config/nix/nix.conf <<'EOF'
experimental-features = nix-command flakes
sandbox = false
build-users-group =
build-dir = /build
accept-flake-config = true
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE=
EOF
nix --version
GUFO_RUNTIME=$(nix build -L --no-link --no-write-lock-file --no-update-lock-file --print-out-paths path:/gufo#loom-runtime)
EXTENSION_RUNTIME=$(nix build -L --no-link --no-write-lock-file --no-update-lock-file --print-out-paths path:/extension#runtime)
export GUFO_RUNTIME EXTENSION_RUNTIME
nix path-info --json "$GUFO_RUNTIME" "$EXTENSION_RUNTIME" > /proof/nix-outputs.json

toolchain='let
  f = builtins.getFlake "path:/gufo";
  p = f.inputs.nixpkgs.legacyPackages.x86_64-linux;
in p.mkShell { packages = [ p.gcc p.patchelf ]; }'
nix-shell -E "$toolchain" --run 'bash /recipes/package-nix-probes.sh'
chown -R "$PROOF_OWNER" /proof/embed
