#!/usr/bin/env bash
set -euo pipefail
sha256sum "$GUFO_RUNTIME/lib/libloom_text_embed.so" "$EXTENSION_RUNTIME/lib/libloom_text_embed.so" > /proof/runtime-sha256.txt

package() {
  local name=$1 sdk=$2 out="/proof/embed/$1" loader
  mkdir -p "$out/lib"
  gcc -I "$sdk/include" /extension/validation/devlab/embed_probe.c \
    -L "$sdk/lib" -Wl,-rpath,"$sdk/lib" -lloom_text_embed -lm -o "$out/logits-probe"
  gcc -I "$sdk/include" /recipes/controller-abi.c \
    -L "$sdk/lib" -Wl,-rpath,"$sdk/lib" -lloom_text_embed -o "$out/controller-abi"
  loader=$(patchelf --print-interpreter "$out/controller-abi")
  cp -L "$loader" "$out/lib/ld-linux-x86-64.so.2"
  {
    "$loader" --list "$out/controller-abi"
    "$loader" --list "$out/logits-probe"
  } | awk '/=> \// { print $3 }' | sort -u | while IFS= read -r lib; do
    cp -L "$lib" "$out/lib/"
  done
  test -s "$out/lib/libloom_text_embed.so"
  "$out/lib/ld-linux-x86-64.so.2" --library-path "$out/lib" --list "$out/controller-abi" \
    > "/proof/$name-dependencies.txt"
}
package gufo "$GUFO_RUNTIME"
package extension "$EXTENSION_RUNTIME"
