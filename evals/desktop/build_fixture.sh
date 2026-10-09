#!/usr/bin/env bash
set -euo pipefail

app="${1:-/tmp/S1ADocumentFixture.app}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$app/Contents/MacOS"
swiftc -O "$root/evals/desktop/fixture.swift" -framework AppKit -o "$app/Contents/MacOS/S1ADocumentFixture"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>S1ADocumentFixture</string>
  <key>CFBundleIdentifier</key><string>org.thinkflowlab.s1a.document-fixture</string>
  <key>CFBundleName</key><string>S1ADocumentFixture</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
</dict></plist>
PLIST
printf '%s\n' "$app"
