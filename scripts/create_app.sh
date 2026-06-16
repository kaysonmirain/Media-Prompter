#!/usr/bin/env bash
set -e

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Desktop/Media Prompter.app"
ICON_PNG="$PROJECT/assets/app-icon.png"

"$PROJECT/.venv/bin/python3" "$PROJECT/scripts/build_app_icon.py"

mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>launcher</string>
  <key>CFBundleIconFile</key>
  <string>app-icon</string>
  <key>CFBundleIdentifier</key>
  <string>com.mediaprompter.app</string>
  <key>CFBundleName</key>
  <string>Media Prompter</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>LSMinimumSystemVersion</key>
  <string>10.13</string>
  <key>LSUIElement</key>
  <false/>
</dict>
</plist>
EOF

cp "$ICON_PNG" "$APP/Contents/Resources/app-icon.png"

cat > "$APP/Contents/MacOS/launcher" <<EOF
#!/usr/bin/env bash
cd "$PROJECT"
exec ./start.sh
EOF

chmod +x "$APP/Contents/MacOS/launcher"

if command -v fileicon >/dev/null 2>&1; then
  fileicon set "$APP" "$ICON_PNG"
elif command -v sips >/dev/null 2>&1; then
  mkdir -p /tmp/mp.iconset
  for s in 16 32 128 256 512; do
    sips -z $s $s "$ICON_PNG" --out "/tmp/mp.iconset/icon_${s}x${s}.png" >/dev/null
    s2=$((s * 2))
    sips -z $s2 $s2 "$ICON_PNG" --out "/tmp/mp.iconset/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns /tmp/mp.iconset -o "$APP/Contents/Resources/app-icon.icns"
  rm -rf /tmp/mp.iconset
fi

echo "Created $APP"
