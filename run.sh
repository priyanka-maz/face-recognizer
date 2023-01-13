#!/bin/sh

if [ "$XDG_SESSION_TYPE" == "wayland" ]; then
  QT_QPA_PLATFORM=wayland python face.py
else
  QT_QPA_PLATFORM=xcb python face.py
fi
