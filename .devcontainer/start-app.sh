#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../genai-nlp-annotation-tool"

if python - <<'PY'
import socket
with socket.socket() as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", 8501)) == 0 else 1)
PY
then
  echo "Annotation product is already listening on port 8501."
else
  nohup python -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    >/tmp/annotation-product.log 2>&1 &
  echo "Annotation product started. Logs: /tmp/annotation-product.log"
fi
