#!/usr/bin/env bash
set -euo pipefail

PORT="${TRTLLM_PORT:-8000}"
MODEL="${TRTLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

pretty_json() {
    if command -v jq >/dev/null 2>&1; then
        jq .
    else
        python3 -m json.tool
    fi
}

curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"用一句中文说明 TensorRT-LLM 的作用。\"}
      ],
      \"max_tokens\": 128,
      \"temperature\": 0.2
    }" | pretty_json
