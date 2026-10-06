#!/usr/bin/env bash
# protein local — 원샷 설치·실행 (Linux + NVIDIA GPU)
#   bash setup.sh          # Python 확인 + Boltz/ProteinMPNN 환경·가중치 확인 + GPU + LLM(선택) + selftest + 웹 서버
#   bash setup.sh stop
# 환경변수: BOLTZ_ENV (~/miniforge3/envs/boltz), BOLTZ_CACHE (~/.boltz), MPNN_ENV (~/miniforge3/envs/proteinmpnn),
#           CUDA_VISIBLE_DEVICES 또는 PROTEIN_GPUS (쓸 GPU 목록),
#           MAX_RES (1500), JOB_TIMEOUT (3600초), PORT (8781), WORKSPACE (작업·결과), LLM_BASE_URL / LLM_API / LLM_MODEL (결과 해설용, 선택)
# 웹 서버는 표준 라이브러리만 쓴다. 예측은 Boltz conda 환경에서 runner.py, 서열 설계는 proteinmpnn 환경에서 runner_mpnn.py (MSA 서버 호출 없음).
set -euo pipefail
PORT="${PORT:-8781}"
if [ -t 1 ]; then B=$'\033[1m'; D=$'\033[2m'; C=$'\033[36m'; G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; N=$'\033[0m'; else B= D= C= G= R= Y= N=; fi
STEP=0; step() { STEP=$((STEP+1)); printf '  %s[%d/8]%s %s%-14s%s ' "$D" "$STEP" "$N" "$B" "$1" "$N"; }
ok() { printf '%s✔%s %s\n' "$G" "$N" "${1:-}"; }; skip() { printf '%s–%s %s\n' "$D" "$N" "${1:-}"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "${1:-}"; }
die() { printf '%s✘ %s%s\n\n' "$R" "$*" "$N" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }; probe() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }
wait_for() { for _ in $(seq 1 "${2:-30}"); do probe "$1" && return 0; sleep 1; done; return 1; }
printf '\n%s  protein local%s  구조 예측 · 결합 친화도 · 서열 설계 — Boltz-2 + ProteinMPNN · 3D 뷰어 동봉 · 폐쇄망(MSA 없음)\n\n' "$B" "$N"

step "OS 감지"; case "$(uname -s)" in Linux*) OS=linux ;; Darwin*) OS=mac ;; *) die "지원하지 않는 OS: $(uname -s) (GPU 서버용)" ;; esac; ok "$OS ($(uname -m))"
step "패키지 확보"; cd "$(dirname "${BASH_SOURCE[0]}")"; [ -f app.py ] || die "app.py 가 없습니다"; [ -f static/3Dmol-min.js ] || die "static/3Dmol-min.js 가 없습니다"; ok "$(pwd)"
if [ "${1:-}" = "stop" ]; then [ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null && rm -f .server.pid && ok "웹 서버 종료" || skip "실행 중인 서버 없음"; exit 0; fi

step "Python"
PY=""; for c in python3 python; do has "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { PY=$c; break; }; done
[ -n "$PY" ] || die "Python 3.9+ 가 없습니다"
ok "$($PY --version 2>&1) (웹 서버 — 외부 패키지 없음)"

step "Boltz 엔진"
export BOLTZ_ENV="${BOLTZ_ENV:-$HOME/miniforge3/envs/boltz}" BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/.boltz}"
[ -x "$BOLTZ_ENV/bin/python" ] || die "Boltz 환경이 없습니다: $BOLTZ_ENV (conda create -n boltz python=3.11 && pip install boltz — README 참고)"
BV=$("$BOLTZ_ENV/bin/python" -c 'import importlib.metadata as m; print(m.version("boltz"))' 2>/dev/null) || die "$BOLTZ_ENV 에 boltz 패키지가 없습니다"
W=""; [ -f "$BOLTZ_CACHE/boltz2_conf.ckpt" ] && W="$W boltz2"; [ -f "$BOLTZ_CACHE/boltz1_conf.ckpt" ] && W="$W boltz1"
[ -n "$W" ] && [ -d "$BOLTZ_CACHE/mols" ] || die "가중치가 없습니다: $BOLTZ_CACHE/{boltz2_conf.ckpt,mols/} (README — 인터넷 되는 곳에서 받아 복사)"
ok "boltz $BV · 가중치:$W$( [ -f "$BOLTZ_CACHE/boltz2_aff.ckpt" ] && echo ' + 친화도')"

step "ProteinMPNN"
export MPNN_ENV="${MPNN_ENV:-$HOME/miniforge3/envs/proteinmpnn}"
if [ -x "$MPNN_ENV/bin/python" ] && ls "$MPNN_ENV"/lib/python3*/site-packages/ligandmpnn/data/model_params/proteinmpnn_v_48_020.pt >/dev/null 2>&1; then
  ok "ligandmpnn $("$MPNN_ENV/bin/python" -c 'import importlib.metadata as m; print(m.version("ligandmpnn"))' 2>/dev/null) (가중치 동봉)"
else warn "없음 → '서열 설계' 탭만 꺼짐 (구조 예측·친화도는 그대로). pip install ligandmpnn 한 conda 환경을 MPNN_ENV 로 지정"; fi

step "GPU"
if has nvidia-smi; then
  GSEL="${PROTEIN_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"   # G 는 색 변수라 쓰지 말 것
  if [ -n "$GSEL" ]; then ok "GPU $GSEL 만 사용 — $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader -i "$GSEL" 2>/dev/null | tr '\n' ' ' | sed 's/, / 여유 /g')"
  else warn "CUDA_VISIBLE_DEVICES 가 비어 있어 모든 GPU 중 여유가 큰 1장을 씁니다 (공용 서버면 PROTEIN_GPUS=2,3 처럼 지정)"; fi
else warn "nvidia-smi 없음 — GPU 없이 예측은 실패합니다 (화면·selftest 는 동작)"; fi

step "LLM (선택)"
export LLM_API="${LLM_API:-}" LLM_BASE_URL="${LLM_BASE_URL:-}" LLM_MODEL="${LLM_MODEL:-}"
if [ -n "$LLM_BASE_URL" ]; then [ -n "$LLM_API" ] || { case "$LLM_BASE_URL" in *1143*) LLM_API=ollama ;; *) LLM_API=openai ;; esac; }
elif probe http://localhost:11434/api/tags; then LLM_API=ollama LLM_BASE_URL=http://localhost:11434
else for p in 8000 1234 8080; do probe "http://localhost:$p/v1/models" && { LLM_API=openai LLM_BASE_URL="http://localhost:$p/v1"; break; }; done; fi
if [ -n "$LLM_BASE_URL" ] && [ -z "$LLM_MODEL" ]; then
  if [ "$LLM_API" = ollama ]; then LLM_MODEL=$(curl -fsS -m 3 "$LLM_BASE_URL/api/tags" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["models"][0]["name"])' 2>/dev/null || true)
  else LLM_MODEL=$(curl -fsS -m 3 "$LLM_BASE_URL/models" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true); fi
fi
if [ -n "$LLM_BASE_URL" ]; then ok "$LLM_API $LLM_BASE_URL ${LLM_MODEL:-}"; else skip "없음 → '결과 해설'만 꺼짐 (예측은 그대로)"; LLM_API=ollama; fi

step "자가검증"; env -u WORKSPACE "$PY" selftest.py >/dev/null 2>&1 || die "selftest 실패 (python3 selftest.py 로 확인)"; ok "검증·큐·결과 파싱·친화도 순위·설계·RMSD·HTTP (임시 폴더, 가짜 엔진)"

[ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null || true
LLM_API=$LLM_API LLM_BASE_URL=$LLM_BASE_URL LLM_MODEL=$LLM_MODEL MPNN_ENV=$MPNN_ENV PORT=$PORT nohup "$PY" app.py > server.log 2>&1 & echo $! > .server.pid
wait_for "http://localhost:$PORT/api/health" 20 || { cat server.log; die "웹 서버 기동 실패 (server.log 확인)"; }
URL="http://localhost:$PORT"
printf '\n  %s준비 완료%s  %s%s%s   종료: bash setup.sh stop   로그: server.log\n\n' "$B" "$N" "$C" "$URL" "$N"
