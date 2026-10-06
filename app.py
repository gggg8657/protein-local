#!/usr/bin/env python3
"""protein local — 단백질 구조 예측. 서열(FASTA)을 넣으면 Boltz-2 로 3D 구조·pLDDT·PAE 를 보여 주고 PDB/mmCIF 로 내려준다.

  python3 app.py            # http://localhost:8781
폐쇄망 전제: MSA 서버를 부르지 않는다(단일 서열 모드). 3D 뷰어(3Dmol.js)는 static/ 에 동봉.
엔진: BOLTZ_ENV(기본 ~/miniforge3/envs/boltz) 의 python 으로 runner.py 를 서브프로세스 실행. 가중치는 BOLTZ_CACHE(~/.boltz).
GPU: CUDA_VISIBLE_DEVICES(또는 PROTEIN_GPUS) 목록 중 여유 메모리가 가장 큰 1장. 작업은 한 번에 1개, 나머지는 대기열.
LLM(선택): 결과 해설 — LLM_API=ollama|openai, LLM_BASE_URL, LLM_MODEL
"""
import array
import ast
import glob
import math
import datetime
import io
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.join(ROOT, "_workspace")  # 포털이 AGENT_DATA/<도구> 로 모아 줌
LLM_API = os.environ.get("LLM_API", "ollama")
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1" if LLM_API == "openai" else "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
PORT = int(os.environ.get("PORT", "8781"))
BOLTZ_ENV = os.path.expanduser(os.environ.get("BOLTZ_ENV", "~/miniforge3/envs/boltz"))
BOLTZ_CACHE = os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz"))
GPUS = [g.strip() for g in (os.environ.get("PROTEIN_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if g.strip()]
MAX_RES = int(os.environ.get("MAX_RES", "1500"))          # 잔기 + 리간드 원자 합 (H100 에서 ~1500 토큰 ≈ 30GB 안팎)
MAX_CHAINS = int(os.environ.get("MAX_CHAINS", "10"))
MAX_SAMPLES = 5
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "3600"))  # 초
PAE_MAX = 500                                              # 화면용 PAE 행렬 최대 크기 (넘으면 평균 풀링)
AA = set("ACDEFGHIKLMNPQRSTVWY")
MODES = {  # recycling, sampling steps
    "fast": {"label": "빠름", "recycling": 1, "steps": 50},
    "standard": {"label": "표준", "recycling": 3, "steps": 200},
    "precise": {"label": "정밀", "recycling": 10, "steps": 200},
}
EXAMPLES = [
    {"name": "Trp-cage (20aa)", "note": "가장 작은 접힘 단백질 TC5b — 몇 초", "fasta": ">TrpCage_TC5b\nNLYIQWLKDGGPSSGRPPPS"},
    {"name": "유비퀴틴 (76aa)", "note": "사람 유비퀴틴 (PDB 1UBQ)",
     "fasta": ">Ubiquitin_human\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"},
    {"name": "인슐린 A+B 사슬 (복합체)", "note": "사람 인슐린 — 2개 사슬 (21 + 30aa)",
     "fasta": ">Insulin_A\nGIVEQCCTSICSLYQLENYCN\n>Insulin_B\nFVNQHLCGSHLVEALYLVCGERGFFYTPKT"},
    {"name": "GFP (238aa)", "note": "해파리 녹색형광단백질 (UniProt P42212) — MSA 없이 어려운 예",
     "fasta": ">GFP_Aequorea\nMSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"},
]
LOCK = threading.RLock()
WAKE = threading.Condition(LOCK)
QUEUE = []           # 대기 중 작업 id (앞이 먼저)
RUNNING = {}         # id → Popen
RUNNER = os.path.join(ROOT, "runner.py")
RUNNER_MPNN = os.path.join(ROOT, "runner_mpnn.py")
MPNN_ENV = os.path.expanduser(os.environ.get("MPNN_ENV", "~/miniforge3/envs/proteinmpnn"))
MAX_AFF_LIGANDS = 20                                       # 친화도 일괄 비교 리간드 수
MAX_DESIGNS = 64                                           # ProteinMPNN 서열 수
MAX_PDB_RES = 3000
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
         "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def jdir(jid, *p):
    if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", jid or ""):
        raise FileNotFoundError(jid)
    return os.path.join(WS, "jobs", jid, *p)


# ── 엔진 ────────────────────────────────────────────────────────────────
def boltz_py():
    return os.path.join(BOLTZ_ENV, "bin", "python")


def engines():
    """실제로 돌릴 수 있는 엔진만 (python + 가중치가 있어야)"""
    out = []
    if os.path.exists(boltz_py()) and os.path.isdir(os.path.join(BOLTZ_CACHE, "mols")):
        if os.path.exists(os.path.join(BOLTZ_CACHE, "boltz2_conf.ckpt")):
            out.append({"id": "boltz2", "label": "Boltz-2", "ligand": True,
                        "affinity": os.path.exists(os.path.join(BOLTZ_CACHE, "boltz2_aff.ckpt"))})
        if os.path.exists(os.path.join(BOLTZ_CACHE, "boltz1_conf.ckpt")):
            out.append({"id": "boltz1", "label": "Boltz-1", "ligand": True, "affinity": False})
    return out


def mpnn_py():
    return os.path.join(MPNN_ENV, "bin", "python")


def mpnn_ok():
    """ligandmpnn 패키지(가중치 동봉)가 깔린 환경인지"""
    return os.path.exists(mpnn_py()) and bool(glob.glob(os.path.join(
        MPNN_ENV, "lib", "python3*", "site-packages", "ligandmpnn", "data", "model_params", "proteinmpnn_v_48_020.pt")))


def gpu_status():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        rows = [[x.strip() for x in l.split(",")] for l in r.stdout.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []
    return [{"index": a, "name": b, "used": int(c), "free": int(d), "total": int(e)}
            for a, b, c, d, e in rows if not GPUS or a in GPUS]


def pick_gpu():
    g = sorted(gpu_status(), key=lambda x: -x["free"])
    return g[0] if g else None


# ── 입력 검증 ───────────────────────────────────────────────────────────
def parse_fasta(text):
    """FASTA(여러 레코드 → 여러 사슬) 또는 맨 서열. 반환: (chains, errors)"""
    chains, errors, cur = [], [], None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            cur = {"name": line[1:].strip()[:60] or f"chain{len(chains) + 1}", "seq": ""}
            chains.append(cur)
            continue
        if cur is None:
            cur = {"name": "chain1", "seq": ""}
            chains.append(cur)
        cur["seq"] += re.sub(r"[\s\d*]", "", line).upper()
    for i, c in enumerate(chains):
        c["id"] = chr(65 + i) if i < 26 else None
        bad = sorted(set(c["seq"]) - AA)
        if not c["seq"]:
            errors.append(f"{c['name']}: 서열이 비어 있습니다")
        elif bad:
            errors.append(f"{c['name']}: 허용하지 않는 문자 {''.join(bad)} — 표준 아미노산 20종(ACDEFGHIKLMNPQRSTVWY)만 씁니다 "
                          "(X·B·Z 등 모호 문자는 지우거나 실제 잔기로 바꾸세요)")
        elif len(c["seq"]) < 4:
            errors.append(f"{c['name']}: 서열이 너무 짧습니다 (4잔기 이상)")
    if not chains:
        errors.append("서열을 넣어 주세요 (FASTA 또는 한 줄 서열)")
    return chains, errors


def lig_atoms(smiles):
    """SMILES 의 무거운 원자 수 대략 (토큰 수 추정용)"""
    return len(re.findall(r"Cl|Br|\[[^\]]+\]|[BCNOPSFI]|[cnops]", smiles))


def ccd_known(code):
    d = os.path.join(BOLTZ_CACHE, "mols")
    return os.path.exists(os.path.join(d, code + ".pkl")) if os.path.isdir(d) else True


def parse_ligands(text):
    """한 줄에 하나. 대문자·숫자 1~5자이고 CCD 사전에 있으면 CCD 코드(ATP, HEM, SO4…), 아니면 SMILES.
    CCO·CO 처럼 둘 다 되는 짧은 글자는 SMILES 로 본다 — CCD 로 쓰려면 'ccd:CCO', SMILES 강제는 'smiles:…'"""
    ligs, errors = [], []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        force = None
        m = re.match(r"(ccd|smiles)\s*:\s*(.+)", s, re.I)
        if m:
            force, s = m.group(1).lower(), m.group(2).strip()
        s, _, name = s.partition(" ") if " " in s else s.partition("\t")
        name = name.strip()[:40]  # 'SMILES 이름' — 친화도 비교표에 쓰는 이름(선택)
        code = s.upper()
        as_code = re.fullmatch(r"[A-Za-z0-9]{1,5}", s) and (force == "ccd" or (force is None and not re.fullmatch(r"[CNOPSF]+", s)))
        if as_code and (ccd_known(code) or force == "ccd"):
            if not ccd_known(code):
                errors.append(f"리간드 {code}: CCD 코드 사전에 없습니다")
            ligs.append({"ccd": code, "label": code, "name": name or code, "atoms": 40})
        elif re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$/\\.%:*]+", s):
            label = s if len(s) <= 24 else s[:22] + "…"
            ligs.append({"smiles": s, "label": label, "name": name or label, "atoms": max(1, lig_atoms(s))})
        else:
            errors.append(f"리간드 '{s[:30]}': SMILES 또는 CCD 코드(예: ATP, HEM)가 아닙니다")
    return ligs, errors


def validate(req):
    chains, errors = parse_fasta(req.get("fasta"))
    eng = {e["id"]: e for e in engines()}
    engine = req.get("engine") or "boltz2"
    ligs, lerr = parse_ligands(req.get("ligands")) if req.get("ligands") else ([], [])
    errors += lerr
    if engine not in eng:
        errors.append(f"엔진 {engine} 을(를) 쓸 수 없습니다 (가중치·환경 없음)")
    n_items = len(chains) + len(ligs)
    if n_items > MAX_CHAINS:
        errors.append(f"사슬+리간드가 {n_items}개 — 최대 {MAX_CHAINS}개")
    if n_items > 26:
        errors.append("사슬 이름(A–Z)이 모자랍니다")
    for i, l in enumerate(ligs):
        l["id"] = chr(65 + len(chains) + i) if len(chains) + i < 26 else None
    tokens = sum(len(c["seq"]) for c in chains) + sum(l["atoms"] for l in ligs)
    if tokens > MAX_RES:
        errors.append(f"총 크기 {tokens} 토큰(잔기 + 리간드 원자) — 이 서버 한도 {MAX_RES} (GPU 메모리). 도메인만 잘라 넣어 보세요")
    affinity = bool(req.get("affinity")) and engine == "boltz2"
    if affinity and len(ligs) != 1:
        errors.append("결합 친화도 예측은 리간드가 정확히 1개일 때만 됩니다")
    samples = max(1, min(MAX_SAMPLES, int(req.get("samples") or 1)))
    mode = req.get("mode") if req.get("mode") in MODES else "standard"
    return {"chains": chains, "ligands": ligs, "engine": engine, "mode": mode, "samples": samples,
            "affinity": affinity, "tokens": tokens, "seed": req.get("seed")}, errors


def build_yaml(spec):
    """Boltz 입력 YAML — 모든 단백질 msa: empty (외부 MSA 서버를 부르지 않는다)"""
    L = ["version: 1", "sequences:"]
    for c in spec["chains"]:
        L += ["  - protein:", f"      id: {c['id']}", f"      sequence: {c['seq']}", "      msa: empty"]
    for l in spec["ligands"]:
        L += ["  - ligand:", f"      id: {l['id']}"]
        L.append(f"      ccd: {l['ccd']}" if l.get("ccd") else f"      smiles: {json.dumps(l['smiles'])}")
    if spec["affinity"]:
        L += ["properties:", "  - affinity:", f"      binder: {spec['ligands'][0]['id']}"]
    return "\n".join(L) + "\n"


def validate_affinity(req):
    """친화도: 단백질(사슬 1개 이상) × 리간드 여러 개 — 리간드마다 따로 Boltz-2 예측(구조 + 친화도)"""
    chains, errors = parse_fasta(req.get("fasta"))
    ligs, lerr = parse_ligands(req.get("ligands"))
    errors += lerr
    eng = {e["id"]: e for e in engines()}
    if not (eng.get("boltz2") or {}).get("affinity"):
        errors.append("Boltz-2 친화도 가중치(boltz2_aff.ckpt)가 없습니다")
    if not ligs and not lerr:
        errors.append("리간드를 1개 이상 넣어 주세요 (한 줄에 하나)")
    if len(ligs) > MAX_AFF_LIGANDS:
        errors.append(f"리간드 {len(ligs)}개 — 한 번에 {MAX_AFF_LIGANDS}개까지")
    if len(chains) + 1 > MAX_CHAINS:
        errors.append(f"사슬이 너무 많습니다 (최대 {MAX_CHAINS - 1}개 + 리간드)")
    lid = chr(65 + min(len(chains), 25))
    for i, l in enumerate(ligs):
        l["id"], l["file"] = lid, f"lig_{i + 1:02d}"
    tokens = sum(len(c["seq"]) for c in chains) + max([l["atoms"] for l in ligs] or [0])
    if tokens > MAX_RES:
        errors.append(f"총 크기 {tokens} 토큰 — 이 서버 한도 {MAX_RES}. 결합 부위 도메인만 넣어 보세요")
    mode = req.get("mode") if req.get("mode") in MODES else "standard"
    return {"chains": chains, "ligands": ligs, "engine": "boltz2", "mode": mode, "samples": 1, "affinity": True,
            "tokens": tokens, "seed": req.get("seed")}, errors


# ── PDB (서열 설계 입력) ─────────────────────────────────────────────────
def pdb_info(text):
    """ATOM 의 CA 기준 사슬·잔기. ProteinMPNN 과 같은 순서(사슬 알파벳순, 사슬 안은 파일 순서)."""
    ch = {}
    for line in (text or "").splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            c = line[21].strip() or "A"
            ch.setdefault(c, []).append((line[22:27].strip(), THREE.get(line[17:20].strip(), "X")))
    return [{"id": c, "n": len(v), "seq": "".join(a for _, a in v), "res": [r for r, _ in v]} for c, v in sorted(ch.items())]


CIF2PDB = """import sys, gemmi
doc = gemmi.cif.read_string(sys.stdin.read())
st = gemmi.make_structure_from_block(doc.sole_block())
st.setup_entities()
st.remove_ligands_and_waters()
st.shorten_chain_names()
sys.stdout.write(st.make_pdb_string())
"""


def is_cif(text):
    return bool(re.search(r"^data_", text or "", re.M)) and "_atom_site." in text


def cif_to_pdb(text):
    """mmCIF → PDB 문자열 (Boltz 환경의 gemmi 로 — 웹 서버는 표준 라이브러리만). 리간드·물은 뺀다(설계 대상 아님)"""
    if not os.path.exists(boltz_py()):
        raise ValueError("mmCIF 변환에 쓸 gemmi(Boltz 환경)가 없습니다 — PDB 로 바꿔 올리세요")
    r = subprocess.run([boltz_py(), "-c", CIF2PDB], input=text, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or "ATOM" not in r.stdout:
        raise ValueError("mmCIF 를 읽지 못했습니다: " + (r.stderr.strip().splitlines() or ["원자 없음"])[-1][:200])
    return r.stdout


def parse_fixed(text, info):
    """'A1-10, A25, B' → ['A1', …, 'A10', 'A25', B 사슬 전부]. 없는 잔기는 오류"""
    res = {c["id"]: c["res"] for c in info}
    out, errors = [], []
    for tok in re.split(r"[,\s]+", (text or "").strip()):
        if not tok:
            continue
        m = re.fullmatch(r"([A-Za-z0-9])(?:(-?\d+[A-Z]?)(?:-(-?\d+))?)?", tok)
        if not m or m.group(1) not in res:
            errors.append(f"고정 잔기 '{tok}': 형식은 A12, A1-10, B(사슬 전체) — 사슬 {', '.join(res) or '없음'}")
            continue
        c, a, b = m.groups()
        if a is None:
            out += [c + r for r in res[c]]
        elif b is None:
            if a not in res[c]:
                errors.append(f"고정 잔기 {c}{a}: 구조에 없는 번호")
            else:
                out.append(c + a)
        else:
            lo, hi = int(re.match(r"-?\d+", a).group()), int(b)
            sel = [r for r in res[c] if lo <= int(re.match(r"-?\d+", r).group()) <= hi]
            if not sel:
                errors.append(f"고정 잔기 {tok}: 그 범위의 잔기가 없음")
            out += [c + r for r in sel]
    return list(dict.fromkeys(out)), errors


def validate_design(req):
    errors = []
    if not mpnn_ok():
        errors.append(f"ProteinMPNN 환경이 없습니다 ({MPNN_ENV})")
    src = None
    if req.get("from_job"):
        k = int(req.get("k") or 0)
        try:
            text = read(jdir(req["from_job"], "models", f"model_{k}.pdb"))
            src = {"job": req["from_job"], "k": k, "title": load_job(req["from_job"]).get("title")}
        except (OSError, FileNotFoundError):
            text, _ = "", errors.append("가져올 예측 구조가 없습니다")
    else:
        text = req.get("pdb") or ""
        if len(text) > 8 * 2 ** 20:
            errors.append("구조 파일이 너무 큽니다 (8MB 이하)")
        src = {"upload": (req.get("filename") or "업로드.pdb")[:80]}
        if is_cif(text):
            try:
                text = cif_to_pdb(text)
                src["converted"] = "mmCIF→PDB"
            except (ValueError, subprocess.SubprocessError) as e:
                errors.append(str(e))
                text = ""
    info = pdb_info(text)
    if not info and not errors:
        errors.append("구조 파일에서 단백질 CA 원자를 찾지 못했습니다 (PDB 또는 mmCIF)")
    n_res = sum(c["n"] for c in info)
    if n_res > MAX_PDB_RES:
        errors.append(f"잔기 {n_res}개 — 최대 {MAX_PDB_RES}")
    ids = [c["id"] for c in info]
    chains = [c for c in (req.get("chains") or ids) if c in ids]
    if info and not chains:
        errors.append("설계할 사슬을 하나 이상 고르세요")
    fixed, ferr = parse_fixed(req.get("fixed"), info)
    errors += ferr
    if info and chains and len(fixed) >= sum(c["n"] for c in info if c["id"] in chains):
        errors.append("설계할 사슬의 잔기가 모두 고정되었습니다")
    n = max(1, min(MAX_DESIGNS, int(req.get("n") or 8)))
    bs = min(n, 16)
    try:
        temp = float(req.get("temperature") or 0.1)
    except ValueError:
        temp = 0.1
    temp = max(0.01, min(1.5, temp))
    model = req.get("model") if req.get("model") in ("protein_mpnn", "soluble_mpnn") else "protein_mpnn"
    omit = "".join(sorted(set((req.get("omit") or "").upper()) & AA))
    return {"text": text, "info": info, "chains": chains, "fixed": fixed, "n": bs * math.ceil(n / bs), "batch_size": bs,
            "batches": math.ceil(n / bs), "temperature": temp, "model": model, "omit": omit, "src": src,
            "seed": req.get("seed"), "tokens": n_res}, errors


# ── 작업 저장 ───────────────────────────────────────────────────────────
def load_job(jid):
    return json.loads(read(jdir(jid, "job.json")))


def save_job(job):
    with LOCK:
        p = jdir(job["id"], "job.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
        os.replace(p + ".tmp", p)


def update_job(jid, **kw):
    with LOCK:
        job = load_job(jid)
        job.update(kw)
        save_job(job)
        return job


def list_jobs(limit=200):
    d = os.path.join(WS, "jobs")
    if not os.path.isdir(d):
        return []
    out = []
    for jid in sorted(os.listdir(d), reverse=True)[:limit]:
        try:
            j = load_job(jid)
        except (OSError, ValueError):
            continue
        s = j.get("summary") or {}
        out.append({k: j.get(k) for k in ("id", "kind", "title", "status", "created", "engine", "mode", "tokens", "error", "elapsed", "ref")}
                   | {"plddt": s.get("plddt"), "ptm": s.get("ptm"), "iptm": s.get("iptm"), "rmsd": s.get("rmsd"), "best": s.get("best"),
                      "n_chains": len(j.get("chains") or []), "n_ligands": len(j.get("ligands") or []), "position": position(jid)})
    return out


def position(jid):
    with LOCK:
        return QUEUE.index(jid) + 1 if jid in QUEUE else 0


def new_id():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def enqueue(job, params, files):
    os.makedirs(jdir(job["id"]), exist_ok=True)
    for name, text in files.items():
        os.makedirs(os.path.dirname(jdir(job["id"], name)), exist_ok=True)
        with open(jdir(job["id"], name), "w", encoding="utf-8") as f:
            f.write(text)
    with open(jdir(job["id"], "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)
    with LOCK:
        save_job(job)
        QUEUE.append(job["id"])
        WAKE.notify_all()
    return job


def base_job(jid, kind, title, **kw):
    return {"id": jid, "kind": kind, "title": title, "status": "queued", "created": datetime.datetime.now().isoformat(timespec="seconds"), **kw}


def seed_of(spec):
    return int(spec["seed"]) if str(spec.get("seed") or "").isdigit() else None


def submit(req):
    kind = req.get("kind") or "fold"
    if kind == "affinity":
        return submit_affinity(req)
    if kind == "design":
        return submit_design(req)
    spec, errors = validate(req)
    ref = req.get("ref")
    if ref:  # 설계 서열 검증 — 원구조(설계 작업의 input.pdb)와 RMSD 비교
        if not os.path.exists(jdir(ref.get("job"), "input.pdb")):
            errors.append("비교할 원구조가 없습니다")
        ref = {"job": ref["job"], "design": int(ref.get("design") or 0)}
    if errors:
        raise ValueError("\n".join(errors))
    m = MODES[spec["mode"]]
    names = [c["name"] for c in spec["chains"]] + [l["label"] for l in spec["ligands"]]
    title = (req.get("title") or "").strip()[:80] or " + ".join(names)[:80]
    job = base_job(new_id(), "fold", title, engine=spec["engine"], mode=spec["mode"], samples=spec["samples"],
                   affinity=spec["affinity"], tokens=spec["tokens"], chains=spec["chains"], ligands=spec["ligands"], msa="none",
                   **({"ref": ref} if ref else {}))
    params = {"kind": "fold", "engine": spec["engine"], "samples": spec["samples"], "recycling": m["recycling"], "steps": m["steps"],
              "cache": BOLTZ_CACHE, "ligands": spec["ligands"], "seed": seed_of(spec)}
    return enqueue(job, params, {"input.yaml": build_yaml(spec)})


def submit_affinity(req):
    spec, errors = validate_affinity(req)
    if errors:
        raise ValueError("\n".join(errors))
    m = MODES[spec["mode"]]
    names = " + ".join(c["name"] for c in spec["chains"])
    title = (req.get("title") or "").strip()[:80] or f"{names[:50]} × 리간드 {len(spec['ligands'])}개"
    job = base_job(new_id(), "affinity", title, engine="boltz2", mode=spec["mode"], samples=1, affinity=True,
                   tokens=spec["tokens"], chains=spec["chains"], ligands=spec["ligands"], msa="none")
    files = {f"inputs/{l['file']}.yaml": build_yaml({"chains": spec["chains"], "ligands": [l], "affinity": True}) for l in spec["ligands"]}
    params = {"kind": "affinity", "input": "inputs", "engine": "boltz2", "samples": 1, "recycling": m["recycling"], "steps": m["steps"],
              "cache": BOLTZ_CACHE, "ligands": spec["ligands"], "seed": seed_of(spec)}
    return enqueue(job, params, files)


def submit_design(req):
    spec, errors = validate_design(req)
    if errors:
        raise ValueError("\n".join(errors))
    src = spec["src"]
    title = (req.get("title") or "").strip()[:80] or "설계: " + (src.get("title") or src.get("upload") or "구조")[:70]
    job = base_job(new_id(), "design", title, engine="proteinmpnn", model=spec["model"], n=spec["n"], temperature=spec["temperature"],
                   design_chains=spec["chains"], fixed=spec["fixed"], omit=spec["omit"], source=src, tokens=spec["tokens"],
                   chains=[{"id": c["id"], "n": c["n"], "seq": c["seq"]} for c in spec["info"]], ligands=[])
    params = {"kind": "design", "model": spec["model"], "batch_size": spec["batch_size"], "batches": spec["batches"],
              "temperature": spec["temperature"], "chains": spec["chains"], "fixed": spec["fixed"], "omit": spec["omit"],
              "seed": seed_of(spec)}
    return enqueue(job, params, {"input.pdb": spec["text"]})


def cancel(jid):
    with LOCK:
        if jid in QUEUE:
            QUEUE.remove(jid)
            return update_job(jid, status="canceled", error="사용자가 취소함")
        p = RUNNING.get(jid)
    if p:
        update_job(jid, cancel=True)
        kill(p)
        return load_job(jid)
    return load_job(jid)


def kill(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass


def delete_job(jid):
    with LOCK:
        if jid in RUNNING:
            raise ValueError("실행 중인 작업은 먼저 취소하세요")
        if jid in QUEUE:
            QUEUE.remove(jid)
    shutil.rmtree(jdir(jid), ignore_errors=True)


# ── 실행 (워커 1개 = GPU 작업 1개) ───────────────────────────────────────
NOISE = re.compile(r"tensorboardX|Lightning v|Tensor Cores|set_float32_matmul|warnings\.warn|UserWarning|FutureWarning|"
                   r"^\s*$|GPU available|TPU available|HPU available|lightning\.ai|pytorch_lightning/")


def run_job(jid):
    job = update_job(jid, status="running", started=datetime.datetime.now().isoformat(timespec="seconds"))
    log = open(jdir(jid, "log.txt"), "a", encoding="utf-8", buffering=1)
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TQDM_MININTERVAL="2")
    g = pick_gpu()
    if g:
        env["CUDA_VISIBLE_DEVICES"] = g["index"]
        log.write(f"[app] GPU {g['index']} ({g['name']}) 여유 {g['free']} MiB 에서 실행\n")
        update_job(jid, gpu=g["index"], gpu_free_mb=g["free"])
    kind = job.get("kind") or "fold"
    if kind == "design":
        log.write(f"[app] ProteinMPNN {job['model']} · 서열 {job['n']}개 · T={job['temperature']} · 설계 사슬 {','.join(job['design_chains'])}"
                  f" · 고정 {len(job['fixed'])}잔기\n")
        cmd = [mpnn_py(), RUNNER_MPNN, jdir(jid)]
    else:
        what = f"리간드 {len(job['ligands'])}개 친화도" if kind == "affinity" else f"샘플 {job['samples']}"
        log.write(f"[app] 엔진 {job['engine']} · 모드 {MODES[job['mode']]['label']} · {what} · MSA 없음(단일 서열)\n")
        cmd = [boltz_py(), RUNNER, jdir(jid)]
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=jdir(jid), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    with LOCK:
        RUNNING[jid] = p
    timer = threading.Timer(JOB_TIMEOUT, lambda: (update_job(jid, timeout=True), kill(p)))
    timer.start()
    tail, last = [], ""
    try:
        buf = b""
        while True:
            ch = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not ch:
                break
            buf += ch
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").rstrip()
                if NOISE.search(line) or line == last:
                    continue
                if "Predicting" in line and "Predicting" in last and "100%" not in line:
                    continue  # 진행 막대는 처음과 끝만
                last = line
                tail = (tail + [line])[-60:]
                log.write(line + "\n")
        rc = p.wait()
    finally:
        timer.cancel()
        with LOCK:
            RUNNING.pop(jid, None)
    elapsed = round(time.time() - t0, 1)
    job = load_job(jid)
    if job.get("cancel"):
        log.write("[app] 취소됨\n")
        return update_job(jid, status="canceled", error="사용자가 취소함", elapsed=elapsed)
    if job.get("timeout"):
        log.write(f"[app] 시간 초과 ({JOB_TIMEOUT}s)\n")
        return update_job(jid, status="error", error=f"시간 초과 ({JOB_TIMEOUT // 60}분)", elapsed=elapsed)
    try:
        if rc != 0:
            raise RuntimeError(explain_failure(tail) or f"엔진이 오류로 끝났습니다 (코드 {rc}) — 로그 확인")
        res = {"design": postprocess_design, "affinity": postprocess_affinity}.get(kind, postprocess)(jid)
        log.write(f"[app] 완료 — {res['summary'].get('best') or ''} {elapsed}s\n")
        return update_job(jid, status="done", elapsed=elapsed, finished=datetime.datetime.now().isoformat(timespec="seconds"),
                          summary=res["summary"])
    except Exception as e:
        msg = str(e)
        if "out of memory" in "\n".join(tail).lower():  # Boltz 는 OOM 이면 건너뛰고 정상 종료한다
            msg = explain_failure(tail)
        log.write(f"[app] 실패: {msg}\n")
        return update_job(jid, status="error", error=msg, elapsed=elapsed)


def explain_failure(tail):
    t = "\n".join(tail)
    if "out of memory" in t.lower():
        return "GPU 메모리 부족 — 서열을 줄이거나 샘플 수를 줄이세요 (다른 작업이 GPU 를 쓰는 중일 수 있음)"
    m = re.search(r"(리간드 SMILES[^\n]*|돌릴 리간드가[^\n]*|SystemExit: [^\n]*)", t)
    if m:
        return m.group(1)
    errs = [l for l in tail if re.match(r"\w*(Error|Exception)\b", l)]
    return errs[-1][:300] if errs else ""


def worker():
    while True:
        with LOCK:
            while not QUEUE:
                WAKE.wait()
            jid = QUEUE.pop(0)
        try:
            run_job(jid)
        except Exception as e:  # 작업 하나가 워커를 죽이지 않게
            try:
                update_job(jid, status="error", error=f"{type(e).__name__}: {e}")
            except Exception:
                pass


def recover():
    """서버 재시작 시: 돌던 작업은 실패 처리, 대기 작업은 다시 줄 세운다"""
    d = os.path.join(WS, "jobs")
    if not os.path.isdir(d):
        return
    for jid in sorted(os.listdir(d)):
        try:
            j = load_job(jid)
        except (OSError, ValueError):
            continue
        if j.get("status") == "running":
            update_job(jid, status="error", error="서버 재시작으로 중단됨 — 다시 실행하세요")
        elif j.get("status") == "queued":
            QUEUE.append(jid)


# ── 결과 파싱 (stdlib 만: PDB, confidence JSON, npz) ──────────────────────
def read_npy(raw):
    """.npy (float32/float64, C 순서) → (shape, flat list)"""
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("npy 아님")
    major = raw[6]
    hl = int.from_bytes(raw[8:10] if major == 1 else raw[8:12], "little")
    off = (10 if major == 1 else 12)
    hdr = ast.literal_eval(raw[off:off + hl].decode("latin1"))
    if hdr.get("fortran_order"):
        raise ValueError("fortran 순서 미지원")
    code = {"<f4": "f", "<f8": "d"}[hdr["descr"]]
    a = array.array(code)
    a.frombytes(raw[off + hl:])
    if sys.byteorder != "little":
        a.byteswap()
    return tuple(hdr["shape"]), a


def read_npz(path, key=None):
    with zipfile.ZipFile(path) as z:
        name = key + ".npy" if key else z.namelist()[0]
        return read_npy(z.read(name))


def parse_pdb(text):
    """토큰(단백질=잔기, 리간드=원자) 목록과 pLDDT(B 인자). Boltz 의 토큰 순서와 같다(사슬 순서대로)."""
    toks, seen = [], set()
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        chain, resn, resi, b = line[21], line[17:20].strip(), int(line[22:26]), float(line[60:66])
        if rec == "ATOM":
            key = (chain, resi)
            if key in seen:
                continue
            seen.add(key)
            toks.append({"chain": chain, "resi": resi, "resn": resn, "plddt": b, "het": False})
        else:
            toks.append({"chain": chain, "resi": resi, "resn": resn, "atom": line[12:16].strip(), "plddt": b, "het": True})
    return toks


def segments(toks, lo=50.0, min_len=3):
    """사슬별 pLDDT < lo 가 min_len 잔기 이상 이어진 구간"""
    out, cur = [], None
    for t in toks + [None]:
        bad = t is not None and not t["het"] and t["plddt"] < lo and (cur is None or (t["chain"] == cur["chain"] and t["resi"] == cur["end"] + 1))
        if bad:
            if cur is None:
                cur = {"chain": t["chain"], "start": t["resi"], "end": t["resi"], "sum": 0.0}
            cur["end"], cur["sum"] = t["resi"], cur["sum"] + t["plddt"]
            continue
        if cur and cur["end"] - cur["start"] + 1 >= min_len:
            n = cur["end"] - cur["start"] + 1
            out.append({"chain": cur["chain"], "start": cur["start"], "end": cur["end"], "len": n, "mean": round(cur["sum"] / n, 1)})
        cur = None
        if t is not None and not t["het"] and t["plddt"] < lo:
            cur = {"chain": t["chain"], "start": t["resi"], "end": t["resi"], "sum": t["plddt"]}
    return out


def pool(mat, n, size):
    """n×n 행렬을 size×size 로 평균 풀링"""
    if n <= size:
        return [[round(mat[i * n + j], 1) for j in range(n)] for i in range(n)], 1
    f = -(-n // size)
    m = -(-n // f)
    out = []
    for bi in range(m):
        row = []
        for bj in range(m):
            s = c = 0
            for i in range(bi * f, min(n, (bi + 1) * f)):
                base = i * n
                for j in range(bj * f, min(n, (bj + 1) * f)):
                    s += mat[base + j]
                    c += 1
            row.append(round(s / c, 1))
        out.append(row)
    return out, f


def bands(vals):
    n = len(vals) or 1
    return {"vh": round(100 * sum(v >= 90 for v in vals) / n, 1), "h": round(100 * sum(70 <= v < 90 for v in vals) / n, 1),
            "l": round(100 * sum(50 <= v < 70 for v in vals) / n, 1), "vl": round(100 * sum(v < 50 for v in vals) / n, 1)}


def pred_dirs(jid):
    """Boltz 출력의 predictions/<이름>/ 폴더들 {이름: 경로}"""
    out = {}
    for dp, ds, _ in os.walk(jdir(jid, "out")):
        if os.path.basename(dp) == "predictions":
            out.update({d: os.path.join(dp, d) for d in ds})
    return out


def parse_model(jid, pdir, name, k, ids, prefix=""):
    """예측 하나(model_k) → 지표 dict. models/{prefix}model_k.{pdb,cif}, tokens_k.json, pae_k.json 을 남긴다."""
    pdb = os.path.join(pdir, f"{name}_model_{k}.pdb")
    if not os.path.exists(pdb):
        return None
    os.makedirs(jdir(jid, "models"), exist_ok=True)
    shutil.copy(pdb, jdir(jid, "models", f"{prefix}model_{k}.pdb"))
    if os.path.exists(pdb[:-4] + ".cif"):
        shutil.copy(pdb[:-4] + ".cif", jdir(jid, "models", f"{prefix}model_{k}.cif"))
    conf_p = os.path.join(pdir, f"confidence_{name}_model_{k}.json")
    conf = json.loads(read(conf_p)) if os.path.exists(conf_p) else {}
    toks = parse_pdb(read(pdb))
    prot = [t["plddt"] for t in toks if not t["het"]]
    chains = []
    for idx, cid in enumerate(ids):
        v = [t["plddt"] for t in toks if t["chain"] == cid]
        chains.append({"id": cid, "plddt": round(sum(v) / len(v), 1) if v else None, "n": len(v),
                       "ptm": round(conf["chains_ptm"][str(idx)], 3) if str(idx) in (conf.get("chains_ptm") or {}) else None})
    pair = conf.get("pair_chains_iptm") or {}
    pairs = [{"a": ids[int(a)], "b": ids[int(b)], "iptm": round(v, 3)}
             for a, row in pair.items() for b, v in row.items() if int(a) < int(b) and int(b) < len(ids)]
    has_lig = any(t["het"] for t in toks)
    s = {"k": k, "plddt": round(sum(prot) / len(prot), 1) if prot else None, "ptm": round(conf.get("ptm", 0), 3) or None,
         "iptm": round(conf.get("iptm", 0), 3) if len(ids) > 1 else None, "confidence": round(conf.get("confidence_score", 0), 3),
         "ligand_iptm": round(conf.get("ligand_iptm", 0), 3) if has_lig else None,
         "complex_pde": round(conf.get("complex_pde", 0), 2), "chains": chains, "pairs": pairs, "bands": bands(prot),
         "low_segments": segments(toks)}
    pae_p = os.path.join(pdir, f"pae_{name}_model_{k}.npz")
    if os.path.exists(pae_p):
        shape, mat = read_npz(pae_p, "pae")
        n = shape[0]
        s["pae_mean"] = round(sum(mat) / len(mat), 2)
        chain_of = [t["chain"] for t in toks] if len(toks) == n else None
        if chain_of and len(ids) > 1:  # 사슬 쌍 PAE 평균 (사슬 간 상대 위치 신뢰도)
            acc = {}
            for i in range(n):
                for j in range(n):
                    if chain_of[i] != chain_of[j]:
                        a = acc.setdefault(tuple(sorted((chain_of[i], chain_of[j]))), [0.0, 0])
                        a[0] += mat[i * n + j]
                        a[1] += 1
            for pr in pairs:
                a = acc.get(tuple(sorted((pr["a"], pr["b"]))))
                pr["pae"] = round(a[0] / a[1], 1) if a else None
        grid, f = pool(mat, n, PAE_MAX)
        bounds = [i for i in range(1, n) if chain_of and chain_of[i] != chain_of[i - 1]]
        with open(jdir(jid, "models", f"{prefix}pae_{k}.json"), "w") as fp:
            json.dump({"n": n, "factor": f, "grid": grid, "bounds": bounds, "max": 31.75}, fp)
    with open(jdir(jid, "models", f"{prefix}tokens_{k}.json"), "w") as fp:
        json.dump(toks, fp)
    return s


def runner_info(jid):
    return json.loads(read(jdir(jid, "runner.json"))) if os.path.exists(jdir(jid, "runner.json")) else {}


def save_result(jid, res):
    with open(jdir(jid, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False)
    return res


def affinity_values(path):
    """Boltz-2 affinity_*.json → 표시용. affinity_pred_value = log10(IC50), IC50 단위 μM (Boltz README).
    pIC50 = -log10(IC50[M]) = 6 - log10(IC50[μM])"""
    a = json.loads(read(path))
    v = a.get("affinity_pred_value")
    return {"log10_ic50_uM": round(v, 3), "ic50_uM": round(10 ** v, 4), "pic50": round(6 - v, 2),
            "p_binder": round(a.get("affinity_probability_binary", 0), 3),
            "models": [round(a[k], 3) for k in ("affinity_pred_value1", "affinity_pred_value2") if k in a]}


def postprocess(jid):
    job = load_job(jid)
    preds = pred_dirs(jid)
    if not preds:
        raise RuntimeError("예측 결과가 없습니다 — GPU 메모리 부족 등으로 건너뛰었을 수 있습니다 (로그 확인)")
    name, pdir = next(iter(preds.items()))
    ids = [c["id"] for c in job["chains"]] + [l["id"] for l in job["ligands"]]
    samples = [s for s in (parse_model(jid, pdir, name, k, ids) for k in range(job["samples"])) if s]
    if not samples:
        raise RuntimeError("예측 PDB 가 없습니다 (로그 확인)")
    ap = os.path.join(pdir, f"affinity_{name}.json")
    aff = affinity_values(ap) if os.path.exists(ap) else None
    runner = runner_info(jid)
    best = samples[0]  # Boltz 는 confidence_score 순으로 model_0 이 최상위
    summary = {"plddt": best["plddt"], "ptm": best["ptm"], "iptm": best["iptm"], "confidence": best["confidence"],
               "gpu_peak_mb": runner.get("gpu_peak_mb"), "gpu_name": runner.get("gpu_name"), "engine_seconds": runner.get("seconds"),
               "affinity": aff, "best": f"평균 pLDDT {best['plddt']}"}
    if job.get("ref"):
        summary.update(compare_ref(jid, job["ref"]["job"]))
        summary["best"] += f" · RMSD {summary.get('rmsd')}Å"
    return save_result(jid, {"summary": summary, "samples": samples})


def postprocess_affinity(jid):
    job = load_job(jid)
    preds = pred_dirs(jid)
    skipped = json.loads(read(jdir(jid, "skipped.json"))) if os.path.exists(jdir(jid, "skipped.json")) else {}
    ids = [c["id"] for c in job["chains"]] + [job["ligands"][0]["id"]]
    rows = []
    for i, l in enumerate(job["ligands"]):
        row = {"i": i, "file": l["file"], "name": l.get("name") or l["label"], "label": l["label"],
               "smiles": l.get("smiles"), "ccd": l.get("ccd")}
        pdir = preds.get(l["file"])
        s = parse_model(jid, pdir, l["file"], 0, ids, prefix=f"l{i:02d}_") if pdir else None
        ap = os.path.join(pdir, f"affinity_{l['file']}.json") if pdir else ""
        if s and os.path.exists(ap):
            row.update(affinity_values(ap))
            row.update({"plddt": s["plddt"], "iptm": s["iptm"], "ligand_iptm": s["ligand_iptm"], "confidence": s["confidence"],
                        "lig_plddt": next((c["plddt"] for c in s["chains"] if c["id"] == ids[-1]), None), "sample": s})
        else:
            row["error"] = skipped.get(l["file"]) or "예측 실패 (로그 확인)"
        rows.append(row)
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise RuntimeError("모든 리간드 예측이 실패했습니다: " + "; ".join(f"{r['name']}: {r['error']}" for r in rows))
    for key, rk in (("p_binder", "rank_p"), ("log10_ic50_uM", "rank_ic50")):
        for n, r in enumerate(sorted(ok, key=lambda r: -r[key] if key == "p_binder" else r[key])):
            r[rk] = n + 1
    top = min(ok, key=lambda r: r["rank_p"])
    runner = runner_info(jid)
    summary = {"n": len(rows), "n_ok": len(ok), "gpu_peak_mb": runner.get("gpu_peak_mb"), "gpu_name": runner.get("gpu_name"),
               "engine_seconds": runner.get("seconds"), "top": top["name"], "top_p": top["p_binder"],
               "best": f"1위 {top['name']} (결합 확률 {top['p_binder']}, IC50 {top['ic50_uM']:.3g} μM)"}
    return save_result(jid, {"summary": summary, "ligands": rows})


def parse_fa(text):
    recs, cur = [], None
    for line in text.splitlines():
        if line.startswith(">"):
            cur = {"hdr": line[1:], "seq": ""}
            recs.append(cur)
        elif cur is not None:
            cur["seq"] += line.strip()
    for r in recs:
        r["kv"] = dict(kv.split("=", 1) for kv in (x.strip() for x in r["hdr"].split(",")) if "=" in kv)
    return recs


def postprocess_design(jid):
    job = load_job(jid)
    fa = glob.glob(jdir(jid, "out", "seqs", "*.fa"))
    if not fa:
        raise RuntimeError("설계 결과(FASTA)가 없습니다 (로그 확인)")
    recs = parse_fa(read(fa[0]))
    native = recs[0]["seq"]
    designs = []
    for r in recs[1:]:
        conf = float(r["kv"].get("overall_confidence", "nan"))
        seq = r["seq"]
        muts = sum(1 for a, b in zip(native, seq) if a != b and a != ":")
        designs.append({"i": int(r["kv"].get("id", len(designs) + 1)), "seq": seq, "confidence": round(conf, 4),
                        "score": round(-math.log(conf), 4) if conf > 0 else None, "recovery": round(float(r["kv"].get("seq_rec", "nan")), 4),
                        "mutations": muts})
    sc = {x["id"]: x for x in json.loads(read(jdir(jid, "scores.json")))} if os.path.exists(jdir(jid, "scores.json")) else {}
    for d in designs:  # runner_mpnn 이 log_probs 로 다시 계산한 값 — score 는 헤더의 −ln(confidence) 와 같아야 한다
        x = sc.get(d["i"])
        if x:
            d["global_score"] = round(x["global_score"], 4)
            if d["score"] is not None and abs(x["score"] - d["score"]) > 2e-3:
                raise RuntimeError(f"설계 {d['i']}: score 불일치 (헤더 {d['score']} / 재계산 {x['score']:.4f}) — 순서 대응을 확인하세요")
            d["score"] = round(x["score"], 4)
        else:
            d["global_score"] = None
    designs.sort(key=lambda d: d["score"] if d["score"] is not None else 99)
    chain_ids = [c["id"] for c in job["chains"]]
    parts = native.split(":")
    if len(parts) != len(chain_ids) or any(len(p) != c["n"] for p, c in zip(parts, job["chains"])):
        raise RuntimeError(f"설계 출력의 사슬 구성이 입력과 다릅니다 ({[len(p) for p in parts]})")
    with open(jdir(jid, "designs.fa"), "w", encoding="utf-8") as f:
        f.write(f">native T={job['temperature']} model={job['model']}\n{native}\n")
        for d in designs:
            f.write(f">design_{d['i']} score={d['score']} global_score={d['global_score']} confidence={d['confidence']} "
                    f"seq_recovery={d['recovery']}\n{d['seq']}\n")
    runner = runner_info(jid)
    best = designs[0] if designs else {}
    summary = {"n": len(designs), "gpu_peak_mb": runner.get("gpu_peak_mb"), "gpu_name": runner.get("gpu_name"),
               "engine_seconds": runner.get("seconds"), "best_score": best.get("score"),
               "mean_recovery": round(sum(d["recovery"] for d in designs) / len(designs), 3) if designs else None,
               "best": f"설계 {len(designs)}개, 최저 score {best.get('score')}"}
    return save_result(jid, {"summary": summary, "native": native, "chain_ids": chain_ids, "designs": designs})


# ── 원구조 비교 (설계 서열 재예측 검증) — CA 중첩 RMSD, 표준 라이브러리만 ──────────
def ca_list(text):
    """사슬 알파벳순·파일 순서의 CA 좌표 (ProteinMPNN 서열 순서와 같다)"""
    ch = {}
    for line in text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            ch.setdefault(line[21].strip() or "A", []).append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return [xyz for c in sorted(ch) for xyz in ch[c]]


def jacobi_max(N):
    """4×4 대칭행렬의 최대 고유값·고유벡터 (Jacobi 회전)"""
    a = [row[:] for row in N]
    v = [[float(i == j) for j in range(4)] for i in range(4)]
    for _ in range(100):
        off = max((abs(a[i][j]), i, j) for i in range(4) for j in range(i + 1, 4))
        if off[0] < 1e-12:
            break
        _, p, q = off
        th = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(th), math.sin(th)
        for k in range(4):
            akp, akq = a[k][p], a[k][q]
            a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
        for k in range(4):
            apk, aqk = a[p][k], a[q][k]
            a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
        for k in range(4):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    i = max(range(4), key=lambda i: a[i][i])
    return a[i][i], [v[k][i] for k in range(4)]


def superpose(P, Q):
    """Q 를 P 에 겹치는 회전(Horn 사원수). 반환: rmsd, R(3×3), cP, cQ — x' = R(x - cQ) + cP"""
    n = len(P)
    cP = [sum(p[i] for p in P) / n for i in range(3)]
    cQ = [sum(q[i] for q in Q) / n for i in range(3)]
    X = [[p[i] - cP[i] for i in range(3)] for p in P]
    Y = [[q[i] - cQ[i] for i in range(3)] for q in Q]
    S = [[sum(y[i] * x[j] for x, y in zip(X, Y)) for j in range(3)] for i in range(3)]
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = S
    N = [[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz],
         [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]]
    lam, (q0, q1, q2, q3) = jacobi_max(N)
    g = sum(c * c for x in X for c in x) + sum(c * c for y in Y for c in y)
    rmsd = math.sqrt(max(0.0, (g - 2 * lam) / n))
    R = [[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
         [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
         [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]]
    return rmsd, R, cP, cQ


def compare_ref(jid, ref_jid):
    """예측 model_0 과 원구조(설계 입력) 를 CA 로 겹쳐 RMSD. 원구조를 예측 좌표계로 옮겨 models/ref_aligned.pdb 로 저장(뷰어 겹쳐 보기)."""
    ref_text = read(jdir(ref_jid, "input.pdb"))
    P, Q = ca_list(read(jdir(jid, "models", "model_0.pdb"))), ca_list(ref_text)
    if len(P) != len(Q) or len(P) < 3:
        return {"rmsd": None, "ref_note": f"잔기 수가 달라 비교 불가 (예측 {len(P)} / 원구조 {len(Q)})"}
    rmsd, R, cP, cQ = superpose(P, Q)
    out = []
    for line in ref_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x, y, z = (float(line[30:38]) - cQ[0], float(line[38:46]) - cQ[1], float(line[46:54]) - cQ[2])
            nx, ny, nz = (R[i][0] * x + R[i][1] * y + R[i][2] * z + cP[i] for i in range(3))
            line = f"{line[:30]}{nx:8.3f}{ny:8.3f}{nz:8.3f}{line[54:]}"
        if line.startswith(("ATOM", "HETATM", "TER", "END")):
            out.append(line)
    with open(jdir(jid, "models", "ref_aligned.pdb"), "w") as f:
        f.write("\n".join(out) + "\n")
    return {"rmsd": round(rmsd, 2), "ref_n": len(P)}


def validations(jid):
    """이 설계 작업에서 나온 검증(재예측) 작업들"""
    out = []
    for j in list_jobs(1000):
        if (j.get("ref") or {}).get("job") == jid:
            out.append({"id": j["id"], "design": j["ref"]["design"], "status": j["status"], "plddt": j.get("plddt"),
                        "ptm": j.get("ptm"), "rmsd": j.get("rmsd"), "error": j.get("error")})
    return out


def validate_designs(jid, req):
    """설계 서열(+원서열) → 구조 예측 작업들 (원구조와 RMSD 비교)"""
    job = load_job(jid)
    res = json.loads(read(jdir(jid, "result.json")))
    by_i = {d["i"]: d["seq"] for d in res["designs"]}
    picks = [int(i) for i in req.get("designs") or []][:20]
    if req.get("native"):
        picks = [0] + picks
    if not picks:
        raise ValueError("검증할 설계를 고르세요")
    jobs = []
    for i in picks:
        seq = res["native"] if i == 0 else by_i.get(i)
        if not seq:
            continue
        fasta = "\n".join(f">{'native' if i == 0 else 'design' + str(i)}_{c}\n{s}" for c, s in zip(res["chain_ids"], seq.split(":")))
        title = f"{job['title'][:50]} — {'원서열' if i == 0 else '설계 ' + str(i)} 재예측"
        jobs.append(submit({"fasta": fasta, "mode": req.get("mode") or "standard", "samples": 1, "title": title,
                            "ref": {"job": jid, "design": i}}))
    return jobs


def zip_job(jid):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("job.json", "result.json", "input.yaml", "input.pdb", "designs.fa", "skipped.json", "log.txt", "explain.md"):
            if os.path.exists(jdir(jid, f)):
                z.write(jdir(jid, f), f)
        md = jdir(jid, "models")
        if os.path.isdir(md):
            for f in sorted(os.listdir(md)):
                if f.endswith((".pdb", ".cif")) or "pae_" in f:
                    z.write(os.path.join(md, f), "models/" + f)
        for dp, _, fs in os.walk(jdir(jid, "out")):
            for f in fs:
                if (f.endswith(".npz") or f.startswith("affinity_")) and "predictions" in dp:
                    z.write(os.path.join(dp, f), "raw/" + f)
        for f in glob.glob(jdir(jid, "inputs", "*.yaml")):
            z.write(f, "inputs/" + os.path.basename(f))
    return b.getvalue()


# ── LLM 결과 해설 ────────────────────────────────────────────────────────
EXPLAIN = """너는 단백질 구조생물학자다. 구조 예측 도구(Boltz)의 신뢰도 지표를 연구자에게 한국어로 해설한다.
반드시 지킬 것:
- 아래 [지표]에 있는 숫자만 근거로 쓴다. 지표에 없는 사실(기능, 결합 부위, 실험 결과, 알려진 구조와의 비교)을 지어내지 않는다.
- 숫자를 인용할 때는 지표의 값을 그대로 쓴다(반올림 바꾸지 말 것).
- 해석 기준: pLDDT ≥90 매우 높음(골격·곁사슬까지 신뢰), 70–90 높음(골격 신뢰), 50–70 낮음(위상 정도만), <50 매우 낮음(무질서 영역이거나 예측 실패 — 해당 구간 모양을 믿지 말 것).
  pTM >0.5 이면 전체 접힘이 맞을 가능성이 높고, ipTM ≥0.8 은 사슬 간 배치 신뢰, 0.6–0.8 은 회색 지대, <0.6 은 복합체 배치가 틀렸을 가능성이 높다.
  사슬 간 PAE(Å)가 낮을수록(<5 좋음, >15 나쁨) 사슬의 상대 위치를 믿을 수 있다.
  친화도 값은 log10(IC50, μM) 예측치(낮을수록 강한 결합)와 결합체일 확률이며, 히트 선별용 상대 지표이지 실측이 아니다.
- 이 예측은 MSA(다중서열정렬) 없이 단일 서열로 돌렸다. 진화 정보가 없어 정확도가 떨어질 수 있으며, 특히 낮은 pLDDT 는 실제 무질서인지 정보 부족인지 구분할 수 없다는 점을 꼭 언급한다.
형식(마크다운, 400~700자):
**한 줄 요약** — 전반적 신뢰도
**신뢰도 해석** — 전체·사슬별 수치 해석 (2~4개 글머리표)
**주의할 구간** — 낮은 pLDDT 구간이 있으면 사슬·잔기 번호로, 없으면 '없음'
**활용 권고** — 이 결과로 할 수 있는 것과 하지 말아야 할 것, 정확도를 높이려면(예: MSA 사용, 실험 구조 확인) 1~3개 글머리표"""


EXPLAIN_AFF = """너는 약물 설계를 돕는 구조생물학자다. Boltz-2 결합 친화도 예측 결과(여러 리간드)를 연구자에게 한국어로 해설한다.
반드시 지킬 것:
- 아래 [지표]의 숫자만 근거로 쓴다. 리간드의 실제 활성, 알려진 약물 정보, 결합 부위 잔기 등 지표에 없는 사실을 지어내지 않는다. 숫자는 그대로 인용.
- 두 값의 뜻(Boltz 문서 기준):
  · 결합체 확률(affinity_probability_binary, 0–1): 리간드가 결합체(binder)일 확률. 결합체와 미끼(decoy)를 가르는 용도 — 히트 발굴 단계.
  · 친화도 값(affinity_pred_value): log10(IC50), IC50 단위 μM. 낮을수록 강한 결합(예: -1 → 0.1 μM, 1 → 10 μM). 결합체들 사이의 세기 차이·작은 구조 변화 비교 용도 — 히트→리드·리드 최적화 단계. pIC50 = 6 − 이 값.
  · 두 값은 다른 데이터로 학습되어 순위가 다를 수 있다. 확률이 낮은 리간드의 IC50 값은 의미가 약하다.
- 구조 신뢰도: 리간드 ipTM·리간드 pLDDT 가 낮으면(대략 0.5·50 미만) 결합 자세(pose)를 믿기 어렵고 친화도도 조심해서 본다.
- 단백질은 MSA 없이 단일 서열로 예측했다(폐쇄망). 단백질 구조가 부정확하면 친화도 예측도 흔들릴 수 있음을 언급한다.
- 예측치이며 실측이 아님. 실험(결합 측정)으로 확인해야 한다.
형식(마크다운, 400~800자):
**한 줄 요약**
**순위 해석** — 결합체 확률 순위와 IC50 순위를 각각, 서로 다르면 그 점 (글머리표 2~4개)
**신뢰도·주의** — 구조 신뢰도가 낮은 리간드, 실패·건너뛴 리간드
**다음 단계** — 1~3개 글머리표"""


def chat(system, user, model=None):
    model = model or MODEL
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        if LLM_API == "openai":
            hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})}
            req = urllib.request.Request(LLM_BASE + "/chat/completions",
                                         json.dumps({"model": model, "temperature": 0.2, "messages": msgs}).encode(), hdr)
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        body = {"model": model, "stream": False, "think": False, "options": {"temperature": 0.2}, "messages": msgs}
        for attempt in (0, 1):
            try:
                req = urllib.request.Request(LLM_BASE + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)["message"]["content"]
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                if attempt == 0 and "think" in msg:
                    body.pop("think")
                    continue
                raise RuntimeError(f"LLM HTTP {e.code}: {msg[:200]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM 서버 연결 실패 ({LLM_BASE}): {e.reason}")


def facts(job, res, k=0):
    s = next((x for x in res["samples"] if x["k"] == k), res["samples"][0])
    names = {c["id"]: f"{c['name']} ({len(c['seq'])}잔기)" for c in job["chains"]}
    names.update({l["id"]: f"리간드 {l['label']}" for l in job["ligands"]})
    f = {"엔진": f"{job['engine']} ({MODES[job['mode']]['label']} 모드, 샘플 {job['samples']}개 중 {k + 1}번)", "MSA": "없음(단일 서열)",
         "사슬": [f"{cid}: {names.get(cid, cid)}" for cid in names], "평균 pLDDT(단백질)": s["plddt"], "pTM": s["ptm"],
         "pLDDT 분포(%)": {"≥90": s["bands"]["vh"], "70-90": s["bands"]["h"], "50-70": s["bands"]["l"], "<50": s["bands"]["vl"]},
         "사슬별": [{"사슬": c["id"], "평균 pLDDT": c["plddt"], "pTM": c["ptm"]} for c in s["chains"]],
         "pLDDT<50 연속 구간": [f"사슬 {x['chain']} {x['start']}-{x['end']} ({x['len']}잔기, 평균 {x['mean']})" for x in s["low_segments"]] or "없음"}
    if s.get("iptm") is not None:
        f["ipTM"] = s["iptm"]
        f["사슬 쌍"] = [{"쌍": f"{p['a']}-{p['b']}", "ipTM": p["iptm"], "평균 PAE(Å)": p.get("pae")} for p in s["pairs"]]
    if s.get("ligand_iptm") is not None:
        f["리간드 ipTM"] = s["ligand_iptm"]
    if res["summary"].get("affinity") and k == 0:
        a = res["summary"]["affinity"]
        f["결합 친화도"] = {"log10(IC50 μM)": a["log10_ic50_uM"], "IC50(μM)": a["ic50_uM"], "결합체 확률": a["p_binder"]}
    if s.get("pae_mean") is not None:
        f["전체 평균 PAE(Å)"] = s["pae_mean"]
    return f


def facts_aff(job, res):
    f = {"엔진": f"Boltz-2 친화도 ({MODES[job['mode']]['label']} 모드)", "MSA": "없음(단일 서열)",
         "단백질": [f"{c['id']}: {c['name']} ({len(c['seq'])}잔기)" for c in job["chains"]], "리간드": []}
    for r in res["ligands"]:
        if "error" in r:
            f["리간드"].append({"이름": r["name"], "결과": "실패: " + r["error"]})
            continue
        f["리간드"].append({"이름": r["name"], "결합체 확률": r["p_binder"], "확률 순위": r["rank_p"],
                          "log10(IC50 μM)": r["log10_ic50_uM"], "IC50(μM)": r["ic50_uM"], "pIC50": r["pic50"], "IC50 순위": r["rank_ic50"],
                          "리간드 ipTM": r["ligand_iptm"], "리간드 pLDDT": r["lig_plddt"], "단백질 평균 pLDDT": r["plddt"]})
    return f


def explain(jid, k=0, model=None):
    job = load_job(jid)
    res = json.loads(read(jdir(jid, "result.json")))
    if job.get("kind") == "affinity":
        out = chat(EXPLAIN_AFF, "[지표]\n" + json.dumps(facts_aff(job, res), ensure_ascii=False, indent=1), model)
    else:
        out = chat(EXPLAIN, "[지표]\n" + json.dumps(facts(job, res, k), ensure_ascii=False, indent=1), model)
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    with open(jdir(jid, "explain.md"), "w", encoding="utf-8") as f:
        f.write(out)
    return out


def models():
    try:
        if LLM_API == "openai":
            req = urllib.request.Request(LLM_BASE + "/models", headers={"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})
            with urllib.request.urlopen(req, timeout=3) as r:
                return [m["id"] for m in json.load(r)["data"]]
        with urllib.request.urlopen(LLM_BASE + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except Exception:
        return []


def meta():
    return {"engines": engines(), "mpnn": mpnn_ok(), "max_aff_ligands": MAX_AFF_LIGANDS, "max_designs": MAX_DESIGNS, "modes": [{"id": k, **v} for k, v in MODES.items()], "max_res": MAX_RES,
            "max_chains": MAX_CHAINS, "max_samples": MAX_SAMPLES, "timeout": JOB_TIMEOUT, "examples": EXAMPLES,
            "gpus": gpu_status(), "model": MODEL, "llm": LLM_BASE, "queue": len(QUEUE), "running": list(RUNNING)}


# ── HTTP ────────────────────────────────────────────────────────────────
HTML = read(os.path.join(ROOT, "ui.html")) if os.path.exists(os.path.join(ROOT, "ui.html")) else "ui.html 없음"
STATIC = {".js": "application/javascript", ".txt": "text/plain; charset=utf-8", ".css": "text/css"}
FILES = {".pdb": "chemical/x-pdb", ".cif": "chemical/x-mmcif", ".json": "application/json"}

# ── 저작권 표기 (LICENSE·NOTICE 참고) ─────────────────────────────────────
_SIG = __import__("base64").b64decode("wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==").decode()
_SIG_A = __import__("base64").b64decode("Z2dnZzg2NTcgPGRvbmdqdWtpbS5kZXZAZ21haWwuY29tPg==").decode()


def signed(html):
    """화면에 저작권 표기를 붙인다. ui.html 에서 지워져도 서버가 내보낼 때 다시 붙는다."""
    name, mail = _SIG.split(" · ")
    if 'name="author"' not in html:
        meta_ = f'<meta name="author" content="{name[7:]} <{mail}>">'
        html = html.replace("<head>", "<head>" + meta_, 1) if "<head>" in html else meta_ + html
    if "data-sig" not in html:
        tag = (f'<!-- {_SIG} --><div data-sig title="{mail}" style="text-align:center;font-size:11px;color:#9aa0a6;'
               f'opacity:.55;margin:28px 0 8px">{name}</div>')
        html = html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag
    return html


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if a and isinstance(a[0], str) and a[0].startswith("POST") and "/api/validate" not in a[0]:
            super().log_message(fmt, *a)

    def _send(self, body, ctype="application/json", code=200, filename=None):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("X-Author", _SIG_A)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + urllib.parse.quote(filename))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path, q = u.path, urllib.parse.parse_qs(u.query)
        try:
            if path == "/api/health":
                return self._send({"ok": True})
            if path == "/api/meta":
                return self._send(meta())
            if path == "/api/models":
                return self._send(models())
            if path == "/api/jobs":
                return self._send(list_jobs())
            m = re.fullmatch(r"/static/([\w.\-]+)", path)
            if m and os.path.splitext(m.group(1))[1] in STATIC:
                with open(os.path.join(ROOT, "static", m.group(1)), "rb") as f:
                    return self._send(f.read(), STATIC[os.path.splitext(m.group(1))[1]])
            m = re.fullmatch(r"/api/job/([\w\-]+)(/[\w\-]+)?(/[\w.\-]+)?", path)
            if m:
                jid, sub, arg = m.group(1), (m.group(2) or "")[1:], (m.group(3) or "")[1:]
                if not sub:
                    return self._send(load_job(jid) | {"position": position(jid)})
                if sub == "log":
                    lines = read(jdir(jid, "log.txt")).splitlines() if os.path.exists(jdir(jid, "log.txt")) else []
                    since = int((q.get("since") or ["0"])[0])
                    return self._send({"lines": lines[since:], "next": len(lines)})
                if sub == "result":
                    out = json.loads(read(jdir(jid, "result.json")))
                    if os.path.exists(jdir(jid, "explain.md")):
                        out["explain"] = read(jdir(jid, "explain.md"))
                    return self._send(out)
                if sub == "validations":
                    return self._send(validations(jid))
                if sub == "fasta":
                    with open(jdir(jid, "designs.fa"), "rb") as f:
                        return self._send(f.read(), "text/plain; charset=utf-8", filename=f"designs_{jid}.fa")
                if sub == "input":
                    with open(jdir(jid, "input.pdb"), "rb") as f:
                        return self._send(f.read(), FILES[".pdb"])
                if sub == "file" and re.fullmatch(r"(l\d{2}_)?(model|pae|tokens)_\d\.(pdb|cif|json)|ref_aligned\.pdb", arg):
                    with open(jdir(jid, "models", arg), "rb") as f:
                        body = f.read()
                    ext = os.path.splitext(arg)[1]
                    dl = (q.get("dl") or [""])[0] == "1"
                    return self._send(body, FILES[ext], filename=f"{jid}_{arg}" if dl else None)
                if sub == "zip":
                    return self._send(zip_job(jid), "application/zip", filename=f"protein_{jid}.zip")
            if path in ("/", "/index.html"):
                return self._send(signed(HTML).encode(), "text/html; charset=utf-8")
            self._send({"error": "없는 경로"}, code=404)
        except FileNotFoundError:
            self._send({"error": "없음"}, code=404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/api/validate":
                spec, errors = validate(req)
                return self._send({"errors": errors, "tokens": spec["tokens"],
                                   "chains": [{"id": c["id"], "name": c["name"], "len": len(c["seq"])} for c in spec["chains"]],
                                   "ligands": spec["ligands"]})
            if self.path == "/api/pdbinfo":
                text = req.get("pdb") or ""
                try:
                    text = cif_to_pdb(text) if is_cif(text) else text
                except (ValueError, subprocess.SubprocessError) as e:
                    return self._send({"error": str(e)}, code=400)
                info = pdb_info(text)
                return self._send({"chains": [{"id": c["id"], "n": c["n"], "first": c["res"][0], "last": c["res"][-1]} for c in info]})
            if self.path == "/api/validate_affinity":
                spec, errors = validate_affinity(req)
                return self._send({"errors": errors, "tokens": spec["tokens"], "ligands": spec["ligands"],
                                   "chains": [{"id": c["id"], "name": c["name"], "len": len(c["seq"])} for c in spec["chains"]]})
            if self.path == "/api/submit":
                try:
                    return self._send(submit(req))
                except ValueError as e:
                    return self._send({"error": str(e)}, code=400)
            m = re.fullmatch(r"/api/job/([\w\-]+)/(cancel|delete|explain|validate)", self.path)
            if m:
                jid, act = m.groups()
                if act == "validate":
                    try:
                        return self._send(validate_designs(jid, req))
                    except ValueError as e:
                        return self._send({"error": str(e)}, code=400)
                if act == "cancel":
                    return self._send(cancel(jid))
                if act == "delete":
                    try:
                        delete_job(jid)
                    except ValueError as e:
                        return self._send({"error": str(e)}, code=400)
                    return self._send({"ok": True})
                return self._send({"text": explain(jid, int(req.get("k") or 0), req.get("model"))})
            self._send({"error": "없는 경로"}, code=404)
        except FileNotFoundError:
            self._send({"error": "없음"}, code=404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, code=500)


def serve():
    os.makedirs(os.path.join(WS, "jobs"), exist_ok=True)
    recover()
    threading.Thread(target=worker, daemon=True).start()
    eng = ",".join(e["id"] for e in engines()) or "없음"
    print(f"protein local → http://localhost:{PORT}  (engines={eng}, gpus={','.join(GPUS) or 'all'}, max_res={MAX_RES}, "
          f"llm={LLM_API} {LLM_BASE} {MODEL}, workspace={WS})  {_SIG}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    serve()
