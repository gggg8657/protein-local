#!/usr/bin/env python3
"""protein-local 자가검증 — GPU·모델 없이 경로만 확인한다. 임시 WORKSPACE 에서 돌고 지운다.
  서열·리간드 검증 → YAML(msa: empty) → 큐(동시 1개) → 가짜 엔진 출력 → PDB/PAE 파싱·pLDDT 추출·저장 → HTTP·다운로드·취소·재시작 복구
"""
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile

TMP = tempfile.mkdtemp(prefix="protein-selftest-")
os.environ["WORKSPACE"] = os.path.join(TMP, "ws")
os.environ["MAX_RES"] = "300"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402

FAKE = r"""
import json, os, struct, sys, time, zipfile
job = sys.argv[1]
p = json.load(open(os.path.join(job, "params.json")))
kind = p.get("kind", "fold")
bad = [l for l in (p.get("ligands") or []) if (l.get("smiles") or "") == "C1CC"]
if bad and kind != "affinity":
    raise SystemExit("리간드 SMILES 를 읽을 수 없습니다: C1CC")
skipped = {}
for l in bad:                                   # 친화도 일괄: 문제 리간드만 빼고 나머지는 돈다
    skipped[l["file"]] = "SMILES 를 읽을 수 없음 (CCD 코드라면 사전에 없는 코드)"
    os.remove(os.path.join(job, "inputs", l["file"] + ".yaml"))
if skipped:
    json.dump(skipped, open(os.path.join(job, "skipped.json"), "w"), ensure_ascii=False)
open(os.path.join(job, "..", "..", "concurrency.log"), "a").write("start %f\n" % time.time())
print("[runner] fake boltz", flush=True)
for i in range(3):
    print("Predicting DataLoader 0: %d%%\r" % (i * 50), end="", flush=True)
    time.sleep(0.15)
print("Predicting DataLoader 0: 100%", flush=True)


def seqs_of(path):
    seqs, ligs = [], []
    for line in open(path):
        if "sequence:" in line: seqs.append(line.split(":")[1].strip())
        if "ccd:" in line or "smiles:" in line: ligs.append(1)
    return seqs, ligs


def write_pred(name, yaml_path, samples, affinity):
    seqs, ligs = seqs_of(yaml_path)
    d = os.path.join(job, "out", "boltz_results_x", "predictions", name)
    os.makedirs(d, exist_ok=True)
    n = sum(len(s) for s in seqs) + 3 * len(ligs)
    for k in range(samples):
        L, no = [], 1
        for ci, s in enumerate(seqs):
            ch = chr(65 + ci)
            for r in range(len(s)):
                b = 30.0 if (ci == 0 and r < 5) else 92.0 - k   # 사슬 A 앞 5잔기는 무질서처럼
                for an in ("N", "CA", "C"):
                    L.append("ATOM  %5d  %-3s ALA %s%4d    %8.3f%8.3f%8.3f  1.00%6.2f           %s"
                             % (no, an, ch, r + 1, r * 3.8, 0, 0, b, an[0])); no += 1
        for li in range(len(ligs)):
            ch = chr(65 + len(seqs) + li)
            for an in ("C1", "C2", "O1"):
                L.append("HETATM%5d  %-3s LIG %s   1    %8.3f%8.3f%8.3f  1.00%6.2f           %s"
                         % (no, an, ch, 0, 5, 0, 80.0, an[0])); no += 1
        open(os.path.join(d, "%s_model_%d.pdb" % (name, k)), "w").write("\n".join(L) + "\nEND\n")
        open(os.path.join(d, "%s_model_%d.cif" % (name, k)), "w").write("data_fake\n")
        nc = len(seqs) + len(ligs)
        conf = {"confidence_score": 0.8 - k / 10, "ptm": 0.71, "iptm": 0.55 if nc > 1 else 0.0, "ligand_iptm": 0.4,
                "complex_pde": 1.2, "chains_ptm": {str(i): 0.7 for i in range(nc)},
                "pair_chains_iptm": {str(i): {str(j): 0.55 for j in range(nc)} for i in range(nc)}}
        json.dump(conf, open(os.path.join(d, "confidence_%s_model_%d.json" % (name, k)), "w"))
        hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (n, n)
        hdr += " " * (118 - len(hdr)) + "\n"
        body = b"".join(struct.pack("<f", 2.0 if (i < len(seqs[0])) == (j < len(seqs[0])) else 20.0)
                        for i in range(n) for j in range(n))
        with zipfile.ZipFile(os.path.join(d, "pae_%s_model_%d.npz" % (name, k)), "w") as z:
            z.writestr("pae.npy", b"\x93NUMPY\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode() + body)
    if affinity:   # 리간드 이름 끝자리로 값이 달라지게 (순위 확인용)
        t = int(name[-1]) if name[-1].isdigit() else 1
        json.dump({"affinity_pred_value": 0.5 * t - 1, "affinity_probability_binary": 0.9 - 0.2 * t,
                   "affinity_pred_value1": 0.1, "affinity_pred_value2": 0.2},
                  open(os.path.join(d, "affinity_%s.json" % name), "w"))


if kind == "affinity":
    for f in sorted(os.listdir(os.path.join(job, "inputs"))):
        write_pred(f[:-5], os.path.join(job, "inputs", f), 1, True)
else:
    write_pred("input", os.path.join(job, "input.yaml"), p["samples"], False)
json.dump({"seconds": 0.5, "gpu_name": "fake", "gpu_peak_mb": 123}, open(os.path.join(job, "runner.json"), "w"))
open(os.path.join(job, "..", "..", "concurrency.log"), "a").write("end %f\n" % time.time())
"""

FAKE_MPNN = r"""
import json, os, random, sys, time
job = sys.argv[1]
p = json.load(open(os.path.join(job, "params.json")))
print("[runner] fake proteinmpnn", flush=True)
ch = {}
for line in open(os.path.join(job, "input.pdb")):
    if line.startswith("ATOM") and line[12:16].strip() == "CA":
        ch.setdefault(line[21], []).append("ALA")
native = ":".join("A" * len(v) for k, v in sorted(ch.items()))
n = p["batch_size"] * p["batches"]
random.seed(1)
out = os.path.join(job, "out", "seqs")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "input.fa"), "w") as f:
    f.write(">input, T=%s\n%s\n" % (p["temperature"], native))
    for i in range(1, n + 1):
        seq = "".join(c if c == ":" or random.random() < 0.5 else "G" for c in native)
        f.write(">input, id=%d, T=%s, seed=1, overall_confidence=%.4f, ligand_confidence=0.3, seq_rec=%.4f\n%s\n"
                % (i, p["temperature"], 0.30 + i / 100, 0.4 + i / 100, seq))
import math
json.dump([{"id": i, "score": -math.log(0.30 + i / 100), "global_score": -math.log(0.30 + i / 100) - 0.05}
           for i in range(1, n + 1)], open(os.path.join(job, "scores.json"), "w"))
json.dump({"seconds": 0.3, "gpu_name": "fake", "gpu_peak_mb": 45}, open(os.path.join(job, "runner.json"), "w"))
"""

N = 0


def ok(cond, msg):
    global N
    if not cond:
        print("FAIL:", msg)
        shutil.rmtree(TMP, ignore_errors=True)
        sys.exit(1)
    N += 1


def wait(jid, t=30):
    for _ in range(t * 10):
        if app.load_job(jid)["status"] not in ("queued", "running"):
            return app.load_job(jid)
        time.sleep(0.1)
    ok(False, f"시간 초과 {jid}")


def main():
    fake = os.path.join(TMP, "fake_runner.py")
    open(fake, "w").write(FAKE)
    fake_mpnn = os.path.join(TMP, "fake_mpnn.py")
    open(fake_mpnn, "w").write(FAKE_MPNN)
    app.RUNNER, app.RUNNER_MPNN = fake, fake_mpnn
    app.boltz_py = app.mpnn_py = lambda: sys.executable
    app.mpnn_ok = lambda: True
    app.engines = lambda: [{"id": "boltz2", "label": "Boltz-2", "ligand": True, "affinity": True}]
    app.pick_gpu = lambda need=0: {"index": "9", "name": "fake", "free": 1000}

    # ── 서열 검증 ──
    ch, err = app.parse_fasta(">a\nMQIF VKTL\n12 TGK*\n>b desc\nacdefg")
    ok(not err and [c["seq"] for c in ch] == ["MQIFVKTLTGK", "ACDEFG"] and [c["id"] for c in ch] == ["A", "B"], f"FASTA 파싱 {ch} {err}")
    ok(app.parse_fasta("MKVLAAGIX")[1] and "X" in app.parse_fasta("MKVLAAGIX")[1][0], "모호 문자 X 거부")
    ok(app.parse_fasta("MQ1")[1], "너무 짧은 서열 거부")
    ok(app.parse_fasta("  ")[1], "빈 입력 거부")
    ok(app.parse_fasta("MKTAYIAK")[0][0]["name"] == "chain1", "헤더 없는 한 줄 서열")
    ligs, lerr = app.parse_ligands("atp\nCCO\nCC(=O)Oc1ccccc1C(=O)O\n한글\nccd:CCO")
    ok([("ccd" in l) for l in ligs] == [True, False, False, True] and ligs[0]["ccd"] == "ATP" and len(lerr) == 1, f"리간드 판별 {ligs} {lerr}")
    spec, e = app.validate({"fasta": "M" * 301})
    ok(e and "한도" in e[0], "길이 한도(MAX_RES)")
    spec, e = app.validate({"fasta": "MKTAYIAKQR", "ligands": "ATP\nHEM", "affinity": True})
    ok(any("친화도" in x for x in e), "친화도는 리간드 1개만")
    spec, e = app.validate({"fasta": "MKTAYIAKQR", "engine": "esmfold"})
    ok(any("esmfold" in x for x in e), "없는 엔진 거부")
    spec, e = app.validate({"fasta": ">x\nMKTAYIAKQR\n>y\nGSHMLE", "ligands": "ATP", "affinity": True, "samples": 9})
    ok(not e and spec["samples"] == app.MAX_SAMPLES and spec["ligands"][0]["id"] == "C", f"정상 입력 {e}")
    y = app.build_yaml(spec)
    ok(y.count("msa: empty") == 2 and "binder: C" in y and "ccd: ATP" in y and "msa_server" not in y, "YAML (MSA 없음)")

    # ── npy 리더 ──
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }"
    hdr += " " * (118 - len(hdr)) + "\n"
    shape, a = app.read_npy(b"\x93NUMPY\x01\x00" + len(hdr).to_bytes(2, "little") + hdr.encode() + struct.pack("<4f", 1, 2, 3, 4.5))
    ok(shape == (2, 2) and list(a) == [1, 2, 3, 4.5], "npy 파싱")
    grid, f = app.pool(array_of(10), 10, 4)
    ok(f == 3 and len(grid) == 4, f"PAE 풀링 {f} {len(grid)}")

    # ── 큐: 동시에 1개, 순서대로 ──
    threading.Thread(target=app.worker, daemon=True).start()
    j1 = app.submit({"fasta": ">A_chain\nMKTAYIAKQRQISFVKSHFSRQ\n>B_chain\nGSHMLEDPV", "ligands": "ATP", "samples": 2, "mode": "fast", "affinity": False})
    j2 = app.submit({"fasta": "MQIFVKTLTGKTITLEVEPS", "title": "두 번째"})
    j3 = app.submit({"fasta": "MQIFVKTLTG"})
    ok(app.position(j2["id"]) >= 1, "대기 순번")
    app.cancel(j3["id"])
    ok(app.load_job(j3["id"])["status"] == "canceled", "대기 작업 취소")
    peak = 0
    for _ in range(300):
        peak = max(peak, len(app.RUNNING))
        if app.load_job(j2["id"])["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    r1, r2 = wait(j1["id"]), wait(j2["id"])
    ok(r1["status"] == "done" and r2["status"] == "done", f"작업 완료 {r1.get('error')} {r2.get('error')}")
    ok(peak <= 1, "GPU 작업 동시 1개")
    ev = open(os.path.join(app.WS, "concurrency.log")).read().split()
    ok(ev[::2] == ["start", "end", "start", "end"] and float(ev[3]) <= float(ev[5]), f"순차 실행 {ev}")

    # ── 결과 파싱 ──
    res = json.load(open(app.jdir(j1["id"], "result.json")))
    s0 = res["samples"][0]
    exp = round((5 * 30 + 26 * 92) / 31, 1)
    ok(s0["plddt"] == exp, f"평균 pLDDT {s0['plddt']} != {exp}")
    ok([c["id"] for c in s0["chains"]] == ["A", "B", "C"] and s0["chains"][1]["plddt"] == 92.0 and s0["chains"][2]["n"] == 3, "사슬별 pLDDT·리간드 토큰")
    ok(s0["low_segments"] == [{"chain": "A", "start": 1, "end": 5, "len": 5, "mean": 30.0}], f"저신뢰 구간 {s0['low_segments']}")
    ok(s0["iptm"] == 0.55 and s0["ligand_iptm"] == 0.4 and len(s0["pairs"]) == 3, "ipTM·사슬 쌍")
    ab = next(p for p in s0["pairs"] if (p["a"], p["b"]) == ("A", "B"))
    ok(ab["pae"] == 20.0, f"사슬 간 PAE {ab}")
    ok(len(res["samples"]) == 2 and res["samples"][1]["plddt"] < s0["plddt"], "샘플 2개")
    ok(res["summary"]["gpu_peak_mb"] == 123, "GPU 메모리 기록")
    pae = json.load(open(app.jdir(j1["id"], "models", "pae_0.json")))
    ok(pae["n"] == 34 and pae["bounds"] == [22, 31], f"PAE 사슬 경계 {pae['bounds']}")
    ok(os.path.exists(app.jdir(j1["id"], "models", "model_1.cif")), "mmCIF 복사")
    log = open(app.jdir(j1["id"], "log.txt")).read()
    ok("fake boltz" in log and "100%" in log and "완료" in log, "진행 로그")

    # ── 실패 경로: 엔진 오류 메시지 ──
    j4 = app.submit({"fasta": "MKTAYIAKQR", "ligands": "C1CC"})
    r4 = wait(j4["id"])
    ok(r4["status"] == "error" and "SMILES" in r4["error"], f"엔진 오류 전달 {r4.get('error')}")

    # ── 해설 프롬프트: 지표 숫자만 ──
    seen = {}
    app.chat = lambda sys_, user, model=None: seen.setdefault("u", user) and "**한 줄 요약** 테스트"
    txt = app.explain(j1["id"])
    ok("테스트" in txt and str(exp) in seen["u"] and "MSA" in seen["u"] and "1-5" in seen["u"], "해설 입력에 지표 포함")

    # ── HTTP ──
    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    get = lambda p: urllib.request.urlopen(base + p, timeout=10)
    r = get("/")
    html = r.read().decode()
    ok("3Dmol-min.js" in html and "data-sig" in html and r.headers["X-Author"], "UI·저작자 표기")
    ok(len(get("/static/3Dmol-min.js").read()) > 100000, "3Dmol.js 동봉")
    ok(get(f"/api/job/{j1['id']}/file/model_0.pdb").read().startswith(b"ATOM"), "PDB 내려받기")
    ok("attachment" in get(f"/api/job/{j1['id']}/file/model_0.cif?dl=1").headers["Content-Disposition"], "mmCIF 첨부")
    z = zipfile.ZipFile(__import__("io").BytesIO(get(f"/api/job/{j1['id']}/zip").read()))
    ok("models/model_0.pdb" in z.namelist() and "result.json" in z.namelist(), "ZIP")
    ok(json.load(get(f"/api/job/{j1['id']}/result"))["explain"].startswith("**"), "결과 + 해설 API")
    ok(len(json.load(get("/api/jobs"))) == 4, "작업 목록")
    try:
        get("/api/job/../../etc/file/model_0.pdb")
        ok(False, "경로 탈출")
    except urllib.error.HTTPError as e:
        ok(e.code == 404, "경로 탈출 차단")
    v = json.load(urllib.request.urlopen(urllib.request.Request(base + "/api/validate", json.dumps({"fasta": "MKZ"}).encode(),
                                                                {"Content-Type": "application/json"})))
    ok(v["errors"], "검증 API")
    srv.shutdown()

    # ── 친화도: 일괄 비교·순위·건너뛴 리간드 ──
    ja = app.submit({"kind": "affinity", "fasta": ">T\nMKTAYIAKQRQISFVKSHFSRQ",
                     "ligands": "ATP 아데노신\nCC(=O)Oc1ccccc1C(=O)O 아스피린\nC1CC 깨짐", "mode": "fast"})
    ok(ja["kind"] == "affinity" and len(ja["ligands"]) == 3 and ja["ligands"][0]["name"] == "아데노신", f"친화도 입력 {ja['ligands']}")
    ok(os.path.exists(app.jdir(ja["id"], "inputs", "lig_01.yaml")), "리간드별 YAML")
    y = app.read(app.jdir(ja["id"], "inputs", "lig_02.yaml"))
    ok("binder: B" in y and "msa: empty" in y and "smiles" in y, f"친화도 YAML {y}")
    ra = wait(ja["id"])
    ok(ra["status"] == "done", f"친화도 완료 {ra.get('error')}")
    resa = json.load(open(app.jdir(ja["id"], "result.json")))
    rows = resa["ligands"]
    ok(len(rows) == 3 and "error" in rows[2] and "SMILES" in rows[2]["error"], f"깨진 리간드만 건너뜀 {rows[2]}")
    ok(resa["summary"]["n_ok"] == 2 and resa["summary"]["n"] == 3, "성공·전체 개수")
    # 가짜 엔진: lig_01 → log10=-0.5, p=0.7 / lig_02 → log10=0.0, p=0.5
    ok(rows[0]["log10_ic50_uM"] == -0.5 and rows[0]["ic50_uM"] == round(10 ** -0.5, 4), f"IC50 변환 {rows[0]}")
    ok(rows[0]["pic50"] == 6.5, f"pIC50 = 6 - log10 {rows[0]['pic50']}")
    ok(rows[0]["p_binder"] == 0.7 and rows[0]["rank_p"] == 1 and rows[1]["rank_p"] == 2, "결합 확률 순위")
    ok(rows[0]["rank_ic50"] == 1 and rows[1]["rank_ic50"] == 2, "IC50 순위")
    ok(rows[0]["lig_plddt"] == 80.0 and rows[0]["ligand_iptm"] == 0.4, f"리간드 신뢰도 {rows[0]['lig_plddt']}")
    ok(os.path.exists(app.jdir(ja["id"], "models", "l00_model_0.pdb")), "리간드별 구조 저장")
    seen2 = {}
    app.chat = lambda sys_, user, model=None: seen2.setdefault("u", user) and "**한 줄 요약** 친화도"
    ok("친화도" in app.explain(ja["id"]) and "결합체 확률" in seen2["u"] and "log10(IC50" in seen2["u"], "친화도 해설 지표")

    # ── 설계: ProteinMPNN ──
    pdb_text = app.read(app.jdir(j1["id"], "models", "model_0.pdb"))
    info = app.pdb_info(pdb_text)
    ok([c["id"] for c in info] == ["A", "B"] and info[0]["n"] == 22 and info[0]["seq"] == "A" * 22, f"PDB 파싱 {[(c['id'], c['n']) for c in info]}")
    fx, fe = app.parse_fixed("A1-3, A10, B", info)
    ok(not fe and fx[:4] == ["A1", "A2", "A3", "A10"] and len([x for x in fx if x[0] == "B"]) == info[1]["n"], f"고정 잔기 {fx[:6]}")
    ok(app.parse_fixed("Z1", info)[1] and app.parse_fixed("A999", info)[1], "없는 사슬·잔기 거부")
    _, de = app.validate_design({"pdb": pdb_text, "chains": ["A"], "fixed": "A"})
    ok(any("모두 고정" in x for x in de), "전부 고정 거부")
    _, de = app.validate_design({"pdb": "ATOM  없음"})
    ok(any("CA" in x for x in de), "CA 없는 PDB 거부")
    jd = app.submit({"kind": "design", "pdb": pdb_text, "filename": "x.pdb", "chains": ["A", "B"],
                     "fixed": "A1-3", "n": 6, "temperature": 0.2, "omit": "cx"})
    ok(jd["n"] == 6 and jd["omit"] == "C" and jd["fixed"] == ["A1", "A2", "A3"], f"설계 입력 {jd['n']} {jd['omit']}")
    rd = wait(jd["id"])
    ok(rd["status"] == "done", f"설계 완료 {rd.get('error')}")
    resd = json.load(open(app.jdir(jd["id"], "result.json")))
    ok(len(resd["designs"]) == 6 and resd["chain_ids"] == ["A", "B"], f"설계 {len(resd['designs'])}개")
    d0 = resd["designs"][0]
    ok(abs(d0["score"] + math.log(d0["confidence"])) < 1e-3, f"score = -ln(confidence) {d0}")
    ok(resd["designs"] == sorted(resd["designs"], key=lambda d: d["score"]), "score 오름차순")
    ok(d0["mutations"] == sum(1 for a, b in zip(resd["native"], d0["seq"]) if a != b), "치환 수")
    ok(d0["global_score"] is not None and abs(d0["global_score"] - (d0["score"] - 0.05)) < 2e-3, f"global_score {d0}")
    fa = app.read(app.jdir(jd["id"], "designs.fa"))
    ok(fa.startswith(">native") and fa.count(">") == 7, f"FASTA {fa.count('>')}개 레코드")

    # ── score 어긋나면 실패로 (FASTA 헤더 ↔ 재계산 대응) ──
    bad = app.submit({"kind": "design", "pdb": pdb_text, "chains": ["A"], "n": 4})
    wait(bad["id"])
    sp = app.jdir(bad["id"], "scores.json")
    rows = json.load(open(sp))
    rows[0]["score"] += 1.0
    json.dump(rows, open(sp, "w"))
    try:
        app.postprocess_design(bad["id"])
        ok(False, "score 불일치를 잡지 못함")
    except RuntimeError as e:
        ok("불일치" in str(e), f"score 불일치 감지 {e}")

    # ── mmCIF 업로드 ──
    ok(app.is_cif("data_x\n_atom_site.id\n") and not app.is_cif(pdb_text), "mmCIF 판별")

    # ── 재예측 검증 루프 (RMSD) ──
    jv = app.validate_designs(jd["id"], {"native": True, "designs": [d0["i"]], "mode": "fast"})
    ok(len(jv) == 2 and jv[0]["ref"]["design"] == 0 and jv[1]["ref"]["design"] == d0["i"], "검증 작업 생성")
    for x in jv:
        wait(x["id"])
    vs = app.validations(jd["id"])
    ok(len(vs) == 2 and all(v["status"] == "done" for v in vs), f"검증 완료 {vs}")
    ok(all(v["rmsd"] == 0.0 for v in vs), f"같은 좌표 → RMSD 0 {[v['rmsd'] for v in vs]}")
    ok(os.path.exists(app.jdir(jv[0]["id"], "models", "ref_aligned.pdb")), "겹친 원구조 저장")

    # ── 겹침 수학 (회전·평행이동 복원) ──
    import random as _r
    _r.seed(7)
    Q = [(_r.uniform(-9, 9), _r.uniform(-9, 9), _r.uniform(-9, 9)) for _ in range(30)]
    th = 0.9
    P = [(q[0] * math.cos(th) - q[1] * math.sin(th) + 3, q[0] * math.sin(th) + q[1] * math.cos(th) - 2, q[2] + 7) for q in Q]
    ok(app.superpose(P, Q)[0] < 1e-6, "회전+평행이동만 다르면 RMSD 0")
    ok(abs(app.superpose([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 0), (1, 0, 0), (0, 2, 0)])[0] - 0.451476) < 1e-5, "RMSD 값 (numpy SVD Kabsch 와 대조)")

    # ── HTTP: 새 경로 ──
    srv2 = app.ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    b2 = f"http://127.0.0.1:{srv2.server_address[1]}"
    g2 = lambda p: urllib.request.urlopen(b2 + p, timeout=10)
    ok(g2(f"/api/job/{jd['id']}/fasta").read().startswith(b">native"), "FASTA 내려받기")
    ok(g2(f"/api/job/{jd['id']}/input").read().startswith(b"ATOM"), "원구조 PDB")
    ok(len(json.load(g2(f"/api/job/{jd['id']}/validations"))) == 2, "검증 목록 API")
    ok(g2(f"/api/job/{ja['id']}/file/l00_model_0.pdb").read().startswith(b"ATOM"), "리간드별 PDB 경로")
    ok(g2(f"/api/job/{jv[0]['id']}/file/ref_aligned.pdb").read().startswith(b"ATOM"), "겹친 원구조 경로")
    z2 = zipfile.ZipFile(__import__("io").BytesIO(g2(f"/api/job/{ja['id']}/zip").read()))
    ok(any(n.startswith("inputs/") for n in z2.namelist()) and "models/l00_model_0.pdb" in z2.namelist(), f"친화도 ZIP {z2.namelist()[:3]}")
    pi = json.load(urllib.request.urlopen(urllib.request.Request(b2 + "/api/pdbinfo", json.dumps({"pdb": pdb_text}).encode(),
                                                                 {"Content-Type": "application/json"})))
    ok([c["id"] for c in pi["chains"]] == ["A", "B"], "PDB 정보 API")
    srv2.shutdown()

    # ── 재시작 복구 ──
    app.update_job(j2["id"], status="running")
    app.QUEUE.clear()
    app.recover()
    ok(app.load_job(j2["id"])["status"] == "error", "재시작 시 실행 중 작업 실패 처리")
    app.delete_job(j3["id"])
    ok(not os.path.exists(app.jdir(j3["id"])), "작업 삭제")


def array_of(n):
    return [float(i) for i in range(n * n)]


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"selftest OK ({N} checks)")
