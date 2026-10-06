#!/usr/bin/env python3
"""protein-local 엔진 실행기 — boltz conda 환경의 python 으로 돈다 (app.py 가 서브프로세스로 부름).

  ~/miniforge3/envs/boltz/bin/python runner.py <job_dir>
job_dir/params.json(engine·옵션) + job_dir/input.yaml 을 읽어 Boltz 를 같은 프로세스에서 돌리고,
GPU 최대 메모리를 job_dir/runner.json 에 남기고, 예측 PDB 마다 mmCIF 를 옆에 만든다.
외부 호출 없음: --use_msa_server 를 쓰지 않는다 (입력 YAML 에 msa: empty).
"""
import glob
import json
import os
import sys
import time


def check_ligands(job, params):
    """SMILES 는 RDKit 으로 미리 읽어 본다 — Boltz 의 긴 traceback 대신 알아보기 쉬운 오류.
    친화도 일괄(inputs/ 폴더)에서는 문제 리간드만 빼고(skipped.json) 나머지는 돌린다. 친화도는 무거운 원자 128개까지(Boltz 제한)."""
    from rdkit import Chem
    bad, skipped = [], {}
    for lig in params.get("ligands") or []:
        why = None
        if lig.get("smiles"):
            m = Chem.MolFromSmiles(lig["smiles"])
            if m is None:
                why = "SMILES 를 읽을 수 없음 (CCD 코드라면 사전에 없는 코드)"
            elif params.get("kind") == "affinity" and m.GetNumHeavyAtoms() > 128:
                why = f"무거운 원자 {m.GetNumHeavyAtoms()}개 — Boltz 친화도는 128개까지"
        if why and lig.get("file"):
            skipped[lig["file"]] = why
            os.replace(os.path.join(job, "inputs", lig["file"] + ".yaml"), os.path.join(job, lig["file"] + ".skipped.yaml"))
            print(f"[runner] 건너뜀 {lig.get('name') or lig.get('label')}: {why}", flush=True)
        elif why:
            bad.append(f"{lig['smiles']} ({why})")
    if skipped:
        json.dump(skipped, open(os.path.join(job, "skipped.json"), "w"), ensure_ascii=False)
    if bad:
        raise SystemExit("리간드 SMILES 를 읽을 수 없습니다: " + ", ".join(bad))
    if params.get("kind") == "affinity" and not os.listdir(os.path.join(job, "inputs")):
        raise SystemExit("돌릴 리간드가 없습니다 (모두 건너뜀)")


def to_cif(job):
    import gemmi
    for p in glob.glob(os.path.join(job, "out", "**", "*.pdb"), recursive=True):
        st = gemmi.read_structure(p)
        st.setup_entities()
        st.make_mmcif_document().write_file(p[:-4] + ".cif")


def main(job):
    params = json.load(open(os.path.join(job, "params.json"), encoding="utf-8"))
    check_ligands(job, params)
    args = ["predict", os.path.join(job, params.get("input") or "input.yaml"), "--out_dir", os.path.join(job, "out"),
            "--cache", params.get("cache") or os.path.expanduser("~/.boltz"), "--model", params["engine"],
            "--output_format", "pdb", "--diffusion_samples", str(params["samples"]),
            "--recycling_steps", str(params["recycling"]), "--sampling_steps", str(params["steps"]),
            "--write_full_pae", "--override", "--num_workers", "0"]
    if params.get("seed") is not None:
        args += ["--seed", str(params["seed"])]
    if params.get("no_kernels", True):
        args.append("--no_kernels")  # cuequivariance 커널이 이 환경에서 import 되지 않는다 — 기본값으로 끈다
    print("[runner] boltz " + " ".join(args), flush=True)
    import torch
    from boltz.main import cli
    t0 = time.time()
    try:
        cli.main(args, standalone_mode=False)
    finally:
        info = {"seconds": round(time.time() - t0, 1)}
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
        json.dump(info, open(os.path.join(job, "runner.json"), "w"))
    print(f"[runner] 예측 끝 {info['seconds']}s, GPU peak {info.get('gpu_peak_mb', '?')} MiB", flush=True)
    to_cif(job)
    print("[runner] mmCIF 변환 완료", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
