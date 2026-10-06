#!/usr/bin/env python3
"""protein-local 서열 설계 실행기 — proteinmpnn conda 환경(ligandmpnn 패키지, 가중치 동봉)의 python 으로 돈다.

  ~/miniforge3/envs/proteinmpnn/bin/python runner_mpnn.py <job_dir>
job_dir/params.json + job_dir/input.pdb → job_dir/out/seqs/input.fa, GPU 최대 메모리는 job_dir/runner.json.
--save_stats 로 남는 log_probs 에서 ProteinMPNN 의 두 점수를 다시 계산해 job_dir/scores.json 에 쓴다:
  score        = 설계한 잔기(mask·chain_mask) 평균 음의 로그확률  (= −ln overall_confidence, FASTA 헤더와 같은 값)
  global_score = 구조의 모든 잔기(mask) 평균 음의 로그확률         (고정·비설계 사슬 포함)
"""
import json
import os
import sys
import time


def main(job):
    p = json.load(open(os.path.join(job, "params.json"), encoding="utf-8"))
    argv = ["mpnn", "--model_type", p["model"], "--pdb_path", os.path.join(job, "input.pdb"),
            "--out_folder", os.path.join(job, "out"), "--batch_size", str(p["batch_size"]),
            "--number_of_batches", str(p["batches"]), "--temperature", str(p["temperature"]),
            "--seed", str(p.get("seed") or 1), "--chains_to_design", ",".join(p["chains"]), "--verbose", "1",
            "--save_stats", "1"]
    if p.get("fixed"):
        argv += ["--fixed_residues", " ".join(p["fixed"])]
    if p.get("omit"):
        argv += ["--omit_AA", p["omit"]]
    print("[runner] " + " ".join(argv), flush=True)
    import torch
    from ligandmpnn.run import main as mpnn_main
    t0 = time.time()
    sys.argv = argv
    try:
        mpnn_main()
    finally:
        info = {"seconds": round(time.time() - t0, 1)}
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
        json.dump(info, open(os.path.join(job, "runner.json"), "w"))
    print(f"[runner] 설계 끝 {info['seconds']}s, GPU peak {info.get('gpu_peak_mb', '?')} MiB", flush=True)
    scores(job)


def scores(job):
    """stats/*.pt → scores.json. 행 순서 = FASTA 의 id-1 (run.py 가 id = ix + 1 로 쓴다)"""
    import glob
    import torch
    st = torch.load(glob.glob(os.path.join(job, "out", "stats", "*.pt"))[0], map_location="cpu")
    S, lp = st["generated_sequences"], st["log_probs"]               # [B, L], [B, L, 21]
    nll = -torch.gather(lp, -1, S.unsqueeze(-1)).squeeze(-1)          # [B, L]
    m_all = st["mask"].float()
    m_des = m_all * st["chain_mask"].float()
    out = [{"id": i + 1, "score": float((nll[i] * m_des).sum() / (m_des.sum() + 1e-8)),
            "global_score": float((nll[i] * m_all).sum() / (m_all.sum() + 1e-8))} for i in range(S.shape[0])]
    json.dump(out, open(os.path.join(job, "scores.json"), "w"))
    print(f"[runner] score·global_score {len(out)}개 계산", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
