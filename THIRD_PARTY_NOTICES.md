# THIRD_PARTY_NOTICES — protein-local

이 저장소의 코드(`app.py`·`runner.py`·`runner_mpnn.py`·`gpu_pick.py`·`ui.html`·`selftest.py`·`setup.sh`)는 DongJu Kim(gggg8657)이 작성했으며 PolyForm Noncommercial 1.0.0 입니다. 웹 서버는 Python 표준 라이브러리만 씁니다. 아래는 동봉했거나, 서버의 별도 conda 환경에 설치해 서브프로세스로 부르거나, 예시로 쓰는 제3자 구성요소입니다.

조사 기준: 2026-10-12. 실제 파일(import 문, `setup.sh`, `static/`)을 보고, 라이선스는 설치된 패키지의 dist-info LICENSE·메타데이터와 원 저장소에서 확인했습니다. 버전은 이 서버에 설치된 값입니다(고정 아님).

**ESM(ESM2/ESM3/ESM C)·AlphaFold·ColabFold 는 쓰지 않습니다.** 코드·setup 어디에도 없고, MSA 서버도 부르지 않습니다(`msa: empty`, `--use_msa_server` 안 씀).

## 구성요소

| 이름 | 버전 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|---|
| 3Dmol.js | 2.5.5 | **동봉** `static/3Dmol-min.js` | BSD-3-Clause (© 2014 University of Pittsburgh and contributors) | https://github.com/3dmol/3Dmol.js | 가능 | 고지 의무 — 전문 `static/3Dmol.LICENSE.txt` |
| └ GLmol 0.47 (3Dmol.js 안) | 0.47 | 동봉(번들 안) | MIT 또는 LGPL-3.0 이중 — MIT 선택 | (3Dmol.LICENSE.txt) | 가능 | |
| └ three.js, jQuery (3Dmol.js 안) | — | 동봉(번들 안) | MIT | (3Dmol.LICENSE.txt) | 가능 | |
| └ pako (3Dmol.js 안, deflate/inflate) | 1.x | 동봉(번들 안) | MIT (+ zlib) | https://github.com/nodeca/pako | 가능 | 전문 `licenses/pako.LICENSE.txt` |
| Boltz (boltz-1 / boltz-2) | 2.2.1 | 설치(conda env `boltz`), 서브프로세스 | MIT (© 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro) | https://github.com/jwohlwend/boltz | 가능 | dist-info LICENSE 확인 |
| Boltz 가중치 `boltz2_conf.ckpt`·`boltz2_aff.ckpt`·`boltz1_conf.ckpt`, CCD 사전 `mols/` | — | 내려받는 가중치(`BOLTZ_CACHE`, 저장소에 없음) | MIT (Boltz README: "All the code and weights are provided under MIT license") | https://github.com/jwohlwend/boltz | 가능 | `mols/` 는 wwPDB Chemical Component Dictionary 유래 |
| LigandMPNN / ProteinMPNN (`ligandmpnn` pip 패키지, 가중치 포함) | 0.1.2 | 설치(conda env `proteinmpnn`), 서브프로세스 `mpnn` | MIT (© 2024 Justas Dauparas) | https://github.com/dauparas/LigandMPNN | 가능 | 모델 `protein_mpnn`·`soluble_mpnn` 만 씀 |
| PyTorch | 2.5.1 (boltz) / 2.2.1 (mpnn) | 설치(위 환경 의존성) | BSD-3-Clause | https://pytorch.org | 가능 | CUDA 런타임 휠은 NVIDIA 라이선스 |
| RDKit | 2025.9.6 | 설치(boltz 의존성 — SMILES 확인) | BSD-3-Clause | https://github.com/rdkit/rdkit | 가능 | |
| gemmi | 0.6.5 | 설치(boltz 의존성 — mmCIF→PDB 변환) | MPL-2.0 | https://github.com/project-gemmi/gemmi | 가능(조건부) | 수정해 배포할 때만 그 파일 소스 공개 의무. 수정·동봉 안 함 |
| 예시 서열: 사람 유비퀴틴(UniProt P0CG48, PDB 1UBQ), 인슐린(UniProt P01308), GFP(UniProt P42212), Trp-cage TC5b | — | `app.py` 예시 | UniProt: CC BY 4.0 / PDB: CC0 1.0 | https://www.uniprot.org/help/license , https://www.wwpdb.org/about/usage-policies | 가능(출처 표기) | 서열은 사실 데이터. UniProt 출처 표기 |
| 예시 리간드: ATP·NAG (CCD 코드), 아스피린·에탄올·갈락토스(SMILES) | — | `app.py`·`ui.html` 예시 | CCD: wwPDB (CC0 1.0) / SMILES 는 사실 데이터 | https://www.wwpdb.org/data/ccd | 가능 | |
| Google Gemma 4 `gemma4:31b` (Ollama GGUF 빌드) | — | 외부 서비스(OpenAI 호환 API, 결과 설명 LLM) | Apache-2.0 (Ollama `api/show` 의 license 필드로 확인) | https://ai.google.dev/gemma | 가능 | 가중치 동봉 안 함 |
| Ollama | — | 외부 서비스(LLM 서버) | MIT | https://github.com/ollama/ollama | 가능 | 동봉 안 함 |

인용: Boltz-1 — Wohlwend et al., bioRxiv (2024), doi:10.1101/2024.11.19.624167; Boltz-2 — Passaro et al., bioRxiv (2025), doi:10.1101/2025.06.14.659707; ProteinMPNN — Dauparas et al., *Science* (2022), doi:10.1126/science.add2187; LigandMPNN — Dauparas et al., bioRxiv (2023), doi:10.1101/2023.12.22.573103. 결과를 발표할 때는 해당 모델 논문을 인용하세요.

## 확인 필요 (상업 이용 전)

현재 상업 이용 확인이 필요한 항목은 찾지 못했습니다(모두 MIT·BSD·Apache·MPL·CC0/CC BY). 다른 예측 모델(ESM3·ESM C·AlphaFold 3 등 비상업·제한 라이선스)을 추가할 때는 이 표와 이슈로 먼저 확인하세요.

이 문서는 법률 자문이 아니며, 라이선스는 각 원문이 우선합니다.
