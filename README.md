# MuJoCo Grasp Demo — Kuka + Allegro가 책상 위 큐브를 잡은 모습

IsaacLab에서 학습된 **사전학습 Lift 정책이 만든 grasp**(Kuka iiwa + Allegro 손이 2.9cm 큐브를 쥔 자세)를
**MuJoCo에서 볼 수 있게** 한 최소 데모입니다. 강화학습은 포함돼 있지 않습니다 — RL은 이 씬을 기반으로
팀에서 입맛에 맞게 붙이면 됩니다.

> 렌더 결과(`*.png`)는 레포에 커밋하지 않습니다 — `view_grasp.py` 를 실행하면 `grasp.png` 가 생성됩니다.

## 빠른 시작 (셋업)

> 이 레포는 메시·모델·데이터(`.obj/.stl/.npz/.png` 등)를 **Git LFS**로 저장하고, 파이썬 가상환경(`.venv`)은
> 커밋하지 않습니다(`.gitignore`). 따라서 클론 후 **① LFS 받기 → ② 가상환경 만들기** 두 단계가 필요합니다.

### 0. 사전 준비 (한 번만)
```bash
# git-lfs (이미 있으면 건너뜀)
git lfs install
# uv (파이썬/패키지 관리자, 권장). 없으면 아래 'pip 대안' 사용
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. 레포 받기 + LFS 실파일 내려받기
```bash
git clone <레포주소> mujoco_grasp_demo      # 또는 이미 폴더가 있으면 생략
cd mujoco_grasp_demo
git lfs pull                                # ★ 메시/모델/데이터 실파일 받기 (안 하면 OBJ가 포인터만 옴)
```
> 확인: `ls -la assets/isaaclab_kuka_allegro/index_link_3.obj` 가 수십 KB~MB면 정상.
> 약 130B로 작으면 LFS pull이 안 된 것(포인터) → `git lfs pull` 다시 실행.

### 2. 가상환경 구성 (uv 권장)
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python mujoco numpy pillow
```
**pip 대안 (uv 없이):**
```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install mujoco numpy pillow
```
의존성: `mujoco>=3.10`, `numpy`, `pillow` (PNG 저장용). 그 외 없음.

### 3. 실행 (동작 확인)
```bash
.venv/bin/python view_grasp.py                 # IsaacLab 모델로 grasp 렌더 → grasp.png
.venv/bin/python view_grasp.py --idx 100       # 다른 grasp (0 ~ 6199)
.venv/bin/python view_grasp.py --view          # 인터랙티브 뷰어 (디스플레이 있을 때)
```
- 헤드리스(서버)면 위 PNG 렌더가 그대로 동작합니다(EGL 오프스크린). 디스플레이가 있으면 `--view`로 마우스 회전 가능.
- 카메라는 grasp를 자동으로 측면에서 비춥니다. 수동 조정: `--azimuth 90 --elevation -15 --dist 0.3`,
  팔까지 다 보려면 `--dist 1.0`.

### 트러블슈팅
- `OBJ 로드 실패 / 메시가 비었다` → `git lfs pull` 안 함. 1번 다시.
- `MUJOCO_GL / EGL 오류` → 헤드리스에서 `MUJOCO_GL=egl`(스크립트가 기본 설정). GPU EGL이 없으면
  `MUJOCO_GL=osmesa`로 시도(`uv pip install ...` 시 osmesa 백엔드 필요할 수 있음).
- 종료 시 나오는 `EGLError ... context free` 경고는 무해(PNG는 정상 저장됨).

## 구성

| 파일 | 설명 |
|---|---|
| `grasp_scene.py` | 씬 조립: MuJoCo Menagerie **iiwa14 + Allegro(right)** 를 합치고 책상·큐브·카메라 추가. `build_model()` 이 컴파일된 `MjModel` 반환. 단독 실행 시 결합 XML(`kuka_allegro_cube.xml`)도 export. |
| `view_grasp.py` | harvest된 grasp 자세를 적용해 **손이 큐브를 쥔 그림을 렌더(PNG)** 하거나 인터랙티브 뷰어 실행. |
| `kuka_allegro_isaaclab.xml` | 단독 로드 가능한 **IsaacLab 모델**(메시 포함, self-contained). RL 프레임워크에 바로 물리기 좋음. (menagerie 단독 XML은 `grasp_scene.py` 실행 시 생성 — gitignore) |
| `data/grasp_states.npz` | IsaacLab 사전학습 Lift 정책에서 수확한 **6,200개 grasp 상태**(관절각 + 큐브 포즈). grasp 자세의 출처. |
| `assets/` | `isaaclab_kuka_allegro/`(추출 메시+model.json) + Menagerie 에셋(`wonik_allegro`, `kuka_iiwa_14`). |
| `.venv/` | uv로 만든 전용 가상환경 (mujoco 3.10, numpy, pillow). |

## ★ 두 가지 손 모델 (source)

- **`isaaclab` (기본, 정확)**: IsaacLab의 실제 `kuka.usd`에서 메시·운동학을 추출(`extract_kuka.py`)해
  MuJoCo로 재조립한 **진짜 IsaacLab Kuka iiwa7 + Allegro**. 손끝까지 IsaacSim과 동일 지오메트리.
  씬 빌더 `grasp_scene_isaaclab.py`, 에셋 `assets/isaaclab_kuka_allegro/`(OBJ + model.json),
  단독 모델 `kuka_allegro_isaaclab.xml`. harvest 관절명이 그대로 일치해 팔 각도까지 직접 적용(IK 불필요).
- **`menagerie` (근사, 폴백)**: MuJoCo Menagerie iiwa14 + wonik_allegro(손끝 메시 교체). `grasp_scene.py`.

```bash
.venv/bin/python view_grasp.py --source isaaclab   # 정확(기본)
.venv/bin/python view_grasp.py --source menagerie  # 근사
.venv/bin/python extract_kuka.py                   # (컨테이너에서) USD->OBJ 재추출
```

## 실행

```bash
# (PNG 렌더 — 헤드리스 OK)
.venv/bin/python view_grasp.py                 # 기본 grasp(idx 0) → grasp.png
.venv/bin/python view_grasp.py --idx 7         # 다른 harvest grasp 선택 (0 ~ 6199)
.venv/bin/python view_grasp.py --view          # 인터랙티브 뷰어 (디스플레이 필요)

# (씬만 다시 빌드 + 결합 XML export)
.venv/bin/python grasp_scene.py            # menagerie XML
.venv/bin/python grasp_scene_isaaclab.py   # isaaclab XML
```
> 가상환경 만드는 법은 위 **빠른 시작 §2** 참고.

## 어떻게 동작하나 (정확성 메모)

- **grasp 출처**: `data/grasp_states.npz` 는 IsaacLab 사전학습 Kuka+Allegro **Lift 정책**을 2.9cm 큐브에
  굴려, 큐브를 엄지·검지로 쥔 프레임을 포착 → 섭동·정착·재검증(HORA식)해 모은 6,200개 상태다.
  (이 정책 계보 = DexPBT. 자세한 내용은 `dexsuite_dg5f/docs/ALLEGRO_INHAND_REPORT.md`.)
- **손가락 매핑**: IsaacLab `{index/middle/ring/thumb}_joint_J` → Menagerie `{ff/mf/rf/th}j{J}`.
  손가락 16관절 각도는 **그대로 전이**된다.
- **팔(arm)은 재계산**: harvest는 IsaacLab의 **iiwa7**, 이 씬은 Menagerie **iiwa14**(다른 기종, 링크 길이
  다름)라 팔 각도는 그대로 못 쓴다. 대신 **damped-LS IK**로 손을 책상 위로 가져가게 푼다(팔 자세는 근사,
  손가락 grasp는 정확).
- **큐브 배치**: 프레임 규약 차이를 피하려고 큐브를 **엄지·검지 손끝 사이**에 둔다 → 항상 "쥔" 것으로 보인다.
- **손끝 메시**: Menagerie 기본 손끝은 둥근 캡이라 IsaacLab과 달라 보여서, **표준 Allegro 뾰족 손끝 메시**
  (`assets/wonik_allegro/assets/pointed_fingertip.obj`, graspqp의 `allegro/meshes/fingertip.obj)`)로 교체하고
  둥근 캡을 제거했다. distal 4개의 visual 메시만 바뀌고 관절/운동학은 동일.
- ⚠️ 이건 **시각화 데모**다. 물리적으로 grasp가 중력 하에서 안정적으로 유지된다는 보장은 아니다
  (정적 포즈 + 운동학 배치). RL/제어는 팀원이 이 씬 위에서 설계.

## 팀원에게 (RL 붙일 때)

- 모델: `grasp_scene_isaaclab.build_model()`/`grasp_scene.build_model()` (Python) 또는 `kuka_allegro_isaaclab.xml` (단독 로드).
- 관절: 팔 `joint1..7`, 손가락 `rh_{ff/mf/rf/th}j{0..3}`, 큐브는 free joint(body `cube`).
- 초기 grasp 상태가 필요하면 `data/grasp_states.npz`(6,200개)를 reset 분포로 쓰면 된다
  (`joint_pos[23]`, `obj_pos_r`/`obj_quat_r`=robot-root 프레임, `obj_pos_p`/`obj_quat_p`=palm 프레임).
