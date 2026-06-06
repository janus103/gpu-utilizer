Algorithm 2 (후) Entropy Minimization 모사를 위한 강화학습 에피소드 (수정 스케치)

목적: 모든 클래스 대표 이미지(각 1장) 에 대해, 모든 augmentation policy(강도 포함) 를 적용한 입력에서 분류 모델의 예측 확률분포 엔트로피를 감소시키도록, 정규화 레이어(예: BN/LN/GN 등)의 weight(γ), bias(β) 업데이트량을 RL로 학습한다.

정의

대표 데이터셋:

X = {x^1, …, x^C}

각 x^k는 클래스 k의 대표 이미지 1개

Augmentation policy 집합:

Π = {π_1, …, π_{N_Aug}}

각 π_i는 연구자가 정한 augmentation type(예: brightness/contrast/blur 등)

각 적용에는 강도 λ ∈ [0, 1]가 포함됨

분류 모델(적응 대상): M_Pre

정규화 레이어 집합: L_norm

각 레이어 ℓ ∈ L_norm에 대해 파라미터 (γ_ℓ, β_ℓ) 존재

RL 정책/에이전트: M_RL

리플레이 버퍼: B

엔트로피 정의(분류 결과 확률분포):

p = softmax(logits)

H(p) = - Σ_{c=1..C} p[c] * log(p[c])

에피소드(1회) 의 총 step 수

각 클래스 대표 이미지(총 C개) × augmentation 개수(N_Aug) × 한 데이터 포인트당 반복(N_step)

N_episode_step = C * N_Aug * N_step



# 알고리즘 (하기 내용)

Input: X={x^1,…,x^C}, Π={π_1,…,π_{N_Aug}}, M_Pre, M_RL, N_step, replay buffer B


0) Reset
M_Pre ← M_Pre^(0) # 기본: 매 에피소드 시작 시 모델(적응 파라미터)을 초기 상태로 리셋


1) Augmented Set 정의(명시)
# 각 클래스당 이미지가 1개이므로, 전체 augmented 입력 개수는 C * N_Aug
for k = 1..C:
for i = 1..N_Aug:
x_tilde^{k,i} ← π_i(x^k; λ_{k,i})


N_set = C * N_Aug


2) 모든 클래스 대표 데이터의 모든 augmentation에 대해 inner-loop 적응
for k = 1..C:
for i = 1..N_Aug:
x_tilde ← x_tilde^{k,i}


# 기준 엔트로피(업데이트 전)
p_0 ← M_Pre(x_tilde) # softmax 확률분포
H_0 ← H(p_0) = - Σ_{c=1..C} p_0[c] * log(p_0[c])


# 한 데이터 포인트에 대한 반복
for t = 1..N_step:


2.1) State 구성
s_t ← BuildState(
x_tilde, k, i, λ_{k,i}, t,
p_t, H_t,
summary(γ,β), summary(norm_stats)
)


2.2) Action 샘플링 (정규화 파라미터 변경량)
a_t ← M_RL(s_t)
a_t = { (Δγ_ℓ, Δβ_ℓ) } for all ℓ in L_norm


2.3) Environment update (정규화 파라미터 갱신)
for each ℓ in L_norm:
γ_ℓ ← γ_ℓ + Δγ_ℓ
β_ℓ ← β_ℓ + Δβ_ℓ
# (선택) clip/constraint: Δγ,Δβ 혹은 γ,β 범위 제한


2.4) Update 후 엔트로피 계산
p_{t+1} ← M_Pre(x_tilde)
H_{t+1} ← H(p_{t+1})


2.5) Reward 계산
r_ent ← H_t - H_{t+1} # 엔트로피 감소량(감소하면 +)
if argmax(p_{t+1}) != k:
r_cls ← -2 # 정답(대표 클래스) 불일치 페널티
else:
r_cls ← 0
r_t ← r_ent + r_cls


2.6) Next state
s_{t+1} ← BuildState(..., t+1, p_{t+1}, H_{t+1}, ...)


2.7) Replay buffer에 transition 등록 (매 step 종료 시)
B ← B ∪ {(s_t, a_t, r_t, s_{t+1})}


2.8) Roll forward
p_t ← p_{t+1}
H_t ← H_{t+1}


3) Episode 종료
모든 (k,i)에 대해 N_step 반복이 끝나면 episode 종료.


4) Training 반복
위 episode를 수천~수만 회 반복 수행하며, B에서 샘플링해 M_RL을 업데이트.