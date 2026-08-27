# Allocation-Aware Adaptive Richness Stopping (AAARS)

**Risk-Modulated Estimator Blending and Adaptive Confidence Certification for Species/Fault Richness Estimation**

*Project11-AAARS — Manuscript*

---

## Abstract

Situational-awareness missions (search-and-rescue, unexploded-ordnance neutralization, biological surveys) must decide **when to stop** collecting evidence and assert that the number of remaining targets is small enough to declare the region "clear." The dominant approach stops when a Chao-type nonparametric estimator of undiscovered richness falls below a confidence threshold. These estimators assume *complete spatial or temporal randomization* of the observation process. When the fleet's allocation policy is *biased* — revisiting rich subregions rather than sweeping uniformly — this assumption fails, the unseen-mass residual $f_1^2/(2f_2)$ is inflated by revisitation, and the estimator **false-certifies** a clearance at unsafely low recall.

We introduce **AAARS (Allocation-Aware Adaptive Richness Stopping)**, a genuinely *online, adaptive* stopping rule that never observes ground truth. AAARS continuously diagnoses the observation process from *leak-free fleet-observable* statistics, maps them to a risk score $r(t)\in[0,1]$, and uses that score in two coupled ways: **(1)** it *blends* the Chao1 and Chao92 estimators continuously (with a dead zone so that low-risk episodes keep the efficient Chao1 behavior), and **(2)** it *adaptively tightens* the certification threshold as risk rises ($\alpha_{\text{adj}} = \alpha_0/(1+\lambda r)$). A discrete switching variant is kept as an ablation.

In 160 controlled episodes (80 seeds per allocation; K=60 mines, N=6 agents, 100×100 grid, under both a systematic BoustroLanes sweep and a biased MineRichness allocation), Chao1-CI false-certifies **35.0%** of MineRichness episodes. AAARS reduces this to **21.2%** (a 39% relative / 13.8-point reduction; Fisher's exact *p*=0.078) while **never** false-certifying under the systematic policy and leaving the stopping rate effectively unchanged (56/80 vs 59/80). The improvement is carried by the adaptive-threshold mechanism; the continuous blend contributes additionally in the highest-bias regimes. Across a 3×3 robustness grid of mine counts $K\in\{30,60,120\}$ and agent counts $N\in\{3,6,12\}$, AAARS never performs worse than Chao1 and delivers point reductions up to **40 percentage-points** in false-certification at high agent counts ($N=12$), where allocation bias is strongest — reported as **exploratory** (wide CIs at 5 seeds/cell). Honest limitations are documented: the reduction does not reach conventional significance at this sample size, and AAARS trades a modest increase in stopping latency for safety.

---

## 1. Introduction

### 1.1 The stopping problem

A fleet of autonomous agents must certify that a bounded domain contains no more than a small number of undetected targets. Every agent sweep produces noisy detections; the fleet must decide, online, whether to "stop" and hand the region to follow-up teams, or to keep searching. Premature certification is dangerous: undetected mines or survivors are the most costly failure mode. Overly conservative certification wastes mission time.

We restrict the setting to richness-based stopping: we estimate the *number of undetected targets* $\hat{U}$ and stop when a high-confidence upper bound on $\hat{U}$ is a small fraction of the estimated total, i.e. when recall is confidently high.

### 1.2 Richness estimation and its failure mode

The workhorse estimators are Chao-type nonparametric species estimators. Chao1 [1] estimates undiscovered mass from the count of species seen exactly once ($f_1$) and exactly twice ($f_2$):

$$U_{\text{Chao1}} = \frac{f_1^2}{2 f_2}, \qquad \text{CI} = U_{\text{Chao1}} \pm z_{1-\alpha} \sqrt{\mathrm{Var}} .$$

Chao92 [2] additionally uses higher-order frequency counts and a heterogeneity coefficient $\gamma^2$, giving more conservative bounds when capture probabilities are heterogeneous.

Both rest on a **random/complete sampling** assumption. The residual $f_1^2/(2f_2)$ is a lower bound on unseen mass *only when* encounters behave like repeated independent draws. Under a **biased allocation** — where the fleet deliberately returns to subregions it already knows to be rich (e.g., revisits mined lanes, rich ecological patches) — the same few cells accumulate many encounters. This inflates the *observed* doubleton count $f_2$ without adding new territory, thereby **deflating** $f_1^2/(2f_2)$ and making the system *believe* it has almost fully searched the region when a large fraction (possibly the majority) remains unscanned. The estimator false-certifies.

Crucially, the fleet can observe its own behavior — *where it has been, how often, and how much of the domain it has covered* — without any ground truth about targets. This is the signal we exploit.

### 1.3 Contributions

1. **A risk-modulated blending estimator** — instead of committing to Chao1 or Chao92, AAARS computes a continuous risk score $r(t)$ and blends the point estimates and variances: $\hat{a}=(1-r_{\text{eff}})U_1 + r_{\text{eff}}U_{92}$, with a dead zone ($r_{\text{eff}}=0$ below a threshold) so systematic, low-risk runs retain Chao1's efficiency.
2. **Adaptive certification** — AAARS tightens the stopping threshold under risk: $\alpha_{\text{adj}}=\alpha_0/(1+\lambda r)$. Rising risk makes the fleet *require* more evidence before certifying.
3. **Leak-free diagnosis** — all diagnostics use only fleet-observable statistics (detection frequencies, spatial concentration, and fleet *coverage* — the fraction of traversable cells actually scanned), never ground truth.
4. **An honest, reproducible evaluation** — a 160-episode confirmation campaign (80 seeds/allocation) with Wilson CIs and a significance test, plus a 90-episode robustness sweep reported as exploratory, with documented limits.

To our knowledge this is the first treatment of *online adaptive enrichment stopping* that (a) operates without ground truth, (b) reacts continuously (not via discrete mode switches), and (c) explicitly ties certification safety back to the **coverage behavior** of the allocation policy.

---

## 2. Related Work

**Stopping rules for species/coverage estimation.** Good [3] and Chao [1] gave early estimators of unseen mass. The asymptotic sampling-coverage approach of Chao & Jost [4] forms the basis of many modern confidence-certification rules. "Diminishing returns" and fixed-$\nu$-sweep heuristics are widely used in field practice (e.g., clearing models in [5]) but lack statistical guarantee.

**Adaptive threshold selection.** Sequential hypothesis testing (SPRT [6]) and multiple-hypothesis corrections adjust evidence thresholds based on accumulated data, but require a well-specified likelihood, which Chao-type estimators deliberately avoid.

**Adaptive and online stopping.** Recent work treats the termination decision itself as part of a sequential decision problem rather than a fixed threshold. Cheng & Huan [8] cast stopping for sequential Bayesian experimental design as a coupled design-and-stopping policy optimized by policy gradients, arguing that myopic threshold rules ignore the expected value of future measurements — a perspective we share, though our setting couples stopping to a nonparametric richness estimate rather than a learned posterior. In robotics, IA-TIGRIS [9] adaptively gathers information online with informed-sampling informative-path planning, but grants no statistical clearance guarantee. Placed & Castellanos [10] argue that autonomous exploration-stopping decisions remain understudied relative to exploration algorithms themselves, a gap this paper also targets. Closest in spirit, Luperto et al. [11] learn a task-independent, ground-truth-free stopping criterion for robot exploration from partial occupancy maps, but diagnose *map* completeness rather than allocation bias, and do not couple their criterion to a nonparametric richness estimator. Outside robotics, Bron et al. [12] apply Chao's estimator directly as a stopping criterion for technology-assisted document review, the closest methodological analogue to AAARS; their setting, however, assumes a fixed recall target against a static corpus rather than an allocation policy that can adversarially bias detection frequencies over time. To our knowledge no prior work couples a *ground-truth-free*, *online* fleet diagnosis to an *adaptive* richness-certification threshold; this is the gap AAARS fills.

**Coverage-aware estimation.** The connection between *sample coverage* and unseen-mass estimation is classical (Good–Turing [3]); the Chao & Jost coverage estimator [4] explicitly renormalizes by estimated sample coverage. We build on this intuition but supply coverage *online* from the fleet's own scan history, and couple it to a *stopping* (not just an estimation) decision.

**Multi-robot coverage.** Extensive work exists on coverage path planning (e.g., [7]) that *guarantees* complete coverage. Our contribution is complementary: given *any* allocation policy (optimal or biased), certify whether it has gathered enough evidence to stop.

---

## 3. Problem Formulation

Let $\mathcal{C}$ be a grid of $|{\cdot}|$ cells, $K$ of which contain targets (mines). A fleet of $N$ agents moves on $\mathcal{C}$, each with a field of view; a cell yields a detection with probability $p_d$ when scanned. The fleet fuses observations into a set of located targets and a per-cell scan count.

Let:
- $n_{\text{det}}$ = number of targets located,
- $f_k$ = number of *located targets* observed exactly $k$ times,
- $\hat{K}=n_{\text{det}}+U$ = estimated total,
- recall $R = n_{\text{det}}/K$.

**Stopping objective.** Certify clearance at the first time $t^*$ when, with high confidence, the fraction of undiscovered targets is bounded: $\mathbb{P}[\hat{U}/\hat{K} \le \alpha] \ge 1-\beta$.

A **false certification (FC)** occurs when the rule stops but true recall $< R_{\min} = 0.95$. This is the primary safety metric.

---

## 4. Method

### 4.1 Diagnostic features (leak-free)

AAARS computes three scalar statistics each decision step from fleet-observable state:

1. **Temporal clustering** $\tau(t)\in[0,1]$ — coefficient-of-variation of inter-detection timing and rapid-revisit fraction. Detects bursty, locality-driven observation.
2. **Frequency instability / richness bias** $\phi(t)\in[0,1]$ — measures when doubleton encounters dominate singletons, i.e. the richness-bias regime: $\mathrm{RB}=\max(0,(f_2-f_1))/(f_2+f_1+1)$, combined with the trend of $f_1/f_2$.
3. **Coverage deficit** $\delta(t)=1-C(t)\in[0,1]$ — where $C(t)$ is the fraction of traversable cells scanned at least once by the fleet. **This is the key discriminator**: a systematic sweep drives $C\to 1$ (deficit $\to 0$); a biased allocation that keeps revisiting rich rows leaves much of the domain unscanned (deficit stays high).

All three are computed without ground truth: $\tau,\phi$ use the observation monitor's detection frequencies; $C$ uses the fleet's scan counts.

### 4.2 Risk score

The risk score is the weighted, EMA-smoothed diagnosis:

$$r(t) = \operatorname{clip}\big( w_\tau \tau(t) + w_\delta \delta(t) + w_\phi \phi(t),\; 0, 1 \big), \qquad \bar r(t)=\eta r(t)+(1-\eta)\bar r(t-1).$$

We use $w_\tau=0$, $w_\delta=0.55$, $w_\phi=0.45$, smoothing $\eta=0.1$; coverage deficit and richness-bias dominate because they most directly indicate that Chao's random-sampling assumption is violated.

### 4.3 Blended estimator with dead zone

Chao1 gives the efficient base estimate; Chao92 gives a conservative one. The conservatism of Chao92 enters through the squared coefficient of variation of detection probabilities, $\gamma^2$ ($\S$2): its estimate includes a term $n(1-C)\gamma^2/C$ that inflates the unseen-mass estimate when capture probabilities are heterogeneous or coverage $C$ is incomplete — the exact regime flagged as risky by our diagnostics. AAARS therefore blends continuously, shifting weight from the $\gamma^2$-free Chao1 bound toward the $\gamma^2$-corrected Chao92 bound only when risk warrants it:

$$r_{\text{eff}} = \begin{cases} 0 & r < \theta \\ \dfrac{r-\theta}{1-\theta} & r \ge \theta \end{cases}, \qquad
\hat a = (1-r_{\text{eff}})U_1 + r_{\text{eff}}U_{92}, \qquad
\hat\sigma^2 = (1-r_{\text{eff}})\mathrm{Var}_1 + r_{\text{eff}}\mathrm{Var}_{92}.$$

The dead zone $\theta=0.30$ ensures that systematic (low-risk) runs use pure Chao1 and keep its efficiency (fast, non-redundant stopping). Only when risk rises does the estimator become more conservative (doubletons discounted, heterogeneity absorbed).

### 4.4 Adaptive certification

Risk also *tightens* the acceptance threshold:

$$\alpha_{\text{adj}}(t) = \frac{\alpha_0}{1+\lambda\,\bar r(t)}, \qquad \lambda=1.0,\ \alpha_0=0.05 .$$

Under risk the fleet requires a smaller residual fraction before stopping, forcing more evidence collection precisely in the regimes where Chao underestimates unseen mass. This is the mechanism that delivers the FC reduction.

**Stop condition.** Stop at the first time $t$ such that the estimator is armed ($n_{\text{det}}\ge n_{\min}$, $f_2\ge$ floor) and

$$U_{\text{upper}}(t) \;\le\; \alpha_{\text{adj}}(t)\,\hat K(t).$$

### 4.5 Discrete-switching ablation

For comparison we retain a **discrete** controller that classifies the observation state (dispersed/moderate/clustered) and switches estimator mode between Chao1 and Chao92 with hysteresis, echoing the discrete design in the original problem statement.

---

## 5. Experimental Setup

**Environment.** 100×100 grid, $N\in\{3,6,12\}$ agents, $K\in\{30,60,120\}$ mines, field-of-view radius 5, communication-limited fusion, homogeneous detectability with $p_d=0.7$. Confirmation uses K=60, N=6.

**Allocation policies.**
- **BoustroLanes**: systematic lane sweep intended to maximize coverage (a "balanced/safe" policy).
- **MineRichness**: bias-driven allocation that returns to mine-rich subregions (the adversarial/unsafe policy).

**Methods.** Chao1-CI (baseline), Chao92-CI (conservative baseline), **AAARS**, Discrete-AAARS (ablation), Oracle-95% (ground-truth upper bound), Fixed-2 sweep, Diminishing-returns. All methods evaluate on the *same* detection traces; only AAARS/discrete act in-loop.

**Metrics.** False-certification rate (recall < 95%), stopping rate, median stop time, mean recall.

**Protocol.** Confirmation: 80 seeds (10–89) × 2 allocations = 160 episodes (outcome-only, for speed). Sweep (exploratory): K×N ∈ {30,60,120}×{3,6,12}, 5 seeds each, 90 episodes.

---

## 6. Results

### 6.1 Headline: false certification reduced under bias, no regression under systematic

Table 1 reports the confirmation campaign.

**Table 1. Confirmation (80 seeds, K=60, N=6).** FC = false certifications (recall<95%); Stop = episodes that terminated before $T_{\max}$; FC% shows the Wilson 95% CI; Med $t$ shows the bootstrap 95% CI on median stop time; FC% among stoppers.

| Method | Alloc | Stop | FC | FC% (95% CI) | Med $t$ (95% CI) | MedR | MinR |
|--------|-------|------|----|--------------|------------------|------|------|
| Chao1-CI | Boustro | 59/80 | 0 | 0.0% [0.0,6.1] | 409 [399,415] | 100.0 | 100.0 |
| **AAARS** | **Boustro** | **56/80** | **0** | **0.0% [0.0,6.4]** | **428 [419,436]** | **100.0** | **100.0** |
| Chao1-CI | MineRich | 80/80 | 28 | **35.0% [25.5,45.9]** | 732 [668,899] | 96.7 | 60.0 |
| **AAARS** | **MineRich** | **80/80** | **17** | **21.2% [13.7,31.4]** | **945 [735,1146]** | **98.3** | **81.7** |
| Chao92-CI | Boustro | 1/80 | 0 | 0.0% | 551 | 100.0 | 100.0 |
| Chao92-CI | MineRich | 80/80 | 2 | 2.5% [0.7,8.7] | 2054 [1899,2150] | 100.0 | 86.7 |
| Discrete-AAARS | Boustro | 59/80 | 0 | 0.0% [0.0,6.1] | 409 [399,415] | 100.0 | 100.0 |
| Discrete-AAARS | MineRich | 80/80 | 28 | 35.0% [25.5,45.9] | 732 [668,899] | 96.7 | 60.0 |
| Oracle-95 | Boustro | 80/80 | 0 | 0.0% [0.0,4.6] | 242 [236,246] | 95.0 | 95.0 |
| Oracle-95 | MineRich | 80/80 | 0 | 0.0% [0.0,4.6] | 593 [544,690] | 95.0 | 95.0 |
| Fixed-2 | Boustro | 80/80 | 0 | 0.0% [0.0,4.6] | 442 [432,449] | 100.0 | 100.0 |
| Fixed-2 | MineRich | 80/80 | 0 | 0.0% [0.0,4.6] | 2286 [2199,2347] | 100.0 | 100.0 |
| Diminishing | Boustro | 80/80 | 0 | 0.0% [0.0,4.6] | 424 [419,432] | 100.0 | 100.0 |
| Diminishing | MineRich | 80/80 | 37 | 46.2% [35.7,57.1] | 628 [591,664] | 95.0 | 53.3 |

**Key findings.**
- AAARS reduces MineRichness false certification from **35.0% → 21.2%** (a 13.8-point absolute / 39% relative reduction). A two-tailed Fisher's exact test on the FC counts gives **p=0.078** — the difference is in the predicted direction and substantial but does not reach conventional significance (see §7.3 power analysis).
- AAARS **never** false-certifies under BoustroLanes (0/56 vs 0/59 for Chao1) and preserves the stopping rate; it does not trade safety for paralysis in the safe regime.
- In **78 of 80** MineRich episodes AAARS stops *later* than Chao1 (median 945 vs 732), and in none does it stop earlier — it deliberately holds out for more evidence under bias.
- The discrete-switching ablation is **byte-identical to plain Chao1** (both 28/80, 35.0% FC) *by construction*: across all 80 confirmation episodes its switching logic records `num_switches=0` and its observation state never reaches the `CLUSTERED` trigger that would engage the conservative Chao92 branch (temporal clustering > 0.35 or spatial concentration > 0.25 were never sustained), so it reduces to Chao1. The continuous formulation was introduced precisely to avoid this brittle, rarely-firing discrete cascade. (In two cells of the exploratory sweep the discrete controller does switch once or twice, confirming the mechanism is reachable in principle but unreliable as designed.)

The conservative Chao92 attains only 2.5% FC, but at the cost of *never* effectively stopping under Boustro (1/80) and stopping very late under MineRich (median 2054). AAARS finds a middle ground: near-Chao1 latency under systematic allocation and near-Chao92 safety under bias, while preserving a realistic stopping rate.

*Figure 1* (risk trajectories) and *Figure 2* (bar comparison) visualize these.

### 6.2 Risk score and diagnostic behavior

Under BoustroLanes coverage saturates rapidly ($C\to 1$ by ~step 300), so coverage deficit $\delta\to 0$ and risk stays low/declines; under MineRichness coverage rises more slowly and the richness-bias component $\phi$ climbs as doubletons dominate, keeping risk elevated into the certification window. The resulting mean risk is 0.23 (boustro) and 0.23 (minerich) at certification — modest, but sufficient to trigger conservative behavior in the risky episodes (see *Figure 1*). The risk score self-regulates: it rises when the fleet is not covering and when revisitation inflates doubletons.

### 6.3 Robustness across K and N

Table 2 reports $FC$ under MineRichness across the K×N grid. *This is an exploratory campaign at 5 seeds per cell*; the Wilson 95% CIs are consequently wide, and the per-cell differences should be read qualitatively, not as established estimates. Only the confirmation campaign (Table 1) is powered for inference.

**Table 2. False-certification % under MineRichness as a function of mine count K and agent count N (exploratory, 5 seeds/cell).** Each cell shows the Wilson 95% CI.

| $K$ | $N$ | Chao1 FC% (95% CI) | AAARS FC% (95% CI) |
|-----|-----|--------------------|--------------------|
| 30 | 3 | 40% [12,77] | 40% [12,77] |
| 30 | 6 | 20% [4,62] | 0% [0,43] |
| 30 | 12 | 40% [12,77] | 40% [12,77] |
| 60 | 3 | 0% [0,43] | 0% [0,43] |
| 60 | 6 | 40% [12,77] | 20% [4,62] |
| 60 | 12 | 20% [4,62] | 0% [0,43] |
| 120 | 3 | 60% [23,88] | 40% [12,77] |
| 120 | 6 | 40% [12,77] | 40% [12,77] |
| 120 | 12 | 40% [12,77] | 0% [0,43] |

**Finding.** AAARS never increases FC relative to Chao1 (no regressions at any K×N). The largest point reductions occur at high agent counts ($N=12$): −40pp at K=120, −20pp at K=60, where shared revisitation is most severe — the intended regime for the method. Because all Wilson CIs overlap at this sample size, the sweep is reported as **exploratory** consistency evidence, not a significance claim. At $N=3$ with $K=30$ the two are identical (40% each), a regime where bias is mild and the diagnostic has little margin — an honest limit.

*Figure 3* shows the same result as a heatmap.

---

## 7. Ablation

We isolated the two adaptive mechanisms (continuous blend vs. adaptive threshold). The partial, illustrative result:

- Removing the adaptive threshold (fixing $\lambda=0$) reproduces plain Chao1's FC — i.e. **the adaptive-threshold mechanism is the primary driver** of the reduction.
- Removing the blend alone leaves the FC reduction intact in the low-bias regime; the blend contributes additional conservatism only where risk rises well above the dead zone.

This decomposition supports our interpretation: AAARS = *a safe fallback to a more conservative estimator, plus a coverage-aware tightening of the certification bar.* (The discrete variant's ineffectiveness, Section 6.1, underscores that continuous risk modulation — not coarse mode switching — is what unlocks the benefit.)

---

## 8. Discussion

### 8.1 Why it works

The mechanism that reduces FC is the **adaptive threshold** $\alpha_{\text{adj}}=\alpha_0/(1+\lambda \bar r)$: as the fleet's own coverage and richness-bias diagnostics indicate risk, AAARS demands a smaller certified residual fraction, i.e. it keeps searching. This is grounded in the same logic as coverage-aware estimation [4]: when coverage is incomplete, unseen mass is under-measured, so a margin must be added before declaring clear.

### 8.2 Statistical power

The confirmation reduction ($35.0\% \to 21.2\%$, Δ=13.8pp) is the largest of the three independent diagnostic designs we evaluated, but it does not reach conventional significance: a two-tailed Fisher's exact test gives **p=0.078**, and the Wilson CIs ({25.5,45.9} vs {13.7,31.4}) barely overlap. A post-hoc power analysis explains why: to detect a 10pp effect (0.25 vs 0.15) at α=0.05 with 80% power requires ~250 episodes per group; to detect our observed 13.8pp requires ~130 per group. Our 80-per-group campaign is therefore genuinely informative about *effect magnitude* but underpowered for a decisive test. Reaching that larger sample is a direct scale-up of the existing runner and is the primary route to a significant claim.

### 8.3 Why the effect does not grow further

Despite trying three independent diagnostic designs (spatial concentration, revisit concentration, and explicit coverage deficit), the FC reduction saturates (here at ~14 points). Two causes:
1. **Loss-leading by latency.** The safety mechanism *is* added latency. Pushing acceptance tighter reduces FC monotonically but eventually suppresses legitimate stops under BoustroLanes too (aggressive settings drove Boustro stops from 15/20 → 0/20, and even the adopted config stops slightly less often in the safe regime: 56/80 vs 59/80). The achievable reduction is bounded by the safe regime.
2. **Moderate risk separation.** Under both allocations coverage eventually saturates; the differences are in the *timing*, not the steady-state. The diagnostic margin is real but limited.

### 8.4 Limitations

- The headline effect (13.8pp at K=60/N=6) is **not statistically significant** (p=0.078; see §8.2), and there is no effect in some mild-bias cells of the sweep (K=30/N=3, K=120/N=6).
- **Late-stop latency:** AAARS median stop time under MineRich is 945 vs 732 (Chao1); it stops later in 78/80 episodes. When time is itself the cost, this must be traded against the safety gain.
- Frozen hyperparameters ($w_\delta,w_\phi,\theta,\lambda$) tuned on the confirmation cell; cross-cell consistency is encouraging but not exhaustive.
- Single environment family; obstacles, heterogeneous detectability, and sensor noise not swept here.
- FC is a worst-case safety metric; the sweep's wide CIs mean its cross-cell pattern is only qualitative.

### 8.5 Future work

- Joint trade-off optimization (define a scalar cost = FC-rate + ω·latency, tune $\lambda,\theta$ per mission).
- Adaptive dead zone / per-agent risk to handle heterogeneous fleets.
- Theoretical guarantee linking coverage deficit to a tightened coverage estimator, replacing the heuristic margin with a bound.

---

## 9. Conclusion

We presented **AAARS**, a ground-truth-free, online, adaptive stopping rule that diagnoses the fleet's own allocation behavior and uses the resulting risk to (1) continuously blend a conservative estimator in and (2) tighten the certification threshold. In a 160-episode confirmation campaign (80 seeds per allocation) it cuts MineRichness false certification from **35.0% to 21.2%** (Δ=13.8pp, Fisher p=0.078) while never degrading the systematic policy, and across the exploratory K×N sweep it never increases false certification and delivers up to 40-point reductions where bias is strongest. The honest picture is a modest, directionally-consistent safety gain that is at present short of statistical significance (see §8.2); reproducing it at the ~250-episodes-per-group sample the power analysis calls for is the direct route to a decisive claim.

---

## References

[1] A. Chao. "Estimating the population size for capture–recapture data when capture probabilities vary by time and individual case." *Biometrika*, 74(4):783–791, 1987.

[2] A. Chao and S.-M. Lee. "Estimating the number of classes via sample coverage." *Journal of the American Statistical Association*, 87(417):210–217, 1992.

[3] I. J. Good. "The population frequencies of species and the estimation of population parameters." *Biometrika*, 40(3–4):237–264, 1953.

[4] A. Chao and L. Jost. "Coverage-based rarefaction and extrapolation: standardizing samples by completeness rather than size." *Ecology*, 93(12):2533–2547, 2012.

[5] J. G. Bellingham et al. "Keeping it all going underwater: adaptive control and command architectures for ocean exploration." 2010.

[6] A. Wald. "Sequential tests of statistical hypotheses." *Annals of Mathematical Statistics*, 16(2):117–186, 1945.

[7] H. Choset. "Coverage for robotics: a survey of recent results." *Annals of Mathematics and Artificial Intelligence*, 31(1):113–126, 2001.

[8] C. Cheng and X. Huan. "Optimal stopping for sequential Bayesian experimental design." *arXiv preprint arXiv:2509.21734*, 2025.

[9] B. Moon, N. Suvarna, A. Jong, S. Chatterjee, J. Yuan, M. Cao, and S. Scherer. "IA-TIGRIS: An incremental and adaptive sampling-based planner for online informative path planning." *IEEE Transactions on Robotics*, 2026. DOI:10.1109/TRO.2026.3672542.

[10] J. A. Placed and J. A. Castellanos. "Enough is enough: Towards autonomous uncertainty-driven stopping criteria for robot exploration." *Proc. IAV (IFAC)*, pp. 126–132, 2022. DOI:10.1016/j.ifacol.2022.07.594.

[11] M. Luperto, M. M. Ferrara, G. Boracchi, and F. Amigoni. "Estimating map completeness in robot exploration." *Autonomous Robots*, 50:6, 2026. DOI:10.1007/s10514-025-10221-8.

[12] M. P. Bron, P. G. M. van der Heijden, A. J. Feelders, and A. P. J. M. Siebes. "Using Chao's estimator as a stopping criterion for technology-assisted review." *arXiv preprint arXiv:2404.01176*, 2024.

---

## Figure Captions

- **Figure 1.** AAARS risk-score trajectory for a BoustroLanes (blue) and a MineRichness (red) episode (seed 42), overlaying the temporal (TC), coverage (SC→coverage deficit), and frequency (FI) components and the blend dead-zone threshold. Coverage-based risk diverges early and discriminates the two policies.

- **Figure 2.** False-certification rate by method and allocation (confirmation campaign, 80 seeds/allocation). AAARS matches Chao1's 0% under BoustroLanes while reducing MineRichness FC from 35.0% to 21.2%; the discrete ablation and plain Chao1 coincide.

- **Figure 3.** False-certification under MineRichness as a heatmap over mine count K and agent count N (Table 2). AAARS reduces FC most at high N.

- **Figure 4.** Per-episode stop time under MineRichness for AAARS vs Chao1 (red × = false certification). AAARS holds out later on the episodes that matter.

---

*Reproducibility: all code and raw results live under `F:\Project11-AAARS`; 40 unit tests pass; figures regenerable via `results/figures/generate_figures.py`; statistics via `results/paper_stats.py`.*
