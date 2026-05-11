# Faits empiriques pour le repositionnement de l'article (2026-05-11)

> Document de référence : valeurs **reproductibles** mesurées sur les CSV de prédiction `results/tft_ablations/vertex/*.csv` (runs full sur Vertex AI, `smoke:false`, 20–37 epochs, checkpoints 5 MB).
> Toute édition future de l'article qui parle de TFT v3 doit utiliser ces nombres et **pas** le 77,4 % historique.

---

## 1. Headline reproductible TFT v3

Test slice Vertex : N=6 782 lignes, 43 tickers, 2025-01-07 → 2025-12-10, **UP ratio = 54,51 %** (skewed UP, contrairement au 49,98 % du test set N=10 314 du papier).

| Variante | Acc | 95 % CI bootstrap | Macro-F1 | F1_UP | F1_DOWN |
|---|---|---|---|---|---|
| **TFT v3 baseline** (= « v3 reference »)              | **50,47 %** | [49.28, 51.65] | 0,504 | 0,488 | 0,521 |
| TFT − VSN (uniform variable weights)                   | 51,77 % | [50.56, 52.99] | 0,502 | 0,591 | 0,412 |
| TFT − per-stock norm (TorchNormalizer global)          | **53,79 %** | [52.55, 54.95] | 0,494 | 0,643 | 0,344 |

Sur le **même slice** (N=6 782, UP 54,51 %) :
- always-UP scorerait 54,51 % par construction.
- always-DOWN scorerait 45,49 %.

**Conclusion immédiate** : aucune des trois variantes TFT v3 ne bat always-UP sur sa propre slice de test.

---

## 2. McNemar par paires

| Paire | $\chi^2$ | p | Interprétation |
|---|---|---|---|
| baseline vs global_norm  | 17,55 | 2,8×10⁻⁵ | TFT − per-stock norm **strictement meilleur** que baseline |
| baseline vs no_vsn       |  5,23 | 0,022    | TFT − VSN **meilleur** que baseline ($p < 0{,}05$) |
| global_norm vs no_vsn    |  8,31 | 0,004    | TFT − per-stock norm **meilleur** que TFT − VSN |
| always-UP vs baseline    | 19,00 | 1,3×10⁻⁵ | always-UP **strictement meilleur** que TFT v3 baseline |
| always-UP vs global_norm |  1,36 | 0,243    | always-UP ≈ TFT − per-stock norm (indistinct) |
| always-UP vs no_vsn      | 13,82 | 2,0×10⁻⁴ | always-UP **strictement meilleur** que TFT − VSN |

---

## 3. Conséquences pour l'article

### 3.1 Hypothèses falsifiables de §V.F : **toutes falsifiées en sens inverse**

L'article (§V.F après nos edits du 10 mai) écrivait :

> - If TFT − VSN drops to within ±2 pp of the standard Transformer baseline (52,26 %), VSN gating accounts for most of the gap.
> - If TFT − per-stock norm drops by ≥10 pp, per-stock target standardisation accounts for a substantial slice of the gap.

Mesures réelles :
- TFT − VSN ne **drop pas** : il **monte** à 51,77 % (+1,30 pp). VSN gating ne contribue **rien de positif** ; il dégrade légèrement.
- TFT − per-stock norm ne **drop pas** : il **monte** à 53,79 % (+3,32 pp, McNemar p=2,8×10⁻⁵). Per-stock target standardisation **dégrade** l'accuracy.

Les composants présentés comme contributions architecturales clés (§I.C contrib 1 : « per-stock target normalization » + « confidence-gated feature selection ») sont **falsifiés comme contributions positives sur ce dataset**.

### 3.2 Le 77,4 % du papier n'est pas reproductible

- Aucun checkpoint `models/tft_quantile_*.ckpt` n'existe dans le repo public.
- Le retrain Vertex complet (mêmes hparams, mêmes data path, GroupNormalizer per ticker, pinball Q10/Q50/Q90) du « baseline » converge à 50,47 % directionnel.
- Sans le checkpoint original, on ne peut pas distinguer entre :
  - **(H1)** Fuite temporelle dans le pipeline pré-Vertex (les rows 2025 étaient présentes en train).
  - **(H2)** Test set différent (e.g. mesuré sur validation 2024 par erreur, ou sur un subset).
  - **(H3)** Règle de décision différente non documentée.

**Action article** : reconnaître honnêtement la non-reproductibilité du 77,4 % et la **traiter comme leçon de reproductibilité**, pas comme résultat principal.

### 3.3 Le narratif « TFT bat les baselines de +27 pp » s'effondre

Comparaison correcte sur le test set du papier N=10 314 (UP 49,98 %) — baselines mesurés :
- logistic regression L2 : 52,88 %
- standard Transformer : 52,26 %
- simple LSTM : 52,45 %

Comparaison TFT v3 reproductible sur slice Vertex N=6 782 (UP 54,51 %) :
- TFT v3 baseline : 50,47 %
- TFT − per-stock norm : 53,79 %

**Apples-to-apples impossible sans re-évaluation des baselines sur le slice Vertex**, mais l'ordre de grandeur est clair : TFT v3 reproductible est dans le même band 50–54 % que les baselines classiques. **Le titre, l'abstract et la contrib 1 ne sont plus défendables**.

---

## 4. Plan de repositionnement (proposé, en attente de validation)

### 4.1 Titre

Trois options, du plus conservateur au plus disruptif :

- **(O1)** « An Auditable Multi-Component Forecasting Pipeline for Thin Emerging Markets: A Reproducibility Study »
- **(O2)** « When Headline Numbers Don't Reproduce: Lessons from a Vertex-AI Retrain of a Multi-Component Pipeline on the Tunisian Stock Exchange »
- **(O3)** « Reproducibility-First Multi-Component Forecasting for Illiquid Equity Markets: Baselines, Ablations, and a Documented Discrepancy »

### 4.2 Abstract

Squelette : annoncer (i) le contexte BVMT thin market, (ii) le pipeline 7-agents, (iii) **le résultat reproductible 50,47 %** sur Vertex, (iv) la documentation honnête de l'écart vs le 77,4 % historique, (v) les ablations qui *falsifient* les contributions architecturales annoncées, (vi) le repositionnement sur auditabilité / EU AI Act / leçon de reproductibilité.

### 4.3 Contribution 1 (§I.C)

**Avant** : « A TFT model architecture combining per-stock target normalization, per-encoder-window feature scaling, confidence-gated feature selection, and multi-horizon quantile regression, achieving 77.4% directional accuracy. »

**Après** : « A reproducible Vertex-AI training pipeline for TFT-on-thin-market with full ablation suite. The reference (per-stock GroupNormalizer + VSN + pinball) reaches **50.47 % directional accuracy** (95 % CI [49.28, 51.65]) on the 2025 test slice (N=6 782, UP ratio 54.51 %), significantly below the always-UP baseline on the same slice (54.51 %, McNemar p = 1.3×10⁻⁵). Two architectural ablations *improve* over the reference: removing the per-stock target normalizer raises accuracy to 53.79 % (p = 2.8×10⁻⁵), and disabling the Variable Selection Network raises it to 51.77 % (p = 0.022). The per-stock-normalization and VSN-gating components originally hypothesized to contribute positively are **falsified as positive contributors on this dataset**. »

### 4.4 §IV.C, §V.A, §V.F : réécriture complète

À faire dans un second temps, après validation du titre et de l'abstract.

### 4.5 §V.G Multi-Agent Ablation

Le placeholder « pending » de §V.G ajouté le 10 mai n'a plus le même cadre : la « value-add » se mesure désormais par rapport à un baseline TFT à 50,47 %, pas 77,4 %. Reste à exécuter pour mesurer si l'orchestration multi-agent rajoute quelque chose au-dessus de 50,47 %. Section à laisser en placeholder pour le moment.

### 4.6 §VI Robustness

Survit : la défense par per-stock normalization est *architecturale* et reste défendable même si elle ne produit pas le gain d'accuracy attendu. À reformuler en : « per-stock norm reduces poisoning attack surface; on this dataset it also costs some accuracy, suggesting a robustness-accuracy trade-off rather than a free lunch ».

### 4.7 §VII Conclusion + §IX Limitations

Réécrire : la limitation centrale n'est plus « per-stock breakdown pending » mais « TFT-on-thin-market doesn't beat always-UP on its own test slice ; the reported 77,4 % is not reproducible from released artefacts and we document it as a reproducibility lesson ».

---

## 5. Question ouverte avant édit

Le repositionnement transforme un papier « SOTA accuracy on thin markets » en un papier « reproducibility lesson ». Choix éditorial :

- **(a)** Garder le 77,4 % comme « initial internal estimate » dans une box de transparence, mais centrer toute la table de résultats sur 50,47 %.
- **(b)** Retirer entièrement le 77,4 % de l'article. Honnête mais perd le contexte historique.
- **(c)** Garder le 77,4 % mais le déplacer en annexe avec une note disant qu'il n'est pas reproductible et appeler à de futurs benchmarks.

Mon avis : **(a)** est le plus défendable pour un reviewer IEEE Trustworthy AI — la transparence est exactement la valeur qu'on défend. Mais c'est la décision de l'auteur.

---

*Source : `results/tft_ablations/vertex/{baseline,global_norm,no_vsn}_test_predictions.csv` + `*_summary.json` (runs Vertex AI du 2026-05-10, 71–161 min GPU chacun).*
