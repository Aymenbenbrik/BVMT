# Plan d'amélioration — Article IEEE BVMT Multi-Agent

> **Pré-requis** : avoir lu `critique_synthesis.md` (synthèse des 6 critiques).
> **Objectif** : transformer l'article actuel en une version *defensible* devant un comité IEEE, en répondant point par point aux six critiques.
> **Date de plan** : 2026-05-09

---

## Vue d'ensemble en 4 phases

```
Phase 0 : AUDIT          (1 semaine)   — Vérifier qu'il n'y a pas de fuite
Phase 1 : BASELINES      (2 semaines)  — Entraîner les modèles compétitifs
Phase 2 : ABLATIONS      (2 semaines)  — Démontrer la valeur du multi-agent
Phase 3 : RÉDACTION      (1 semaine)   — Réécriture, références, claims
                       ─────────────
                       ≈ 6 semaines au total
```

Les phases 1 et 2 peuvent partiellement se chevaucher (semaines 2–3 et 4–5).

---

## Phase 0 — Audit anti-fuite (réponse à C3)

**Objectif** : avant tout nouvel entraînement, garantir qu'aucune information du futur ne fuit dans les features ou la normalisation.

### Action 0.1 — Audit des indicateurs techniques

**Fichiers à inspecter** :
- `agents/technical_agent.py`
- `training/train_tft_v3.py`
- Tout script qui calcule MA, RSI, volatilité

**Vérifications** :
- [ ] MA5/20/50 calculés en **rolling causal** (`pandas.rolling(window).mean()` avec `closed='left'` ou décalage explicite).
- [ ] RSI(14) et `rsi_momentum` (RSI(t) − RSI(t−5)) utilisent uniquement le passé.
- [ ] `volatility_20` = std des rendements **strictement antérieurs** à $t$.
- [ ] `price_to_ma20`, `volume_spike` calculés à partir des MA causales.

**Test recommandé** : pour chaque feature, recalculer à la date $t$ en n'utilisant que les lignes $\leq t-1$. Comparer avec la valeur stockée. Tolérance : 0.

### Action 0.2 — Audit de la normalisation par action

**Fichiers** : `training/train_tft_v3.py`, `agents/technical_agent.py`

**Vérification critique** : les statistiques $(\mu_i, \sigma_i)$ par action doivent être calculées **uniquement** sur la fenêtre de train (2016–2023), puis appliquées telles quelles à 2024 (val) et 2025 (test). Si elles sont recalculées sur la série complète, c'est une fuite.

**Action** : ajouter un commentaire et un assert dans le code, et un paragraphe explicite dans l'article (§III.B).

### Action 0.3 — Audit des news et fondamentaux

- [ ] Le pré-scoring offline du corpus de 7 937 articles produit-il une feature **disponible à $t$** ? Si oui, filtrer par `publication_date < t`.
- [ ] Les ratios financiers utilisés pour le `FundamentalAgent` respectent-ils le **délai de publication** (typiquement N+90 jours pour les comptes annuels) ?

### Livrable Phase 0

**Document** : `paper/leakage_audit.md` listant chaque feature, son origine temporelle, et la confirmation d'absence de fuite. À citer dans §III.B (« Walk-Forward Validation Protocol »).

---

## Phase 1 — Baselines compétitives (réponse à C2)

**Objectif** : entraîner et évaluer 7 baselines + 3 ablations TFT internes, sur exactement le même split walk-forward.

### Action 1.1 — Baseline triviale et naïve

| Modèle | Implémentation | Effort |
|---|---|---|
| **Always-DOWN** | Classe majoritaire (58,1 %) | 1h |
| **Always-UP** | Pour symétrie | 1h |
| **Momentum 5/20** | Si MA5 > MA20 → UP, sinon DOWN | 2h |
| **MA crossover** | Variante classique | 2h |

> ⚠️ **Le baseline correct n'est pas 50 % mais 58,1 %** (always-DOWN). Toutes les comparaisons de l'article doivent être recalculées sur cette base.

### Action 1.2 — Baselines ML classiques

Tous entraînés sur les mêmes 15 features × 60 jours, mais aplaties (le modèle voit un vecteur de 900 features) :

| Modèle | Bibliothèque | Hyperparams à tuner |
|---|---|---|
| Logistic Regression L2 | `sklearn` | C ∈ {0.01, 0.1, 1, 10} |
| Random Forest | `sklearn` | n_estimators=500, max_depth ∈ {5, 10, None} |
| XGBoost technique | `xgboost` | mêmes settings que `train_fundamental.py` |

### Action 1.3 — Baselines deep learning

| Modèle | Architecture cible | Bibliothèque |
|---|---|---|
| LSTM simple | 2 couches LSTM(64) + dense | `pytorch` |
| Transformer standard | 2 blocs encodeur (h=4, d=64) | `pytorch` |

### Action 1.4 — Ablations TFT internes

| Variante | Modification | Cible |
|---|---|---|
| TFT − VSN | Remplacer la VSN par concat brut | Isoler l'apport de la VSN |
| TFT − norm/action | Normalisation globale (un seul $\mu, \sigma$) | Quantifier le gain de la normalisation par action |
| TFT − confidence gating | Désactiver Eq. 3 | Quantifier l'apport du gating |

### Action 1.4b — CNN-LSTM seed sweep statistiquement puissant (réponse à B3)

**Constat** : la table actuelle `tab:cnn_lstm_ablation` rapporte 4 runs avec des hyperparamètres différents (batch_size 64/128, version v1/v2). Sur un test set de quelques milliers de prédictions, l'écart-type d'une accuracy binaire est ~0,5–1,0 pp, donc la fourchette observée [51,7%, 53,8%] (2,1 pp) est **dans le bruit**, pas un plafond statistique.

**Outils livrés** :
- `training/train_cnn_lstm.py` patché pour lire `BVMT_SEED` (numpy + torch + cuDNN deterministic) et `BVMT_SMOKE` (override de la config par défaut).
- `training/cnn_lstm_seed_sweep.py` : harness qui spawn N processus enfants, agrège les `results/results_cnnlstm_v2_*_seed<S>.json`, calcule mean ± IC 95% bootstrap, médiane, Q1/Q3, et test de Wilcoxon vs always-DOWN (58,1%).

**Protocole** :

```bash
# 1. Sanity check (3 seeds smoke, ~5 min sur GPU)
python training/cnn_lstm_seed_sweep.py --n 3 --smoke true

# 2. Sweep complet (20 seeds, ~6-12 h sur GPU T4 / Lightning AI)
python training/cnn_lstm_seed_sweep.py --n 20 --smoke false
```

**Critère de validation du « ceiling »** : la médiane et le 3e quartile (Q3) du test_acc sur 20 seeds restent < 56% (`CEILING_CANDIDATE_PCT` dans le harness). Si Q3 < 56%, le claim « architectural ceiling » est défendable. Sinon, la formulation doit rester « consistent four-run signature ».

**Tests statistiques** :
- Wilcoxon (deux côtés) vs always-DOWN 58,1% → p-value attendue petite si CNN-LSTM ne bat pas la majorité bear.
- Wilcoxon (deux côtés) vs random 50% → indique si CNN-LSTM extrait au moins du signal.

**Livrable** : nouvelle table `tab:cnn_lstm_seed_sweep` dans §V.A :

| Modèle | N seeds | mean test_acc (%) | IC 95% | médiane | Q3 | p (vs always-DOWN) |
|---|---|---|---|---|---|---|
| CNN-LSTM v2 | 20 | … | […, …] | … | … | … |
| TFT v3 | 1 (déterministe) | 77,4 | n/a | — | — | … |

→ Si Q3 < 56%, la prose actuelle (§V.A « consistent four-run signature ») peut redevenir « empirical ceiling ». Sinon, la formulation prudente reste.

**Garde-fou** : le harness journalise le seed dans `run_name` et le JSON, donc une re-exécution est traçable. Le fail d'un seed individuel n'invalide pas le sweep — le summary skip et liste les seeds échoués.

### Action 1.5 — Métriques à calculer pour CHAQUE modèle

Pour répondre à C1 simultanément :

```
- Précision globale + IC 95% (bootstrap n=1000)
- Précision par classe (UP / DOWN)
- F1-score par classe + macro-F1
- ROC-AUC + IC 95%
- Matrice de confusion
- Précision par action (68 valeurs) + écart-type cross-stock
- Test de McNemar vs TFT v3 (p-value)
- Test de Diebold-Mariano sur les pertes log-loss
```

### Action 1.5b — Sensitivity de la règle quantile→direction (réponse à B1)

La règle déployée (`agents/technical_agent.py`, `interpret_quantile_prediction`) est :

$$
\hat{d}(x) = \begin{cases}
\text{UP}   & \text{si } \hat{Q}_{0.5}(x) \geq 0 \\
\text{DOWN} & \text{sinon.}
\end{cases}
$$

Pour chaque échantillon de test 2025, persister `(Q10, Q50, Q90, true_return_7d)`. Calculer ensuite la précision sous quatre règles concurrentes :

| Règle | Définition | Couverture | Précision attendue |
|---|---|---|---|
| **R1** (déployée) | UP si $Q_{50} \geq 0$ | 100% | 77,4 % (référence) |
| **R2** strict | UP si $Q_{10} > 0$, DOWN si $Q_{90} < 0$, sinon abstain | < 100% | + plus haut, - couverture |
| **R3** HOLD band | HOLD si $|Q_{50}| < \tau$ ; $\tau$ tuné sur 2024 | < 100% | + stable sur low-confidence |
| **R4** CDF | $\hat{p}(\text{ret}>0)$ via inverse-CDF linéaire ; seuil 0,5 | 100% | $\approx$ R1 si quantiles monotones |

Reporter dans une nouvelle table `tab:rule_sensitivity` (colonnes : Rule, Coverage, Acc, F1\_UP, F1\_DOWN). Si R1 et R2/R3 divergent de plus de 5 pp sur la couverture commune, l'article doit identifier laquelle est utilisée pour le headline 77,4 %.

### Action 1.6 — Métriques économiques (réponse à C1)

Sur le test set 2025, simuler une stratégie *long-short* basée sur les prédictions :

```
- Rendement cumulé annualisé
- Ratio de Sharpe (rf = taux BCT)
- Ratio de Sortino
- Drawdown maximum
- Profit Factor
- Hit ratio
- Avec/sans coûts de transaction (50 bps round-trip BVMT)
- Capacité de remplissage (volume disponible vs taille position)
- Deflated Sharpe ratio (Bailey & López de Prado, 2014)
```

### Livrable Phase 1

**Tableau dans l'article** (remplace/étend Tab. III actuel) :

| Modèle | Acc | Acc IC 95% | UP-acc | DOWN-acc | Sharpe | Max DD | p-value vs TFT |
|---|---|---|---|---|---|---|---|

**Code** : `training/baselines/` (un fichier par baseline).

### Action 1.8 — Per-stock breakdown TFT v3 (réponse à D3)

**Constat** : le 77,4 % est un agrégat **pooled** sur 10 314 prédictions × 43 tickers. Aucun fichier de prédictions per-row n'existe pour le v3 quantile (le script `training/evaluate_tft.py` cible le v1 binaire). Donc on ne peut pas dire si le 77,4 % est porté par 8 grands tickers liquides ou réparti uniformément.

**Outils livrés** :
- `training/evaluate_tft_v3_per_row.py` : évaluation TFT v3 sur le test 2025, persiste `(ticker, ticker_id, isin_code, date, time_idx, q10, q50, q90, true_return_7d, true_direction, pred_direction_R1)` dans `results/tft_v3_predictions_2025.csv`.
- `scripts/per_stock_breakdown.py` : agrège la CSV → table per-ticker avec accuracy, 95% bootstrap CI, lift vs always-DOWN ; produit le summary JSON + LaTeX rows pour `tab:per_stock_breakdown`.

**Protocole** :

```bash
# 1. Re-générer les prédictions per-row (requiert le checkpoint v3 + GPU recommandé)
python training/evaluate_tft_v3_per_row.py \
    --ckpt models/tft_quantile_v2_fixed_*.ckpt \
    --out  results/tft_v3_predictions_2025.csv

# 2. Agréger par ticker
python scripts/per_stock_breakdown.py \
    --in   results/tft_v3_predictions_2025.csv \
    --out  reports/per_stock/breakdown.json \
    --csv  reports/per_stock/per_ticker_table.csv \
    --rule R1
```

**Critères et lectures à faire**

| Indicateur | Cible saine | Inquiétant |
|---|---|---|
| `mean per-ticker acc` | proche du 77,4 % pooled | ≥ 5 pp en dessous → l'agrégat est tiré par les volumes |
| `median per-ticker acc` | ≥ 70 % | < 60 % → agrégat porté par une minorité |
| `IQR` (Q3 − Q1) | ≤ 15 pp | > 25 pp → forte hétérogénéité |
| `fraction < 50 %` | ≤ 10 % | > 30 % → de nombreux tickers en dessous du hasard |
| `fraction < 58,1 %` (always-DOWN) | ≤ 25 % | > 50 % → modèle fait pire que la baseline naïve sur la majorité |
| Lift moyen vs always-DOWN | + 19 pp | < + 5 pp → l'avantage TFT s'évapore une fois pondéré par stock |

**Livrable article** : nouvelle table `tab:per_stock_breakdown` dans §V, avec **Top 5 / Bottom 5** :

| Catégorie | Ticker | n | accuracy (%) | IC 95% | lift vs always-DOWN |
|---|---|---|---|---|---|
| Best 1 | … | … | … | […, …] | … |
| … | … | … | … | … | … |
| Worst 5 | … | … | … | […, …] | … |

**Bonus utile pour la critique** : le même CSV alimente Action 1.5b (sensitivity quantile→direction R1/R2/R3/R4) sans re-exécuter le modèle.

**Garde-fou éditorial** : si la médiane per-ticker est < 60 %, l'article doit reformuler le headline « 77,4 % directional accuracy » en « 77,4 % pooled directional accuracy, with median per-ticker accuracy of XX,X % », et faire suivre d'une discussion sur la concentration de la performance.

### Action 1.7 — Validation expert du modèle de sentiment (réponse à B5)

**Constat** : le `SentimentAgent` utilise `bardsai/finance-sentiment-fr-base` comme boîte noire pré-entraînée. La performance reportée par Rguibi et al. 2023 (5–15 pp d'amélioration) est mesurée sur **leur** benchmark, pas sur ilboursa.com. ~15% des articles sont arabes, scorés par un modèle français → garbage in / garbage out probable.

**Outils livrés** :
- `scripts/sample_news_for_labelling.py` : échantillonnage stratifié par (année, langue, label-modèle), 200 articles, export CSV avec colonnes `expert_label` / `expert_confidence` / `expert_notes` à remplir.
- `scripts/evaluate_sentiment_labels.py` : calcul accuracy, F1 par classe, macro-F1, Cohen κ, breakdown FR/AR, row LaTeX paste-ready pour `tab:sentiment_validation`.

**Protocole** :

```bash
# 1. Échantillonner 200 articles (DB doit être accessible)
python scripts/sample_news_for_labelling.py --n 200 \
    --out reports/sentiment_validation/sample_to_label.csv

# 2. Le labeller (étape humaine, ~4-8 h)
#    - Ouvrir le CSV dans LibreOffice/Excel/Sheets
#    - Remplir expert_label ∈ {positive, neutral, negative}
#    - expert_confidence ∈ {high, medium, low}
#    - expert_notes pour les cas ambigus (sarcasme, signaux mixtes, arabe)
#    - Convention : labelliser sur l'impact financier perçu pour le ticker,
#      pas sur le ton émotionnel pur de l'article.

# 3. Évaluer
python scripts/evaluate_sentiment_labels.py \
    --in reports/sentiment_validation/sample_to_label.csv \
    --out reports/sentiment_validation/eval_report.json
```

**Critères de succès** :

| Métrique | Cible (FR) | Acceptable | Échec |
|---|---|---|---|
| Accuracy | ≥ 70% | 60–70% | < 60% |
| Macro-F1 | ≥ 0,65 | 0,55–0,65 | < 0,55 |
| Cohen κ | ≥ 0,50 (modéré-substantial Landis-Koch) | 0,30–0,50 | < 0,30 |

**Si échec** : pivoter vers (i) fine-tuning du modèle bardsai sur 200–500 labels supplémentaires, ou (ii) remplacer par un modèle plus récent (e.g. `cmarkea/distilcamembert-base-sentiment` finetuné finance), ou (iii) downgrader la weight base du SentimentAgent en dessous de 0,20.

**Sur l'arabe** : on s'attend à κ ≈ 0 sur le subset AR (modèle français, articles arabes). Cela motive l'intégration AraBERT déjà listée en Future Work — mais ne ressuscitera pas le SentimentAgent sur AR pour le papier actuel.

**Livrable article** : nouvelle table `tab:sentiment_validation` dans §V (ou en annexe si l'espace manque) :

| Modèle | n labels | n FR/AR | Accuracy (%) | Macro-F1 (%) | Cohen κ |
|---|---|---|---|---|---|
| bardsai/finance-sentiment-fr-base | … | …/… | … | … | … |

→ Si κ ≥ 0,50, le claim de qualité du sentiment devient defendable et l'article peut quantifier l'incertitude. Si κ < 0,30, le SentimentAgent doit être remis en question (poids réduit ou agent désactivé pendant la Phase 2 d'ablation).

---

## Phase 2 — Ablations multi-agent (réponse à C4)

**Objectif** : démontrer empiriquement que chaque agent contribue à la performance.

### Action 2.1 — Configurations à évaluer

```
A1 : TFT seul
A2 : TFT + Fundamental
A3 : TFT + Sentiment
A4 : TFT + Graph
A5 : TFT + Fundamental + Sentiment
A6 : Système complet (7 agents)
A7 : Complet − DataAgent confidence (forcer à 1)
A8 : Complet − Confidence Gating
A9 : Complet − Weight Redistribution
```

### Action 2.2 — Performance par segment

Décomposer les résultats par type d'entreprise :

| Type | N stocks | TFT seul | Multi-agent | Δ |
|---|---|---|---|---|
| Banques | 12 | … | … | … |
| Assurances | 5 | … | … | … |
| Non-banques | 51 | … | … | … |

→ **Justifie ou invalide** la matrice de poids adaptatifs (Tab. 2 actuel).

### Action 2.3 — Mesurer le confidence gating

Sur l'ensemble du test 2025 :
- [ ] Combien de prédictions ont un poids `Technical` modifié ? (% de gating actif)
- [ ] Distribution des confidences par agent (histogramme).
- [ ] Précision conditionnelle : précision **quand** le gating est actif vs **quand** il ne l'est pas.

### Action 2.4 — Apprentissage des poids du multi-agent (réponse à B2)

**Constat** : la version actuelle utilise un vecteur de poids fixe $w^{(0)} = (0.34, 0.26, 0.20, 0.20)$ et un ensemble de règles d'ajustement heuristiques (Table~\ref{tab:weight_adjustments} de l'article) qui n'ont pas été tunées par CV. Une version antérieure du papier prétendait à une matrice par type d'entreprise (banques / assurances / non-banques) qui ne correspondait à aucun chemin de code et a été retirée.

**Procédure d'apprentissage** :

1. **Données** : utiliser le set de validation 2024 (10 228 lignes), exclu du training et du test 2025.
2. **Espace de recherche** :
   - Vecteur de base $w^{(0)}_{c} \in \Delta^4$ par type d'entreprise $c \in \{\text{bank, insurance, non-bank, leasing}\}$ ($4 \times 4 = 16$ paramètres dans le simplexe).
   - Optionnel : 4 paramètres $\alpha$ par règle d'ajustement (donor fraction de la Table~\ref{tab:weight_adjustments}) tunés en commun.
3. **Méthode** :
   - **Grid search** grossier ($\Delta = 0{,}05$ sur chaque dimension du simplexe) pour calibrer la maille.
   - **Bayesian optimization** (TPE, Optuna) pour raffinement, 200 itérations.
   - **Critère** : précision directionnelle macro-F1 sur 2024, plutôt que accuracy brute, pour pondérer la classe minoritaire (UP).
4. **Validation hors-échantillon** : appliquer les poids sélectionnés au test 2025, comparer à la baseline $w^{(0)}$ déployée actuellement. Reporter :
   - Précision avant/après.
   - p-value (McNemar) sur le différentiel.
   - Test de robustesse : permuter les labels de validation 100 fois, vérifier que le gain n'est pas dû à du sur-ajustement (gain réel doit être > 95-percentile des gains permutés).
5. **Garde-fou** : si l'amélioration sur 2025 est < 0,5 pp, **ne pas** modifier les poids déployés ; reporter le résultat négatif honnêtement (le hand-tuning informel a produit des poids déjà proches de l'optimum).

**Livrable** :
- Code dans `training/learn_weights.py`.
- Nouvelle table `tab:learned_weights` dans §V de l'article :

| Type entreprise | $w_{\mathrm{tech}}$ | $w_{\mathrm{fund}}$ | $w_{\mathrm{sent}}$ | $w_{\mathrm{graph}}$ | Δ accuracy 2025 |
|---|---|---|---|---|---|

- Si gain significatif → la table remplace `tab:weight_adjustments`. Sinon, la table reste en annexe avec discussion.

### Livrable Phase 2

Section nouvelle §V.D « Multi-Agent Ablation Study » avec :
- Tableau A1–A9.
- Figure : barplot des accuracy par configuration.
- Tableau performance par segment.
- Tableau des poids appris (`tab:learned_weights`).
- Texte interprétatif intégrant B2 (poids appris vs heuristiques).

---

## Phase 3 — Rédaction et réponses C5–C6

### Action 3.1 — Reformulation des claims réglementaires (C5)

**Recherche/remplacement dans `bvmt_multiagent_ieee_article.tex`** :

| Avant | Après |
|---|---|
| « satisfies EU AI Act transparency requirements » (abstract) | « is **designed to support** EU AI Act transparency and auditability requirements » |
| « satisfy Articles 13 and 14 » (introduction) | « **align with** Articles 13 and 14 » |
| « satisfies the three properties above » (§II.D, SHAP) | OK (factuel sur SHAP) |
| « These audit trails satisfy regulatory requirements » (§VI.B) | « These audit trails **provide the technical building blocks** for regulatory requirements; full compliance also depends on governance, risk management, post-market monitoring, and independent validation, which are out of scope here. » |

Ajouter un paragraphe **explicite** §VI.B :
> *« We emphasize that EU AI Act conformity is not solely a function of explainability technology. It additionally requires governance frameworks, risk management procedures, technical documentation per Annex IV, post-market monitoring (Art. 72), and where applicable conformity assessment by a notified body. The present work delivers the technical layer; integration into a compliant operational pipeline is left to deploying institutions. »*

### Action 3.2 — Renforcement bibliographique (C6)

Ajouter au minimum **15 références récentes (2022–2025)**, regroupées par thème :

**Financial transformers** :
- Zhou et al., *Informer*, AAAI 2021.
- Wu et al., *Autoformer*, NeurIPS 2021.
- Nie et al., *PatchTST*, ICLR 2023.
- Zhou et al., *FEDformer*, ICML 2022.

**Marchés émergents & illiquidité** :
- Bekaert & Harvey, *Emerging Equity Markets*, 1995/2017 update.
- Chen et al., *Forecasting illiquid stocks*, 2023.

**GNN pour finance** :
- Hsu et al., *FinGAT*, 2021.
- Sawhney et al., *MAN-SF*, EMNLP 2020.
- Cheng & Li, *Modeling momentum spillovers via GNN*, 2024.

**Adversarial / poisoning en finance** :
- Goldblum et al., *Adversarial attacks on financial DNNs*, 2021.
- Chen et al., *Targeted backdoor attacks on time series*, 2023.

**Calibration & backtesting** :
- Kuleshov, Fenner & Ermon, *Accurate uncertainties for deep learning via calibrated regression*, ICML 2018.
- Bailey & López de Prado, *The Deflated Sharpe Ratio*, 2014.
- Harvey, Liu & Zhu, *... and the Cross-Section of Expected Returns*, RFS 2016 (multiple testing).

**NLP arabe financier** :
- Antoun et al., *AraBERT*, 2020.
- Khondaker et al., *Arabic NLP for financial text*, 2024.

### Action 3.3 — Réécriture de l'abstract et de la conclusion

Une fois Phases 0–2 terminées :

**Abstract** : remplacer « 25.5 % improvement over the random baseline » par une formulation économique : « Sharpe ratio X, deflated Sharpe Y, beating the always-DOWN baseline by Z accuracy points ».

**Conclusion** : ajouter dans **Limitations** :
- Validation empirique de la calibration des quantiles encore à mener (déjà mentionné).
- Pas de backtest avec coûts de marché réels pour 2026 (out-of-sample post-publication).
- Performance hétérogène entre actions liquides et très peu liquides.

---

## Calendrier proposé

| Semaine | Phase | Livrables |
|---|---|---|
| 1 | Phase 0 | `leakage_audit.md` validé |
| 2 | Phase 1 (start) | Baselines triviales + ML classiques |
| 3 | Phase 1 (end) | LSTM, Transformer, ablations TFT |
| 4 | Phase 2 (start) | Configs A1–A5 |
| 5 | Phase 2 (end) | A6–A9 + analyse par segment |
| 6 | Phase 3 | Réécriture, refs, soumission |

---

## Critères de réussite (definition of done)

L'article sera prêt pour soumission IEEE quand **les six conditions** sont réunies :

- ✅ **C1** : chaque résultat de précision est accompagné d'un IC 95 %, d'un test statistique versus le meilleur compétiteur, et d'au moins une métrique économique.
- ✅ **C2** : tableau comparatif incluant *au minimum* always-DOWN, XGBoost technique, LSTM simple, Transformer standard, et trois ablations TFT internes.
- ✅ **C3** : audit de fuite documenté dans `paper/leakage_audit.md` et résumé dans §III.
- ✅ **C4** : section ablation multi-agent avec performances pour A1–A9 + tableau par segment d'entreprise.
- ✅ **C5** : aucune occurrence de « satisfies EU AI Act » ; remplacée par « designed to support » + paragraphe sur les exigences hors-scope.
- ✅ **C6** : bibliographie ≥ 30 entrées avec au moins 15 publications 2022–2025.

---

## Risques et plans B

| Risque | Plan B |
|---|---|
| L'audit Phase 0 révèle une fuite réelle → 77,4 % retombe à ~55 % | Repositionner l'article comme méthode + leçons d'ingénierie sur marché thin, et viser un workshop/journal applicatif plutôt qu'IEEE conf flagship |
| XGBoost technique atteint ≥ TFT | Pivoter le narratif : *« le TFT n'est pas systématiquement supérieur, mais il fournit la calibration probabiliste et l'interprétabilité que XGBoost n'a pas »* |
| Pas le temps pour Phase 2 complète | Présenter A1, A6 et A8 (TFT seul, complet, sans gating) comme minimum viable, et déclarer le reste « future work » |

---

*Auteur du plan : revue automatique des 6 captures de `paper/A faire/`, croisée avec lecture de `paper/bvmt_multiagent_ieee_article.tex`.*
