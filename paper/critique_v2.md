# Critique v2 — Article IEEE BVMT Multi-Agent (post-edits 2026-05-11)

> Lecture fraîche, en tant que reviewer IEEE, de `paper/bvmt_multiagent_ieee_article.tex` après les modifications du 10–11 mai 2026.
> La critique v1 (`critique_synthesis.md`, 6 critiques C1–C6) est en grande partie adressée. Les items ci-dessous sont **résiduels ou nouveaux**.

---

## Résumé exécutif

L'article a fait des progrès substantiels : baselines compétitives chiffrées, IC bootstrap, fuite documentée, reformulation EU AI Act, bibliographie élargie. **Trois familles de problèmes restent** :

| Famille | Items | Sévérité dominante |
|---|---|---|
| **A. Le 77,4 % reste un slogan, pas un résultat vérifié** | A1–A5 | 🔴 Bloquante |
| **B. Le titre et l'identité du papier sont en avance sur les preuves** | B1–B3 | 🔴 Bloquante |
| **C. Cohérence interne, figures, présentation** | C1–C8 | 🟠 Importante / 🟡 Modérée |

Le pattern récurrent : **l'article est devenu honnête (« pending », « shipped »), mais cette honnêteté souligne maintenant un déséquilibre entre le titre/abstract et ce qui est mesuré**. Un relecteur exigeant va lire « pending » comme « non fait ».

---

## A. Le 77,4 % reste un slogan, pas un résultat vérifié

### A1 — Aucun IC, aucun test statistique sur le headline lui-même 🔴
- Les 10 baselines ont chacun un IC 95 % bootstrap (Table 4). **TFT v3 lui-même n'en a pas** : la dernière ligne de Tab. 4 affiche `IC pending Action 1.8`.
- Aucun McNemar entre TFT v3 et un seul concurrent. La p-value du « gap +24–27 pp » est implicite.
- **Ce que dira un reviewer** : « Vous avez calculé des IC pour tout le monde sauf pour votre propre résultat. C'est exactement le seul IC qui compte. »
- **Action** : générer `results/tft_v3_predictions_2025.csv` (script existant `evaluate_tft_v3_per_row.py`) et populer Tab. 4 + McNemar contre logreg / Transformer. Avant publication.

### A2 — Pas de matrice de confusion, pas de précision par classe pour TFT v3 🔴
- Pour toutes les baselines : confusion matrix, precision/recall par classe UP / DOWN, macro-F1.
- Pour TFT v3 : rien. Or 77,4 % sur 49,98 % UP peut cacher un déséquilibre (e.g. 90 % sur DOWN, 65 % sur UP).
- **Action** : ajouter une ligne TFT v3 complète dans Tab. 4 (acc, IC, F1_UP, F1_DOWN, macro-F1) une fois le CSV généré.

### A3 — Pas de breakdown per-stock pour TFT v3 🔴
- §IV.C contient déjà l'avertissement (« 77,4 % can be produced either by a model that is uniformly informative ... or highly accurate on a small high-liquidity subset »).
- Le breakdown existe pour les 10 baselines (champ `per_stock` dans chaque `*_summary.json`) mais pas pour TFT v3.
- **Ce que dira un reviewer** : « Vous me dites vous-même que votre headline peut être porté par 5 tickers ; alors vérifiez-le avant de soumettre. »
- **Action** : `scripts/per_stock_breakdown.py` est déjà livré, à exécuter sur le CSV TFT v3.

### A4 — Le « +24 à +27 pp » n'a pas de mécanisme quantifié 🟠
- §V.A bullet (iv) admet deux explications : avantage architectural réel OU fuite résiduelle non détectée.
- Les ablations TFT internes (Tab. `tab:tft_ablations`) sont toutes en *pending*. Seuls smoke tests faits (5 epochs, `smoke:true`, `best_q50_dir_acc:null` → val_loss inutilisables).
- **Argument de Bayes priors** : sur marché développé avec données plus riches, TFT plafonne à 55–65 %. Sur marché thin, le prior conditionnel est *plus bas*, pas plus haut. +24 pp au-dessus d'un Transformer standard sur les mêmes inputs est extraordinaire. Le rasoir d'Ockham penche vers une fuite résiduelle.
- **Action** : exécuter au moins une des trois ablations (TFT − per-stock norm est la moins coûteuse) avant publication. Si le drop est faible, l'avantage n'est PAS la normalisation et il reste à expliquer.

### A5 — Paradoxe « marché thin = harder + accuracy supérieure » non résolu 🟠
- §I.B argumente que les marchés émergents sont **plus difficiles** (sparse, no derivatives, no benchmarks).
- §V annonce 77,4 % directionnel, 12+ pp au-dessus du SOTA marchés développés.
- L'article ne réconcilie pas explicitement. Le reviewer lira : « Soit votre marché est difficile et votre résultat est suspect, soit votre marché n'est pas si difficile et votre framing est faux. »
- **Action** : ajouter un paragraphe court §V.A reconnaissant la tension et hypothèse de mécanisme (e.g. inefficacité structurelle, frictions qui suppriment l'arbitrage rapide → mais alors pourquoi un LSTM/Transformer standard ne l'exploite pas ?).

---

## B. Titre et identité en avance sur les preuves

### B1 — « Robustness-Aware » dans le titre, zéro expérience adversariale 🔴
- §VII.A déclare explicitement : *« The analyses in this section are architectural [...] We do not run end-to-end poisoning experiments »*.
- §VIII Future Work liste « Empirical poisoning study » comme à faire.
- Le titre vend une propriété **mesurée** ; le contenu livre une propriété **désignée**.
- **Ce que dira un reviewer SaTML/Trustworthy AI** : « Title says ‘Robustness-Aware', show me one adversarial number. » Reject argument.
- **Action** : choisir entre (a) retirer « Robustness-Aware » du titre (e.g. « Auditable Multi-Component Pipeline ») OU (b) exécuter un mini poisoning expérimental (1–2 jours de code, injection ±5 % sur prix d'un ticker, mesurer dérive de prédiction sur ce ticker vs voisins).

### B2 — « Multi-Agent Intelligence » dans le titre, zéro ablation multi-agent 🔴
- Le §V.G (Multi-Agent Ablation Study) que nous venons d'ajouter contient 9 lignes A1–A9, toutes en *pending*.
- Le titre annonce une contribution multi-agent ; le test empirique de cette contribution n'existe pas.
- §III.A.A.A redéfinit prudemment « agent » comme « specialized deterministic module » (bonne décision), mais le mot du titre reste « Multi-Agent Intelligence », qui évoque autonomie / émergence.
- **Action** : exécuter A1, A6, A8 (TFT seul, complet, sans gating) comme minimum viable. Le script wrapper n'existe pas encore — c'est un sprint d'une journée maximum. Sinon, retitrer « Multi-Component Decision Pipeline for [...] ».

### B3 — Contribution 1 de §I.C affirme 77,4 % sans caveat 🟠
- « *A TFT model architecture [...] achieving 77.4% directional accuracy on seven-day forecasts* ».
- Comparer à §V.A bullet (iv) qui est honnête : « consistent with two non-exclusive explanations ... residual leak ».
- L'introduction et la conclusion devraient pré-avertir le lecteur du gap entre headline et vérification.
- **Action** : ajouter une demi-phrase de hedge à la contribution 1 (« achieving a pooled 77.4 % directional accuracy that is large enough to require the internal-ablation suite of Section ... to discriminate between architectural advantage and residual leakage »).

---

## C. Cohérence, figures, présentation

### C1 — `fig:model_comparison` est obsolète 🟠
- Affiche : Random / CNN-LSTM v1 / CNN-LSTM v2 / TFT v1 / TFT v3.
- Tab. 4 contient maintenant 10 baselines + TFT v3.
- La figure devrait montrer le band 50–53 % (10 baselines superposées en boxplot) + TFT v3 à 77,4 %, ou disparaître au profit d'une figure dérivée de Tab. 4.
- **Action** : refondre en boxplot ou supprimer la figure et renvoyer à la table.

### C2 — `fig:training_curves` cite « TFT v1 best: 0.6036 » mais la valeur correcte selon Tab. III est 0,6051 🟡
- Tab. `tab:tft_progression` : `TFT v1 (consolidated) ... 0.6051`.
- `fig:training_curves` (line ~614) : `v1 best: 0.6036`.
- Soit la figure est ancienne (avant consolidation v1), soit la table. Incohérence visible.
- **Action** : harmoniser à 0,6051 dans la figure (modifier la coordonnée `(6,0.6036)` → `(6,0.6051)` + label).

### C3 — §IV.C début continue à comparer TFT v3 à CNN-LSTM uniquement 🟠
- « *TFT v3 achieves 77.4% directional accuracy [...] compared to 53.8% for the best CNN-LSTM hyperparameter configuration* ».
- La table de référence pour la comparaison est maintenant Tab. 4 (logreg 52,88 %, Transformer 52,26 %, etc.). CNN-LSTM n'est qu'un comparateur parmi d'autres.
- **Action** : remplacer « best CNN-LSTM » par « strongest of the ten non-TFT baselines » et pointer Tab. 4.

### C4 — Ordre logique §V.B (Economic) / §V.C (Quantile→Direction) inversé 🟠
- §V.B (Economic Backtest) utilise les prédictions discrètes UP/DOWN.
- §V.C (From Quantile to Direction) **définit** comment la prédiction discrète est obtenue.
- Lecteur arrive au backtest avant de connaître la règle. Solution : interchanger l'ordre des deux sous-sections.

### C5 — §V.B Economic Backtest a deux fragilités numériques 🟠
- **(a)** TFT v3 absent de Tab. `tab:economic_metrics`. Sans la PnL TFT v3, on compare 9 baselines entre elles → le « gagnant » Standard Transformer (Sharpe 4,40) devient le headline économique implicite, ce qui contredit le narratif TFT-best du reste de l'article.
- **(b)** `n_trials_for_dsr = 9` dans `economic_summary.json`. Mais le développement réel a essayé bien plus de variants (4 CNN-LSTM, plusieurs TFT v1–v3, ratios sentiment threshold, etc.). Le deflated Sharpe avec k=9 est **optimiste** : un reviewer López-de-Prado-aware demandera k ≥ 30.
- **Action (a)** : ajouter ligne TFT v3 après l'IC (dépend de Action 1.8).
- **Action (b)** : justifier le choix de k=9 dans la légende OU augmenter à k=15–20 pour absorber le multiple-testing du développement.

### C6 — Mapping Article 17 EU AI Act incorrect 🟠
- §VII.B mappe Article 17 (Quality Management System) au « data quality score 0–100 » du DataAgent.
- Article 17 concerne le **système de management qualité** (procédures, change management, traçabilité documentaire, post-market monitoring), pas un score numérique de qualité de données.
- Le mapping correct serait : Article 10 (data governance) pour le score, Article 17 reste hors-scope.
- **Action** : remplacer « Article 17 — Quality Management » par « Article 10 — Data Governance », ou retirer le bullet.

### C7 — `fig:architecture` ne montre pas l'XAI Agent dans le flux 🟡
- Le diagramme dessine XAI comme un node entrant dans l'orchestrateur par le bas (`xai.east) -- ++(0.9,0) |- (orch.south west)`), pas dans la chaîne Technical→Fundamental→Sentiment→Graph.
- Or §III.A.B liste XAI comme 6e agent.
- L'arrow set est confus. Préciser : XAI consomme les sorties des autres et alimente l'orchestrateur, pas un agent en série.

### C8 — Absence d'un Reproducibility Statement consolidé 🟡
- Info repro éparpillée : Vertex container §V.F, scripts baselines/ §V.A et §V.B, harness seed sweep §IV.A, evaluate per-row §IV.C.
- Format IEEE typique attendu : un paragraphe ou sous-section « Reproducibility » donnant repo URL (anonymisée), commit hash, container digest, seed, hardware, durée totale, coût GCP.
- **Action** : ajouter `\subsection{Reproducibility and Artifact Availability}` avant la Conclusion ou en annexe.

### C9 — Aucune Data Availability Statement 🟡
- BVMT prix : ilboursa.com (scraping) — licence pas claire.
- News : 7 937 articles scrapés — licence absente.
- Financial statements : CMF Tunisie (organisme public) — généralement ouvert.
- IEEE et la plupart des journaux exigent maintenant une « Data Availability Statement ».
- **Action** : ajouter 1 paragraphe Conclusion ou avant : ce qui est public (code, modèles ?), ce qui est restreint (données brutes), comment un relecteur peut reproduire (instructions de scraping ?).

### C10 — Inconsistance numérique mineure : STAR n=192 dans Tab.4 caption (`241 trading days`) 🟡
- §IV.C dit « 43 of 68 tickers, 241 trading days ».
- Per-stock JSON montre STAR `n=192` (probablement listé partiellement en 2025).
- Caption Tab. 4 devrait préciser : « 43 tickers actively trading, with $N$ per ticker ranging from $192$ to $241$ days ».

---

## D. Items mineurs (présentation, style)

- **D1** §V.D « Loss Interpretation Across Versions » : trop court (6 lignes), pourrait être merge dans §V.E ou §IV.B.
- **D2** §V.E « Quantile Regression » : ne justifie pas pourquoi (0,1 ; 0,5 ; 0,9) et pas (0,05 ; 0,5 ; 0,95).
- **D3** Author block : « Submitted to IEEE Conference on Secure and Trustworthy AI » — ce nom exact n'existe pas. Soit IEEE SaTML, soit IEEE BigData TrustAI workshop, soit IEEE TIFS. Préciser ou retirer.
- **D4** §VIII Future Work : 7 items, mélangent extension scientifique (graph temporel, Arabic NLP) et items qui sont des Limitations actuelles (calibration validation, empirical poisoning). Séparer ces deux types.
- **D5** §VII Title « Robustness and Auditability Implications » est OK, mais §VII.C « Agent Trust and Failure Isolation » répète des éléments déjà couverts §III.A.C–D. Réduire au strict minimum.
- **D6** Bibliographie : maintenant 35 entrées. Une référence manquante notable : **Lim et al. 2021 (TFT)** est cité comme `\cite{IEEEhowto:kopka}` — clé bizarre, à renommer en `\cite{ref:tft}` pour homogénéité.

---

## Recommandation de priorisation

**Avant soumission (bloquants)** :
1. **A1+A2+A3** ensemble : un seul script (`evaluate_tft_v3_per_row.py`) débloque IC TFT v3, McNemar vs logreg, breakdown per-stock. ~30 min wall-clock GPU.
2. **B1 ou B2** : trancher entre retirer un mot du titre OU faire l'expérience qui le supporte. C'est une décision éditoriale, à prendre **maintenant**.
3. **A4** : exécuter au moins 1 ablation TFT interne (`TFT − per-stock norm`) pour disqualifier la fuite la plus probable. 3–6 h GPU.

**Avant camera-ready (importants)** :
4. **C3, C4, C5, C6** : edits ciblés (1–2 h chacun).
5. **C1** : refonte `fig:model_comparison` en boxplot 10 baselines + TFT v3.

**Cosmétique (modéré / mineur)** :
6. **C7, C8, C9, C10, D1–D6** : ramassage final.

---

## Critères de DoD mis à jour (par rapport à `critique_synthesis.md`)

| # | Original | Statut au 2026-05-11 |
|---|---|---|
| C1 | Vérification 77,4 % | 🟠 baselines OK, TFT v3 propre IC manquant |
| C2 | Baselines compétitives | ✅ 10 baselines avec IC |
| C3 | Fuite temporelle | ✅ `leakage_audit.md` + normalisation §III.B |
| C4 | Ablation multi-agent | 🟠 §V.G existe en placeholder ; valeurs pending |
| C5 | Claims EU AI Act | ✅ « designed to support » partout |
| C6 | État de l'art | ✅ 35 entrées |
| **B1 nouv.** | Robustness empirique | 🔴 pas une seule mesure adversariale |
| **B2 nouv.** | Multi-agent empirique | 🔴 placeholder seul |
| **A4 nouv.** | Mécanisme du gap | 🔴 ablations pending |

---

*Auteur de la critique : revue indépendante du `bvmt_multiagent_ieee_article.tex` post-edits 10 mai 2026, croisée avec les `results/baselines/*.json` actuellement présents dans le repo.*
