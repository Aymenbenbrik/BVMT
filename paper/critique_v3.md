# Critique v3 — Article IEEE BVMT post-repositionnement (2026-05-11)

> Lecture fraîche de `paper/bvmt_multiagent_ieee_article.tex` après le pivot vers le narratif « Reproducibility-First » du 11 mai (titre O3, 77,4 % en transparency box, TFT v3 → 50,47 %, Tab. 4 réorganisée, biblio à 50 entrées).
> Les critiques v1 (`critique_synthesis.md` C1–C6) et v2 (`critique_v2.md` A1–A5, B1–B3, C1–C10) sont en grande partie résolues. Cette v3 liste les **résidus** et les **tensions nouvelles** introduites par le pivot.

---

## Résumé exécutif

Le repositionnement **fonctionne** : le narratif est cohérent, le titre matche le contenu, les ablations sont chiffrées, la biblio reproductibilité ancre le pitch. **Mais le pivot a fragilisé trois zones** :

| Zone | Statut post-pivot | Sévérité |
|---|---|---|
| **§V.G Multi-Agent Ablation** | Encore plus vide qu'avant. Si TFT seul = 50,47 %, l'attente sur A6 est explicite ; ne rien donner devient bloquant. | 🔴 |
| **Slice mismatch baselines $N=10{,}314$ vs TFT $N=6{,}782$** | Plus visible qu'avant. Le lecteur ne peut pas comparer directement. | 🟠 |
| **Reading 2 « slice-bias explanation » de §V.F** | Suggère que les « improvements » des ablations sont des artefacts de biais UP, pas des découvertes architecturales. Honnête mais inconfortable. | 🟠 |

Trois pré-existants non résolus :
- **§IV.A CNN-LSTM** section autonome qui n'a plus sa place (CNN-LSTM est juste une baseline parmi d'autres).
- **`fig:training_curves`** montre encore la courbe non-reproductible v3 (val_loss 0,5161).
- **§VII.B Audit trail** est over-engineered pour un modèle 50 % — la valeur d'un audit trail sur un classifier indistinguable d'always-UP est questionnable.

---

## A. Tensions nouvelles introduites par le repositionnement

### A1 — §V.G Multi-Agent Ablation est maintenant un trou narratif béant 🔴

Avant le pivot, A1 référence = 77,4 % et la question implicite était : *« est-ce que les 6 autres agents montent encore ce nombre ? »*. Le manque de mesure était embêtant mais pas central.

Après le pivot, A1 référence = 50,47 % et la question devient : *« est-ce que la pipeline 7-agents fait au moins mieux que la TFT seule à 50,47 % ? »* — et le titre du papier promet une réponse (« Multi-Component Forecasting »).

La table `tab:multiagent_ablation` reste entièrement *pending*. Or :
- Le CSV v3 (`baseline_test_predictions.csv`) existe.
- Les agents (`agents/fundamental_agent.py`, `sentiment_agent.py`, `graph_agent.py`, `orchestrator.py`) existent.
- L'évaluation est *post-hoc CPU-only* (article §V.G : « approximately 20--40 minutes of CPU-only inference »).
- Donc ne pas livrer ces chiffres est un choix éditorial, pas une contrainte technique.

**Ce que dira un reviewer** : « Vous avez retitré le papier en mettant ‘Multi-Component’ en avant et la seule table qui mesurerait cette contribution est entièrement vide. Soit vous la remplissez, soit vous retirez ‘Multi-Component’ du titre. »

**Action** : exécuter le sweep A1–A9 sur le CSV Vertex. C'est ~1 journée de scripting (créer le wrapper Python qui charge le CSV TFT, calcule les sorties des autres agents, agrège via l'orchestrator). Sans ça, la critique B2 de v2 (titre vs preuve) revient sous une forme aggravée.

### A2 — Mismatch slice baselines (N=10 314, UP 49,98 %) vs TFT (N=6 782, UP 54,51 %) 🟠

Le pivot a clarifié qu'on a **deux grilles d'évaluation différentes** :
- *Baseline grid* : produits par `training/baselines/*.py`, $N=10{,}314$, UP $49{,}98\%$.
- *Vertex grid* : produits par `training/tft_ablations/runner.py` sur Vertex, $N=6{,}782$, UP $54{,}51\%$.

La différence vient probablement de la politique de *boundary rows* du `TimeSeriesDataSet` de pytorch_forecasting (encoder warm-up + horizon = $30+7=37$ rows perdus per ticker × 43 tickers ≈ $1{,}591$, soit $10{,}314 - 1{,}591 \approx 8{,}723$ ; mais on observe $6{,}782$, donc il y a $\sim 1{,}900$ rows de plus perdues). Possiblement le `min_encoder_length=15` + des trous calendaires.

**Conséquence** : Tab. 4 a une légende qui dit « TFT v3 reference is reported separately... the two slices are not row-identical » mais n'explique pas la cause technique. Un reviewer va demander : *« Why aren't the baselines re-evaluated on the Vertex slice? »*

**Deux solutions** :
- **(a)** Réévaluer les 10 baselines sur le slice Vertex N=6 782 → harmonise le tableau. Effort : 1–2 h (les baselines sont déterministes, juste filtrer les rows).
- **(b)** Documenter la cause technique en caption Tab. 4 (« the Vertex grid drops an additional ~X rows per ticker due to pytorch_forecasting's strict encoder-window policy »).

**Action recommandée** : (a). C'est cheap et tue la critique.

### A3 — Reading 2 (« slice-bias explanation ») de §V.F est un poignard pour les ablations 🟠

§V.F observation Reading 2 (que j'ai écrite) dit :

> The Vertex slice is 54.51 % UP, so any model with even a mild UP bias gets a tail-wind on this slice; the variants TFT − VSN and TFT − per-stock norm happen to be more UP-biased than the reference (TFT − per-stock norm predicts UP on 4569/6782 = 67.4% of rows, well above the 54.5% slice prior, and that bias is what carries its 53.79%).

C'est honnête. Mais ça implique que les « improvements » +1,30 / +3,32 pp ne sont *pas* des découvertes architecturales — c'est juste un biais UP qui colle à un slice UP-skewed.

**Ce que dira un reviewer méchant** : *« Vous présentez ‘TFT − per-stock norm beats TFT v3 by +3.3 pp’ comme un résultat, mais votre propre Reading 2 explique que c'est juste un biais UP sur slice UP. Pourquoi reporter le chiffre du tout ? »*

**Solutions possibles** :
- **(a)** Calculer le « UP-bias-adjusted accuracy » : pour chaque variante, mesurer la précision **conditionnelle au true_direction**. Si TFT − per-stock norm a F1_UP = 0,643 mais F1_DOWN = 0,344 (qui sont déjà dans tab:baselines), on voit que la précision sur les DOWNs vrais est sous le hasard (50 %). Donc le +3,32 pp pooled est un artefact UP-bias.
- **(b)** Reporter macro-F1 comme métrique primaire (équilibrée par classe), pas accuracy pooled. Or les macro-F1 sont : TFT v3 = 0,504, TFT − VSN = 0,502, TFT − per-stock norm = **0,494**. Avec macro-F1, le baseline TFT v3 est en fait *le meilleur* des trois.

**Conséquence si on fait (b)** : le narratif change encore. Les ablations ne sont plus des « improvements » mais des **dégradations sur macro-F1**, qui contredit l'observation +1,30 / +3,32 pp sur l'accuracy pooled.

**Action** : ajouter une phrase explicite §V.F : *« Note that the +1.30/+3.32 pp accuracy improvements coincide with macro-F1 decreases (0.504 → 0.502 / 0.494), indicating the ablations trade balanced performance for slice-conditional UP bias rather than extracting more directional skill. »*

C'est l'honnêteté complète et ça resserre le narratif : aucune des trois variantes TFT n'extrait du signal directionnel utile sur ce dataset.

### A4 — La narrative *positive* du papier devient maigre 🟡

Après le pivot, ce que le papier *démontre* :
1. ✅ TFT v3 reproductible reproduit dans un container Vertex (livrable méthodo).
2. ✅ 10 baselines classiques avec IC, McNemar, macro-F1.
3. ✅ 2 ablations TFT internes qui falsifient les hypothèses architecturales originales.
4. ✅ Documentation honnête d'un écart non reproductible 77,4 % → 50,47 %.

Ce qu'il *ne démontre pas* (mais que le titre / abstract suggèrent) :
- Le « Multi-Component » (§V.G vide).
- L'auditabilité (§VII.B est de la prose architecturale, pas une mesure).
- La robustesse (§VII.A pareil, pas d'expérience).
- Quoi que ce soit qui dépasse 50–54 % d'accuracy directionnelle.

**Ce que dira un reviewer SaTML** : *« Reproducibility lesson is fine but a single null result isn't a paper by itself. What's the constructive contribution? »*

**Trois sorties possibles** :
- **(s1)** Ajouter une expérience constructive : par exemple le sweep A1–A9 multi-agent (A1) + un mini-poisoning expérimental sur §VII.A (1–2 jours de code).
- **(s2)** Repositionner explicitement comme « short paper / workshop paper » plutôt que conf full paper. Réduire à 6–8 pages.
- **(s3)** Étoffer la partie ablation : 3 seeds × 3 variantes = 9 retrains pour donner des IC sur les chiffres TFT eux-mêmes, pas seulement bootstrap sur 1 seed.

### A5 — La transparency box dans §IV.C est efficace mais isolée 🟡

L'encadré `\fbox{}` lit bien mais est *un seul block* au milieu de §IV.C. Le lecteur peut le sauter. Pour un reviewer trustworthy AI, ce devrait être au moins **également signalé dans l'abstract et dans la conclusion**.

**État actuel** :
- Abstract : *« An earlier internal estimate of 77.4% directional accuracy is documented in Section ... but is not reproducible »* — ✅ présent.
- Conclusion : *« An earlier internal estimate ... is documented as non-reproducible in Section ... »* — ✅ présent (paragraphe 1).
- Limitations bullet 1 : *« Reproducibility of the historical 77.4% figure ... is the dominant limitation »* — ✅ présent.

**OK, c'est bien signalé**. Mais la transparency box elle-même pourrait s'enrichir avec un mini-diagramme : ligne de temps « 2026-04 : 77,4 % reporté en dev / 2026-05 : Vertex retrain → 50,47 % / écart documenté » pour rendre la chronologie nette.

**Action** : optionnel ; bon-à-avoir, pas bloquant.

---

## B. Pré-existants v2 non résolus

### B1 — §IV.A CNN-LSTM Ablation est devenue redondante 🟠

Avant le pivot, §IV.A était une comparaison externe (CNN-LSTM ne marche pas, donc TFT marche). Maintenant, CNN-LSTM band 51,7–53,8 % est **dans le même 50–54 % band** que TFT v3 (50,47 %), logreg (52,88 %), Transformer (52,26 %).

§IV.A continue d'avoir 80 lignes : 4 runs, val_loss, class collapse formula, fig:training_curves... Tout ça sert maintenant à prouver que CNN-LSTM est dans le band — qui est trivialement attendu. Une seule ligne dans Tab. 4 suffirait.

**Action** : couper §IV.A à 1 paragraphe (« CNN-LSTM run cluster 51.7–53.8% ; consistent with the rest of the 50–54% band ; full ablation reported in appendix / repo »). Récupérer 1 page.

### B2 — `fig:training_curves` montre la courbe v3 non-reproductible 🟡

Lines 575–622 du tex : pgfplots montre la val_loss TFT v3 atteignant 0,5161 à epoch 16. Cette courbe vient du run *non reproductible*. La courbe Vertex (qui converge à val_loss 0,0159 à epoch 4) n'est pas montrée.

Le lecteur regarde la figure, voit « TFT v3 best: 0.5161 » au-dessus du marker bleu, et n'a aucune indication que ce n'est pas le run reproductible.

**Action** : soit (a) annoter la figure avec « non-reproducible development run », soit (b) remplacer par les val_loss Vertex (le `*_summary.json` les contient sous `dir_acc_history` mais pas le full val_loss curve). (a) est plus simple.

### B3 — §V.E « Quantile Regression and Uncertainty Quantification » est mal cadré 🟡

La section dit *« This is particularly valuable for thin markets where liquidity can change rapidly »*. Mais maintenant que TFT à 50,47 % a une accuracy directionnelle au niveau du hasard, la valeur du quantile n'est plus directionnelle — elle est dans la quantification du risque.

**Action** : reformuler : *« On this dataset where the directional median is essentially at chance, the value of the quantile head is not in the directional decision but in producing a calibrated $80\%$ prediction interval that downstream position-sizing logic can consume. Empirical calibration of these intervals remains to be measured (cf. Limitations). »*

### B4 — §VII.B Audit trail prétend être un building block pour conformité 🟡

Bien réécrite par les critiques précédentes. Mais après le pivot, on est dans la situation : audit trail détaillé pour un modèle qui ne bat pas always-UP. Un reviewer peut dire : *« You can audit a model that's no better than a coin flip — but should you? »*

Une phrase explicite §VII.B ferait du bien : *« We note that the value of audit-trail infrastructure is independent of headline accuracy: a regulator requires the trail whether the model performs at $50\%$ or $77\%$, and the trail's mechanical correctness (every prediction has feature attributions, decision narratives, and data-quality flags attached) is what the EU AI Act Articles 13--14 actually demand. »*

### B5 — Multiple testing absent du panel global 🟡

Tab. 4 a maintenant 13 lignes (10 baselines + 3 TFT variants). On rapporte McNemar pair-wise dans certaines observations mais aucune correction Bonferroni / Holm pour le fait qu'on teste $\binom{13}{2} = 78$ paires. La citation `ref:harvey_multiple` est dans la bibliographie depuis avant — utilisée seulement en §II.A. Devrait être citée explicitement §V.A pour reconnaître le risque.

**Action** : ajouter une phrase §V.A : *« We report pairwise McNemar tests but do not apply a family-wise correction (e.g. Bonferroni) across the $\binom{13}{2}=78$ pairs; readers should adjust $p$-thresholds accordingly~\cite{ref:harvey_multiple}. »*

### B6 — Author block toujours non-soumis à conférence concrète 🟡

`\author{\IEEEauthorblockN{Anonymous}\IEEEauthorblockA{Submitted to IEEE Conference on Secure and Trustworthy AI}}` — ce nom n'existe pas comme conférence IEEE distincte. Plus probable : *IEEE SaTML* (Symposium on Secure and Trustworthy ML) ou *IEEE TIFS*.

**Action** : préciser le venue. Reproducibility-first papers se placent bien à : ICML Reproducibility Workshop, MLOps Workshop @ NeurIPS, ou IEEE BigData Industry Track.

---

## C. Cleanups mineurs

- **C1** Tab `tab:tft_progression` mélange 4 lignes avec val_loss à 3 échelles différentes (0,6931 BCE / 0,6051 BCE / 0,5161 pinball-raw / 0,0159 pinball-zscored). Visuellement déroutant. Suggestion : ajouter une colonne « Loss scale » avec valeurs {BCE-binary, pinball-raw, pinball-zscored} pour rendre les non-comparabilités explicites.

- **C2** Article §IV.C subsubsection « Heterogeneity caveat » dit que le `breakdown.json` est à `reports/per_stock/breakdown.json`. À vérifier : ce fichier existe-t-il déjà ? Sinon le commentaire est promissoire.

- **C3** §III.C Confidence Gating mentionne `threshold_{technical} = 0.60`. Si on appliquait Eq. 3 sur le baseline TFT v3 à 50,47 %, le confidence score serait calculé sur des quantiles très souvent non-monotones → la `multiplicative penalty of 0.6 when quantiles cross` activée souvent. Worth checking si le `confidence_gating_sweep.csv` qu'on a vu (vide actuellement) gives meaningful coverage curves once populé.

- **C4** `fig:architecture` (TikZ) : le bullet précédent C7 de v2 (XAI agent flux) reste valide — aucun edit. Pas bloquant.

- **C5** Section §IX Future Work : 7 items. Maintenant que le papier est repositionné, les 3 plus pertinents sont (i) sweep multi-agent A1–A9, (ii) baselines re-évalués sur slice Vertex, (iii) per-stock breakdown TFT v3 dans le main text. Les 4 autres (Arabic NLP, federated, streaming, surveillance) sont des extensions ; les déplacer dans une sous-section « Extensions » distincte.

- **C6** Aucune section « Reproducibility and Artifact Availability » dédiée (déjà mentionné C8 de v2). Avec le repositionnement, c'est encore plus important. Suggestion : sous-section avant la Conclusion, listant repo URL anonymisée, commit hash, container digest, hardware, durée, coût total.

---

## Tableau de DoD mis à jour

| # | Origine | Statut au 11 mai post-pivot |
|---|---|---|
| C1 (v1) Vérification 77,4 % | critique_synthesis | ✅ Résolu — 50,47 % avec IC bootstrap |
| C2 (v1) Baselines compétitives | critique_synthesis | ✅ Résolu — 10 baselines + 3 TFT variants |
| C3 (v1) Fuite temporelle | critique_synthesis | ✅ Documentée (3 hypothèses dans transparency box) |
| C4 (v1) Ablation multi-agent | critique_synthesis | 🔴 **Pire qu'avant** — A1 baseline maintenant explicite à 50,47 %, gap béant |
| C5 (v1) Claims EU AI Act | critique_synthesis | ✅ Résolu — « designed to support » |
| C6 (v1) État de l'art | critique_synthesis | ✅ Résolu — 50 entrées |
| **A1 (v3)** Multi-agent §V.G | nouveau | 🔴 |
| **A2 (v3)** Slice mismatch | nouveau | 🟠 |
| **A3 (v3)** Reading 2 slice-bias | nouveau | 🟠 |
| **A4 (v3)** Narrative positive maigre | nouveau | 🟡 |
| **B1 (v3)** CNN-LSTM autonome | héritage v2 | 🟠 |
| **B2 (v3)** training_curves non-reproductible | héritage v2 | 🟡 |
| **B5 (v3)** Multiple testing | héritage v2 | 🟡 |

---

## Recommandation prioritaire

Trois actions à faire avant soumission **dans cet ordre** :

1. **Réévaluer les 10 baselines sur slice Vertex N=6 782** (résout A2, harmonise Tab. 4 et `tab:tft_ablations`). Effort : 1–2 h.
2. **Exécuter sweep multi-agent A1–A9** sur CSV Vertex (résout A1, donne du sens au titre). Effort : 1 jour de scripting + ~1 h CPU.
3. **Trancher A4** : workshop reproductibilité (court papier 4–6 pages) OU full conférence avec sprint multi-agent / poisoning (option s1 ci-dessus).

L'item #2 est de loin le plus important. Sans A1–A9 chiffré, le titre « Multi-Component » reste un slogan, le papier reste un « TFT-doesn't-reproduce paper » et un reviewer va dire à juste titre que c'est trop maigre pour une publication full.

---

*Source : lecture critique de la version compilée 19 pages PDF générée le 2026-05-11 09:46.*
