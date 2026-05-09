# Critique indépendante de l'article — second tour

> **Article** : `paper/bvmt_multiagent_ieee_article.tex` (état post-Étape 6, commit `bf2aa2f`)
> **Référence du premier tour** : `paper/critique_synthesis.md` (six critiques superviseur).
> **Date** : 2026-05-09.
>
> **Objectif** : critique de relecteur indépendante. Ne reprend pas C1–C6 du superviseur ; pointe des problèmes que ces six critiques ne couvrent pas, en particulier des **incohérences internes** que tout relecteur sérieux trouvera en moins d'une heure.

---

## TL;DR

| Famille | Sévérité moyenne |
|---|---|
| **A — Contradictions internes / erreurs factuelles** | 🔴 Bloquantes (objectives, vérifiables) |
| **B — Lacunes méthodologiques nouvelles** | 🟠 Importantes |
| **C — Claims encore trop forts après Étape 6** | 🟡 Modérées |
| **D — Hygiène numérique** | 🟡 Modérée |

Les points A1–A5 sont les plus dangereux : un relecteur les détectera mécaniquement, et chacun individuellement peut suffire à un *desk reject* IEEE. Ils doivent être corrigés en priorité, avant même de lancer la Phase 1 du `improvement_plan.md`.

---

## A. Contradictions internes et erreurs factuelles

### A1. 🔴 TFT v1 Run 1 ≡ Run 2 mais «vs Random» diffère

**Localisation** : Table `tab:tft_progression`, lignes 611-612.

```
TFT v1 Run 1   0.6036   +12.9%   epoch 6   Initial configuration
TFT v1 Run 2   0.6036   +13.1%   epoch 6   dropout=0.3 fix
```

Si la `Val Loss` est strictement identique à 4 décimales (0.6036), le pourcentage d'amélioration vis-à-vis du baseline 0.6931 doit l'être aussi. Or l'article affiche 12.9 % et 13.1 % :

$$
\frac{0.6931 - 0.6036}{0.6931} = 12{,}91\,\% \quad\text{(les deux)}
$$

→ **Erreur de transcription ou copier-coller**. Soit les deux runs ont une `Val Loss` différente (et la table est arrondie incorrectement), soit le `+13.1 %` est faux. Dans les deux cas, c'est un signal alarme pour un relecteur : si ce détail trivial est faux, qu'en est-il du `0.5161` du v3 ?

**Action** : recalculer la valeur exacte de `Val Loss` pour chaque run depuis `lightning_logs/`, mettre la table à jour avec 4 décimales **vraies**, et recalculer les pourcentages.

---

### A2. 🔴 La section §V.B raisonne sur une cross-entropy que le modèle n'optimise pas

**Localisation** : §V.B « Information-Theoretic Interpretation », lignes 666-673.

L'article écrit :

> *Validation loss is cross-entropy:*
> $$\mathcal{L} = -\frac{1}{N} \sum [y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$
> *A loss of 0.6931 nats represents zero information beyond the class prior.*

Or **§III.C indique explicitement** que la cible du TFT v3 est `future_return_7d` (rendement fractionnaire continu) avec **quantile regression** sur Q10/Q50/Q90. Le code (`training/train_tft_v3.py` ligne 134) confirme :

```python
print(f"Target        : future_return_7d (fractional, clipped ±15%)")
print(f"Quantiles     : {QUANTILES}")
```

→ La perte du TFT v3 est une **pinball loss multi-quantile**, pas une cross-entropy binaire.

Conséquences :
- La valeur 0.6931 (= ln 2) **n'est pas** la baseline pour une pinball loss. Elle est la baseline pour une BCE binaire.
- L'« interprétation entropique » qui occupe toute §V.B (mutual information, prior baseline) est **techniquement non-sequitur** par rapport au modèle entraîné.
- La précision directionnelle 77,4 % n'est pas obtenue par une probabilité de classe `p̂` mais par un seuillage sur Q50. La transformation quantile→direction n'est jamais documentée (cf. B1).

**Action** : soit (a) reporter la pinball loss empirique et l'interpréter dans son cadre propre (e.g. CRPS skill score), soit (b) reproduire l'expérience avec une **vraie** classification binaire et BCE, et alors §V.B devient cohérente. Mais on ne peut pas garder le mélange.

---

### A3. 🔴 Encoder window : 60 jours dans la prose, 30 jours dans le code

**Localisations** :

| Endroit | Valeur |
|---|---|
| Table `tab:tft_config`, ligne 452 | `encoder_window 60 days` |
| §III.B description CNN-LSTM, ligne 506 | `[60, 15] tensor (60 days, 15 features)` |
| §V.B, ligne 671 | « the 60-day feature window provides meaningful predictive signal » |
| §III.B paragraphe Étape 6 (que je viens d'ajouter), ligne ~409 | `each L=30-day encoder window` |
| **Code** `training/train_tft_v3.py` ligne 102 | `MAX_ENCODER_LENGTH = 30` |

→ La vérité est dans le code : **30**. L'article n'a pas mis à jour quand l'auteur est passé de 60 à 30. Mon patch Étape 6 a inséré la valeur correcte (30) sans toucher aux occurrences fautives.

**Action** : un *find-and-replace* `60 days → 30 days` dans le contexte de l'encoder window TFT (en gardant 60 si 60 est le bon chiffre pour CNN-LSTM, ce que le code de `train_cnn_lstm.py` doit confirmer). Mettre à jour aussi la figure `fig:validation_timeline` si elle fait référence à 60.

---

### A4. 🔴 Learning rate dans `tab:tft_config` est auto-contradictoire

**Localisation** : Table `tab:tft_config`, ligne 454.

```
learning_rate    0.0003   Reduced from 0.001 post-epoch-6
```

Et la légende de `fig:training_curves` (ligne 595) :

> « TFT v1 peaks at epoch 6 and degrades (premature convergence due to **LR=0.001**) »

Mais le code `train_tft_v3.py` ligne 131 :

```python
LEARNING_RATE = 0.0003
```

et il s'agit du LR **du v3**, pas un changement intra-run. Donc la note « Reduced from 0.001 post-epoch-6 » est trompeuse :
- soit elle décrit l'historique entre versions (v1 utilisait 0.001, v3 utilise 0.0003) — formulation actuelle ambiguë,
- soit elle suggère un LR scheduler qui passe de 0.001 à 0.0003 au sein du run, ce que le code ne fait pas.

**Action** : remplacer la note par `Reduced from 0.001 (v1) to 0.0003 (v3)` pour lever l'ambiguïté.

---

### A5. 🔴 Le « naive baseline » utilisé est le mauvais

**Localisation** : §V.A « Contextualizing Directional Accuracy », lignes 662-664.

```
A naive "always predict UP" strategy achieves only 41.9% accuracy,
substantially below the 50% random baseline due to class imbalance.
TFT's 77.4% accuracy is therefore particularly meaningful: the model
not only exceeds random but does so in a market with structural bias
toward the minority class.
```

Le baseline naïf pertinent quand 58,1 % des fenêtres sont DOWN n'est **pas** « always-UP » (qui est garanti perdant), c'est **always-DOWN** :

| Stratégie | Précision | Lift TFT v3 |
|---|---|---|
| Always-UP | 41,9 % | +35,5 pp |
| Random 50/50 | 50,0 % | +27,4 pp |
| **Always-DOWN** | **58,1 %** | **+19,3 pp** |

La phrase « TFT's 77.4% accuracy is therefore particularly meaningful » est honnête seulement si on compare à la bonne baseline. Choisir always-UP comme « naive baseline » et déduire que 77,4 % « not only exceeds random » est un **homme de paille rhétorique** — exactement le grief C2 du superviseur, mais avec un degré supplémentaire de sélectivité.

**Action** : remplacer la baseline par always-DOWN ; recalculer le lift réel ; expliciter que le marché BVMT a une biais structurel vendeur. C'est plus honnête et **n'invalide pas** le résultat — 19,3 pp reste un lift significatif.

---

## B. Lacunes méthodologiques nouvelles

### B1. 🟠 Le mapping quantile → direction n'est pas spécifié

§III.C dit que le TFT prédit Q10/Q50/Q90 du rendement à 7 jours.
§V.A reporte une précision directionnelle 77,4 %.

**Manquant** : comment Q10/Q50/Q90 ∈ ℝ devient `{UP, DOWN, HOLD}` ?

Plusieurs choix raisonnables :
- `direction = sign(Q50)`
- `direction = UP si P(Q50 > 0) selon la dist. quantile, DOWN sinon`
- `direction = UP si Q10 > 0 (signal fort), DOWN si Q90 < 0, sinon HOLD`
- Seuillage avec une bande autour de zéro pour HOLD

Chaque choix donne une précision différente. Le HOLD est même mentionné dans `AgentSignal.direction` (ligne 234 de l'article) mais le résultat 77,4 % est bipolaire.

**Action** : spécifier explicitement la règle, idéalement avec une équation. Reporter la précision pour 2-3 règles concurrentes (sensitivity analysis).

---

### B2. 🟠 La matrice de poids adaptative est entièrement à la main

`Tab tab:weights` (lignes 271-280) :

| Type | Tech | Fund | Sent | Graph |
|---|---|---|---|---|
| Banks | 0.42 | 0.31 | 0.27 | Gated |
| Insurance | 0.45 | 0.28 | 0.27 | Gated |
| Non-banks | 0.59 | 0.16 | 0.27 | Gated |

Aucune justification empirique ni procédure d'apprentissage. Les nombres sortent du chapeau et sont présentés comme un *résultat* de l'architecture. C'est de l'**hyperparameter tuning caché** sans validation croisée.

→ La supervision C4 (« multi-agent peu évalué ») est valide ; ce point B2 est complémentaire : la *combinaison* des agents est non seulement non-évaluée, elle est **non-apprise**.

**Action** : soit (a) apprendre les poids par CV bayésienne / grid search sur la période de validation, soit (b) reporter une étude d'ablation où les poids varient et montrer le plateau.

---

### B3. 🟠 Le « plafond » CNN-LSTM avec N=4 runs n'a aucune puissance statistique

`Tab tab:cnn_lstm_ablation` reporte 4 runs : 51.7 %, 51.8 %, 51.7 %, 53.8 %. Le texte (ligne 521) écrit :

> *« The narrow band [51.7%, 53.8%] across all four runs demonstrates that the CNN-LSTM ceiling is not due to hyperparameter mis-tuning but to fundamental architectural unsuitability. »*

Sur un test set de quelques milliers d'échantillons, l'écart-type d'une accuracy binaire est ~0,5–1,0 pp. Quatre seeds dans une plage de 2,1 pp est **dans le bruit**, pas un *ceiling* statistique.

Pour démontrer un plafond, il faut typiquement N ≥ 20 seeds + un test t apparié, ou un test de Wilcoxon, ou une borne supérieure de confiance.

**Action** : étendre à 10–20 seeds (faisable, le CNN-LSTM est petit) et reporter `mean ± 95% CI`. Si le top-quartile reste sous 56 %, alors « ceiling » est défendable.

---

### B4. 🟠 Aucune comparaison aux travaux BVMT existants ni claim explicite « first »

L'article décrit BVMT comme « no public benchmarks exist for these markets » (§I.A) mais ne dit jamais explicitement « to the best of our knowledge, we are the first ML system on BVMT ». Sans cette assertion, le relecteur pourrait suspecter que des travaux antérieurs existent et n'ont pas été cités.

Si l'auteur sait qu'il n'y en a pas (recherche bibliographique faite), **dire-le** est plus fort qu'éviter le sujet.

---

### B5. 🟠 Le modèle de sentiment n'est jamais validé sur le corpus BVMT

§II.F mentionne `bardsai/finance-sentiment-fr-base` comme outperformant les modèles génériques de 5–15 pp sur du sentiment financier. Mais :

- Cette amélioration est rapportée par Rguibi et al. 2023 sur leur propre benchmark, pas sur ilboursa.com.
- Aucune métrique de qualité du sentiment sur le corpus BVMT n'est donnée.
- Aucun échantillon expert-labellisé pour validation.
- Les ~15 % d'articles arabes sont scorés par un modèle français → garbage out probable.

**Action** : labelliser à la main 200 articles ilboursa.com, calculer accuracy/F1 du modèle, reporter dans une table d'annexe. Sans ça, le `SentimentAgent` est une boîte noire qui pèse 0,27 dans la décision finale.

---

### B6. 🟠 « Multi-agent » est un abus de langage

L'article s'intitule « Multi-Agent Intelligence » et décrit une « architecture multi-agent », mais l'implémentation (`agents/orchestrator.py`) est un appel séquentiel de 7 fonctions Python avec un `asyncio.gather` ponctuel. Pas de :

- communication peer-to-peer entre agents,
- négociation, vote, consensus,
- comportement émergent,
- autonomie (chaque « agent » est un objet sans état persistant).

Pour la communauté MAS (Multi-Agent Systems), ce sont des **modules** dans un pipeline, pas des agents.

**Risque** : un relecteur de la track *AI Safety / MAS* va demander l'étude d'interaction entre agents. Reformulation possible : « modular ensemble pipeline » ou « multi-component decision framework » ; garder « multi-agent » dans le titre seulement si on assume une définition opérationnelle (que l'article devrait alors fournir).

---

## C. Claims encore trop forts après Étape 6

### C1. 🟡 Titre « Architecting Secure » sur-promet

Le titre actuel :

> **Architecting Secure Multi-Agent Intelligence for Illiquid Financial Ecosystems**

« Secure » est un mot fort en sécurité informatique. L'article propose un *modèle de menace* (§VI.A) et une *défense conceptuelle* (per-stock + per-window normalization), mais **aucune attaque empirique** n'est exécutée. Pas de quantification du coût d'attaque (capital nécessaire pour manipuler un stock illiquide vs. profit espéré). Pas de stress-test adversarial.

**Action** : soit livrer une expérience de poisoning empirique (faisable : injecter des trades synthétiques dans une copie de la BD, mesurer le shift de prédiction), soit alléger le titre — par exemple « Robustness-Aware Multi-Agent Intelligence ».

---

### C2. 🟡 §II.D : SHAP « satisfait les trois propriétés » est trop fort

§II.D ligne ~146 :

> *« SHAP values satisfy the three properties above [faithfulness, contrastiveness, selectivity], making them suitable for regulatory compliance. »*

La littérature critique sérieusement SHAP sur :
- la faithfulness (Kumar et al. 2020 « Problems with SHAP », Slack et al. 2020 « Fooling LIME and SHAP »),
- la stabilité (Alvarez-Melis & Jaakkola 2018),
- l'interprétation causale (SHAP capture des corrélations, pas des causalités).

Une formulation prudente : *« SHAP provides feature attributions widely used in financial ML; faithfulness and stability remain active research areas (Kumar et al. 2020). »*

---

### C3. 🟡 Graph surveillance §VI.D : promesse sans implémentation

§VI.D présente la « Graph-Based Market Surveillance » comme un cinquième contribution majeur (annoncée dès §I.C contribution 5). Mais §VI.D ne contient **aucune expérience** : pas de cas détecté, pas de précision/rappel, pas de comparaison avec une baseline d'anomaly detection.

C'est de la **spéculation bien écrite**. Pour un papier IEEE, c'est trop. Soit déplacer en *Future Work*, soit produire au moins une étude de cas (un stock dont la corrélation avec son secteur a chuté avant un évènement BVMT documenté).

---

## D. Hygiène numérique

### D1. 🟡 116 626 vs 105 191 lignes — trou non expliqué

| Source | Lignes |
|---|---|
| Article `Tab tab:dataset` ligne 326 | `daily_prices = 116 626` |
| Fichier réel `data/features/tft_features.csv` | **105 191** (vérifié) |

Δ = 11 435 lignes. L'article mentionne « 6,896 rows where close_price fell outside [low, high] : corrected » et d'autres nettoyages mineurs, mais le total ne reconcilie pas. Un relecteur fera la soustraction.

**Action** : ajouter un paragraphe « Final dataset size: 105 191 rows post-cleaning (cf. Table~\ref{tab:dataset_post})» ou ajouter une colonne « Rows after cleaning » dans `tab:dataset`.

---

### D2. 🟡 Taille du test set 2025 jamais reportée

Tout repose sur le 77,4 % en test 2025. Mais combien de prédictions ce test contient-il ? Si c'est 200, l'IC à 95 % via Wilson est ~[71 %, 83 %] — large. Si c'est 5 000, c'est [76 %, 79 %] — étroit.

Le superviseur (C1) demande l'IC. Cet aspect D2 est un sous-cas : **on ne peut pas calculer l'IC sans la taille N**, et N est absent.

---

### D3. 🟡 Per-stock breakdown : le vrai différenciateur thèse

C1 superviseur demande déjà « précision par action ». J'ajoute : ne pas se contenter d'une moyenne et d'un écart-type, **donner la liste des 5 stocks les plus mal prédits**. Si le 77,4 % moyen est porté par 8 grandes capitalisations liquides et que les 60 autres sont à 55 %, c'est l'observation la plus importante de l'article.

---

## Tableau récapitulatif (à utiliser comme checklist avant soumission)

| # | Problème | Sévérité | Effort | Fichier(s) |
|---|---|---|---|---|
| A1 | Val Loss identique mais % différents (TFT v1 R1/R2) | 🔴 | 5 min | `paper/.tex` Tab II |
| A2 | §V.B parle BCE alors que le modèle fait pinball | 🔴 | 1 j (réécriture §V.B + bonne perte) | `paper/.tex` §V.B |
| A3 | Encoder 60 vs 30 partout | 🔴 | 30 min | `paper/.tex` × 4 endroits |
| A4 | LR 0.001 vs 0.0003 ambigu | 🔴 | 5 min | `paper/.tex` Tab III + caption Fig 2 |
| A5 | Naive baseline = always-UP au lieu d'always-DOWN | 🔴 | 30 min | `paper/.tex` §V.A |
| B1 | Mapping quantile→direction non spécifié | 🟠 | 1 j (équation + sensitivity) | `paper/.tex` §V.A + nouveau §V.A.1 |
| B2 | Poids du multi-agent non appris | 🟠 | 2 j | nouvelle expérience |
| B3 | « Ceiling » CNN-LSTM avec N=4 | 🟠 | 1 j (10 seeds suppl.) | `paper/.tex` Tab IV |
| B4 | Pas de claim explicite « first BVMT » | 🟠 | 30 min | `paper/.tex` §I.A ou §I.C |
| B5 | Sentiment non validé sur corpus BVMT | 🟠 | 2 j (200 labels manuels) | nouvelle annexe |
| B6 | « Multi-agent » abusif | 🟠 | 1 h | titre + termes |
| C1 | Titre « Secure » sans test adversarial | 🟡 | 1 h ou 2 j | titre OU nouvelle expérience |
| C2 | SHAP « satisfait » à modérer | 🟡 | 5 min | `paper/.tex` §II.D |
| C3 | Surveillance graphe sans expérience | 🟡 | 0,5 j (étude de cas) ou 5 min (déplacer) | `paper/.tex` §VI.D |
| D1 | 116k vs 105k lignes | 🟡 | 30 min | `paper/.tex` Tab II |
| D2 | Taille test set 2025 absente | 🟡 | 5 min (calcul) | `paper/.tex` §V |
| D3 | Per-stock top-5 worst | 🟡 | 1 j | nouvelle table |

---

## Recommandation d'ordonnancement

**Avant n'importe quelle Phase 1+ du `improvement_plan.md`** :

1. Exécuter A1, A3, A4, A5, C2, C3, D2 — **moins d'une demi-journée** au total et fait disparaître toute trace d'incohérence évidente.
2. Décider sur B6 et C1 (titre / framing) — décision éditoriale, pas technique.
3. A2 demande plus de réflexion : soit reporter la pinball loss correctement (1 j), soit ré-entraîner en classification binaire (3-5 j).

Une fois ces points levés, l'article devient relisable sans rejets mécaniques. **Phase 1** (baselines compétitives) peut alors démarrer sur une base saine.

---

*Document indépendant complétant `critique_synthesis.md`. Les deux ensemble forment le corpus de pré-soumission IEEE.*
