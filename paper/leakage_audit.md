# Audit anti-fuite (leakage audit) — Phase 0

> **Objectif** : pour chaque feature et chaque flux de données, vérifier qu'aucune information du futur ne contamine l'entraînement ou les prédictions historiques.
>
> **Méthode** : revue manuelle du code source.
> **Date d'audit** : 2026-05-09
> **Reviewer** : Claude (audit assisté)
>
> **Statut global** : 🔴 **3 fuites confirmées + 2 zones grises** — corrections requises avant ré-évaluation du 77,4 %.

---

## TL;DR — Score par feature/agent

| Source | Statut | Risque |
|---|---|---|
| MA5 / MA20 / MA50 | ✅ OK | Causales (`pandas.rolling`) |
| RSI(14) | ✅ OK | Causal (`ewm`, `diff`) |
| Volatility 20 | ✅ OK | Causale (`rolling.std`) |
| Volume MA20 | ✅ OK | Causale |
| Daily return | ✅ OK | Causal (`diff`) |
| `price_to_ma20`, `volume_spike`, `rsi_momentum` | ✅ OK | Calculés à partir de features causales |
| Cible `direction_7d` / `future_return_7d` | ✅ Corrigé (Étape 3) | Section 5b drop des bornes ; ~0,9 % de lignes droppées |
| Normalisation cible (GroupNormalizer) | ✅ OK | Fittée sur `train_clean` uniquement |
| Normalisation features d'entrée | ✅ Documenté (Étape 4) | EncoderNormalizer per-window (causal), bloc d'introspection à chaque run |
| Sentiment (inférence live) | ✅ OK | `published_at <= reference_ts` |
| Sentiment (fallback `latest_ts`) | ✅ Corrigé (Étape 2) | `upper_bound_ts=reference_ts` propagé sur les 2 fallbacks |
| Fundamental (`_query_latest_ratios`) | ✅ Corrigé (Étape 1) | Filtre `period_end_date + 4 mois <= prediction_date` |
| Article (paper) vs code | ✅ Aligné (Étape 6) | §III.B + §III.C + §VI.A décrivent maintenant les deux normalizers utilisés |
| Tests anti-fuite (CI) | ✅ Ajoutés (Étape 5) | `tests/test_no_leakage.py` — 12 tests, 2,5 s |

---

## 1. Indicateurs techniques (C3 — Action 0.1)

**Source** : `notebooks/compute_indicators.ipynb`

### 1.1 Calculs vérifiés

```python
# Cellule 2 — RSI (Wilder's EMA)
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()                                  # ✅ causal
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()  # ✅ causal (EWM est causal par défaut)
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

# Cellule de calcul des MA et volatilité
df['ma_5']          = close.rolling(5).mean()             # ✅ causal (closed='right' par défaut)
df['ma_20']         = close.rolling(20).mean()
df['ma_50']         = close.rolling(50).mean()
df['rsi_14']        = compute_rsi(close, period=14)
df['volatility_20'] = df['daily_return'].rolling(20).std()
df['volume_ma_20']  = volume.rolling(20).mean()
df['price_to_ma20'] = close / df['ma_20']
```

### 1.2 Application par stock (cellule 132)

```python
for isin, group in df_all.groupby('isin_code'):
    # ... applique compute_rsi et rolling à chaque groupe
```

✅ **Pas de contamination cross-stock** : les rolling windows sont appliqués strictement à chaque ticker.

### 1.3 Verdict 0.1

| Feature | Causale ? | Cross-stock leak ? | Statut |
|---|---|---|---|
| `daily_return` | ✅ (`diff`) | ✅ (groupby) | OK |
| `ma_5/20/50` | ✅ (`rolling`) | ✅ | OK |
| `rsi_14` | ✅ (`ewm`) | ✅ | OK |
| `volatility_20` | ✅ (`rolling.std`) | ✅ | OK |
| `volume_ma_20` | ✅ (`rolling.mean`) | ✅ | OK |
| `price_to_ma20` | ✅ (composé causal) | ✅ | OK |
| `volume_spike` | ✅ | ✅ | OK |
| `rsi_momentum` | ✅ (RSI - RSI.shift(5)) | ✅ (`groupby.transform`) | OK |

**Conclusion 0.1** : ✅ aucune fuite dans les features techniques. Le seul caveat est que `pandas.rolling(W).mean()` inclut la valeur à $t$ dans la fenêtre $[t-W+1, t]$. C'est correct pour prédire $t+1, \ldots, t+7$ (le futur n'est pas vu).

---

## 2. Cible et normalisation (C3 — Action 0.2)

### 2.1 Construction de la cible

**Fichier** : `training/train_tft_v3.py` (lignes 228-242) et `notebooks/feature_engineering.ipynb` (cellule 6)

```python
df['_future_close'] = (
    df.groupby('ticker_id')['close_price']
      .transform(lambda x: x.shift(-MAX_PREDICTION_LENGTH))   # MAX_PREDICTION_LENGTH = 7
)
df['future_return_7d'] = (
    (df['_future_close'] - df['close_price']) / df['close_price']
).fillna(np.nan)
df = df.dropna(subset=['future_return_7d'])
```

✅ La cible est correctement définie comme un *forward return* à 7 jours, calculée par stock.

### 2.2 🟠 Fuite de bord au split train/val

**Problème** : la cible à $t$ utilise `close_price` à $t+7$. Au split :

```python
TRAIN_END = '2024-01-01'
train_df = df[df['date'] < TRAIN_END]   # rows up to 2023-12-31
```

Pour une ligne en train avec `date = 2023-12-22`, la cible utilise la close du `2023-12-29` → OK.

Pour une ligne en train avec `date = 2023-12-29`, la cible utilise la close du `2024-01-08` → **données de validation utilisées comme label de train**.

**Magnitude** : 7 jours × 68 stocks ≈ 476 lignes contaminées sur ~88 000 lignes de train ≈ **0,54 %**.

**Sévérité** : 🟠 mineure mais doit être citée et corrigée.

**Patch recommandé** :

```python
# Avant le split, retirer les rows dont la cible tombe dans la période suivante
train_cutoff_for_target = pd.to_datetime(TRAIN_END) - pd.Timedelta(days=MAX_PREDICTION_LENGTH + 7)  # marge week-ends
train_df = df[df['date'] < train_cutoff_for_target]

val_cutoff_for_target = pd.to_datetime(VAL_END) - pd.Timedelta(days=MAX_PREDICTION_LENGTH + 7)
# val_df reste construit à partir de history < VAL_END pour garder le contexte encoder
# mais n'inclure dans la loss que les rows avec target < VAL_END
```

### 2.3 GroupNormalizer (normalisation cible)

**Fichier** : `training/train_tft_v3.py` (lignes 366-388)

```python
training_dataset = TimeSeriesDataSet(
    train_clean,                   # ✅ uniquement train
    ...
    target_normalizer = GroupNormalizer(
        groups=['ticker_id'],
        transformation=None,
    ),
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,              # ✅ hérite des stats train, pas de re-fit sur val
    val_clean,
    predict=True,
    stop_randomization=True,
)
```

✅ **Pas de fuite** : GroupNormalizer fittée sur `train_clean` uniquement (avant 2024-01-01). Les stats $\mu_i, \sigma_i$ par stock sont gelées et appliquées telles quelles à la val.

### 2.4 ✅ Normalisation des features d'entrée — résolu (Étape 4)

Dans `train_tft_v3.py`, le constructeur `TimeSeriesDataSet` ne reçoit pas d'argument explicite `scalers={...}`. Conséquence : `pytorch_forecasting` applique son default scaler (`EncoderNormalizer` / `TorchNormalizer`) aux `time_varying_unknown_reals`.

| Default behavior | Causal ? |
|---|---|
| `TorchNormalizer` (depuis pytorch_forecasting >=1.0) | Fitté sur la même donnée passée à `TimeSeriesDataSet` (= `train_clean`) → ✅ pas de fuite |
| `EncoderNormalizer` | Normalise à l'intérieur de chaque fenêtre encoder (30 j passés) → ✅ causal par construction |

**Patch livré (Étape 4)** : un bloc `ANTI-LEAK CHECK` dans `training/train_tft_v3.py` (juste après la construction des datasets) loggue désormais à chaque run :

- le type du `target_normalizer`, ses `groups` et sa `transformation` ;
- le scaler effectivement attaché à chacune des 15 features d'entrée ;
- les flags qui affectent l'input du modèle (`add_target_scales`, `add_relative_time_idx`, …) ;
- un tag `[OK]` ou `[WARN]` par scaler, basé sur la fonction `_classify_scaler`. Toute classe inattendue introduite par un upgrade futur déclenche l'avertissement.

### 2.5 ✅ Discordance article ↔ code — résolue (Étape 6)

L'ancienne version de l'article (LaTeX, §III.B et §VI.A) revendiquait :
> *« Per-stock normalization (defended): $\mu_i$ and $\sigma_i$ computed independently for each stock. »*

Or le code utilise par défaut `EncoderNormalizer` (per-window) pour les features d'entrée, et `GroupNormalizer` (per-stock-static) uniquement pour la cible.

**Patches livrés (Étape 6)** dans `paper/bvmt_multiagent_ieee_article.tex` :

1. **Abstract** : « Per-stock normalization » remplacé par « Per-stock target normalization combined with per-encoder-window feature scaling ».
2. **§I.C contribution 1** : la TFT adaptation est désormais décrite comme « (i) per-stock target normalization … (ii) per-encoder-window feature scaling … (iii) confidence-gated feature selection … (iv) multi-horizon quantile regression ».
3. **§II.A** : phrasing analogue dans la related work.
4. **§III.B (table features)** : la note « Per-stock normalized » est remplacée par « Encoder-window scaled » sur les 7 features concernées.
5. **§III.C (Feature Engineering)** : nouveau paragraphe **« Normalization strategy and walk-forward discipline »** qui détaille les deux normalizers, la fitting strategy train-only, le tronquage de borne et le bloc de logging.
6. **§VI.A (Adversarial Robustness)** : le développement formel propose désormais trois régimes (global, per-stock-static, per-encoder-window) et explicite le rôle de chacun dans la défense anti-poisoning.

---

## 3. Données externes — News & Fondamentaux (C3 — Action 0.3)

### 3.1 SentimentAgent (`agents/sentiment_agent.py`)

#### 3.1.1 ✅ Inférence live — pas de fuite

```sql
SELECT sentiment_score, sentiment_label, published_at
FROM news_articles
WHERE ticker = $1
  AND sentiment_label IS NOT NULL
  AND published_at <= $3                       -- ✅ filtre temporel correct
  AND published_at >= ($3 - ($4 * INTERVAL '1 day'))
```

`$3 = reference_ts` est résolu via `_resolve_reference_timestamp(state)` qui prend `state.seance` ou `state.created_at`. ✅

#### 3.1.2 🔴 Fuite confirmée — fallback `latest_ts`

Lignes 159-175 :

```python
if len(selected_rows) == 0:
    latest_ts = await self._fetch_latest_scored_timestamp(state)   # MAX(published_at) — sans filtre
    if latest_ts is not None:
        for win in WINDOW_CANDIDATES:
            rows = await self._fetch_window_rows(state, latest_ts, win)
            # ↑ La fenêtre est ancrée sur latest_ts, qui peut être DANS LE FUTUR
            #   par rapport à la prediction_date demandée.
```

**Scénario qui fuit** : on back-teste une prédiction au 2023-03-15. Aucun article scoré dans la fenêtre `[2023-02-13, 2023-03-15]`. Le fallback récupère `latest_ts = 2026-01-31`, puis charge les articles `published_at ∈ [2025-08-04, 2026-01-31]`. **Le modèle voit du sentiment 2026 pour prédire 2023.**

Lignes 178-184 (fallback final `_fetch_latest_rows_without_window`) : même problème, encore plus permissif.

**Sévérité** : 🔴 **bloquante pour le back-test 2025**. Si ce fallback s'est déclenché ne serait-ce qu'une fois pendant l'évaluation, le 77,4 % est partiellement corrompu.

**Patch recommandé** :

```python
# Dans _fetch_latest_scored_timestamp et _fetch_latest_rows_without_window,
# ajouter un argument upper_bound_ts et un filtre WHERE published_at <= upper_bound_ts.
async def _fetch_latest_scored_timestamp(self, state: AgentState, upper_bound_ts: datetime):
    rows = await self.query_db(
        '''
        SELECT MAX(published_at) AS latest_at
        FROM news_articles
        WHERE ticker = $1 AND sentiment_label IS NOT NULL
          AND published_at <= $2                                 -- ✅ borne dure
        ''',
        state.ticker, upper_bound_ts,
    )
    ...
```

Et propager `reference_ts` comme `upper_bound_ts` dans toute la chaîne.

### 3.2 FundamentalAgent (`agents/fundamental_agent.py`)

#### 3.2.1 🔴 Fuite majeure — `_query_latest_ratios`

Lignes 98-142 :

```sql
SELECT ..., period_end_date, ...
FROM financial_ratios
WHERE ticker = %(key)s
  AND extraction_confidence >= %(min_conf)s
  AND log_total_assets IS NOT NULL
ORDER BY match_rank DESC, period_end_date DESC
LIMIT 1
```

**Pas de filtre `period_end_date <= prediction_date`. Pas de filtre sur le délai de publication.**

**Conséquences** :
- Au back-test pour 2025-06-15, le modèle voit les ratios de **2024 annual** (publiés ~mars 2025) → OK.
- Au back-test pour **2023-06-15**, le modèle voit les ratios **les plus récents disponibles dans la table** = 2024 ou 2025 → **fuite massive**.
- Même pour 2025-01-15, le modèle voit les ratios 2024 alors que l'annual report 2024 n'est typiquement publié qu'au Q1/Q2 2025.

#### 3.2.2 Délai de publication réglementaire BVMT

| Type de rapport | Délai légal | Délai effectif observé |
|---|---|---|
| Comptes annuels | N+4 mois (CMF) | N+5 à 6 mois en pratique |
| Comptes semestriels | S+2 mois | S+2 à 3 mois |
| Indicateurs trimestriels | T+45 jours | T+1 à 2 mois |

Aucune trace de cette logique dans le code.

#### 3.2.3 Patch recommandé

Ajouter une colonne `publication_date` dans `financial_ratios` (à défaut, dériver `publication_date = period_end_date + INTERVAL '4 months'` comme proxy conservateur), puis :

```sql
WHERE ticker = %(key)s
  AND publication_date <= %(prediction_date)s    -- ✅ borne par la date de prédiction
  AND extraction_confidence >= %(min_conf)s
  ...
ORDER BY publication_date DESC
LIMIT 1
```

Et passer `prediction_date` depuis l'orchestrator vers `_query_latest_ratios`.

**Sévérité** : 🔴 **bloquante**. Le `FundamentalAgent` participe à toutes les prédictions et son poids varie de 0,16 à 0,31 selon le type d'entreprise. Une fuite ici contamine **toutes** les sorties du système multi-agent.

---

## 4. Vérifications complémentaires recommandées

### 4.1 Test de fuite par randomization

Méthode standard pour détecter empiriquement une fuite :

1. Permuter aléatoirement les labels (`direction_7d`) sur le **train uniquement**.
2. Re-entraîner le TFT.
3. Mesurer la précision sur le test 2025.

| Précision attendue | Diagnostic |
|---|---|
| ≈ 50–58 % (random / always-DOWN) | ✅ Pas de fuite |
| > 60 % | 🔴 Fuite résiduelle dans les features ou le pipeline |

### 4.2 Test de cohérence cross-window

Pour 10 dates de prédiction tirées au hasard dans 2025, comparer manuellement :
- les MA, RSI, volatilité que voit le modèle ;
- les MA, RSI, volatilité recalculés à partir d'un dump des prix limité à `< prediction_date`.

Tolérance : 0 (égalité stricte).

### 4.3 Test sur la BD — qu'a vraiment vu chaque prédiction

Logger pour chaque prédiction du test 2025 :
- `prediction_date`
- `min/max published_at` des news utilisées
- `period_end_date` du ratio fondamental utilisé
- `min/max date` des prix utilisés dans l'encoder window

Toute valeur > `prediction_date` = preuve d'une fuite.

---

## 5. Plan de correction priorisé

| # | Action | Sévérité | Effort | Fichier(s) |
|---|---|---|---|---|
| 1 | Patcher `_query_latest_ratios` avec filtre `publication_date <= prediction_date` | 🔴 Bloquante | 1–2 j (incl. ajout colonne BD + remplissage proxy) | `agents/fundamental_agent.py`, `sql/schema.sql`, scripts de re-population |
| 2 | Patcher fallback Sentiment pour borner `latest_ts <= reference_ts` | 🔴 Bloquante | 0,5 j | `agents/sentiment_agent.py` |
| 3 | Tronquer le train de `MAX_PREDICTION_LENGTH + 7` jours pour éviter la fuite de bord cible | 🟠 Mineure | 0,5 j | `training/train_tft_v3.py` |
| 4 | Loguer explicitement les scalers utilisés par `TimeSeriesDataSet` | ⚪ Vérification | 0,2 j | `training/train_tft_v3.py` |
| 5 | Aligner article et code sur la stratégie de normalisation features-d'entrée | 🟠 Discordance | 0,5 j | `paper/bvmt_multiagent_ieee_article.tex` |
| 6 | Ajouter le test de label-shuffle dans `tests/` | ⚪ Validation | 1 j | nouveau fichier `tests/test_no_leakage.py` |
| 7 | Logger pour chaque prédiction le min/max des dates utilisées | ⚪ Audit récurrent | 0,5 j | `agents/orchestrator.py` |

**Total effort estimé Phase 0** : 4–5 jours (compatible avec les 7 j alloués au planning).

---

## 6. Implications pour le résultat 77,4 %

Selon les fuites confirmées :

- **Fundamental (3.2.1)** participe avec un poids de 0,16 à 0,31. Une fuite ici peut artificiellement rehausser la précision du système multi-agent **mais pas celle du TFT seul** (qui est l'agent technique).
- **Sentiment (3.1.2)** : le fallback peut s'être déclenché ou non selon la couverture news 2025. À mesurer.
- **TFT seul** : reste défendable une fois la fuite de bord cible (2.2) corrigée.

→ **Conclusion** : le 77,4 % du **TFT seul** est probablement défendable après correction de §2.2. Mais le 77,4 % attribué au **système multi-agent** est compromis tant que §3.1.2 et §3.2.1 ne sont pas corrigés ET que les baselines n'ont pas été ré-entraînées dans les mêmes conditions causales.

→ **Action immédiate suggérée** : 
1. corriger les patches §1 et §2 ; 
2. relancer l'évaluation 2025 ;
3. comparer les nouveaux 77,4 % aux anciens. Si écart < 1 pp, le résultat est solidifié. Si écart > 3 pp, le résultat doit être révisé dans l'article.

---

*Document à mettre à jour après chaque correction. Voir aussi `improvement_plan.md` Phase 1+ pour les baselines et les ablations qui dépendent de ce nettoyage préalable.*
