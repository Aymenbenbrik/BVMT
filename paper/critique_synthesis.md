# Synthèse de la critique — Article IEEE BVMT Multi-Agent

> **Article concerné** : `paper/bvmt_multiagent_ieee_article.tex`
> *« Architecting Secure Multi-Agent Intelligence for Illiquid Financial Ecosystems: A Temporal Fusion Approach for North African Equity Markets »*
>
> **Source des critiques** : `paper/A faire/` (6 captures, datées 2026-05-08)
> **Date de synthèse** : 2026-05-09

---

## Résumé exécutif

Six critiques majeures ont été identifiées. Elles se regroupent en **trois familles** :

| Famille | Critiques concernées | Sévérité |
|---|---|---|
| **A. Validité empirique** des résultats | C1 (résultat 77,4 %), C2 (baselines), C3 (fuite de données) | 🔴 Bloquante pour publication |
| **B. Évaluation du système multi-agent** | C4 (ablation multi-agent) | 🟠 Importante — la contribution principale n'est pas démontrée |
| **C. Positionnement scientifique** | C5 (claims réglementaires), C6 (état de l'art) | 🟡 Modérée — affecte la crédibilité, pas la validité |

En l'état actuel, le **résultat central de 77,4 % de précision directionnelle n'est pas défendable** devant un comité de relecture IEEE : il manque les contrôles statistiques et économiques élémentaires, les baselines compétitives, et la preuve d'absence de fuite temporelle.

---

## C1 — Le résultat de 77,4 % est insuffisamment vérifié

**Sévérité** : 🔴 Bloquante

**Constat** : 77,4 % de précision directionnelle sur 7 jours est exceptionnellement élevé pour de la prédiction boursière, **même** sur un marché peu liquide. Sur les marchés développés, l'état de l'art TFT plafonne à 55–65 % (l'article le rappelle lui-même, §II.A). Un résultat 12+ points au-dessus de la littérature exige une vérification proportionnée. Or l'article n'en fournit aucune.

**Manquements identifiés** :
1. Pas d'**intervalle de confiance** (bootstrap ou DeLong) autour du 77,4 %.
2. Pas de **test statistique de significativité** (McNemar, Diebold-Mariano) versus les baselines.
3. Pas de **matrice de confusion** ni de précision par classe (UP/DOWN/HOLD).
4. Pas de **performance par action** (l'article admet 68 actions ; la moyenne peut masquer une forte hétérogénéité).
5. Pas de **comparaison versus des baselines simples mais fortes** (cf. C2).
6. Aucune **mesure économique** : rendement cumulé, ratio de Sharpe, drawdown maximum, coûts de transaction, capacité de remplissage des ordres.

**Risque sous-jacent** : sur un marché illiquide, un modèle peut être directionnellement correct **sans être économiquement exploitable** — le spread bid-ask, l'absence de volume et l'impossibilité d'exécuter les ordres peuvent annuler tout signal.

---

## C2 — Les baselines de comparaison sont trop limitées

**Sévérité** : 🔴 Bloquante

**Constat** : la comparaison principale oppose le TFT à (i) un baseline aléatoire à 50 % et (ii) quatre runs CNN-LSTM. C'est insuffisant pour démontrer la supériorité du TFT sur l'**ensemble** des méthodes classiques.

**Baselines minimales attendues** :

| # | Baseline | Pourquoi c'est nécessaire |
|---|---|---|
| 1 | **Always-DOWN / classe majoritaire** | L'article déclare 41,9 % de fenêtres UP. Le baseline pertinent n'est donc **pas 50 %** mais 58,1 % (always-DOWN). C'est un point critique : la « +25,5 % vs random » s'effondre à +19,3 % vs always-DOWN. |
| 2 | **Logistic regression** | Référence linéaire incompressible. |
| 3 | **Random Forest** | Référence non-paramétrique standard. |
| 4 | **XGBoost technique** | Le projet utilise déjà XGBoost en fondamental ; appliqué aux features techniques, c'est la baseline tabulaire moderne. |
| 5 | **LSTM simple** | Pour isoler l'apport de l'attention. |
| 6 | **Transformer standard** | Pour isoler l'apport spécifique du TFT (VSN + gating + multi-horizon) versus l'attention seule. |
| 7 | **Modèle naïf momentum / moving-average crossover** | Stratégie sans ML, exigée par la communauté quant. |

**Ablations TFT internes manquantes** :
- TFT **sans** normalisation par action.
- TFT **sans** Variable Selection Network.
- TFT **sans** confidence gating.

**Implication** : l'argument selon lequel « CNN-LSTM plafonne à 53,8 % donc TFT est supérieur » est un **homme de paille**. La vraie question est : *TFT est-il supérieur à XGBoost tabulaire et à un LSTM bien tuné* ? L'article ne répond pas.

---

## C3 — Risque de fuite de données à clarifier

**Sévérité** : 🔴 Bloquante (potentiel) / 🟠 Importante (si à corriger en simple clarification)

**Constat** : le protocole walk-forward (train 2016–2023, val 2024, test 2025) est **conceptuellement correct** mais l'article ne précise pas si l'implémentation respecte la **causalité temporelle stricte**.

**Points à clarifier impérativement** :

1. **Indicateurs techniques** (MA20, MA50, RSI, volatilité, ratios price/MA20) :
   - Sont-ils calculés *uniquement* à partir du passé disponible à chaque date $t$ ?
   - Ou pré-calculés sur l'ensemble de la série (y compris 2024–2025) puis splittés ? **Cette deuxième pratique introduit une fuite invisible.**

2. **Normalisation par action** ($\mu_i, \sigma_i$) :
   - Ajustée sur le set de **train uniquement** ?
   - Ou recalculée sur la série complète, ce qui ferait fuiter de l'information du test (la moyenne et l'écart-type 2025 sont vus à l'entraînement) ?

3. **News & données fondamentales** :
   - Seules les nouvelles **publiées avant** la date de prédiction sont-elles utilisées ?
   - Les états financiers utilisés respectent-ils le délai de publication réglementaire (rapport Q4 souvent publié N+3 mois) ?
   - Le pré-scoring offline du corpus 7 937 articles (mentionné §II.F) est-il **temporellement filtré** au moment de l'inférence ?

**Risque** : si même un seul de ces points laisse fuiter du futur, le 77,4 % peut s'expliquer entièrement par cette fuite. C'est l'**explication la plus parcimonieuse** du résultat élevé, et c'est la première chose qu'un relecteur exigera de vérifier.

---

## C4 — La contribution multi-agent est peu évaluée empiriquement

**Sévérité** : 🟠 Importante

**Constat** : l'article s'intitule *« Multi-Agent Intelligence »* et présente sept agents, mais **toute la section expérimentale porte sur le TFT seul**. Le caractère multi-agent reste un cadre conceptuel non quantifié.

**Ablations exigées** :

| Configuration | Mesure attendue |
|---|---|
| TFT seul | Baseline du système |
| TFT + Fundamental | Gain marginal de l'agent fondamental |
| TFT + Sentiment | Gain marginal de l'agent NLP |
| TFT + Graph | Gain marginal du graphe de corrélation |
| TFT + XAI | (n'affecte pas la précision mais doit être documenté) |
| Système complet (7 agents) | Performance d'ensemble |
| Système avec failure isolation forcée d'un agent | Validation empirique de la robustesse |

**Effets à mesurer** :
- Effet **réel** du confidence gating (Eq. 3) : combien de fois est-il déclenché ? quel impact sur la précision ?
- Effet de la **redistribution des poids** (Eq. 1) : amélioration ou dégradation lorsqu'un agent est désactivé ?
- **Performance par type d'entreprise** : l'article propose des poids différenciés banques / assurances / non-banques (Tab. 2) sans démontrer empiriquement que cette différenciation améliore les résultats.

**Conclusion du relecteur attendu** : *« Sans ces ablations, je ne peux pas savoir si le multi-agent apporte quoi que ce soit ou s'il s'agit d'un emballage architectural autour d'un seul TFT. »*

---

## C5 — Les claims réglementaires sont trop forts

**Sévérité** : 🟡 Modérée mais visible

**Constat** : l'article affirme que le système **« satisfait »** les Articles 13, 14 et 17 de l'EU AI Act (2024) via SHAP, attention weights et explications en langage naturel.

**Problème** : la conformité à l'EU AI Act ne dépend **pas uniquement** de l'explicabilité technique. Elle implique également :

- **Gouvernance** (rôles, responsabilités, comité d'éthique)
- **Gestion des risques** (système de management du risque, art. 9)
- **Documentation technique** (annexe IV obligatoire)
- **Procédures humaines** (formation, supervision, escalade)
- **Surveillance post-déploiement** (post-market monitoring, art. 72)
- **Validation indépendante** (organisme notifié pour les systèmes à haut risque)

**Reformulation prudente** suggérée : remplacer *« satisfies / satisfies »* par *« is **designed to support** transparency and auditability requirements »*. Cela protège l'article de toute attaque sur le terrain réglementaire et reste défendable.

---

## C6 — Références et état de l'art à renforcer

**Sévérité** : 🟡 Modérée

**Constat** : la bibliographie (17 entrées) couvre les classiques (Vaswani, LSTM, TFT, XGBoost, SHAP, GAT, EU AI Act) mais reste **trop générale** pour positionner précisément la contribution.

**Domaines à enrichir** :

1. **Financial transformers** récents (Informer, Autoformer, FEDformer pour finance ; PatchTST appliqué à la finance ; PrincipleNet, FinFormer).
2. **Forecasting en marchés émergents** (études BRICS, MENA, Afrique).
3. **Learning under illiquidity** (microstructure, impact-aware learning).
4. **Market microstructure** (Almgren-Chriss, Hasbrouck, modèles de carnet d'ordres).
5. **GNN pour marchés financiers** (FinGAT, HATS, MAN-SF, temporal graphs financiers récents 2023–2025).
6. **Robust learning / data poisoning en finance** (Steinhardt et al., Goldblum et al., adversarial attacks sur séries temporelles).
7. **Calibration des modèles probabilistes** (Kuleshov et al., Pinball loss empirique, calibration de quantiles).
8. **Backtesting et évaluation économique** (Bailey & López de Prado, deflated Sharpe ratio, probabilistic Sharpe).

**Effet attendu** : positionner précisément la contribution du papier par rapport à chaque sous-domaine, plutôt que de citer des classiques en bloc.

---

## Tableau de synthèse

| # | Critique | Sévérité | Effort de correction | Type d'action |
|---|---|---|---|---|
| C1 | Vérification du 77,4 % | 🔴 Bloquante | Élevé | Calculs supplémentaires + backtest |
| C2 | Baselines compétitives | 🔴 Bloquante | Élevé | Entraînement de 7 nouveaux modèles |
| C3 | Fuite temporelle | 🔴 Bloquante (à vérifier) | Moyen | Audit du code de feature engineering |
| C4 | Ablation multi-agent | 🟠 Importante | Moyen | Désactivation séquentielle des agents |
| C5 | Claims EU AI Act | 🟡 Modérée | Faible | Reformulation textuelle |
| C6 | État de l'art | 🟡 Modérée | Faible | Enrichissement bibliographique |

---

*Voir `improvement_plan.md` pour le plan d'action détaillé répondant à ces critiques.*
