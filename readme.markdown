# Thermo-Stats

Analyse thermique et prévisions intelligientes pour habitation chauffée au poêle

Master 2 Data Engineer – YNOV Nanterre

Auteur : **Marc-Alfred FALIGANT**

---

# 1. Présentation du projet

Thermo-Stats est un outil de simulation thermique et de visualisation météo destiné aux particuliers souhaitant anticiper l’allumage de leur poêle afin de maintenir une température intérieure stable.

Le tableau de bord permet :

- d'anticiper l'allumage optimal du poêle
- de visualiser la météo locale (température, vent)
- de simuler la montée en température suivant l'inertie de l'habitation
- de calculer des KPIs énergétiques
- d'obtenir un diagnostic thermique simple
- de visualiser une carte thermique interactive de la France
- de comprendre l'effet du vent, de la puissance et de l'isolation sur le confort thermique

Le projet s'appuie sur un **modèle thermique RC** simple mais efficace, des **données météo horaires**, et un affichage construit avec **Streamlit**.

---

# 2. Architecture générale

```mermaid
graph TD
    U[Utilisateur] -->|Paramètres habitation| UI[Interface Streamlit]
    UI --> METEO["MeteoClient<br/>(Open-Meteo)"]
    METEO --> DF[DataFrame météo]
    UI --> TM[ThermalModel]
    DF --> TM
    TM --> SIM[Températures simulées]
    UI --> KPI[KPIEngine]
    DF --> KPI
    KPI --> OUT1[KPIs]
    SIM --> OUT2[Projection 24h]
    UI --> DIAG[Diagnostic]
    KPI --> DIAG
    DIAG --> OUT3[Diagnostic thermique]
    UI --> MAP[Carte France]
    METEO --> MAP
```

---

# 3. Structure du projet

```mermaid
graph TD
    A[thermostats/] --> B[app.py]

    A --> C[core/]
    C --> C1[user_inputs.py]
    C --> C2[kpi_engine.py]
    C --> C3[diagnostic.py]

    A --> D[models/]
    D --> D1[thermal_model.py]

    A --> E[data/]
    E --> E1[data_meteo.py]
    E --> E2[data_france.py]

    A --> F[utils/]
    F --> F1[helpers.py]

    A --> G[tests/]
    G --> G1[test_thermal_model.py]
    G --> G2[test_kpi_engine.py]
    G --> G3[test_diagnostic.py]

    A --> H[notebooks/]
```

---

# 4. Modèle thermique RC

```mermaid
graph LR
    Tin[Température intérieure] --> Loss["Pertes thermiques<br/>(W/K)"]
    Text[Température extérieure] --> Loss
    Poele["Poêle<br/>(kW × rendement)"] --> Heat[Apport de chaleur]
    Heat --> Tin
    Loss --> Tin
    Tin <-->|Inertie thermique<br/>(tau)| C[Capacité thermique]
```

---

# 5. Séquence de montée en température

```mermaid
sequenceDiagram
    participant M as ThermalModel
    participant T as T° intérieure
    participant W as Vent
    participant E as T° ext. effective

    M->>W: Reçoit vent (km/h)
    M->>E: T_ext_eff = T_ext - 0.2 × vent
    M->>T: heat_step(T_int, T_ext_eff)
    T-->>M: Nouvelle T_int
    M->>T: Répète jusqu'à la consigne
```

---

# 6. Carte thermique France

```mermaid
graph TD
    FM[FranceMeteo] --> API[(API Open-Meteo)]
    API --> DATA[Temp° / Vent par ville]
    DATA --> MAP[Plotly Mapbox]
    MAP --> UI[Streamlit]
```

---

# 7. Couverture cartographique enrichie

La carte embarque un maillage hors-ligne d'environ 35 000 communes synthétiques couvrant chaque département et région. Les pastilles de température sont disponibles simultanément par niveaux (régions, départements, communes) et sont animées sur 24h (température, vent, humidité). Un cache `data/communes_cache.csv` est généré automatiquement si l'accès aux jeux de données publics est bloqué.

# 8. KPIs dynamiques

Les KPIs (coût journalier, degrés-jours ramenés à 24h, pertes W/K) se recalculent en temps réel à partir des paramètres saisis dans le panneau latéral : structure/isolant/vitrage, VMC, volume et prix du sac de pellets. Le coût inclut la consommation pellets et l'électricité de veille du poêle pour rester cohérent avec les réglages en cours.

---

# 9. Installation

## 7.1 Cloner le dépôt

<pre class="overflow-visible!" data-start="3422" data-end="3497"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>git </span><span>clone</span><span> https://github.com/MAF1995/thermostats.git
</span><span>cd</span><span> thermostats
</span></span></code></div></div></pre>

---

# 8. Environnement virtuel et lancement

## 8.1 Création du venv

<pre class="overflow-visible!" data-start="3570" data-end="3602"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m venv .venv
</span></span></code></div></div></pre>

## 8.2 Activation

### Windows (PowerShell)

<pre class="overflow-visible!" data-start="3649" data-end="3689"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-powershell"><span><span>.venv\Scripts\activate
</span></span></code></div></div></pre>

Windows (CMD)

<pre class="overflow-visible!" data-start="3706" data-end="3745"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-cmd"><span>.\.venv\Scripts\activate.bat
</span></code></div></div></pre>

### Linux / macOS

<pre class="overflow-visible!" data-start="3766" data-end="3803"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>source</span><span> .venv/bin/activate
</span></span></code></div></div></pre>

---

## 8.3 Installation des dépendances

<pre class="overflow-visible!" data-start="3847" data-end="3890"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
</span></span></code></div></div></pre>

---

## 8.4 Lancer Streamlit

<pre class="overflow-visible!" data-start="3922" data-end="3954"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>streamlit run app.py
</span></span></code></div></div></pre>

L'application s'ouvrira automatiquement dans votre navigateur.

---

## 8.5 Lancer les tests

<pre class="overflow-visible!" data-start="4050" data-end="4068"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pytest
</span></span></code></div></div></pre>

---

## 8.6 Geler les dépendances

<pre class="overflow-visible!" data-start="4105" data-end="4146"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip freeze > requirements.txt
</span></span></code></div></div></pre>

---

## 8.7 Désactiver le venv

<pre class="overflow-visible!" data-start="4180" data-end="4202"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>deactivate
</span></span></code></div></div></pre>

---

## 8.8 Activation automatique dans VS Code

Assurez-vous que le fichier suivant existe dans le projet :

**`.vscode/settings.json`**

<pre class="overflow-visible!" data-start="4344" data-end="4510"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
    </span><span>"python.defaultInterpreterPath"</span><span>:</span><span></span><span>".venv/Scripts/python.exe"</span><span>,</span><span>
    </span><span>"python.terminal.activateEnvironment"</span><span>:</span><span></span><span>true</span><span></span><span>,</span><span>
    </span><span>"markdown.mermaid.enabled"</span><span>:</span><span></span><span>true</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

Ainsi, VS Code active automatiquement l'environnement à l'ouverture du projet.

---

# 9. Tests unitaires

Le projet utilise PyTest :

<pre class="overflow-visible!" data-start="4647" data-end="4661"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>pytest</span><span>
</span></span></code></div></div></pre>

Les tests couvrent :

- modèle thermique
- moteur de KPIs
- diagnostic habitation

---

# 10. Sources des données

- API Open-Meteo (modèle Météo-France) — Licence Ouverte v2.0
- Données climatologiques horaires — data.gouv.fr

---

# 11. Auteur

Projet réalisé par :

**Marc-Alfred FALIGANT**

Master 2 Data Engineer — YNOV Campus Nanterre.
