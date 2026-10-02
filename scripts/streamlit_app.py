"""
streamlit_app.py
=================

Frontend de démonstration pour l'assistant IA de recommandation
d'incidents IT (DHL Maroc).

Deux onglets :
    1. Recommandation -- recherche d'incidents similaires pour un
       ticket donné (retrieval -> clustering -> scoring -> reco).
    2. Tendances -- analyse globale de l'historique (volume dans le
       temps, groupes les plus sollicités, problèmes récurrents).
       Répond à la section 4 du cahier des charges ("Analyse
       intelligente des incidents").

LANCER (depuis la racine du projet) :
    streamlit run scripts/streamlit_app.py

Le LLM local est optionnel (case à cocher). Si le modèle n'est pas
en cache ou que l'espace disque est insuffisant, l'app bascule
automatiquement sur la recommandation heuristique -- la démo ne
plante jamais pour cette raison.
"""

from __future__ import annotations

import base64
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import recommendation_core as rc  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESOLVED_PATH = DATA_DIR / "Inc_resolved by MAIT.xlsx"
ESCALATED_PATH = DATA_DIR / "Inc_registred by MAIT.xlsx"
TICKET_EMBEDDINGS_PATH = DATA_DIR / "ticket_embeddings.npy"
SOLUTION_EMBEDDINGS_PATH = DATA_DIR / "solution_embeddings.npy"

st.set_page_config(
    page_title="DHL AI Incident Assistant",
    page_icon="🛠️",
    layout="wide",
)

# ================================================================
# CHARTE DHL (rouge #D40511 / jaune #FFCC00)
# ================================================================
#
# Place le logo officiel DHL (récupéré sur l'intranet / le portail
# de marque DHL) dans assets/dhl_logo.png à la racine du projet.
# S'il est absent, un bandeau texte stylé est utilisé à la place --
# l'app ne plante jamais pour cette raison.

DHL_RED = "#D40511"
DHL_YELLOW = "#FFCC00"
LOGO_PATH = PROJECT_ROOT / "assets" / "dhl_logo.png"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 3px solid {DHL_RED};
        }}
        div.stButton > button[kind="primary"] {{
            background-color: {DHL_RED};
            border-color: {DHL_RED};
            color: white;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: #A8030D;
            border-color: #A8030D;
        }}
        .stTabs [aria-selected="true"] {{
            color: {DHL_RED};
            border-bottom-color: {DHL_RED} !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {DHL_RED};
        }}
        .dhl-header {{
            background-color: {DHL_RED};
            padding: 18px 26px;
            border-radius: 8px;
            border-left: 10px solid {DHL_YELLOW};
            margin-bottom: 1.4rem;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .dhl-header img {{
            height: 68px;
            border-radius: 6px;
            background: {DHL_YELLOW};
            padding: 6px 10px;
        }}
        .dhl-header h1 {{
            color: white;
            margin: 0;
            font-size: 1.7rem;
        }}
        .dhl-header p {{
            color: #FFEFC2;
            margin: 4px 0 0 0;
            font-size: 0.9rem;
        }}
        .reco-card {{
            background-color: white;
            border: 1px solid #EAEAEA;
            border-left: 6px solid {DHL_YELLOW};
            border-radius: 8px;
            padding: 20px 24px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            margin-bottom: 1rem;
        }}
        .reco-card h3 {{
            margin-top: 0;
            color: {DHL_RED};
        }}
        div[data-testid="stMetric"] {{
            background-color: white;
            border: 1px solid #EAEAEA;
            border-radius: 8px;
            padding: 10px 6px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _logo_html(height_px: int = 68) -> str:
    """Retourne un <img> base64 du logo DHL, ou une chaîne vide si absent."""
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" style="height:{height_px}px;" />'


_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"(?<!\d)(\+?\d[\d .-]{7,}\d)(?!\d)")


def _mask_pii(text) -> str:
    """
    Masque les emails et numéros de téléphone dans un texte affiché
    à l'écran (tickets historiques réels, donc potentiellement des
    données personnelles de collègues DHL). Complète -- sans la
    remplacer -- la sanitization déjà faite côté prompt LLM dans
    llm_local._sanitize() (mots de passe / identifiants).
    """
    if text is None:
        return text
    s = str(text)
    s = _EMAIL_PATTERN.sub("[email masqué]", s)
    s = _PHONE_PATTERN.sub("[téléphone masqué]", s)
    return s


def _mask_pii_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique _mask_pii à toutes les colonnes de type texte d'un DataFrame."""
    if df is None:
        return df
    masked = df.copy()
    for col in masked.columns:
        if masked[col].dtype == object:
            masked[col] = masked[col].apply(_mask_pii)
    return masked


# ================================================================
# CHARGEMENT (mis en cache -- une seule fois par session serveur)
# ================================================================

@st.cache_resource(show_spinner="Chargement de la base de connaissance...")
def load_kb():
    kb = rc.load_knowledge_base(str(RESOLVED_PATH), str(ESCALATED_PATH))
    kb = rc.add_solution_types(kb)
    return kb


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def load_embed_model():
    return rc.get_model()


@st.cache_resource(show_spinner="Chargement des embeddings pré-calculés...")
def load_embeddings(n_rows: int):
    ticket_embeddings = None
    solution_embeddings = None

    if TICKET_EMBEDDINGS_PATH.exists():
        ticket_embeddings = np.load(TICKET_EMBEDDINGS_PATH)
        if ticket_embeddings.shape[0] != n_rows:
            st.warning(
                "ticket_embeddings.npy semble désaligné avec la base "
                "actuelle -- relancez scripts/03_backtest.py."
            )

    if SOLUTION_EMBEDDINGS_PATH.exists():
        solution_embeddings = np.load(SOLUTION_EMBEDDINGS_PATH)

    return ticket_embeddings, solution_embeddings


kb = load_kb()
embed_model = load_embed_model()
ticket_embeddings, solution_embeddings = load_embeddings(len(kb))

if ticket_embeddings is None:
    st.error(
        "ticket_embeddings.npy introuvable. Lancez d'abord "
        "`python scripts/03_backtest.py` une fois pour le générer."
    )
    st.stop()


# ================================================================
# BARRE LATÉRALE
# ================================================================

with st.sidebar:
    logo_html_small = _logo_html(height_px=48)
    if logo_html_small:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:10px;">{logo_html_small}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="background:white;border:1px solid #EAEAEA;
                    border-radius:8px;padding:14px 16px;margin-bottom:14px;">
            <h4 style="margin-top:0;color:{DHL_RED};">📚 Base de connaissance</h4>
            <p style="font-size:1.4rem;font-weight:700;margin:0;">{len(kb)}</p>
            <p style="font-size:0.8rem;color:#666;margin-top:2px;">
                tickets historiques (locaux + escaladés à des équipes internationales)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**⚙️ Comment ça marche**")
    st.markdown(
        """
        1. Recherche des tickets historiques similaires
        2. Filtrage des solutions exploitables
        3. Regroupement des solutions proches
        4. Sélection du groupe le plus fiable
        5. Reformulation de la recommandation
        """
    )

    st.markdown("**🔒 Pourquoi c'est fiable**")
    st.caption(
        "Les étapes 1 à 4 sont déterministes et auditables -- pas "
        "de boîte noire. Le LLM local (optionnel, étape 5) ne fait "
        "que reformuler les preuves déjà sélectionnées ; il "
        "n'invente jamais de cause ni de procédure, et se "
        "désactive automatiquement si les preuves sont trop "
        "faibles."
    )

    st.divider()
    use_llm = st.checkbox(
        "Utiliser le LLM local pour reformuler la recommandation",
        value=False,
        help=(
            "Si désactivé (par défaut) : recommandation heuristique, "
            "instantanée. Si activé : un petit LLM local (Qwen2.5) "
            "reformule la même preuve en langage plus naturel. "
            "En cas de problème (modèle absent, espace disque), "
            "l'app revient automatiquement à l'heuristique."
        ),
    )

    st.divider()
    st.caption("Département IT, DHL Maroc")


# ================================================================
# EN-TÊTE + ONGLETS
# ================================================================

st.markdown(
    f"""
    <div class="dhl-header">
        {_logo_html()}
        <div>
            <h1>🛠️ DHL AI Incident Assistant</h1>
            <p>Département IT -- DHL Maroc</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _max_ticket_similarity(result: dict) -> float | None:
    """
    Lit la similarité max calculée par recommend_incident()
    (recommendation_core.py -- source de vérité unique, utilisée
    aussi par le backtest). Filet de sécurité pour compatibilité
    si un ancien recommendation_core.py sans ce champ est utilisé.
    """
    if result.get("max_ticket_similarity") is not None:
        return result["max_ticket_similarity"]

    # Repli si le module central ne fournit pas encore ce champ
    evidence = result.get("evidence") or {}
    tickets = evidence.get("Evidence_Tickets") or []
    sims = [
        t.get("Ticket_Similarity")
        for t in tickets
        if isinstance(t.get("Ticket_Similarity"), (int, float))
    ]
    if sims:
        return max(sims)

    candidates = result.get("candidates")
    if candidates is not None and "Ticket_Similarity" in getattr(candidates, "columns", []):
        col = candidates["Ticket_Similarity"].dropna()
        if not col.empty:
            return float(col.max())

    return None


tab_reco, tab_tendances = st.tabs(["🔍 Recommandation", "📊 Tendances"])


# ================================================================
# ONGLET 1 -- RECOMMANDATION PAR TICKET
# ================================================================

with tab_reco:
    st.caption(
        "Décrivez un problème IT -- l'assistant recherche des incidents "
        "similaires dans l'historique et propose une recommandation."
    )

    query = st.text_area(
        "Description du problème",
        placeholder=(
            "Ex : GCA printer is not working and delivery sheets "
            "cannot be printed."
        ),
        height=100,
    )

    col_btn, _ = st.columns([1, 4])
    run_clicked = col_btn.button("Analyser", type="primary", use_container_width=True)

    if run_clicked:
        if not query.strip():
            st.warning("Merci de décrire un problème avant de lancer l'analyse.")
            st.stop()

        recommendation_generator = None
        llm_notice = None

        if use_llm:
            try:
                import functools
                import llm_local

                with st.spinner("Chargement du LLM local (peut prendre un moment au premier appel)..."):
                    llm_pipe = llm_local.get_llm_pipeline()
                recommendation_generator = functools.partial(
                    llm_local.generate_recommendation_llm, pipe=llm_pipe
                )
            except Exception as error:
                llm_notice = (
                    f"LLM local indisponible ({error}) -- "
                    "repli automatique sur la recommandation heuristique."
                )

        with st.spinner("Recherche des incidents similaires..."):
            result = rc.recommend_incident(
                query=query,
                knowledge_base=kb,
                ticket_embeddings=ticket_embeddings,
                solution_embeddings=solution_embeddings,
                model=embed_model,
                recommendation_generator=recommendation_generator,
            )

        if llm_notice:
            st.info(llm_notice)
        elif use_llm and recommendation_generator is not None and result.get("evidence"):
            # Le LLM a été appelé -- mais generate_recommendation_llm peut
            # lui-même avoir décidé de ne pas s'en servir (garde-fou de
            # qualité des preuves) et être retombé sur l'heuristique en
            # silence côté terminal. On rend cette décision visible ici.
            should_run, reason = llm_local._llm_should_run(result["evidence"])
            if should_run:
                st.caption(f"✅ LLM local utilisé pour reformuler la recommandation ({reason}).")
            else:
                st.info(
                    f"⚠️ Le LLM local n'a pas été utilisé -- {reason}. "
                    "Repli automatique sur la recommandation heuristique "
                    "(garde-fou anti-hallucination)."
                )

        status = result["status"]

        if status != "SUCCESS":
            st.warning(
                result.get("message")
                or f"Statut : {status} -- aucune recommandation fiable trouvée."
            )
        else:
            confidence_sim = _max_ticket_similarity(result)

            if confidence_sim is not None and confidence_sim < 0.45:
                st.error(
                    f"🔴 Confiance faible -- les tickets historiques du "
                    f"groupe retenu n'ont qu'une similarité moyenne de "
                    f"{confidence_sim:.2f} avec ta description. Aucun "
                    "incident vraiment comparable n'a été trouvé dans "
                    "l'historique : la recommandation ci-dessous risque de ne "
                    "pas correspondre au vrai problème -- **à vérifier "
                    "manuellement avant toute action**."
                )
            elif confidence_sim is not None and confidence_sim < 0.65:
                st.warning(
                    f"🟠 Confiance moyenne -- similarité moyenne de "
                    f"{confidence_sim:.2f} entre ta description et les "
                    "tickets historiques du groupe retenu. À vérifier "
                    "avant d'appliquer."
                )
            else:
                sim_txt = f" (similarité {confidence_sim:.2f})" if confidence_sim is not None else ""
                st.success(f"🟢 Recommandation générée à partir de l'historique DHL{sim_txt}.")

            reco_html = "<br>".join(
                line.replace("- ", "• ", 1) if line.strip().startswith("- ") else line
                for line in result["recommendation"].split("\n")
            )
            st.markdown(
                f"""
                <div class="reco-card">
                    <h3>Recommandation</h3>
                    {reco_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Score du cluster gagnant", f"{result['cluster_score']:.2f}")
            col_b.metric("Ticket représentatif", str(result["representative_ticket"]))
            candidates = result.get("candidates")
            col_c.metric(
                "Tickets similaires analysés",
                len(candidates) if candidates is not None else "?",
            )
            col_d.metric(
                "Similarité du cluster retenu",
                f"{confidence_sim:.2f}" if confidence_sim is not None else "?",
            )

            with st.expander("Solution représentative (texte brut)"):
                st.write(_mask_pii(result["representative_solution"]))

            evidence = result.get("evidence") or {}
            evidence_tickets = evidence.get("Evidence_Tickets", [])

            if evidence_tickets:
                st.subheader("Preuves utilisées (tickets historiques du cluster gagnant)")
                evidence_df = _mask_pii_in_dataframe(pd.DataFrame(evidence_tickets))
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)

            with st.expander("Voir tous les tickets candidats (avant clustering)"):
                if candidates is not None:
                    st.dataframe(
                        _mask_pii_in_dataframe(candidates),
                        use_container_width=True,
                        hide_index=True,
                    )


# ================================================================
# ONGLET 2 -- TENDANCES / ANALYSE GLOBALE
# (répond à la section 4 du cahier des charges)
# ================================================================

_STOPWORDS = {
    "the", "and", "for", "with", "not", "are", "was", "were", "has",
    "have", "had", "this", "that", "from", "please", "issue", "ticket",
    "user", "account", "problem", "request", "new", "unable", "need",
    "des", "les", "pour", "avec", "dans", "une", "sur", "est", "pas",
    "que", "qui", "aux", "par", "son", "ses", "nous", "vous", "sont",
    "été", "être", "svp", "merci", "bonjour",
}


def _cluster_keyword_label(descriptions: pd.Series, top_n_words: int = 2) -> str:
    """
    Étiquette lisible pour un groupe de tickets = ses mots-clés les
    plus fréquents DANS ce groupe (donc dérivé des données, pas une
    catégorie fixée à l'avance). Ex: "password / ldap" plutôt que
    "Cluster 7".
    """
    counter = Counter()
    for text in descriptions.dropna().astype(str):
        for word in text.lower().split():
            cleaned = "".join(c for c in word if c.isalpha())
            if len(cleaned) >= 4 and cleaned not in _STOPWORDS:
                counter[cleaned] += 1

    top_words = [w for w, _ in counter.most_common(top_n_words)]
    return " / ".join(top_words) if top_words else "divers"


@st.cache_data(show_spinner="Regroupement sémantique des tickets historiques...")
def _cluster_recurring_issues(_embeddings: np.ndarray, n_clusters: int, kb_len: int):
    """
    Regroupe TOUT l'historique par similarité sémantique (mêmes
    embeddings que le pipeline de recommandation -- cohérent avec
    le reste du projet) pour faire émerger les problématiques
    récurrentes, plutôt qu'une simple fréquence de mots qui ne
    capture ni le sens ni les synonymes/paraphrases.

    Retourne un DataFrame trié par taille de cluster décroissante,
    avec pour chaque cluster : sa taille et jusqu'à 3 exemples de
    description courte.
    """
    from sklearn.cluster import KMeans

    n_clusters = min(n_clusters, kb_len)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(_embeddings)
    return labels


with tab_tendances:
    st.caption(
        "Vue d'ensemble de l'historique complet des tickets : volume "
        "dans le temps, groupes les plus sollicités, problèmes "
        "récurrents et types de solutions."
    )

    kb_dates = kb.copy()
    kb_dates["Opened"] = pd.to_datetime(kb_dates["Opened"], errors="coerce")
    n_undated = kb_dates["Opened"].isna().sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total tickets", len(kb))
    col2.metric("Groupes distincts", kb["Assigned Group"].nunique())
    if n_undated:
        col3.metric("Tickets sans date exploitable", n_undated)

    st.subheader("Volume de tickets dans le temps")
    monthly = (
        kb_dates.dropna(subset=["Opened"])
        .set_index("Opened")
        .resample("MS")
        .size()
    )
    if not monthly.empty:
        st.line_chart(monthly)
    else:
        st.info("Pas de dates exploitables pour tracer l'évolution.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Groupes les plus sollicités")
        st.bar_chart(kb["Assigned Group"].value_counts().head(15))

    with col_right:
        st.subheader("Répartition des types de solutions")
        st.bar_chart(kb["solution_type"].value_counts())

    st.subheader("Problématiques récurrentes (regroupement sémantique)")
    st.caption(
        "Regroupe automatiquement tous les tickets par similarité de "
        "sens (mêmes embeddings que le pipeline de recommandation), "
        "pour faire émerger les types de problèmes qui reviennent le "
        "plus souvent -- plus fiable qu'un simple comptage de mots, "
        "qui ne capture ni les synonymes ni les reformulations."
    )

    n_clusters = st.slider(
        "Nombre de groupes de problématiques à identifier",
        min_value=5, max_value=30, value=12,
    )

    try:
        cluster_labels = _cluster_recurring_issues(
            ticket_embeddings, n_clusters, len(kb)
        )
        kb_clustered = kb.copy()
        kb_clustered["Cluster"] = cluster_labels

        cluster_sizes = kb_clustered.groupby("Cluster").size()

        # Étiquette lisible pour chaque groupe = ses 2 mots-clés
        # dominants (calculés uniquement sur les tickets DE ce groupe,
        # donc entièrement piloté par les données, pas une catégorie
        # inventée à l'avance).
        readable_labels = {}
        for cluster_id, members in kb_clustered.groupby("Cluster"):
            label = _cluster_keyword_label(members["Short Description"])
            readable_labels[cluster_id] = f"{label} ({len(members)})"

        chart_series = (
            cluster_sizes.rename(index=readable_labels)
            .sort_values(ascending=False)
        )
        chart_series.name = "Nombre de tickets"

        st.bar_chart(chart_series)

        st.markdown("**Détail des problématiques les plus fréquentes**")
        for cluster_id in cluster_sizes.sort_values(ascending=False).index[:8]:
            members = kb_clustered[kb_clustered["Cluster"] == cluster_id]
            examples = members["Short Description"].dropna().astype(str).head(3).tolist()
            label = _cluster_keyword_label(members["Short Description"])
            with st.expander(f"{label} -- {len(members)} tickets"):
                for example in examples:
                    st.markdown(f"- {_mask_pii(example)}")

    except ImportError:
        st.warning(
            "scikit-learn n'est pas installé -- `pip install scikit-learn` "
            "pour activer le regroupement automatique des problématiques "
            "récurrentes."
        )

    with st.expander("Filtrer par groupe assigné"):
        selected_group = st.selectbox(
            "Groupe",
            options=["(tous)"] + sorted(kb["Assigned Group"].dropna().unique().tolist()),
        )
        if selected_group != "(tous)":
            subset = kb[kb["Assigned Group"] == selected_group]
            st.write(f"{len(subset)} tickets pour le groupe **{selected_group}**")
            st.dataframe(
                _mask_pii_in_dataframe(
                    subset[["Number", "Opened", "Short Description", "solution_type"]]
                ),
                use_container_width=True,
                hide_index=True,
            )
