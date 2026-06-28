import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

NIVEAU_LABELS = {0: "Non urgent", 1: "Urgence relative", 2: "Urgence vitale"}

st.set_page_config(page_title="Triage IA - Urgences", page_icon="🚑")
st.title("🚑 Triage IA - Aide à la décision d'urgence")

st.header("Nouvelle évaluation")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        sexe = st.selectbox("Sexe", ["F", "H"])
        tranche_age = st.selectbox("Tranche d'âge", ["0-18", "18-40", "40-65", "65+"])
        source = st.selectbox("Source", ["appel", "chat"])
        antecedents = st.checkbox("Antécédents médicaux (pathologie chronique)")
        duree_symptomes = st.number_input("Durée des symptômes (heures)", min_value=0.0, value=1.0)
    with col2:
        freq_cardiaque = st.number_input("Fréquence cardiaque (bpm)", min_value=0, value=80)
        tension_sys = st.number_input("Tension systolique (mmHg)", min_value=0, value=120)
        temp = st.number_input("Température (°C)", min_value=30.0, max_value=45.0, value=37.0)
        sat_oxygene = st.number_input("Saturation O2 (%)", min_value=0, max_value=100, value=98)

    description_texte = st.text_area("Description des symptômes (texte libre)")

    submitted = st.form_submit_button("Évaluer le niveau d'urgence")

if submitted:
    payload = {
        "sexe": sexe,
        "tranche_age": tranche_age,
        "source": source,
        "antecedents": antecedents,
        "duree_symptomes": duree_symptomes,
        "freq_cardiaque": freq_cardiaque,
        "tension_sys": tension_sys,
        "temp": temp,
        "sat_oxygene": sat_oxygene,
        "description_texte": description_texte,
    }
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Erreur lors de l'appel à l'API : {exc}")
    else:
        niveau = result["niveau_urgence"]
        st.session_state["last_prediction_id"] = result["id_prediction"]

        if niveau == 2:
            st.error(f"⚠️ Niveau prédit : {NIVEAU_LABELS[niveau]}")
        elif niveau == 1:
            st.warning(f"Niveau prédit : {NIVEAU_LABELS[niveau]}")
        else:
            st.success(f"Niveau prédit : {NIVEAU_LABELS[niveau]}")

        st.bar_chart(
            {NIVEAU_LABELS[int(k)]: v for k, v in result["probabilities"].items()}
        )
        st.caption(f"Prédiction #{result['id_prediction']} — modèle : {result['model_version']}")

st.divider()
st.header("Donner un retour (feedback) sur une prédiction")
st.caption(
    "Une fois la vraie décision médicale connue, l'enregistrer permet d'alimenter "
    "le réentraînement futur du modèle."
)

with st.form("feedback_form"):
    id_prediction = st.number_input(
        "ID de la prédiction",
        min_value=1,
        value=st.session_state.get("last_prediction_id", 1),
        step=1,
    )
    actual_niveau = st.selectbox(
        "Niveau d'urgence réel (décision médicale)",
        options=[0, 1, 2],
        format_func=lambda x: NIVEAU_LABELS[x],
    )
    feedback_submitted = st.form_submit_button("Envoyer le feedback")

if feedback_submitted:
    try:
        response = requests.patch(
            f"{API_URL}/predict/{int(id_prediction)}/feedback",
            json={"actual_niveau_urgence": actual_niveau},
            headers={"X-API-Key": API_KEY},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Erreur lors de l'envoi du feedback : {exc}")
    else:
        st.success("Feedback enregistré.")
