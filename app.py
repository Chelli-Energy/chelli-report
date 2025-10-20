import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import date

# -------------------------
# 1) Gate con password unica
# -------------------------
def check_password():
    pw = st.secrets.get("APP_PASSWORD", None)
    if pw is None:
        st.error("Password non configurata. Imposta APP_PASSWORD nei Secrets.")
        st.stop()
    if st.session_state.get("auth_ok", False):
        return True
    st.title("Accesso riservato")
    pwd = st.text_input("Password", type="password")
    if st.button("Entra"):
        if pwd == pw:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Password errata")
    st.stop()

# -------------------------
# 2) Utility
# -------------------------
ANAGRAFICA_PATH = "schema/anagrafica.csv"

ANAG_COLS = ["denominazione","indirizzo","provincia","potenza_kw","data_installazione"]

def load_anagrafica():
    try:
        df = pd.read_csv(ANAGRAFICA_PATH)
        # garantisce colonne nell’ordine atteso
        missing = [c for c in ANAG_COLS if c not in df.columns]
        for c in missing: df[c] = ""
        return df[ANAG_COLS]
    except Exception:
        return pd.DataFrame(columns=ANAG_COLS)

def to_download_button(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")

# -------------------------
# 3) App
# -------------------------
def main():
    st.title("Chelli Report")
    st.caption("Step: Anagrafica clienti — selezione e aggiunta")

    # carica anagrafica (da CSV locale finché non usiamo Google Sheets)
    if "anag_df" not in st.session_state:
        st.session_state.anag_df = load_anagrafica()
    anag = st.session_state.anag_df.copy()

    # --- TENDINA DENOMINAZIONE ORDINATA ---
    st.subheader("Seleziona cliente")
    if anag.empty or "denominazione" not in anag.columns:
        st.warning("Nessun cliente presente. Aggiungine uno qui sotto.")
        selected = None
    else:
        # lista denominazioni uniche ordinate
        denoms = sorted([d for d in anag["denominazione"].dropna().astype(str).unique() if d.strip()])
        selected = st.selectbox("Denominazione", options=denoms, index=0 if denoms else None, placeholder="Scegli...")

        # mostra dettagli sintetici del cliente scelto
        if selected:
            row = anag[anag["denominazione"] == selected].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Indirizzo:** {row.get('indirizzo','')}")
                st.markdown(f"**Provincia:** {row.get('provincia','')}")
            with col2:
                st.markdown(f"**Potenza (kW):** {row.get('potenza_kw','')}")
                st.markdown(f"**Data installazione:** {row.get('data_installazione','')}")

    st.divider()
    st.subheader("Aggiungi nuovo cliente")

    from datetime import date
    with st.form("nuovo_cliente"):
        c1, c2 = st.columns(2)
        with c1:
            denominazione = st.text_input("Denominazione*", "")
            indirizzo = st.text_input("Indirizzo*", "")
            provincia = st.text_input("Provincia*", placeholder="es. FI, PI, SI")
        with c2:
            potenza_kw = st.number_input("Potenza (kW)*", min_value=0.1, step=0.1, value=5.0)
            data_installazione = st.date_input("Data installazione*", value=date.today())

        submitted = st.form_submit_button("Aggiungi all’elenco")
        if submitted:
            if not denominazione or not indirizzo or not provincia:
                st.error("Compila i campi obbligatori contrassegnati con *")
            else:
                new_row = {
                    "denominazione": denominazione.strip(),
                    "indirizzo": indirizzo.strip(),
                    "provincia": provincia.strip(),
                    "potenza_kw": potenza_kw,
                    "data_installazione": data_installazione.strftime("%d/%m/%Y"),
                }
                st.session_state.anag_df = pd.concat(
                    [anag, pd.DataFrame([new_row])],
                    ignore_index=True
                )
                st.success("Cliente aggiunto. Scarica il CSV aggiornato per salvarlo in modo permanente su GitHub.")

    st.divider()
    st.subheader("Esporta anagrafica aggiornata")
    st.caption("Scarica il CSV aggiornato. Per renderlo permanente, caricalo in GitHub al posto di schema/anagrafica.csv.")
    to_download_button(st.session_state.anag_df, "anagrafica_aggiornata.csv", "Scarica CSV aggiornato")


if __name__ == "__main__":
    if check_password():
        main()
