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

    # --- ANAGRAFICA (come prima) ---
    if "anag_df" not in st.session_state:
        st.session_state.anag_df = load_anagrafica()
    anag = st.session_state.anag_df.copy()

    st.subheader("Seleziona cliente")
    if anag.empty or "denominazione" not in anag.columns:
        st.warning("Nessun cliente presente. Aggiungine uno qui sotto.")
        selected = None
    else:
        denoms = sorted([d for d in anag["denominazione"].dropna().astype(str).unique() if d.strip()])
        selected = st.selectbox("Denominazione", options=denoms, index=0 if denoms else None, placeholder="Scegli...")
        if selected:
            row = anag[anag["denominazione"] == selected].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Indirizzo:** {row.get('indirizzo','')}")
                st.markdown(f"**Provincia:** {row.get('provincia','')}")
            with c2:
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

    # =======================================================================
    # NUOVA SEZIONE: CARICAMENTO EXCEL (12 mesi) + RIEPILOGO PRODUZIONE/MESI
    # =======================================================================
    st.divider()
    st.subheader("File Excel 12 mesi — caricamento e lettura")

    up = st.file_uploader("Carica il file Excel annuale (.xlsx)", type=["xlsx"])
    if up is None:
        st.info("Carica un file con gli ultimi 12 mesi. Colonne minime: Data e ora, Produzione totale, Consumo totale, Autoconsumo, Energia alimentata nella rete, Energia prelevata.")
        return

    try:
        # prova a leggere i fogli; se esiste un foglio chiamato “Anno” usalo, altrimenti il primo
        xls = pd.read_excel(up, sheet_name=None)
        if "Anno" in xls:
            df = xls["Anno"].copy()
        else:
            # prende il primo foglio disponibile
            first_sheet = next(iter(xls))
            df = xls[first_sheet].copy()

        # normalizzazione colonne attese
        # nomi possibili per la produzione (accetta “Energia per inverter …” o “Produzione totale”)
        prod_col = None
        for c in df.columns:
            cs = str(c)
            if "Energia per inverter" in cs or cs.strip().lower() == "produzione totale":
                prod_col = c; break
        if prod_col is None:
            st.error("Colonna produzione non trovata. Attesa: 'Produzione totale' oppure 'Energia per inverter | ...'")
            return

        # Data e ora → datetime
        if "Data e ora" not in df.columns:
            st.error("Colonna 'Data e ora' non trovata.")
            return
        df["Data e ora"] = pd.to_datetime(df["Data e ora"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Data e ora"])

        # valori numerici (Wh) → kWh
        for col in [prod_col, "Consumo totale", "Autoconsumo", "Energia alimentata nella rete", "Energia prelevata"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0  # se manca, metto 0 per non bloccare
        df["Produzione_kWh"]     = df[prod_col] / 1000.0
        df["Consumo_kWh"]        = df["Consumo totale"] / 1000.0
        df["Autoconsumo_kWh"]    = df["Autoconsumo"] / 1000.0
        df["Rete_immessa_kWh"]   = df["Energia alimentata nella rete"] / 1000.0
        df["Rete_prelevata_kWh"] = df["Energia prelevata"] / 1000.0

        # aggregazione per mese (YYYY-MM)
        df["mese"] = df["Data e ora"].dt.to_period("M")
        agg = (df.groupby("mese")[["Produzione_kWh","Consumo_kWh","Autoconsumo_kWh","Rete_immessa_kWh","Rete_prelevata_kWh"]]
                 .sum()
                 .reset_index())
        agg["mese"] = agg["mese"].astype(str)
        # reformat mese → MM-YYYY
        agg["Mese"] = agg["mese"].apply(lambda s: f"{s.split('-')[1]}-{s.split('-')[0]}")

        # ordina per data crescente e limita a 12 mesi
        agg = agg.sort_values("mese").tail(12).reset_index(drop=True)

        # mostra tabella riepilogo
        cols_show = ["Mese","Produzione_kWh","Consumo_kWh","Autoconsumo_KWh","Rete_immessa_kWh","Rete_prelevata_kWh"]
        # rinomina per etichette pulite
        show = agg.rename(columns={
            "Produzione_kWh":"Produzione (kWh)",
            "Consumo_kWh":"Consumo (kWh)",
            "Autoconsumo_kWh":"Autoconsumo (kWh)",
            "Rete_immessa_kWh":"Rete immessa (kWh)",
            "Rete_prelevata_kWh":"Rete prelevata (kWh)"
        })[["Mese","Produzione (kWh)","Consumo (kWh)","Autoconsumo (kWh)","Rete immessa (kWh)","Rete prelevata (kWh)"]]

        st.subheader("Riepilogo 12 mesi (kWh)")
        st.dataframe(show.style.format("{:.1f}"), use_container_width=True)

        st.success("Lettura completata. Prossimo passo: grafico + PDF (quando arrivano le credenziali Google useremo l’anagrafica online).")

    except Exception as e:
        st.error(f"Errore lettura Excel: {e}")


if __name__ == "__main__":
    if check_password():
        main()

