import json
import streamlit as st
import pandas as pd
from io import BytesIO
import gspread
from google.oauth2.service_account import Credentials
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader

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
# 2) Utility base
# -------------------------
ANAGRAFICA_PATH = "schema/anagrafica.csv"
ANAG_COLS = ["denominazione","indirizzo","provincia","potenza_kw","data_installazione","derating_percent"]


def load_anagrafica():
    try:
        df = pd.read_csv(ANAGRAFICA_PATH)
        missing = [c for c in ANAG_COLS if c not in df.columns]
        for c in missing: df[c] = ""
        return df[ANAG_COLS]
    except Exception:
        return pd.DataFrame(columns=ANAG_COLS)

def to_download_button(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")

def gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    info_raw = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]

    # Accetta sia dict (TOML table) che stringa JSON
    if isinstance(info_raw, dict):
        info = dict(info_raw)
    else:
        import json, ast
        try:
            info = json.loads(info_raw)
        except Exception:
            info = ast.literal_eval(info_raw)

    # Normalizza private_key:
    # - se contiene "\\n" -> converte in veri "\n"
    # - se è multiline (TOML table) -> lascia com'è
    pk = info.get("private_key", "")
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    info["private_key"] = pk

    # Controlli minimi prima di creare le credenziali
    if not pk or "BEGIN PRIVATE KEY" not in pk or "END PRIVATE KEY" not in pk:
        raise ValueError("private_key non valida: manca BEGIN/END PRIVATE KEY")

    if not info.get("client_email"):
        raise ValueError("client_email mancante nei secrets")

    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["GSHEET_ID"])
    return sh



def sheet_to_df(ws):
    rows = ws.get_all_records()
    return pd.DataFrame(rows)

def df_append_row(ws, row_dict):
    headers = ws.row_values(1)
    if not headers:
        ws.append_row(list(row_dict.keys()))
        headers = ws.row_values(1)
    values = [row_dict.get(h, "") for h in headers]
    ws.append_row(values)

def load_anagrafica_gs():
    """Legge il foglio 'anagrafica' e normalizza la provincia a sigla."""
    try:
        sh = gs_client(); ws = sh.worksheet("anagrafica")
        df = sheet_to_df(ws)
        for c in ANAG_COLS:
            if c not in df.columns: df[c] = ""

        # normalizza provincia -> sigla
        df["provincia"] = (
            df["provincia"].astype(str).str.strip().str.upper()
                .map(lambda x: PROVINCE_MAP.get(x, x))
        )

        # normalizza derating_percent (0–99 int)
        if "derating_percent" not in df.columns: df["derating_percent"] = ""
        df["derating_percent"] = pd.to_numeric(df["derating_percent"], errors="coerce") \
            .clip(lower=0, upper=99).fillna(0).astype(int)

        return df[ANAG_COLS]

    except Exception as e:
        st.error(f"Errore lettura anagrafica (GS): {e}")
        return pd.DataFrame(columns=ANAG_COLS)

def append_anagrafica_gs(row: dict):
    """Aggiunge un cliente al foglio 'anagrafica'."""
    sh = gs_client(); ws = sh.worksheet("anagrafica")
    df_append_row(ws, row)

def load_coeff_gs():
    """Legge il foglio 'province_coeff' e forza i mesi a numerico."""
    sh = gs_client(); ws = sh.worksheet("province_coeff")
    df = sheet_to_df(ws)
    df["provincia"] = (
        df["provincia"].astype(str).str.strip().str.upper()
          .map(lambda x: PROVINCE_MAP.get(x, x))
    )
    for col in ["gennaio","febbraio","marzo","aprile","maggio","giugno",
                "luglio","agosto","settembre","ottobre","novembre","dicembre"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df

def atteso_for_last_month(prov_sigla: str, potenza_kw: float, last_mm: str, coeff_df: pd.DataFrame) -> float:
    """kWh attesi del mese corrente: coeff(prov, mese)*potenza."""
    mese_col = MESE_COL.get(last_mm, None)
    if not mese_col: return 0.0
    row = coeff_df[coeff_df["provincia"] == prov_sigla]
    if row.empty: return 0.0
    coeff = float(row.iloc[0][mese_col])  # kWh per kW
    return coeff * float(potenza_kw or 0.0)

# -------------------------
# 2bis) Helper grafico e PDF
# -------------------------
GREEN_MAIN = "#006633"
ORANGE = "#F9A825"
RED = "#C62828"
GRAY_PRIOR = "#9b9b9b"
TXT_DARK = "#212121"
TXT_BASE = "#424242"
GRID = "#DDDDDD"

PROVINCE_MAP = {
    "FIRENZE":"FI","FI":"FI","PISA":"PI","PI":"PI","SIENA":"SI","SI":"SI",
    "GROSSETO":"GR","GR":"GR","LIVORNO":"LI","LI":"LI","LUCCA":"LU","LU":"LU",
    "AREZZO":"AR","AR":"AR","PISTOIA":"PT","PT":"PT","MASSA-CARRARA":"MS","MS":"MS","MASSA CARRARA":"MS"
}
MESE_COL = {
    "01":"gennaio","02":"febbraio","03":"marzo","04":"aprile","05":"maggio","06":"giugno",
    "07":"luglio","08":"agosto","09":"settembre","10":"ottobre","11":"novembre","12":"dicembre"
}

def build_monthly_chart(month_labels, prod_values, atteso_last=None, last_ok_class="verde"):
    """Ritorna bytes PNG del grafico a barre."""
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    x = np.arange(len(month_labels))
    widths = np.array([0.5]*len(month_labels))
    colors_bars = [GRAY_PRIOR]*len(month_labels)

    last_color = GREEN_MAIN if last_ok_class=="verde" else (ORANGE if last_ok_class=="arancione" else RED)
    if len(colors_bars) > 0:
        colors_bars[-1] = last_color
        widths[-1] = 0.7

    bars = ax.bar(x, prod_values, width=widths, color=colors_bars, edgecolor="none")
    ax.set_ylabel("Produzione (kWh)")
    ax.set_xticks(x, month_labels, rotation=45, ha="right", fontsize=9)
    ymax = max(prod_values) if prod_values else 1
    ax.set_ylim(0, ymax*1.2)

    # segna la media 12 mesi sul mese corrente
    if len(prod_values) > 0:
        mean_val = float(np.mean(prod_values))
        if len(bars) > 0:
            last_bar = bars[-1]
            bx, bw = last_bar.get_x(), last_bar.get_width()
            ax.hlines(y=mean_val, xmin=bx, xmax=bx+bw,
                      colors="#555555", linewidth=1.0, linestyles=(0,(3,2)))
            # posiziona la scritta sopra l'area del grafico
            ax.text(bx + bw/2, ymax * 1.18,
                    "media 12 mesi", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#333333")



    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + ymax*0.012, f"{h:.0f}", ha='center', va='bottom', fontsize=8)

    if atteso_last is not None and len(bars) > 0:
        last_bar = bars[-1]
        bx, bw = last_bar.get_x(), last_bar.get_width()
        ax.hlines(y=atteso_last, xmin=bx, xmax=bx+bw, colors="white", linewidth=1.2, linestyles=(0,(2,2)))
        ax.text(bx + bw/2, atteso_last + 10, "valore standard del mese", ha="center", va="bottom",
                color="black", fontsize=6)

    buf = BytesIO()
    plt.subplots_adjust(bottom=0.24)  # più spazio per le etichette 
    plt.tight_layout()
    fig.savefig(buf, format="PNG", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def compose_pdf(path_out, logo_path, title_mmYYYY, anag_dict, table_rows, last_class, chart_img):
    """Crea PDF A4 completo."""
    c = canvas.Canvas(path_out, pagesize=A4)
    page_w, page_h = A4
    LM, RM, TM, BM = 2*cm, 2*cm, 2*cm, 2*cm

    # Logo
    logo_w = 7*cm; logo_h = 2.3*cm
    logo_y = page_h - TM - logo_h + 0.3*cm
    c.drawImage(logo_path, (page_w - logo_w)/2, logo_y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')

    # Titolo
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor(TXT_DARK))
    c.drawCentredString(page_w/2, logo_y - 0.9*cm, f"Report produzione fotovoltaica {title_mmYYYY}")

    # Anagrafica
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor(TXT_BASE))
    y = logo_y - 1.9*cm
    for label in ["Denominazione","Indirizzo","Provincia","Potenza (kWh)","Derating (%)","Data installazione"]:
        c.drawString(LM, y, f"{label}:")
        c.drawString(LM + 4.5*cm, y, str(anag_dict.get(label, '')))
        y -= 0.42*cm

    # Grafico
    chart_w = page_w - LM - RM; chart_h = 7.0*cm
    chart_y = y - 0.7*cm - chart_h
    c.drawImage(chart_img, LM, chart_y, width=chart_w, height=chart_h, preserveAspectRatio=True, mask='auto')

    # Tabella
    table_y = chart_y - 2.6*cm - 5.2*cm
    hdr = ["Mese","Produzione\nkWh","Consumo\nkWh","Autoconsumo\nkWh","Rete\nimmessa","Rete\nprelevata","Atteso\nkWh","Scost.\n%"]
    data_table = [hdr] + table_rows
    col_widths = [1.6*cm, 1.8*cm, 1.8*cm, 2.0*cm, 1.8*cm, 2.0*cm, 1.8*cm, 1.6*cm]
    tbl = Table(data_table, colWidths=col_widths)
    ts = TableStyle([
        ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 6.5),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(TXT_DARK)),
        ('FONT', (0,1), (-1,-1), 'Helvetica', 6.5),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor(GRID)),
        ('INNERGRID', (0,1), (-1,-1), 0.2, colors.HexColor(GRID)),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor(GRID)),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
    ])
    hl = "#E8F5E9" if last_class=="verde" else ("#FFF8E1" if last_class=="arancione" else "#FFEBEE")
    ts.add('BACKGROUND', (0, len(data_table)-1), (-1, len(data_table)-1), colors.HexColor(hl))
    tbl.setStyle(ts)
    w, h = tbl.wrap(page_w, 5.2*cm)
    tbl.drawOn(c, (page_w - w)/2, table_y)

    # Caption
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor(TXT_BASE))
    c.drawString(LM, table_y - 1.8*cm, "Nota: Valore atteso calcolato su provincia × potenza impianto.")
    c.setFont("Helvetica-Bold", 10)
    msg = {
        "verde": "Risultato eccellente: produzione del mese in linea o superiore alla media attesa.",
        "arancione": "Risultato buono: produzione del mese leggermente sotto la media attesa.",
        "rosso": "Risultato inferiore agli standard: produzione del mese sensibilmente sotto la media attesa.",
    }[last_class]
    c.drawString(LM, table_y - 2.6*cm, msg)

    # Footer
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#616161"))
    c.drawCentredString(page_w/2, 1.5*cm, "Chelli Energy Solutions — Report automatico")

    c.showPage(); c.save()

# -------------------------
# 3) App
# -------------------------
def main():
    st.title("Chelli Report")
    st.caption("Step: Anagrafica clienti — selezione, caricamento e report")

    # --- ANAGRAFICA ---
    if "anag_df" not in st.session_state:
        st.session_state.anag_df = load_anagrafica_gs()
    anag = st.session_state.anag_df.copy()

    st.subheader("Seleziona cliente")
    if anag.empty or "denominazione" not in anag.columns:
        st.warning("Nessun cliente presente.")
        selected = None
    else:
        denoms = ["-- Seleziona cliente --"] + sorted(
            [d for d in anag["denominazione"].dropna().astype(str).unique() if d.strip()]
        )
        selected = st.selectbox(
            "Seleziona cliente",
            options=denoms,
            index=0,
            label_visibility="collapsed"
        )

        if selected == "-- Seleziona cliente --":
            st.info("Seleziona dall’elenco o aggiungi un nuovo cliente per procedere.")
            selected = None

        if selected:
            row = anag[anag["denominazione"] == selected].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Indirizzo:** {row.get('indirizzo','')}")
                st.markdown(f"**Provincia:** {row.get('provincia','')}")
            with c2:
                st.markdown(f"**Potenza (kW):** {row.get('potenza_kw','')}")
                st.markdown(f"**Data installazione:** {row.get('data_installazione','')}")
                st.markdown(f"**Derating (%):** {row.get('derating_percent', 0)}")


    # --- Aggiungi nuovo cliente ---
    st.divider()
    st.subheader("Aggiungi nuovo cliente")
    with st.form("nuovo_cliente"):
        c1, c2 = st.columns(2)
        with c1:
            denominazione = st.text_input("Denominazione*", "")
            indirizzo = st.text_input("Indirizzo*", "")
            provincia = st.text_input("Provincia*", placeholder="es. FI, PI, SI")
        with c2:
            potenza_kw = st.number_input("Potenza (kW)*", min_value=0.1, step=0.1, value=5.0)
            data_installazione = st.date_input("Data installazione*", value=date.today())
            derating_percent = st.number_input("Derating impianto (%)", min_value=0, max_value=99, value=0, step=1)

        submitted = st.form_submit_button("Aggiungi all’elenco")
        if submitted:
            if not denominazione or not indirizzo or not provincia:
                st.error("Compila i campi obbligatori contrassegnati con *")
            else:
                new_row = {
                    "denominazione": denominazione.strip(),
                    "indirizzo": indirizzo.strip(),
                    "provincia": PROVINCE_MAP.get(provincia.strip().upper(), provincia.strip().upper()),
                    "potenza_kw": float(potenza_kw),
                    "data_installazione": data_installazione.strftime("%d/%m/%Y"),
                    "derating_percent": int(derating_percent),

                }
                try:
                    append_anagrafica_gs(new_row)
                    st.session_state.anag_df = load_anagrafica_gs()
                    st.success("Cliente aggiunto su Google Sheets.")
                except Exception as e:
                    st.error(f"Errore salvataggio su Google Sheets: {e}")


    st.divider()
    st.subheader("File Excel 12 mesi — caricamento e lettura")

    up = st.file_uploader("Carica il file Excel annuale (.xlsx)", type=["xlsx"])
    if up is None:
        return

    try:
        xls = pd.read_excel(up, sheet_name=None)
        if "Anno" in xls:
            df = xls["Anno"].copy()
        else:
            first_sheet = next(iter(xls))
            df = xls[first_sheet].copy()

        prod_col = None
        for c in df.columns:
            cs = str(c)
            if "Energia per inverter" in cs or cs.strip().lower() == "produzione totale":
                prod_col = c; break
        if prod_col is None:
            st.error("Colonna produzione non trovata.")
            return

        if "Data e ora" not in df.columns:
            st.error("Colonna 'Data e ora' non trovata.")
            return
        df["Data e ora"] = pd.to_datetime(df["Data e ora"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Data e ora"])

        for col in [prod_col, "Consumo totale", "Autoconsumo", "Energia alimentata nella rete", "Energia prelevata"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0
        df["Produzione_kWh"]     = df[prod_col] / 1000.0
        df["Consumo_kWh"]        = df["Consumo totale"] / 1000.0
        df["Autoconsumo_kWh"]    = df["Autoconsumo"] / 1000.0
        df["Rete_immessa_kWh"]   = df["Energia alimentata nella rete"] / 1000.0
        df["Rete_prelevata_kWh"] = df["Energia prelevata"] / 1000.0

        df["mese"] = df["Data e ora"].dt.to_period("M")
        agg = (df.groupby("mese")[["Produzione_kWh","Consumo_kWh","Autoconsumo_kWh","Rete_immessa_kWh","Rete_prelevata_kWh"]]
                 .sum().reset_index())
        agg["mese"] = agg["mese"].astype(str)
        agg["Mese"] = agg["mese"].apply(lambda s: f"{s.split('-')[1]}-{s.split('-')[0]}")
        agg = agg.sort_values("mese").tail(12).reset_index(drop=True)

        show = agg.rename(columns={
            "Produzione_kWh":"Produzione (kWh)",
            "Consumo_kWh":"Consumo (kWh)",
            "Autoconsumo_kWh":"Autoconsumo (kWh)",
            "Rete_immessa_kWh":"Rete immessa (kWh)",
            "Rete_prelevata_kWh":"Rete prelevata (kWh)"
        })[["Mese","Produzione (kWh)","Consumo (kWh)","Autoconsumo (kWh)","Rete immessa (kWh)","Rete prelevata (kWh)"]]

        # colonne numeriche
        num_cols = [
            "Produzione (kWh)",
            "Consumo (kWh)",
            "Autoconsumo (kWh)",
            "Rete immessa (kWh)",
            "Rete prelevata (kWh)"
        ]
        for c in num_cols:
            show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0.0)

        # --- calcolo atteso_last da GS (province_coeff) ---
        month_labels = show["Mese"].tolist()
        prod_values  = show["Produzione (kWh)"].astype(float).tolist()
        mese_corrente = month_labels[-1] if month_labels else "MM-YYYY"
        prod_last = float(prod_values[-1]) if prod_values else 0.0

        atteso_last = 0.0
        try:
            if selected:
                row_sel = st.session_state.anag_df[st.session_state.anag_df["denominazione"] == selected].iloc[0]
                prov_sigla = str(row_sel.get("provincia","")).strip().upper()
                potenza_sel = float(row_sel.get("potenza_kw", 0) or 0)
                if prov_sigla and potenza_sel > 0:
                    coeff_df = load_coeff_gs()
                    last_mm = mese_corrente.split("-")[0] if month_labels else None  # "MM"
                    atteso_last = atteso_for_last_month(prov_sigla, potenza_sel, last_mm, coeff_df)

                    # applica derating percentuale se presente
                    der = float(row_sel.get("derating_percent", 0) or 0)
                    if 0 < der <= 99 and atteso_last > 0:
                        atteso_last = atteso_last * (1 - der / 100.0)
                    
        except Exception as e:
            st.warning(f"Valore atteso non calcolabile: {e}")

        # --- classificazione soglie richieste ---
        if atteso_last > 0:
            delta = (prod_last - atteso_last) / atteso_last
            if delta >= -0.10:
                last_class = "verde"
            elif -0.20 <= delta < -0.10:
                last_class = "arancione"
            else:
                last_class = "rosso"
        else:
            last_class = "verde"

        # Grafico
        img_bytes = build_monthly_chart(month_labels, prod_values,
                                        atteso_last if atteso_last > 0 else None,
                                        last_class)

        # Tabella
        table_rows = []
        for i, r in show.iterrows():
            p    = float(r["Produzione (kWh)"])
            cons = float(r["Consumo (kWh)"])
            aut  = float(r["Autoconsumo (kWh)"])
            imm  = float(r["Rete immessa (kWh)"])
            prel = float(r["Rete prelevata (kWh)"])
            if i == len(show) - 1 and atteso_last > 0:
                scost = f"{(p/atteso_last - 1)*100:.1f}%"
                att   = f"{atteso_last:.1f}"
            else:
                scost, att = "", ""
            table_rows.append([r["Mese"], f"{p:.1f}", f"{cons:.1f}", f"{aut:.1f}",
                               f"{imm:.1f}", f"{prel:.1f}", att, scost])

        # Dati anagrafici nel PDF
        if selected:
            row = anag[anag["denominazione"] == selected].iloc[0]
            anag_dict = {
                "Denominazione": str(row.get("denominazione","")),
                "Indirizzo": str(row.get("indirizzo","")),
                "Provincia": str(row.get("provincia","")),
                "Potenza (kWh)": str(row.get("potenza_kw","")),
                "Derating (%)": str(row.get("derating_percent", 0)),
                "Data installazione": str(row.get("data_installazione","")),
            }

            denom_safe = str(row.get("denominazione","")).replace(" ", "")
        else:
            anag_dict = {"Denominazione":"","Indirizzo":"","Provincia":"","Potenza":"","Data installazione":""}
            denom_safe = "Cliente"

        mesi_it = {"01":"Gennaio","02":"Febbraio","03":"Marzo","04":"Aprile","05":"Maggio","06":"Giugno",
                   "07":"Luglio","08":"Agosto","09":"Settembre","10":"Ottobre","11":"Novembre","12":"Dicembre"}
        mm, yyyy = (mese_corrente.split("-") + ["",""])[:2]
        titolo_esteso = f"{mesi_it.get(mm, mm)} {yyyy}"

        # PDF
        pdf_buf = BytesIO()
        compose_pdf(
            path_out=pdf_buf,
            logo_path="assets/logo.jpg",
            title_mmYYYY=titolo_esteso,
            anag_dict=anag_dict,
            table_rows=table_rows,
            last_class=last_class,
            chart_img=ImageReader(BytesIO(img_bytes))
        )
        pdf_data = pdf_buf.getvalue()
        st.download_button("Scarica PDF",
                           data=pdf_data,
                           file_name=f"Report_{denom_safe}_{mese_corrente}.pdf",
                           mime="application/pdf")

    except Exception as e:
        st.error(f"Errore lettura Excel: {e}")

if __name__ == "__main__":
    if check_password():
        main()
