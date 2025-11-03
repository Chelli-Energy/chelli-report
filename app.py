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
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
import ssl
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader
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
import smtplib
from email.message import EmailMessage
from email.utils import formataddr


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
ANAG_COLS = ["denominazione","indirizzo","provincia","potenza_kw","data_installazione","derating_percent","email"]

def gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    info_raw = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    if isinstance(info_raw, dict):
        info = dict(info_raw)
    else:
        import ast
        info = json.loads(info_raw) if isinstance(info_raw, str) and info_raw.strip().startswith("{") else ast.literal_eval(info_raw)
    pk = info.get("private_key", "")
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    info["private_key"] = pk
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets["GSHEET_ID"])

def sheet_to_df(ws):
    return pd.DataFrame(ws.get_all_records())

def df_append_row(ws, row_dict):
    headers = ws.row_values(1)
    if not headers:
        ws.append_row(list(row_dict.keys()))
        headers = ws.row_values(1)
    values = [row_dict.get(h, "") for h in headers]
    ws.append_row(values)

def load_anagrafica_gs():
    """Legge il foglio 'anagrafica' e normalizza la provincia a sigla + derating."""
    try:
        sh = gs_client(); ws = sh.worksheet("anagrafica")
        df = sheet_to_df(ws)
        for c in ANAG_COLS:
            if c not in df.columns: df[c] = ""
        df["provincia"] = (
            df["provincia"].astype(str).str.strip().str.upper()
              .map(lambda x: PROVINCE_MAP.get(x, x))
        )
        if "derating_percent" not in df.columns: df["derating_percent"] = ""
        df["derating_percent"] = pd.to_numeric(df["derating_percent"], errors="coerce").clip(0,99).fillna(0).astype(int)
        return df[ANAG_COLS]
    except Exception as e:
        st.error(f"Errore lettura anagrafica (GS): {e}")
        return pd.DataFrame(columns=ANAG_COLS)

def append_anagrafica_gs(row: dict):
    sh = gs_client(); ws = sh.worksheet("anagrafica")
    df_append_row(ws, row)

def load_coeff_gs():
    """Legge 'province_coeff' e normalizza i numeri (virgola/punto, migliaia)."""
    sh = gs_client(); ws = sh.worksheet("province_coeff")
    df = sheet_to_df(ws)
    df["provincia"] = (
        df["provincia"].astype(str).str.strip().str.upper()
          .map(lambda x: PROVINCE_MAP.get(x, x))
    )
    for col in ["gennaio","febbraio","marzo","aprile","maggio","giugno",
                "luglio","agosto","settembre","ottobre","novembre","dicembre"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)
            mask_both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
            s = s.where(~mask_both, s.str.replace(".", "", regex=False))
            s = s.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
            df.loc[df[col] > 1000, col] = df.loc[df[col] > 1000, col] / 100.0
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

def build_monthly_chart(month_labels, prod_values, atteso_last=None, last_ok_class="verde", atteso_label=None):
    """Ritorna bytes PNG del grafico a barre."""
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    x = np.arange(len(month_labels))
    widths = np.array([0.5]*len(month_labels))
    colors_bars = [GRAY_PRIOR]*len(month_labels)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 2.50, h * 2.20)
    base = plt.rcParams.get("font.size", 10)
    scale_factor = max(2.5, 2.2)  # i tuoi fattori
    bump = base * scale_factor * 0.9

    base = plt.rcParams.get("font.size", 10)
    scale_factor = max(2.5, 2.2)  # lo stesso usato per la figura
    bump = base * scale_factor * 0.9  # leggermente ridotto per equilibrio
    ax.tick_params(labelsize=bump)
    ax.xaxis.label.set_fontsize(bump)
    ax.yaxis.label.set_fontsize(bump)
    if ax.title: ax.title.set_fontsize(bump)
    # Se usi legende:
    leg = ax.get_legend()
    if leg: 
        for t in leg.get_texts():
            t.set_fontsize(bump)
    # Se usi etichette sulle barre (bar_label):
    try:
        # etichette sopra barre
        for c in ax.containers:
            lbls = ax.bar_label(c, padding=4)
            for t in lbls:
                t.set_fontsize(bump * 1.4)
    
        # etichette asse X
        for lab in ax.get_xticklabels():
            lab.set_fontsize(bump * 1.3)
    
        # testo della linea "standard ..."
        for t in ax.texts:
            if "standard" in t.get_text().lower():
                t.set_fontsize(bump * 1.4)
    except Exception:
        pass
    


    last_color = GREEN_MAIN if last_ok_class=="verde" else (ORANGE if last_ok_class=="arancione" else RED)
    if colors_bars:
        colors_bars[-1] = last_color
        widths[-1] = 0.7

    bars = ax.bar(x, prod_values, width=widths, color=colors_bars, edgecolor="none")
    ax.set_ylabel("Produzione (kWh)")
    ax.set_xticks(x, month_labels, rotation=45, ha="right", fontsize=16)
    ymax = max(prod_values) if prod_values else 1
    ax.set_ylim(0, ymax*1.2)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + ymax*0.012, f"{h:.0f}", ha='center', va='bottom', fontsize=14)

    if atteso_last is not None and len(bars) > 0:
        last_bar = bars[-1]
        bx, bw = last_bar.get_x(), last_bar.get_width()
        ax.hlines(y=atteso_last, xmin=bx, xmax=bx+bw, colors="#333333", linewidth=1.2, linestyles=(0,(2,2)))
        label_txt = atteso_label if atteso_label else "standard mese × kW"
        ax.text(bx + bw + 0.15, atteso_last, label_txt, ha="left", va="center",
                fontsize=16, color="#333333", backgroundcolor="white")


    buf = BytesIO()
    plt.subplots_adjust(bottom=0.24)
    plt.tight_layout()
    fig.savefig(buf, format="PNG", dpi=300)
    # ingrandisci etichette barre, testo linea "standard" e ogni altro testo
    for c in ax.containers:
        for t in c.datavalues if hasattr(c, "datavalues") else []:
            pass  # no-op, serve solo a non rompere vecchie versioni
    try:
        for c in ax.containers:
            lbls = ax.bar_label(c, padding=4)  # ricrea labels se non esistono
            for t in lbls:
                t.set_fontsize(bump * 1.6)
    except Exception:
        pass
    
    # etichette asse X
    for lab in ax.get_xticklabels():
        lab.set_fontsize(bump * 1.5)
    
    # testo “standard …” e qualsiasi altro testo
    for t in ax.texts:
        t.set_fontsize(bump * 1.6)

    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def compose_pdf(path_out, logo_path, title_mmYYYY, anag_dict, table_rows, last_class, chart_img):
    """Crea PDF A4 completo."""
    c = canvas.Canvas(path_out, pagesize=A4)
    page_w, page_h = A4
    LM, RM, TM, BM = 1.0*cm, 1.0*cm, 2*cm, 2*cm  # più larghezza utile

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
    for label in ["Denominazione","Indirizzo","Provincia","Potenza (kWp)","Data installazione"]:
        c.drawString(LM, y, f"{label}:")
        c.drawString(LM + 4.5*cm, y, str(anag_dict.get(label, '')))
        y -= 0.42*cm

    # Grafico
    chart_w = page_w - LM - RM; chart_h = 9.0*cm
    chart_y = y - 0.7*cm - chart_h
    c.drawImage(chart_img, LM, chart_y, width=chart_w, height=chart_h, preserveAspectRatio=True, mask='auto')

    # Tabella + Caption con posizionamento sicuro
    hdr = ["Mese","Produzione kWh","Consumo kWh","Autoconsumo kWh","Rete immessa kWh","Rete prelevata kWh","Atteso kWh*","Scost. %"]
    data_table = [hdr] + table_rows
    
    content_w = page_w - LM - RM
    fractions = [0.11, 0.14, 0.14, 0.15, 0.15, 0.15, 0.09, 0.07]
    col_widths = [f * content_w for f in fractions]
    
    tbl = Table(data_table, colWidths=col_widths)
    ts = TableStyle([
        ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 7),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(TXT_DARK)),
        ('FONT', (0,1), (-1,-1), 'Helvetica', 8),
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
    
    # misura tabella e posiziona sopra al margine
    w, h = tbl.wrapOn(c, content_w, page_h)
    table_y = max(BM + 2.2*cm, chart_y - 0.8*cm - h)
    tbl.drawOn(c, (page_w - w)/2, table_y)
    
    # caption sotto tabella con guardia
    cap1_y = table_y - 1.0*cm
    cap2_y = table_y - 1.8*cm
    min_y  = BM + 0.9*cm
    if cap2_y < min_y:
        shift = (min_y - cap2_y)
        table_y += shift; cap1_y += shift; cap2_y += shift
    
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor(TXT_BASE))
    c.drawString(LM, cap1_y, "*Valore di produzione medio mensile atteso")
    
    c.setFont("Helvetica-Bold", 10)
    msg = {
        "verde": "Risultato buono: produzione del mese in linea alla media attesa.",
        "arancione": "Risultato inferiore agli standard: produzione del mese leggermente sotto la media attesa.",
        "rosso": "Risultato non sufficiente: produzione del mese sensibilmente sotto la media attesa.",
    }[last_class]
    c.drawString(LM, cap2_y, msg)


    # Footer
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#616161"))
    c.drawCentredString(page_w/2, 1.5*cm, "Chelli Energy Solutions — Report automatico")

    c.showPage(); c.save()

ITALIAN_MONTHS = {
    1:"Gennaio",2:"Febbraio",3:"Marzo",4:"Aprile",5:"Maggio",6:"Giugno",
    7:"Luglio",8:"Agosto",9:"Settembre",10:"Ottobre",11:"Novembre",12:"Dicembre"
}

def subject_for_last_month(today=None):
    from datetime import datetime
    today = today or datetime.now()
    y = today.year
    m = today.month - 1 or 12
    if today.month == 1:
        y -= 1
    return f"Report produzione fotovoltaica — {ITALIAN_MONTHS[m]} {y}"

def send_pdf_via_email(pdf_bytes: bytes, filename: str, to_email: str):
    import streamlit as st
    s = st.secrets["EMAIL"]
    msg = EmailMessage()
    msg["From"] = formataddr(("Chelli Report", s["SMTP_USER"]))
    msg["To"] = f"{to_email}, assistenza@chellienergysolutions.it"
    msg["Cc"] = ""
    msg["Reply-To"] = "assistenza@chellienergysolutions.it"
    msg["Subject"] = subject_for_last_month()
    html_body = """
    <html>
      <body style="font-family:Arial,sans-serif; color:#222;">
        <p>Gentile cliente;
trasmettiamo in allegato il report mensile del suo impianto fotovoltaico contenente i dati che abbiamo registrato. Può contattare il nostro servizio tecnico per approfondimenti e informazioni in merito.
 
Cordiali saluti.</p>
        <hr style="border:0; border-top:1px solid #ccc; margin:20px 0;" />
        <img src="https://chellienergysolutions.it/wp-content/uploads/2025/09/logo-per-mail-form.jpg"
             alt="Logo Chelli Energy Solution"
             width="140"
             style="display:block; margin-bottom:10px;">

        <p style="font-size:16px; font-weight:bold; margin:0;">
          Chelli Energy Solution
        </p>
        <p style="font-size:14px; margin:2px 0;">
          via Lisbona, 37<br>
          50065 Pontassieve (FI) Italy<br>
          <a href="https://www.chellienergysolutions.it" style="color:#1155cc; text-decoration:none;">
            www.chellienergysolutions.it
          </a>
        </p>
    
        <p style="font-size:14px; font-weight:bold; margin:12px 0 0 0;">
          Assistenza tecnica
        </p>
        <p style="font-size:14px; margin:2px 0;">
          Mobile: +39 347 399 9592<br>
          Tel: 055 8323264
        </p>
    
        <p style="font-size:8px; color:#666; margin-top:18px; line-height:1.3;">
          Questo documento è formato esclusivamente per il destinatario. Tutte le informazioni ivi contenute,
          compresi eventuali allegati, sono da ritenere esclusivamente confidenziali e riservate secondo i termini
          del vigente D.Lgs. 196/2003 in materia di privacy e del Regolamento europeo 679/2016 – GDPR – e quindi ne
          è proibita l’utilizzazione ulteriore non autorizzata. Se avete ricevuto per errore questo messaggio,
          Vi preghiamo cortesemente di contattare immediatamente il mittente e cancellare la e-mail. Grazie.<br><br>
          Confidentiality Notice – This e-mail message including any attachments is for the sole use of the intended
          recipient and may contain confidential and privileged information pursuant to Legislative Decree 196/2003
          and the European General Data Protection Regulation 679/2016 – GDPR –. Any unauthorized review, use,
          disclosure or distribution is prohibited. If you are not the intended recipient, please contact the sender
          by reply e-mail and destroy all copies of the original message.
        </p>
      </body>
    </html>
    """

    msg.set_content("In allegato il report mensile in PDF.")  # fallback testo
    msg.add_alternative(html_body, subtype="html")
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(s["SMTP_SERVER"], s["SMTP_PORT"], context=context) as server:
        server.login(s["SMTP_USER"], s["SMTP_PASS"])
        server.send_message(msg)

    def send_email_gmail_cid(pdf_bytes: bytes, pdf_filename: str, to_addr: str, cc_addr: str | None, subject: str, html_body: str):
        user = st.secrets.get("SMTP_USER") or st.secrets.get("EMAIL_USER")
        pwd  = st.secrets.get("SMTP_PASS") or st.secrets["APP_PASSWORD"]
        if not user:
            raise ValueError("SMTP_USER mancante nei secrets")
    
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        if cc_addr:
            msg["Cc"] = cc_addr
    
        alt = MIMEMultipart("alternative")
        msg.attach(alt)
        alt.attach(MIMEText(html_body, "html", "utf-8"))
    
        with open("assets/logo.jpg", "rb") as f:
            img = MIMEImage(f.read(), name="logo.jpg")
        img.add_header("Content-ID", "<logo>")
        img.add_header("Content-Disposition", "inline", filename="logo.jpg")
        msg.attach(img)
    
        pdf = MIMEApplication(pdf_bytes, _subtype="pdf")
        pdf.add_header("Content-Disposition", "attachment", filename=pdf_filename)
        msg.attach(pdf)
    
        rcpts = [to_addr] + ([cc_addr] if cc_addr else [])
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo()
            s.starttls()
            s.login(user, pwd)
            s.sendmail(user, rcpts, msg.as_string())


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
        selected = st.selectbox("Seleziona cliente", options=denoms, index=0, label_visibility="collapsed")
        if selected == "-- Seleziona cliente --":
            st.info("Seleziona dall’elenco o aggiungi un nuovo cliente per procedere.")
            selected = None

        if selected:
            row = anag[anag["denominazione"] == selected].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Indirizzo:** {row.get('indirizzo','')}")
                st.markdown(f"**Provincia:** {row.get('provincia','')}")
                st.markdown(f"**Email:** {row.get('email','')}")
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
            email = st.text_input("Email*", "")
        with c2:
            potenza_kw = st.number_input("Potenza (kW)*", min_value=0.1, step=0.1, value=5.0)
            data_installazione = st.date_input("Data installazione*", value=date.today())
            derating_percent = st.number_input("Derating impianto (%)", min_value=0, max_value=99, value=0, step=1)
        submitted = st.form_submit_button("Aggiungi all’elenco")
        if submitted:
            if not denominazione or not indirizzo or not provincia or not email:
                st.error("Compila i campi obbligatori contrassegnati con *")
            else:
                new_row = {
                    "denominazione": denominazione.strip(),
                    "indirizzo": indirizzo.strip(),
                    "provincia": PROVINCE_MAP.get(provincia.strip().upper(), provincia.strip().upper()),
                    "potenza_kw": float(potenza_kw),
                    "data_installazione": data_installazione.strftime("%d/%m/%Y"),
                    "derating_percent": int(derating_percent),
                    "email": email.strip(),
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
        # Lettura Excel
        xls = pd.read_excel(up, sheet_name=None)
        df = xls["Anno"].copy() if "Anno" in xls else xls[next(iter(xls))].copy()

        # Colonne base
        prod_col = None
        for c in df.columns:
            cs = str(c)
            if "Energia per inverter" in cs or cs.strip().lower() == "produzione totale":
                prod_col = c; break
        if prod_col is None:
            st.error("Colonna produzione non trovata."); return
        if "Data e ora" not in df.columns:
            st.error("Colonna 'Data e ora' non trovata."); return

        df["Data e ora"] = pd.to_datetime(df["Data e ora"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Data e ora"])

        for col in [prod_col, "Consumo totale", "Autoconsumo", "Energia alimentata nella rete", "Energia prelevata"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
        df["Produzione_kWh"]     = df[prod_col] / 1000.0
        df["Consumo_kWh"]        = df.get("Consumo totale", 0) / 1000.0
        df["Autoconsumo_kWh"]    = df.get("Autoconsumo", 0) / 1000.0
        df["Rete_immessa_kWh"]   = df.get("Energia alimentata nella rete", 0) / 1000.0
        df["Rete_prelevata_kWh"] = df.get("Energia prelevata", 0) / 1000.0

        # Aggregazione mensile
        import re
        
        def _to_num(x):
            s = str(x).strip().replace("−", "-")  # meno Unicode
            if s in ("", "-", "nan", "None", "NULL"): 
                return 0.0
            # togli caratteri non numerici comuni (spazi, NBSP, tab)
            s = re.sub(r"[^\d,.\-]", "", s)
            # se c'è la virgola, tratta la virgola come decimale ed elimina i punti migliaia
            if "," in s:
                s = s.replace(".", "").replace(",", ".")
            return pd.to_numeric(s, errors="coerce")
        
        for col in ["Produzione_kWh","Consumo_kWh","Autoconsumo_kWh","Rete_immessa_kWh","Rete_prelevata_kWh"]:
            df[col] = df[col].map(_to_num).fillna(0.0).astype(float)
        
        df["mese"] = df["Data e ora"].dt.to_period("M")
        # Normalizza colonne rete: scegli la prima disponibile con dati
        def _pick_col(cands):
            for c in cands:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                    if s.sum() > 0:
                        return s.astype(float)
            return pd.Series(0.0, index=df.index)
        
        # Sovrascrivi le colonne normalizzate usate dal report
        df["Rete_prelevata_kWh"] = _pick_col(["Energia prelevata dalla rete","Energia prelevata","Rete_prelevata_kWh"])
        df["Rete_immessa_kWh"]   = _pick_col(["Energia alimentata nella rete","Rete_immessa_kWh"])

        agg = (
            df.groupby("mese")[["Produzione_kWh","Consumo_kWh","Autoconsumo_kWh","Rete_immessa_kWh","Rete_prelevata_kWh"]]
              .sum()
              .reset_index()
        )
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

        # Numeriche safe
        for c in ["Produzione (kWh)","Consumo (kWh)","Autoconsumo (kWh)","Rete immessa (kWh)","Rete prelevata (kWh)"]:
            show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0.0)

        # Label e mese corrente
        month_labels = show["Mese"].tolist()
        prod_values  = show["Produzione (kWh)"].astype(float).tolist()
        mese_corrente = month_labels[-1] if month_labels else "MM-YYYY"
        prod_last = float(prod_values[-1]) if prod_values else 0.0
        last_mm = mese_corrente.split("-")[0] if month_labels else None

        # Atteso dal foglio province_coeff
        atteso_last = 0.0
        atteso_label = None
        if selected and last_mm:
            row_sel = st.session_state.anag_df[st.session_state.anag_df["denominazione"] == selected].iloc[0]
            prov_sigla = str(row_sel.get("provincia","")).strip().upper()
            potenza_sel = float(row_sel.get("potenza_kw", 0) or 0)
            if prov_sigla and potenza_sel > 0:
                coeff_df = load_coeff_gs()
                atteso_last = atteso_for_last_month(prov_sigla, potenza_sel, last_mm, coeff_df)
                der = float(row_sel.get("derating_percent", 0) or 0)
                if 0 < der <= 99 and atteso_last > 0:
                    atteso_last *= (1 - der / 100.0)
                mesi_it = {"01":"Gennaio","02":"Febbraio","03":"Marzo","04":"Aprile","05":"Maggio","06":"Giugno",
                           "07":"Luglio","08":"Agosto","09":"Settembre","10":"Ottobre","11":"Novembre","12":"Dicembre"}
                atteso_label = f"standard {mesi_it.get(last_mm,last_mm)} ({prov_sigla}) × {potenza_sel:.1f} kW = {atteso_last:.1f} kWh"

        # Classificazione
        if atteso_last > 0:
            delta = (prod_last - atteso_last) / atteso_last
            if delta >= -0.10:        last_class = "verde"
            elif delta >= -0.20:      last_class = "arancione"
            else:                     last_class = "rosso"
        else:
            last_class = "verde"

        # Grafico
        img_bytes = build_monthly_chart(month_labels, prod_values,
                                        atteso_last if atteso_last > 0 else None,
                                        last_class,
                                        atteso_label=atteso_label)

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
                "Potenza (kWp)": str(row.get("potenza_kw","")),
                "Data installazione": str(row.get("data_installazione","")),
            }
            denom_safe = str(row.get("denominazione","")).replace(" ", "")
        else:
            anag_dict = {"Denominazione":"","Indirizzo":"","Provincia":"","Potenza (kWp)":"","Data installazione":""}
            denom_safe = "Cliente"

        # Titolo PDF
        mm, yyyy = (mese_corrente.split("-") + ["",""])[:2]
        mesi_it = {"01":"Gennaio","02":"Febbraio","03":"Marzo","04":"Aprile","05":"Maggio","06":"Giugno",
                   "07":"Luglio","08":"Agosto","09":"Settembre","10":"Ottobre","11":"Novembre","12":"Dicembre"}
        titolo_esteso = f"{mesi_it.get(mm, mm)} {yyyy}"

        # PDF
        st.subheader("PDF")
        pdf_buf = BytesIO()
        compose_pdf(path_out=pdf_buf, logo_path="assets/logo.jpg", title_mmYYYY=titolo_esteso,
                    anag_dict=anag_dict, table_rows=table_rows, last_class=last_class,
                    chart_img=ImageReader(BytesIO(img_bytes)))
        pdf_name = f"Report_{denom_safe}_{mese_corrente}.pdf"
        st.download_button("Scarica PDF",
                           data=pdf_buf.getvalue(),
                           file_name=f"Report_{denom_safe}_{mese_corrente}.pdf",
                           mime="application/pdf",
                           key="download_pdf")

                # --- Spedisci PDF ---
        st.subheader("Spedisci PDF")
        email_to = st.text_input("Email destinatario", value="", key="mail_to")
        email_cc = st.text_input("CC (opzionale)", value="", key="mail_cc")
        default_subj = f"Report produzione fotovoltaica {titolo_esteso} — {denom_safe}"
        email_subj = st.text_input("Oggetto", value=default_subj, key="mail_subj")

        if st.button("Invia PDF", key="send_pdf"):
            if not email_to:
                st.error("Inserisci l'email del destinatario.")
            else:
                body_html = f"""
                <p>Trasmettiamo in allegato il report mensile del suo impianto fotovoltaico.</p>
                <hr style="border:0; border-top:1px solid #ccc; margin:20px 0;" />
                <img src="cid:logo" alt="Logo Chelli Energy Solution" width="140" style="display:block; margin-bottom:10px;">
                <p style="font-size:16px; font-weight:bold; margin:0;">Chelli Energy Solution</p>
                <p style="font-size:14px; margin:2px 0;">
                  via Lisbona, 37<br>50065 Pontassieve (FI) Italy<br>
                  <a href="https://www.chellienergysolutions.it" style="color:#1155cc; text-decoration:none;">www.chellienergysolutions.it</a>
                </p>
                <p style="font-size:14px; font-weight:bold; margin:12px 0 0 0;">Assistenza tecnica</p>
                <p style="font-size:14px; margin:2px 0;">
                  Mobile: +39 347 399 9592<br>Tel: 055 8323264
                </p>
                """
                try:
                    send_email_gmail_cid(
                        pdf_bytes=pdf_buf.getvalue(),
                        pdf_filename=f"Report_{denom_safe}_{mese_corrente}.pdf",
                        to_addr=email_to.strip(),
                        cc_addr=email_cc.strip() or None,
                        subject=email_subj.strip(),
                        html_body=body_html,
                    )
                    st.success("Email inviata.")
                except Exception as e:
                    st.error(f"Errore invio email: {e}")


    except Exception as e:
        st.error(f"Errore lettura Excel: {e}")

if __name__ == "__main__":
    if check_password():
        main()


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
ANAG_COLS = ["denominazione","indirizzo","provincia","potenza_kw","data_installazione","derating_percent","email"]

def gs_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    info_raw = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    if isinstance(info_raw, dict):
        info = dict(info_raw)
    else:
        import ast
        info = json.loads(info_raw) if isinstance(info_raw, str) and info_raw.strip().startswith("{") else ast.literal_eval(info_raw)
    pk = info.get("private_key", "")
    if "\\n" in pk:
        pk = pk.replace("\\n", "\n")
    info["private_key"] = pk
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(st.secrets["GSHEET_ID"])

def sheet_to_df(ws):
    return pd.DataFrame(ws.get_all_records())

def df_append_row(ws, row_dict):
    headers = ws.row_values(1)
    if not headers:
        ws.append_row(list(row_dict.keys()))
        headers = ws.row_values(1)
    values = [row_dict.get(h, "") for h in headers]
    ws.append_row(values)

def load_anagrafica_gs():
    """Legge il foglio 'anagrafica' e normalizza la provincia a sigla + derating."""
    try:
        sh = gs_client(); ws = sh.worksheet("anagrafica")
        df = sheet_to_df(ws)
        for c in ANAG_COLS:
            if c not in df.columns: df[c] = ""
        df["provincia"] = (
            df["provincia"].astype(str).str.strip().str.upper()
              .map(lambda x: PROVINCE_MAP.get(x, x))
        )
        if "derating_percent" not in df.columns: df["derating_percent"] = ""
        df["derating_percent"] = pd.to_numeric(df["derating_percent"], errors="coerce").clip(0,99).fillna(0).astype(int)
        return df[ANAG_COLS]
    except Exception as e:
        st.error(f"Errore lettura anagrafica (GS): {e}")
        return pd.DataFrame(columns=ANAG_COLS)

def append_anagrafica_gs(row: dict):
    sh = gs_client(); ws = sh.worksheet("anagrafica")
    df_append_row(ws, row)

def load_coeff_gs():
    """Legge 'province_coeff' e normalizza i numeri (virgola/punto, migliaia)."""
    sh = gs_client(); ws = sh.worksheet("province_coeff")
    df = sheet_to_df(ws)
    df["provincia"] = (
        df["provincia"].astype(str).str.strip().str.upper()
          .map(lambda x: PROVINCE_MAP.get(x, x))
    )
    for col in ["gennaio","febbraio","marzo","aprile","maggio","giugno",
                "luglio","agosto","settembre","ottobre","novembre","dicembre"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)
            mask_both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
            s = s.where(~mask_both, s.str.replace(".", "", regex=False))
            s = s.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce").fillna(0.0)
            df.loc[df[col] > 1000, col] = df.loc[df[col] > 1000, col] / 100.0
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

def build_monthly_chart(month_labels, prod_values, atteso_last=None, last_ok_class="verde", atteso_label=None):
    """Ritorna bytes PNG del grafico a barre."""
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    x = np.arange(len(month_labels))
    widths = np.array([0.5]*len(month_labels))
    colors_bars = [GRAY_PRIOR]*len(month_labels)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 2.50, h * 2.20)
    base = plt.rcParams.get("font.size", 10)
    scale_factor = max(2.5, 2.2)  # i tuoi fattori
    bump = base * scale_factor * 0.9

    base = plt.rcParams.get("font.size", 10)
    scale_factor = max(2.5, 2.2)  # lo stesso usato per la figura
    bump = base * scale_factor * 0.9  # leggermente ridotto per equilibrio
    ax.tick_params(labelsize=bump)
    ax.xaxis.label.set_fontsize(bump)
    ax.yaxis.label.set_fontsize(bump)
    if ax.title: ax.title.set_fontsize(bump)
    # Se usi legende:
    leg = ax.get_legend()
    if leg: 
        for t in leg.get_texts():
            t.set_fontsize(bump)
    # Se usi etichette sulle barre (bar_label):
    try:
        # etichette sopra barre
        for c in ax.containers:
            lbls = ax.bar_label(c, padding=4)
            for t in lbls:
                t.set_fontsize(bump * 1.4)
    
        # etichette asse X
        for lab in ax.get_xticklabels():
            lab.set_fontsize(bump * 1.3)
    
        # testo della linea "standard ..."
        for t in ax.texts:
            if "standard" in t.get_text().lower():
                t.set_fontsize(bump * 1.4)
    except Exception:
        pass
    


    last_color = GREEN_MAIN if last_ok_class=="verde" else (ORANGE if last_ok_class=="arancione" else RED)
    if colors_bars:
        colors_bars[-1] = last_color
        widths[-1] = 0.7

    bars = ax.bar(x, prod_values, width=widths, color=colors_bars, edgecolor="none")
    ax.set_ylabel("Produzione (kWh)")
    ax.set_xticks(x, month_labels, rotation=45, ha="right", fontsize=16)
    ymax = max(prod_values) if prod_values else 1
    ax.set_ylim(0, ymax*1.2)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + ymax*0.012, f"{h:.0f}", ha='center', va='bottom', fontsize=14)

    if atteso_last is not None and len(bars) > 0:
        last_bar = bars[-1]
        bx, bw = last_bar.get_x(), last_bar.get_width()
        ax.hlines(y=atteso_last, xmin=bx, xmax=bx+bw, colors="#333333", linewidth=1.2, linestyles=(0,(2,2)))
        label_txt = atteso_label if atteso_label else "standard mese × kW"
        ax.text(bx + bw + 0.15, atteso_last, label_txt, ha="left", va="center",
                fontsize=16, color="#333333", backgroundcolor="white")


    buf = BytesIO()
    plt.subplots_adjust(bottom=0.24)
    plt.tight_layout()
    fig.savefig(buf, format="PNG", dpi=300)
    # ingrandisci etichette barre, testo linea "standard" e ogni altro testo
    for c in ax.containers:
        for t in c.datavalues if hasattr(c, "datavalues") else []:
            pass  # no-op, serve solo a non rompere vecchie versioni
    try:
        for c in ax.containers:
            lbls = ax.bar_label(c, padding=4)  # ricrea labels se non esistono
            for t in lbls:
                t.set_fontsize(bump * 1.6)
    except Exception:
        pass
    
    # etichette asse X
    for lab in ax.get_xticklabels():
        lab.set_fontsize(bump * 1.5)
    
    # testo “standard …” e qualsiasi altro testo
    for t in ax.texts:
        t.set_fontsize(bump * 1.6)

    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def compose_pdf(path_out, logo_path, title_mmYYYY, anag_dict, table_rows, last_class, chart_img):
    """Crea PDF A4 completo."""
    c = canvas.Canvas(path_out, pagesize=A4)
    page_w, page_h = A4
    LM, RM, TM, BM = 1.0*cm, 1.0*cm, 2*cm, 2*cm  # più larghezza utile

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
    for label in ["Denominazione","Indirizzo","Provincia","Potenza (kWp)","Data installazione"]:
        c.drawString(LM, y, f"{label}:")
        c.drawString(LM + 4.5*cm, y, str(anag_dict.get(label, '')))
        y -= 0.42*cm

    # Grafico
    chart_w = page_w - LM - RM; chart_h = 9.0*cm
    chart_y = y - 0.7*cm - chart_h
    c.drawImage(chart_img, LM, chart_y, width=chart_w, height=chart_h, preserveAspectRatio=True, mask='auto')

    # Tabella + Caption con posizionamento sicuro
    hdr = ["Mese","Produzione kWh","Consumo kWh","Autoconsumo kWh","Rete immessa kWh","Rete prelevata kWh","Atteso kWh*","Scost. %"]
    data_table = [hdr] + table_rows
    
    content_w = page_w - LM - RM
    fractions = [0.11, 0.14, 0.14, 0.15, 0.15, 0.15, 0.09, 0.07]
    col_widths = [f * content_w for f in fractions]
    
    tbl = Table(data_table, colWidths=col_widths)
    ts = TableStyle([
        ('FONT', (0,0), (-1,0), 'Helvetica-Bold', 7),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor(TXT_DARK)),
        ('FONT', (0,1), (-1,-1), 'Helvetica', 8),
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
    
    # misura tabella e posiziona sopra al margine
    w, h = tbl.wrapOn(c, content_w, page_h)
    table_y = max(BM + 2.2*cm, chart_y - 0.8*cm - h)
    tbl.drawOn(c, (page_w - w)/2, table_y)
    
    # caption sotto tabella con guardia
    cap1_y = table_y - 1.0*cm
    cap2_y = table_y - 1.8*cm
    min_y  = BM + 0.9*cm
    if cap2_y < min_y:
        shift = (min_y - cap2_y)
        table_y += shift; cap1_y += shift; cap2_y += shift
    
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor(TXT_BASE))
    c.drawString(LM, cap1_y, "*Valore di produzione medio mensile atteso")
    
    c.setFont("Helvetica-Bold", 10)
    msg = {
        "verde": "Risultato buono: produzione del mese in linea alla media attesa.",
        "arancione": "Risultato inferiore agli standard: produzione del mese leggermente sotto la media attesa.",
        "rosso": "Risultato non sufficiente: produzione del mese sensibilmente sotto la media attesa.",
    }[last_class]
    c.drawString(LM, cap2_y, msg)


    # Footer
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#616161"))
    c.drawCentredString(page_w/2, 1.5*cm, "Chelli Energy Solutions — Report automatico")

    c.showPage(); c.save()

ITALIAN_MONTHS = {
    1:"Gennaio",2:"Febbraio",3:"Marzo",4:"Aprile",5:"Maggio",6:"Giugno",
    7:"Luglio",8:"Agosto",9:"Settembre",10:"Ottobre",11:"Novembre",12:"Dicembre"
}

def subject_for_last_month(today=None):
    from datetime import datetime
    today = today or datetime.now()
    y = today.year
    m = today.month - 1 or 12
    if today.month == 1:
        y -= 1
    return f"Report produzione fotovoltaica — {ITALIAN_MONTHS[m]} {y}"

def send_pdf_via_email(pdf_bytes: bytes, filename: str, to_email: str):
    import streamlit as st
    s = st.secrets["EMAIL"]
    msg = EmailMessage()
    msg["From"] = formataddr(("Chelli Report", s["SMTP_USER"]))
    msg["To"] = f"{to_email}, assistenza@chellienergysolutions.it"
    msg["Cc"] = ""
    msg["Reply-To"] = "assistenza@chellienergysolutions.it"
    msg["Subject"] = subject_for_last_month()
    html_body = """
    <html>
      <body style="font-family:Arial,sans-serif; color:#222;">
        <p>Gentile cliente;
trasmettiamo in allegato il report mensile del suo impianto fotovoltaico contenente i dati che abbiamo registrato. Può contattare il nostro servizio tecnico per approfondimenti e informazioni in merito.
 
Cordiali saluti.</p>
        <hr style="border:0; border-top:1px solid #ccc; margin:20px 0;" />
        <img src="https://chellienergysolutions.it/wp-content/uploads/2025/09/logo-per-mail-form.jpg"
             alt="Logo Chelli Energy Solution"
             width="140"
             style="display:block; margin-bottom:10px;">

        <p style="font-size:16px; font-weight:bold; margin:0;">
          Chelli Energy Solution
        </p>
        <p style="font-size:14px; margin:2px 0;">
          via Lisbona, 37<br>
          50065 Pontassieve (FI) Italy<br>
          <a href="https://www.chellienergysolutions.it" style="color:#1155cc; text-decoration:none;">
            www.chellienergysolutions.it
          </a>
        </p>
    
        <p style="font-size:14px; font-weight:bold; margin:12px 0 0 0;">
          Assistenza tecnica
        </p>
        <p style="font-size:14px; margin:2px 0;">
          Mobile: +39 347 399 9592<br>
          Tel: 055 8323264
        </p>
    
        <p style="font-size:8px; color:#666; margin-top:18px; line-height:1.3;">
          Questo documento è formato esclusivamente per il destinatario. Tutte le informazioni ivi contenute,
          compresi eventuali allegati, sono da ritenere esclusivamente confidenziali e riservate secondo i termini
          del vigente D.Lgs. 196/2003 in materia di privacy e del Regolamento europeo 679/2016 – GDPR – e quindi ne
          è proibita l’utilizzazione ulteriore non autorizzata. Se avete ricevuto per errore questo messaggio,
          Vi preghiamo cortesemente di contattare immediatamente il mittente e cancellare la e-mail. Grazie.<br><br>
          Confidentiality Notice – This e-mail message including any attachments is for the sole use of the intended
          recipient and may contain confidential and privileged information pursuant to Legislative Decree 196/2003
          and the European General Data Protection Regulation 679/2016 – GDPR –. Any unauthorized review, use,
          disclosure or distribution is prohibited. If you are not the intended recipient, please contact the sender
          by reply e-mail and destroy all copies of the original message.
        </p>
      </body>
    </html>
    """

    msg.set_content("In allegato il report mensile in PDF.")  # fallback testo
    msg.add_alternative(html_body, subtype="html")
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(s["SMTP_SERVER"], s["SMTP_PORT"], context=context) as server:
        server.login(s["SMTP_USER"], s["SMTP_PASS"])
        server.send_message(msg)


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
        selected = st.selectbox("Seleziona cliente", options=denoms, index=0, label_visibility="collapsed")
        if selected == "-- Seleziona cliente --":
            st.info("Seleziona dall’elenco o aggiungi un nuovo cliente per procedere.")
            selected = None

        if selected:
            row = anag[anag["denominazione"] == selected].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Indirizzo:** {row.get('indirizzo','')}")
                st.markdown(f"**Provincia:** {row.get('provincia','')}")
                st.markdown(f"**Email:** {row.get('email','')}")
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
            email = st.text_input("Email*", "")
        with c2:
            potenza_kw = st.number_input("Potenza (kW)*", min_value=0.1, step=0.1, value=5.0)
            data_installazione = st.date_input("Data installazione*", value=date.today())
            derating_percent = st.number_input("Derating impianto (%)", min_value=0, max_value=99, value=0, step=1)
        submitted = st.form_submit_button("Aggiungi all’elenco")
        if submitted:
            if not denominazione or not indirizzo or not provincia or not email:
                st.error("Compila i campi obbligatori contrassegnati con *")
            else:
                new_row = {
                    "denominazione": denominazione.strip(),
                    "indirizzo": indirizzo.strip(),
                    "provincia": PROVINCE_MAP.get(provincia.strip().upper(), provincia.strip().upper()),
                    "potenza_kw": float(potenza_kw),
                    "data_installazione": data_installazione.strftime("%d/%m/%Y"),
                    "derating_percent": int(derating_percent),
                    "email": email.strip(),
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
        # Lettura Excel
        xls = pd.read_excel(up, sheet_name=None)
        df = xls["Anno"].copy() if "Anno" in xls else xls[next(iter(xls))].copy()

        # Colonne base
        prod_col = None
        for c in df.columns:
            cs = str(c)
            if "Energia per inverter" in cs or cs.strip().lower() == "produzione totale":
                prod_col = c; break
        if prod_col is None:
            st.error("Colonna produzione non trovata."); return
        if "Data e ora" not in df.columns:
            st.error("Colonna 'Data e ora' non trovata."); return

        df["Data e ora"] = pd.to_datetime(df["Data e ora"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Data e ora"])

        for col in [prod_col, "Consumo totale", "Autoconsumo", "Energia alimentata nella rete", "Energia prelevata"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
        df["Produzione_kWh"]     = df[prod_col] / 1000.0
        df["Consumo_kWh"]        = df.get("Consumo totale", 0) / 1000.0
        df["Autoconsumo_kWh"]    = df.get("Autoconsumo", 0) / 1000.0
        df["Rete_immessa_kWh"]   = df.get("Energia alimentata nella rete", 0) / 1000.0
        df["Rete_prelevata_kWh"] = df.get("Energia prelevata", 0) / 1000.0

        # Aggregazione mensile
        import re
        
        def _to_num(x):
            s = str(x).strip().replace("−", "-")  # meno Unicode
            if s in ("", "-", "nan", "None", "NULL"): 
                return 0.0
            # togli caratteri non numerici comuni (spazi, NBSP, tab)
            s = re.sub(r"[^\d,.\-]", "", s)
            # se c'è la virgola, tratta la virgola come decimale ed elimina i punti migliaia
            if "," in s:
                s = s.replace(".", "").replace(",", ".")
            return pd.to_numeric(s, errors="coerce")
        
        for col in ["Produzione_kWh","Consumo_kWh","Autoconsumo_kWh","Rete_immessa_kWh","Rete_prelevata_kWh"]:
            df[col] = df[col].map(_to_num).fillna(0.0).astype(float)
        
        df["mese"] = df["Data e ora"].dt.to_period("M")
        # Normalizza colonne rete: scegli la prima disponibile con dati
        def _pick_col(cands):
            for c in cands:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce").fillna(0)
                    if s.sum() > 0:
                        return s.astype(float)
            return pd.Series(0.0, index=df.index)
        
        # Sovrascrivi le colonne normalizzate usate dal report
        df["Rete_prelevata_kWh"] = _pick_col(["Energia prelevata dalla rete","Energia prelevata","Rete_prelevata_kWh"])
        df["Rete_immessa_kWh"]   = _pick_col(["Energia alimentata nella rete","Rete_immessa_kWh"])

        agg = (
            df.groupby("mese")[["Produzione_kWh","Consumo_kWh","Autoconsumo_kWh","Rete_immessa_kWh","Rete_prelevata_kWh"]]
              .sum()
              .reset_index()
        )
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

        # Numeriche safe
        for c in ["Produzione (kWh)","Consumo (kWh)","Autoconsumo (kWh)","Rete immessa (kWh)","Rete prelevata (kWh)"]:
            show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0.0)

        # Label e mese corrente
        month_labels = show["Mese"].tolist()
        prod_values  = show["Produzione (kWh)"].astype(float).tolist()
        mese_corrente = month_labels[-1] if month_labels else "MM-YYYY"
        prod_last = float(prod_values[-1]) if prod_values else 0.0
        last_mm = mese_corrente.split("-")[0] if month_labels else None

        # Atteso dal foglio province_coeff
        atteso_last = 0.0
        atteso_label = None
        if selected and last_mm:
            row_sel = st.session_state.anag_df[st.session_state.anag_df["denominazione"] == selected].iloc[0]
            prov_sigla = str(row_sel.get("provincia","")).strip().upper()
            potenza_sel = float(row_sel.get("potenza_kw", 0) or 0)
            if prov_sigla and potenza_sel > 0:
                coeff_df = load_coeff_gs()
                atteso_last = atteso_for_last_month(prov_sigla, potenza_sel, last_mm, coeff_df)
                der = float(row_sel.get("derating_percent", 0) or 0)
                if 0 < der <= 99 and atteso_last > 0:
                    atteso_last *= (1 - der / 100.0)
                mesi_it = {"01":"Gennaio","02":"Febbraio","03":"Marzo","04":"Aprile","05":"Maggio","06":"Giugno",
                           "07":"Luglio","08":"Agosto","09":"Settembre","10":"Ottobre","11":"Novembre","12":"Dicembre"}
                atteso_label = f"standard {mesi_it.get(last_mm,last_mm)} ({prov_sigla}) × {potenza_sel:.1f} kW = {atteso_last:.1f} kWh"

        # Classificazione
        if atteso_last > 0:
            delta = (prod_last - atteso_last) / atteso_last
            if delta >= -0.10:        last_class = "verde"
            elif delta >= -0.20:      last_class = "arancione"
            else:                     last_class = "rosso"
        else:
            last_class = "verde"

        # Grafico
        img_bytes = build_monthly_chart(month_labels, prod_values,
                                        atteso_last if atteso_last > 0 else None,
                                        last_class,
                                        atteso_label=atteso_label)

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
                "Potenza (kWp)": str(row.get("potenza_kw","")),
                "Data installazione": str(row.get("data_installazione","")),
            }
            denom_safe = str(row.get("denominazione","")).replace(" ", "")
        else:
            anag_dict = {"Denominazione":"","Indirizzo":"","Provincia":"","Potenza (kWp)":"","Data installazione":""}
            denom_safe = "Cliente"

        # Titolo PDF
        mm, yyyy = (mese_corrente.split("-") + ["",""])[:2]
        mesi_it = {"01":"Gennaio","02":"Febbraio","03":"Marzo","04":"Aprile","05":"Maggio","06":"Giugno",
                   "07":"Luglio","08":"Agosto","09":"Settembre","10":"Ottobre","11":"Novembre","12":"Dicembre"}
        titolo_esteso = f"{mesi_it.get(mm, mm)} {yyyy}"

        # PDF
        st.subheader("PDF")
        pdf_buf = BytesIO()
        compose_pdf(path_out=pdf_buf, logo_path="assets/logo.jpg", title_mmYYYY=titolo_esteso,
                    anag_dict=anag_dict, table_rows=table_rows, last_class=last_class,
                    chart_img=ImageReader(BytesIO(img_bytes)))
        pdf_name = f"Report_{denom_safe}_{mese_corrente}.pdf"
        st.download_button("Scarica PDF",
                           data=pdf_buf.getvalue(),
                           file_name=f"Report_{denom_safe}_{mese_corrente}.pdf",
                           mime="application/pdf",
                           key="download_pdf")

        if st.button("Spedisci PDF", key="send_pdf"):
            try:
                cliente_email = row.get("email", "")
                send_pdf_via_email(pdf_buf.getvalue(), pdf_name, cliente_email)
                st.success("Email inviata.")
            except Exception as e:
                st.error(f"Invio fallito: {e}")


    except Exception as e:
        st.error(f"Errore lettura Excel: {e}")

if __name__ == "__main__":
    if check_password():
        main()
