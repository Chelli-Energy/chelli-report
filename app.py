import streamlit as st

# ---- GATE DI ACCESSO CON PASSWORD UNICA ----
def check_password():
    pw = st.secrets.get("APP_PASSWORD", None)
    if pw is None:
        st.error("Password non configurata. Imposta APP_PASSWORD nei Secrets.")
        st.stop()

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    st.title("Accesso riservato")
    pwd = st.text_input("Password", type="password")
    if st.button("Entra"):
        if pwd == pw:
            st.session_state.auth_ok = True
            st.experimental_rerun()
        else:
            st.error("Password errata")
    st.stop()

# ---- APP ----
def main():
    st.write("Chelli Report — setup ok.")
    st.info("Accesso riuscito. Prossimo passo: anagrafica clienti e upload Excel.")

if __name__ == "__main__":
    if check_password():
        main()
