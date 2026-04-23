import streamlit as st
import base64
from pathlib import Path


def get_tum_logo_base64():
    # 1. Absoluten Pfad zur Datei ermitteln (ausgehend von dieser Skript-Datei)
    # Annahme: Ihr Skript liegt im Hauptordner, das Logo in app/assets/
    current_dir = Path(__file__).parent.parent
    logo_path = current_dir / "assets" / "tum_logo.svg"

    # Debugging-Info (nur für Sie zur Kontrolle, ob der Pfad stimmt)
    if not logo_path.exists():
        st.error(f"Logo nicht gefunden unter: {logo_path.resolve()}")
        return ""

    try:
        svg_content = logo_path.read_text(encoding="utf-8")
        b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception as e:
        st.error(f"Fehler beim Einlesen: {e}")
        return ""


def tum_header(page_title_line1):
    # Definition der Texte
    university_name = "Technical University of Munich"
    link_title = "Home www.tum.de"

    logo_url = get_tum_logo_base64()

    # CSS für das Layout
    st.markdown(
        f"""
        <style>
            /* Entfernen der Standard-Abstände von Streamlit */
            .block-container {{
                padding-top: 0rem;
            }}

            header {{
                visibility: hidden;
            }}

            .tum-container {{
                width: 100%;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                margin-bottom: 20px;
            }}

            # /* Funktionsleiste */
            # .tum-top-bar {{
            #     background-color: #f2f2f2;
            #     height: 36px;
            #     width: 100%;
            #     display: flex;
            #     align-items: center;
            #     justify-content: flex-end;
            #     padding: 0 0px;
            #     font-size: 14px;
            #     color: #333;
            # }}

            /* Weißer Header-Bereich */
            .tum-main-header {{
                background-color: #ffffff;
                height: 104px;
                padding: 0 0px;
                display: flex;
                justify-content: space-between;
                align-items: flex-end; /* Alignierung an der Unterkante des Logos */
                padding-bottom: 20px; /* Schutzzone unten */
            }}

            /* Textbereich links */
            .tum-text-area {{
                color: #3070b3;
                line-height: 1.2;
                margin-bottom: 2px; /* Feinjustierung an Logo-Basis */
            }}

            .tum-line{{
                font-size: 14px;
                font-weight: normal;
            }}

            /* Logo rechts */
            .tum-logo-link img {{
                width: 73px;
                height: 38px;
            }}

            /* Responsive Anpassung für Mobile */
            @media (max-width: 768px) {{
                .tum-main-header {{
                    height: 90px;
                }}
            }}
        </style>

        <div class="tum-container">
            <div class="tum-main-header">
                <a href="https://www.bgd.ed.tum.de/" title="Professorship Big Geospatial Data Management"
                style="text-decoration: none">
                <div class="tum-text-area">
                    <div class="tum-line">Professorship Big Geospatial Data Management</div>
                    <div class="tum-line">TUM School of Engineering and Design</div>
                    <div class="tum-line">{university_name}</div>
                </div>
                </a>
                <a href="https://www.tum.de" class="tum-logo-link" title="{link_title}">
                    <img src="{logo_url}" alt="TUM Logo">
                </a>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )
