import streamlit as st


def tum_footer(is_english=False):
    # Sprachspezifische Inhalte
    imprint_label = "Imprint"
    privacy_label = "Privacy Policy"
    
    footer_html = f"""
    <style>
        /* Container für den seiteneigenen Footer */
        .tum-footer-container {{
            width: 100%;
            border-top: 1px solid #e5e5e5;
            margin-top: 40px;
            padding: 16px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            background-color: #ffffff;
        }}

        /* Link-Sektion links */
        .tum-footer-links {{
            display: flex;
            gap: 24px;
        }}

        .tum-footer-links a {{
            color: #333333;
            text-decoration: none;
            font-size: 14px;
        }}

        .tum-footer-links a:hover {{
            text-decoration: underline;
            color: #3070b3;
        }}


        }}
    </style>

    <div class="tum-footer-container">
        <div class="tum-footer-links">
            <a href="https://www.tum.de/impressum/" target="_blank">{imprint_label}</a>
            <a href="https://www.tum.de/datenschutz/" target="_blank">{privacy_label}</a>
        </div>
        
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True)

