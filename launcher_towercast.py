import subprocess
import webbrowser
import os
import sys
from PIL import Image

# Ruta al script de Streamlit
base = os.path.expanduser("~")
ruta_script = os.path.join(
    base,
    "DERCO CHILE REPUESTOS SpA",
    "Planificación y abastecimiento - Documentos",
    "KPI",
    "Instock Semanal",
    "tablero instock - VERSION",
    "Parquet_Vigencia",
    "TowerCast.py"
)

# Ruta del logo compatible con .exe y .py
if getattr(sys, 'frozen', False):
    ruta_logo = os.path.join(sys._MEIPASS, "logo_towercast.png")
else:
    ruta_logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_towercast.png")

# Verificar si existe el logo
if not os.path.exists(ruta_logo):
    print(f"❌ Logo no encontrado en: {ruta_logo}")
else:
    Image.open(ruta_logo).close()

# Ejecutar Streamlit
subprocess.Popen(["streamlit", "run", ruta_script])
webbrowser.open("http://localhost:8501")
