import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import re


def calcular_tiempo_fermentacion(df):
    # df: columnas 'Fecha' (o 'FechaHora'), 'D', 'T'
    # Si tienes múltiples horarios por día, asegúrate de tener un timestamp por fila.
    # Detecta primer instante con D==995 (término)
    df = df.sort_values('Fecha')
    inicio = df['Fecha'].iloc[0]
    fin = df.loc[df['D'] == 995, 'Fecha'].min()
    if pd.isna(fin):
        raise ValueError("No se encontró D==995 (fermentación no terminada en el dataset).")
    t_ref_days = (fin - inicio).total_seconds() / 86400.0
    T_ref = df.loc[df['Fecha'] <= fin, 'T'].mean()
    return t_ref_days, T_ref

def tiempo_por_Q10(T_obj, t_ref, T_ref, Q10=2.5):
    # t(T) = t_ref / Q10^((T - T_ref)/10)
    return t_ref / (Q10 ** ((T_obj - T_ref) / 10.0))

#===============================================================================
excel_path = Path("/home/camilo/Documents/cii-vct-tesis/Resultados/2024/I+D Nutricion Temperatura.xlsx")
sheet_name = "T° y D°"  # cámbialo si tu pestaña tiene otro nombre exacto

if not excel_path.exists():
    raise FileNotFoundError(f"No existe el archivo:\n{excel_path}")

# Lee con doble encabezado: arriba horas, abajo D/T
raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])

# Identificar la columna de fecha (queda como MultiIndex: ('Fecha', NaN) o similar)
fecha_candidates = [c for c in raw.columns if isinstance(c, tuple) and str(c[0]).strip().lower().startswith("fecha")]
if not fecha_candidates:
    # En algunos archivos la fecha puede venir como columna simple (sin MultiIndex)
    fecha_candidates = [c for c in raw.columns if isinstance(c, str) and c.strip().lower().startswith("fecha")]
if not fecha_candidates:
    raise KeyError(f"No se encontró columna de fecha. Columnas: {list(raw.columns)}")
fecha_col = fecha_candidates[0]

# Normaliza símbolos posibles en subcolumnas
def is_d_label(x: str) -> bool:
    s = str(x).strip().lower().replace("º", "°")
    return s in {"d", "d°", "densidad"}
def is_t_label(x: str) -> bool:
    s = str(x).strip().lower().replace("º", "°")
    return s in {"t", "t°", "temperatura"}

# Parseo robusto de hora (acepta datetime.time y strings como '9:00', '09:00:00')
time_pattern = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$")
def parse_hour(val) -> int | None:
    if isinstance(val, dt.time):
        return val.hour
    s = str(val).strip()
    m = time_pattern.match(s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    return None

# Pre-indexar columnas por hora objetivo y tipo (D/T)
target_hours = [3, 9, 15, 21]
cols_by_hour: dict[int, dict[str, tuple]] = {}
for col in raw.columns:
    if not isinstance(col, tuple):
        continue
    h = parse_hour(col[0])
    if h is None:
        continue
    kind = "D" if is_d_label(col[1]) else ("T" if is_t_label(col[1]) else None)
    if kind:
        cols_by_hour.setdefault(h, {})[kind] = col

registros = []

for _, row in raw.iterrows():
    # Fecha del día (sin hora)
    fecha_base = row[fecha_col] if isinstance(fecha_col, tuple) else row.get(fecha_col)
    fecha_base = pd.to_datetime(fecha_base, errors="coerce", dayfirst=True)
    if pd.isna(fecha_base):
        continue

    # Para cada hora objetivo, usa columnas detectadas (MultiIndex)
    for h in target_hours:
        pair = cols_by_hour.get(h, {})
        d_col = pair.get("D")
        t_col = pair.get("T")

        if d_col is None or t_col is None:
            # No hay datos para esta hora; continuar
            continue

        d_val = pd.to_numeric(row.get(d_col, np.nan), errors="coerce")
        t_val = pd.to_numeric(row.get(t_col, np.nan), errors="coerce")
        if pd.notna(d_val) and pd.notna(t_val):
            ts = pd.Timestamp(
                year=fecha_base.year, month=fecha_base.month, day=fecha_base.day,
                hour=h, minute=0, second=0
            )
            registros.append({"Fecha": ts, "D": float(d_val), "T": float(t_val)})

# Construir DataFrame y validar columnas antes de usar dropna/sort_values
df = pd.DataFrame(registros)

if df.empty:
    # Diagnóstico útil: muestra primeras columnas y etiquetas de nivel superior si hay MultiIndex
    raw_cols_preview = list(raw.columns)[:12]
    top_levels = sorted({str(c[0]).strip() for c in raw.columns if isinstance(c, tuple)})
    raise ValueError(
        "No se pudo construir el DataFrame (Fecha, D, T) porque no se generaron registros.\n"
        f"- Verifica 'sheet_name' ('{sheet_name}') y los encabezados esperados.\n"
        f"- Primeras columnas leídas: {raw_cols_preview}\n"
        f"- Encabezados de nivel superior detectados (horas): {top_levels}\n"
        f"- Columna de fecha detectada: {fecha_col}"
    )

expected_cols = ["Fecha", "D", "T"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    example_keys = list(registros[0].keys()) if registros else []
    raise ValueError(
        f"Faltan columnas esperadas en los registros: {missing}. "
        f"Ejemplo de claves en un registro: {example_keys}"
    )

df = df.dropna(subset=expected_cols).sort_values("Fecha")

t_ref, T_ref = calcular_tiempo_fermentacion(df)
t_pred_14C = tiempo_por_Q10(14.0, t_ref, T_ref, Q10=2.5)
t_pred_20C = tiempo_por_Q10(20.0, t_ref, T_ref, Q10=2.5)

print(t_pred_14C, t_pred_20C)