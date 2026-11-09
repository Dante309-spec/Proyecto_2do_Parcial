"""Generación del informe PDF del entrenamiento."""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


def generar_pdf(resultado, parametros):

    # Crear archivo PDF en memoria
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm
    )

    estilos = getSampleStyleSheet()

    elementos = [

        Paragraph(
            "Informe de entrenamiento: Perceptrón AND",
            estilos["Title"]
        ),

        Paragraph(
            "Perceptrón monocapa (SLP) - Función de activación escalón",
            estilos["Heading2"]
        ),

        Spacer(1, 8),

        Paragraph(
            f"Parámetros iniciales: "
            f"w1={parametros['w1']}, "
            f"w2={parametros['w2']}, "
            f"b={parametros['b']}, "
            f"eta={parametros['eta']}. "
            f"Máximo de épocas: {parametros['max_epocas']}.",
            estilos["BodyText"]
        ),

        Spacer(1, 8),

        Paragraph(
            "Reglas matemáticas",
            estilos["Heading2"]
        ),

        Paragraph(
            "z = (x1 * w1) + (x2 * w2) + b",
            estilos["BodyText"]
        ),

        Paragraph(
            "y = 1 si z >= 0; de lo contrario y = 0",
            estilos["BodyText"]
        ),

        Paragraph(
            "e = objetivo - y",
            estilos["BodyText"]
        ),

        Paragraph(
            "w1' = w1 + eta * e * x1",
            estilos["BodyText"]
        ),

        Paragraph(
            "w2' = w2 + eta * e * x2",
            estilos["BodyText"]
        ),

        Paragraph(
            "b' = b + eta * e",
            estilos["BodyText"]
        ),

        Spacer(1, 10),

        Paragraph(
            "Resumen por épocas",
            estilos["Heading2"]
        )
    ]

    # Crear tabla de épocas
    tabla = [[
        "Época",
        "w1 final",
        "w2 final",
        "Bias final",
        "Errores"
    ]]

    for r in resultado["epocas"]:

        tabla.append([
            str(r["epoca"]),
            f'{r["w1"]:.4f}',
            f'{r["w2"]:.4f}',
            f'{r["b"]:.4f}',
            str(r["errores"])
        ])

    tabla_epocas = Table(
        tabla,
        repeatRows=1,
        hAlign="LEFT"
    )

    tabla_epocas.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#173B57")),

        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.white),

        ("GRID", (0, 0), (-1, -1),
         0.5, colors.HexColor("#B8C7D1")),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EEF4F7")]),

        ("PADDING", (0, 0), (-1, -1), 7)

    ]))

    elementos.append(tabla_epocas)

    elementos.append(Spacer(1, 10))

    elementos.append(
        Paragraph(
            "Procedimiento muestra por muestra",
            estilos["Heading2"]
        )
    )

    # Tabla con los cálculos completos
    pasos = [[
        "Época",
        "Entrada",
        "t",
        "z",
        "y",
        "e",
        "w1 nuevo",
        "w2 nuevo",
        "b nuevo"
    ]]

    for p in resultado["pasos"]:

        pasos.append([
            str(p["epoca"]),
            f'({p["x1"]},{p["x2"]})',
            str(p["objetivo"]),
            f'{p["z"]:.4f}',
            str(p["salida"]),
            str(p["error"]),
            f'{p["w1_despues"]:.4f}',
            f'{p["w2_despues"]:.4f}',
            f'{p["b_despues"]:.4f}'
        ])

    tabla_pasos = Table(
        pasos,
        repeatRows=1,
        colWidths=[
            13*mm, 17*mm, 9*mm, 14*mm,
            9*mm, 9*mm, 19*mm, 19*mm, 19*mm
        ]
    )

    tabla_pasos.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#173B57")),

        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.white),

        ("GRID", (0, 0), (-1, -1),
         0.35, colors.HexColor("#AAB8C2")),

        ("FONTSIZE", (0, 0), (-1, -1), 7),

        ("PADDING", (0, 0), (-1, -1), 4),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F6F8")])

    ]))

    elementos.append(tabla_pasos)

    # Construir documento
    doc.build(elementos)

    buffer.seek(0)

    return buffer.getvalue()