"""Lógica de entrenamiento e inferencia del perceptrón AND."""

# Conjunto de entrenamiento de la compuerta AND
AND_DATA = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]


def funcion_escalon(z):
    """
    Función de activación escalón.
    Si z >= 0 devuelve 1.
    Si z < 0 devuelve 0.
    """
    return 1 if z >= 0 else 0


def entrenar(w1, w2, b, eta, max_epocas, datos=None):
    """
    Entrena el perceptrón muestra por muestra.

    Regla de actualización:
    w1 = w1 + eta * error * x1
    w2 = w2 + eta * error * x2
    b  = b  + eta * error
    """

    if datos is None:
        datos = AND_DATA

    historial_pasos = []
    resumen_epocas = []

    for epoca in range(1, max_epocas + 1):

        errores_epoca = 0

        for x1, x2, objetivo in datos:

            # 1. Calcular suma ponderada
            z = x1 * w1 + x2 * w2 + b

            # 2. Aplicar función de activación
            salida = funcion_escalon(z)

            # 3. Calcular error
            error = objetivo - salida

            # Guardar parámetros anteriores
            w1_anterior = w1
            w2_anterior = w2
            b_anterior = b

            # 4. Actualizar pesos y bias
            w1 = w1 + eta * error * x1
            w2 = w2 + eta * error * x2
            b = b + eta * error

            # Contabilizar errores
            if error != 0:
                errores_epoca += 1

            # Guardar procedimiento matemático
            historial_pasos.append({
                "epoca": epoca,
                "x1": x1,
                "x2": x2,
                "objetivo": objetivo,
                "z": z,
                "salida": salida,
                "error": error,
                "w1_antes": w1_anterior,
                "w2_antes": w2_anterior,
                "b_antes": b_anterior,
                "w1_despues": w1,
                "w2_despues": w2,
                "b_despues": b
            })

        # Guardar resumen de la época
        resumen_epocas.append({
            "epoca": epoca,
            "w1": w1,
            "w2": w2,
            "b": b,
            "errores": errores_epoca
        })

        # Detener si no hubo errores
        if errores_epoca == 0:
            break

    return {
        "w1": w1,
        "w2": w2,
        "b": b,
        "pasos": historial_pasos,
        "epocas": resumen_epocas
    }


def predecir(x1, x2, w1, w2, b):
    """
    Realiza una predicción con los parámetros finales.
    """

    z = x1 * w1 + x2 * w2 + b

    salida = funcion_escalon(z)

    return {
        "x1": x1,
        "x2": x2,
        "z": z,
        "salida": salida
    }