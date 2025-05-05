# Palabras clave que suelen ser organizaciones
ORG_KEYWORDS = {"Ministerio", "Fundación", "Agencia", "Centro", "Junta", "Comisión", "Colegio"}
PREPS = {"en", "de", "del", "al"}

# Diccionario para tokens ya etiquetados
token_to_tag = {}


# Leer archivo original
with open("data/ner-es.train.csv", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f]

# Extraer etiquetas conocidas
for line in lines:
    if line and len(line.split()) == 2:
        token, tag = line.split()
        if tag != "O":
            # Guardamos los tokens con las etiquetas distintas a O
            token_to_tag[token] = tag

# Aplicar baseline con reglas
output_lines = []
previous_token = ""
for line in lines:
    if not line:
        output_lines.append("")
        previous_token = ""
        continue

    parts = line.split()

    # Línea con etiqueta
    if len(parts) == 2:
        token, tag = parts
        output_lines.append(f"{token} {tag}")
    # Línea sin etiqueta → completar
    elif len(parts) == 1:
        token = parts[0]
        tag = token_to_tag.get(token, "O")  # Etiqueta por defecto es O

        # Reglas adicionales
        if tag == "O":
            if token in ORG_KEYWORDS:
                tag = "B-ORG"
            elif token[0].isupper() and previous_token.lower() in PREPS:
                tag = "B-LOC"

        output_lines.append(f"{token} {tag}")
    else:
        output_lines.append(line)  # No debería pasar, pero por seguridad

    previous_token = parts[0]

# Guardar archivo final
with open("archivo_baseline_completado.csv", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")
