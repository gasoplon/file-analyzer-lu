import os
import glob
import re
from openai import OpenAI
import PyPDF2
import csv

# Configura la clave de la API de OpenAI
client = OpenAI(api_key="")  # Asegúrate de poner tu clave aquí

def extract_text_from_pdf(pdf_path):
    """
    Extrae todo el texto de un archivo PDF.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"No se pudo leer el PDF {pdf_path}: {e}")
    return text

def normalize_text(text):
    """
    Normaliza el texto reemplazando múltiples espacios y signos de separación contiguos por un solo espacio.
    """
    text = re.sub(r'\s+', ' ', text)  # Reemplaza los espacios consecutivos por uno solo
    text = re.sub(r'[.]+', '.', text)  # Reemplaza puntos consecutivos
    text = re.sub(r'[;]+', ';', text)  # Reemplaza puntos y comas consecutivos
    text = re.sub(r'[!]+', '!', text)  # Reemplaza signos de exclamación consecutivos
    text = re.sub(r'[?]+', '?', text)  # Reemplaza signos de interrogación consecutivos
    return text.strip()

def extract_section(text, primerDel, segundoDel):
    """
    Extrae el texto entre dos delimitadores dados (primerDel y segundoDel).
    """
    text = normalize_text(text)  # Normalizamos el texto
    pattern = re.escape(primerDel) + r"(.*?)" + re.escape(segundoDel)
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return "no aplica"

def generate_embeddings(text):
    """
    Genera el embedding de un texto utilizando la nueva API de OpenAI >= 1.0.0.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        # Acceder a los embeddings correctamente con el nuevo formato de la respuesta
        embedding = response.data[0].embedding
        
        return embedding
    except Exception as e:
        print(f"Error al generar embedding: {e}")
        return None

def ask_openai_question(text, question):
    """
    Envía un texto a la API de OpenAI con una pregunta específica.
    """
    try:
        # print(text)
        # Hacemos la solicitud a la API de OpenAI para obtener la respuesta
        response = client.chat.completions.create(
           model="gpt-3.5-turbo",  # Asegúrate de usar el modelo correcto
            messages=[
                {"role": "system", "content": "you are an assistant, I will pass you texts in natural language, and I will ask you to extract a field from that text, for example: ‘Hello I am David’ and I ask you ‘extract me the text’, you must return me only the name or a text without explanation: David."},
                {"role": "user", "content": f"Texto: {text}\nPregunta: {question}"}
            ],
            max_tokens=100,  # Puedes ajustar el número de tokens según lo necesites
            temperature=0.0   # Usamos 0.0 para obtener respuestas más precisas
        )
        
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error al hacer la pregunta: {e}")
        return "No se pudo obtener la respuesta."

def main():
    docs_folder = "./docs"
    output_csv = "results.csv"
    # Lista de pares de delimitadores y preguntas asociadas con la columna correspondiente en el CSV
    delimiters_and_questions = [
        ("Uso previsto", "Principio del test", "¿Cuál sería el nombre del producto en este texto?", "Nombre del producto"),
        ("Conservación y estabilidad", "Material suministrado", "¿Cuales son los grados entre los que se tiene que conservar?", "Grados de conservación"),
        ("Intervalo de medici", "Material de", "¿Cuales son intervalos de medición?", "Límite de detección"),
        ("Conservación y estabilidad", "Material suministrado", "¿Hasta cuando puede usarse este producto?", "Estabilidad reactivo"),
        ("Principio del test", "Intervalo de medici", "¿cual es la medicion en una frase pero con todos los datos?", "Principio del test"),
        ("", "Espa", "Extrae en base a este ejemplo. El texto es '06688969001V 7.02024-09CoaguChek PT Test 066887212 x 24CoaguChek ® Pro II' y yo quiero el '2 x 24', pero para mi texto", "Test por kit"),
        ("Uso previsto", "Princ", "¿Cual es el tipo de muestra sobre el que se aplica?", "Tipo de muestra"),
        # Puedes añadir más pares de delimitadores y preguntas según sea necesario
    ]

    # Crear el archivo CSV de salida
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        # Escribimos la cabecera del archivo CSV
        headers = ["filename"]  # Empezamos con 'filename' que será común en todas las filas
        headers.extend([col[3] for col in delimiters_and_questions])  # Añadimos las columnas configuradas
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        # Procesar cada archivo PDF
        for pdf_path in glob.glob(os.path.join(docs_folder, "*.pdf")):
            pdf_name = os.path.basename(pdf_path)
            print(f"Procesando: {pdf_name}")

            # Extraer todo el texto del PDF
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text:
                print(f"No se pudo extraer texto de {pdf_name}.")
                continue

            # Resultados por archivo PDF
            row = [pdf_name]

            # Iterar sobre cada par de delimitadores, preguntas y columna
            for primerDel, segundoDel, question, col_name in delimiters_and_questions:
                # Extraer la sección entre los delimitadores
                section_text = extract_section(full_text, primerDel, segundoDel)
                if section_text == "no aplica":
                    print(f"No se encontró la sección entre '{primerDel}' y '{segundoDel}' en {pdf_name}.")
                    row.append("No se encontró")
                    continue

                # Generar embeddings del texto extraído
                embedding = generate_embeddings(section_text)
                if embedding is None:
                    print(f"No se pudo generar embedding para {pdf_name}.")
                    row.append("No se generó embedding")
                    continue

                # Hacer una pregunta a OpenAI sobre la sección extraída
                answer = ask_openai_question(section_text, question)

                # Añadir la respuesta de la pregunta en la columna correspondiente
                row.append(answer)

            # Guardar los resultados en el CSV
            writer.writerow(row)
            print(f"{pdf_name}: Datos procesados y guardados.")

if __name__ == "__main__":
    main()
