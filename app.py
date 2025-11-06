import os
from flask import Flask, request, jsonify, send_from_directory
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import numpy as np # Necessário para processar a imagem

# --- Importações de Análise de Sentimento (Azure) ---
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# --- Importações para a Detecção Facial (DeepFace) ---
import base64
import io
import cv2 # OpenCV para decodificar a imagem
from deepface import DeepFace

# Carrega as variáveis de ambiente (.env)
load_dotenv()

# Inicializa o Flask
app = Flask(__name__)

""" # === 1. Configuração do Cliente Azure OpenAI (LLM) ===
try:
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([OPENAI_API_KEY, OPENAI_ENDPOINT, openai_deployment_name]):
        raise ValueError("Credenciais do Azure OpenAI não definidas.")
    
    openai_client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01",
        azure_endpoint=OPENAI_ENDPOINT
    )
    print("Cliente Azure OpenAI inicializado.")
except Exception as e:
    print(f"Erro na configuração do OpenAI: {e}")
    openai_client = None """

# === NOVO: Configuração do Cliente LOCAL (Ollama) ===
try:
    # O Ollama não precisa de chave de API por padrão
    # A URL base é o padrão do Ollama
    openai_client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama' # (O valor real não importa, mas não pode ser nulo)
    )
    # Define o nome do modelo que você baixou no Ollama
    openai_deployment_name = "deepseek-r1:14b" # <-- MUDE ISSO para o modelo que você baixou
    
    print(f"Cliente OpenAI (Local/Ollama) inicializado. Usando modelo: {openai_deployment_name}")
except Exception as e:
    print(f"Erro ao inicializar o cliente local OpenAI: {e}")
    openai_client = None

# === 2. Configuração do Cliente Azure Language (Análise de Texto) ===
try:
    LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
    LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
    
    if not all([LANGUAGE_ENDPOINT, LANGUAGE_KEY]):
        raise ValueError("Credenciais do Azure Language não definidas.")
    
    text_analytics_client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT, 
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )
    print("Cliente Azure AI Language (Text Analytics) inicializado.")
except Exception as e:
    print(f"Erro na configuração do Language API: {e}")
    text_analytics_client = None

# === 3. NOVO: "Aquecendo" o DeepFace ===
# A primeira chamada ao DeepFace baixa os modelos.
# Fazemos uma chamada falsa na inicialização para evitar um longo delay na primeira requisição.
try:
    print("Aquecendo o modelo de emoção do DeepFace (isso pode levar um momento)...")
    # Cria uma imagem preta falsa para o aquecimento
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    DeepFace.analyze(
        img_path=dummy_image, 
        actions=['emotion'], 
        enforce_detection=False
    )
    print("Modelo DeepFace carregado e pronto.")
except Exception as e:
    print(f"Erro ao inicializar o DeepFace: {e}")

# -----------------------------------------------------------------

# NOVO: Função auxiliar para mapear emoções (DEEPFACE) para sentimentos (AZURE)
def map_visual_to_sentiment(visual_emotion_str):
    """Mapeia a emoção do DeepFace para o sentimento da API de Texto."""
    # Emoções do DeepFace: 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    # Emoções do Azure: 'positive', 'negative', 'neutral'
    mapping = {
        'happy': 'positive',
        'sad': 'negative',
        'angry': 'negative',
        'disgust': 'negative',
        'fear': 'negative',
        'neutral': 'neutral',
        'surprise': 'neutral'
    }
    return mapping.get(visual_emotion_str, 'desconhecido')

# -----------------------------------------------------------------

# Rota 1: Servir a página de chat (index.html)
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

# Rota 2: O endpoint da API de chat (MODIFICADO)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    history = data.get("history", [])
    image_data_base64 = data.get("image_data") # Imagem da webcam

    if not user_message:
        return jsonify({"error": "Nenhuma mensagem fornecida"}), 400

    # --- 1. Bloco de Análise de Sentimento (Texto) ---
    detected_sentiment = "desconhecido"
    if text_analytics_client:
        try:
            result = text_analytics_client.analyze_sentiment(documents=[user_message])[0]
            if not result.is_error:
                detected_sentiment = result.sentiment
                print(f"Sentimento (Texto) detectado: {detected_sentiment}")
            else:
                detected_sentiment = "erro na análise"
        except Exception as e:
            print(f"Erro ao chamar a Language API: {e}")
            detected_sentiment = "erro na análise"
    
    # --- 2. NOVO: Bloco de Análise de Emoção (Visual com DeepFace) ---
    visual_emotion_raw = "desconhecido" # ex: 'happy'
    mapped_visual_sentiment = "desconhecido" # ex: 'positive'

    if image_data_base64:
        try:
            # Decodifica a imagem Base64 recebida do JS
            img_bytes = base64.b64decode(image_data_base64.split(',')[1])
            
            # Converte bytes para um array numpy
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decodifica o array numpy para uma imagem OpenCV
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Chama a análise do DeepFace
            # enforce_detection=False evita que o app quebre se nenhum rosto for encontrado
            analysis_result = DeepFace.analyze(
                img_path=img_cv2, 
                actions=['emotion'], 
                enforce_detection=False
            )

            # DeepFace retorna uma lista, pegamos o primeiro rosto
            if analysis_result and isinstance(analysis_result, list) and analysis_result[0].get('dominant_emotion'):
                visual_emotion_raw = analysis_result[0]['dominant_emotion']
                mapped_visual_sentiment = map_visual_to_sentiment(visual_emotion_raw)
                print(f"Emoção (Visual) detectada: {visual_emotion_raw} (Mapeado para: {mapped_visual_sentiment})")
            else:
                print("Análise visual (DeepFace): Nenhum rosto detectado.")
                visual_emotion_raw = "nenhum rosto"

        except Exception as e:
            print(f"Erro ao chamar o DeepFace: {e}")
            visual_emotion_raw = "erro na análise"
            
    # --- 3. Lógica de Decisão (Sua regra de negócio) ---
    
    final_emotion_for_llm = "nenhuma"

    if mapped_visual_sentiment == detected_sentiment and detected_sentiment not in ["desconhecido", "erro na análise"]:
        final_emotion_for_llm = detected_sentiment
        print(f"*** CONFIRMAÇÃO DE EMOÇÃO: Texto ({detected_sentiment}) e Visual ({visual_emotion_raw}) coincidem.")
    else:
        print(f"*** Emoções não coincidem: Texto={detected_sentiment}, Visual={visual_emotion_raw} (Mapeado={mapped_visual_sentiment})")

    # --- 4. Bloco do LLM (Adaptado) ---
    try:
        if not openai_client:
             return jsonify({"error": "Cliente OpenAI não inicializado."}), 500

        if final_emotion_for_llm != "nenhuma":
            system_prompt = (
                "Você é um amigo virtual, que utiliza técnicas da terapia cognitivo-comportamental para conversar com o usuário e tranquilizá-lo."
                "O sentimento do usuário foi detectado tanto no texto quanto na sua expressão facial, confirmando um estado emocional. "
                f"O sentimento confirmado do usuário é: **{final_emotion_for_llm}**. "
                "Adapte seu tom sutilmente para refletir essa emoção confirmada."
            )
        else:
            system_prompt = (
                "Você é um amigo virtual, que utiliza técnicas da terapia cognitivo-comportamental para conversar com o usuário e tranquilizá-lo."
                "Responda de forma prestativa. O sentimento do usuário é incerto ou misto, "
                "então mantenha um tom neutro e compreensivo, sem assumir um estado emocional."
            )
        
        messages_for_api = [
            {"role": "system", "content": system_prompt}
        ] + history

        response = openai_client.chat.completions.create(
            model=openai_deployment_name, # <-- Usará "llama3:8b" (ou o que você definiu)
            messages=messages_for_api,
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({
            "response": ai_response, 
            "sentiment": detected_sentiment, 
            "visual_emotion": visual_emotion_raw # Retorna a emoção bruta do deepface
        })

    except Exception as e:
        print(f"Erro na API do OpenAI (local): {e}")
        return jsonify({"error": str(e)}), 500

# Inicia o servidor Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)