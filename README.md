# chat_tcc
Amigo Virtual - Chat terapeuta com reconhecimento de emoções

Como Usar:
 - Clone o projeto no seu computador
 - Crie um ambiente virtual:
    - python -m venv .venv
 - Ative o ambiente:
    - Mac: source .venv/bin/activate
    - Windowns: .\.venv\Scripts\activate
 - Crie um arquivo .env com as seguintes informações:
    #Chaves do LLM
    AZURE_OPENAI_API_KEY=CHAVE_API_OPENAI_AZURE
    AZURE_OPENAI_ENDPOINT=ENDPOINT_OPENAI_AZURE
    AZURE_OPENAI_DEPLOYMENT_NAME=NOME_DO_DEPLOY_DO_MODELO

    #Chaves da Deteccao Emocao Textual
    AZURE_LANGUAGE_KEY=CHAVE_AZURE_AI
    AZURE_LANGUAGE_ENDPOINT=ENDPOINT_AZURE_AI
 
 - Baixe as bibliotecas necessárias: pip install -r requirements.txt
 - Rode o app: python app.py
