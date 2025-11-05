# chat_tcc
Amigo Virtual - Chat terapeuta com reconhecimento de emoções

Como Usar:
 1) Clone o projeto no seu computador

 2) Crie um ambiente virtual:
```sh
python -m venv .venv
```
 3) Ative o ambiente:

   - Mac: 
```sh
source .venv/bin/activate
```
   - Windowns: 
```sh
.\.venv\Scripts\activate
```
 4) Crie um arquivo .env com as seguintes informações:

```sh
   #Chaves do LLM

   AZURE_OPENAI_API_KEY=CHAVE_API_OPENAI_AZURE
   AZURE_OPENAI_ENDPOINT=ENDPOINT_OPENAI_AZURE
   AZURE_OPENAI_DEPLOYMENT_NAME=NOME_DO_DEPLOY_DO_MODELO

   #Chaves da Deteccao Emocao Textual

   AZURE_LANGUAGE_KEY=CHAVE_AZURE_AI
   AZURE_LANGUAGE_ENDPOINT=ENDPOINT_AZURE_AI
```

 5) Baixe as bibliotecas necessárias: pip install -r requirements.txt

 6) Rode o app: python app.py
