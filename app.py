import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# --- LangChain imports ---
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# --- UPDATED IMPORT ---
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app) 

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Model ---
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(50), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, sender, text):
        self.sender = sender
        self.text = text

    def to_dict(self):
        return {
            'sender': self.sender,
            'text': self.text,
            'timestamp': self.timestamp.isoformat()
        }

# --- Load the Chatbot Components ---
print("Loading chatbot components...")
try:
    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
except Exception as e:
    print(f"Error loading LLM: {e}")
    llm = None
db_path = "./chroma_db"
# --- UPDATED CLASS NAME ---
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

prompt_templates = """You are a compassionate mental health chatbot named WellMind Assistant. Your primary function is to answer questions about mental health based ONLY on the context provided.
If the user's question is related to mental health and the context contains relevant information, provide a thoughtful and supportive answer.
If the user asks a question that is NOT related to mental health or if the context does not contain the information, you MUST politely decline. Respond with something like, "I'm sorry, but my purpose is to provide support and information about mental health. I can't answer questions on other topics."
Context:
{context}
User's Question:
{question}
Chatbot's Answer:"""
PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(), chain_type_kwargs={'prompt': PROMPT})
print("Chatbot is ready and loaded.")


@app.route("/")
def home():
    """This function will serve the index.html file from the templates folder."""
    return render_template("index.html")

@app.route("/history", methods=["GET"])
def get_history():
    """Retrieves all messages from the database."""
    messages = Message.query.order_by(Message.timestamp.asc()).all()
    return jsonify([msg.to_dict() for msg in messages])

@app.route("/chat", methods=["POST"])
def handle_chat():
    """This function is called when your website sends a message."""
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    user_message = Message(sender='user', text=user_query)
    db.session.add(user_message)
    
    try:
        response_text = qa_chain.run(user_query)
        bot_message = Message(sender='bot', text=response_text)
        db.session.add(bot_message)
        db.session.commit()
        return jsonify({"answer": response_text})
    except Exception as e:
        db.session.rollback()
        print(f"Error during QA chain execution: {e}")
        return jsonify({"error": "Failed to process the request"}), 500

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

