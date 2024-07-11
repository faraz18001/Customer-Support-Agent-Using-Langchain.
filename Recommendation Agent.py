import os
from typing import Any, List, Dict
import logging
import pathlib
from langchain.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader
)
from langchain.schema import Document, BaseRetriever, HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

# Set the OpenAI API key (consider using environment variables for security)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | List[str], **kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    pass

class DocumentLoader:
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        ext = pathlib.Path(file_path).suffix
        loader = DocumentLoader.supported_extensions.get(ext)
        if not loader:
            raise DocumentLoaderException(
                f'Invalid Extension Type {ext}, cannot load this type of file'
            )
        loader = loader(file_path)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents")
        return docs

def configure_retriever(docs: List[Document]) -> BaseRetriever:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def create_custom_prompt():
    system_template = """You are an expert technical support agent specializing in Apple products. 
    Your job is to provide clear, solutions to user problems based on the given context, 
    user questions, and chat history. Always provide practical, actionable advice.

    If the user's question is not clear or you need more information, ask for clarification.
    If the solution is not evident from the context, use your general knowledge about Apple products 
    to provide the best possible advice, but mention that it's based on general knowledge.

    Remember:
    Don't give step by step solution
    just provide 4 most closest probable solutions
    each solution must be in one line.
    only answer the question from you database
    if the user asks you any question which is not related to your database then say sorry i can't assist with that.

    Chat History:
    {chat_history}
    """
    
    human_template = """Context: {context}

    User Question: {question}

    Please provide your recommendation:"""
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

class TechSupportAgent:
    def __init__(self, retriever: BaseRetriever, llm: Any):
        self.retriever = retriever
        self.llm = llm
        self.prompt = create_custom_prompt()
        self.chat_history = ChatMessageHistory()
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def get_recommendation(self, query: str) -> Dict[str, Any]:
        relevant_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Convert chat history to a string
        chat_history_str = "\n".join([f"{message.type}: {message.content}" for message in self.chat_history.messages])
        
        # Prepare inputs for the LLMChain
        inputs = {
            "context": context,
            "question": query,
            "chat_history": chat_history_str
        }
        
        result = self.llm_chain.predict(**inputs)
        
        # Update chat history
        self.chat_history.add_user_message(query)
        self.chat_history.add_ai_message(result)
        
        return {
            "result": result,
            "source_documents": relevant_docs
        }

    def chat(self):
        print("Tech Support Agent: Hello! I'm here to help you with any Apple product issues. What problem can I assist you with today?")
        
        while True:
            user_query = input("You: ")
            if user_query.lower() in ['quit', 'exit', 'bye']:
                print("Tech Support Agent: Thank you for using our service. Have a great day!")
                break

            with get_openai_callback() as cb:
                response = self.get_recommendation(user_query)
                print("\nTech Support Agent:", response['result'])
                print(f'\n[Debug] Total Tokens: {cb.total_tokens}')
                print(f'[Debug] Total Cost: ${cb.total_cost:.4f}')

            print("\nIs there anything else I can help you with?")
def main():
    document_path = "apple-support-dataset.pdf"  # Replace with your document path
    
    docs = DocumentLoader.load_document(document_path)
    retriever = configure_retriever(docs)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    agent = TechSupportAgent(retriever, llm)

    agent.chat()

if __name__ == "__main__":
    main()