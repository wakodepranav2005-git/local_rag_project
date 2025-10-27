import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB

# --- New Imports for Local LLM ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
# --- End New Imports ---


# Load environment variables
load_dotenv()


def load_documents() -> List[Dict[str, Any]]:
    """
    Load .txt documents from the 'data/' directory.

    Returns:
        List of document dictionaries, each with 'content' and 'metadata'
    """
    # === IMPLEMENTATION (Step 2) ===
    results = []
    data_dir = "data"
    
    # Find all .txt files in the data directory
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    print(f"Found {len(txt_files)} documents to load.")
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use the filename as metadata
            metadata = {"source": os.path.basename(file_path)}
            
            results.append({
                "content": content,
                "metadata": metadata
            })
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            
    return results


class RAGAssistant:
    """
    A RAG-based AI assistant using ChromaDB and a local HuggingFace LLM.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM
        print("Initializing local LLM...")
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError("Failed to initialize local LLM.")

        # Initialize vector database
        self.vector_db = VectorDB()

        # === IMPLEMENTATION (Step 6) ===
        # Create RAG prompt template
        template = """
You are a helpful AI assistant. You must answer the user's question based *only* on the provided context.
If the context does not contain the answer, you must state that you cannot find the information in the provided documents.
Do not use any external knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        self.prompt_template = ChatPromptTemplate.from_template(template)
        # === End Step 6 ===

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the local HuggingFace LLM.
        """
        model_id = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        
        print(f"Loading model ({model_id}) with 4-bit quantization...")

        # --- Model Loading (from your example) ---
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" # Automatically uses GPU if available
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print("Model loaded successfully.")

            # Create the transformers pipeline
            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=500,  # Max tokens to generate
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                return_full_text=False
            )

            # Wrap the transformers pipeline in a LangChain object
            return HuggingFacePipeline(pipeline=text_gen_pipeline)

        except Exception as e:
            print(f"Error loading local model: {e}")
            return None


    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the knowledge base.
        """
        self.vector_db.add_documents(documents)

    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Answer questions using retrieved context.

        Args:
            question: User's question
            n_results: Number of context chunks to retrieve

        Returns:
            Dictionary with answer and context information
        """
        # === IMPLEMENTATION (Step 7) ===
        print(f"\nProcessing query: '{question}'")
        
        # 1. Search for relevant chunks
        search_results = self.vector_db.search(question, n_results=n_results)
        retrieved_docs = search_results.get("documents", [])
        
        if not retrieved_docs:
            print("No relevant documents found in the vector database.")
            return {
                "answer": "I'm sorry, I couldn't find any relevant information in the loaded documents to answer your question.",
                "context": []
            }

        # 2. Combine chunks into a single context string
        context = "\n\n---\n\n".join(retrieved_docs)

        # 3. Generate response using LLM + context
        print(f"Generating answer with {len(retrieved_docs)} context chunk(s)...")
        answer_raw = self.chain.invoke({
            "context": context,
            "question": question
        })

        # --- FIX: Split the response and take the first part ---
        # The model sometimes repeats the answer with a new "Assistant: ANSWER:" prefix.
        # We split by this prefix (or similar) and take only the first generated block.
        answer = answer_raw.split("\n\nAssistant:")[0].strip()
        
        # 4. Return structured results
        return {
            "answer": answer,
            "context": retrieved_docs
        }
        # === End Step 7 ===


def main():
    """Main function to run the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        
        if not sample_docs:
            print("No documents found in 'data/' directory. Please add some .txt files.")
            return

        # Add documents to the vector DB
        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("\nEnter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(question)
                
                # --- MODIFIED: Prettier output ---
                print("\n" + "="*50)
                print("🤖 AI ANSWER:")
                print(result['answer'])
                print("\n--- RETRIEVED CONTEXT ---")
                if result['context']:
                    for i, doc in enumerate(result['context']):
                        # Print first 150 chars of each context chunk
                        print(f"[{i+1}] {doc[:150]}...")
                else:
                    print("No context was used.")
                print("="*50)
                # --- End Modification ---

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()