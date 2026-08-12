import torch
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# 2. LOAD LLM
# ============================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    ),
    device_map="auto"
)


# ============================================================
# 3. TEXT GENERATION PIPELINE
# ============================================================

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.1,
    do_sample=False,
    return_full_text=False
)

llm = HuggingFacePipeline(
    pipeline=text_generation_pipeline
)

print("LLM loaded successfully!")


# ============================================================
# 4. PROMPT
# ============================================================

prompt_template = """
You are a research paper assistant.

Answer the question using ONLY the information
provided in the context below.

Do not use outside knowledge.

If the answer is not present in the context, say:

"I could not find this information in the uploaded paper."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# ============================================================
# 5. TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# ============================================================
# 6. GLOBAL RETRIEVER
# ============================================================

retriever = None


# ============================================================
# 7. PROCESS PDF
# ============================================================

def process_pdf(pdf_file):

    global retriever

    if pdf_file is None:
        return "Please upload a valid PDF file."

    try:

        loader = PyPDFLoader(pdf_file)

        documents = loader.load()

        chunks = text_splitter.split_documents(
            documents
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return (
            f"PDF processed successfully! "
            f"{len(documents)} pages and "
            f"{len(chunks)} chunks created."
        )

    except Exception as e:

        return f"Error processing PDF: {str(e)}"


# ============================================================
# 8. ASK QUESTION
# ============================================================

def ask_question(question):

    global retriever

    if retriever is None:
        return (
            "Please upload and process a PDF first.",
            []
        )

    if not question.strip():
        return (
            "Please enter a question.",
            []
        )

    try:

        retrieved_docs = retriever.invoke(
            question
        )

        context_parts = []

        for doc in retrieved_docs:

            page_number = (
                doc.metadata.get("page", 0) + 1
            )

            context_parts.append(
                f"[Page {page_number}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(
            context_parts
        )

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        response = llm.invoke(
            final_prompt
        )

        pages = sorted(
            set(
                doc.metadata.get("page", 0) + 1
                for doc in retrieved_docs
            )
        )

        return response, pages

    except Exception as e:

        return (
            f"Error generating answer: {str(e)}",
            []
        )


# ============================================================
# 9. CHATBOT FUNCTION
# ============================================================

def chatbot(question):

    answer, pages = ask_question(question)

    sources = "\n".join(
        f"- Page {page}"
        for page in pages
    )

    return answer, sources


# ============================================================
# 10. GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Research Paper Chatbot"
) as demo:

    gr.Markdown(
        "# 📚 Research Paper Chatbot"
    )

    gr.Markdown(
        "Upload a research paper and ask "
        "questions about its content."
    )

    pdf_upload = gr.File(
        label="Upload Research Paper",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button(
        "📄 Process Paper"
    )

    status_box = gr.Textbox(
        label="Status",
        interactive=False
    )

    question_box = gr.Textbox(
        label="Ask a Question",
        placeholder=(
            "Example: What methodology was used?"
        )
    )

    ask_button = gr.Button(
        "🔍 Ask Question"
    )

    answer_box = gr.Markdown(
        label="Answer"
    )

    source_box = gr.Markdown(
        label="Sources"
    )

    process_button.click(
        fn=process_pdf,
        inputs=pdf_upload,
        outputs=status_box
    )

    ask_button.click(
        fn=chatbot,
        inputs=question_box,
        outputs=[
            answer_box,
            source_box
        ]
    )


# ============================================================
# 11. LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()
