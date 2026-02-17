from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STATE
# ============================================================

class FileState(TypedDict):
    # Input
    file_path: str
    question: str
    
    # Processing
    content: str
    is_large: bool
    chunks: List[str]
    chunk_summaries: List[str]
    
    # Output
    answer: str


# ============================================================
# AGENT
# ============================================================

class AgenticFileReader:
    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=150
        )
        self.graph = self._build_graph()
    
    # --------------------------------------------------------
    # NODE 1: Load File
    # --------------------------------------------------------
    def load_file(self, state: FileState) -> FileState:
        """Load file and check size"""
        print(f"📂 Loading: {state['file_path']}")
        
        with open(state['file_path'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        is_large = len(content) > 8000
        print(f"   {'Large' if is_large else 'Small'} file ({len(content)} chars)")
        
        return {
            **state,
            "content": content,
            "is_large": is_large
        }
    
    # --------------------------------------------------------
    # ROUTER: Small or Large?
    # --------------------------------------------------------
    def route_by_size(self, state: FileState) -> str:
        """Decide which path to take"""
        return "large" if state["is_large"] else "small"
    
    # --------------------------------------------------------
    # NODE 2a: Small File - Skip to Answer
    # --------------------------------------------------------
    def handle_small(self, state: FileState) -> FileState:
        """Small files don't need chunking"""
        print("✓ Small file - answering directly")
        return state
    
    # --------------------------------------------------------
    # NODE 2b: Large File - Chunk and Summarize
    # --------------------------------------------------------
    def handle_large(self, state: FileState) -> FileState:
        """Chunk and summarize large files"""
        print("📄 Chunking large file...")
        
        # Split
        chunks = self.splitter.split_text(state["content"])
        print(f"   Split into {len(chunks)} chunks")
        
        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            prompt = ChatPromptTemplate.from_template(
                "Summarize in 1-2 sentences:\n\n{text}"
            )
            summary = self.llm.invoke(prompt.format(text=chunk)).content
            summaries.append(summary)
            print(f"   Summarized {i}/{len(chunks)}")
        
        return {
            **state,
            "chunks": chunks,
            "chunk_summaries": summaries
        }
    
    # --------------------------------------------------------
    # NODE 3: Answer Question
    # --------------------------------------------------------
    def answer(self, state: FileState) -> FileState:
        """Answer the question"""
        print(f"💭 Answering: {state['question']}")
        
        if state["is_large"]:
            answer = self._answer_large_file(state)
        else:
            answer = self._answer_small_file(state)
        
        print("✓ Answer generated")
        
        return {**state, "answer": answer}
    
    def _answer_small_file(self, state: FileState) -> str:
        """Answer from full content"""
        prompt = ChatPromptTemplate.from_template(
            """Answer based on this content:

{content}

Question: {question}

Answer:"""
        )
        
        return self.llm.invoke(
            prompt.format(
                content=state["content"],
                question=state["question"]
            )
        ).content
    
    def _answer_large_file(self, state: FileState) -> str:
        """Two-step: find relevant chunks, then answer"""
        
        # Step 1: Find relevant chunks
        summaries_text = "\n\n".join([
            f"Chunk {i+1}: {summary}"
            for i, summary in enumerate(state["chunk_summaries"])
        ])
        
        find_prompt = ChatPromptTemplate.from_template(
            """Which chunks are relevant to the question?

Summaries:
{summaries}

Question: {question}

List chunk numbers (e.g., "1, 3, 5"):"""
        )
        
        relevant_str = self.llm.invoke(
            find_prompt.format(
                summaries=summaries_text,
                question=state["question"]
            )
        ).content
        
        # Parse chunk numbers
        try:
            chunk_nums = [
                int(n.strip()) - 1
                for n in relevant_str.replace(",", " ").split()
                if n.strip().isdigit()
            ]
            chunk_nums = [n for n in chunk_nums if 0 <= n < len(state["chunks"])]
        except:
            chunk_nums = [0]  # Default to first chunk
        
        if not chunk_nums:
            chunk_nums = [0]
        
        print(f"   Using chunks: {[n+1 for n in chunk_nums[:5]]}")
        
        # Step 2: Answer from relevant chunks
        relevant_content = "\n\n---\n\n".join([
            f"[Section {i+1}]\n{state['chunks'][i]}"
            for i in chunk_nums[:5]
        ])
        
        answer_prompt = ChatPromptTemplate.from_template(
            """Answer based on these sections:

{content}

Question: {question}

Answer:"""
        )
        
        return self.llm.invoke(
            answer_prompt.format(
                content=relevant_content,
                question=state["question"]
            )
        ).content
    
    # --------------------------------------------------------
    # Build Graph
    # --------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """Create the workflow"""
        workflow = StateGraph(FileState)
        
        # Add nodes
        workflow.add_node("load", self.load_file)
        workflow.add_node("small", self.handle_small)
        workflow.add_node("large", self.handle_large)
        workflow.add_node("answer", self.answer)
        
        # Flow
        workflow.set_entry_point("load")
        
        workflow.add_conditional_edges(
            "load",
            self.route_by_size,
            {"small": "small", "large": "large"}
        )
        
        workflow.add_edge("small", "answer")
        workflow.add_edge("large", "answer")
        workflow.add_edge("answer", END)
        
        return workflow.compile()
    
    # --------------------------------------------------------
    # Public Interface
    # --------------------------------------------------------
    def query(self, file_path: str, question: str) -> str:
        """Ask a question about a file"""
        print("\n" + "="*60)
        print(f"Question: {question}")
        print("="*60 + "\n")
        
        result = self.graph.invoke({
            "file_path": file_path,
            "question": question,
            "content": "",
            "is_large": False,
            "chunks": [],
            "chunk_summaries": [],
            "answer": ""
        })
        
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result["answer"])
        print()
        
        return result["answer"]


# ============================================================
# USAGE
# ============================================================

def main():
    # Create reader
    reader = AgenticFileReader(model="gpt-4")
    
    # Ask questions
    reader.query("example.py", "What does this code do?")
    
    reader.query("example.py", "Are there any security issues?")
    
    reader.query("example.py", "What are the main functions?")
