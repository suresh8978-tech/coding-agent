from typing import TypedDict, List, Dict, Any
import os
from langgraph.graph import StateGraph, END
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from tools.ansible_log_chunker import AnsibleLogChunker


class AnsibleLogState(TypedDict):
    file_path: str
    question: str
    log_content: str
    chunks: List[Dict[str, Any]]
    answer: str


class AnsibleLogReader:
    def __init__(self, model: str | None = None):
        # Use env vars consistent with agent.py
        llm_name = model or os.getenv("LLM_NAME", "bedrock-sonnet-4-5")
        api_key = os.getenv("LITELLM_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
        api_url = os.getenv("LITELLM_API_BASE", os.getenv("ANTHROPIC_API_URL", ""))
        
        llm_kwargs = {
            "model": llm_name,
            "api_key": api_key,
            "max_tokens": 4096,
            "drop_params": True,
        }
        if api_url:
            llm_kwargs["api_base"] = api_url

        self.llm = ChatLiteLLM(**llm_kwargs)
        self.chunker = AnsibleLogChunker()
        self.graph = self._build_graph()
    
    def load_log(self, state: AnsibleLogState) -> AnsibleLogState:
        """Load and parse Ansible log"""
        print(f"📂 Loading Ansible log: {state['file_path']}")
        
        with open(state['file_path'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk by Ansible structure
        chunks = self.chunker.chunk_log_content(content)
        
        print(f"   Parsed {len(chunks)} execution units")
        
        # Show summary
        failed_count = sum(1 for c in chunks if c.get('status') == 'failed')
        changed_count = sum(1 for c in chunks if c.get('status') == 'changed')
        
        print(f"   Status: {failed_count} failed, {changed_count} changed")
        
        return {
            **state,
            "log_content": content,
            "chunks": chunks
        }
    
    def answer_question(self, state: AnsibleLogState) -> AnsibleLogState:
        """Answer question about Ansible execution"""
        print(f"💭 Analyzing: {state['question']}")
        
        question_lower = state['question'].lower()
        
        # Smart filtering based on question type
        if any(word in question_lower for word in ['failed', 'error', 'problem']):
            relevant_chunks = [c for c in state['chunks'] if c.get('status') == 'failed']
            context = "failed tasks"
        elif any(word in question_lower for word in ['changed', 'modified']):
            relevant_chunks = [c for c in state['chunks'] if c.get('status') == 'changed']
            context = "changed tasks"
        elif 'summary' in question_lower or 'overview' in question_lower:
            relevant_chunks = [c for c in state['chunks'] if c['type'] == 'recap']
            if not relevant_chunks:
                relevant_chunks = state['chunks']
            context = "execution summary"
        else:
            # General question - use LLM to find relevant chunks
            relevant_chunks = self._find_relevant_chunks(state['chunks'], state['question'])
            context = "relevant tasks"
        
        print(f"   Found {len(relevant_chunks)} {context}")
        
        # Create context from relevant chunks
        if not relevant_chunks:
            return {
                **state,
                "answer": "No relevant information found in the log."
            }
        
        context_text = self._format_chunks(relevant_chunks)
        
        # Answer using LLM
        prompt = ChatPromptTemplate.from_template(
            """You are analyzing an Ansible execution log. Answer the question based on these log entries.

Log entries:
{context}

Question: {question}

Provide a clear, concise answer focusing on:
1. What happened
2. Which hosts were affected
3. Any errors or issues
4. Recommendations if applicable

Answer:"""
        )
        
        answer = self.llm.invoke(
            prompt.format(context=context_text, question=state['question'])
        ).content
        
        print("✓ Analysis complete")
        
        return {**state, "answer": answer}
    
    def _find_relevant_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        question: str
    ) -> List[Dict[str, Any]]:
        """Use LLM to find relevant chunks for general questions"""
        summaries = "\n\n".join([
            f"[{i+1}] {chunk['summary']}"
            for i, chunk in enumerate(chunks)
            if chunk['type'] == 'task'
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """Which of these Ansible tasks are relevant to the question?

Tasks:
{summaries}

Question: {question}

List relevant task numbers (e.g., "1, 3, 5") or "all":"""
        )
        
        response = self.llm.invoke(
            prompt.format(summaries=summaries, question=question)
        ).content
        
        if "all" in response.lower():
            return [c for c in chunks if c['type'] == 'task']
        
        try:
            indices = [
                int(n.strip()) - 1
                for n in response.replace(",", " ").split()
                if n.strip().isdigit()
            ]
            return [chunks[i] for i in indices if 0 <= i < len(chunks)]
        except:
            return chunks[:5]  # Fallback
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for LLM context"""
        formatted = []
        
        for chunk in chunks:
            if chunk['type'] == 'task':
                formatted.append(
                    f"PLAY: {chunk['play']}\n"
                    f"TASK: {chunk['task']}\n"
                    f"STATUS: {chunk['status']}\n"
                    f"RESULTS: {chunk['results']}\n"
                    f"ERROR: {chunk.get('error_message', 'None')}\n"
                    f"---\n{chunk['content']}"
                )
            elif chunk['type'] == 'recap':
                formatted.append(f"EXECUTION SUMMARY:\n{chunk['content']}")
        
        return "\n\n" + "="*60 + "\n\n".join(formatted)
    
    def _build_graph(self):
        workflow = StateGraph(AnsibleLogState)
        
        workflow.add_node("load", self.load_log)
        workflow.add_node("answer", self.answer_question)
        
        workflow.set_entry_point("load")
        workflow.add_edge("load", "answer")
        workflow.add_edge("answer", END)
        
        return workflow.compile()
    
    def query(self, file_path: str, question: str) -> str:
        """Ask a question about an Ansible log"""
        print("\n" + "="*60)
        print(f"Question: {question}")
        print("="*60 + "\n")
        
        result = self.graph.invoke({
            "file_path": file_path,
            "question": question,
            "log_content": "",
            "chunks": [],
            "answer": ""
        })
        
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result["answer"])
        print()
        
        return result["answer"]


if __name__ == "__main__":
    # Usage
    reader = AnsibleLogReader()
    reader.query("deploy.log", "What tasks failed?")
    reader.query("deploy.log", "Which hosts had changes?")
    reader.query("deploy.log", "Why did the deployment fail on server2?")
    reader.query("deploy.log", "Give me an execution summary")
