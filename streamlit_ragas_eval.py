import streamlit as st
import pandas as pd
import requests
import urllib3
from datetime import datetime
from datasets import Dataset
from typing import List, Dict, Any
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from ragas import evaluate
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import faithfulness, context_recall, context_precision, answer_relevancy
from model_config import get_model_config
import io
import time
import random

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleAPIRetriever:
    def __init__(self, api_url: str, bearer_token: str, tenant: str, knowledge_base_name: str):
        self.api_url = api_url
        self.bearer_token = bearer_token
        self.tenant = tenant
        self.knowledge_base_name = knowledge_base_name
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }

    def get_relevant_documents(self, query: str, max_retries: int = 3) -> List[Document]:
        payload = {
            "tenant": self.tenant,
            "message": query,
            "model": "claude-3-7-sonnet",
            "knowledgeBaseName": self.knowledge_base_name
        }
       
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=self.headers, verify=False, timeout=30)
                response.raise_for_status()
                result = response.json()
               
                documents = []
               
                if isinstance(result, list) and len(result) > 1:
                    bot_response = result[1]
                   
                    if 'references' in bot_response:
                        for ref in bot_response['references']:
                            if 'content' in ref and ref['content'].strip():
                                doc = Document(
                                    page_content=ref['content'].strip(),
                                    metadata={
                                        'source': ref.get('name', 'unknown'),
                                        'location': ref.get('location', '')
                                    }
                                )
                                documents.append(doc)
                   
                    if not documents and 'message' in bot_response:
                        doc = Document(
                            page_content=bot_response['message'],
                            metadata={'source': 'bot_response'}
                        )
                        documents.append(doc)
               
                elif isinstance(result, dict):
                    if 'sources' in result:
                        for source in result['sources']:
                            doc = Document(
                                page_content=source.get('content', ''),
                                metadata=source.get('metadata', {})
                            )
                            documents.append(doc)
                    elif 'answer' in result:
                        doc = Document(
                            page_content=result['answer'],
                            metadata={'source': 'api_response'}
                        )
                        documents.append(doc)
               
                return documents
           
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Error retrieving documents after {max_retries} attempts: {e}")
                    return []
               
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
       
        return []

def generate_answer_from_context(question, contexts, llm_model=None):
    if not contexts or not any(ctx.strip() for ctx in contexts):
        return "No relevant information found to answer this question."
   
    valid_contexts = []
    for ctx in contexts:
        if ctx and ctx.strip():
            cleaned = ctx.strip().replace('\\n', ' ').replace('\n', ' ')
            cleaned = ' '.join(cleaned.split())
            if len(cleaned) > 10:
                valid_contexts.append(cleaned)
   
    if not valid_contexts:
        return "No valid context available to generate an answer."
   
    # Improved keyword matching with synonyms and related terms
    question_lower = question.lower()
    key_terms = []
   
    # Extract meaningful terms (longer than 3 chars, not common words)
    stop_words = {'what', 'when', 'where', 'why', 'how', 'does', 'the', 'and', 'for', 'with', 'are', 'you'}
    for term in question_lower.split():
        if len(term) > 3 and term not in stop_words:
            key_terms.append(term)
   
    # Add related terms for better matching
    if 'purpose' in question_lower or 'why' in question_lower:
        key_terms.extend(['goal', 'objective', 'reason', 'benefit', 'aim'])
    if 'step' in question_lower or 'first' in question_lower:
        key_terms.extend(['begin', 'start', 'initial', 'process'])
   
    combined_context = ' '.join(valid_contexts)
   
    # Try LLM-based answer generation first with retry logic
    if llm_model:
        for attempt in range(3):
            try:
                prompt = f"""Based on the following context, answer the question concisely and accurately.

Context: {combined_context[:1000]}

Question: {question}

Answer:"""
               
                response = llm_model.invoke(prompt)
                if hasattr(response, 'content'):
                    answer = response.content.strip()
                else:
                    answer = str(response).strip()
               
                if answer and len(answer) > 10 and answer != combined_context:
                    return answer
                break
            except Exception as e:
                if attempt == 2:
                    print(f"LLM generation failed after 3 attempts: {e}")
                else:
                    time.sleep(2 ** attempt)
   
    # Fallback to improved extractive method
    sentences = combined_context.replace('!', '.').replace('?', '.').split('.')
    relevant_sentences = []
   
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 15:
            sentence_lower = sentence.lower()
            # Score sentences based on keyword matches
            score = sum(1 for term in key_terms if term in sentence_lower)
            if score > 0:
                relevant_sentences.append((sentence, score))
   
    if relevant_sentences:
        # Sort by relevance score and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in relevant_sentences[:2]]
        answer = '. '.join(top_sentences)
        if not answer.endswith('.'):
            answer += '.'
        return answer
   
    # Final fallback - return summarized context
    first_context = valid_contexts[0]
    if len(first_context) > 200:
        # Try to find the most relevant part
        parts = first_context.split('. ')
        for part in parts:
            if any(term in part.lower() for term in key_terms):
                return part + '.'
        # If no relevant part found, truncate
        return first_context[:200] + "..."
   
    return first_context

def run_ragas_evaluation(test_data, api_url, bearer_token, tenant, knowledge_base_name, model_id, embedding_model_id):
       
    retriever = SimpleAPIRetriever(api_url, bearer_token, tenant, knowledge_base_name)
   
    questions = []
    answers = []
    contexts = []
    ground_truths = []
   
    progress_bar = st.progress(0)
   
    for i, item in enumerate(test_data):
        question = str(item['question']).strip()
        ground_truth = str(item['ground_truth']).strip()

        if not question or question == 'nan':
            continue

        if not ground_truth or ground_truth == 'nan':
            ground_truth = "No ground truth provided."

        documents = retriever.get_relevant_documents(question)
        context_list = [doc.page_content for doc in documents if doc.page_content and doc.page_content.strip()]
       
        # Ensure we have valid contexts
        if not context_list:
            context_list = ["No relevant context found for this question."]
       
        try:
            llm_model = ChatBedrockConverse(
                region_name="us-gov-west-1",
                model=model_id,
                temperature=0.1,
                max_tokens=200,
            )
        except Exception:
            llm_model = None
       
        # Generate answer from contexts using LLM
        answer = generate_answer_from_context(question, context_list, llm_model)
       
        if not answer or not answer.strip():
            answer = "Unable to generate answer from available context."
       
        questions.append(str(question).strip())
        answers.append(str(answer).strip())
        contexts.append([str(ctx).strip() for ctx in context_list])
        ground_truths.append(str(ground_truth).strip())
       
        progress_bar.progress((i + 1) / len(test_data))
   
    if not questions:
        st.error("No valid data generated for evaluation")
        return None, None
   
    # Create dataset
    dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    })
   
    model_cfg = get_model_config(model_id)

    bedrock_model = ChatBedrockConverse(
        region_name="us-gov-west-1",
        model=model_id,
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )

    bedrock_embeddings = BedrockEmbeddings(
        region_name="us-gov-west-1",
        model_id=embedding_model_id,
    )
   
    # Run evaluation with retry logic
    for attempt in range(3):
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, context_recall, context_precision, answer_relevancy],
                llm=bedrock_model,
                embeddings=bedrock_embeddings
            )
            break
        except Exception as e:
            if attempt == 2:
                st.error(f"Evaluation failed after 3 attempts: {e}")
                return None, None
            st.warning(f"Evaluation attempt {attempt + 1} failed, retrying...")
            time.sleep(5)
   
    return result, questions

from streamlit_ui import StreamlitUI

# Main application
st.title("RAGAS Evaluation Tool")

ui = StreamlitUI()
config = ui.render_sidebar()
test_data = ui.render_file_upload()

if test_data:
    ui.render_evaluation_section(test_data, config, run_ragas_evaluation)