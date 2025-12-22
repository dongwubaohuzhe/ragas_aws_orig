import streamlit as st
import pandas as pd
from datetime import datetime
import io

class StreamlitUI:
    def __init__(self):
        self.api_url = "https://api.awsurl.pvt/chat"
        self.tenant = "chat_tenant"
        self.knowledge_base_name = "chat_kb"
       
    def render_sidebar(self):
        st.sidebar.header("⚙️ Configuration")
       
        api_url = st.sidebar.text_input("API URL", value=self.api_url)
        bearer_token = st.sidebar.text_input("Bearer Token", type="password")
        tenant = st.sidebar.text_input("Tenant", value=self.tenant)
        knowledge_base_name = st.sidebar.text_input("Knowledge Base Name", value=self.knowledge_base_name)
       
        model_id = st.sidebar.selectbox(
            "LLM Model",
            ["anthropic.claude-3-5-sonnet-20240620-v1:0","anthropic.claude-3-7-sonnet-20250219-v1:0","amazon.titan-text-express-v1"]
        )
       
        embedding_model_id = st.sidebar.selectbox(
            "Embedding Model",
            ["amazon.titan-embed-text-v2:0"]
        )
       
        self._render_instructions()
       
        return api_url, bearer_token, tenant, knowledge_base_name, model_id, embedding_model_id
   
    def render_file_upload(self):
        st.header("1. Upload Test Plan")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
       
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} questions from test plan")
                st.dataframe(df.head())
               
                if 'question' not in df.columns or 'ground_truth' not in df.columns:
                    st.error("CSV must contain 'question' and 'ground_truth' columns")
                    return None
               
                return df.to_dict('records')
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None
       
        return None
   
    def render_evaluation_section(self, test_data, config, evaluation_func):
        st.header("2. Run Evaluation")
        if st.button("Start RAGAS Evaluation", type="primary"):
            api_url, bearer_token, tenant, knowledge_base_name, model_id, embedding_model_id = config
           
            if not all([api_url, bearer_token, tenant, knowledge_base_name]):
                st.error("Please fill in all configuration fields")
                return
           
            with st.spinner("Running RAGAS evaluation..."):
                try:
                    result, evaluation_data = evaluation_func(
                        test_data, api_url, bearer_token, tenant,
                        knowledge_base_name, model_id, embedding_model_id
                    )
                   
                    if result is not None:
                        self._render_results(result, knowledge_base_name, model_id, embedding_model_id)
                       
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
   
    def _render_results(self, result, knowledge_base_name, model_id, embedding_model_id):
        st.success("Evaluation completed!")
       
        st.header("3. Download Results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_evaluation_{knowledge_base_name}_{timestamp}.csv"
       
        results_df = result.to_pandas()
       
        results_df['evaluation_timestamp'] = timestamp
        results_df['knowledge_base_name'] = knowledge_base_name
        results_df['model_id'] = model_id
        results_df['embedding_model_id'] = embedding_model_id
       
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
       
        st.download_button(
            "Download Results CSV",
            csv_buffer.getvalue(),
            filename,
            "text/csv"
        )
       
        st.info(f"Results saved with {len(results_df)} evaluations")
        st.dataframe(results_df)
   
    def _render_instructions(self):
        st.sidebar.markdown("---")
        with st.sidebar.expander("📋 Instructions"):
            st.markdown("""
            1. Upload Test Plan (csv) with 'question' and 'ground_truth' columns
            2. Configure API settings and models
            3. Run evaluation
            4. Download results (csv) with timestamp
            """)
       
        with st.sidebar.expander("📊 Metric Definitions"):
            st.markdown("""
            **Faithfulness** (0-1)  
            Measures if the answer is factually consistent with the given context. Higher is better.
           
            **Context Recall** (0-1)  
            Measures how much of the ground truth can be attributed to the retrieved context. Higher is better.
           
            **Context Precision** (0-1)  
            Measures how relevant the retrieved contexts are to the question. Higher is better.
           
            **Answer Relevancy** (0-1)  
            Measures how relevant the generated answer is to the question. Higher is better.
            """)