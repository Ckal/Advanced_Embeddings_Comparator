import os
import time
import pdfplumber
import docx
import nltk
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import jellyfish
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from collections import Counter
from tokenizers import Tokenizer, models, trainers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from functools import lru_cache
from langchain.retrievers import MultiQueryRetriever
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# NLTK Resource Download
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'snowball_data']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Failed to download {resource}: {str(e)}")

download_nltk_resources()

FILES_DIR = './files'

# Model Management
class ModelManager:
    def __init__(self):
        self.models = {
            'HuggingFace': {
                'e5-base-de': "danielheinz/e5-base-sts-en-de",
                'paraphrase-miniLM': "paraphrase-multilingual-MiniLM-L12-v2",
                'paraphrase-mpnet': "paraphrase-multilingual-mpnet-base-v2",
                'gte-large': "gte-large",
                'gbert-base': "gbert-base"
            },
            'OpenAI': {
                'text-embedding-ada-002': "text-embedding-ada-002"
            },
            'Cohere': {
                'embed-multilingual-v2.0': "embed-multilingual-v2.0"
            }
        }

    def add_model(self, provider, name, model_path):
        if provider not in self.models:
            self.models[provider] = {}
        self.models[provider][name] = model_path

    def remove_model(self, provider, name):
        if provider in self.models and name in self.models[provider]:
            del self.models[provider][name]

    def get_model(self, provider, name):
        return self.models.get(provider, {}).get(name)

    def list_models(self):
        return {provider: list(models.keys()) for provider, models in self.models.items()}

model_manager = ModelManager()

# File Handling
class FileHandler:
    @staticmethod
    def extract_text(file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            return FileHandler._extract_from_pdf(file_path)
        elif ext == '.docx':
            return FileHandler._extract_from_docx(file_path)
        elif ext == '.txt':
            return FileHandler._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _extract_from_pdf(file_path):
        with pdfplumber.open(file_path) as pdf:
            return ' '.join([page.extract_text() for page in pdf.pages])

    @staticmethod
    def _extract_from_docx(file_path):
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])

    @staticmethod
    def _extract_from_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

# Text Processing
def simple_tokenize(text):
    return text.split()

def preprocess_text(text, lang='german', apply_preprocessing=True):
    if not apply_preprocessing:
        return text
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        tokens = word_tokenize(text, language=lang)
    except LookupError:
        print(f"Warning: NLTK punkt tokenizer for {lang} not found. Using simple tokenization.")
        tokens = simple_tokenize(text)
    
    try:
        stop_words = set(stopwords.words(lang))
    except LookupError:
        print(f"Warning: Stopwords for {lang} not found. Skipping stopword removal.")
        stop_words = set()
    tokens = [token for token in tokens if token not in stop_words]
    
    try:
        stemmer = SnowballStemmer(lang)
        tokens = [stemmer.stem(token) for token in tokens]
    except ValueError:
        print(f"Warning: SnowballStemmer for {lang} not available. Skipping stemming.")
    
    return ' '.join(tokens)

def phonetic_match(text, query, method='levenshtein_distance', apply_phonetic=True):
    if not apply_phonetic:
        return 0
    if method == 'levenshtein_distance':
        text_phonetic = jellyfish.soundex(text)
        query_phonetic = jellyfish.soundex(query)
        return jellyfish.levenshtein_distance(text_phonetic, query_phonetic)
    return 0

def optimize_query(query, llm_model):
    llm = HuggingFacePipeline.from_model_id(
        model_id=llm_model,
        task="text2text-generation",
        model_kwargs={"temperature": 0, "max_length": 64},
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=get_retriever(vector_store, search_type, search_kwargs),
        llm=llm
    )
    optimized_queries = multi_query_retriever.generate_queries(query)
    return optimized_queries
    

def create_custom_embedding(texts, model_type='word2vec', vector_size=100, window=5, min_count=1):
    tokenized_texts = [text.split() for text in texts]
    
    if model_type == 'word2vec':
        model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    elif model_type == 'fasttext':
        model = FastText(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    else:
        raise ValueError("Unsupported model type")
    
    return model

class CustomEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_path):
        self.model = Word2Vec.load(model_path)  # or FastText.load() for FastText models
    
    def embed_documents(self, texts):
        return [self.model.wv[text.split()] for text in texts]
    
    def embed_query(self, text):
        return self.model.wv[text.split()]

# Custom Tokenizer
def create_custom_tokenizer(file_path, model_type='WordLevel', vocab_size=10000, special_tokens=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if model_type == 'WordLevel':
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    elif model_type == 'BPE':
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    elif model_type == 'Unigram':
        tokenizer = Tokenizer(models.Unigram())
    else:
        raise ValueError(f"Unsupported tokenizer model: {model_type}")

    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.WordLevelTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.train_from_iterator([text], trainer)

    return tokenizer

def custom_tokenize(text, tokenizer):
    return tokenizer.encode(text).tokens

# Embedding and Vector Store
#@lru_cache(maxsize=None)

# Helper functions

def get_text_splitter(split_strategy, chunk_size, overlap_size, custom_separators=None):
    if split_strategy == 'token':
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    elif split_strategy == 'recursive':
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=custom_separators or ["\n\n", "\n", " ", ""]
        )
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

def get_embedding_model(model_type, model_name):
    model_path = model_manager.get_model(model_type, model_name)
    if model_type == 'HuggingFace':
        return HuggingFaceEmbeddings(model_name=model_path)
    elif model_type == 'OpenAI':
        return OpenAIEmbeddings(model=model_path)
    elif model_type == 'Cohere':
        return CohereEmbeddings(model=model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_vector_store(vector_store_type, chunks, embedding_model):
    chunks_tuple = tuple(chunks)
    if vector_store_type == 'FAISS':
        return FAISS.from_texts(chunks, embedding_model)
    elif vector_store_type == 'Chroma':
        return Chroma.from_texts(chunks, embedding_model)
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")

def get_retriever(vector_store, search_type, search_kwargs):
    if search_type == 'similarity':
        return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    elif search_type == 'mmr':
        return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    elif search_type == 'custom':
        return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

def custom_similarity(query_embedding, doc_embedding, query, doc_text, phonetic_weight=0.3):
    embedding_sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
    phonetic_sim = phonetic_match(doc_text, query)
    combined_sim = (1 - phonetic_weight) * embedding_sim + phonetic_weight * phonetic_sim
    return combined_sim
	
def _create_vector_store(vector_store_type, chunks_tuple, embedding_model):
    chunks = list(chunks_tuple)
    
    if vector_store_type == 'FAISS':
        return FAISS.from_texts(chunks, embedding_model)
    elif vector_store_type == 'Chroma':
        return Chroma.from_texts(chunks, embedding_model)
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")


# Main Processing Functions
def process_files(file_path, model_type, model_name, split_strategy, chunk_size, overlap_size, custom_separators, lang='german', apply_preprocessing=True, custom_tokenizer_file=None, custom_tokenizer_model=None, custom_tokenizer_vocab_size=10000, custom_tokenizer_special_tokens=None):
    if file_path:
        text = FileHandler.extract_text(file_path)
    else:
        text = ""
        for file in os.listdir(FILES_DIR):
            file_path = os.path.join(FILES_DIR, file)
            text += FileHandler.extract_text(file_path)
    
    if custom_tokenizer_file:
        tokenizer = create_custom_tokenizer(custom_tokenizer_file, custom_tokenizer_model, custom_tokenizer_vocab_size, custom_tokenizer_special_tokens)
        text = ' '.join(custom_tokenize(text, tokenizer))
    elif apply_preprocessing:
        text = preprocess_text(text, lang)

    text_splitter = get_text_splitter(split_strategy, chunk_size, overlap_size, custom_separators)
    chunks = text_splitter.split_text(text)

    embedding_model = get_embedding_model(model_type, model_name)

    return chunks, embedding_model, len(text.split())

def search_embeddings(chunks, embedding_model, vector_store_type, search_type, query, top_k, lang='german', apply_phonetic=True, phonetic_weight=0.3):
    preprocessed_query = preprocess_text(query, lang) if apply_phonetic else query
    
    vector_store = get_vector_store(vector_store_type, chunks, embedding_model)
    retriever = get_retriever(vector_store, search_type, {"k": top_k})

    start_time = time.time()
    results = retriever.invoke(preprocessed_query)

    def score_result(doc):
        similarity_score = vector_store.similarity_search_with_score(doc.page_content, k=1)[0][1]
        if apply_phonetic:
            phonetic_score = phonetic_match(doc.page_content, query)
            return (1 - phonetic_weight) * similarity_score + phonetic_weight * phonetic_score
        else:
            return similarity_score

    results = sorted(results, key=score_result, reverse=True)
    end_time = time.time()

    embeddings = []
    for doc in results:
        if hasattr(doc, 'embedding'):
            embeddings.append(doc.embedding)
        else:
            embeddings.append(None)

    results_df = pd.DataFrame({
        'content': [doc.page_content for doc in results],
        'embedding': embeddings
    })

    return results_df, end_time - start_time, vector_store, results



# Evaluation Metrics
# ... (previous code remains the same)

def calculate_statistics(results, search_time, vector_store, num_tokens, embedding_model, query, top_k):
    stats = {
        "num_results": len(results),
        "avg_content_length": np.mean([len(doc.page_content) for doc in results]) if results else 0,
        "search_time": search_time,
        "vector_store_size": vector_store._index.ntotal if hasattr(vector_store, '_index') else "N/A",
        "num_documents": len(vector_store.docstore._dict),
        "num_tokens": num_tokens,
        "embedding_vocab_size": embedding_model.client.get_vocab_size() if hasattr(embedding_model, 'client') and hasattr(embedding_model.client, 'get_vocab_size') else "N/A",
        "embedding_dimension": len(embedding_model.embed_query(query)),
        "top_k": top_k,
    }
    
    if len(results) > 1000:
        embeddings = [embedding_model.embed_query(doc.page_content) for doc in results]
        pairwise_similarities = np.inner(embeddings, embeddings)
        stats["result_diversity"] = 1 - np.mean(pairwise_similarities[np.triu_indices(len(embeddings), k=1)])
        
        if len(embeddings) > 2:
            stats["silhouette_score"] = silhouette_score(embeddings, range(len(embeddings)))
        else:
            stats["silhouette_score"] = "N/A"
    else:
        stats["result_diversity"] = "N/A"
        stats["silhouette_score"] = "N/A"
    
    query_embedding = embedding_model.embed_query(query)
    result_embeddings = [embedding_model.embed_query(doc.page_content) for doc in results]
    similarities = [np.inner(query_embedding, emb) for emb in result_embeddings]
    rank_correlation, _ = spearmanr(similarities, range(len(similarities)))
    stats["rank_correlation"] = rank_correlation
    
    return stats

# Visualization
def visualize_results(results_df, stats_df):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    sns.barplot(x='model', y='search_time', data=stats_df, ax=axs[0, 0])
    axs[0, 0].set_title('Search Time by Model')
    axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    sns.scatterplot(x='result_diversity', y='rank_correlation', hue='model', data=stats_df, ax=axs[0, 1])
    axs[0, 1].set_title('Result Diversity vs. Rank Correlation')
    
    sns.boxplot(x='model', y='avg_content_length', data=stats_df, ax=axs[1, 0])
    axs[1, 0].set_title('Distribution of Result Content Lengths')
    axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    embeddings = np.array([embedding for embedding in results_df['embedding'] if isinstance(embedding, np.ndarray)])
    if len(embeddings) > 1:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=results_df['model'][:len(embeddings)], ax=axs[1, 1])
        axs[1, 1].set_title('t-SNE Visualization of Result Embeddings')
    else:
        axs[1, 1].text(0.5, 0.5, "Not enough data for t-SNE visualization", ha='center', va='center')
    
    plt.tight_layout()
    return fig

def optimize_vocabulary(texts, vocab_size=10000, min_frequency=2):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    word_freq = Counter(word for text in texts for word in text.split())
    
    optimized_texts = [
        ' '.join(word for word in text.split() if word_freq[word] >= min_frequency)
        for text in texts
    ]
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(optimized_texts, trainer)
    
    return tokenizer, optimized_texts

# New preprocessing function
def optimize_query(query, llm):
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=get_retriever(vector_store, search_type, search_kwargs),
        llm=llm
    )
    optimized_queries = multi_query_retriever.generate_queries(query)
    return optimized_queries

# New postprocessing function
def rerank_results(results, query, reranker):
    reranked_results = reranker.rerank(query, [doc.page_content for doc in results])
    return reranked_results

# Main Comparison Function
def compare_embeddings(file, query, embedding_models, custom_embedding_model, split_strategy, chunk_size, overlap_size, custom_separators, vector_store_type, search_type, top_k, lang='german', apply_preprocessing=True, optimize_vocab=False, apply_phonetic=True, phonetic_weight=0.3, custom_tokenizer_file=None, custom_tokenizer_model=None, custom_tokenizer_vocab_size=10000, custom_tokenizer_special_tokens=None, use_query_optimization=False, query_optimization_model="google/flan-t5-base", use_reranking=False):
    all_results = []
    all_stats = []
    settings = {
        "split_strategy": split_strategy,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "custom_separators": custom_separators,
        "vector_store_type": vector_store_type,
        "search_type": search_type,
        "top_k": top_k,
        "lang": lang,
        "apply_preprocessing": apply_preprocessing,
        "optimize_vocab": optimize_vocab,
        "apply_phonetic": apply_phonetic,
        "phonetic_weight": phonetic_weight,
        "use_query_optimization": use_query_optimization,
        "use_reranking": use_reranking
    }

    # Parse the embedding models from the checkbox group
    models = [model.split(':') for model in embedding_models]
    if custom_embedding_model:
        models.append(custom_embedding_model.strip().split(':'))

    for model_type, model_name in models:
        chunks, embedding_model, num_tokens = process_files(
            file.name if file else None,
            model_type,
            model_name,
            split_strategy,
            chunk_size,
            overlap_size,
            custom_separators.split(',') if custom_separators else None,
            lang,
            apply_preprocessing,
            custom_tokenizer_file,
            custom_tokenizer_model,
            int(custom_tokenizer_vocab_size),
            custom_tokenizer_special_tokens.split(',') if custom_tokenizer_special_tokens else None
        )

        if optimize_vocab:
            tokenizer, optimized_chunks = optimize_vocabulary(chunks)
            chunks = optimized_chunks

        if use_query_optimization:
            optimized_queries = optimize_query(query, query_optimization_model)
            query = " ".join(optimized_queries)

        results, search_time, vector_store, results_raw = search_embeddings(
            chunks,
            embedding_model,
            vector_store_type,
            search_type,
            query,
            top_k,
            lang,
            apply_phonetic,
            phonetic_weight
        )
        
        if use_reranking:
            reranker = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-12-v2")
            results_raw = rerank_results(results_raw, query, reranker)

        result_embeddings = [doc.metadata.get('embedding', None) for doc in results_raw]

        stats = calculate_statistics(results_raw, search_time, vector_store, num_tokens, embedding_model, query, top_k)
        stats["model"] = f"{model_type} - {model_name}"
        stats.update(settings)

        formatted_results = format_results(results_raw, stats)
        for i, result in enumerate(formatted_results):
            result['embedding'] = result_embeddings[i]

        all_results.extend(formatted_results)
        all_stats.append(stats)

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)

    fig = visualize_results(results_df, stats_df)

    return results_df, stats_df, fig

def format_results(results, stats):
    formatted_results = []
    for doc in results:
        result = {
            "Model": stats["model"],
            "Content": doc.page_content,
            "Embedding": doc.embedding if hasattr(doc, 'embedding') else None,
            **doc.metadata,
            **{k: v for k, v in stats.items() if k not in ["model"]}
        }
        formatted_results.append(result)
    return formatted_results

# Gradio Interface
def launch_interface(share=True):
    with gr.Blocks() as iface:
        gr.Markdown("# Advanced Embedding Comparison Tool")
        
        with gr.Tab("Simple"):
            file_input = gr.File(label="Upload File (Optional)")
            query_input = gr.Textbox(label="Search Query")
            embedding_models_input = gr.CheckboxGroup(
                choices=[
                    "HuggingFace:paraphrase-miniLM",
                    "HuggingFace:paraphrase-mpnet",
                    "OpenAI:text-embedding-ada-002",
                    "Cohere:embed-multilingual-v2.0"
                ],
                label="Embedding Models"
            )
            top_k_input = gr.Slider(1, 10, step=1, value=5, label="Top K")
        
        with gr.Tab("Advanced"):
            custom_embedding_model_input = gr.Textbox(label="Custom Embedding Model (optional, format: type:name)")
            split_strategy_input = gr.Radio(choices=["token", "recursive"], label="Split Strategy", value="recursive")
            chunk_size_input = gr.Slider(100, 1000, step=100, value=500, label="Chunk Size")
            overlap_size_input = gr.Slider(0, 100, step=10, value=50, label="Overlap Size")
            custom_separators_input = gr.Textbox(label="Custom Split Separators (comma-separated, optional)")
            vector_store_type_input = gr.Radio(choices=["FAISS", "Chroma"], label="Vector Store Type", value="FAISS")
            search_type_input = gr.Radio(choices=["similarity", "mmr", "custom"], label="Search Type", value="similarity")
            lang_input = gr.Dropdown(choices=["german", "english", "french"], label="Language", value="german")
        
        with gr.Tab("Optional"):
            apply_preprocessing_input = gr.Checkbox(label="Apply Text Preprocessing", value=True)
            optimize_vocab_input = gr.Checkbox(label="Optimize Vocabulary", value=False)
            apply_phonetic_input = gr.Checkbox(label="Apply Phonetic Matching", value=True)
            phonetic_weight_input = gr.Slider(0, 1, step=0.1, value=0.3, label="Phonetic Matching Weight")
            custom_tokenizer_file_input = gr.File(label="Custom Tokenizer File (Optional)")
            custom_tokenizer_model_input = gr.Textbox(label="Custom Tokenizer Model (e.g., WordLevel, BPE, Unigram)")
            custom_tokenizer_vocab_size_input = gr.Textbox(label="Custom Tokenizer Vocab Size", value="10000")
            custom_tokenizer_special_tokens_input = gr.Textbox(label="Custom Tokenizer Special Tokens (comma-separated)")
            use_query_optimization_input = gr.Checkbox(label="Use Query Optimization", value=False)
            query_optimization_model_input = gr.Textbox(label="Query Optimization Model", value="google/flan-t5-base")
            use_reranking_input = gr.Checkbox(label="Use Reranking", value=False)

        results_output = gr.Dataframe(label="Results", interactive=False)
        stats_output = gr.Dataframe(label="Statistics", interactive=False)
        plot_output = gr.Plot(label="Visualizations")

        submit_button = gr.Button("Compare Embeddings")
        submit_button.click(
            fn=compare_embeddings,
            inputs=[
                file_input, query_input, embedding_models_input, custom_embedding_model_input,
                split_strategy_input, chunk_size_input, overlap_size_input, custom_separators_input,
                vector_store_type_input, search_type_input, top_k_input, lang_input,
                apply_preprocessing_input, optimize_vocab_input, apply_phonetic_input,
                phonetic_weight_input, custom_tokenizer_file_input, custom_tokenizer_model_input,
                custom_tokenizer_vocab_size_input, custom_tokenizer_special_tokens_input,
                use_query_optimization_input, query_optimization_model_input, use_reranking_input
            ],
            outputs=[results_output, stats_output, plot_output]
        )


    tutorial_md = """
    # Advanced Embedding Comparison Tool Tutorial

    This tool allows you to compare different embedding models and retrieval strategies for document search and similarity matching.

    ## How to use:

    1. Upload a file (optional) or use the default files in the system.
    2. Enter a search query.
    3. Enter embedding models as a comma-separated list (e.g., HuggingFace:paraphrase-miniLM,OpenAI:text-embedding-ada-002).
    4. Set the number of top results to retrieve.
    5. Optionally, specify advanced settings such as custom embedding models, text splitting strategies, and vector store types.
    6. Choose whether to use optional features like vocabulary optimization, query optimization, or result reranking.
    7. If you have a custom tokenizer, upload the file and specify its attributes.

    The tool will process your query and display results, statistics, and visualizations to help you compare the performance of different models and strategies.
    """

    iface = gr.TabbedInterface(
        [iface, gr.Markdown(tutorial_md)],
        ["Embedding Comparison", "Tutorial"]
    )

    iface.launch(share=share)

if __name__ == "__main__":
    launch_interface()