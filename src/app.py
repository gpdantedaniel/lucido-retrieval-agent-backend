import pandas as pd
import gradio as gr
from utils import TopicWeaver

hf_models = [
    'Qwen/Qwen3-Embedding-0.6B',
    'Qwen/Qwen3-Embedding-4B',
    'Qwen/Qwen3-Embedding-8B',
    'BAAI/bge-large-en-v1.5',
    'intfloat/multilingual-e5-large-instruct'
]

azureopenai_models = [
    'text-embedding-3-large',
    'text-embedding-3-small',
    'text-embedding-ada-002'
]

weaver = None

def pipeline(csv_file, model_name, provider, cluster_min, top_n, chunk_size):
    """
    A convenient pipeline that orchestrates all the steps
    involved in topic modeling, from keyword extraction to the
    generation of an interactive D3S.js graph of the articles.
    """

    # Extract content
    df = pd.read_csv(csv_file.name)
    docs, titles = df['content'], df['title']

    # Initialize progress bar and the Hyperbook Weaver
    progress = gr.Progress() 
    weaver = TopicWeaver(
        model_name=model_name, 
        inference_mode=provider, 
        top_n=top_n, 
        cluster_min=cluster_min, 
        chunk_size=chunk_size
    )

    # Extract keywords from the documents
    docs_keywords = [weaver.get_keywords(doc) for doc in progress.tqdm(docs, desc='Extracting keywords')]

    # Step 2. Embed and cluster the unique keywords
    unique_keywords = list(set([kw for keywords in docs_keywords for kw in keywords]))
    cluster_map, keyword_embeddings = weaver.cluster_keywords(unique_keywords)

    # Step 3. Find names for every cluster
    cluster_names = {
        cluster_id: weaver.name_cluster(keywords, keyword_embeddings) 
        for cluster_id, keywords in progress.tqdm(cluster_map.items(), desc='Naming clusters')
    }

    # Step 4. Display the clusters
    fig2d = weaver.display_2d(cluster_map, cluster_names, keyword_embeddings)
    fig3d = weaver.display_3d(cluster_map, cluster_names, keyword_embeddings)

    # Step 5. Generate the graph and its HTML display
    G, kw2cluster = weaver.create_graph(cluster_map, cluster_names, titles, docs_keywords)
    html_graph = weaver.display_graph(G)

    # Step 6. Give each article its tag
    tags_df = weaver.assign_tags(titles, docs_keywords, kw2cluster, cluster_names)

    del keyword_embeddings  # Free up some GPU memory
    
    return fig2d, fig3d, html_graph, tags_df

with gr.Blocks(title='Hyperbook Weaver 🕷️') as demo:
    gr.Markdown('# 🕷️ Hyperbook Weaver')

    """
    with gr.Tab(label='Introduction'):
        gr.Markdown("# To begin weaving 🕸️")
        gr.Markdown('## 1. Compile your articles into a CSV table with the columns "title" and "content."')
        gr.Markdown('## 2. Drop your CSV file below "📁 Data Upload" in the dropzone.')
        gr.Markdown('## 3. Click on "Process data" to weave your articles into a network!')
    """

    with gr.Tab(label='Topic plots'):
        plot_2d = gr.Plot(label='2D topic plot')
        plot_3d = gr.Plot(label='3D topic plot')
   
    with gr.Tab(label='Article tags'):
        articles_tags = gr.DataFrame(headers=['title', 'topics', 'keywords'], wrap=True)
    
    with gr.Tab(label='Outputs'):
        gr.Markdown('## 🕸️ HTML graph file')
        html_out = gr.File(label='Download HTML', file_types=['.html'])

    with gr.Sidebar(width=500):
        gr.Markdown('## 📁 Data upload')
        csv_in = gr.File(label='CSV', file_types=['.csv'], height=150)
        gr.Markdown('⚠️ The csv must have the columns "<b>title</b>" and "<b>content</b>"')

        run_btn = gr.Button('Process', variant='primary')

        gr.Markdown('## ⚙️ Parameters')
        cluster_min = gr.Number(value=2, label='Minimum topic cluster size', interactive=True, minimum=2)
        top_n = gr.Number(value=10, label='Keywords to extract by paper', interactive=True, minimum=1)
        chunk_size = gr.Number(value=1024, label='Chunk size', interactive=True, minimum=64)
        
        gr.Markdown('## 🔮 Embedding Model')
        gr.Markdown('⚠️ To use cloud inference, add the right keys to .env')

        provider = gr.Radio(
            label='Inference provider', 
            choices=['ollama', 'hf inference', 'azure openai'], 
            value='ollama', 
            interactive=True
        )

        model_name = gr.Dropdown(
            choices=hf_models + azureopenai_models, 
            label='Models', 
            interactive=True,
            allow_custom_value=True
        )

    run_btn.click(
        fn=pipeline, 
        inputs=[csv_in, model_name, provider, cluster_min, top_n, chunk_size], 
        outputs=[plot_2d, plot_3d, html_out, articles_tags]
    )

demo.launch()