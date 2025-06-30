import tempfile, uuid
import pandas as pd
import gradio as gr
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from utils import TopicWeaver

embedding_models = [
    'Qwen/Qwen3-Embedding-0.6B',
    'Qwen/Qwen3-Embedding-4B',
    'Qwen/Qwen3-Embedding-8B'
]

weaver = None

def pipeline(csv_file, model_name, cluster_min, top_n):

    df = pd.read_csv(csv_file.name)
    docs, titles = df['content'], df['title']

    progress = gr.Progress() 
    weaver = TopicWeaver(
        model_name=model_name,
        top_n=top_n,
        cluster_min=cluster_min
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

    # 3. Produce JSON & HTML artifacts
    tmpdir = Path(tempfile.mkdtemp())
    json_path = tmpdir / f'{uuid.uuid4()}.json'
    html_path = tmpdir / f'{uuid.uuid4()}.html'

    df.to_json(json_path, orient='records', indent=2)
    df.to_html(html_path, index=False)

    del keyword_embeddings  # Free up some GPU memory
    
    return fig2d, fig3d, str(json_path), str(html_path)

with gr.Blocks(title='Hyperbook Weaver 🕷️') as demo:
    with gr.Sidebar():
        gr.Markdown('## 📁 Data upload')
        csv_in = gr.File(label='CSV', file_types=['.csv'])
        gr.Markdown('⚠️ Make sure that your csv table has the columns "<b>title</b>" and "<b>content</b>"')

        run_btn = gr.Button('Process data', variant='primary')
        restart_btn = gr.Button("Restart")

        gr.Markdown('## ⚙️ Parameters')
        cluster_min = gr.Number(value=2, label='Minimum topic cluster size', interactive=True, minimum=2)
        top_n = gr.Number(value=10, label='Keywords to extract by paper', interactive=True, minimum=1)
        model_name = gr.Dropdown(choices=embedding_models, label='Embedding model', interactive=True)

        json_out = gr.File(label='Download JSON', file_types=['.json'], render=False)
        html_out = gr.File(label='Download HTML', file_types=['.html'], render=False)

    gr.Markdown('# 🕷️ Hyperbook Weaver')
    plot_2d = gr.Plot(label='2D topic plot')
    plot_3d = gr.Plot(label='3D topic plot')

    run_btn.click(
        fn=pipeline, 
        inputs=[csv_in, model_name, cluster_min, top_n], 
        outputs=[plot_2d, plot_3d, json_out, html_out]
    )

demo.launch()