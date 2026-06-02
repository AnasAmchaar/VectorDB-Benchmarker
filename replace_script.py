import os

replacements = {
    'rag_benchmark': 'rag_benchmark',
    'vectordb-bench': 'vectordb-bench',
    'benchmark': 'benchmark',
    'VectorDB Benchmarker': 'VectorDB Benchmarker',
    'benchmark_run': 'benchmark_run',
    'Standard analyzer': 'Standard analyzer',
    '"analyzer": "standard"': '"analyzer": "standard"',
    'RAG Bench Team': 'RAG Bench Team',
    'VectorDB Benchmark': 'VectorDB Benchmark',
    'VectorDB Benchmarker Dependencies': 'VectorDB Benchmark Dependencies',
    'multilingual-supporting': 'multilingual-supporting',
    'Multilingual': 'Multilingual',
    'dataset corpus': 'dataset corpus',
    'VectorDB Benchmarker (VectorDB Benchmark)': 'VectorDB Benchmarker',
    'Corpus': 'Corpus'
}

files_to_process = []
for root, dirs, files in os.walk('.'):
    if '.git' in root or '.venv' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith(('.py', '.md', '.txt', '.env', '.example', '.example copy')):
            files_to_process.append(os.path.join(root, file))

for file_path in files_to_process:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    # specific lowercase replace for rag_benchmark and dataset corpus
    new_content = new_content.replace('dataset corpus', 'dataset corpus')
    new_content = new_content.replace('rag_benchmark', 'rag_benchmark')
    new_content = new_content.replace('multilingual', 'multilingual') # careful with this one, let's only do specific ones
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {file_path}')
