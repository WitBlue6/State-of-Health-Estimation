conda create -n llm python=3.9
conda init
conda activate llm
# 1 基础深度学习库
python -m pip install torch==2.1.0
python -m pip install torch-npu==2.1.0.post13

conda install -c conda-forge \
    numpy=1.26 \
    scipy \
    scikit-learn=1.5.2 \

# 2 核心LLM依赖
python -m pip install transformers==4.36.2 tokenizers==0.15.0 --no-deps
python -m pip install sentence-transformers==2.3.0 scikit-learn sentencepiece --no-deps
python -m pip install huggingface-hub==0.19.3
python -m pip install chromadb --no-deps

# 3 其他辅助库
python -m pip install overrides tenacity pybase64 jsonschema httpx pydantic python-dotenv opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
python -m pip install pandas openai websockets matplotlib
python -m pip install regex
python -m pip install safetensors
python -m pip install nltk
